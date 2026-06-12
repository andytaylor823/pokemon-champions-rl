---
last_synced: 6061c64
watches:
  - src/
  - sim/src/sim-worker.ts
  - docs/architecture/state-encoding.md
  - docs/architecture/repo-architecture.md
---

# Encoder — Design Spec (Phase 1, with Phase 4 hooks)

> Module ref: `docs/architecture/repo-architecture.md` §3.2
> Design ref: `docs/architecture/state-encoding.md`

## Status

Phase 1 implementation complete. Modules: `src/encoder.py`, `src/action_space.py`,
`src/vocab.py`, `src/obs_bundle.py`. Extended snapshot in `sim/src/sim-worker.ts`.
Key numbers: A = 1036, F = 65 (entity feature dim), N = 12 at team preview / 8 after.

## Context

The `Encoder` translates a battle observation into the neural net's tensor input bundle
(`ObsBundle`). Its one upstream dependency, `SimClient`, is built and tested
(`src/sim_client.py` + `sim/src/sim-worker.ts`, passing end-to-end self-test), so the
Encoder is the correct next module to build.

This spec covers the **unified Phase-1/Phase-4 schema** with Phase 1 as the degenerate
case (one candidate per opponent slot, all belief weights = 1.0). Phase-4 *population* of
candidates is `BeliefModel`'s job and is out of scope here; we only build the schema that
holds it. Framework is **PyTorch**. NN-internal architecture (how the net consumes the
bundle) is CVPN's concern; this spec stops at the bundle.

## Scope

Translate battle observation + belief into the `ObsBundle` tensor schema consumed by the CVPN.

## Locked Decisions

1. **Categoricals — hybrid.** Encoder emits **integer IDs** for high-cardinality
   categoricals (species, the 4 moves, item, ability); the **CVPN owns the embedding
   tables** and trains them end-to-end. Encoder emits **one-hot / float** directly for
   low-cardinality categoricals (status, nature) and all continuous features (final stats,
   HP%, stat stages, counters). Embedding dims ~16–32, to tune.

2. **Moves → token = Option B** (net-internal; Encoder just emits the 4 move IDs per
   Pokémon). The net runs a small attention step over the 4 move-vectors, then reads ONE
   *wide* summary. ⚠️ **REVISIT** — escalate to per-move tokens (Option C) if move detail
   proves to be lost.

3. **Token + belief layout — variable-length list, weight rides with each token.** Encoder
   outputs however many tokens the state needs (12 in Phase 1, your-6 + Σ candidates in
   Phase 4). Each token carries a `belief_weight`: 1.0 for all your mons and everything in
   Phase 1, <1 for opponent candidates. The weight is a **clean per-token number** (search
   uses it to aggregate per-candidate values) **and** an **input feature** (so the net
   knows candidate likelihood). A `slot_id` tag groups candidates per opponent slot.
   Batching uses **padding + mask**.

4. **Perspective — always relative.** Encoder rewrites every position to "me vs them": the
   `perspective` player's 6 mons first (flagged mine), opponent second. Net only ever
   learns one viewpoint. When both sides' plans are needed (simultaneous moves / self-play),
   call `encode` twice. Per-token **physical-slot feature** (active-left / active-right /
   bench) is included for targeting.

5. **Action index — small shared `action_space` module.** One standalone module owns the
   canonical numbered action list, the **number ↔ Showdown choice-string** map, and the
   **legal-request → mask** helper. `SimClient`, `Encoder`, and `CVPN` all import it →
   single source of truth.

6. **Derived features — none pre-computed initially.** Encoder forwards raw facts only; the
   net learns relationships. ⚠️ **REVISIT** — effective Speed is the first derived feature
   to add back, and if so it must be **computed by Showdown and exposed in the snapshot**
   (source-of-truth rule).

7. **Snapshot gaps — close as prerequisite Task #1.** Extend the Node worker's snapshot to
   carry the raw facts the Encoder needs but today drops (see Task 1 below).

8. **`ObsBundle` = TensorDict.** Adds the `tensordict` dependency; gives batching, stacking,
   and device-movement for free. Pydantic is intentionally NOT used here — wrong tool for a
   hot-path tensor bundle.

## Deferred / Open (record, do not decide now)

- **Per-slot vs joint-team candidates** (state-encoding.md §12.3). The variable-length
  layout supports per-slot candidates naturally; the joint-vs-per-slot choice is a
  Phase-4 / `BeliefModel` decision.
- **Top-K candidates per opponent slot** — `BeliefModel`'s concern (Phase 4).
- **Embedding dims, `d_model`, attention heads/layers** — empirical tuning.

## ObsBundle Schema (the contract)

```
encode(observation: PublicObservation, perspective: Player, belief: Belief | None) -> ObsBundle
```

A `TensorDict` (`N` = number of entity tokens; 12 in Phase 1, variable in Phase 4):

| Key | Shape / dtype | Contents |
|---|---|---|
| `entities` | float `[N, F]` | continuous + low-card one-hot per token: HP%, 6 final stats (norm.), 6 stat stages, status one-hot, nature one-hot, per-move pp/maxpp/disabled `[·,4]`, volatile turn-counters, Substitute HP, turns-on-field, item-consumed flag, is-active/back-row/fainted/not-brought, physical-slot one-hot, side flag, `belief_weight` (also surfaced here as input feature) |
| `ids.species` | int64 `[N]` | for CVPN embedding |
| `ids.ability` | int64 `[N]` | for CVPN embedding |
| `ids.item` | int64 `[N]` | for CVPN embedding |
| `ids.moves` | int64 `[N, 4]` | for CVPN embedding |
| `belief_weight` | float `[N]` | clean per-token weight (1.0 in Phase 1); used by search to aggregate |
| `slot_id` | int `[N]` | owner + opponent-slot grouping for candidates |
| `field` | float `[Ff]` | weather/terrain/TR/gravity + one-hot duration counters |
| `sides` | float `[2, Fs]` | tailwind / screens / hazards / mega-used, per side |
| `scalars` | float `[Fg]` | turn #, phase, whose-decision |
| `action_mask` | bool `[A]` | legal actions over the shared canonical index |
| `padding_mask` | bool `[N]` | true = real token (for batching) |

**Invariants** (from §3.2): keep the entity axis (don't flatten across Pokémon); field is a
peer token, not broadcast; encode features, never opaque set IDs; probability lives on the
belief weight, never inside a content feature.

## Implementation Tasks

### Task 1: Extend the snapshot (prerequisite)

**File:** `sim/src/sim-worker.ts` — `snapshotPokemon` (≈ lines 128–150), also
`snapshotBattle`/`snapshotSide` as needed.

**Add:**
- Volatile turn-counters (currently only `Object.keys(volatiles)`)
- Status counters (toxic stage / sleep turns)
- Protect/stall counter
- Substitute HP
- Turns-on-field
- Item-consumed distinction

**Showdown accessors:** `volatiles[id].duration`/`.time`, `statusState.stage`/`.time`,
`volatiles['stall']`, `volatiles['substitute'].hp`, `activeTurns`, `lastItem`/`itemState`.

**Tests:** extend `sim/tests/sim-worker.test.ts` — assert Protect counter after stacking,
Substitute HP after Substitute, toxic counter increments.

### Task 2: `src/action_space.py` (new module)

Canonical numbered action list, `index → choice_string` (e.g.
`"move heatwave 1, move protect"`), and `legal_request → bool mask`. The flat-joint vs
per-Pokémon-factored head shape (§3.6) is an open sub-decision; expose `A` and the mask
builder so the choice stays behind this module. Wire `SimClient` to use it for any
index↔string translation.

### Task 3: `src/vocab.py` (new module)

Load id↔name maps from `data/legal/` (~195 species, ~120 items, ~551 moves, abilities,
natures). Bidirectional, transparent (the human-readable anchor; embeddings are never
inverted).

### Task 4: `src/obs_bundle.py` (new module)

`ObsBundle` TensorDict factory + batching/padding helpers + `.to(device)`.

### Task 5: `src/encoder.py` (new module)

`encode(...)`: read the SimClient `StateView`/snapshot (a plain dict), canonicalize to the
perspective player, build the entity tokens (IDs + feature block), field/sides/scalars,
`belief_weight` (1.0 delta in Phase 1, from `belief` in Phase 4), and `action_mask` via
`action_space`. Returns the `ObsBundle`.

### Task 6: `pyproject.toml`

Add `tensordict` and ensure `torch` is a direct dependency.

## Reuse (don't reinvent)

- `src/sim_client.py` `SimClient` — source of snapshots/views.
- `sim/src/sim-worker.ts` `snapshotPokemon`/`snapshotBattle`/`view` — extend, don't replace.
- `data/legal/` lists — the vocabulary for IDs.
- `toy_examples/*/network.py` — torch usage as a reference pattern only (no shared code).

## Verification

- **Unit (Phase-1 encode):** encode a known snapshot → assert `entities` is `[12, F]`,
  `ids.*` shapes, `action_mask` length `A`, **all `belief_weight == 1.0`**, `padding_mask`
  all-true.
- **Perspective symmetry:** `encode(perspective=p1)` vs `encode(perspective=p2)` on the same
  state swaps the mine/theirs blocks and slot flags consistently.
- **Mask correctness:** `action_mask` true exactly for the legal moves in `StateView.legal`,
  false otherwise.
- **End-to-end index round-trip:** sample an index where `action_mask` is true →
  `action_space` → choice string → `SimClient.step` accepts it (no `SimError`
  illegal-choice). Proves Encoder, action_space, and SimClient agree on the numbering.
- **Snapshot tests:** the extended `sim-worker.test.ts` asserts the new raw facts appear.
- **Integration:** drive the `sim_client.py` self-test game; encode every state; assert no
  crashes and stable shapes across turns.
- **Batching:** collate `ObsBundle`s of differing `N` (mock Phase-4 candidates) → assert
  padding + `padding_mask` correct.
