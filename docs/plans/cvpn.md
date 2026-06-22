---
last_synced: dba4d8c
watches:
  - src/
  - docs/architecture/state-encoding.md
  - docs/architecture/search-nn-interface.md
  - docs/architecture/repo-architecture.md
---

# CVPN (Counterfactual Value + Policy Network) — Design Spec (Phase 1, with Phase 4 hooks)

> Module ref: `docs/architecture/repo-architecture.md` §3.3
> Design refs: `docs/architecture/state-encoding.md` (token anatomy, CLS, value-head designs),
> `docs/architecture/search-nn-interface.md` (three-tier split, when the NN is called)

## Status

Design locked (grilling session), **implementation not started**. This spec is the unified
Phase-1/Phase-4 design with Phase 1 as the degenerate case. Phase 1 is built **now**, against
the current action space (`A = 1089`); the Phase-4 pieces are designed-in but pinned to trivial
values so the upgrade is a head swap, not a rewrite.

## Context — why this module, why now

The production data flow is:

```
SimClient → Encoder → CVPN → Search → SelfPlay → Replay → Trainer
 [built]    [built]   ◄── this ──►  (primary seam, blocked on CVPN)
```

`SimClient` and `Encoder` are built and tested. The CVPN is the **next foundational module** for
two reasons:

1. **It is the first and only consumer of the encoder's `ObsBundle`.** Today nothing reads the
   bundle in production; the encoder's "locked decisions" deliberately handed work to the CVPN
   (it *owns the embedding tables* for the integer IDs, and it *runs the attention step over the
   4 move-vectors*) — all currently unverified because no consumer exists. Building the CVPN
   closes the Encoder↔net contract from the consumer side.
2. **`Search` — the project's primary seam — cannot exist without it.** Per
   `search-nn-interface.md` §1, the net is called at exactly two points (node expansion → policy
   prior + value; leaf evaluation → cached value). MCTS/GT-CFR cannot evaluate a leaf or get a
   policy prior without the CVPN. Everything downstream (`SelfPlay`, `Trainer`) is in turn
   blocked on `Search`.

**Phase-pin philosophy (repo-architecture §0.3).** Every module is built so Phase 1
(perfect-info, scalar value, belief = delta) is the *same schema* as the Phase-4 GT-CFR north
star (belief-weighted candidates, vector CFV value head), with the Phase-4 pieces pinned to
trivial values. The CVPN honors this: the **backbone is shared across phases; only the value
head changes.**

## What the CVPN is

One shared Transformer backbone + two heads. It maps the encoder's `ObsBundle` to:
- a **policy prior** over the canonical joint-action index (guides search exploration), and
- a **value** (evaluates search leaves): a scalar in Phase 1, a per-candidate CFV vector in
  Phase 4.

Framework: **PyTorch** (`torch` 2.11, `tensordict` 0.13 — both installed).

## Interface contract

```
forward(obs: ObsBundle) -> (policy_logits: float[A], value: Value)
```

- **Input** — `obs` is the encoder's `ObsBundle` (a `TensorDict`; see `src/obs_bundle.py` and
  `docs/plans/encoder.md`). The CVPN consumes every key:

  | Key | Shape | How the CVPN uses it |
  |---|---|---|
  | `entities` | float `[N, 65]` | per-Pokémon continuous + one-hot features (incl. `belief_weight` as the last feature — do **not** re-add it) |
  | `ids.species` / `ids.ability` / `ids.item` | int64 `[N]` | index the CVPN's embedding tables |
  | `ids.moves` | int64 `[N, 4]` | index the (shared) move embedding table; 4 per Pokémon |
  | `field` | float `[13]` | the field (global-effects) token's raw features |
  | `sides` | float `[2, 11]` | the two side (asymmetric-effects) tokens' raw features |
  | `scalars` | float `[7]` | the meta (turn#/phase/whose-decision) token's raw features |
  | `action_mask` | bool `[A]` | masks illegal policy logits (illegal → `-inf`) |
  | `padding_mask` | bool `[N]` | true = real entity token; drives the Transformer key-padding mask |
  | `belief_weight` | float `[N]` | **Phase 4 only** — search uses it to aggregate per-candidate values; Phase-1 net ignores it (all 1.0, already inside `entities`) |
  | `slot_id` | int `[N]` | **Phase 4 only** — groups candidate tokens per opponent slot for the vector value head |

- **Output** — `policy_logits` is a full `[A]` vector over the canonical action index
  (`src/action_space.py`), with illegal entries at `-inf`; callers apply softmax (search prior)
  or cross-entropy against σ̄ (training). `value` is a scalar in `[-1, 1]` (Phase 1) or a CFV
  vector (Phase 4).

- **Internal structure (private, but designed-in)** — three tiers behind the single public
  `forward()`:
  ```
  backbone(obs)            -> embeddings[n_tokens, d_model]   # the expensive shared trunk
  policy_from(embeddings)  -> policy_logits[A]                # cheap; reads CLS
  value_from(embeddings)   -> value                           # cheap; reads CLS (Phase 1)
  ```
  Phase 1 exposes only `forward()`; the tiers exist internally so that exposing the full
  three-tier seam and swapping in the per-deal vector head later (`search-nn-interface.md` §7)
  is mechanical. See **Phase 1 → Phase 4** below.

## Locked decisions (with rationale)

### D1 — Backbone = Transformer, built now (not flat-MLP-first)
`state-encoding.md` §11.1 offers an MLP-first path ("ship a flat MLP, swap the Transformer in
later"), but `repo-architecture.md` §3.3 requires the backbone to be "byte-for-byte shared
across phases." Building the Transformer now resolves that in favor of *not* shipping a throwaway
net: the encoder already paid the token-layout tax (`entities[N,F]`, `padding_mask`, `slot_id`,
CLS-ready), and the Transformer's structural wins (weight-sharing across the 12+ Pokémon,
permutation-equivariance, cross-entity attention, native variable token count) are exactly what
Phase 4 needs. Cost accepted: Transformers are finickier to train (LayerNorm placement, LR
warmup, mask bugs) than an MLP before the loop is proven end-to-end.

### D2 — Token sequence = peer tokens `[CLS, field, side_p1, side_p2, meta, entities…]`
Every global object is its own **peer token** that entities attend to (not broadcast onto each
Pokémon, not a flat prefix — see `state-encoding.md` §5.3, §11.5). The split:
- **`field` token** — fully-**global** effects only: weather, terrain, gravity, Trick Room.
  (From `field[13]`.)
- **two `side` tokens** — **asymmetric** effects, one per side: Tailwind, screens, hazards,
  mega-used. (From `sides[2, 11]`.) Keeping these separate from `field` preserves the clean
  "field = symmetric, sides = asymmetric" semantics.
- **`meta` token** — global scalars: turn #, phase, whose-decision. (From `scalars[7]`.) A
  dedicated peer token was chosen over folding scalars into `field` (which would muddy the
  field=effects semantics) or into `CLS` (which by design carries no input data).
- **`CLS` token** — a learnable `nn.Parameter[d_model]`, **randomly initialized → trained**, read
  back from position 0 after attention (the BERT `[CLS]` pattern; `state-encoding.md` §6.1).
  The *same* learned vector is prepended to every input; it is **not** re-randomized per forward.
  What differs per input is the attention CLS performs over that input's tokens — it learns
  content-weighted pooling (attend to active threats, ignore fainted mons), which mean-pooling
  cannot. Both heads read CLS in Phase 1.

The 5-token prefix (`CLS, field, side_p1, side_p2, meta`) is always present; only the entity
tokens vary in count (`N = 12` at team preview, `8` afterward in Phase 1; variable in Phase 4).

### D3 — Move embeddings → attention-pool ("Option B"), with a REVISIT flag
The encoder emits 4 move IDs per Pokémon (`ids.moves[N,4]`); the CVPN embeds them and collapses
the 4 vectors into one per-Pokémon summary **via a small attention pooling**, before the
cross-entity Transformer. This shares move-embedding semantics across the 4 slots (no per-slot
weight duplication) and can learn move synergies (Protect + Fake Out). ⚠️ **REVISIT based on
model performance** — if move detail proves lost, escalate to per-move tokens ("Option C",
`search-nn-interface.md` §6 / `encoder.md` Decision 2), which lengthens the token sequence.

### D4 — Policy head = phase-split internally, unified `[A]` output externally
The action space has two disjoint phases (`action_space.py`): team preview
(`TEAM_PREVIEW_COUNT = 360` orderings) and the move phase (`MOVE_PHASE_COUNT = 729` slot-pairs).
Internally the CVPN uses **two heads** (one per phase) read from CLS, so neither head wastes
capacity on the other phase's always-masked region. `forward()` **writes both heads into their
respective regions unconditionally**, then applies `action_mask` — which is phase-exclusive by
construction (`legal_mask` sets `True` only in the active phase's region), so the inactive
phase's logits get masked to `-inf` without any explicit phase detection. This gives every
downstream consumer (Search prior, Trainer cross-entropy) one uniform distribution over the
canonical index that matches `action_space` and the search policy target σ̄ — with no
marginalization. (Rationale: a single flat head would predict a large always-masked region;
fully-factored per-slot heads don't fit team preview's permutation structure and force the joint
search target to be marginalized.)

### D5 — Value head = scalar from CLS
Phase 1 is perfect-info, so position value is a single number: `Linear(d_model → 1)(CLS)` →
`tanh` → `[-1, 1]` (game payoff is ±1; contrast the Kuhn toy's `tanh × 2` for its ±2 payoff).
The Phase-4 per-candidate CFV vector is a **later head swap** (D6 / Phase 1 → Phase 4). It is
deliberately **not** built now: in Phase 1 belief = delta (fixed real mons, no candidates), so
"per-candidate CFV" is ill-defined — `state-encoding.md` §6.3 makes the vector head a Phase-4
swap.

### D6 — Code structure = trunk + 2 heads behind one `forward()`
Internally separate the shared trunk (`backbone`) from the two readers (`policy_from`,
`value_from`); publicly expose only `forward()`. This realizes the phase-pin philosophy without
YAGNI: the per-deal value-caching machinery of the full three-tier seam
(`search-nn-interface.md` §7) is **not** built now (Phase-1 MCTS has no sampled deals, so
`value_from(deal)` would be pure ceremony), but because the trunk/head boundary already exists,
exposing the seam and replacing the value head in Phase 4 is mechanical.

### D7 — Build against `A = 1089` now; derive head widths from `action_space`
The action space will later be **canonicalized** (see "Decisions living in other modules"),
shrinking `A` to ~819. We build now against the current `A = 1089` and make the phase-split head
widths **derive from `action_space` constants** (`TEAM_PREVIEW_COUNT`, `MOVE_PHASE_COUNT`) rather
than hard-coded literals, so canonicalization (team-preview 360 → 90) resizes the heads for free
(retrain only). Do **not** hard-code 360/729/1089 in the CVPN.

### D8 — Categoricals = hybrid; CVPN owns the embedding tables
Per `encoder.md` Decision 1: the encoder emits **integer IDs** for high-cardinality categoricals
(species, ability, item, the 4 moves) and **one-hot/float** for low-cardinality ones (status,
nature, already inside `entities[65]`). The CVPN owns `nn.Embedding` tables for the four ID
categoricals, sized from `src/vocab.py` (`*_VOCAB.size`, which already includes the reserved
**id 0 = UNK/NONE**). Use `padding_idx=0` so the UNK/NONE/pad row contributes a fixed zero
vector. The move table is **shared** across the 4 move slots. Consequence: novel sets (unseen
species/move) degrade gracefully to the UNK embedding rather than crashing or needing a new slot.

### D9 — Config = frozen dataclass (with a flagged tension)
`CVPNConfig` is a `@dataclass(frozen=True)` per the repo rule
(`agent/overview.mdc`: action models, internal structs, **config objects → frozen dataclass**,
*not* Pydantic; this overrides the personal Pydantic preference). ⚠️ **Flag:** the Trainer plan
(`repo-architecture.md` §3.8) says "typed config via Pydantic." That tension is noted, not
resolved here — the CVPN's architecture config is an internal struct, so the repo rule applies.

## Architecture walkthrough (the forward pass)

With concrete Phase-1 dims (`d_model` tentative = 128):

1. **Embed IDs.** species `[N]→[N,32]`, ability `[N]→[N,16]`, item `[N]→[N,16]`,
   moves `[N,4]→[N,4,32]` (shared table). (Embedding dims tentative; see Hyperparameters.)
2. **Move attention-pool (D3).** `[N,4,32] → [N, move_summary_dim]`.
3. **Build entity token inputs.** `concat(entities[N,65], species_emb, ability_emb, item_emb,
   move_summary)` → `Linear(· → d_model)` → `entity_tokens[N, d_model]`.
4. **Build global token inputs** (type-specific input projections, each `→ d_model`):
   `field[13]→[1,d_model]`, `sides[2,11]→[2,d_model]`, `scalars[7]→meta[1,d_model]`,
   `CLS = nn.Parameter[d_model]`.
5. **Assemble sequence** `[CLS, field, side_p1, side_p2, meta, entity_0..N-1]` → `[5+N, d_model]`.
6. **Key-padding mask.** The 5-token prefix is always real; the entity portion comes from
   inverting `obs["padding_mask"]` (encoder uses `true = real`; torch's `src_key_padding_mask`
   uses `true = ignore`). Result `[5+N]`.
7. **Transformer backbone.** Pre-norm encoder stack (`n_layers` blocks, `n_heads`,
   FFN width `ffn_mult × d_model`); **no positional encodings on the token axis** (slot identity
   — active/back, side, left/right — is a per-token *feature*, not a position; `state-encoding.md`
   §8.6). Output `h[5+N, d_model]`; read `CLS_out = h[0]`.
8. **Policy (D4).** team-preview head + move-phase head from `CLS_out` → write both into `[A]`
   → apply `action_mask` (phase-exclusive, masks the inactive region to `-inf`) →
   `policy_logits[A]`.
9. **Value (D5).** `tanh(Linear(d_model→1)(CLS_out))` → scalar `[-1,1]`.

Batching: `obs` may be a single bundle or a `[B]`-batched one (`collate_obs_bundles` pads the
entity axis to max `N` and sets `padding_mask`); the attention mask makes padded entities inert,
so batched and unbatched per-sample outputs must match.

## Hyperparameters (tentative — to tune; all defaults from the Opus-4.6 design docs)

| Name | Default | Source / note |
|---|---|---|
| `d_model` | 128 | `state-encoding.md` §12 |
| `n_heads` | 4 | `state-encoding.md` §12 |
| `n_layers` | 3 | `state-encoding.md` §12 (3–6 typical at this scale) |
| `ffn_mult` | 4 | standard Transformer FFN expansion (§8.5) |
| `dropout` | 0.1 | conventional starting point |
| embed dim: species / move | 32 | `encoder.md` Decision 1 ("~16–32") |
| embed dim: item / ability | 16 | low-to-mid cardinality (118 / 140) |
| value activation | `tanh` | bounds to game payoff `[-1, 1]` |

## Phase 1 → Phase 4 (what swaps, what is shared)

| Piece | Phase 1 | Phase 4 | Changes? |
|---|---|---|---|
| Embeddings, input projections, Transformer backbone | as above | **identical** | no |
| Token sequence | CLS + field + 2 sides + meta + 12/8 entities | + opponent **candidate** tokens (belief weight < 1, grouped by `slot_id`) | encoder-side; backbone is token-count-agnostic |
| `policy_from` | reads CLS | reads CLS (unchanged) | no |
| `value_from` | scalar from CLS | **per-candidate CFV vector**; the three-tier `value_from(embeddings, deal)` concatenates a deal's selected candidate embeddings → small MLP (`search-nn-interface.md` §7) | **head swap** + expose the seam |
| Public surface | `forward()` only | `backbone`/`policy_from`/`value_from` exposed for search caching | additive |

## Decisions made this session that live in OTHER modules (not built here)

These came out of the grilling and are recorded where they belong (`search.md`, `self-play.md`,
`repo-architecture.md` §6):

- **Action-space canonicalization** → `action_space` (deferred follow-up, *not* in the CVPN
  build). Collapse equivalent orderings to one canonical action: team-preview lead-pair order and
  bench-pair order are immaterial (A+B ≡ B+A), and the double-faint forced-switch order is
  immaterial. Team preview `360 → C(6,2)·C(4,2) = 90`; `A → ~819`. The canonical→Showdown-string
  expansion is deterministic and lives in `action_space`. (Move-phase double-switch order is a
  smaller, state-dependent sub-case to handle in the mask.) The CVPN is built against `A = 1089`
  now (D7) and absorbs this for free when it lands.
- **Search single-legal-move skip** → `Search`. When the legal (canonical) action set has size 1,
  bypass search entirely — emit that action with policy mass 1.0 (value from a single NN eval if
  the training target needs it). Composes with canonicalization (double-faint "bring both back"
  collapses to one canonical action → auto-skipped).
- **All other / "intelligent" early-stopping is DEFERRED** → `Search` / `SelfPlay`. Researched
  the AlphaZero lineage: AlphaZero self-play uses a *fixed* 800-sim budget per move (no adaptive
  stop); AlphaGo match play used *dynamic time allocation* (optimize time across moves); Lc0
  "smart pruning" is a *move-preserving* early-abort (stop once the choice provably can't change);
  KataGo "Playout Cap Randomization" runs most self-play moves at a small budget and trains the
  policy only on the occasional full search. **All deferred** for now (incl. the σ̄-convergence
  early-stop). **TODO recorded:** study Lee-Sedol-match & Lc0 dynamic time allocation before
  building the 60 s-turn time manager. Sources: AlphaZero arXiv:1712.01815; KataGo
  arXiv:1902.10565; lczero.org/blog/2018/09/time-management.

## Deferred / open / REVISIT (record, do not decide now)

- **Move aggregation (D3)** — attention-pool now; REVISIT → per-move tokens (Option C) on
  evidence move detail is lost.
- **Value-head combination function (Phase 4)** — concat vs pool vs cross-attention over selected
  candidate embeddings (`search-nn-interface.md` §10.1). Empirical.
- **Per-slot vs joint-team candidates (Phase 4)** — `state-encoding.md` §12.3; `BeliefModel`'s
  call, affects only `value_from`, not the public `search()` interface.
- **Tokenization / embedding dims, `d_model`, heads, layers** — tuning behind the `ObsBundle`
  seam; defaults above.
- **Config framework tension (D9)** — frozen dataclass vs the Trainer plan's Pydantic note.

## Verification

- **Unit (`tests/unit/test_cvpn.py`)**, reusing the `tests/fixtures/real_snapshot.json` → encoder
  fixture pattern:
  - forward-pass shapes: `policy_logits[A]`, scalar value;
  - **mask correctness**: illegal indices → `-inf` logit / 0 probability; legal probabilities sum
    to 1; end-to-end round-trip — an index sampled where `action_mask` is true maps (via
    `action_space`) to a choice string `SimClient.step` accepts;
  - **phase-split routing**: a team-preview state lights only the team-preview region; a move
    state lights only the move region;
  - **value range** ⊂ `[-1, 1]`;
  - **gradient flow**: `loss.backward()` populates grads on backbone + both heads + embeddings;
  - **batch/padding invariance**: collate bundles of differing `N` → per-sample outputs equal the
    unbatched results; padded entities don't leak through attention.
- **Integration (`tests/integration/test_cvpn_pipeline.py`)**: drive the `sim_client` self-test
  game (session fixture in `tests/conftest.py`), `encode` every state, `forward` through the CVPN
  — assert no crashes, stable shapes across turns/phases, valid masked policy distribution.
- **Commands**: `pytest tests/unit/test_cvpn.py tests/integration/test_cvpn_pipeline.py`; full
  `pytest`; `ruff` / `black` / `mypy` per repo standards.

## Reuse (don't reinvent)

- `src/obs_bundle.py` — `ObsBundle` / `collate_obs_bundles` (the input contract).
- `src/vocab.py` — `*_VOCAB.size` for embedding-table sizes (id 0 reserved for UNK/NONE).
- `src/action_space.py` — `A`, `TEAM_PREVIEW_COUNT`, `MOVE_PHASE_COUNT` (derive head widths; D7).
- `src/encoder.py` + `tests/conftest.py` — produce the bundles and the test fixtures.
- `toy_examples/kuhn_poker/network.py` — the "shared trunk → policy + value heads" pattern, as a
  reference only (no shared code; that net is per-card-vector output, which differs here).

## Concrete reference numbers (as built today)

- Encoder feature dims: `entities` F = **65**, `field` = **13**, `sides` = **11** (per side),
  `scalars` = **7**. Status one-hot = 7, nature one-hot = 25 (inside `entities`).
- Action space: `A` = **1089** = team-preview **360** (`P(6,4)`) + move-phase **729**
  (27 × 27 per-slot pairs). Post-canonicalization (future): team preview → **90**, `A` → **~819**.
- Vocab sizes (incl. reserved id 0): species **187**, item **118**, move **552**, ability **140**,
  nature **26**.
- Token count `N` (entities): **12** at team preview, **8** afterward (Phase 1); variable Phase 4.
