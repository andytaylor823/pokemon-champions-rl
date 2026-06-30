---
last_synced: 43728cd
watches:
  - src/
  - docs/architecture/repo-architecture.md
  - docs/plans/search.md
---

# SelfPlay — Phase-1 Design (outer-loop data generation)

> **Module ref:** `docs/architecture/repo-architecture.md` §3.7 (SelfPlay — outer-loop data generation)
> **Design refs:** `docs/architecture/gt-cfr-theory.md` §11–§12 (bootstrapping, the two loops, training targets), `docs/plans/search.md` §7 (the `SearchResult` seam this module consumes), §13 (Phase-1 exit criteria)
> **Consumes (all built & tested):** `src/search/` (`search()`), `src/sim_client.py`, `src/cvpn.py`, `src/encoder.py`, `src/action_space.py`, `src/obs_bundle.py`
> **Produces (seam to the unbuilt outer loop):** a stream of `TrainingTuple` for `ReplayBuffer`/`Trainer`

## Status

**Design complete; not yet implemented.** This document is the complete Phase-1 specification:
responsibilities, the game loop, the training-tuple contract, the output seam, the validation
curriculum, the interfaces consumed, and the seams exposed. Tuning knobs chosen on intuition are
recorded in `docs/vibes-decisions.md` §8 and cross-referenced inline as *(vibes N.N)*.

## Context — what SelfPlay is and where it sits

SelfPlay is the **outer loop's data generator**. The inner loop (`search()`) is built and tested
end-to-end; SelfPlay is the bridge from that inner loop to the (unbuilt) training pipeline. It
plays full self-play games — each decision is one `search()` call — and emits the training tuples
the CVPN learns from. It is a pure **consumer + orchestrator**: it reimplements no game rules
(Showdown via `SimClient` is the sole source of truth) and no search (it calls `search()`).

Phase 1 is the **perfect-information regime** (`repo-architecture.md` §0.3): both teams are fully
revealed, belief = delta, scalar value head. SelfPlay needs **no belief model and no meta-priors**
— it only needs two concrete legal teams to start each game. The Phase-4 upgrade (hidden info,
MCCFR, vector value targets) changes the *shape* of the tuple's value field and activates the
belief model; it does **not** change SelfPlay's loop structure (§14).

```
   MatchupSource ─► new_battle ─► [ game loop: search ─► sample ─► step ] ─► terminal
        │                              │        │         │                    │
   (two legal teams)            (inner loop) (∝ σ̄)   (advance live      stamp z on the
                                                       handle)          game's tuples, flush
                                                                              │
                                                                    Iterator[TrainingTuple] ─► ReplayBuffer
```

## 1. Responsibility & scope

**Responsibility.** Generate training data by playing full self-play games. For each game: obtain
two teams from a `MatchupSource`, drive every decision through `search()`, advance the real game
on the engine, and emit `(β, v_search, σ̄, z)` tuples — one per side that faced a genuine choice —
once the game terminates.

**In scope (Phase-1 first build).**
- The single-process game loop over a live `SimClient` handle.
- Forced-decision skipping (§2.2), action sampling from σ̄ (§3), per-game buffering and the
  result-tag/flush (§4).
- The `MatchupSource` abstraction and the validation curriculum (§6).
- The generator output seam (§5) and the consumer-side checkpoint/net handling (§7).

**Out of scope / deferred (documented seams, not built first).**
- **Parallelism** — single-process, sequential games first; the worker-per-actor seam is §9
  *(vibes 11.3)*.
- **The checkpoint *store* and concurrent hot-reload** — Trainer owns checkpoint publishing and
  the on-disk format; SelfPlay's first build takes an in-process net (§7). Checkpoint *loading*
  (for warm-start and Evaluation) is needed early and is specified here as a consumer.
- **The outer-loop orchestrator** that wires SelfPlay → ReplayBuffer → Trainer → checkpoint →
  reload concurrently. SelfPlay is just the generator; orchestration is a separate concern
  (Trainer + a thin driver).
- **Belief / meta-priors / hidden info** — Phase 4 (§14).

## 2. The game loop

### 2.1 The live-handle timeline

`SimClient` distinguishes **live handles** (the real, advancing game; `session = null`) from
**search-session handles** (ephemeral clones search makes and discards). SelfPlay owns the *live*
handle; `search()` opens its **own** session from a `from_handle` and closes it internally
(`search.md` §10), never touching the live timeline.

- Start a game: `live, view = sim.new_battle(team_a, team_b, seed)`.
- Advance the real game: `res = sim.step(live, choices, seed)` returns a **new child handle**
  (`res.child`, also live). Then `sim.release(live)` the old handle and set `live = res.child`.
  *(Releasing the old live handle each step caps memory; the engine requires the explicit
  release to advance — clones are immutable snapshots.)*
- The real-timeline `step` seed is a **fresh** seed drawn from the per-game RNG (§8); this is the
  actual game roll. Chance seeds *inside* search are search's own concern.

### 2.2 Forced-decision skip (no CVPN, no CFR, no tuple)

A **forced decision** is a state where every acting side has exactly one legal action — pure
Pokémon ceremony (e.g. a forced switch to your only remaining Pokémon, a Choice-locked slot with
one legal target). It carries no strategic choice and its value is fully determined by its
successors, so SelfPlay **skips it entirely**:

```
if not view.terminal and every side in view.to_move has exactly one legal action:
    choices = { side: action_to_choice_contextual(argmax(legal_mask(view.legal[side], view.phase)))
                for side in view.to_move }
    advance the live handle with `choices` (fresh seed); DO NOT call search(); emit NOTHING
    loop (chains of forced moves auto-resolve)
```

No `search()`, no CVPN forward, no regret tables, no tuple. This is **stronger** than the current
in-search "single-legal-move skip" (`search.md` §8), which still does one `cvpn_value`.

> **Cross-module flag (search).** The same principle should hold *inside* the search tree: a
> forced node encountered during expansion should be **collapsed** (stepped through transparently,
> resolving any chance, to the next genuine decision/chance/terminal) rather than instantiated as
> a degenerate decision node with a CVPN eval. This is flagged into `search.md` §8 + §12 as a firm
> preference. Fallback if the in-tree collapse proves too invasive for the first build: at minimum
> drop the root short-circuit's CVPN eval and emit no tuple — but the half-measure is a known code
> smell. *(vibes 8.7)* See §11 for the concrete change list.

A decision where **one** side is forced but the **other** has a real choice is **not** a forced
decision — it is a genuine decision for the choosing side. SelfPlay calls `search()` normally
(search solves the degenerate `1×k` grid) and emits a tuple **only for the choosing side** (§4).

### 2.3 Per-decision procedure (genuine decisions)

For each non-terminal, non-forced state:

1. `result = search(view, sim, net, from_handle=live, config=search_config)`.
2. **Buffer** a `TrainingTuple` for each side that had ≥2 legal actions (§4) — held, not yet
   emitted (z is unknown until the game ends).
3. **Sample** one joint action per acting side from `result.strategy[side]` (∝ σ̄, temperature τ;
   §3). Translate each to a Showdown choice string via `action_space.action_to_choice_contextual`.
4. **Advance** the live handle with the sampled `choices` and a fresh seed (§2.1); `view` becomes
   the child's view.

Team preview is the first decision and is handled by exactly this path (it is a genuine decision —
choosing/ordering 4 of 6 — never forced).

### 2.4 Termination, draws, and the safety cap

- **Terminal:** `view.terminal` is true; `view.utility` is `{"p1": ±1, "p2": ∓1}` for a win, or
  `{"p1": 0, "p2": 0}` for a **draw** (engine returns 0/0 when there is no winner). Map utility to
  the per-side result `z ∈ {+1, 0, −1}` and stamp the game's buffered tuples (§4), then flush.
- **Draws are valid games** — keep their tuples with `z = 0` (the value targets are search values
  regardless).
- **Safety cap.** The engine ends battles on its own (no max-turn field is surfaced in `StateView`),
  but SelfPlay enforces a defensive `max_decisions` cap *(vibes 8.9; generous, e.g. 500)*. On
  exceeding it, **abort and discard** the game's buffered tuples with a warning — never fabricate
  an outcome z. Likewise, if `SimClient` raises mid-game, abort + discard that game and continue.
- **Defensive edge:** if a state is non-terminal with empty `to_move` (engine phase `"none"`),
  step it forward with empty choices rather than searching.

## 3. Action selection

Each acting side samples the joint action it actually plays from its own average strategy
`result.strategy[side]` (a `dict[action_index, prob]` over that side's top-k support). σ̄ is the
game-theoretically correct mixed strategy, so proportional sampling both plays well and yields
natural trajectory diversity. A **temperature** τ rescales the distribution (`p_i ∝ σ̄_i^{1/τ}`):

- **Self-play default: τ = 1.0** — sample exactly from σ̄ *(vibes 8.5)*.
- **Evaluation: τ → 0 (greedy/argmax)** — for measuring strength; **not** used during data
  generation (greedy play departs from the mixed equilibrium and kills diversity).

Both sides are driven by the **same** network (true self-play) and sample independently from their
respective σ̄ in the same solve.

> **Not solved here (search-internal, deferred).** "Top-k blindness" — a strong joint action the
> CVPN prior never ranks in the top-k never enters σ̄ at all, so SelfPlay can never sample it.
> Fixing that requires search-side action-coverage (progressive widening / ε-slot injection;
> `search.md` §12.4), not a SelfPlay knob. Temperature only diversifies among actions already in
> σ̄'s support.

## 4. The TrainingTuple (the seam to ReplayBuffer / Trainer)

The training tuple is the **stable boundary** between the inner and outer loops (the GT-CFR
analogue of AlphaZero's `(state, π, z)`; `repo-architecture.md` §3.7). One frozen dataclass
(per `agent/overview.mdc`: dataclasses over Pydantic for hot-path internals), now defined in the
shared `src/training_types.py` rather than this module (see §12 and `replay-buffer.md` §8):

```python
@dataclass(frozen=True)
class TrainingTuple:
    beta: ObsBundle          # encoded public state for one perspective: encode(view, side)
    value: float             # search-refined value from THIS side's perspective (+v for p1, -v for p2)
    policy: SparsePolicy     # σ̄ for this side, stored sparse: (indices, probs) over the top-k support
    z: float                 # final game result from this side's perspective: +1 / 0 / -1
    meta: TupleMeta          # generation/checkpoint id, game id, decision index, phase, side
```

- **`beta`** is the **encoded `ObsBundle`** (the β contract). Pre-encoding makes each tuple
  self-contained — ReplayBuffer/Trainer never touch the encoder. (A pre-encoded buffer is
  "invalidated" by encoder-schema changes, but such changes alter the CVPN's input dims and force
  a fresh run + fresh net anyway, so this costs nothing extra; the buffer is ephemeral/FIFO.)
- **`value`** is the **search-refined value** (`result.value` is p1-perspective; negate for p2).
  This is **bootstrapping** (`gt-cfr-theory.md` §11) — a target available at every decision,
  already better than the raw network because search folds in real terminals + the CFR operator.
  **Not** the game outcome. *(vibes 8.4)*
- **`z`** is the final game result, used **only** as a side label (diagnostics, and the door to a
  future search-value/z blend or TD target). It is **not** the training target in Phase 1.
  *(vibes 8.4)*
- **`policy`** stores σ̄ **sparse** (≤k index→prob pairs); the Trainer densifies to a `[A]` vector
  for the cross-entropy target. (Dense `[A]` = 1089 floats ≈ 4.3 KB/tuple; sparse is lossless and
  tiny.)

**Emission rule.** A decision emits **one full tuple per side that had ≥2 legal actions**:
normal turn → 2 tuples; force-switch (one side acting) → 1 tuple for the acting side; fully-forced
→ 0 (and the CVPN was never hit, §2.2). No value-only tuples in Phase 1.

**Per-game buffering & flush.** Because each tuple is tagged with z (unknown until terminal),
SelfPlay buffers a whole game's tuples, stamps z on each at terminal, then yields them. (This is
why emission is game-by-game, not decision-by-decision.)

> **Phase-4 doors (no rewrite).** (a) `value` becomes a **vector** of per-candidate CFVs (scalar
> is the length-1 degenerate case). (b) Value-only tuples (absent `policy`) re-enter for
> KataGo-style Playout Cap Randomization (the "deferred adaptive compute" note below). The schema
> is designed so both are additive.

## 5. The output seam — `run() -> Iterator[TrainingTuple]`

```python
def run(net: CVPN,
        matchup_source: MatchupSource,
        config: SelfPlayConfig) -> Iterator[TrainingTuple]:
    """Yield training tuples, flushing one game's worth at each terminal."""
```

A **generator** (matches `repo-architecture.md` §3.7's `stream[TrainingTuple]`). SelfPlay knows
nothing about `ReplayBuffer`; a thin driver (or the Trainer harness) pumps the iterator into the
buffer. Trivially unit-testable by draining the iterator over a few games. `SelfPlayConfig` carries
the temperature, `max_decisions`, master seed, number of games (or "infinite"), and the
`SearchConfig` to pass to `search()`.

## 6. `MatchupSource` & the validation curriculum

### 6.1 The `MatchupSource` seam

```python
class MatchupSource(Protocol):
    def sample(self, rng: random.Random) -> tuple[list[dict], list[dict]]:
        """Return (team_a, team_b) as legal list[dict] sets for new_battle."""
```

Teams are `list[dict]` with keys `species, item, ability, moves, nature, statPoints`
(the format `new_battle` expects; see `tests/conftest.py`). The loop is matchup-agnostic; the pool
grows without touching the loop. Phase-4 belief-driven matchup sampling slots in here later.

### 6.2 The validation curriculum (the spine of Phase-1 validation)

Phase-1 correctness is proven by a **deliberate curriculum** of fixed, hand-crafted matchups of
increasing complexity. At each stage, train a model and verify the architecture finds the
**known-correct solution** — operationally, *the favored team wins far more often* / play converges
to expert expectation. Each stage is a fixed `MatchupSource` (a single matchup, or a tiny set).

| Stage | Matchup | What it proves |
|---|---|---|
| **0** | all-Fire vs all-Grass, **2 moves** each | Smallest action space, obvious type edge, humanly-verifiable correct play. The architecture can learn *anything*. (May be made **simpler** if needed.) |
| **1** | same teams, **4 moves** each | Larger per-slot action space; still an obvious edge. |
| **2** | two "normal" teams, one expert-favored | Realistic complexity; favored side should win far more often. |
| **3** | two balanced normal teams | Full Phase-1 complexity; converges to sensible mixed play. |

**Per-stage initialization:** **from-scratch is the primary correctness proof** (random init →
the architecture discovers the solution at this complexity with no carryover; each stage an
independent, unconfounded experiment). **Warm-start from the previous stage is a side experiment**
measuring how much the curriculum accelerates learning. Both require checkpoint load (§7).

> **Cross-module flags.** The success **metric** (favored-team win-rate on a fixed matchup;
> convergence to known-correct play) lives in `Evaluation` — flagged into `evaluation.md`. The
> per-stage **training runs** (from-scratch + warm-start) are a `Trainer`/harness concern —
> flagged into `trainer.md`. *(vibes 8.8)*

## 7. Net source & checkpoints (consumer side)

- **First build:** `run()` takes an **in-process `CVPN`** object — testable before Trainer exists.
- **Warm-start + Evaluation need checkpoint *load*** (load stage N−1's weights, or a trained
  checkpoint to play head-to-head). Consumer side specified here: `CVPN()` then
  `net.load_state_dict(torch.load(path)["state_dict"])`. No checkpoint save/load exists in the repo
  yet (the poker toys keep `state_dict()` clones in memory only).
- **Concurrent hot-reload (deferred):** when generation and training run together, SelfPlay reloads
  the latest checkpoint **between games, never mid-game** (a game is played by one fixed net).
- **Producer side (the on-disk format + publishing) is owned by `Trainer`** — flagged into
  `trainer.md`. Proposed minimal format: `torch.save({"generation": int, "state_dict": ...,
  "config": CVPNConfig}, path)`.

## 8. Seeding & reproducibility

A master seed (in `SelfPlayConfig`) seeds a top-level RNG; each game draws a game seed; a per-game
RNG derives the team-preview/move/step seeds and the action-sampling draws. Reproducible: same
master seed + same net ⇒ same games. (Per-step `SimClient` seeds remain mandatory — cloning copies
PRNG state, so each real step must be reseeded; `search.md` §2.1.)

## 9. Parallelism (deferred)

Single-process, sequential games first. The scaling seam *(vibes 11.3)*: N worker processes, each
owning its own `SimClient` (its own Node subprocess) and a periodically-reloaded net, all feeding
one `ReplayBuffer`. The generator interface (§5) is unchanged — parallelism wraps it. Ray vs plain
multiprocessing is deferred until self-play throughput is the measured bottleneck.

## 10. Dependencies & reused interfaces (exact, as built)

- `src/sim_client.py` — `new_battle(team_a, team_b, seed) -> (handle:int, StateView)`;
  `step(handle, choices: dict[Side,str], seed: list[int]) -> StepResult{child:int, view, outcome}`;
  `view(handle)`; `release(handle)`; `close()`. Live handles vs search sessions per §2.1.
- `src/search/` — `search(view, sim, net, from_handle:int, config:SearchConfig|None) ->
  SearchResult{strategy: dict[Side, dict[int,float]], value: float (p1), policy_target:
  dict[Side, np.ndarray[A]]}`.
- `src/action_space.py` — `legal_mask(request, phase) -> [A] bool`; `action_to_choice_contextual(idx)
  -> str`; constants `A`, `MOVE_PHASE_OFFSET`, etc.
- `src/encoder.py` — `encode(view, perspective: "p1"|"p2", belief=None) -> ObsBundle`.
- `src/cvpn.py` — `CVPN(config=None)`; `forward(obs) -> (policy_logits[A], value scalar in
  [-1,1])`. (SelfPlay uses the net only via `search()`; it does not call `forward` directly.)
- `src/obs_bundle.py` — `ObsBundle = TensorDict`; `collate_obs_bundles(list)` (used by the Trainer
  when batching stored β, not by SelfPlay).
- `StateView` (`src/state_types.py`) — `phase ∈ {teamPreview, move, forceSwitch, terminal, none}`,
  `to_move: list[str]`, `legal: dict[str, Any]`, `snapshot.turn:int`, `terminal: bool`,
  `utility: dict[str,float]|None` (draw = `{0,0}`).

## 11. Required changes to other modules (ripple from this design)

This SelfPlay design implies concrete changes to **already-built** modules. They are in scope
(the grilling was SelfPlay-focused, but its outcomes propagate upstream) and are gathered here so
the ripple lives in one place rather than only being discoverable inline.

**Search (`src/search/`, `docs/plans/search.md`) — forced-node collapse (firm).**
- *Why:* §2.2 — a forced decision carries no strategic choice and its value is fully determined by
  its successors, so it must never cost a CVPN forward or regret tables, at the root *or* inside
  the tree.
- *Root:* `src/search/core.py` `_single_move_result` currently returns `value = cvpn_value(net,
  view)` for a fully-forced decision. SelfPlay no longer calls `search()` on forced decisions, so
  that eval is dead for the SelfPlay path; the short-circuit should not hit the CVPN when no value
  is needed.
- *In-tree (the real work):* `src/search/expansion.py` node creation/expansion should **collapse**
  forced nodes — step transparently through the single legal action (resolving any chance) to the
  next genuine decision / chance / terminal, evaluating the CVPN only there, instead of
  instantiating a degenerate decision node.
- *Fallback (acceptable only if the in-tree collapse is too invasive for the first build, and
  flagged as a known code smell):* keep degenerate nodes but skip their CVPN/CFR cost and never
  emit from them.
- *Docs:* flag `search.md` §8 (adaptive compute) and §12 (open questions).

**`action_space` (`src/action_space.py`) — verify team-preview choice-string coverage.**
- *Why:* SelfPlay samples and *plays* team-preview actions (§2.3) via `action_to_choice_contextual`.
  The search integration test plays team preview with a literal `"team 1234"`, so it is unverified
  that `action_to_choice_contextual` covers the team-preview region.
- *Action:* confirm `action_to_choice_contextual` maps team-preview indices → `"team ...."` strings; add
  the mapping if missing. No change if already covered.

**`SimClient` (`src/sim_client.py`) — no change required for Phase 1.**
- `new_battle` / `step` / `view` / `release` already support the live-handle timeline (§2.1).
  Structured chance outcomes and snapshot completeness remain Phase-4 items, not needed here.

**`CVPN` / `Encoder` / `ObsBundle` — no schema change required.**
- `encode(view, side)` already yields the `ObsBundle` stored as β; the scalar value head already
  matches the Phase-1 value target; sparse→dense σ̄ is a Trainer-side batching concern.

**Downstream (new modules, listed for completeness, not "earlier"):** `ReplayBuffer`
(`add`/`sample` over the §4 `TrainingTuple`), `Trainer` (owns checkpoint format/publishing §7,
per-stage curriculum runs §6.2, value loss toward search value not z), `Evaluation` (favored-team
win-rate on the curriculum §6.2).

## 12. Module shape

> **Seam-type relocation (from the ReplayBuffer grilling, `replay-buffer.md` §8, vibes 8.15).**
> `TrainingTuple` / `SparsePolicy` / `TupleMeta` are **extracted to a new `src/training_types.py`**
> so the inner↔outer seam sits on neutral ground that `self_play`, `replay_buffer`, and `trainer`
> all import. `self_play.py` re-imports them from there; behavior is unchanged.

```
src/training_types.py
  TrainingTuple, SparsePolicy, TupleMeta   — frozen dataclasses (the outer-loop seam; extracted from self_play.py)

src/self_play.py
  (imports TrainingTuple, SparsePolicy, TupleMeta from training_types)
  MatchupSource (Protocol) + CurriculumMatchupSource / curated team pools
  SelfPlayConfig                           — frozen dataclass (temperature, max_decisions, seed, games, SearchConfig)
  run(net, matchup_source, config) -> Iterator[TrainingTuple]   — the game loop (generator)
  _is_forced(view) / _forced_choices(view) — the §2.2 skip
  _sample_action(strategy, rng, temperature) — the §3 sampler

data/curriculum/  (or a small module of hand-written teams)   — Stage 0–3 matchups

tests/unit/test_self_play.py        — forced-skip, sampler, emission rule, z-stamping, generator drain (mock net/sim or tiny config)
tests/integration/test_self_play.py — a full Stage-0 game to terminal with a real SimClient + random-weights CVPN, asserting well-formed tuples
```

## 13. Vibes-tagged decisions

Recorded in `docs/vibes-decisions.md` §8: 8.4 (value target = search value + z-tag), 8.5 (sample
∝ σ̄, temperature τ=1), 8.6 (tuple stores encoded ObsBundle + sparse σ̄), 8.7 (forced-decision full
skip; in-loop now, in-tree flagged; fallback noted), 8.8 (validation curriculum; from-scratch +
warm-start), 8.9 (single-process default + `max_decisions` cap).

## 14. Open questions & Phase-4 doors

1. **Vector value target** — Phase 4 makes `value` a per-candidate CFV vector; scalar is the
   degenerate length-1 case. Tuple value-shape change only (`repo-architecture.md` §5).
2. **Value-only tuples + Playout Cap Randomization** — harvest value everywhere, policy from full
   searches only; needs the `policy`-optional schema relaxation. Deferred (the adaptive-compute
   note below).
3. **Belief-driven matchups** — Phase-4 `MatchupSource` draws from the conditional sampler; Phase 1
   uses the fixed curriculum.
4. **Curriculum extent** — Stage 0 may need to be made simpler; later stages (more Pokémon, items,
   speed control, mega) are added empirically.
5. **Parallelism + checkpoint hot-reload** — §9, §7; deferred to measured-bottleneck.

## 15. Exit criteria (what "done" means)

Phase-1 SelfPlay is done when, wired to ReplayBuffer + Trainer + Evaluation: (1) a full self-play
game runs to terminal driving every genuine decision through `search()`, skipping forced
decisions, emitting well-formed tuples; and (2) on the **Stage-0** curriculum matchup, a
from-scratch agent self-improves until the favored team wins far more often / play converges to the
known-correct strategy (`search.md` §13, `gt-cfr-theory.md` §14). Then climb the curriculum;
once Stage 3 holds, the Phase-1 infrastructure is proven and the Phase-4 imperfect-information swap
begins.

## Deferred adaptive compute (decided in the CVPN grilling; see `cvpn.md`)

- **KataGo-style Playout Cap Randomization (deferred):** run most self-play decisions at a *small*
  search budget to finish games fast, and a sampled fraction (~25%) at the *full* budget — emitting
  policy targets (σ̄) *only* from the full searches, while every move still yields a value target.
  This resolves the value-vs-policy tension and guards against the genuinely sticky failure mode
  (under-searched policy targets biasing the net across generations). It is the concrete mechanism
  behind the "value-only tuples" Phase-4 door (§4, §14.2): a value-only tuple is what a *reduced*
  search emits. **Deferred** until self-play throughput is the measured bottleneck. Ref: KataGo
  arXiv:1902.10565. (The per-decision single-legal-move skip *is* planned now — it is the
  forced-decision skip of §2.2.)
