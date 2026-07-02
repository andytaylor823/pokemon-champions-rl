---
alwaysApply: false
paths: src/self_play.py
---

# TrainingTuple — inner/outer loop data contract

The `TrainingTuple` is the stable boundary between the inner loop (search at one decision) and the outer loop (training across self-play). The GT-CFR analogue of AlphaZero's `(state, pi, z)`.

## Schema

```python
@dataclass(frozen=True)
class TrainingTuple:
    beta: ObsBundle          # encoded state for one perspective
    value: float             # search-refined value (this side's perspective)
    policy: SparsePolicy     # σ̄ sparse: (indices, probs) over top-k support
    z: float                 # terminal game result (this side's perspective)
    meta: TupleMeta          # provenance metadata
```

## Field semantics

**`beta`** — pre-encoded `ObsBundle` via `encode(view, side)`. The tuple is self-contained; ReplayBuffer and Trainer never touch the encoder. A buffer of pre-encoded tuples is "invalidated" by encoder-schema changes, but such changes force a fresh net + fresh run anyway.

**`value`** — search-bootstrapped value from **this side's perspective**. The Phase-1 training target. p1 gets `result.value`; p2 gets `-result.value`. This is bootstrapping — a target already better than the raw network because search folds in terminals + the CFR operator. **Not the game outcome.**

**`policy`** — `SparsePolicy(indices: tuple[int, ...], probs: tuple[float, ...])` over the top-k support of sigma-bar. Frozen dataclass. The Trainer densifies to a full `[A]` vector for the cross-entropy target. Sparse storage is lossless and tiny (<=k entries vs 1089 floats dense).

**`z`** — terminal game result from this side's perspective: `+1` (win), `0` (draw), `-1` (loss). A side label only in Phase 1 — not the training target. Stamped at game end from `view.utility[side]`.

**`meta`** — `TupleMeta(generation, game_id, decision_idx, phase, side)`. All frozen. `phase` is one of `"teamPreview"`, `"move"`, `"forceSwitch"`.

## Emission rule

One tuple per side that had **>=2 legal actions** at that decision:

| Situation | Tuples emitted |
|-----------|---------------|
| Normal turn (both sides choosing) | 2 (one per side) |
| Unilateral decision (one side forced, one choosing) | 1 (choosing side only) |
| Fully forced decision | 0 (skipped entirely — no search, no CVPN) |

## Per-game buffering and flush

Tuples are held as mutable `_PendingTuple` (same fields minus `z`) during the game. At terminal, `z` is stamped from `view.utility` and each is converted to a frozen `TrainingTuple`, then yielded. Aborted games (max_decisions exceeded, SimError) discard all pending tuples — never fabricate an outcome.

## Phase-4 doors (additive, no rewrite)

- `value` becomes a **vector** of per-candidate CFVs (scalar is the length-1 degenerate case).
- Value-only tuples (absent `policy`) re-enter for KataGo-style Playout Cap Randomization.

See `self-play/overview.md` (game loop), `docs/plans/self-play.md` §4, `docs/architecture/gt-cfr-theory.md` §11–12.
