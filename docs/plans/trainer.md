---
last_synced: a39ccc9
watches:
  - src/
  - docs/architecture/repo-architecture.md
---

# Trainer — Implementation Plan

> Module ref: `docs/architecture/repo-architecture.md` §3.8

## Status

Not yet started.

## Scope

Minibatch SGD: `L = ‖v̂−v_search‖² + CE(π̂, σ̄) + λ‖θ‖²`. Publishes checkpoints consumed by SelfPlay workers. Training and generation run concurrently.

## Notes from the SelfPlay design

> Added from the SelfPlay grilling — see `docs/plans/self-play.md` §4, §6.2, §7 and `docs/vibes-decisions.md` §8.4, §8.8.

- **Value target = the tuple's `value` (search-refined value), not z.** The value loss
  `‖v̂−v_search‖²` trains toward the bootstrapped search value. Each `TrainingTuple` *also* carries
  the final game result `z`, but in Phase 1 z is a **logged side signal only** (diagnostics; the
  door to a future z-blend / TD target), never the regression target (vibes 8.4).
- **Policy target densification.** Tuples store σ̄ **sparse** (≤k index→prob pairs); the Trainer
  densifies to a `[A]` vector for the cross-entropy term, then masks/zeros illegal entries
  consistent with the CVPN policy head (`self-play.md` §4).
- **Trainer owns the checkpoint format + publishing** consumed by SelfPlay (`self-play.md` §7).
  Proposed minimal format: `torch.save({"generation": int, "state_dict": ..., "config": CVPNConfig}, path)`.
  No checkpoint save/load exists in the repo yet (the poker toys keep `state_dict()` clones in
  memory only) — this is greenfield.
- **Per-curriculum-stage runs.** Drive each validation-curriculum stage (`self-play.md` §6.2) as
  **from-scratch** (the primary correctness proof) and, as a side experiment, **warm-start** from
  the previous stage's checkpoint (vibes 8.8). Warm-start needs checkpoint *load*, which is also
  required by Evaluation — so the checkpoint format is needed earlier than full concurrent training.

## Notes from the ReplayBuffer design

> Added from the ReplayBuffer grilling — see `docs/plans/replay-buffer.md` §4, §5, §12 and
> `docs/vibes-decisions.md` §8.10–§8.15.

- **The Trainer owns batching — the buffer is a dumb container.** `ReplayBuffer.sample(n)` returns a
  plain `list[TrainingTuple]`; the Trainer turns it into the minibatch the loss needs:
  (1) **collate** the `n` per-decision `ObsBundle`s into one batched bundle via the existing
  `collate_obs_bundles` (`src/obs_bundle.py`); (2) **densify** each `SparsePolicy` into a dense
  `[n, A]` cross-entropy target, masking illegal entries to match the policy head; (3) **stack** the
  scalar `value` fields into a `[n]` value target; (4) **move to the training device**. The buffer
  never imports `A` or touches `ObsBundle` internals — a schema/`A` change ripples into the Trainer
  (which owns the loss), not the storage layer.
- **The Trainer owns the warmup threshold.** "Don't start training until the buffer holds ≥ N
  tuples" is a Trainer/orchestrator config knob, not a buffer concern. The Trainer checks
  `len(buffer) >= warmup_threshold` before the first gradient step and never asks for more than the
  buffer holds (`buffer.sample(n)` raises `ValueError` on over-ask) (vibes 8.11).
- **Buffer-snapshot ↔ checkpoint coordination.** `ReplayBuffer` ships `save`/`load` for mid-stage
  resume (replay-buffer.md §5). To resume *consistently*, the Trainer/driver should snapshot the
  buffer at the **same cadence/generation as the net checkpoint**, so restored weights and restored
  data match. The buffer just provides `save`/`load`; pairing them is the Trainer's job.
- **Imports `TrainingTuple` from `src/training_types.py`** (the extracted seam module), not from
  `self_play` (vibes 8.15).
