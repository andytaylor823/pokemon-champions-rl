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
