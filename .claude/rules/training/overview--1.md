---
alwaysApply: false
paths: src/train_loop.py
---

# Training — Phase-1 outer-loop driver + learner

The outer loop turns self-play data into a better CVPN. Two modules split the work: `train_loop.py` (the thin driver/conductor) orchestrates the generation loop, and `trainer.py` (the stateful learner) owns the CVPN + optimizer and does one gradient step at a time.

## Data flow

```
SelfPlay.run()  ──add(tuples)──►  ReplayBuffer  ──sample(n)──►  Trainer.train_step()
(Iterator[TrainingTuple])         (FIFO window)                  (collate β, densify σ̄,
                                                                  MSE + masked-CE loss,
                                                                  AdamW step)
                                                                       │
                                                            save() ────┘
                                                               │
                                                       gen_NNNN.pt  ◄── paired with ──►  buffer_gen_NNNN.pt
```

`train_loop._drive()` orchestrates all three — SelfPlay, ReplayBuffer, and Trainer — and pairs checkpoint + buffer snapshots at the same generation.

## The generation loop

One **generation** = G games into the buffer → warmup gate → M gradient steps → checkpoint.

```python
run_training(net, matchup_source, sim, config) -> list[GenerationLog]
resume_training(matchup_source, sim, config, checkpoint_path, buffer_path) -> list[GenerationLog]
```

Both funnel into `_drive()`, which loops `start_gen .. n_generations`:

1. Play `games_per_generation` self-play games (fresh per-generation seed).
2. `buffer.add(new_tuples)`.
3. If `len(buffer) >= max(warmup_threshold, batch_size)`: do `train_steps_per_generation` gradient steps.
4. Every `checkpoint_interval` generations: publish `gen_NNNN.pt` + paired `buffer_gen_NNNN.pt`.

The **warmup threshold** is owned by the driver, not the buffer — the buffer never refuses a sample on policy grounds; it only raises on `n > len(self)`.

## Trainer interface

```python
class Trainer:
    def __init__(self, net: CVPN, config: TrainerConfig | None = None)
    def train_step(self, tuples: list[TrainingTuple]) -> TrainStepLog
    def save(self, path, generation, *, rng_state=None) -> None
    @classmethod
    def resume(cls, path, *, device="cpu") -> tuple[Trainer, int, dict | None]
```

The Trainer is a *stateful learner*, not the loop. This keeps `train_step` unit-testable on a fake batch (no SimClient, no games) and survives the future concurrent generation/training split.

## The loss

$$L = w_v \cdot \text{MSE}(\hat{v},\, v_\text{search}) \;+\; w_\pi \cdot \text{CE}_\text{masked}(\hat{\pi},\, \bar{\sigma})$$

- **Value target** = the search-refined value (bootstrapping), **never z**. `z` is logged as a diagnostic only.
- **Policy target** = sigma-bar (the search's average strategy), densified from `SparsePolicy` into `[n, A]`.
- **Masked CE guard** — the CVPN emits `-inf` logits at illegal actions; naive `0 * log(-inf) = NaN`. Fix: `log_softmax`, then `masked_fill` illegal log-probs to `0` before the multiply. Sigma-bar is already 0 on illegal actions, so the fill value is annihilated; the gradient is exactly `(softmax - σ̄)` on legal logits and 0 on illegal ones.
- **L2** via AdamW `weight_decay` (default 0.0) — not an explicit loss term.

## Batching discipline

The **Trainer** owns all tensor work on sampled tuples. The buffer returns raw `list[TrainingTuple]`:

1. Collate β via `collate_obs_bundles` → batched `ObsBundle [n, ...]`.
2. Densify each `SparsePolicy` → dense `[n, A]` target (0 on unlisted/illegal actions).
3. Stack `value` fields → `[n]` value target.
4. Move to device.

The buffer never imports the action-space size `A` or touches `ObsBundle` internals.

## Two config objects

| Config | Scope | Style | Key fields |
|--------|-------|-------|------------|
| `TrainerConfig` | Learner knobs | Pydantic `BaseModel` | `learning_rate`, `weight_decay`, `batch_size`, `value_weight`, `policy_weight`, `device` |
| `TrainLoopConfig` | Loop knobs | Pydantic `BaseModel` (composes `TrainerConfig` + `SelfPlayConfig`) | `games_per_generation`, `train_steps_per_generation`, `warmup_threshold`, `checkpoint_interval`, `n_generations`, `buffer_capacity`, `checkpoint_dir`, `master_seed` |

## Resume

`resume_training()` restores the Trainer (weights + AdamW state + saved `TrainerConfig`, with `device` overridden) and the buffer from a paired snapshot. The loop RNG is restored from `rng_state["loop_rng"]` so per-generation seeds continue the original sequence. On resume, **learner knobs come from the checkpoint**; only loop knobs and `trainer.device` are honored from the passed config.

## Seeding

`master_seed` → `loop_rng` → per-generation seed → per-generation `SelfPlayConfig.master_seed`. Each generation plays different games; same `master_seed` + same net = same games.

## Phase 1 pins

- Single-process, turn-based: play G games, then train M steps (no concurrent generation/training).
- Scalar value target only (no vector CFV head).
- No Evaluation module yet — `scripts/train.py` prints losses as the manual feasibility signal.

See `replay-buffer.mdc` (what the buffer stores), `checkpoint.mdc` (the save/load seam), `self-play/overview.mdc` (data generator), `self-play/training-tuple.mdc` (tuple schema), `docs/plans/trainer.md`.
