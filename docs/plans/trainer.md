---
last_synced: f6be4a3
watches:
  - src/
  - docs/architecture/repo-architecture.md
  - docs/plans/replay-buffer.md
  - docs/plans/self-play.md
---

# Trainer + Outer-Loop Driver — Phase-1 Design

> **Module ref:** `docs/architecture/repo-architecture.md` §3.8 (ReplayBuffer / Trainer / Evaluation), §4 (the checkpoint store seam)
> **Design refs:** `docs/plans/self-play.md` §4 (`TrainingTuple`), §7 (checkpoints); `docs/plans/replay-buffer.md` §4 (who batches), §5 (snapshot pairing); `docs/vibes-decisions.md` §8.4, §12
> **Consumes (built & tested):** `src/training_types.py`, `src/replay_buffer.py`, `src/self_play.py`, `src/cvpn.py`, `src/obs_bundle.py` (`collate_obs_bundles`), `src/curriculum.py`
> **Implemented in:** `src/trainer.py` (`Trainer`, `TrainerConfig`), `src/checkpoint.py` (save/load seam), `src/train_loop.py` (`run_training` driver, `TrainLoopConfig`), `scripts/train.py` (smoke entrypoint)

## Status

**Built & tested.** The Trainer closes the Phase-1 outer loop: with SelfPlay + ReplayBuffer
already in place, it is the link that turns the assembled pipeline into a *learning* system.
Decisions made by intuition/convention are recorded in `docs/vibes-decisions.md` §12 and
cross-referenced inline as *(vibes 12.N)*.

## Context — what the Trainer is and where it sits

The outer loop is `SimClient -> Encoder -> CVPN -> Search -> SelfPlay -> ReplayBuffer -> Trainer ->
checkpoint -> (reloaded by SelfPlay)`. Everything up to and including `ReplayBuffer` was built first;
the Trainer (and a thin driver) is the next foundational piece — and is strictly upstream of
`Evaluation`, which depends on the **checkpoint format the Trainer owns**.

```
   ReplayBuffer.sample(n) ──► Trainer.train_step(tuples) ──► checkpoint (gen_NNNN.pt)
   (list[TrainingTuple])      (collate β, densify σ̄,         + paired buffer snapshot
                               MSE+masked-CE loss, AdamW)      consumed by SelfPlay / Evaluation
        ▲                              ▲                                   │
        └──────── train_loop.run_training (the driver) ───────────────────┘
              (generate G games -> warmup gate -> M steps -> checkpoint -> repeat)
```

## 1. Responsibility & scope

**Trainer (the learner).** Hold the CVPN + optimizer; turn one minibatch of `TrainingTuple`s into one
gradient step; publish/restore checkpoints. Nothing else.

**Driver (`run_training`).** Own the generation loop, the warmup gate, the generation counter, and the
checkpoint↔buffer-snapshot pairing. Build a fresh ReplayBuffer per stage (no cross-stage bleed,
`replay-buffer.md` §12). It is deliberately thin so the later concurrent generation/training split
(`replay-buffer.md` §7) wraps it without touching the Trainer.

**In scope (Phase-1 first build).** The combined loss + AdamW step; sparse->dense σ̄ densification +
masked cross-entropy; β collation + device placement; the rich checkpoint format + save/resume; the
single-process generation loop + warmup gate + checkpoint cadence; a manual smoke entrypoint.

**Out of scope / deferred.** Concurrent generation/training (Phase 1 is turn-based, single-process);
`Evaluation` (separate module — consumes `checkpoint.load_checkpoint`); vector CFV value target
(Phase 4 — scalar is the length-1 degenerate case); Playout Cap Randomization / value-only tuples
(`self-play.md` deferred adaptive compute); prioritized replay (`replay-buffer.md` §14).

## 2. The learner — `src/trainer.py`

**The split (vibes 12.1).** The Trainer is a *stateful learner*, not the loop. This keeps a gradient
step unit-testable on a fake batch (no SimClient/games), survives the concurrent split, and is reused
unchanged across curriculum stages — only the driver's inputs vary. Rejected: the poker toy's fat
`Trainer.train()` (tangles data-generation orchestration into the learner).

```python
class TrainerConfig(BaseModel):                 # learner knobs (Pydantic — architecture §3.8)
    learning_rate: float = 1e-3
    weight_decay: float = 0.0                    # AdamW decoupled L2; default off (vibes 12.5)
    batch_size: int = 64
    value_weight: float = 1.0
    policy_weight: float = 1.0
    device: str = "cpu"                          # reproducible default; knob for mps/cuda

class Trainer:
    def __init__(self, net: CVPN, config: TrainerConfig | None = None)
    def train_step(self, tuples: list[TrainingTuple]) -> TrainStepLog
    def save(self, path, generation: int, *, rng_state: dict | None = None) -> None
    @classmethod
    def resume(cls, path, *, device="cpu") -> tuple[Trainer, int, dict | None]
```

**`train_step` owns batching (vibes 12.1; `replay-buffer.md` §4):** collate β via `collate_obs_bundles`,
densify each `SparsePolicy` into a dense `[n, A]` target, stack the search-refined `value` fields into
`[n]`, move to device, forward, loss, backward, AdamW step. The buffer stays a dumb container.

### 2.1 The loss

```
L = value_weight * MSE(v_hat, v_search)  +  policy_weight * CE_masked(pi_hat, sigma-bar)   (+ AdamW weight_decay)
```

- **Value** = `MSE(v_hat, tuple.value)` toward the **search-refined value** (bootstrapping), never `z`
  (vibes 8.4, 12.6). `z` is logged only as a diagnostic.
- **Policy** = masked soft-target cross-entropy (vibes 12.3). The CVPN emits `-inf` logits at illegal
  actions; the naive `-(target * log_softmax(logits)).sum()` computes `0 * (-inf) = NaN` (IEEE-754)
  there and poisons backprop. Fix: `log_softmax`, then `masked_fill` the illegal log-probs to `0`
  **before** the multiply. σ̄ is already 0 on illegal actions, so the fill value is annihilated; the
  gradient is exactly `(softmax - sigma-bar)` on legal logits and **0** on illegal ones — provably
  signal-free. Reuses the same `action_mask` the net used (`collate_obs_bundles` stacks it to `[n, A]`);
  every emitted tuple has ≥2 legal actions, so the normalizer is finite.
- **L2** via AdamW `weight_decay` (default `0.0`, vibes 12.5) — not an explicit loss term.

`train_step` returns a `TrainStepLog{value_loss, policy_loss, total_loss, grad_norm}` (grad_norm is a
no-clip measurement — a cheap NaN/explosion diagnostic).

## 3. The checkpoint seam — `src/checkpoint.py`

A **neutral module** (vibes 12.2), so `self_play`/`evaluation` load checkpoints without importing the
learner (the same reasoning that put `TrainingTuple` in `training_types`).

```python
CHECKPOINT_FORMAT_VERSION = 1
save_checkpoint(path, *, net, generation, optimizer=None, trainer_config=None, rng_state=None)
load_checkpoint(path, *, map_location="cpu") -> LoadedCheckpoint   # raises CheckpointFormatError on mismatch
LoadedCheckpoint = {net (rebuilt + weight-loaded), generation, optimizer_state_dict, trainer_config, rng_state}
```

**One rich format, supersets all needs (vibes 12.2).** Blob = `{format_version, generation,
model_config (CVPNConfig via dataclasses.asdict — a plain dict, not a pickled class), model_state_dict,
optimizer_state_dict, trainer_config, rng_state}`. Evaluation + warm-start read only `model_config` +
`model_state_dict` (rebuild `CVPN(CVPNConfig(**d))`, then `load_state_dict`); resume reads everything.
`load` **fails loudly** on a format-version mismatch or missing keys (like the buffer's schema guard).

## 4. The driver — `src/train_loop.py`

```python
class TrainLoopConfig(BaseModel):                # loop knobs; composes SelfPlayConfig (vibes 12.4)
    games_per_generation, train_steps_per_generation, warmup_threshold,
    checkpoint_interval, n_generations, buffer_capacity, checkpoint_dir, master_seed,
    trainer: TrainerConfig, self_play: SelfPlayConfig

def run_training(net, matchup_source, sim, config: TrainLoopConfig) -> list[GenerationLog]
```

One **generation** (vibes 12.4) = play G games into the buffer -> if `len(buffer) >= warmup_threshold`
do M gradient steps -> every `checkpoint_interval` generations publish `gen_NNNN.pt` + the paired
`buffer_gen_NNNN.pt` snapshot at the same generation (`replay-buffer.md` §5). A fresh per-generation
seed (from `master_seed`) makes successive generations play different games. The driver owns the
**warmup threshold** (vibes 8.11) and never over-asks the buffer. `GenerationLog` records mean losses,
buffer size, and the `z`-vs-`value` correlation diagnostic.

**Per-curriculum-stage runs (vibes 8.8).** The caller hands `run_training` a fresh `CVPN`
(from-scratch — the primary correctness proof) or a checkpoint-loaded net (warm-start — the side
experiment); the matchup source is a `curriculum.STAGE_*`. Each stage gets a fresh buffer.

## 5. Dependencies & reused interfaces (exact, as built)

- `src/training_types.py` — `TrainingTuple{beta: ObsBundle, value: float, policy: SparsePolicy, z, meta}`.
- `src/obs_bundle.py` — `collate_obs_bundles(list) -> ObsBundle` (stacks `action_mask` -> `[n, A]`).
- `src/cvpn.py` — `CVPN(config)`, `forward(obs) -> (policy_logits[n,A] masked, value[n] in [-1,1])`, `CVPNConfig`.
- `src/replay_buffer.py` — `ReplayBuffer(capacity, seed)`, `add`, `sample(n)`, `save`/`load`.
- `src/self_play.py` — `run(net, matchup_source, sim, SelfPlayConfig) -> Iterator[TrainingTuple]`.
- `src/curriculum.py` — `STAGE_0`, `STAGE_1` `MatchupSource`s.

## 6. Module shape

```
src/checkpoint.py                     — save/load seam + CHECKPOINT_FORMAT_VERSION + CheckpointFormatError
src/trainer.py                        — TrainerConfig, Trainer, TrainStepLog, _densify_policy
src/train_loop.py                     — TrainLoopConfig, GenerationLog, run_training (the driver)
scripts/train.py                      — manual smoke entrypoint (Stage 0/1)
tests/unit/test_checkpoint.py         — round-trip, config rebuild, lean vs rich, fail-loud guards
tests/unit/test_trainer.py            — masked-CE no-NaN, loss decreases, value-not-z, AdamW wiring, save/resume
tests/integration/test_train_loop.py  — tiny real Stage-0 run: checkpoints + paired buffer snapshots, finite losses
```

## 7. Verification (what gates the build — empirical)

1. **Unit (checkpoint):** save->load round-trips weights byte-for-byte; `CVPNConfig` rebuilds from the
   stored dict; lean (no-optimizer) and rich payloads both load; a forged `format_version` / missing
   key raises `CheckpointFormatError`.
2. **Unit (trainer):** masked CE produces **no NaN** with mostly-illegal (`-inf`) logits and all
   gradients stay finite (the headline guard); `train_step` on a fixed batch reduces total loss; with
   `policy_weight=0` the value loss collapses toward a target that differs from `z` (proves the target
   is `value`, not `z`); the optimizer is AdamW with the configured `lr`/`weight_decay`; empty batch
   raises; save->resume reproduces weights + generation + config.
3. **Integration (slow):** a 2-generation Stage-0 run on a real SimClient + fresh CVPN trains, writes
   `gen_NNNN.pt` + paired `buffer_gen_NNNN.pt` per generation, logs finite losses, and the published
   checkpoint + buffer snapshot reload.
4. **Manual smoke (`scripts/train.py`, not in CI):** a few generations on Stage 0 show policy + value
   loss falling — the actual Phase-1 feasibility signal (`self-play.md` §15). Full convergence metrics
   (favored-team win-rate, Elo) belong to the next module, `Evaluation`.

All of the above pass; the full suite (820 tests) is green and the new code is ruff-clean.

## 8. Open questions & future doors

1. **Vector value target (Phase 4)** — `value` becomes a per-candidate CFV vector; the MSE generalizes,
   scalar is the length-1 case (`self-play.md` §14, `repo-architecture.md` §5).
2. **Concurrent generation/training** — N self-play workers feeding one buffer while the Trainer trains
   (`replay-buffer.md` §7); the driver is the wrap point, the Trainer is unchanged.
3. **Checkpoint migration** — `format_version` exists; a real cross-version migration story is deferred
   (Phase-1 checkpoints are disposable/per-stage).
4. **z-blend / TD value target** — the `z`-vs-`value` diagnostic is the door; not a target in Phase 1.
5. **Warm-start curriculum experiment** — `Trainer.resume` (or `load_checkpoint` -> fresh Trainer)
   already supports loading stage N−1's weights; wiring the side experiment is a driver/harness concern.

## 9. Vibes-tagged decisions

Recorded in `docs/vibes-decisions.md` §12: 12.1 (learner + thin driver), 12.2 (rich checkpoint format in
a neutral module), 12.3 (masked policy cross-entropy / the 0*(-inf) guard), 12.4 (two configs +
generation definition), 12.5 (AdamW weight_decay default 0), 12.6 (value target = search value, not z).
Also restates §8.4 (value target) and §8.11 (warmup-threshold in the caller).
