---
alwaysApply: false
paths: src/checkpoint.py
---

# Checkpoint — neutral save/load seam

`checkpoint.py` is a standalone module for persisting and restoring CVPN snapshots. It lives outside `trainer.py` so that `self_play`, `evaluation`, and any future consumer can load checkpoints **without importing the learner** — the same neutral-ground reasoning that put `TrainingTuple` in `training_types.py`.

## Interface

```python
CHECKPOINT_FORMAT_VERSION = 1

save_checkpoint(path, *, net, generation, optimizer=None, trainer_config=None, rng_state=None)
load_checkpoint(path, *, map_location="cpu") -> LoadedCheckpoint   # raises CheckpointFormatError on mismatch
```

```python
@dataclass(frozen=True)
class LoadedCheckpoint:
    net: CVPN              # rebuilt from stored CVPNConfig + weight-loaded
    generation: int
    optimizer_state_dict: dict | None
    trainer_config: dict | None
    rng_state: dict | None
```

## One rich format, all needs

The blob supersets every consumer's needs:

```
{
  "format_version": int,
  "generation": int,
  "model_config": dict,           # CVPNConfig via dataclasses.asdict (plain dict, not a pickled class)
  "model_state_dict": dict,
  "optimizer_state_dict": dict | None,
  "trainer_config": dict | None,
  "rng_state": dict | None,
}
```

| Consumer | Reads |
|----------|-------|
| Evaluation / warm-start | `model_config` + `model_state_dict` → rebuild net, ignore the rest |
| `Trainer.resume()` | Everything — weights, optimizer state, trainer config, RNG state |

Optional keys (`optimizer_state_dict`, `trainer_config`, `rng_state`) are simply `None` when not saved, so a lean "published" checkpoint and a full "resume" checkpoint share one format.

## CVPNConfig reconstruction

`model_config` is stored as a **plain dict** (`dataclasses.asdict(net.config)`), never a pickled class. On load, the net is rebuilt via `CVPN(CVPNConfig(**d))` then `load_state_dict`. This is forward-compatible: adding new `CVPNConfig` fields with defaults survives old checkpoints without migration.

## Fail-loud guard

`load_checkpoint` raises `CheckpointFormatError` on:

- `format_version` mismatch (saved != `CHECKPOINT_FORMAT_VERSION`).
- Missing required keys (`generation`, `model_config`, `model_state_dict`).
- Non-dict payload (corrupted file).

It never silently loads a file the current code cannot interpret.

## Pairing with buffer snapshots

The driver writes `gen_NNNN.pt` (checkpoint) and `buffer_gen_NNNN.pt` (buffer snapshot) **at the same generation** so the restored buffer matches the restored weights. The checkpoint module does not own this pairing — the driver does (`train_loop._drive`, every `checkpoint_interval` generations). `resume_training()` expects both files.

## `weights_only=False` caveat

Uses pickle-based deserialization (`torch.load(path, weights_only=False)`) because the payload contains arbitrary Python objects (config dicts, RNG state tuples). Only load files you produced yourself — a malicious `.pt` file can execute arbitrary code.

## Phase 1 pins

- `CHECKPOINT_FORMAT_VERSION = 1` — no migration story yet (Phase-1 checkpoints are disposable, per-stage).
- Scalar value head only — the stored `model_config` will naturally grow a vector-CFV field in Phase 4.

See `overview.mdc` (the driver that pairs checkpoint + buffer), `replay-buffer.mdc` (the paired buffer snapshot), `cvpn/overview.mdc` (what the loaded net is), `docs/plans/trainer.md` §3.
