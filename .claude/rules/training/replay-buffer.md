---
alwaysApply: false
paths: src/replay_buffer.py
---

# ReplayBuffer — Phase-1 outer-loop data store

`replay_buffer.py` is a fixed-capacity FIFO sliding window over `TrainingTuple`s produced by SelfPlay and consumed by the Trainer. It is deliberately **thin**: it stores, evicts, samples, and persists tuples — and nothing else.

## Interface

```python
class ReplayBuffer:
    def __init__(self, capacity: int, seed: int | None = None)
    def add(self, tuples: Iterable[TrainingTuple]) -> None
    def sample(self, n: int) -> list[TrainingTuple]
    def __len__(self) -> int
    def save(self, path: str | Path) -> None
    @classmethod
    def load(cls, path: str | Path) -> ReplayBuffer   # raises SchemaMismatchError on mismatch
```

## FIFO contract

Storage = `collections.deque(maxlen=capacity)` of `TrainingTuple`. When full, each appended tuple evicts the single oldest (per-tuple FIFO). Capacity is counted in **tuples**, not games — tuples from one game enter together (SelfPlay flushes a game's worth at terminal) but age out individually.

## Sampling contract

Uniform, without-replacement, seeded via the buffer's own `random.Random(seed)`. Same seed + same add/sample sequence = same draws.

- `sample(n)` raises `ValueError` if `n > len(self)` — the caller (driver) gates on `len(buffer)` and never over-asks.
- `sample(n)` returns **shared, immutable references** — `TrainingTuple` is a frozen dataclass and `ObsBundle` is not mutated downstream, so no defensive copies.

## Thin-container discipline

The buffer does **not**:

- Collate `ObsBundle`s into batched tensors (Trainer's job via `collate_obs_bundles`).
- Densify `SparsePolicy` into dense `[n, A]` vectors (Trainer's job).
- Place tensors on a device (Trainer's job).
- Inspect or validate tuple fields.
- Import the action-space size `A` (except indirectly via the schema fingerprint).

**Deletion test:** remove the buffer and only the sliding window + sampling + persistence disappears — nothing about batching, the loss, or `A` smears into other modules.

## Schema fingerprint

`_current_schema()` builds a structural + manual fingerprint from **live module constants** (never from stored data):

| Key | Source | Catches |
|-----|--------|---------|
| `action_space_A` | `action_space.A` | Action-space resize |
| `encoder_schema_version` | `encoder.ENCODER_SCHEMA_VERSION` | Semantic-but-same-width change (e.g. feature reordering) |
| `entities_F`, `field_Ff`, `sides_Fs`, `scalars_Fg`, `move_slots` | Encoder dimension constants | Any feature-width change |

`save` records the fingerprint; `load` re-reads the live constants and compares. Mismatch raises `SchemaMismatchError` — it never silently loads stale-schema data. The fingerprint reads **code identity, not data shape**: a signature read back off the saved data could only ever match itself, so it could never detect a code change since save. The live module constants are the source of truth on both sides.

## Persistence

`torch.save` / `torch.load` (pickle-based, `weights_only=False`). Payload:

```
{
  "format_version": int,         # buffer file format (bump on payload-shape changes)
  "schema": dict,                # encoder/action-space fingerprint (see above)
  "capacity": int,
  "tuples": list[TrainingTuple], # FIFO order, oldest first
  "rng_state": tuple,            # random.Random.getstate() for exact-resume reproducibility
}
```

Two distinct version guards: `_FORMAT_VERSION` (the payload shape itself) and `schema` (the meaning of the tensors inside each tuple). Both must match for `load` to succeed.

## Fresh buffer per curriculum stage

No `clear()` method — the orchestrator constructs a new `ReplayBuffer` for each validation-curriculum stage. Each stage is an independent from-scratch experiment; old-stage tuples (off-distribution positions scored by a different net) must never bleed in. Even the warm-start side experiment carries net *weights*, not data.

## Concurrency

Single-threaded, no locking. Phase 1 plays and trains in turns — locking would guard a situation that cannot occur. The future concurrent shape (N producers → one buffer) will likely be multi-process, where an in-process lock is the wrong primitive.

See `self-play/training-tuple.mdc` (what the buffer stores), `overview.mdc` (who calls `sample`, warmup gate), `checkpoint.mdc` (paired snapshot pairing), `docs/plans/replay-buffer.md`.
