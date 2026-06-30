---
last_synced: 393fbd5
watches:
  - src/
  - docs/architecture/repo-architecture.md
  - docs/plans/self-play.md
  - docs/plans/trainer.md
---

# ReplayBuffer — Phase-1 Design (outer-loop data store)

> **Module ref:** `docs/architecture/repo-architecture.md` §3.8 (ReplayBuffer / Trainer / Evaluation)
> **Design refs:** `docs/plans/self-play.md` §4 (the `TrainingTuple` contract), §6.2 (validation curriculum); `docs/vibes-decisions.md` §8.1 (FIFO sliding window), §8.6 (tuple stores encoded ObsBundle + sparse σ̄)
> **Consumes (new):** `src/training_types.py` — `TrainingTuple`, `SparsePolicy`, `TupleMeta` (extracted from `self_play.py`; see §8)
> **Produced into by:** `self_play.run()` (a stream of `TrainingTuple`)
> **Consumed by:** `Trainer` (`sample(n)` → minibatch; the Trainer collates + densifies — see §4)

## Status

**Design complete; not yet implemented.** This document is the complete Phase-1 specification:
responsibility, the interface seam, storage/eviction, sampling, the curriculum lifecycle,
persistence, and the test plan. Decisions made on intuition or convention are recorded in
`docs/vibes-decisions.md` §8.10–§8.15 and cross-referenced inline as *(vibes N.N)*.

## Context — what ReplayBuffer is and where it sits

The inner loop (`SimClient → Encoder → CVPN → Search`) and the data generator (`SelfPlay`) are
built and tested. `SelfPlay.run()` already yields `TrainingTuple`s, but **nothing consumes them
yet** — the outer-loop training side (`ReplayBuffer`, `Trainer`, `Evaluation`) is unbuilt.
`ReplayBuffer` is the **next foundational module**: the immediate downstream consumer of SelfPlay's
finished output seam, and a hard prerequisite of the `Trainer` (which samples minibatches from it).

It is a **fixed-capacity FIFO sliding window** over training tuples (CONTEXT.md "Replay buffer";
vibes §8.1): newest data enters, oldest (from weak early checkpoints) is evicted. Not everything
forever, not everything discarded each generation.

```
   SelfPlay.run()  ──add(tuples)──►   [ deque(maxlen=capacity) of TrainingTuple ]   ──sample(n)──►  Trainer
   (Iterator[TrainingTuple],          (newest in, oldest auto-evicted; per-tuple FIFO)              (collates β,
    one game's worth per flush)                                                                       densifies σ̄,
                                                                                                       gradient step)
```

The buffer is deliberately **thin** — the architecture calls it "thin but real" (§3.8). The whole
design discipline: keep it a dumb, well-tested container, and push every *policy* decision (how to
batch, when to start training, what device tensors live on) to the `Trainer`/orchestrator.

## 1. Responsibility & scope

**Responsibility.** Hold a bounded, most-recent window of `TrainingTuple`s; accept new tuples,
evict the oldest when full, hand back uniform random samples on request, and optionally
checkpoint/restore its contents to disk.

**In scope (Phase-1 first build).**
- The FIFO window (`add`, automatic eviction, `__len__`).
- Uniform, seeded, no-replacement sampling (`sample(n)`).
- Save/load of the buffer's contents with a fail-loud schema guard (§5, §6).

**Out of scope — explicitly *not* the buffer's job.**
- **Collation & densification** — turning sampled tuples into batched tensors (a batched
  `ObsBundle`, a dense `[n, A]` policy target) is the **Trainer's** job (§4). The buffer never
  imports the action-space size or touches `ObsBundle` internals.
- **Device placement** — the buffer stores CPU tensors as produced by the Encoder; the Trainer
  moves batches to the training device.
- **"When to start training"** — the warmup threshold (minimum tuples before the first gradient
  step) is a Trainer/orchestrator config knob, not the buffer's (§3) *(vibes 8.11)*.
- **Game-boundary bookkeeping** — eviction is per-tuple, not per-game (§2).
- **Concurrency** — single-threaded first; the simultaneous-access seam is deferred (§7)
  *(vibes 8.12)*.
- **Data quality / coverage** — top-k blindness, trajectory diversity, and target quality are
  upstream (`search.md` §12.4, `self-play.md` §3) concerns; the buffer stores whatever it is given.

## 2. The interface (the seam)

```python
class ReplayBuffer:
    def __init__(self, capacity: int, seed: int | None = None) -> None: ...

    def add(self, tuples: Iterable[TrainingTuple]) -> None:
        """Append tuples; when full, the oldest are evicted one-for-one (FIFO)."""

    def sample(self, n: int) -> list[TrainingTuple]:
        """Uniform, without-replacement sample of n tuples.
        Raises ValueError if n > len(self) (the caller must not over-ask)."""

    def __len__(self) -> int: ...

    def save(self, path: str | Path) -> None:
        """torch.save the contents + a schema fingerprint + the RNG state (§5)."""

    @classmethod
    def load(cls, path: str | Path) -> "ReplayBuffer":
        """Reconstruct a buffer. Raises if the saved schema fingerprint does not
        match the current encoder/action-space schema (§6)."""
```

**Invariants callers rely on.**
- **Per-tuple FIFO eviction** — once `len == capacity`, each `add` of one tuple drops exactly the
  single oldest tuple.
- **Uniform, no-replacement, seeded sampling** — within one `sample(n)` call no tuple repeats;
  the same `seed` + same call sequence ⇒ the same draws (reproducibility consistent with
  SelfPlay's `master_seed` discipline, `self-play.md` §8).
- **Over-ask raises** — `sample(n)` with `n > len(self)` raises `ValueError` rather than silently
  returning fewer; the Trainer gates on `len(buffer)` before sampling.
- **`load` fails loudly** — a schema-fingerprint mismatch raises; it never silently loads
  stale-schema data (§6).
- **Stored tuples are shared, immutable references** — `TrainingTuple` is a frozen dataclass and
  `ObsBundle` is not mutated downstream, so `sample` returns references without defensive copies.

**Config style.** The buffer takes plain constructor args — only two knobs (`capacity`, `seed`), so
no config object. `capacity` is *sourced from* the Trainer's config (the poker toys already bundle
`buffer_capacity` into `TrainingConfig`). The repo's "typed config via Pydantic" note (§3.8) scopes
to the **Trainer**; the buffer follows the frozen-dataclass/plain-args convention of `SelfPlayConfig`
and `SearchConfig`.

## 3. Storage, eviction & sampling

**Storage = `collections.deque(maxlen=capacity)` of `TrainingTuple`** *(vibes 8.10)*. O(1) append;
when full, appending auto-drops the single oldest tuple. This is the literal poker-toy pattern
(`toy_examples/*/self_play.py`: `deque[TrainingTuple](maxlen=buffer_capacity)`), scaled up to the
heavier Pokémon tuple.

- **Capacity is counted in tuples** (not games). Tuples from one game enter together (SelfPlay
  flushes a game's worth at each terminal) but age out individually — correct, because sampling
  draws i.i.d. decision points, not whole games.
- **Memory.** One Pokémon `TrainingTuple` carries a full `ObsBundle` (entities `[12, 65]`, an
  `action_mask [1089]`, the id tensors, etc.) plus a sparse σ̄ ≈ a few KB. At a large capacity
  (~100k tuples) the buffer is still well under 1 GB — comfortable for a single process.

**Sampling = uniform, without replacement, seeded** *(vibes 8.11)*. Implementation: draw `n`
distinct indices from the current contents with the buffer's own `random.Random(seed)` (e.g.
`random.Random.sample` over a snapshot of the deque). Uniform + FIFO is fixed by vibes §8.1;
**prioritized replay** (weight by surprise/TD-error) is a logged REVISIT, not Phase 1.

- **Over-ask contract.** `sample(n)` raises `ValueError` if `n > len(self)`. The "wait until there
  is enough data" rule lives in the Trainer: it checks `len(buffer) >= warmup_threshold` before the
  first gradient step and never asks for more than the buffer holds. *Mechanism* is in the buffer;
  *training-schedule policy* is in the caller.

## 4. The Trainer boundary (who batches)

`sample(n)` returns a **plain `list[TrainingTuple]`** — the buffer does no tensor work. The
**Trainer** turns that list into the minibatch its loss needs *(vibes 8.10; trainer.md)*:

1. **Collate β** — stack the `n` per-decision `ObsBundle`s into one batched `ObsBundle [n, …]` via
   the existing `collate_obs_bundles` ([`src/obs_bundle.py`](../../src/obs_bundle.py)) (pads the
   entity axis, sets `padding_mask`).
2. **Densify σ̄** — expand each `SparsePolicy` (≤k index→prob pairs) into a dense `[n, A]` target
   for the cross-entropy term, masking/zeroing illegal entries consistent with the CVPN policy head.
3. **Stack values** — gather the `n` scalar `value` fields into a `[n]` value target.
4. **Move to device.**

Keeping all four steps in the Trainer means the buffer never depends on the encoder schema or the
action-space size `A`, so a change to either ripples into the Trainer (which owns the loss), not the
storage layer. Deletion test: delete the buffer and only "the FIFO window + sampling + persistence"
disappears — nothing about batching, the loss, or `A` smears into other modules.

## 5. Persistence — `save` / `load`

The buffer supports disk persistence in the first build *(vibes 8.14)* so a long single-stage run
can resume after a crash or pause without losing accumulated tuples.

**Payload (one `torch.save` blob).** `torch.save` is pickle-based, handles tensors natively, and is
the same mechanism the Trainer will use for net checkpoints (`self-play.md` §7).
```
{
  "format_version": int,          # the buffer file format itself (bump on payload-shape changes)
  "schema": <fingerprint>,        # encoder/action-space schema fingerprint (§6)
  "capacity": int,
  "tuples": list[TrainingTuple],  # the deque contents, in FIFO order (oldest first)
  "rng_state": tuple,             # random.Random.getstate(), so a resumed run reproduces draws
}
```
`load` reconstructs `deque(self.tuples, maxlen=capacity)` and restores the sampling RNG state for
**exact-resume reproducibility** (identical subsequent draw sequence).

> **Coordination note (orchestrator's job, not the buffer's).** To resume a stage consistently, the
> buffer snapshot should be saved at the **same cadence/generation as the net checkpoint** so the
> restored buffer matches the restored weights. The buffer just provides `save`/`load`; pairing them
> is the Trainer/driver's responsibility (flagged into `trainer.md`).

## 6. The schema fingerprint (what makes "fail loudly" enforceable)

A stored tuple's β tensors and `action_mask [1089]` are only meaningful under the **encoder/
action-space schema** that produced them (vibes §8.6: "a pre-encoded buffer is invalidated by
encoder-schema changes"). Reloading a buffer under a changed schema would silently feed the network
tensors that no longer mean what it reads — corrupting training with no error.

**`load` therefore fails loudly** *(vibes 8.14)*: on a fingerprint mismatch it raises and refuses to
load, forcing a deliberate "start fresh" rather than a hidden bug. (A schema change already forces a
fresh net + fresh run anyway, so refusing costs nothing real.)

**Proposed fingerprint.** A structural signature plus a manual version, both checked:
- **Structural:** `action_space.A` (= 1089) and the `ObsBundle` feature dims — entity `F` (= 65),
  `field Ff`, `sides Fs`, `scalars Fg`, and the move-slot count. Catches any dimension change
  automatically.
- **Manual:** an `ENCODER_SCHEMA_VERSION` integer the encoder owns and bumps on a
  *semantic-but-same-width* change (e.g. reordering features within the same `F`), which the
  structural signature alone would miss.

> **Cross-module flag → `encoder.md` / `action-space.md`.** For the guard to work, the
> encoder/action-space modules must expose a stable **version/dim constant**. Today neither does
> (the dims live implicitly in `src/encoder.py` and `A` in `src/action_space.py`). Building the guard
> requires surfacing them. Until that exists, a pragmatic interim is to fingerprint on
> `action_space.A` + the observed `entities`/`field`/`sides`/`scalars` shapes of the first stored
> bundle.

## 7. Concurrency seam (deferred)

**Single-threaded, no locking** in the first build *(vibes 8.12)*. Phase 1 plays games and trains in
**turns**, never simultaneously (`self-play.md` §9, vibes §8.9), so locking would guard a situation
that cannot occur. The eventual concurrent shape (architecture §4: generation and training run
together) will most likely be **multi-process** (vibes §11.3: Ray vs multiprocessing, deferred until
throughput is the measured bottleneck) — where an in-process `threading.Lock` is the *wrong*
primitive and would give false confidence. The right later mechanism is a shared queue / inference
server feeding one buffer (N producers → one buffer). The generator/store interfaces here are
unchanged by that wrapping; only a new transport is added around them.

## 8. The seam type — `src/training_types.py` (new module)

`TrainingTuple` / `SparsePolicy` / `TupleMeta` are **extracted from `self_play.py` into a new
`src/training_types.py`** *(vibes 8.15)*. The architecture (§3.7) calls `TrainingTuple` "the stable
seam between the inner and outer loops"; it belongs on neutral ground, imported by all three of
`self_play`, `replay_buffer`, and `trainer`, rather than living inside the producer. This is a small
one-time refactor: move the three frozen dataclasses, then update `self_play.py` and its test imports
(behavior unchanged). The shape is exactly as built today
([self_play.py:39–71](../../src/self_play.py)):

```python
@dataclass(frozen=True)
class SparsePolicy:        # σ̄ stored sparse
    indices: tuple[int, ...]
    probs: tuple[float, ...]

@dataclass(frozen=True)
class TupleMeta:           # provenance
    generation: int
    game_id: int
    decision_idx: int
    phase: str
    side: str

@dataclass(frozen=True)
class TrainingTuple:       # the inner↔outer seam
    beta: ObsBundle        # encoded public state for one perspective
    value: float           # search-refined value target (this side's perspective)
    policy: SparsePolicy   # σ̄, sparse
    z: float               # final game result (Phase-1 side label only)
    meta: TupleMeta
```

The buffer **stores the whole tuple unchanged** (including `z` and `meta` — `z` for diagnostics,
`meta` for provenance); it never inspects fields.

## 9. Dependencies & reused interfaces

- `src/training_types.py` (new) — `TrainingTuple`, `SparsePolicy`, `TupleMeta`.
- `torch.save` / `torch.load` — persistence (handles the `ObsBundle` tensors inside each tuple).
- `random.Random` — seeded sampling and persisted RNG state.
- `src/obs_bundle.py` `collate_obs_bundles` — **not** called by the buffer; named here because the
  **Trainer** uses it on `sample()` output (§4).
- At runtime the buffer imports **nothing** from `encoder` / `cvpn` / `action_space` except the
  schema-fingerprint constants for the load guard (§6).

## 10. Module shape

```
src/training_types.py             — TrainingTuple, SparsePolicy, TupleMeta (extracted from self_play.py)
src/replay_buffer.py               — ReplayBuffer
tests/unit/test_replay_buffer.py   — see §11
```

## 11. Verification (the test plan that gates the build)

A small, fully deterministic unit suite (mock/tiny `TrainingTuple`s; no SimClient or net needed):

- **FIFO eviction & capacity.** Add `capacity + k` tuples; assert `len == capacity` and the first
  `k` are gone, newest retained, order is per-tuple FIFO.
- **Over-ask raises.** `sample(len + 1)` raises `ValueError`.
- **Seeded-sampling determinism.** Two buffers with the same `seed` and identical add/sample
  sequences return identical samples; a different seed differs. No repeats within one `sample(n)`.
- **Save/load round-trip.** `save` then `load` reproduces the exact contents (FIFO order) and, via
  the restored RNG state, the exact subsequent `sample` sequence.
- **Version-guard rejection.** A saved blob with a mismatched schema fingerprint raises on `load`.
- **Fresh-buffer isolation.** Two buffers (simulating two curriculum stages) share no state; adding
  to one never affects the other (guards the §12 "fresh buffer per stage" lifecycle).

## 12. Curriculum-stage lifecycle

A **brand-new buffer is constructed per validation-curriculum stage** *(vibes 8.13;
`self-play.md` §6.2)*. Each stage is an independent from-scratch experiment with different teams and
a freshly-initialized net, so old-stage tuples (off-distribution positions scored by a different net)
must never bleed in. There is **no `clear()` method** — the orchestrator simply builds a new
`ReplayBuffer` (construction is cheap). Even the warm-start side-experiment carries net *weights*,
not data, so it too gets a fresh buffer.

Within a single stage there is **no age-based purge**: the fixed-size FIFO window naturally ages out
tuples from weaker earlier networks (§3.8 explicitly forbids wholesale per-generation discard;
`generation` stays metadata, not an eviction key).

## 13. Vibes-tagged decisions

Recorded in `docs/vibes-decisions.md` §8: 8.10 (deque thin container), 8.11 (uniform/no-replace/
seeded sampling; warmup-in-Trainer), 8.12 (single-threaded; concurrency-seam-flagged), 8.13
(fresh-buffer-per-stage), 8.14 (save/load persistence in the first build + fail-loud schema guard),
8.15 (seam-type extraction to `training_types.py`). See also the prior §8.1 (FIFO sliding window) and
§8.6 (encoded ObsBundle + sparse σ̄).

## 14. Open questions & future doors

1. **Prioritized replay** — weight sampling by surprise/TD-error instead of uniform. Logged REVISIT
   (vibes §8.1); additive behind the same `sample(n)` interface.
2. **Persistence format versioning** — `format_version` exists for the payload; a real migration
   story is deferred (Phase-1 buffers are disposable, per-stage).
3. **Multi-process buffer** — the §7 concurrency seam: N self-play workers feeding one buffer via a
   queue/server; mechanism (Ray vs multiprocessing) deferred to measured bottleneck (vibes §11.3).
4. **Per-tuple importance weighting** — a door opened by storing the whole tuple (`z`, `meta`,
   `generation` all available) if a future target blend or recency weighting is wanted.
5. **Optional `stats()` hook** — size, fill ratio, per-generation histogram for Evaluation/logging.
   Not needed for the first build; cheap to add.

## 15. Exit criteria (what "done" means)

ReplayBuffer is done when: (1) the §11 unit suite passes (FIFO eviction, over-ask, seeded
determinism, save/load round-trip, version-guard rejection, fresh-buffer isolation); and (2) it
composes end-to-end — `for t in self_play.run(...): buffer.add([t])` fills the buffer, and
`buffer.sample(n)` yields a `list[TrainingTuple]` the (future) Trainer can collate + densify into a
minibatch. That end-to-end wiring is exercised for real once the Trainer exists.
