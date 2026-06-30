"""ReplayBuffer — Phase-1 outer-loop data store.

A fixed-capacity FIFO sliding window over the ``TrainingTuple``s produced by SelfPlay and
consumed by the Trainer. Deliberately *thin*: it stores, evicts, samples, and persists
tuples — and nothing else. Collation, sparse->dense policy densification, and device
placement are the Trainer's job; ``sample`` returns raw ``TrainingTuple``s.

Design: ``docs/plans/replay-buffer.md`` (storage §3, sampling §3, persistence §5, schema
fingerprint §6, concurrency seam §7).
"""

from __future__ import annotations

import random
from collections import deque
from typing import TYPE_CHECKING

import torch

import action_space

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from training_types import TrainingTuple

# Bump only when the on-disk save *payload* shape changes — distinct from the encoder
# schema fingerprint (§6), which tracks what the stored tensors mean.
_FORMAT_VERSION = 1


class SchemaMismatchError(RuntimeError):
    """Raised by :meth:`ReplayBuffer.load` when a saved buffer's schema fingerprint does
    not match the current encoder/action-space schema.

    A stored tuple's ``beta`` tensors and ``action_mask`` are only meaningful under the
    encoder/action-space version that produced them. Loading mismatched data would feed
    the network tensors that no longer mean what it reads, so we refuse rather than
    silently corrupt training (``docs/plans/replay-buffer.md`` §6).
    """


def _current_schema() -> dict:
    """Fingerprint of the encoder/action-space schema the *currently loaded code* produces.

    Recorded at ``save`` and re-read at ``load``, then compared. It is built from live
    module constants — ``action_space.A`` and the encoder's derived dim + version constants
    — never from the stored tuples. That distinction is the whole point: a fingerprint read
    back off the saved data could only ever match itself, so it could not detect a code
    change since save. Reading the live modules on both sides means a real schema drift is
    caught (``docs/plans/replay-buffer.md`` §6):

    - ``action_space_A`` — catches an action-space resize (e.g. the planned 1089 -> 819).
    - ``entities_F`` / ``field_Ff`` / ``sides_Fs`` / ``scalars_Fg`` / ``move_slots`` — the
      encoder's feature widths; catch any dimension change automatically.
    - ``encoder_schema_version`` — the encoder's manual version; catches a semantic-but-
      same-width change (e.g. feature reordering) the widths alone would miss.
    """
    import encoder

    return {
        "action_space_A": int(action_space.A),
        "encoder_schema_version": int(encoder.ENCODER_SCHEMA_VERSION),
        "entities_F": int(encoder.ENTITY_FEATURE_DIM),
        "field_Ff": int(encoder.FIELD_FEATURE_DIM),
        "sides_Fs": int(encoder.SIDE_FEATURE_DIM),
        "scalars_Fg": int(encoder.SCALAR_FEATURE_DIM),
        "move_slots": int(encoder.NUM_MOVE_SLOTS),
    }


class ReplayBuffer:
    """Fixed-capacity FIFO sliding window of training tuples.

    Args:
        capacity: Maximum number of tuples retained. When full, each added tuple evicts
            the single oldest one (per-tuple FIFO).
        seed: Seed for the sampling RNG. ``None`` uses an unseeded ``random.Random``.

    Not safe for simultaneous use from multiple threads/processes — Phase 1 plays and
    trains in turns (``docs/plans/replay-buffer.md`` §7).
    """

    def __init__(self, capacity: int, seed: int | None = None) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        self._capacity = capacity
        self._buf: deque[TrainingTuple] = deque(maxlen=capacity)
        self._rng = random.Random(seed)

    @property
    def capacity(self) -> int:
        return self._capacity

    def __len__(self) -> int:
        return len(self._buf)

    def add(self, tuples: Iterable[TrainingTuple]) -> None:
        """Append tuples; when full, the oldest are evicted one-for-one (FIFO)."""
        self._buf.extend(tuples)

    def sample(self, n: int) -> list[TrainingTuple]:
        """Uniform, without-replacement sample of ``n`` tuples.

        Returns raw ``TrainingTuple``s (the Trainer collates + densifies). Raises
        ``ValueError`` if ``n`` is negative or exceeds the current size — the caller
        gates on ``len(buffer)`` and never over-asks (``docs/plans/replay-buffer.md`` §3).
        """
        if n < 0:
            raise ValueError(f"n must be >= 0, got {n}")
        if n > len(self._buf):
            raise ValueError(f"sample(n={n}) exceeds buffer size {len(self._buf)}")
        return self._rng.sample(list(self._buf), n)

    def save(self, path: str | Path) -> None:
        """Persist contents + RNG state + schema fingerprint via ``torch.save`` (§5)."""
        payload = {
            "format_version": _FORMAT_VERSION,
            "schema": _current_schema(),
            "capacity": self._capacity,
            "tuples": list(self._buf),  # FIFO order, oldest first
            "rng_state": self._rng.getstate(),
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str | Path) -> ReplayBuffer:
        """Reconstruct a buffer, failing loudly on a schema-fingerprint mismatch (§6).

        Raises ``SchemaMismatchError`` if the file's format version or encoder/
        action-space schema does not match the current code — never silently loads
        stale-schema data.

        .. warning::
            Uses ``weights_only=False`` (pickle-based deserialization) because tuples
            contain arbitrary Python objects. Only load files you produced yourself —
            a malicious ``.pt`` file can execute arbitrary code.
        """
        # map_location="cpu" ensures portability: a buffer saved on a CUDA machine
        # can be loaded on a CPU-only one without a device error.
        payload = torch.load(path, weights_only=False, map_location="cpu")

        saved_format = payload.get("format_version")
        if saved_format != _FORMAT_VERSION:
            raise SchemaMismatchError(f"buffer file format_version {saved_format!r} != current {_FORMAT_VERSION}; refusing to load.")

        # Validate expected keys — a corrupted/partial file should raise a clear error
        # rather than a raw KeyError deep in the reconstruction logic.
        _REQUIRED_KEYS = ("schema", "capacity", "tuples", "rng_state")
        missing = [k for k in _REQUIRED_KEYS if k not in payload]
        if missing:
            raise SchemaMismatchError(f"corrupted or incomplete buffer file: missing key(s) {missing}")

        stored, current = payload["schema"], _current_schema()
        if stored != current:
            diffs = {k: (stored.get(k), current.get(k)) for k in stored.keys() | current.keys() if stored.get(k) != current.get(k)}
            raise SchemaMismatchError(f"encoder/action-space schema changed since save (saved -> current): {diffs}. Refusing to load a stale-schema buffer.")

        buf = cls(capacity=int(payload["capacity"]), seed=None)
        buf._buf.extend(payload["tuples"])
        buf._rng.setstate(payload["rng_state"])
        return buf
