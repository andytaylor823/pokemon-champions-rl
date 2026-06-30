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
    from collections.abc import Iterable, Sequence
    from pathlib import Path

    from obs_bundle import ObsBundle
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


def _current_encoder_schema_version() -> int | None:
    """The encoder's manual schema version, if it exposes one.

    The encoder does not expose ``ENCODER_SCHEMA_VERSION`` yet (flagged in
    ``docs/plans/encoder.md``). Until it does, this returns ``None`` and the
    same-width-layout half of the guard is inactive; the structural action-space-size
    check (§6) still fires. Once the constant lands, the guard activates automatically.
    """
    try:
        from encoder import ENCODER_SCHEMA_VERSION

        return int(ENCODER_SCHEMA_VERSION)
    except (ImportError, AttributeError):
        return None


def _bundle_dims(beta: ObsBundle) -> dict[str, int]:
    """Structural feature dims of one (unbatched) ObsBundle — the recorded signature."""
    return {
        "entities_F": int(beta["entities"].shape[-1]),
        "field_Ff": int(beta["field"].shape[-1]),
        "sides_Fs": int(beta["sides"].shape[-1]),
        "scalars_Fg": int(beta["scalars"].shape[-1]),
        "move_slots": int(beta["ids", "moves"].shape[-1]),
        "action_mask_A": int(beta["action_mask"].shape[-1]),
    }


def _schema_fingerprint(tuples: Sequence[TrainingTuple] | deque[TrainingTuple]) -> dict:
    """Signature of the encoder/action-space schema the stored tuples assume.

    ``action_space_A`` is read from the *current* ``action_space`` module, so it is
    recomputed on load and a resize (e.g. the planned 1089 -> 819 canonicalization) is
    caught. ``bundle_dims`` (from the first tuple) document the layout and guard against
    a corrupt file. ``encoder_schema_version`` activates the same-width-layout check once
    the encoder exposes the constant. See ``docs/plans/replay-buffer.md`` §6.
    """
    first = next(iter(tuples), None)
    return {
        "action_space_A": int(action_space.A),
        "encoder_schema_version": _current_encoder_schema_version(),
        "bundle_dims": _bundle_dims(first.beta) if first is not None else None,
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
            "schema": _schema_fingerprint(self._buf),
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
        """
        payload = torch.load(path, weights_only=False)

        if payload.get("format_version") != _FORMAT_VERSION:
            raise SchemaMismatchError(f"buffer file format_version {payload.get('format_version')!r} != current {_FORMAT_VERSION}; refusing to load.")

        stored = payload["schema"]
        current_a = int(action_space.A)
        if stored.get("action_space_A") != current_a:
            raise SchemaMismatchError(f"action-space size changed: saved A={stored.get('action_space_A')}, current A={current_a}. Refusing to load a stale-schema buffer.")

        current_ver = _current_encoder_schema_version()
        saved_ver = stored.get("encoder_schema_version")
        if current_ver is not None and saved_ver is not None and current_ver != saved_ver:
            raise SchemaMismatchError(f"encoder schema version changed: saved {saved_ver}, current {current_ver}. Refusing to load a stale-schema buffer.")

        tuples = payload["tuples"]
        saved_dims = stored.get("bundle_dims")
        if saved_dims is not None and tuples:
            actual = _bundle_dims(tuples[0].beta)
            if actual != saved_dims:
                raise SchemaMismatchError(f"stored bundle dims {saved_dims} != loaded tuple dims {actual} (corrupt file?).")

        buf = cls(capacity=int(payload["capacity"]), seed=None)
        buf._buf.extend(tuples)
        buf._rng.setstate(payload["rng_state"])
        return buf
