"""Unit tests for the ReplayBuffer (docs/plans/replay-buffer.md §11).

Covers: FIFO eviction & capacity, the over-ask contract, seeded sampling determinism,
save/load round-trip + RNG-state restore, the fail-loud schema guard, fresh-buffer
isolation, and edge cases (empty save/load, invalid capacity). Pure unit tests — tiny
synthetic ObsBundles, no SimClient or net.
"""

from __future__ import annotations

import action_space
import encoder
import pytest
import torch

from obs_bundle import make_obs_bundle
from replay_buffer import ReplayBuffer, SchemaMismatchError
from training_types import SparsePolicy, TrainingTuple, TupleMeta


# ---------------------------------------------------------------------------
# Helpers: minimal synthetic TrainingTuples
# ---------------------------------------------------------------------------


def _bundle(n: int = 2, f: int = 3, a: int = 5):
    """A minimal (unbatched) ObsBundle — shapes are what the schema fingerprint reads."""
    return make_obs_bundle(
        entities=torch.zeros(n, f),
        species_ids=torch.zeros(n, dtype=torch.long),
        ability_ids=torch.zeros(n, dtype=torch.long),
        item_ids=torch.zeros(n, dtype=torch.long),
        move_ids=torch.zeros(n, 4, dtype=torch.long),
        belief_weight=torch.ones(n),
        slot_id=torch.zeros(n, dtype=torch.long),
        field=torch.zeros(7),
        sides=torch.zeros(2, 4),
        scalars=torch.zeros(3),
        action_mask=torch.ones(a, dtype=torch.bool),
        padding_mask=torch.ones(n, dtype=torch.bool),
    )


def _tuple(i: int, value: float = 0.0, z: float = 0.0) -> TrainingTuple:
    """A tuple tagged with decision_idx=i so contents are identifiable after sampling."""
    return TrainingTuple(
        beta=_bundle(),
        value=value,
        policy=SparsePolicy(indices=(i,), probs=(1.0,)),
        z=z,
        meta=TupleMeta(generation=0, game_id=0, decision_idx=i, phase="move", side="p1"),
    )


def _ids(tuples) -> list[int]:
    return [t.meta.decision_idx for t in tuples]


# ---------------------------------------------------------------------------
# Storage & eviction
# ---------------------------------------------------------------------------


def test_fifo_eviction_and_capacity():
    buf = ReplayBuffer(capacity=3, seed=0)
    buf.add(_tuple(i) for i in range(5))
    assert len(buf) == 3
    # Oldest two (0, 1) evicted; newest three (2, 3, 4) retained.
    assert set(_ids(buf.sample(3))) == {2, 3, 4}


def test_add_accepts_any_iterable():
    buf = ReplayBuffer(capacity=10)
    buf.add([_tuple(0)])  # list
    buf.add(_tuple(i) for i in (1, 2))  # generator
    assert len(buf) == 3


def test_invalid_capacity_raises():
    with pytest.raises(ValueError):
        ReplayBuffer(capacity=0)
    with pytest.raises(ValueError):
        ReplayBuffer(capacity=-1)


# ---------------------------------------------------------------------------
# Sampling contract
# ---------------------------------------------------------------------------


def test_over_ask_raises():
    buf = ReplayBuffer(capacity=10, seed=0)
    buf.add(_tuple(i) for i in range(3))
    with pytest.raises(ValueError):
        buf.sample(4)
    # Exactly-full ask is fine.
    assert len(buf.sample(3)) == 3


def test_sample_negative_raises():
    buf = ReplayBuffer(capacity=10, seed=0)
    buf.add([_tuple(0)])
    with pytest.raises(ValueError):
        buf.sample(-1)


def test_sample_zero_returns_empty():
    buf = ReplayBuffer(capacity=10, seed=0)
    buf.add(_tuple(i) for i in range(3))
    assert buf.sample(0) == []


def test_sample_no_repeats_within_batch():
    buf = ReplayBuffer(capacity=100, seed=0)
    buf.add(_tuple(i) for i in range(50))
    drawn = _ids(buf.sample(20))
    assert len(set(drawn)) == 20  # without replacement


def test_seeded_determinism():
    tuples = [_tuple(i) for i in range(50)]
    b1 = ReplayBuffer(capacity=100, seed=123)
    b2 = ReplayBuffer(capacity=100, seed=123)
    b1.add(tuples)
    b2.add(tuples)
    assert _ids(b1.sample(10)) == _ids(b2.sample(10))


def test_different_seed_differs():
    tuples = [_tuple(i) for i in range(50)]
    b1 = ReplayBuffer(capacity=100, seed=1)
    b3 = ReplayBuffer(capacity=100, seed=999)
    b1.add(tuples)
    b3.add(tuples)
    # Exact-ordered match of 10-from-50 across different seeds is astronomically unlikely.
    assert _ids(b1.sample(10)) != _ids(b3.sample(10))


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_save_load_round_trip(tmp_path):
    buf = ReplayBuffer(capacity=20, seed=7)
    buf.add(_tuple(i, value=i * 0.1, z=1.0) for i in range(10))
    path = tmp_path / "buf.pt"
    buf.save(path)

    loaded = ReplayBuffer.load(path)
    assert len(loaded) == 10
    assert loaded.capacity == 20
    # Contents preserved in FIFO order (compare scalar fields, not the TensorDict β).
    orig = [(t.meta.decision_idx, t.value, t.z) for t in buf._buf]
    got = [(t.meta.decision_idx, t.value, t.z) for t in loaded._buf]
    assert got == orig
    # β tensors survive the round-trip.
    assert loaded._buf[0].beta["entities"].shape == buf._buf[0].beta["entities"].shape


def test_load_restores_rng_state(tmp_path):
    buf = ReplayBuffer(capacity=50, seed=7)
    buf.add(_tuple(i) for i in range(30))
    path = tmp_path / "buf.pt"
    buf.save(path)  # captures RNG state *before* any sampling

    loaded = ReplayBuffer.load(path)
    # Same contents + restored RNG state => identical subsequent draw sequence.
    assert _ids(buf.sample(8)) == _ids(loaded.sample(8))


def test_save_load_empty(tmp_path):
    buf = ReplayBuffer(capacity=5, seed=1)
    path = tmp_path / "empty.pt"
    buf.save(path)
    loaded = ReplayBuffer.load(path)
    assert len(loaded) == 0
    assert loaded.capacity == 5


# ---------------------------------------------------------------------------
# Fail-loud schema guard
# ---------------------------------------------------------------------------


def test_load_rejects_action_space_resize(tmp_path, monkeypatch):
    buf = ReplayBuffer(capacity=10, seed=0)
    buf.add(_tuple(i) for i in range(5))
    path = tmp_path / "buf.pt"
    buf.save(path)

    # Simulate the planned canonicalization resize (1089 -> 819, etc.).
    monkeypatch.setattr(action_space, "A", action_space.A - 1)
    with pytest.raises(SchemaMismatchError):
        ReplayBuffer.load(path)


def test_load_rejects_bad_format_version(tmp_path):
    buf = ReplayBuffer(capacity=10, seed=0)
    buf.add([_tuple(0)])
    path = tmp_path / "buf.pt"
    buf.save(path)

    payload = torch.load(path, weights_only=False)
    payload["format_version"] = 999
    torch.save(payload, path)
    with pytest.raises(SchemaMismatchError):
        ReplayBuffer.load(path)


def test_load_rejects_encoder_dim_change(tmp_path, monkeypatch):
    buf = ReplayBuffer(capacity=10, seed=0)
    buf.add(_tuple(i) for i in range(3))
    path = tmp_path / "buf.pt"
    buf.save(path)

    # Simulate widening an entity feature in the encoder (an automatic-catch width change).
    monkeypatch.setattr(encoder, "ENTITY_FEATURE_DIM", encoder.ENTITY_FEATURE_DIM + 1)
    with pytest.raises(SchemaMismatchError):
        ReplayBuffer.load(path)


def test_load_rejects_encoder_version_bump(tmp_path, monkeypatch):
    buf = ReplayBuffer(capacity=10, seed=0)
    buf.add(_tuple(i) for i in range(3))
    path = tmp_path / "buf.pt"
    buf.save(path)

    # Simulate a semantic-but-same-width encoder change announced via the manual version.
    monkeypatch.setattr(encoder, "ENCODER_SCHEMA_VERSION", encoder.ENCODER_SCHEMA_VERSION + 1)
    with pytest.raises(SchemaMismatchError):
        ReplayBuffer.load(path)


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_fresh_buffers_are_isolated():
    """Two buffers (simulating two curriculum stages) share no state."""
    b1 = ReplayBuffer(capacity=10, seed=0)
    b2 = ReplayBuffer(capacity=10, seed=0)
    b1.add(_tuple(i) for i in range(3))
    assert len(b1) == 3
    assert len(b2) == 0
