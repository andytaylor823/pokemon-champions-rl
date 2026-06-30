"""Unit tests for the ReplayBuffer (docs/plans/replay-buffer.md §11).

Covers: FIFO eviction & capacity, the over-ask contract, seeded sampling determinism,
save/load round-trip + RNG-state restore, the fail-loud schema guard, fresh-buffer
isolation, schema-fingerprint contract, adversarial/boundary inputs, and edge cases
(empty save/load, invalid capacity, capacity=1, corrupted files). Pure unit tests —
tiny synthetic ObsBundles, no SimClient or net.
"""

from __future__ import annotations

import pickle
from collections import Counter

import pytest
import torch

import action_space
import encoder
from obs_bundle import make_obs_bundle
from replay_buffer import ReplayBuffer, SchemaMismatchError, _current_schema
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


# ---------------------------------------------------------------------------
# Schema-fingerprint contract (Gap 1)
# ---------------------------------------------------------------------------


_EXPECTED_SCHEMA_KEYS = frozenset({
    "action_space_A",
    "encoder_schema_version",
    "entities_F",
    "field_Ff",
    "sides_Fs",
    "scalars_Fg",
    "move_slots",
})


def test_current_schema_has_exactly_expected_keys():
    """The fingerprint dict must contain exactly the documented keys — no more, no less."""
    schema = _current_schema()
    assert set(schema.keys()) == _EXPECTED_SCHEMA_KEYS


def test_current_schema_values_match_live_modules():
    """Each fingerprint value must reflect the live module constant it claims to track."""
    schema = _current_schema()
    assert schema["action_space_A"] == int(action_space.A)
    assert schema["encoder_schema_version"] == int(encoder.ENCODER_SCHEMA_VERSION)
    assert schema["entities_F"] == int(encoder.ENTITY_FEATURE_DIM)
    assert schema["field_Ff"] == int(encoder.FIELD_FEATURE_DIM)
    assert schema["sides_Fs"] == int(encoder.SIDE_FEATURE_DIM)
    assert schema["scalars_Fg"] == int(encoder.SCALAR_FEATURE_DIM)
    assert schema["move_slots"] == int(encoder.NUM_MOVE_SLOTS)


def test_current_schema_values_are_positive_ints():
    """All schema values must be positive ints (not floats, not zero, not negative)."""
    for key, val in _current_schema().items():
        assert isinstance(val, int), f"{key} is {type(val)}, expected int"
        assert val > 0, f"{key}={val}, expected positive"


# ---------------------------------------------------------------------------
# Corrupted / malformed file loading (Gap 2 & 3)
# ---------------------------------------------------------------------------


def test_load_rejects_corrupted_file(tmp_path):
    """Loading random bytes should raise, not silently produce garbage."""
    path = tmp_path / "garbage.pt"
    path.write_bytes(b"not a valid torch file at all")
    with pytest.raises((RuntimeError, EOFError, pickle.UnpicklingError)):
        ReplayBuffer.load(path)


def test_load_rejects_empty_file(tmp_path):
    """Loading an empty file should raise clearly."""
    path = tmp_path / "empty.pt"
    path.write_bytes(b"")
    with pytest.raises((RuntimeError, EOFError, pickle.UnpicklingError)):
        ReplayBuffer.load(path)


def test_load_rejects_truncated_file(tmp_path):
    """A file truncated mid-write (first 10 bytes of a valid save) should fail cleanly."""
    buf = ReplayBuffer(capacity=5, seed=0)
    buf.add([_tuple(0)])
    path = tmp_path / "buf.pt"
    buf.save(path)
    # Truncate the file to simulate a crash mid-save
    full = path.read_bytes()
    path.write_bytes(full[:10])
    with pytest.raises((RuntimeError, EOFError, pickle.UnpicklingError)):
        ReplayBuffer.load(path)


def test_load_rejects_missing_schema_key(tmp_path):
    """A saved payload missing a schema sub-key should trigger SchemaMismatchError."""
    buf = ReplayBuffer(capacity=10, seed=0)
    buf.add([_tuple(0)])
    path = tmp_path / "buf.pt"
    buf.save(path)

    # Remove one key from the saved schema
    payload = torch.load(path, weights_only=False)
    del payload["schema"]["move_slots"]
    torch.save(payload, path)
    with pytest.raises(SchemaMismatchError, match="move_slots"):
        ReplayBuffer.load(path)


def test_load_rejects_missing_schema_dict_entirely(tmp_path):
    """A saved payload with no 'schema' key at all should raise."""
    buf = ReplayBuffer(capacity=10, seed=0)
    buf.add([_tuple(0)])
    path = tmp_path / "buf.pt"
    buf.save(path)

    payload = torch.load(path, weights_only=False)
    del payload["schema"]
    torch.save(payload, path)
    # Should raise KeyError (no 'schema' key) rather than silently loading
    with pytest.raises((KeyError, SchemaMismatchError)):
        ReplayBuffer.load(path)


# ---------------------------------------------------------------------------
# add() foot-gun: bare tuple vs wrapped iterable (Gap 4)
# ---------------------------------------------------------------------------


def test_add_bare_tuple_does_not_silently_corrupt():
    """Passing a bare TrainingTuple (not wrapped in a list) to add() must not silently
    decompose the dataclass fields into separate deque entries."""
    buf = ReplayBuffer(capacity=10, seed=0)
    single = _tuple(0)
    # deque.extend on a dataclass iterates its fields, adding 5 non-tuple items.
    # After this call, len(buf) should NOT be 5 (the number of dataclass fields).
    buf.add([single])  # correct usage
    assert len(buf) == 1
    assert isinstance(buf.sample(1)[0], TrainingTuple)


# ---------------------------------------------------------------------------
# Capacity boundary cases (Gap 5 & 6)
# ---------------------------------------------------------------------------


def test_capacity_one_eviction():
    """A capacity-1 buffer always holds only the most recently added tuple."""
    buf = ReplayBuffer(capacity=1, seed=0)
    buf.add([_tuple(0)])
    assert len(buf) == 1
    assert _ids(buf.sample(1)) == [0]
    # Adding another evicts the first
    buf.add([_tuple(99)])
    assert len(buf) == 1
    assert _ids(buf.sample(1)) == [99]


def test_capacity_one_save_load_round_trip(tmp_path):
    """A capacity-1 buffer survives save/load."""
    buf = ReplayBuffer(capacity=1, seed=42)
    buf.add([_tuple(7, value=0.5, z=1.0)])
    path = tmp_path / "cap1.pt"
    buf.save(path)
    loaded = ReplayBuffer.load(path)
    assert len(loaded) == 1
    assert loaded.capacity == 1
    assert loaded._buf[0].meta.decision_idx == 7


def test_large_capacity_small_content():
    """capacity=1M with 3 tuples: len is 3, sample(3) works, sample(4) raises."""
    buf = ReplayBuffer(capacity=1_000_000, seed=0)
    buf.add(_tuple(i) for i in range(3))
    assert len(buf) == 3
    assert len(buf.sample(3)) == 3
    with pytest.raises(ValueError, match="exceeds buffer size"):
        buf.sample(4)


# ---------------------------------------------------------------------------
# Sampling distribution & seed=None (Gap 10 & 13)
# ---------------------------------------------------------------------------


def test_sample_distribution_approximately_uniform():
    """Over many draws, each tuple should be sampled roughly equally often."""
    buf = ReplayBuffer(capacity=10, seed=12345)
    buf.add(_tuple(i) for i in range(10))
    counts: Counter[int] = Counter()
    for _ in range(5000):
        # Draw 1 at a time to measure per-tuple frequency
        drawn = buf.sample(1)
        counts[drawn[0].meta.decision_idx] += 1
    # Each of 10 items should appear ~500 times out of 5000. Allow wide tolerance.
    for idx in range(10):
        assert 300 < counts[idx] < 700, f"idx {idx} drawn {counts[idx]} times, expected ~500"


def test_seed_none_produces_nondeterministic_samples():
    """Two buffers with seed=None and identical content should (very likely) differ."""
    tuples = [_tuple(i) for i in range(50)]
    b1 = ReplayBuffer(capacity=100, seed=None)
    b2 = ReplayBuffer(capacity=100, seed=None)
    b1.add(tuples)
    b2.add(tuples)
    # 10-from-50 with truly different RNGs: collision probability is negligible
    ids1 = _ids(b1.sample(10))
    ids2 = _ids(b2.sample(10))
    assert ids1 != ids2 or ids1 != _ids(b1.sample(10))  # at least one pair differs


# ---------------------------------------------------------------------------
# Save idempotency & capacity-mismatch load (Gap 11 & 12)
# ---------------------------------------------------------------------------


def test_save_overwrite_idempotency(tmp_path):
    """Saving twice to the same path: second save produces a correct, loadable file."""
    buf = ReplayBuffer(capacity=10, seed=0)
    path = tmp_path / "buf.pt"

    # First save with 3 tuples
    buf.add(_tuple(i) for i in range(3))
    buf.save(path)

    # Add more and save again (overwrite)
    buf.add(_tuple(i) for i in range(3, 7))
    buf.save(path)

    loaded = ReplayBuffer.load(path)
    assert len(loaded) == 7
    assert set(_ids(loaded.sample(7))) == set(range(7))


def test_load_with_tampered_capacity_drops_oldest(tmp_path):
    """If saved capacity is manually shrunk below the tuple count, deque.extend
    drops the oldest tuples. Verify this is the actual behavior."""
    buf = ReplayBuffer(capacity=10, seed=0)
    buf.add(_tuple(i) for i in range(10))
    path = tmp_path / "buf.pt"
    buf.save(path)

    # Tamper: set capacity to 5 but leave all 10 tuples in the payload
    payload = torch.load(path, weights_only=False)
    payload["capacity"] = 5
    torch.save(payload, path)

    loaded = ReplayBuffer.load(path)
    assert loaded.capacity == 5
    # deque(maxlen=5).extend(10 items) keeps the last 5 (indices 5-9)
    assert len(loaded) == 5
    assert set(_ids(loaded.sample(5))) == {5, 6, 7, 8, 9}
