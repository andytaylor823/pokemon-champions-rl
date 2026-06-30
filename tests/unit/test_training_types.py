"""Unit tests for training_types.py — the seam types between inner and outer loops.

Covers: SparsePolicy and TupleMeta construction, frozen immutability, edge cases
(empty policy, mismatched lengths), and torch.save/load serialization round-trip.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock

import pytest
import torch

from training_types import SparsePolicy, TrainingTuple, TupleMeta

# ---------------------------------------------------------------------------
# SparsePolicy
# ---------------------------------------------------------------------------


class TestSparsePolicyContract:
    """Verify SparsePolicy's data contract and edge-case behavior."""

    def test_empty_policy(self):
        """An empty support (no legal actions) should be representable."""
        sp = SparsePolicy(indices=(), probs=())
        assert len(sp.indices) == 0
        assert len(sp.probs) == 0

    def test_single_action_policy(self):
        """A deterministic single-action policy stores correctly."""
        sp = SparsePolicy(indices=(42,), probs=(1.0,))
        assert sp.indices == (42,)
        assert sp.probs == (1.0,)

    def test_multi_action_policy(self):
        """Multiple actions with fractional probabilities."""
        sp = SparsePolicy(indices=(0, 5, 10), probs=(0.5, 0.3, 0.2))
        assert sp.indices == (0, 5, 10)
        assert sp.probs == (0.5, 0.3, 0.2)

    def test_mismatched_lengths_not_validated(self):
        """SparsePolicy is a plain frozen dataclass — no __post_init__ validation.
        This test documents the current behavior: mismatched lengths are *silently accepted*.
        Callers must ensure consistency."""
        sp = SparsePolicy(indices=(0, 1, 2), probs=(0.5, 0.5))
        assert len(sp.indices) == 3
        assert len(sp.probs) == 2

    def test_frozen_rejects_mutation(self):
        """Frozen dataclass should reject attribute assignment."""
        sp = SparsePolicy(indices=(0,), probs=(1.0,))
        with pytest.raises(FrozenInstanceError):
            sp.indices = (1,)  # type: ignore
        with pytest.raises(FrozenInstanceError):
            sp.probs = (0.5,)  # type: ignore

    def test_equality(self):
        """Two SparsePolicy instances with identical fields should be equal."""
        a = SparsePolicy(indices=(1, 2), probs=(0.6, 0.4))
        b = SparsePolicy(indices=(1, 2), probs=(0.6, 0.4))
        assert a == b

    def test_inequality(self):
        """Different fields should produce inequality."""
        a = SparsePolicy(indices=(1,), probs=(1.0,))
        b = SparsePolicy(indices=(2,), probs=(1.0,))
        assert a != b


# ---------------------------------------------------------------------------
# TupleMeta
# ---------------------------------------------------------------------------


class TestTupleMetaContract:
    """Verify TupleMeta's data contract."""

    def test_fields_stored(self):
        meta = TupleMeta(generation=3, game_id=17, decision_idx=5, phase="move", side="p1")
        assert meta.generation == 3
        assert meta.game_id == 17
        assert meta.decision_idx == 5
        assert meta.phase == "move"
        assert meta.side == "p1"

    def test_frozen_rejects_mutation(self):
        meta = TupleMeta(generation=0, game_id=0, decision_idx=0, phase="move", side="p1")
        with pytest.raises(FrozenInstanceError):
            meta.generation = 1  # type: ignore

    def test_force_switch_phase(self):
        """Phase can be 'forceSwitch' — a valid non-move decision point."""
        meta = TupleMeta(generation=0, game_id=0, decision_idx=0, phase="forceSwitch", side="p2")
        assert meta.phase == "forceSwitch"
        assert meta.side == "p2"

    def test_equality(self):
        a = TupleMeta(generation=1, game_id=2, decision_idx=3, phase="move", side="p1")
        b = TupleMeta(generation=1, game_id=2, decision_idx=3, phase="move", side="p1")
        assert a == b


# ---------------------------------------------------------------------------
# TrainingTuple
# ---------------------------------------------------------------------------


class TestTrainingTupleContract:
    """Verify TrainingTuple's frozen contract and field access."""

    def _make(self, value: float = 0.5, z: float = 1.0) -> TrainingTuple:
        return TrainingTuple(
            beta=MagicMock(),
            value=value,
            policy=SparsePolicy(indices=(0, 1), probs=(0.7, 0.3)),
            z=z,
            meta=TupleMeta(generation=0, game_id=0, decision_idx=0, phase="move", side="p1"),
        )

    def test_fields_accessible(self):
        tt = self._make(value=0.75, z=-1.0)
        assert tt.value == 0.75
        assert tt.z == -1.0
        assert tt.policy.indices == (0, 1)
        assert tt.meta.phase == "move"

    def test_frozen_rejects_mutation(self):
        tt = self._make()
        with pytest.raises(FrozenInstanceError):
            tt.value = 0.0  # type: ignore
        with pytest.raises(FrozenInstanceError):
            tt.z = -1.0  # type: ignore

    def test_extreme_values(self):
        """Extreme float values should be storable without error."""
        tt = TrainingTuple(
            beta=MagicMock(),
            value=1e6,
            policy=SparsePolicy(indices=(0,), probs=(1.0,)),
            z=-1e6,
            meta=TupleMeta(generation=0, game_id=0, decision_idx=0, phase="move", side="p1"),
        )
        assert tt.value == 1e6
        assert tt.z == -1e6


# ---------------------------------------------------------------------------
# Serialization round-trip via torch.save/load
# ---------------------------------------------------------------------------


class TestTorchSerializationRoundTrip:
    """TrainingTuples are persisted by ReplayBuffer via torch.save. Verify the types
    survive a pickle round-trip (torch.save uses pickle under the hood)."""

    def test_sparse_policy_survives_torch_round_trip(self, tmp_path):
        sp = SparsePolicy(indices=(3, 7, 11), probs=(0.5, 0.3, 0.2))
        path = tmp_path / "sp.pt"
        torch.save(sp, path)
        loaded = torch.load(path, weights_only=False)
        assert loaded == sp
        assert isinstance(loaded, SparsePolicy)

    def test_tuple_meta_survives_torch_round_trip(self, tmp_path):
        meta = TupleMeta(generation=2, game_id=10, decision_idx=4, phase="forceSwitch", side="p2")
        path = tmp_path / "meta.pt"
        torch.save(meta, path)
        loaded = torch.load(path, weights_only=False)
        assert loaded == meta
        assert isinstance(loaded, TupleMeta)

    def test_empty_sparse_policy_survives_round_trip(self, tmp_path):
        sp = SparsePolicy(indices=(), probs=())
        path = tmp_path / "empty_sp.pt"
        torch.save(sp, path)
        loaded = torch.load(path, weights_only=False)
        assert loaded == sp
        assert len(loaded.indices) == 0
