"""Unit tests for search.strategy — extract_average_strategy and build_policy_target.

Migrated from tests/unit/test_search.py:
  - TestStrategyExtraction

New gap tests:
  - Empty InfoSet returns {} (gap #13)
  - Zero-prob actions excluded from output dict (gap #14)
  - Empty strategy yields all-zeros policy target (gap #15)
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from action_space import A
from search import InfoSet, TurnNode, build_policy_target, extract_average_strategy

# ---------------------------------------------------------------------------
# Migrated tests
# ---------------------------------------------------------------------------


class TestStrategyExtraction:
    """Test average strategy extraction and policy target building."""

    def test_extracts_normalized_strategy(self):
        """Strategy sum is normalized to probabilities."""
        mock_view = MagicMock()
        node = TurnNode(handle=0, view=mock_view, to_move=["p1", "p2"])
        node.info = {
            "p1": InfoSet(
                actions=[10, 20, 30],
                prior=np.array([0.33, 0.33, 0.34]),
                regret=np.zeros(3),
                strategy_sum=np.array([100.0, 200.0, 300.0]),
                visits=np.zeros(3, dtype=np.int64),
            ),
        }

        sigma_bar = extract_average_strategy(node, "p1")
        assert abs(sigma_bar[10] - 1 / 6) < 1e-10
        assert abs(sigma_bar[20] - 2 / 6) < 1e-10
        assert abs(sigma_bar[30] - 3 / 6) < 1e-10

    def test_uniform_fallback_when_no_iterations(self):
        """When strategy sum is all zeros, fall back to uniform."""
        mock_view = MagicMock()
        node = TurnNode(handle=0, view=mock_view, to_move=["p1"])
        node.info = {
            "p1": InfoSet(
                actions=[5, 6],
                prior=np.array([0.5, 0.5]),
                regret=np.zeros(2),
                strategy_sum=np.array([0.0, 0.0]),
                visits=np.zeros(2, dtype=np.int64),
            ),
        }

        sigma_bar = extract_average_strategy(node, "p1")
        assert abs(sigma_bar[5] - 0.5) < 1e-10
        assert abs(sigma_bar[6] - 0.5) < 1e-10

    def test_policy_target_correct_shape(self):
        """Policy target should be a full [A] vector with mass on top-k."""
        mock_view = MagicMock()
        node = TurnNode(handle=0, view=mock_view, to_move=["p1"])
        node.info = {
            "p1": InfoSet(
                actions=[100, 200, 300],
                prior=np.array([0.33, 0.33, 0.34]),
                regret=np.zeros(3),
                strategy_sum=np.array([1.0, 2.0, 3.0]),
                visits=np.zeros(3, dtype=np.int64),
            ),
        }

        target = build_policy_target(node, "p1")
        assert target.shape == (A,)
        assert abs(target.sum() - 1.0) < 1e-6
        assert target[100] > 0
        assert target[200] > 0
        assert target[300] > 0
        assert target[0] == 0.0
        assert target[50] == 0.0


# ---------------------------------------------------------------------------
# New gap tests
# ---------------------------------------------------------------------------


class TestStrategyEdgeCases:
    """Edge cases for strategy extraction."""

    def test_empty_infoset_returns_empty_dict(self):
        """An InfoSet with no actions yields an empty strategy dict (gap #13)."""
        mock_view = MagicMock()
        node = TurnNode(handle=0, view=mock_view, to_move=["p1"])
        node.info = {"p1": InfoSet.empty()}

        sigma_bar = extract_average_strategy(node, "p1")
        assert sigma_bar == {}

    def test_zero_prob_actions_excluded(self):
        """Actions with zero probability in strategy_sum are excluded from output (gap #14)."""
        mock_view = MagicMock()
        node = TurnNode(handle=0, view=mock_view, to_move=["p1"])
        node.info = {
            "p1": InfoSet(
                actions=[10, 20, 30],
                prior=np.array([0.33, 0.33, 0.34]),
                regret=np.zeros(3),
                strategy_sum=np.array([10.0, 0.0, 5.0]),
                visits=np.zeros(3, dtype=np.int64),
            ),
        }

        sigma_bar = extract_average_strategy(node, "p1")
        # Action 20 has 0 mass in strategy_sum -> excluded
        assert 20 not in sigma_bar
        assert 10 in sigma_bar
        assert 30 in sigma_bar
        assert abs(sum(sigma_bar.values()) - 1.0) < 1e-10

    def test_empty_strategy_yields_zero_policy_target(self):
        """An empty InfoSet produces an all-zeros [A] policy target (gap #15)."""
        mock_view = MagicMock()
        node = TurnNode(handle=0, view=mock_view, to_move=["p1"])
        node.info = {"p1": InfoSet.empty()}

        target = build_policy_target(node, "p1")
        assert target.shape == (A,)
        assert target.sum() == 0.0

    def test_noop_infoset_strategy(self):
        """A noop InfoSet yields a single-entry strategy {0: 1.0}."""
        mock_view = MagicMock()
        node = TurnNode(handle=0, view=mock_view, to_move=["p2"])
        node.info = {"p2": InfoSet.noop()}
        # Give the noop some strategy_sum weight
        node.info["p2"].strategy_sum = np.array([5.0])

        sigma_bar = extract_average_strategy(node, "p2")
        assert sigma_bar == {0: 1.0}
