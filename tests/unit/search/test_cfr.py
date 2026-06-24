"""Unit tests for search.cfr — regret matching+, CFR+ traversals, tree_value.

Migrated from tests/unit/test_search.py:
  - TestRegretMatching, TestCFRConvergence, TestDeepRecursion

New gap tests:
  - Unilateral node convergence (gap #9)
  - Empty grid does not crash (gap #10)
  - tree_value deep recursion matches manual calculation (gap #11)
  - Single-element regret array (gap #21)
  - Return-value accuracy for matching pennies (gap #22)
  - Dominant-action convergence (gap #28)
  - 1x1 deterministic grid (gap #29)
  - Extreme-magnitude and NaN/inf regret robustness (gap #31)
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from search import (
    ChanceNode,
    ChanceOutcome,
    InfoSet,
    TurnNode,
    cfr_update_recursive,
    extract_average_strategy,
    regret_matching,
    tree_value,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_node_2x2(payoff_grid: dict[tuple[int, int], float]) -> TurnNode:
    """Build a 2x2 simultaneous-move TurnNode from a payoff dict (p1's perspective)."""
    mock_view = MagicMock()
    mock_view.to_move = ["p1", "p2"]
    node = TurnNode(handle=0, view=mock_view, to_move=["p1", "p2"])
    node.info = {
        "p1": InfoSet(
            actions=[0, 1],
            prior=np.array([0.5, 0.5]),
            regret=np.zeros(2),
            strategy_sum=np.zeros(2),
            visits=np.zeros(2, dtype=np.int64),
        ),
        "p2": InfoSet(
            actions=[0, 1],
            prior=np.array([0.5, 0.5]),
            regret=np.zeros(2),
            strategy_sum=np.zeros(2),
            visits=np.zeros(2, dtype=np.int64),
        ),
    }
    node.expanded = True
    node.grid = {
        (i, j): ChanceNode(children=[ChanceOutcome(handle=100 * i + j, leaf_value_p1=v)])
        for (i, j), v in payoff_grid.items()
    }
    return node


# ---------------------------------------------------------------------------
# Regret matching+ tests (migrated)
# ---------------------------------------------------------------------------


class TestRegretMatching:
    """Test the regret matching+ strategy derivation."""

    def test_uniform_when_all_zero(self):
        regret = np.array([0.0, 0.0, 0.0])
        strategy = regret_matching(regret)
        np.testing.assert_allclose(strategy, [1 / 3, 1 / 3, 1 / 3])

    def test_uniform_when_all_negative(self):
        regret = np.array([-1.0, -2.0, -3.0])
        strategy = regret_matching(regret)
        np.testing.assert_allclose(strategy, [1 / 3, 1 / 3, 1 / 3])

    def test_proportional_to_positive_regret(self):
        regret = np.array([3.0, 1.0, 0.0])
        strategy = regret_matching(regret)
        np.testing.assert_allclose(strategy, [0.75, 0.25, 0.0])

    def test_single_positive_regret(self):
        regret = np.array([0.0, 5.0, 0.0])
        strategy = regret_matching(regret)
        np.testing.assert_allclose(strategy, [0.0, 1.0, 0.0])

    def test_probabilities_sum_to_one(self):
        regret = np.array([2.0, 3.0, 5.0, 0.0])
        strategy = regret_matching(regret)
        assert abs(strategy.sum() - 1.0) < 1e-10

    # --- Gap #21: single-element array ---
    def test_single_element_returns_one(self):
        """A single-action regret vector should always return [1.0]."""
        strategy = regret_matching(np.array([0.0]))
        np.testing.assert_allclose(strategy, [1.0])

        strategy_pos = regret_matching(np.array([5.0]))
        np.testing.assert_allclose(strategy_pos, [1.0])

    # --- Gap #31: extreme magnitudes ---
    def test_large_magnitude_regret(self):
        """Very large regret values should still produce a valid distribution."""
        regret = np.array([1e15, 1e14, 0.0])
        strategy = regret_matching(regret)
        assert abs(strategy.sum() - 1.0) < 1e-10
        # Action 0 should dominate
        assert strategy[0] > 0.9

    def test_tiny_magnitude_regret(self):
        """Very small positive regret values should still normalize correctly."""
        regret = np.array([1e-15, 1e-15])
        strategy = regret_matching(regret)
        np.testing.assert_allclose(strategy, [0.5, 0.5], atol=1e-10)

    def test_inf_regret_concentrates_on_inf_actions(self):
        """Inf in regret should concentrate mass uniformly on inf action(s)."""
        regret = np.array([np.inf, 1.0, 0.0])
        strategy = regret_matching(regret)
        assert strategy.shape == (3,)
        assert np.isfinite(strategy).all(), "Strategy must not contain NaN or inf"
        np.testing.assert_allclose(strategy, [1.0, 0.0, 0.0])

    def test_multiple_inf_regrets_split_evenly(self):
        """Multiple inf regrets should split mass uniformly among them."""
        regret = np.array([np.inf, np.inf, 5.0, 0.0])
        strategy = regret_matching(regret)
        assert np.isfinite(strategy).all()
        np.testing.assert_allclose(strategy, [0.5, 0.5, 0.0, 0.0])

    def test_empty_regret_returns_empty(self):
        """Empty regret array (InfoSet.empty()) must not crash with ZeroDivisionError."""
        strategy = regret_matching(np.array([]))
        assert strategy.shape == (0,)
        assert len(strategy) == 0


# ---------------------------------------------------------------------------
# CFR+ convergence tests (migrated + new)
# ---------------------------------------------------------------------------


class TestCFRConvergence:
    """Test that CFR+ converges to Nash equilibrium on toy games."""

    def _make_matching_pennies_node(self) -> TurnNode:
        return _make_node_2x2({(0, 0): 1.0, (0, 1): -1.0, (1, 0): -1.0, (1, 1): 1.0})

    def test_converges_to_nash(self):
        """Matching Pennies: average strategy converges to 50/50."""
        node = self._make_matching_pennies_node()
        for t in range(1, 1001):
            cfr_update_recursive(node, t)

        sigma_bar_p1 = extract_average_strategy(node, "p1")
        sigma_bar_p2 = extract_average_strategy(node, "p2")
        assert abs(sigma_bar_p1.get(0, 0) - 0.5) < 0.05
        assert abs(sigma_bar_p1.get(1, 0) - 0.5) < 0.05
        assert abs(sigma_bar_p2.get(0, 0) - 0.5) < 0.05
        assert abs(sigma_bar_p2.get(1, 0) - 0.5) < 0.05

    def test_converges_asymmetric_game(self):
        """Asymmetric zero-sum: Nash is (1/4, 3/4) for both players."""
        node = _make_node_2x2({(0, 0): 3.0, (0, 1): 0.0, (1, 0): 0.0, (1, 1): 1.0})
        for t in range(1, 2001):
            cfr_update_recursive(node, t)

        sigma_bar_p1 = extract_average_strategy(node, "p1")
        sigma_bar_p2 = extract_average_strategy(node, "p2")
        assert abs(sigma_bar_p1.get(0, 0) - 0.25) < 0.05
        assert abs(sigma_bar_p1.get(1, 0) - 0.75) < 0.05
        assert abs(sigma_bar_p2.get(0, 0) - 0.25) < 0.05
        assert abs(sigma_bar_p2.get(1, 0) - 0.75) < 0.05

    # --- Gap #22: return-value accuracy ---
    def test_matching_pennies_game_value_near_zero(self):
        """Matching Pennies game value is 0; cfr_update_recursive should return ~0."""
        node = self._make_matching_pennies_node()
        val = 0.0
        for t in range(1, 1001):
            val = cfr_update_recursive(node, t)
        assert abs(val) < 0.1, f"Game value should be ~0 for Matching Pennies, got {val}"

    # --- Gap #28: dominant-action convergence ---
    def test_dominant_action_concentrates_mass(self):
        """When action 0 strictly dominates for p1, regret concentrates on it."""
        # p1 payoff: action 0 always gives +1, action 1 always gives -1
        node = _make_node_2x2({(0, 0): 1.0, (0, 1): 1.0, (1, 0): -1.0, (1, 1): -1.0})
        for t in range(1, 501):
            cfr_update_recursive(node, t)

        sigma_bar_p1 = extract_average_strategy(node, "p1")
        assert sigma_bar_p1.get(0, 0) > 0.95, f"Dominant action should get >95% mass, got {sigma_bar_p1}"

    # --- Gap #29: 1x1 deterministic grid ---
    def test_1x1_grid_deterministic_strategy(self):
        """A 1x1 grid (single action per side) yields deterministic strategy {action: 1.0}."""
        mock_view = MagicMock()
        node = TurnNode(handle=0, view=mock_view, to_move=["p1", "p2"])
        node.info = {
            "p1": InfoSet.from_actions([7], np.array([1.0])),
            "p2": InfoSet.from_actions([3], np.array([1.0])),
        }
        node.expanded = True
        node.grid = {(0, 0): ChanceNode(children=[ChanceOutcome(handle=1, leaf_value_p1=0.42)])}

        for t in range(1, 101):
            cfr_update_recursive(node, t)

        sigma_bar_p1 = extract_average_strategy(node, "p1")
        sigma_bar_p2 = extract_average_strategy(node, "p2")
        assert sigma_bar_p1 == {7: 1.0}
        assert sigma_bar_p2 == {3: 1.0}


# ---------------------------------------------------------------------------
# Unilateral node and empty grid (gaps #9-10)
# ---------------------------------------------------------------------------


class TestCFRUnilateral:
    """CFR+ on nodes where one side has a single noop action."""

    def test_unilateral_converges_to_dominant(self):
        """p1 has 2 actions, p2 has noop. p1 should converge to the dominant action."""
        mock_view = MagicMock()
        node = TurnNode(handle=0, view=mock_view, to_move=["p1"])
        node.info = {
            "p1": InfoSet.from_actions([0, 1], np.array([0.5, 0.5])),
            "p2": InfoSet.noop(),
        }
        node.expanded = True
        # Action 0 gives +1, action 1 gives -1 (from p1's perspective)
        node.grid = {
            (0, 0): ChanceNode(children=[ChanceOutcome(handle=1, leaf_value_p1=1.0)]),
            (1, 0): ChanceNode(children=[ChanceOutcome(handle=2, leaf_value_p1=-1.0)]),
        }

        for t in range(1, 501):
            cfr_update_recursive(node, t)

        sigma_bar = extract_average_strategy(node, "p1")
        assert sigma_bar.get(0, 0) > 0.95, f"Dominant action should get >95% mass, got {sigma_bar}"

        # p2 noop: strategy_sum should be trivially [N] for some N > 0
        sigma_bar_p2 = extract_average_strategy(node, "p2")
        assert sigma_bar_p2 == {0: 1.0}

    def test_empty_grid_returns_zero(self):
        """When all cells failed (empty grid), CFR+ returns 0.0 without crashing."""
        mock_view = MagicMock()
        node = TurnNode(handle=0, view=mock_view, to_move=["p1", "p2"])
        node.info = {
            "p1": InfoSet.from_actions([0, 1], np.array([0.5, 0.5])),
            "p2": InfoSet.from_actions([0, 1], np.array([0.5, 0.5])),
        }
        node.expanded = True
        node.grid = {}  # All cells failed with SimError

        val = cfr_update_recursive(node, 1)
        assert val == 0.0


# ---------------------------------------------------------------------------
# Deep tree recursion (migrated + gap #11)
# ---------------------------------------------------------------------------


class TestDeepRecursion:
    """Regression: CFR+ must recurse through expanded ChanceOutcome.node refs."""

    def test_cfr_recurses_into_child_nodes(self):
        """Values from depth-2 nodes propagate back up through cfr_update_recursive."""
        mock_view = MagicMock()

        child_node = TurnNode(handle=10, view=mock_view, to_move=["p1", "p2"])
        child_node.info = {
            "p1": InfoSet(
                actions=[0, 1],
                prior=np.array([0.5, 0.5]),
                regret=np.zeros(2),
                strategy_sum=np.zeros(2),
                visits=np.zeros(2, dtype=np.int64),
            ),
            "p2": InfoSet(
                actions=[0],
                prior=np.array([1.0]),
                regret=np.zeros(1),
                strategy_sum=np.zeros(1),
                visits=np.zeros(1, dtype=np.int64),
            ),
        }
        child_node.expanded = True
        child_node.grid = {
            (0, 0): ChanceNode(children=[ChanceOutcome(handle=20, leaf_value_p1=0.9)]),
            (1, 0): ChanceNode(children=[ChanceOutcome(handle=21, leaf_value_p1=0.1)]),
        }

        root = TurnNode(handle=0, view=mock_view, to_move=["p1", "p2"])
        root.info = {
            "p1": InfoSet(
                actions=[0, 1],
                prior=np.array([0.5, 0.5]),
                regret=np.zeros(2),
                strategy_sum=np.zeros(2),
                visits=np.zeros(2, dtype=np.int64),
            ),
            "p2": InfoSet(
                actions=[0],
                prior=np.array([1.0]),
                regret=np.zeros(1),
                strategy_sum=np.zeros(1),
                visits=np.zeros(1, dtype=np.int64),
            ),
        }
        root.expanded = True
        root.grid = {
            (0, 0): ChanceNode(children=[ChanceOutcome(handle=10, leaf_value_p1=0.5, node=child_node)]),
            (1, 0): ChanceNode(children=[ChanceOutcome(handle=11, leaf_value_p1=0.3)]),
        }

        for t in range(1, 101):
            cfr_update_recursive(root, t)

        assert child_node.info["p1"].strategy_sum.sum() > 0, "CFR+ did not recurse into child"
        child_sigma = regret_matching(child_node.info["p1"].regret)
        child_value = float(child_sigma @ np.array([0.9, 0.1]))
        assert child_value > 0.5, f"Child value {child_value} should exceed stale leaf value 0.5"

    def test_tree_value_does_not_mutate_strategy_sum(self):
        """tree_value must be read-only — no strategy_sum perturbation."""
        mock_view = MagicMock()
        node = TurnNode(handle=0, view=mock_view, to_move=["p1", "p2"])
        node.info = {
            "p1": InfoSet(
                actions=[0, 1],
                prior=np.array([0.5, 0.5]),
                regret=np.array([2.0, 1.0]),
                strategy_sum=np.array([100.0, 200.0]),
                visits=np.zeros(2, dtype=np.int64),
            ),
            "p2": InfoSet(
                actions=[0, 1],
                prior=np.array([0.5, 0.5]),
                regret=np.array([1.0, 3.0]),
                strategy_sum=np.array([150.0, 50.0]),
                visits=np.zeros(2, dtype=np.int64),
            ),
        }
        node.expanded = True
        node.grid = {
            (0, 0): ChanceNode(children=[ChanceOutcome(handle=1, leaf_value_p1=1.0)]),
            (0, 1): ChanceNode(children=[ChanceOutcome(handle=2, leaf_value_p1=-1.0)]),
            (1, 0): ChanceNode(children=[ChanceOutcome(handle=3, leaf_value_p1=-1.0)]),
            (1, 1): ChanceNode(children=[ChanceOutcome(handle=4, leaf_value_p1=1.0)]),
        }

        p1_sum_before = node.info["p1"].strategy_sum.copy()
        p2_sum_before = node.info["p2"].strategy_sum.copy()
        val = tree_value(node)
        np.testing.assert_array_equal(node.info["p1"].strategy_sum, p1_sum_before)
        np.testing.assert_array_equal(node.info["p2"].strategy_sum, p2_sum_before)
        assert isinstance(val, float)

    # --- Gap #11: tree_value multi-level propagation ---
    def test_tree_value_depth_3_manual_calculation(self):
        """tree_value through a depth-3 chain matches hand-computed value.

        root -> (0,0) -> child -> (0,0) -> grandchild leaf = 0.7
        All nodes are 1x1, so the value should propagate straight through.
        """
        mock_view = MagicMock()

        grandchild = TurnNode(handle=20, view=mock_view, to_move=["p1", "p2"])
        grandchild.info = {
            "p1": InfoSet.from_actions([0], np.array([1.0])),
            "p2": InfoSet.from_actions([0], np.array([1.0])),
        }
        grandchild.expanded = True
        grandchild.grid = {(0, 0): ChanceNode(children=[ChanceOutcome(handle=30, leaf_value_p1=0.7)])}

        child = TurnNode(handle=10, view=mock_view, to_move=["p1", "p2"])
        child.info = {
            "p1": InfoSet.from_actions([0], np.array([1.0])),
            "p2": InfoSet.from_actions([0], np.array([1.0])),
        }
        child.expanded = True
        child.grid = {(0, 0): ChanceNode(children=[ChanceOutcome(handle=20, leaf_value_p1=0.0, node=grandchild)])}

        root = TurnNode(handle=0, view=mock_view, to_move=["p1", "p2"])
        root.info = {
            "p1": InfoSet.from_actions([0], np.array([1.0])),
            "p2": InfoSet.from_actions([0], np.array([1.0])),
        }
        root.expanded = True
        root.grid = {(0, 0): ChanceNode(children=[ChanceOutcome(handle=10, leaf_value_p1=0.0, node=child)])}

        # With all 1x1 grids, value should be the grandchild's leaf value = 0.7
        val = tree_value(root)
        assert abs(val - 0.7) < 1e-10, f"Expected 0.7, got {val}"
