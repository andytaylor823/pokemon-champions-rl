"""Unit tests for the GT-CFR search module.

Tests the core algorithmic components with mocked SimClient + CVPN:
- Regret matching+ convergence on a toy 2x2 matrix game
- Single-legal-move skip
- PUCT selection logic
- Average strategy extraction
- Deep tree recursion through ChanceOutcome.node references
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from search import (
    ChanceNode,
    ChanceOutcome,
    SearchConfig,
    SearchResult,
    TurnNode,
    _build_policy_target,
    _cfr_update_recursive,
    _extract_average_strategy,
    _puct_scores,
    _regret_matching,
    search,
)


# ---------------------------------------------------------------------------
# Regret matching+ tests
# ---------------------------------------------------------------------------


class TestRegretMatching:
    """Test the regret matching+ strategy derivation."""

    def test_uniform_when_all_zero(self):
        # All-zero regret -> uniform strategy
        regret = np.array([0.0, 0.0, 0.0])
        strategy = _regret_matching(regret)
        np.testing.assert_allclose(strategy, [1 / 3, 1 / 3, 1 / 3])

    def test_uniform_when_all_negative(self):
        # Negative regrets are treated as zero -> uniform
        regret = np.array([-1.0, -2.0, -3.0])
        strategy = _regret_matching(regret)
        np.testing.assert_allclose(strategy, [1 / 3, 1 / 3, 1 / 3])

    def test_proportional_to_positive_regret(self):
        # Positive regrets -> proportional strategy
        regret = np.array([3.0, 1.0, 0.0])
        strategy = _regret_matching(regret)
        np.testing.assert_allclose(strategy, [0.75, 0.25, 0.0])

    def test_single_positive_regret(self):
        # Only one positive regret -> all mass on that action
        regret = np.array([0.0, 5.0, 0.0])
        strategy = _regret_matching(regret)
        np.testing.assert_allclose(strategy, [0.0, 1.0, 0.0])

    def test_probabilities_sum_to_one(self):
        # Always sums to 1
        regret = np.array([2.0, 3.0, 5.0, 0.0])
        strategy = _regret_matching(regret)
        assert abs(strategy.sum() - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# CFR+ convergence test on a 2x2 matrix game (Rock-Paper-Scissors simplified)
# ---------------------------------------------------------------------------


class TestCFRConvergence:
    """Test that CFR+ converges to Nash equilibrium on a toy 2x2 game.

    Game: Matching Pennies
      P2: H    P2: T
    P1: H  (+1, -1)  (-1, +1)
    P1: T  (-1, +1)  (+1, -1)

    Nash: both players play 50/50.
    """

    def _make_matching_pennies_node(self) -> TurnNode:
        """Create a TurnNode representing the Matching Pennies game."""
        mock_view = MagicMock()
        mock_view.to_move = ["p1", "p2"]
        mock_view.terminal = False

        node = TurnNode(
            handle=0,
            view=mock_view,
            to_move=["p1", "p2"],
        )

        # 2 actions per player
        node.actions = {"p1": [0, 1], "p2": [0, 1]}
        node.policy_prior = {"p1": np.array([0.5, 0.5]), "p2": np.array([0.5, 0.5])}
        node.cumulative_regret = {"p1": np.zeros(2), "p2": np.zeros(2)}
        node.strategy_sum = {"p1": np.zeros(2), "p2": np.zeros(2)}
        node.visit_counts = {"p1": np.zeros(2, dtype=np.int64), "p2": np.zeros(2, dtype=np.int64)}
        node.expanded = True

        # Payoff matrix from p1's perspective: (H,H)=+1, (H,T)=-1, (T,H)=-1, (T,T)=+1
        node.grid = {
            (0, 0): ChanceNode(children=[ChanceOutcome(handle=100, leaf_value_p1=1.0)]),
            (0, 1): ChanceNode(children=[ChanceOutcome(handle=101, leaf_value_p1=-1.0)]),
            (1, 0): ChanceNode(children=[ChanceOutcome(handle=102, leaf_value_p1=-1.0)]),
            (1, 1): ChanceNode(children=[ChanceOutcome(handle=103, leaf_value_p1=1.0)]),
        }

        return node

    def test_converges_to_nash(self):
        """After many CFR+ iterations, average strategy should converge to 50/50."""
        node = self._make_matching_pennies_node()

        # Run many CFR+ iterations
        for t in range(1, 1001):
            _cfr_update_recursive(node, t)

        # Extract average strategy
        sigma_bar_p1 = _extract_average_strategy(node, "p1")
        sigma_bar_p2 = _extract_average_strategy(node, "p2")

        # Both should be close to 50/50
        assert abs(sigma_bar_p1.get(0, 0) - 0.5) < 0.05
        assert abs(sigma_bar_p1.get(1, 0) - 0.5) < 0.05
        assert abs(sigma_bar_p2.get(0, 0) - 0.5) < 0.05
        assert abs(sigma_bar_p2.get(1, 0) - 0.5) < 0.05

    def test_converges_asymmetric_game(self):
        """Test convergence on an asymmetric game.

        Game matrix (p1 payoff):
          P2: L   P2: R
        P1: U  3     0
        P1: D  0     1

        Nash: P1 plays U with prob 1/4, D with prob 3/4;
              P2 plays L with prob 1/4, R with prob 3/4.
        Wait -- let me recalculate. For zero-sum:
        p1 picks U: 3*q + 0*(1-q) = 3q
        p1 picks D: 0*q + 1*(1-q) = 1-q
        Indifference: 3q = 1-q -> 4q = 1 -> q = 1/4
        p2 picks L: 3*p + 0*(1-p) = 3p
        p2 picks R: 0*p + 1*(1-p) = 1-p
        Indifference: 3p = 1-p -> 4p = 1 -> p = 1/4

        So Nash is: P1 plays (1/4, 3/4), P2 plays (1/4, 3/4).
        """
        mock_view = MagicMock()
        mock_view.to_move = ["p1", "p2"]

        node = TurnNode(
            handle=0,
            view=mock_view,
            to_move=["p1", "p2"],
        )

        node.actions = {"p1": [0, 1], "p2": [0, 1]}
        node.policy_prior = {"p1": np.array([0.5, 0.5]), "p2": np.array([0.5, 0.5])}
        node.cumulative_regret = {"p1": np.zeros(2), "p2": np.zeros(2)}
        node.strategy_sum = {"p1": np.zeros(2), "p2": np.zeros(2)}
        node.visit_counts = {"p1": np.zeros(2, dtype=np.int64), "p2": np.zeros(2, dtype=np.int64)}
        node.expanded = True

        # P1's payoffs: (U,L)=3, (U,R)=0, (D,L)=0, (D,R)=1
        node.grid = {
            (0, 0): ChanceNode(children=[ChanceOutcome(handle=100, leaf_value_p1=3.0)]),
            (0, 1): ChanceNode(children=[ChanceOutcome(handle=101, leaf_value_p1=0.0)]),
            (1, 0): ChanceNode(children=[ChanceOutcome(handle=102, leaf_value_p1=0.0)]),
            (1, 1): ChanceNode(children=[ChanceOutcome(handle=103, leaf_value_p1=1.0)]),
        }

        for t in range(1, 2001):
            _cfr_update_recursive(node, t)

        sigma_bar_p1 = _extract_average_strategy(node, "p1")
        sigma_bar_p2 = _extract_average_strategy(node, "p2")

        # P1 should play ~(0.25, 0.75), P2 should play ~(0.25, 0.75)
        assert abs(sigma_bar_p1.get(0, 0) - 0.25) < 0.05
        assert abs(sigma_bar_p1.get(1, 0) - 0.75) < 0.05
        assert abs(sigma_bar_p2.get(0, 0) - 0.25) < 0.05
        assert abs(sigma_bar_p2.get(1, 0) - 0.75) < 0.05


# ---------------------------------------------------------------------------
# Deep tree recursion regression test
# ---------------------------------------------------------------------------


class TestDeepRecursion:
    """Regression: CFR+ must recurse through expanded ChanceOutcome.node refs.

    Catches the aliasing bug where a module-global registry split across modules
    silently prevented recursion beyond depth 1.
    """

    def test_cfr_recurses_into_child_nodes(self):
        """Values from depth-2 nodes propagate back up through _cfr_update_recursive."""
        mock_view = MagicMock()

        # Child node at depth 2: a 2x1 game with known payoffs
        child_node = TurnNode(handle=10, view=mock_view, to_move=["p1", "p2"])
        child_node.actions = {"p1": [0, 1], "p2": [0]}
        child_node.policy_prior = {"p1": np.array([0.5, 0.5]), "p2": np.array([1.0])}
        child_node.cumulative_regret = {"p1": np.zeros(2), "p2": np.zeros(1)}
        child_node.strategy_sum = {"p1": np.zeros(2), "p2": np.zeros(1)}
        child_node.visit_counts = {"p1": np.zeros(2, dtype=np.int64), "p2": np.zeros(1, dtype=np.int64)}
        child_node.expanded = True
        child_node.grid = {
            (0, 0): ChanceNode(children=[ChanceOutcome(handle=20, leaf_value_p1=0.9)]),
            (1, 0): ChanceNode(children=[ChanceOutcome(handle=21, leaf_value_p1=0.1)]),
        }

        # Root node: cell (0,0) points to the child, cell (1,0) is a leaf
        root = TurnNode(handle=0, view=mock_view, to_move=["p1", "p2"])
        root.actions = {"p1": [0, 1], "p2": [0]}
        root.policy_prior = {"p1": np.array([0.5, 0.5]), "p2": np.array([1.0])}
        root.cumulative_regret = {"p1": np.zeros(2), "p2": np.zeros(1)}
        root.strategy_sum = {"p1": np.zeros(2), "p2": np.zeros(1)}
        root.visit_counts = {"p1": np.zeros(2, dtype=np.int64), "p2": np.zeros(1, dtype=np.int64)}
        root.expanded = True
        root.grid = {
            # Cell (0,0): expanded child — CFR+ must recurse into it
            (0, 0): ChanceNode(children=[ChanceOutcome(handle=10, leaf_value_p1=0.5, node=child_node)]),
            # Cell (1,0): frontier leaf — uses cached value
            (1, 0): ChanceNode(children=[ChanceOutcome(handle=11, leaf_value_p1=0.3)]),
        }

        # Run CFR+ — should recurse into child_node
        for t in range(1, 101):
            _cfr_update_recursive(root, t)

        # Child must have been visited (regrets updated by recursion)
        assert child_node.visit_counts["p1"].sum() > 0, "CFR+ did not recurse into child"
        assert child_node.strategy_sum["p1"].sum() > 0, "Child strategy sum not accumulated"

        # The root's value for cell (0,0) should reflect the child's deep value,
        # not the stale leaf_value_p1=0.5. After CFR+ on the child, p1 should
        # favor action 0 (value 0.9) over action 1 (value 0.1), making the
        # child's converged value closer to 0.9 than to 0.5.
        child_sigma = _regret_matching(child_node.cumulative_regret["p1"])
        child_value = float(child_sigma @ np.array([0.9, 0.1]))
        assert child_value > 0.5, f"Child value {child_value} should exceed stale leaf value 0.5"


# ---------------------------------------------------------------------------
# PUCT score tests
# ---------------------------------------------------------------------------


class TestPUCTScores:
    """Test the PUCT score computation."""

    def test_exploration_bonus_decreases_with_visits(self):
        """Actions that have been visited more should have lower exploration bonus."""
        mock_view = MagicMock()
        node = TurnNode(handle=0, view=mock_view, to_move=["p1", "p2"])
        node.actions = {"p1": [0, 1, 2]}
        node.policy_prior = {"p1": np.array([0.33, 0.33, 0.34])}
        node.cumulative_regret = {"p1": np.zeros(3)}
        node.visit_counts = {"p1": np.array([10, 1, 0], dtype=np.int64)}

        config = SearchConfig(c_puct=2.0)
        scores = _puct_scores(node, "p1", config)

        # Action 2 (0 visits) should have highest exploration bonus
        # Action 0 (10 visits) should have lowest
        assert scores[2] > scores[1] > scores[0]

    def test_prior_biases_scores(self):
        """Higher prior probability should increase the PUCT score."""
        mock_view = MagicMock()
        node = TurnNode(handle=0, view=mock_view, to_move=["p1", "p2"])
        node.actions = {"p1": [0, 1]}
        # Strong prior on action 0
        node.policy_prior = {"p1": np.array([0.9, 0.1])}
        node.cumulative_regret = {"p1": np.zeros(2)}
        # Need at least one total visit for the exploration bonus to activate
        node.visit_counts = {"p1": np.array([1, 1], dtype=np.int64)}

        config = SearchConfig(c_puct=2.0)
        scores = _puct_scores(node, "p1", config)

        # Higher prior -> higher score (when visits are equal)
        assert scores[0] > scores[1]

    def test_c_puct_scales_exploration(self):
        """Higher c_puct should increase the exploration component."""
        mock_view = MagicMock()
        node = TurnNode(handle=0, view=mock_view, to_move=["p1", "p2"])
        node.actions = {"p1": [0, 1]}
        node.policy_prior = {"p1": np.array([0.5, 0.5])}
        node.cumulative_regret = {"p1": np.array([1.0, 0.0])}
        node.visit_counts = {"p1": np.array([5, 0], dtype=np.int64)}

        config_low = SearchConfig(c_puct=0.5)
        config_high = SearchConfig(c_puct=5.0)

        scores_low = _puct_scores(node, "p1", config_low)
        scores_high = _puct_scores(node, "p1", config_high)

        # With higher c_puct, the less-visited action should be more competitive
        gap_low = scores_low[0] - scores_low[1]
        gap_high = scores_high[0] - scores_high[1]
        # Higher c_puct should decrease the gap (more exploration)
        assert gap_high < gap_low


# ---------------------------------------------------------------------------
# Single-legal-move skip
# ---------------------------------------------------------------------------


class TestSingleLegalMoveSkip:
    """Test that search returns immediately when only one action is legal per side."""

    def test_single_action_both_sides(self):
        """When both sides have exactly one legal action, skip search entirely."""
        from unittest.mock import MagicMock

        import action_space as as_mod

        # Create a view where each side has exactly one legal action
        mock_view = MagicMock()
        mock_view.phase = "forceSwitch"
        mock_view.to_move = ["p1", "p2"]
        # forceSwitch with only one switch target available per side
        mock_view.legal = {
            "p1": {"forceSwitch": [True, False], "side": {"pokemon": [
                {"condition": "0 fnt"}, {"condition": "0 fnt"},
                {"condition": "100/100"}, {"condition": "0 fnt"},
            ]}},
            "p2": {"forceSwitch": [True, False], "side": {"pokemon": [
                {"condition": "0 fnt"}, {"condition": "0 fnt"},
                {"condition": "0 fnt"}, {"condition": "120/120"},
            ]}},
        }
        mock_view.terminal = False
        mock_view.utility = None

        # Mock SimClient and CVPN
        mock_sim = MagicMock()
        mock_sim.open_search.return_value = (1, 10, None)
        mock_sim.close_search.return_value = 0

        # CVPN mock that returns a scalar value
        mock_net = MagicMock()
        mock_value = torch.tensor(0.3)
        mock_net.return_value = (torch.zeros(as_mod.A), mock_value)

        # Patch encode to return a dummy ObsBundle
        with patch("search.core.encode") as mock_encode:
            mock_obs = MagicMock()
            mock_encode.return_value = mock_obs

            result = search(mock_view, mock_sim, mock_net, from_handle=99, config=SearchConfig())

        assert isinstance(result, SearchResult)
        # Each side's strategy should have exactly one action with mass 1.0
        for s in ["p1", "p2"]:
            assert len(result.strategy[s]) == 1
            assert list(result.strategy[s].values())[0] == 1.0
        # Value should come from the CVPN
        assert abs(result.value - 0.3) < 1e-5
        # open_search should NOT have been called (skipped search)
        mock_sim.open_search.assert_not_called()


# ---------------------------------------------------------------------------
# Strategy extraction tests
# ---------------------------------------------------------------------------


class TestStrategyExtraction:
    """Test average strategy extraction and policy target building."""

    def test_extracts_normalized_strategy(self):
        """Strategy sum is normalized to probabilities."""
        mock_view = MagicMock()
        node = TurnNode(handle=0, view=mock_view, to_move=["p1", "p2"])
        node.actions = {"p1": [10, 20, 30]}
        node.strategy_sum = {"p1": np.array([100.0, 200.0, 300.0])}

        sigma_bar = _extract_average_strategy(node, "p1")

        assert abs(sigma_bar[10] - 1 / 6) < 1e-10
        assert abs(sigma_bar[20] - 2 / 6) < 1e-10
        assert abs(sigma_bar[30] - 3 / 6) < 1e-10

    def test_uniform_fallback_when_no_iterations(self):
        """When strategy sum is all zeros, fall back to uniform."""
        mock_view = MagicMock()
        node = TurnNode(handle=0, view=mock_view, to_move=["p1"])
        node.actions = {"p1": [5, 6]}
        node.strategy_sum = {"p1": np.array([0.0, 0.0])}

        sigma_bar = _extract_average_strategy(node, "p1")
        assert abs(sigma_bar[5] - 0.5) < 1e-10
        assert abs(sigma_bar[6] - 0.5) < 1e-10

    def test_policy_target_correct_shape(self):
        """Policy target should be a full [A] vector with mass on top-k."""
        from action_space import A

        mock_view = MagicMock()
        node = TurnNode(handle=0, view=mock_view, to_move=["p1"])
        node.actions = {"p1": [100, 200, 300]}
        node.strategy_sum = {"p1": np.array([1.0, 2.0, 3.0])}

        target = _build_policy_target(node, "p1")

        assert target.shape == (A,)
        assert abs(target.sum() - 1.0) < 1e-6
        assert target[100] > 0
        assert target[200] > 0
        assert target[300] > 0
        # All other positions should be zero
        assert target[0] == 0.0
        assert target[50] == 0.0


# ---------------------------------------------------------------------------
# ChanceNode value computation
# ---------------------------------------------------------------------------


class TestChanceNode:
    """Test ChanceNode value averaging."""

    def test_single_child_value(self):
        """With one child, value = that child's value."""
        chance = ChanceNode(children=[ChanceOutcome(handle=1, leaf_value_p1=0.7)])
        assert abs(chance.value_p1 - 0.7) < 1e-10

    def test_multiple_children_uniform_average(self):
        """With multiple children, value = uniform average."""
        chance = ChanceNode(children=[
            ChanceOutcome(handle=1, leaf_value_p1=0.5),
            ChanceOutcome(handle=2, leaf_value_p1=0.3),
            ChanceOutcome(handle=3, leaf_value_p1=0.8),
        ])
        expected = (0.5 + 0.3 + 0.8) / 3
        assert abs(chance.value_p1 - expected) < 1e-10

    def test_empty_children_zero(self):
        """Empty children -> value 0."""
        chance = ChanceNode(children=[])
        assert chance.value_p1 == 0.0
