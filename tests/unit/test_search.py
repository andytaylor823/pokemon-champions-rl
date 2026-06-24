"""Unit tests for the GT-CFR search module.

Tests the core algorithmic components with mocked SimClient + CVPN:
- Regret matching+ convergence on a toy 2x2 matrix game
- Single-legal-move skip
- PUCT selection logic
- Average strategy extraction
- Deep tree recursion through ChanceOutcome.node references
- Regression: expansion actually deepens the tree (not stuck at depth 1)
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import torch

from search import (
    ChanceNode,
    ChanceOutcome,
    InfoSet,
    SearchConfig,
    SearchResult,
    TurnNode,
    build_policy_target,
    cfr_update_recursive,
    extract_average_strategy,
    puct_scores,
    regret_matching,
    search,
    tree_value,
)
from search.expansion import puct_expand_one, puct_select_cell

# ---------------------------------------------------------------------------
# Regret matching+ tests
# ---------------------------------------------------------------------------


class TestRegretMatching:
    """Test the regret matching+ strategy derivation."""

    def test_uniform_when_all_zero(self):
        # All-zero regret -> uniform strategy
        regret = np.array([0.0, 0.0, 0.0])
        strategy = regret_matching(regret)
        np.testing.assert_allclose(strategy, [1 / 3, 1 / 3, 1 / 3])

    def test_uniform_when_all_negative(self):
        # Negative regrets are treated as zero -> uniform
        regret = np.array([-1.0, -2.0, -3.0])
        strategy = regret_matching(regret)
        np.testing.assert_allclose(strategy, [1 / 3, 1 / 3, 1 / 3])

    def test_proportional_to_positive_regret(self):
        # Positive regrets -> proportional strategy
        regret = np.array([3.0, 1.0, 0.0])
        strategy = regret_matching(regret)
        np.testing.assert_allclose(strategy, [0.75, 0.25, 0.0])

    def test_single_positive_regret(self):
        # Only one positive regret -> all mass on that action
        regret = np.array([0.0, 5.0, 0.0])
        strategy = regret_matching(regret)
        np.testing.assert_allclose(strategy, [0.0, 1.0, 0.0])

    def test_probabilities_sum_to_one(self):
        # Always sums to 1
        regret = np.array([2.0, 3.0, 5.0, 0.0])
        strategy = regret_matching(regret)
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
            cfr_update_recursive(node, t)

        # Extract average strategy
        sigma_bar_p1 = extract_average_strategy(node, "p1")
        sigma_bar_p2 = extract_average_strategy(node, "p2")

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

        # P1's payoffs: (U,L)=3, (U,R)=0, (D,L)=0, (D,R)=1
        node.grid = {
            (0, 0): ChanceNode(children=[ChanceOutcome(handle=100, leaf_value_p1=3.0)]),
            (0, 1): ChanceNode(children=[ChanceOutcome(handle=101, leaf_value_p1=0.0)]),
            (1, 0): ChanceNode(children=[ChanceOutcome(handle=102, leaf_value_p1=0.0)]),
            (1, 1): ChanceNode(children=[ChanceOutcome(handle=103, leaf_value_p1=1.0)]),
        }

        for t in range(1, 2001):
            cfr_update_recursive(node, t)

        sigma_bar_p1 = extract_average_strategy(node, "p1")
        sigma_bar_p2 = extract_average_strategy(node, "p2")

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
        """Values from depth-2 nodes propagate back up through cfr_update_recursive."""
        mock_view = MagicMock()

        # Child node at depth 2: a 2x1 game with known payoffs
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

        # Root node: cell (0,0) points to the child, cell (1,0) is a leaf
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
            # Cell (0,0): expanded child — CFR+ must recurse into it
            (0, 0): ChanceNode(children=[ChanceOutcome(handle=10, leaf_value_p1=0.5, node=child_node)]),
            # Cell (1,0): frontier leaf — uses cached value
            (1, 0): ChanceNode(children=[ChanceOutcome(handle=11, leaf_value_p1=0.3)]),
        }

        # Run CFR+ — should recurse into child_node
        for t in range(1, 101):
            cfr_update_recursive(root, t)

        # Child must have been visited (strategy_sum accumulated by recursion)
        assert child_node.info["p1"].strategy_sum.sum() > 0, "CFR+ did not recurse into child"

        # The root's value for cell (0,0) should reflect the child's deep value,
        # not the stale leaf_value_p1=0.5. After CFR+ on the child, p1 should
        # favor action 0 (value 0.9) over action 1 (value 0.1), making the
        # child's converged value closer to 0.9 than to 0.5.
        child_sigma = regret_matching(child_node.info["p1"].regret)
        child_value = float(child_sigma @ np.array([0.9, 0.1]))
        assert child_value > 0.5, f"Child value {child_value} should exceed stale leaf value 0.5"

    def test_tree_value_does_not_mutate_strategy_sum(self):
        """Regression: tree_value must be read-only — no strategy_sum perturbation.

        Previously the final value extraction called cfr_update_recursive (which
        mutates regret, strategy_sum, and visits) with the largest iteration
        weight, skewing the average strategy that extract_average_strategy reads
        immediately after.
        """
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

        # Snapshot strategy_sum before the call
        p1_sum_before = node.info["p1"].strategy_sum.copy()
        p2_sum_before = node.info["p2"].strategy_sum.copy()

        val = tree_value(node)

        # tree_value must not touch strategy_sum, regret, or visits
        np.testing.assert_array_equal(node.info["p1"].strategy_sum, p1_sum_before)
        np.testing.assert_array_equal(node.info["p2"].strategy_sum, p2_sum_before)
        assert isinstance(val, float)


# ---------------------------------------------------------------------------
# Tree deepening regression test (the blocker bug)
# ---------------------------------------------------------------------------


class TestTreeDeepening:
    """Regression: expansion must actually deepen the tree past depth 1.

    The original bug: _puct_select_cell only returned full ChanceNodes if they
    had an outcome with node != None, but node is only set by deepening —
    a chicken-and-egg deadlock that kept the tree permanently at depth 1.
    """

    def test_puct_select_returns_full_cell_with_frontier_leaves(self):
        """A full ChanceNode whose outcomes are all frontier leaves (node=None) is expandable."""
        mock_view = MagicMock()

        node = TurnNode(handle=0, view=mock_view, to_move=["p1", "p2"])
        node.info = {
            "p1": InfoSet(
                actions=[0],
                prior=np.array([1.0]),
                regret=np.zeros(1),
                strategy_sum=np.zeros(1),
                visits=np.zeros(1, dtype=np.int64),
            ),
            "p2": InfoSet(
                actions=[0],
                prior=np.array([1.0]),
                regret=np.zeros(1),
                strategy_sum=np.zeros(1),
                visits=np.zeros(1, dtype=np.int64),
            ),
        }
        node.expanded = True

        # A 1x1 grid with a full ChanceNode (K=2 children, all frontier leaves)
        config = SearchConfig(max_chance_children=2)
        node.grid = {
            (0, 0): ChanceNode(children=[
                ChanceOutcome(handle=1, leaf_value_p1=0.5),
                ChanceOutcome(handle=2, leaf_value_p1=0.6),
            ]),
        }

        # Select should return (0, 0) because frontier leaves are expandable
        cell = puct_select_cell(node, config)
        assert cell == (0, 0), f"Expected (0, 0) for full ChanceNode with frontier leaves, got {cell}"

    def test_puct_expand_deepens_tree(self):
        """puct_expand_one actually promotes a frontier leaf to a TurnNode (depth > 1)."""
        mock_view = MagicMock()
        mock_view.terminal = False
        mock_view.to_move = ["p1", "p2"]
        mock_view.legal = {
            "p1": {"active": [{"moves": [{"id": "thunderbolt"}, {"id": "protect"}]}]},
            "p2": {"active": [{"moves": [{"id": "flamethrower"}, {"id": "protect"}]}]},
        }
        mock_view.phase = "move"

        node = TurnNode(handle=0, view=mock_view, to_move=["p1", "p2"])
        node.info = {
            "p1": InfoSet(
                actions=[0],
                prior=np.array([1.0]),
                regret=np.zeros(1),
                strategy_sum=np.zeros(1),
                visits=np.zeros(1, dtype=np.int64),
            ),
            "p2": InfoSet(
                actions=[0],
                prior=np.array([1.0]),
                regret=np.zeros(1),
                strategy_sum=np.zeros(1),
                visits=np.zeros(1, dtype=np.int64),
            ),
        }
        node.expanded = True

        # Full ChanceNode (K=1) with one frontier leaf
        config = SearchConfig(max_chance_children=1, k_actions=2)
        node.grid = {
            (0, 0): ChanceNode(children=[
                ChanceOutcome(handle=5, leaf_value_p1=0.4),
            ]),
        }

        # Mock SimClient and CVPN for the child expansion
        mock_sim = MagicMock()
        mock_sim.view.return_value = mock_view

        mock_net = MagicMock()
        import action_space as as_mod
        mock_net.return_value = (torch.zeros(2, as_mod.A), torch.tensor([0.5, 0.5]))

        # Build a mock obs bundle with a proper action_mask (2 legal actions)
        action_mask = torch.tensor([True, True] + [False] * (as_mod.A - 2))
        mock_obs = MagicMock()
        mock_obs.__getitem__ = lambda self, key: action_mask if key == "action_mask" else MagicMock()

        with patch("search.expansion.encode", return_value=mock_obs), \
             patch("search.expansion.collate_obs_bundles", return_value=MagicMock()):
            result = puct_expand_one(node, mock_sim, mock_net, config=config)

        # The expansion should have deepened: outcome.node should now be set
        assert result is True, "puct_expand_one should have expanded a node"
        outcome = node.grid[(0, 0)].children[0]
        assert outcome.node is not None, "Frontier leaf was not promoted to a TurnNode"
        assert outcome.node.expanded is True, "Child TurnNode should be marked expanded"

    def test_puct_expand_descends_to_depth_3(self):
        """puct_expand_one descends through a fully-expanded depth-2 node to reach depth 3.

        The bug: puct_select_cell and puct_expand_one used the same predicate, so the
        descent branch (node = expanded_child; depth += 1) was unreachable — the tree
        was permanently capped at depth 2.
        """
        mock_view = MagicMock()
        mock_view.terminal = False
        mock_view.to_move = ["p1", "p2"]
        mock_view.legal = {
            "p1": {"active": [{"moves": [{"id": "thunderbolt"}, {"id": "protect"}]}]},
            "p2": {"active": [{"moves": [{"id": "flamethrower"}, {"id": "protect"}]}]},
        }
        mock_view.phase = "move"

        # Build a tree: root → (0,0) → depth-2 child (fully expanded, with its own
        # full ChanceNode holding a frontier leaf that should be deepened to depth 3).
        config = SearchConfig(max_chance_children=1, k_actions=2)

        # Depth-2 child: expanded, with a full ChanceNode containing a frontier leaf
        depth2_child = TurnNode(handle=10, view=mock_view, to_move=["p1", "p2"])
        depth2_child.info = {
            "p1": InfoSet(
                actions=[0],
                prior=np.array([1.0]),
                regret=np.zeros(1),
                strategy_sum=np.zeros(1),
                visits=np.zeros(1, dtype=np.int64),
            ),
            "p2": InfoSet(
                actions=[0],
                prior=np.array([1.0]),
                regret=np.zeros(1),
                strategy_sum=np.zeros(1),
                visits=np.zeros(1, dtype=np.int64),
            ),
        }
        depth2_child.expanded = True
        # Full ChanceNode (K=1) with a frontier leaf — this is the depth-3 target
        depth2_child.grid = {
            (0, 0): ChanceNode(children=[
                ChanceOutcome(handle=20, leaf_value_p1=0.7),
            ]),
        }

        # Root: cell (0,0) has a full ChanceNode with ALL outcomes expanded (the depth-2 child)
        root = TurnNode(handle=0, view=mock_view, to_move=["p1", "p2"])
        root.info = {
            "p1": InfoSet(
                actions=[0],
                prior=np.array([1.0]),
                regret=np.zeros(1),
                strategy_sum=np.zeros(1),
                visits=np.zeros(1, dtype=np.int64),
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
            (0, 0): ChanceNode(children=[
                # All outcomes are expanded TurnNodes — triggers descent
                ChanceOutcome(handle=10, leaf_value_p1=0.5, node=depth2_child),
            ]),
        }

        # Mock SimClient and CVPN for the depth-3 expansion
        mock_sim = MagicMock()
        mock_sim.view.return_value = mock_view

        mock_net = MagicMock()
        import action_space as as_mod
        mock_net.return_value = (torch.zeros(2, as_mod.A), torch.tensor([0.6, 0.6]))

        action_mask = torch.tensor([True, True] + [False] * (as_mod.A - 2))
        mock_obs = MagicMock()
        mock_obs.__getitem__ = lambda self, key: action_mask if key == "action_mask" else MagicMock()

        with patch("search.expansion.encode", return_value=mock_obs), \
             patch("search.expansion.collate_obs_bundles", return_value=MagicMock()):
            result = puct_expand_one(root, mock_sim, mock_net, config=config)

        # Descent should have gone root → depth2_child → expanded depth-3 leaf
        assert result is True, "puct_expand_one should have expanded at depth 3"
        depth3_outcome = depth2_child.grid[(0, 0)].children[0]
        assert depth3_outcome.node is not None, "Depth-3 frontier leaf was not promoted"
        assert depth3_outcome.node.expanded is True, "Depth-3 TurnNode should be expanded"

    def test_puct_select_returns_saturated_cell_for_descent(self):
        """A full ChanceNode where ALL outcomes are expanded is returned for descent."""
        mock_view = MagicMock()

        # Depth-2 child (fully expanded)
        child_node = TurnNode(handle=10, view=mock_view, to_move=["p1", "p2"])
        child_node.info = {
            "p1": InfoSet.from_actions([0], np.array([1.0])),
            "p2": InfoSet.from_actions([0], np.array([1.0])),
        }
        child_node.expanded = True
        child_node.grid = {(0, 0): ChanceNode(children=[ChanceOutcome(handle=20, leaf_value_p1=0.5)])}

        # Root with a single full cell where the outcome is already expanded
        root = TurnNode(handle=0, view=mock_view, to_move=["p1", "p2"])
        root.info = {
            "p1": InfoSet.from_actions([0], np.array([1.0])),
            "p2": InfoSet.from_actions([0], np.array([1.0])),
        }
        root.expanded = True
        config = SearchConfig(max_chance_children=1)
        root.grid = {
            (0, 0): ChanceNode(children=[
                ChanceOutcome(handle=10, leaf_value_p1=0.5, node=child_node),
            ]),
        }

        # Select should return (0, 0) for descent even though all outcomes are expanded
        cell = puct_select_cell(root, config)
        assert cell == (0, 0), f"Expected (0, 0) for saturated cell (descent target), got {cell}"

    def test_puct_select_skips_dead_cells(self):
        """Cells where SimError occurred (chance is None in grid) are skipped."""
        mock_view = MagicMock()

        node = TurnNode(handle=0, view=mock_view, to_move=["p1", "p2"])
        node.info = {
            "p1": InfoSet(
                actions=[0, 1],
                prior=np.array([0.9, 0.1]),
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
        node.expanded = True

        # Cell (0,0) is dead (not in grid), cell (1,0) has room
        config = SearchConfig(max_chance_children=5)
        node.grid = {
            # (0, 0) missing — SimError during expansion
            (1, 0): ChanceNode(children=[ChanceOutcome(handle=1, leaf_value_p1=0.5)]),
        }

        cell = puct_select_cell(node, config)
        # Should skip (0,0) and pick (1,0) which has room for more children
        assert cell == (1, 0), f"Expected (1, 0) but got {cell}"


# ---------------------------------------------------------------------------
# PUCT score tests
# ---------------------------------------------------------------------------


class TestPUCTScores:
    """Test the PUCT score computation."""

    def test_exploration_bonus_decreases_with_visits(self):
        """Actions visited more often should have a lower exploration bonus."""
        mock_view = MagicMock()
        node = TurnNode(handle=0, view=mock_view, to_move=["p1", "p2"])
        node.info = {
            "p1": InfoSet(
                actions=[0, 1, 2],
                prior=np.array([0.33, 0.33, 0.34]),
                regret=np.zeros(3),
                strategy_sum=np.zeros(3),
                visits=np.array([10, 1, 0], dtype=np.int64),
            ),
        }

        config = SearchConfig(c_puct=2.0)
        scores = puct_scores(node, "p1", config)

        # Action 2 (0 visits) should have highest exploration bonus
        # Action 0 (10 visits) should have lowest
        assert scores[2] > scores[1] > scores[0]

    def test_puct_expand_produces_nonuniform_visits(self):
        """Regression: puct_expand_one must increment per-action visit counts.

        Previously visits was incremented uniformly (+1 to the whole array) in
        cfr_update_recursive, making the PUCT exploration term degenerate to
        prior * constant.  After the fix, only the selected cell's action
        indices get incremented, so a 2-action node with one cell expanded
        should have visits [1, 0] — not [1, 1].
        """
        mock_view = MagicMock()
        mock_view.terminal = False
        mock_view.to_move = ["p1", "p2"]
        mock_view.legal = {
            "p1": {"active": [{"moves": [{"id": "thunderbolt"}, {"id": "protect"}]}]},
            "p2": {"active": [{"moves": [{"id": "flamethrower"}]}]},
        }
        mock_view.phase = "move"

        node = TurnNode(handle=0, view=mock_view, to_move=["p1", "p2"])
        node.info = {
            "p1": InfoSet.from_actions([0, 1], np.array([0.5, 0.5])),
            "p2": InfoSet.from_actions([0], np.array([1.0])),
        }
        node.expanded = True

        # Two cells: (0,0) has room to widen, (1,0) also has room
        config = SearchConfig(max_chance_children=2, k_actions=2)
        node.grid = {
            (0, 0): ChanceNode(children=[ChanceOutcome(handle=1, leaf_value_p1=0.5)]),
            (1, 0): ChanceNode(children=[ChanceOutcome(handle=2, leaf_value_p1=0.3)]),
        }

        mock_sim = MagicMock()
        mock_sim.step.return_value = MagicMock(
            view=MagicMock(terminal=True, utility={"p1": 0.5}),
            child=99,
        )

        mock_net = MagicMock()
        result = puct_expand_one(node, mock_sim, mock_net, config)
        assert result is True

        # The selected cell's action should have visits=1; the other should stay at 0
        p1_visits = node.info["p1"].visits
        assert p1_visits.sum() == 1, f"Expected exactly one action incremented, got {p1_visits}"
        assert (p1_visits[0] == 1) != (p1_visits[1] == 1), (
            f"Visits should be non-uniform: {p1_visits}"
        )

    def test_prior_biases_scores(self):
        """Higher prior probability should increase the PUCT score."""
        mock_view = MagicMock()
        node = TurnNode(handle=0, view=mock_view, to_move=["p1", "p2"])
        # Strong prior on action 0
        node.info = {
            "p1": InfoSet(
                actions=[0, 1],
                prior=np.array([0.9, 0.1]),
                regret=np.zeros(2),
                strategy_sum=np.zeros(2),
                # Need at least one total visit for the exploration bonus to activate
                visits=np.array([1, 1], dtype=np.int64),
            ),
        }

        config = SearchConfig(c_puct=2.0)
        scores = puct_scores(node, "p1", config)

        # Higher prior -> higher score (when visits are equal)
        assert scores[0] > scores[1]

    def test_c_puct_scales_exploration(self):
        """Higher c_puct should increase the exploration component."""
        mock_view = MagicMock()
        node = TurnNode(handle=0, view=mock_view, to_move=["p1", "p2"])
        node.info = {
            "p1": InfoSet(
                actions=[0, 1],
                prior=np.array([0.5, 0.5]),
                regret=np.array([1.0, 0.0]),
                strategy_sum=np.zeros(2),
                visits=np.array([5, 0], dtype=np.int64),
            ),
        }

        config_low = SearchConfig(c_puct=0.5)
        config_high = SearchConfig(c_puct=5.0)

        scores_low = puct_scores(node, "p1", config_low)
        scores_high = puct_scores(node, "p1", config_high)

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

        # Patch encode in expansion.py (where cvpn_value calls it)
        with patch("search.expansion.encode") as mock_encode:
            mock_obs = MagicMock()
            mock_encode.return_value = mock_obs

            result = search(mock_view, mock_sim, mock_net, from_handle=99, config=SearchConfig())

        assert isinstance(result, SearchResult)
        # Each side's strategy should have exactly one action with mass 1.0
        for s in ["p1", "p2"]:
            assert len(result.strategy[s]) == 1
            assert next(iter(result.strategy[s].values())) == 1.0
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
        from action_space import A

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
        # All other positions should be zero
        assert target[0] == 0.0
        assert target[50] == 0.0
