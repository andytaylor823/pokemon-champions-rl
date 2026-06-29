"""Unit tests for search.expansion — PUCT, expand_turn_node, and helpers.

Migrated from tests/unit/test_search.py:
  - TestTreeDeepening, TestPUCTScores

New gap tests:
  - expand_turn_node: simultaneous, unilateral/noop, k=0, all-SimError, mixed terminal (gaps #1, #32, #33)
  - cvpn_value (gap #4)
  - _terminal_utility_p1 (gap #5)
  - _cell_choices (gap #6)
  - _expand_chance_child error paths (gap #7)
  - _expand_child_turn_node edge cases (gap #8)
  - puct_scores zero visits (gap #16)
  - puct_select_cell empty grid/actions (gaps #17, #18)
  - puct_expand_one saturated/depth-cap (gaps #19, #20)
  - _generate_seed (gap #23)
  - Large k sparse grid (gap #30)
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import torch

import action_space as as_mod
from search import (
    ChanceNode,
    ChanceOutcome,
    InfoSet,
    SearchConfig,
    TurnNode,
    puct_scores,
)
from search.expansion import (
    _cell_choices,
    _collapse_forced,
    _expand_chance_child,
    _expand_child_turn_node,
    _forced_view_choices,
    _generate_seed,
    _terminal_utility_p1,
    cvpn_value,
    expand_turn_node,
    puct_expand_one,
    puct_select_cell,
)
from sim_client import SimError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_obs_with_mask(n_legal: int) -> dict:
    """Create a dict-based ObsBundle stub whose action_mask has n_legal True entries."""
    action_mask = torch.tensor([True] * n_legal + [False] * (as_mod.A - n_legal))
    return {"action_mask": action_mask}


# ---------------------------------------------------------------------------
# expand_turn_node (gaps #1, #32, #33)
# ---------------------------------------------------------------------------


class TestExpandTurnNode:
    """Unit tests for expand_turn_node with mocked SimClient + CVPN."""

    def _setup_mocks(self, view, n_legal=2, terminal_results=False):
        """Return (sim, net, obs) mocks for expand_turn_node."""
        mock_sim = MagicMock()
        if terminal_results:
            mock_sim.step.return_value = MagicMock(
                view=MagicMock(terminal=True, utility={"p1": 0.8}), child=200,
            )
        else:
            mock_sim.step.return_value = MagicMock(
                view=MagicMock(terminal=False), child=200,
            )

        mock_net = MagicMock()
        # Return logits and values for a batch (up to 2 sides)
        mock_net.return_value = (torch.zeros(2, as_mod.A), torch.tensor([0.5, 0.5]))

        mock_obs = _mock_obs_with_mask(n_legal)
        return mock_sim, mock_net, mock_obs

    def test_simultaneous_expansion(self):
        """Both sides acting: creates InfoSets for both, fills grid."""
        mock_view = MagicMock()
        mock_view.to_move = ["p1", "p2"]
        mock_view.phase = "move"

        node = TurnNode(handle=0, view=mock_view, to_move=["p1", "p2"])
        config = SearchConfig(k_actions=2, max_chance_children=1)
        mock_sim, mock_net, mock_obs = self._setup_mocks(mock_view, n_legal=2, terminal_results=True)

        with patch("search.expansion.encode", return_value=mock_obs), \
             patch("search.expansion.collate_obs_bundles", return_value=MagicMock()):
            expand_turn_node(node, mock_sim, mock_net, config)

        assert node.expanded is True
        assert "p1" in node.info
        assert "p2" in node.info
        assert len(node.info["p1"].actions) > 0
        assert len(node.info["p2"].actions) > 0
        assert len(node.grid) > 0

    def test_unilateral_expansion_noop_padding(self):
        """Only p1 acting: p2 gets a noop InfoSet (gap #32)."""
        mock_view = MagicMock()
        mock_view.to_move = ["p1"]
        mock_view.phase = "move"

        node = TurnNode(handle=0, view=mock_view, to_move=["p1"])
        config = SearchConfig(k_actions=2, max_chance_children=1)
        mock_sim, mock_net, mock_obs = self._setup_mocks(mock_view, n_legal=2, terminal_results=True)
        # Only one side in the batch
        mock_net.return_value = (torch.zeros(1, as_mod.A), torch.tensor([0.5]))

        with patch("search.expansion.encode", return_value=mock_obs), \
             patch("search.expansion.collate_obs_bundles", return_value=MagicMock()):
            expand_turn_node(node, mock_sim, mock_net, config)

        assert node.expanded is True
        # p2 should have noop InfoSet
        assert "p2" in node.info
        assert node.info["p2"].actions == [0]
        np.testing.assert_array_equal(node.info["p2"].prior, [1.0])

    def test_k_zero_early_exit(self):
        """When all actions are masked (k=0), node is expanded but empty."""
        mock_view = MagicMock()
        mock_view.to_move = ["p1", "p2"]
        mock_view.phase = "move"

        node = TurnNode(handle=0, view=mock_view, to_move=["p1", "p2"])
        config = SearchConfig(k_actions=2, max_chance_children=1)
        mock_sim, mock_net, _ = self._setup_mocks(mock_view, n_legal=0)
        mock_obs_zero = _mock_obs_with_mask(0)
        mock_net.return_value = (torch.zeros(2, as_mod.A), torch.tensor([0.5, 0.5]))

        with patch("search.expansion.encode", return_value=mock_obs_zero), \
             patch("search.expansion.collate_obs_bundles", return_value=MagicMock()):
            expand_turn_node(node, mock_sim, mock_net, config)

        assert node.expanded is True
        assert len(node.grid) == 0

    def test_all_simerror(self):
        """All grid cells fail with SimError -> expanded but empty grid."""
        mock_view = MagicMock()
        mock_view.to_move = ["p1", "p2"]
        mock_view.phase = "move"

        node = TurnNode(handle=0, view=mock_view, to_move=["p1", "p2"])
        config = SearchConfig(k_actions=2, max_chance_children=1)
        mock_sim, mock_net, mock_obs = self._setup_mocks(mock_view, n_legal=2)
        mock_sim.step.side_effect = SimError("illegal")

        with patch("search.expansion.encode", return_value=mock_obs), \
             patch("search.expansion.collate_obs_bundles", return_value=MagicMock()):
            expand_turn_node(node, mock_sim, mock_net, config)

        assert node.expanded is True
        assert len(node.grid) == 0

    def test_mixed_terminal_non_terminal(self):
        """Grid with both terminal and non-terminal children."""
        mock_view = MagicMock()
        mock_view.to_move = ["p1", "p2"]
        mock_view.phase = "move"

        node = TurnNode(handle=0, view=mock_view, to_move=["p1", "p2"])
        config = SearchConfig(k_actions=2, max_chance_children=1)
        _, mock_net, mock_obs = self._setup_mocks(mock_view, n_legal=2)

        # Alternate terminal and non-terminal step results
        call_count = [0]
        def _step_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] % 2 == 0:
                return MagicMock(view=MagicMock(terminal=True, utility={"p1": 1.0}), child=call_count[0] + 300)
            return MagicMock(view=MagicMock(terminal=False), child=call_count[0] + 300)

        mock_sim = MagicMock()
        mock_sim.step.side_effect = _step_side_effect

        with patch("search.expansion.encode", return_value=mock_obs), \
             patch("search.expansion.collate_obs_bundles", return_value=MagicMock()):
            expand_turn_node(node, mock_sim, mock_net, config)

        assert node.expanded is True
        assert len(node.grid) > 0


# ---------------------------------------------------------------------------
# cvpn_value (gap #4)
# ---------------------------------------------------------------------------


class TestCvpnValue:
    """Test the single-state CVPN evaluation helper."""

    def test_returns_known_value(self):
        mock_net = MagicMock()
        mock_net.return_value = (torch.zeros(as_mod.A), torch.tensor(0.42))
        mock_view = MagicMock()

        with patch("search.expansion.encode", return_value=MagicMock()):
            val = cvpn_value(mock_net, mock_view)

        assert abs(val - 0.42) < 1e-5

    def test_no_grad_context(self):
        """CVPN evaluation should not accumulate gradients."""
        mock_net = MagicMock()
        mock_net.return_value = (torch.zeros(as_mod.A), torch.tensor(0.1))
        mock_view = MagicMock()

        with patch("search.expansion.encode", return_value=MagicMock()):
            val = cvpn_value(mock_net, mock_view)

        assert isinstance(val, float)


# ---------------------------------------------------------------------------
# _terminal_utility_p1 (gap #5)
# ---------------------------------------------------------------------------


class TestTerminalUtilityP1:
    """Test terminal utility extraction helper."""

    def test_normal_case(self):
        view = MagicMock()
        view.utility = {"p1": 0.75, "p2": -0.75}
        assert _terminal_utility_p1(view) == 0.75

    def test_none_utility(self):
        view = MagicMock()
        view.utility = None
        assert _terminal_utility_p1(view) == 0.0

    def test_missing_p1_key(self):
        view = MagicMock()
        view.utility = {"p2": -1.0}
        assert _terminal_utility_p1(view) == 0.0


# ---------------------------------------------------------------------------
# _cell_choices (gap #6)
# ---------------------------------------------------------------------------


class TestCellChoices:
    """Test SimClient choice dict construction."""

    def test_simultaneous(self):
        """Both sides in to_move -> both in choices dict."""
        mock_view = MagicMock()
        node = TurnNode(handle=0, view=mock_view, to_move=["p1", "p2"])
        node.info = {
            "p1": InfoSet.from_actions([5, 10], np.array([0.5, 0.5])),
            "p2": InfoSet.from_actions([3, 7], np.array([0.5, 0.5])),
        }

        with patch("search.expansion.action_to_choice_contextual", side_effect=lambda x, _req: f"choice_{x}"):
            choices = _cell_choices(node, 0, 1)  # p1 action index 0 -> action 5, p2 index 1 -> action 7

        assert "p1" in choices
        assert "p2" in choices
        assert choices["p1"] == "choice_5"
        assert choices["p2"] == "choice_7"

    def test_unilateral(self):
        """Only p1 in to_move -> p2 excluded from choices dict."""
        mock_view = MagicMock()
        node = TurnNode(handle=0, view=mock_view, to_move=["p1"])
        node.info = {
            "p1": InfoSet.from_actions([5], np.array([1.0])),
            "p2": InfoSet.noop(),
        }

        with patch("search.expansion.action_to_choice_contextual", side_effect=lambda x, _req: f"choice_{x}"):
            choices = _cell_choices(node, 0, 0)

        assert "p1" in choices
        assert "p2" not in choices


# ---------------------------------------------------------------------------
# _expand_chance_child (gap #7)
# ---------------------------------------------------------------------------


class TestExpandChanceChild:
    """Test ChanceNode widening helper error paths."""

    def test_none_chance_in_grid(self):
        """Cell not in grid -> early return, no crash."""
        mock_view = MagicMock()
        node = TurnNode(handle=0, view=mock_view, to_move=["p1", "p2"])
        node.info = {
            "p1": InfoSet.from_actions([0], np.array([1.0])),
            "p2": InfoSet.from_actions([0], np.array([1.0])),
        }
        node.grid = {}
        config = SearchConfig(max_chance_children=5)

        _expand_chance_child(node, (0, 0), MagicMock(), MagicMock(), config)
        # No crash, grid still empty
        assert len(node.grid) == 0

    def test_already_at_max_children(self):
        """ChanceNode already at max_chance_children -> early return."""
        mock_view = MagicMock()
        node = TurnNode(handle=0, view=mock_view, to_move=["p1", "p2"])
        node.info = {
            "p1": InfoSet.from_actions([0], np.array([1.0])),
            "p2": InfoSet.from_actions([0], np.array([1.0])),
        }
        config = SearchConfig(max_chance_children=1)
        # Already has 1 child (at max)
        node.grid = {(0, 0): ChanceNode(children=[ChanceOutcome(handle=1, leaf_value_p1=0.5)])}

        _expand_chance_child(node, (0, 0), MagicMock(), MagicMock(), config)
        assert len(node.grid[(0, 0)].children) == 1

    def test_simerror_silently_returns(self):
        """SimError during step -> no new child appended."""
        mock_view = MagicMock()
        node = TurnNode(handle=0, view=mock_view, to_move=["p1", "p2"])
        node.info = {
            "p1": InfoSet.from_actions([0], np.array([1.0])),
            "p2": InfoSet.from_actions([0], np.array([1.0])),
        }
        config = SearchConfig(max_chance_children=5)
        node.grid = {(0, 0): ChanceNode(children=[])}

        mock_sim = MagicMock()
        mock_sim.step.side_effect = SimError("bad choice")

        with patch("search.expansion.action_to_choice_contextual", return_value="move 1"):
            _expand_chance_child(node, (0, 0), mock_sim, MagicMock(), config)

        assert len(node.grid[(0, 0)].children) == 0

    def test_terminal_child(self):
        """Terminal child -> appended with real utility."""
        mock_view = MagicMock()
        node = TurnNode(handle=0, view=mock_view, to_move=["p1", "p2"])
        node.info = {
            "p1": InfoSet.from_actions([0], np.array([1.0])),
            "p2": InfoSet.from_actions([0], np.array([1.0])),
        }
        config = SearchConfig(max_chance_children=5)
        node.grid = {(0, 0): ChanceNode(children=[])}

        mock_sim = MagicMock()
        mock_sim.step.return_value = MagicMock(
            view=MagicMock(terminal=True, utility={"p1": -0.5}), child=99,
        )

        with patch("search.expansion.action_to_choice_contextual", return_value="move 1"):
            _expand_chance_child(node, (0, 0), mock_sim, MagicMock(), config)

        assert len(node.grid[(0, 0)].children) == 1
        assert node.grid[(0, 0)].children[0].leaf_value_p1 == -0.5

    def test_non_terminal_child(self):
        """Non-terminal child -> appended with CVPN value."""
        mock_view = MagicMock()
        node = TurnNode(handle=0, view=mock_view, to_move=["p1", "p2"])
        node.info = {
            "p1": InfoSet.from_actions([0], np.array([1.0])),
            "p2": InfoSet.from_actions([0], np.array([1.0])),
        }
        config = SearchConfig(max_chance_children=5)
        node.grid = {(0, 0): ChanceNode(children=[])}

        mock_sim = MagicMock()
        mock_sim.step.return_value = MagicMock(
            view=MagicMock(terminal=False), child=99,
        )
        mock_net = MagicMock()

        with patch("search.expansion.action_to_choice_contextual", return_value="move 1"), \
             patch("search.expansion.cvpn_value", return_value=0.33):
            _expand_chance_child(node, (0, 0), mock_sim, mock_net, config)

        assert len(node.grid[(0, 0)].children) == 1
        assert abs(node.grid[(0, 0)].children[0].leaf_value_p1 - 0.33) < 1e-5


# ---------------------------------------------------------------------------
# _expand_child_turn_node (gap #8)
# ---------------------------------------------------------------------------


class TestExpandChildTurnNode:
    """Test frontier leaf -> TurnNode promotion."""

    def test_terminal_returns_none(self):
        """Terminal child view -> returns None, outcome.node stays None."""
        outcome = ChanceOutcome(handle=5, leaf_value_p1=0.5)
        mock_sim = MagicMock()
        mock_sim.view.return_value = MagicMock(terminal=True)

        result = _expand_child_turn_node(outcome, mock_sim, MagicMock(), SearchConfig())
        assert result is None
        assert outcome.node is None

    def test_empty_to_move_returns_none(self):
        """Child view with empty to_move -> returns None."""
        outcome = ChanceOutcome(handle=5, leaf_value_p1=0.5)
        mock_sim = MagicMock()
        mock_sim.view.return_value = MagicMock(terminal=False, to_move=[])

        result = _expand_child_turn_node(outcome, mock_sim, MagicMock(), SearchConfig())
        assert result is None
        assert outcome.node is None

    def test_normal_expansion(self):
        """Non-terminal child with actions -> outcome.node is set in-place."""
        outcome = ChanceOutcome(handle=5, leaf_value_p1=0.5)
        mock_view = MagicMock()
        mock_view.terminal = False
        mock_view.to_move = ["p1", "p2"]
        mock_view.phase = "move"

        mock_sim = MagicMock()
        mock_sim.view.return_value = mock_view
        mock_sim.step.return_value = MagicMock(
            view=MagicMock(terminal=True, utility={"p1": 0.0}), child=300,
        )

        mock_net = MagicMock()
        mock_obs = _mock_obs_with_mask(2)
        mock_net.return_value = (torch.zeros(2, as_mod.A), torch.tensor([0.5, 0.5]))

        config = SearchConfig(k_actions=2, max_chance_children=1)
        with patch("search.expansion.encode", return_value=mock_obs), \
             patch("search.expansion.collate_obs_bundles", return_value=MagicMock()):
            result = _expand_child_turn_node(outcome, mock_sim, mock_net, config)

        assert result is not None
        assert outcome.node is result
        assert outcome.node.expanded is True


# ---------------------------------------------------------------------------
# PUCT score tests (migrated + gap #16)
# ---------------------------------------------------------------------------


class TestPUCTScores:
    """Test the PUCT score computation."""

    def test_exploration_bonus_decreases_with_visits(self):
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
        assert scores[2] > scores[1] > scores[0]

    def test_prior_biases_scores(self):
        mock_view = MagicMock()
        node = TurnNode(handle=0, view=mock_view, to_move=["p1", "p2"])
        node.info = {
            "p1": InfoSet(
                actions=[0, 1],
                prior=np.array([0.9, 0.1]),
                regret=np.zeros(2),
                strategy_sum=np.zeros(2),
                visits=np.array([1, 1], dtype=np.int64),
            ),
        }

        config = SearchConfig(c_puct=2.0)
        scores = puct_scores(node, "p1", config)
        assert scores[0] > scores[1]

    def test_c_puct_scales_exploration(self):
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

        scores_low = puct_scores(node, "p1", SearchConfig(c_puct=0.5))
        scores_high = puct_scores(node, "p1", SearchConfig(c_puct=5.0))
        gap_low = scores_low[0] - scores_low[1]
        gap_high = scores_high[0] - scores_high[1]
        assert gap_high < gap_low

    # --- Gap #16: zero total visits ---
    def test_zero_visits_degenerates_to_sigma(self):
        """With zero visits, exploration term is 0 -> score equals sigma (uniform)."""
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
        }

        scores = puct_scores(node, "p1", SearchConfig(c_puct=2.0))
        # With zero regret -> sigma is uniform [0.5, 0.5]
        # With zero visits -> exploration = prior * sqrt(0) / 1 = 0
        # Score should be uniform
        np.testing.assert_allclose(scores, [0.5, 0.5])


# ---------------------------------------------------------------------------
# puct_select_cell edge cases (gaps #17, #18)
# ---------------------------------------------------------------------------


class TestPUCTSelectCellEdges:
    """Test puct_select_cell with empty grids and actions."""

    def test_empty_grid_returns_none(self):
        """Empty grid -> returns None (gap #17)."""
        mock_view = MagicMock()
        node = TurnNode(handle=0, view=mock_view, to_move=["p1", "p2"])
        node.info = {
            "p1": InfoSet.from_actions([0, 1], np.array([0.5, 0.5])),
            "p2": InfoSet.from_actions([0, 1], np.array([0.5, 0.5])),
        }
        node.grid = {}
        assert puct_select_cell(node, SearchConfig()) is None

    def test_empty_actions_returns_none(self):
        """Empty actions on one side -> returns None (gap #18)."""
        mock_view = MagicMock()
        node = TurnNode(handle=0, view=mock_view, to_move=["p1", "p2"])
        node.info = {
            "p1": InfoSet.empty(),
            "p2": InfoSet.from_actions([0], np.array([1.0])),
        }
        node.grid = {}
        assert puct_select_cell(node, SearchConfig()) is None


# ---------------------------------------------------------------------------
# Tree deepening (migrated)
# ---------------------------------------------------------------------------


class TestTreeDeepening:
    """Regression: expansion must actually deepen the tree past depth 1."""

    def test_puct_select_returns_full_cell_with_frontier_leaves(self):
        mock_view = MagicMock()
        node = TurnNode(handle=0, view=mock_view, to_move=["p1", "p2"])
        node.info = {
            "p1": InfoSet.from_actions([0], np.array([1.0])),
            "p2": InfoSet.from_actions([0], np.array([1.0])),
        }
        node.expanded = True
        config = SearchConfig(max_chance_children=2)
        node.grid = {
            (0, 0): ChanceNode(children=[
                ChanceOutcome(handle=1, leaf_value_p1=0.5),
                ChanceOutcome(handle=2, leaf_value_p1=0.6),
            ]),
        }

        cell = puct_select_cell(node, config)
        assert cell == (0, 0)

    def test_puct_expand_deepens_tree(self):
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
            "p1": InfoSet.from_actions([0], np.array([1.0])),
            "p2": InfoSet.from_actions([0], np.array([1.0])),
        }
        node.expanded = True
        config = SearchConfig(max_chance_children=1, k_actions=2)
        node.grid = {(0, 0): ChanceNode(children=[ChanceOutcome(handle=5, leaf_value_p1=0.4)])}

        mock_sim = MagicMock()
        mock_sim.view.return_value = mock_view
        mock_net = MagicMock()
        mock_net.return_value = (torch.zeros(2, as_mod.A), torch.tensor([0.5, 0.5]))
        mock_obs = _mock_obs_with_mask(2)

        with patch("search.expansion.encode", return_value=mock_obs), \
             patch("search.expansion.collate_obs_bundles", return_value=MagicMock()):
            result = puct_expand_one(node, mock_sim, mock_net, config=config)

        assert result is True
        outcome = node.grid[(0, 0)].children[0]
        assert outcome.node is not None
        assert outcome.node.expanded is True

    def test_puct_expand_descends_to_depth_3(self):
        mock_view = MagicMock()
        mock_view.terminal = False
        mock_view.to_move = ["p1", "p2"]
        mock_view.legal = {
            "p1": {"active": [{"moves": [{"id": "thunderbolt"}, {"id": "protect"}]}]},
            "p2": {"active": [{"moves": [{"id": "flamethrower"}, {"id": "protect"}]}]},
        }
        mock_view.phase = "move"

        config = SearchConfig(max_chance_children=1, k_actions=2)

        depth2_child = TurnNode(handle=10, view=mock_view, to_move=["p1", "p2"])
        depth2_child.info = {
            "p1": InfoSet.from_actions([0], np.array([1.0])),
            "p2": InfoSet.from_actions([0], np.array([1.0])),
        }
        depth2_child.expanded = True
        depth2_child.grid = {(0, 0): ChanceNode(children=[ChanceOutcome(handle=20, leaf_value_p1=0.7)])}

        root = TurnNode(handle=0, view=mock_view, to_move=["p1", "p2"])
        root.info = {
            "p1": InfoSet.from_actions([0], np.array([1.0])),
            "p2": InfoSet.from_actions([0], np.array([1.0])),
        }
        root.expanded = True
        root.grid = {(0, 0): ChanceNode(children=[ChanceOutcome(handle=10, leaf_value_p1=0.5, node=depth2_child)])}

        mock_sim = MagicMock()
        mock_sim.view.return_value = mock_view
        mock_net = MagicMock()
        mock_net.return_value = (torch.zeros(2, as_mod.A), torch.tensor([0.6, 0.6]))
        mock_obs = _mock_obs_with_mask(2)

        with patch("search.expansion.encode", return_value=mock_obs), \
             patch("search.expansion.collate_obs_bundles", return_value=MagicMock()):
            result = puct_expand_one(root, mock_sim, mock_net, config=config)

        assert result is True
        depth3_outcome = depth2_child.grid[(0, 0)].children[0]
        assert depth3_outcome.node is not None
        assert depth3_outcome.node.expanded is True

    def test_puct_select_returns_saturated_cell_for_descent(self):
        mock_view = MagicMock()
        child_node = TurnNode(handle=10, view=mock_view, to_move=["p1", "p2"])
        child_node.info = {
            "p1": InfoSet.from_actions([0], np.array([1.0])),
            "p2": InfoSet.from_actions([0], np.array([1.0])),
        }
        child_node.expanded = True
        child_node.grid = {(0, 0): ChanceNode(children=[ChanceOutcome(handle=20, leaf_value_p1=0.5)])}

        root = TurnNode(handle=0, view=mock_view, to_move=["p1", "p2"])
        root.info = {
            "p1": InfoSet.from_actions([0], np.array([1.0])),
            "p2": InfoSet.from_actions([0], np.array([1.0])),
        }
        root.expanded = True
        config = SearchConfig(max_chance_children=1)
        root.grid = {(0, 0): ChanceNode(children=[ChanceOutcome(handle=10, leaf_value_p1=0.5, node=child_node)])}

        cell = puct_select_cell(root, config)
        assert cell == (0, 0)

    def test_puct_select_skips_dead_cells(self):
        mock_view = MagicMock()
        node = TurnNode(handle=0, view=mock_view, to_move=["p1", "p2"])
        node.info = {
            "p1": InfoSet(
                actions=[0, 1], prior=np.array([0.9, 0.1]),
                regret=np.zeros(2), strategy_sum=np.zeros(2), visits=np.zeros(2, dtype=np.int64),
            ),
            "p2": InfoSet(
                actions=[0], prior=np.array([1.0]),
                regret=np.zeros(1), strategy_sum=np.zeros(1), visits=np.zeros(1, dtype=np.int64),
            ),
        }
        node.expanded = True
        config = SearchConfig(max_chance_children=5)
        node.grid = {(1, 0): ChanceNode(children=[ChanceOutcome(handle=1, leaf_value_p1=0.5)])}

        cell = puct_select_cell(node, config)
        assert cell == (1, 0)

    def test_puct_expand_produces_nonuniform_visits(self):
        mock_view = MagicMock()
        mock_view.terminal = False
        mock_view.to_move = ["p1", "p2"]
        mock_view.phase = "move"

        node = TurnNode(handle=0, view=mock_view, to_move=["p1", "p2"])
        node.info = {
            "p1": InfoSet.from_actions([0, 1], np.array([0.5, 0.5])),
            "p2": InfoSet.from_actions([0], np.array([1.0])),
        }
        node.expanded = True
        config = SearchConfig(max_chance_children=2, k_actions=2)
        node.grid = {
            (0, 0): ChanceNode(children=[ChanceOutcome(handle=1, leaf_value_p1=0.5)]),
            (1, 0): ChanceNode(children=[ChanceOutcome(handle=2, leaf_value_p1=0.3)]),
        }

        mock_sim = MagicMock()
        mock_sim.step.return_value = MagicMock(
            view=MagicMock(terminal=True, utility={"p1": 0.5}), child=99,
        )

        result = puct_expand_one(node, mock_sim, MagicMock(), config)
        assert result is True
        p1_visits = node.info["p1"].visits
        assert p1_visits.sum() == 1
        assert (p1_visits[0] == 1) != (p1_visits[1] == 1)


# ---------------------------------------------------------------------------
# puct_expand_one saturated / depth cap (gaps #19, #20)
# ---------------------------------------------------------------------------


class TestPUCTExpandOneBoundary:
    """Test puct_expand_one when tree is saturated or depth-capped."""

    def test_skips_terminal_outcome_expands_sibling(self):
        """First outcome is terminal, second is expandable -> should expand the second."""
        mock_view = MagicMock()
        mock_view.terminal = False
        mock_view.to_move = ["p1", "p2"]
        mock_view.phase = "move"

        config = SearchConfig(max_chance_children=2, k_actions=2)

        # Two outcomes: first terminal (node stays None), second expandable
        terminal_outcome = ChanceOutcome(handle=10, leaf_value_p1=1.0)
        expandable_outcome = ChanceOutcome(handle=20, leaf_value_p1=0.5)

        root = TurnNode(handle=0, view=mock_view, to_move=["p1", "p2"])
        root.info = {
            "p1": InfoSet.from_actions([0], np.array([1.0])),
            "p2": InfoSet.from_actions([0], np.array([1.0])),
        }
        root.expanded = True
        root.grid = {(0, 0): ChanceNode(children=[terminal_outcome, expandable_outcome])}

        mock_sim = MagicMock()
        # First call (terminal outcome) returns terminal view
        # Second call (expandable outcome) returns non-terminal view
        mock_sim.view.side_effect = [
            MagicMock(terminal=True),
            mock_view,
        ]
        mock_sim.step.return_value = MagicMock(
            view=MagicMock(terminal=True, utility={"p1": 0.0}), child=300,
        )

        mock_net = MagicMock()
        mock_obs = _mock_obs_with_mask(2)
        mock_net.return_value = (torch.zeros(2, as_mod.A), torch.tensor([0.5, 0.5]))

        with patch("search.expansion.encode", return_value=mock_obs), \
             patch("search.expansion.collate_obs_bundles", return_value=MagicMock()):
            result = puct_expand_one(root, mock_sim, mock_net, config=config)

        assert result is True, "Should have expanded the second (non-terminal) outcome"
        assert expandable_outcome.node is not None, "Second outcome should have been expanded"
        assert terminal_outcome.node is None, "Terminal outcome should remain unexpanded"

    def test_saturated_tree_returns_false(self):
        """All cells full, all outcomes expanded with no further grid -> False (gap #19)."""
        mock_view = MagicMock()

        # Leaf node at bottom: expanded but with an empty grid (nothing to expand)
        leaf = TurnNode(handle=10, view=mock_view, to_move=["p1", "p2"])
        leaf.info = {
            "p1": InfoSet.from_actions([0], np.array([1.0])),
            "p2": InfoSet.from_actions([0], np.array([1.0])),
        }
        leaf.expanded = True
        leaf.grid = {}  # Nothing below

        root = TurnNode(handle=0, view=mock_view, to_move=["p1", "p2"])
        root.info = {
            "p1": InfoSet.from_actions([0], np.array([1.0])),
            "p2": InfoSet.from_actions([0], np.array([1.0])),
        }
        root.expanded = True
        config = SearchConfig(max_chance_children=1)
        root.grid = {(0, 0): ChanceNode(children=[ChanceOutcome(handle=10, leaf_value_p1=0.5, node=leaf)])}

        result = puct_expand_one(root, MagicMock(), MagicMock(), config)
        assert result is False

    def test_max_descent_depth_returns_false(self):
        """A chain of 51+ expanded nodes hits _MAX_DESCENT_DEPTH and returns False (gap #20)."""
        mock_view = MagicMock()
        config = SearchConfig(max_chance_children=1)

        # Build a chain of 55 nodes deep
        nodes = []
        for i in range(55):
            n = TurnNode(handle=i, view=mock_view, to_move=["p1", "p2"])
            n.info = {
                "p1": InfoSet.from_actions([0], np.array([1.0])),
                "p2": InfoSet.from_actions([0], np.array([1.0])),
            }
            n.expanded = True
            nodes.append(n)

        # Wire them: node[i] -> (0,0) -> node[i+1]
        for i in range(len(nodes) - 1):
            nodes[i].grid = {
                (0, 0): ChanceNode(children=[
                    ChanceOutcome(handle=100 + i, leaf_value_p1=0.5, node=nodes[i + 1]),
                ]),
            }
        # Bottom node has empty grid
        nodes[-1].grid = {}

        result = puct_expand_one(nodes[0], MagicMock(), MagicMock(), config)
        assert result is False


# ---------------------------------------------------------------------------
# _generate_seed (gap #23)
# ---------------------------------------------------------------------------


class TestGenerateSeed:
    """Verify _generate_seed returns a valid 4-element seed list."""

    def test_returns_list_of_4_ints(self):
        seed = _generate_seed()
        assert isinstance(seed, list)
        assert len(seed) == 4
        for s in seed:
            assert isinstance(s, int)
            assert 0 <= s < 2**31


# ---------------------------------------------------------------------------
# Large k sparse grid (gap #30)
# ---------------------------------------------------------------------------


class TestSparseGrid:
    """CFR+ with a large k and sparsely populated grid."""

    def test_cfr_on_sparse_6x6_grid(self):
        """k=6 per side (36 possible cells), only 10 populated. CFR+ should not crash."""
        from search import cfr_update_recursive, extract_average_strategy

        mock_view = MagicMock()
        node = TurnNode(handle=0, view=mock_view, to_move=["p1", "p2"])
        node.info = {
            "p1": InfoSet.from_actions(list(range(6)), np.ones(6) / 6),
            "p2": InfoSet.from_actions(list(range(6)), np.ones(6) / 6),
        }
        node.expanded = True
        # Only populate 10 of 36 cells
        node.grid = {}
        handle_counter = 0
        for i in range(6):
            for j in range(6):
                if handle_counter < 10:
                    node.grid[(i, j)] = ChanceNode(
                        children=[ChanceOutcome(handle=handle_counter, leaf_value_p1=float(handle_counter) / 10)]
                    )
                    handle_counter += 1
                if handle_counter >= 10:
                    break
            if handle_counter >= 10:
                break

        # Run CFR+ — should not crash, and strategies should be valid
        for t in range(1, 201):
            cfr_update_recursive(node, t)

        sigma_bar_p1 = extract_average_strategy(node, "p1")
        assert abs(sum(sigma_bar_p1.values()) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# _forced_view_choices (forced-node collapse)
# ---------------------------------------------------------------------------


class TestForcedViewChoices:
    """Test the forced-decision detection helper."""

    def test_terminal_returns_none(self):
        """Terminal view is never forced."""
        view = MagicMock()
        view.terminal = True
        assert _forced_view_choices(view) is None

    def test_empty_to_move_returns_none(self):
        """No acting sides means not forced."""
        view = MagicMock()
        view.terminal = False
        view.to_move = []
        assert _forced_view_choices(view) is None

    def test_non_list_to_move_returns_none(self):
        """Non-list to_move (e.g. unconfigured MagicMock) returns None safely."""
        view = MagicMock()
        view.terminal = False
        # to_move is a MagicMock (not a list) — should not crash
        assert _forced_view_choices(view) is None

    def test_single_legal_action_both_sides(self):
        """Both sides have exactly one legal action -> returns forced choices."""
        view = MagicMock()
        view.terminal = False
        view.phase = "forceSwitch"
        view.to_move = ["p1", "p2"]
        view.legal = {
            "p1": {"forceSwitch": [True, False], "side": {"pokemon": [
                {"condition": "0 fnt"}, {"condition": "0 fnt"},
                {"condition": "100/100"}, {"condition": "0 fnt"},
            ]}},
            "p2": {"forceSwitch": [True, False], "side": {"pokemon": [
                {"condition": "0 fnt"}, {"condition": "0 fnt"},
                {"condition": "0 fnt"}, {"condition": "120/120"},
            ]}},
        }

        result = _forced_view_choices(view)
        assert result is not None
        assert "p1" in result
        assert "p2" in result

    def test_multi_legal_action_returns_none(self):
        """One side has multiple legal actions -> not forced."""
        view = MagicMock()
        view.terminal = False
        view.phase = "move"
        view.to_move = ["p1", "p2"]
        view.legal = {
            "p1": {"active": [
                {"moves": [{"id": "thunderbolt", "pp": 10, "target": "normal"}, {"id": "protect", "pp": 10, "target": "self"}]},
                {"moves": [{"id": "flamethrower", "pp": 10, "target": "normal"}]},
            ], "side": {"pokemon": [
                {"condition": "100/100"}, {"condition": "100/100"},
                {"condition": "100/100"}, {"condition": "100/100"},
            ]}},
            "p2": {"active": [
                {"moves": [{"id": "flamethrower", "pp": 10, "target": "normal"}]},
                {"moves": [{"id": "protect", "pp": 10, "target": "self"}]},
            ], "side": {"pokemon": [
                {"condition": "100/100"}, {"condition": "100/100"},
                {"condition": "100/100"}, {"condition": "100/100"},
            ]}},
        }
        assert _forced_view_choices(view) is None

    def test_unilateral_single_action(self):
        """Only one side acting with one legal action -> forced."""
        view = MagicMock()
        view.terminal = False
        view.phase = "forceSwitch"
        view.to_move = ["p1"]
        view.legal = {
            "p1": {"forceSwitch": [True, False], "side": {"pokemon": [
                {"condition": "0 fnt"}, {"condition": "0 fnt"},
                {"condition": "100/100"}, {"condition": "0 fnt"},
            ]}},
        }

        result = _forced_view_choices(view)
        assert result is not None
        assert "p1" in result


# ---------------------------------------------------------------------------
# _collapse_forced (forced-node collapse)
# ---------------------------------------------------------------------------


class TestCollapseForced:
    """Test the forced-decision chain collapse helper."""

    def test_genuine_decision_returns_immediately(self):
        """Non-forced view is returned as-is without stepping."""
        view = MagicMock()
        view.terminal = False
        view.phase = "move"
        view.to_move = ["p1", "p2"]
        view.legal = {
            "p1": {"active": [
                {"moves": [{"id": "thunderbolt", "pp": 10, "target": "normal"}, {"id": "protect", "pp": 10, "target": "self"}]},
                {"moves": [{"id": "flamethrower", "pp": 10, "target": "normal"}]},
            ], "side": {"pokemon": [
                {"condition": "100/100"}, {"condition": "100/100"},
                {"condition": "100/100"}, {"condition": "100/100"},
            ]}},
            "p2": {"active": [
                {"moves": [{"id": "flamethrower", "pp": 10, "target": "normal"}]},
                {"moves": [{"id": "protect", "pp": 10, "target": "self"}]},
            ], "side": {"pokemon": [
                {"condition": "100/100"}, {"condition": "100/100"},
                {"condition": "100/100"}, {"condition": "100/100"},
            ]}},
        }

        mock_sim = MagicMock()
        result_handle, result_view = _collapse_forced(mock_sim, 42, view)

        assert result_handle == 42
        assert result_view is view
        mock_sim.step.assert_not_called()

    def test_terminal_returns_immediately(self):
        """Terminal view is returned as-is without stepping."""
        view = MagicMock()
        view.terminal = True
        mock_sim = MagicMock()

        result_handle, result_view = _collapse_forced(mock_sim, 42, view)
        assert result_handle == 42
        assert result_view is view
        mock_sim.step.assert_not_called()

    def test_single_forced_step(self):
        """One forced decision is collapsed to the next genuine decision."""
        forced_view = MagicMock()
        forced_view.terminal = False
        forced_view.phase = "forceSwitch"
        forced_view.to_move = ["p1"]
        forced_view.legal = {
            "p1": {"forceSwitch": [True, False], "side": {"pokemon": [
                {"condition": "0 fnt"}, {"condition": "0 fnt"},
                {"condition": "100/100"}, {"condition": "0 fnt"},
            ]}},
        }

        # After stepping the forced decision, we reach a genuine move phase
        genuine_view = MagicMock()
        genuine_view.terminal = False
        genuine_view.to_move = ["p1", "p2"]
        genuine_view.phase = "move"
        genuine_view.legal = {
            "p1": {"active": [
                {"moves": [{"id": "a", "pp": 10, "target": "normal"}, {"id": "b", "pp": 10, "target": "self"}]},
                {"moves": [{"id": "c", "pp": 10, "target": "normal"}]},
            ], "side": {"pokemon": [
                {"condition": "100/100"}, {"condition": "100/100"},
                {"condition": "100/100"}, {"condition": "100/100"},
            ]}},
            "p2": {"active": [
                {"moves": [{"id": "d", "pp": 10, "target": "normal"}]},
                {"moves": [{"id": "e", "pp": 10, "target": "self"}]},
            ], "side": {"pokemon": [
                {"condition": "100/100"}, {"condition": "100/100"},
                {"condition": "100/100"}, {"condition": "100/100"},
            ]}},
        }

        mock_sim = MagicMock()
        mock_sim.step.return_value = MagicMock(child=99, view=genuine_view)

        result_handle, result_view = _collapse_forced(mock_sim, 42, forced_view)

        assert result_handle == 99
        assert result_view is genuine_view
        mock_sim.step.assert_called_once()

    def test_chain_to_terminal(self):
        """Forced chain ending at terminal returns the terminal state."""
        forced_view = MagicMock()
        forced_view.terminal = False
        forced_view.phase = "forceSwitch"
        forced_view.to_move = ["p1"]
        forced_view.legal = {
            "p1": {"forceSwitch": [True, False], "side": {"pokemon": [
                {"condition": "0 fnt"}, {"condition": "0 fnt"},
                {"condition": "100/100"}, {"condition": "0 fnt"},
            ]}},
        }

        terminal_view = MagicMock()
        terminal_view.terminal = True

        mock_sim = MagicMock()
        mock_sim.step.return_value = MagicMock(child=77, view=terminal_view)

        result_handle, result_view = _collapse_forced(mock_sim, 10, forced_view)

        assert result_handle == 77
        assert result_view is terminal_view

    def test_simerror_stops_collapse(self):
        """SimError during forced step -> returns last good state."""
        forced_view = MagicMock()
        forced_view.terminal = False
        forced_view.phase = "forceSwitch"
        forced_view.to_move = ["p1"]
        forced_view.legal = {
            "p1": {"forceSwitch": [True, False], "side": {"pokemon": [
                {"condition": "0 fnt"}, {"condition": "0 fnt"},
                {"condition": "100/100"}, {"condition": "0 fnt"},
            ]}},
        }

        mock_sim = MagicMock()
        mock_sim.step.side_effect = SimError("engine error")

        result_handle, result_view = _collapse_forced(mock_sim, 42, forced_view)

        # Falls back to the original handle/view
        assert result_handle == 42
        assert result_view is forced_view

    def test_mock_view_passthrough(self):
        """Unconfigured MagicMock view (non-list to_move) passes through safely."""
        bare_mock = MagicMock(terminal=False)
        mock_sim = MagicMock()

        result_handle, result_view = _collapse_forced(mock_sim, 42, bare_mock)

        assert result_handle == 42
        assert result_view is bare_mock
        mock_sim.step.assert_not_called()
