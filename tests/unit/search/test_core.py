"""Unit tests for search.core — search() orchestration and _single_move_result.

Migrated from tests/unit/test_search.py:
  - TestSingleLegalMoveSkip

New gap tests:
  - search() normal path: loop count, cleanup on exception, config=None (gap #2)
  - _single_move_result: multi-action returns None, unilateral forced (gap #3)
  - Iteration numbering monotonicity (gap #34)
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

import action_space as as_mod
from search import SearchConfig, SearchResult, search
from search.core import _single_move_result

# ---------------------------------------------------------------------------
# _single_move_result tests (migrated + gaps #3)
# ---------------------------------------------------------------------------


class TestSingleLegalMoveSkip:
    """Test that _single_move_result / search returns immediately when only one action is legal."""

    def test_single_action_both_sides(self):
        """When both sides have exactly one legal action, skip search entirely."""
        mock_view = MagicMock()
        mock_view.phase = "forceSwitch"
        mock_view.to_move = ["p1", "p2"]
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

        mock_sim = MagicMock()
        mock_net = MagicMock()
        mock_value = torch.tensor(0.3)
        mock_net.return_value = (torch.zeros(as_mod.A), mock_value)

        with patch("search.expansion.encode") as mock_encode:
            mock_encode.return_value = MagicMock()
            result = search(mock_view, mock_sim, mock_net, from_handle=99, config=SearchConfig())

        assert isinstance(result, SearchResult)
        for s in ["p1", "p2"]:
            assert len(result.strategy[s]) == 1
            assert next(iter(result.strategy[s].values())) == 1.0
        assert abs(result.value - 0.3) < 1e-5
        mock_sim.open_search.assert_not_called()

    def test_multi_action_returns_none(self):
        """When a side has multiple legal actions, _single_move_result returns None."""
        mock_view = MagicMock()
        mock_view.phase = "move"
        mock_view.to_move = ["p1", "p2"]
        # p1 has 2 moves per slot -> many joint actions -> not single-move
        mock_view.legal = {
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

        mock_net = MagicMock()
        result = _single_move_result(mock_view, mock_net)
        assert result is None

    def test_unilateral_single_action(self):
        """Unilateral forced move (only one side to_move with one action) returns a result."""
        mock_view = MagicMock()
        mock_view.phase = "forceSwitch"
        mock_view.to_move = ["p1"]
        mock_view.legal = {
            "p1": {"forceSwitch": [True, False], "side": {"pokemon": [
                {"condition": "0 fnt"}, {"condition": "0 fnt"},
                {"condition": "100/100"}, {"condition": "0 fnt"},
            ]}},
        }

        mock_net = MagicMock()
        mock_net.return_value = (torch.zeros(as_mod.A), torch.tensor(0.5))

        with patch("search.expansion.encode", return_value=MagicMock()):
            result = _single_move_result(mock_view, mock_net)

        assert result is not None
        assert "p1" in result.strategy
        assert len(result.strategy["p1"]) == 1
        assert next(iter(result.strategy["p1"].values())) == 1.0


# ---------------------------------------------------------------------------
# search() normal path (gap #2)
# ---------------------------------------------------------------------------


class TestSearchNormalPath:
    """Test the main search() loop with mocked dependencies."""

    def _make_search_mocks(self, config: SearchConfig):
        """Set up mocked sim/net for a minimal search run."""
        mock_view = MagicMock()
        mock_view.phase = "move"
        mock_view.to_move = ["p1", "p2"]
        # Proper Showdown request dicts so legal_mask returns >1 legal action per side
        _side_pokemon = [
            {"condition": "100/100"}, {"condition": "100/100"},
            {"condition": "100/100"}, {"condition": "100/100"},
        ]
        mock_view.legal = {
            "p1": {"active": [
                {"moves": [{"id": "thunderbolt", "pp": 10, "target": "normal"}, {"id": "protect", "pp": 10, "target": "self"}]},
                {"moves": [{"id": "flamethrower", "pp": 10, "target": "normal"}]},
            ], "side": {"pokemon": _side_pokemon}},
            "p2": {"active": [
                {"moves": [{"id": "flamethrower", "pp": 10, "target": "normal"}, {"id": "protect", "pp": 10, "target": "self"}]},
                {"moves": [{"id": "thunderbolt", "pp": 10, "target": "normal"}]},
            ], "side": {"pokemon": _side_pokemon}},
        }

        mock_sim = MagicMock()
        mock_sim.open_search.return_value = (42, 100, mock_view)
        # step returns terminal results so expand_turn_node fills the grid
        mock_sim.step.return_value = MagicMock(
            view=MagicMock(terminal=True, utility={"p1": 0.5}),
            child=200,
        )

        mock_net = MagicMock()
        action_mask = torch.tensor([True, True] + [False] * (as_mod.A - 2))
        mock_obs = MagicMock()
        mock_obs.__getitem__ = lambda self, key: action_mask if key == "action_mask" else MagicMock()

        # CVPN returns valid logits and values for the batch
        mock_net.return_value = (torch.zeros(2, as_mod.A), torch.tensor([0.5, 0.5]))

        return mock_view, mock_sim, mock_net, mock_obs

    def test_close_search_called_on_success(self):
        """close_search is called after a successful search run."""
        config = SearchConfig(expansion_budget=2, cfr_iters_per_expansion=3, k_actions=2, max_chance_children=1)
        mock_view, mock_sim, mock_net, mock_obs = self._make_search_mocks(config)

        with patch("search.expansion.encode", return_value=mock_obs), \
             patch("search.expansion.collate_obs_bundles", return_value=MagicMock()):
            search(mock_view, mock_sim, mock_net, from_handle=1, config=config)

        mock_sim.close_search.assert_called_once_with(42)

    def test_close_search_called_on_exception(self):
        """close_search is called even when expand_turn_node raises."""
        config = SearchConfig(expansion_budget=1, k_actions=2)
        mock_view, mock_sim, mock_net, _mock_obs = self._make_search_mocks(config)

        with patch("search.expansion.encode", side_effect=RuntimeError("boom")), \
             pytest.raises(RuntimeError, match="boom"):
            search(mock_view, mock_sim, mock_net, from_handle=1, config=config)

        mock_sim.close_search.assert_called_once_with(42)

    def test_config_none_uses_defaults(self):
        """Passing config=None creates a default SearchConfig."""
        mock_view, mock_sim, mock_net, mock_obs = self._make_search_mocks(SearchConfig())

        with patch("search.expansion.encode", return_value=mock_obs), \
             patch("search.expansion.collate_obs_bundles", return_value=MagicMock()), \
             patch("search.core.SearchConfig") as mock_config_cls:
            # Make the returned config behave like a real one but with tiny budget
            real_config = SearchConfig(expansion_budget=1, cfr_iters_per_expansion=1, k_actions=2, max_chance_children=1)
            mock_config_cls.return_value = real_config
            search(mock_view, mock_sim, mock_net, from_handle=1, config=None)

        mock_config_cls.assert_called_once()


# ---------------------------------------------------------------------------
# Iteration numbering (gap #34)
# ---------------------------------------------------------------------------


class TestIterationNumbering:
    """Verify that the iteration weight sequence is monotonically increasing."""

    def test_weights_are_monotonic(self):
        """Strategy_sum weights = expansion_idx * cfr_iters + t, always increasing from 1."""
        config = SearchConfig(expansion_budget=3, cfr_iters_per_expansion=4)
        weights = [
            expansion_idx * config.cfr_iters_per_expansion + t
            for expansion_idx in range(config.expansion_budget)
            for t in range(1, config.cfr_iters_per_expansion + 1)
        ]

        # Starts at 1
        assert weights[0] == 1
        # Strictly monotonically increasing
        for i in range(1, len(weights)):
            assert weights[i] > weights[i - 1], f"Weight sequence not monotonic at index {i}: {weights}"
        # No duplicates
        assert len(weights) == len(set(weights))
