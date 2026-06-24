"""Unit tests for search.types — data structures and factory methods.

New gap tests:
  - InfoSet.from_actions output shapes, dtypes, values (gap #12)
  - InfoSet.noop() contract (gap #12)
  - InfoSet.empty() contract (gap #12)
  - SearchConfig default snapshot (gap #24)
  - TurnNode default fields (gap #25)
"""
from __future__ import annotations

import numpy as np

from search.types import ChanceNode, InfoSet, SearchConfig, TurnNode

# ---------------------------------------------------------------------------
# InfoSet factory methods (gap #12)
# ---------------------------------------------------------------------------


class TestInfoSetFromActions:
    """Verify InfoSet.from_actions produces correct shapes, dtypes, and initial values."""

    def test_shapes_match_action_count(self):
        actions = [10, 20, 30]
        prior = np.array([0.5, 0.3, 0.2])
        info = InfoSet.from_actions(actions, prior)

        assert info.actions == [10, 20, 30]
        assert info.prior.shape == (3,)
        assert info.regret.shape == (3,)
        assert info.strategy_sum.shape == (3,)
        assert info.visits.shape == (3,)

    def test_regret_and_strategy_sum_zeroed(self):
        info = InfoSet.from_actions([0, 1], np.array([0.6, 0.4]))
        np.testing.assert_array_equal(info.regret, [0.0, 0.0])
        np.testing.assert_array_equal(info.strategy_sum, [0.0, 0.0])
        np.testing.assert_array_equal(info.visits, [0, 0])

    def test_dtypes(self):
        info = InfoSet.from_actions([0], np.array([1.0]))
        assert info.regret.dtype == np.float64
        assert info.strategy_sum.dtype == np.float64
        assert info.visits.dtype == np.int64

    def test_prior_preserved(self):
        prior = np.array([0.7, 0.2, 0.1])
        info = InfoSet.from_actions([5, 6, 7], prior)
        np.testing.assert_array_equal(info.prior, prior)


class TestInfoSetNoop:
    """Verify InfoSet.noop() produces a valid single-action noop."""

    def test_noop_actions(self):
        info = InfoSet.noop()
        assert info.actions == [0]

    def test_noop_prior(self):
        info = InfoSet.noop()
        np.testing.assert_array_equal(info.prior, [1.0])

    def test_noop_array_shapes(self):
        info = InfoSet.noop()
        assert info.regret.shape == (1,)
        assert info.strategy_sum.shape == (1,)
        assert info.visits.shape == (1,)


class TestInfoSetEmpty:
    """Verify InfoSet.empty() produces a zero-length InfoSet."""

    def test_empty_actions(self):
        info = InfoSet.empty()
        assert info.actions == []

    def test_empty_array_shapes(self):
        info = InfoSet.empty()
        assert info.regret.shape == (0,)
        assert info.strategy_sum.shape == (0,)
        assert info.visits.shape == (0,)
        assert info.prior.shape == (0,)


# ---------------------------------------------------------------------------
# SearchConfig defaults snapshot (gap #24)
# ---------------------------------------------------------------------------


class TestSearchConfigDefaults:
    """Snapshot test: default values haven't accidentally changed."""

    def test_default_values(self):
        cfg = SearchConfig()
        assert cfg.k_actions == 6
        assert cfg.max_chance_children == 5
        assert cfg.expansion_budget == 24
        assert cfg.cfr_iters_per_expansion == 10
        assert cfg.c_puct == 2.0


# ---------------------------------------------------------------------------
# TurnNode default fields (gap #25)
# ---------------------------------------------------------------------------


class TestTurnNodeDefaults:
    """Verify TurnNode default_factory fields are empty/False."""

    def test_defaults(self):
        from unittest.mock import MagicMock
        node = TurnNode(handle=0, view=MagicMock())
        assert node.to_move == []
        assert node.info == {}
        assert node.grid == {}
        assert node.expanded is False


class TestChanceNodeDefaults:
    """Verify ChanceNode default_factory for children."""

    def test_defaults(self):
        cn = ChanceNode()
        assert cn.children == []
