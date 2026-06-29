"""Unit tests for the self_play module.

Tests forced-decision detection, action sampling, emission rules, z-stamping,
and the max_decisions safety cap. Uses mocks/stubs — no subprocess needed.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from action_space import forced_actions
from self_play import (
    CurriculumMatchupSource,
    SelfPlayConfig,
    SparsePolicy,
    TrainingTuple,
    TupleMeta,
    _PendingTuple,
    _advance,
    _is_forced,
    _rng_seed,
    _sample_action,
    _sample_and_step,
    run,
)


# ---------------------------------------------------------------------------
# Helpers: minimal StateView stubs
# ---------------------------------------------------------------------------


def _make_view(
    phase: str = "move",
    to_move: list[str] | None = None,
    legal: dict[str, Any] | None = None,
    terminal: bool = False,
    utility: dict[str, float] | None = None,
) -> MagicMock:
    """Create a mock StateView with the given properties."""
    view = MagicMock()
    view.phase = phase
    view.to_move = to_move or []
    view.legal = legal or {}
    view.terminal = terminal
    view.utility = utility
    return view


# ---------------------------------------------------------------------------
# Tests: forced_actions (canonical helper from action_space)
# ---------------------------------------------------------------------------


class TestForcedActions:
    """Tests for the canonical forced-decision predicate."""

    def test_empty_to_move_returns_none(self):
        """Empty to_move means no one acts — not forced."""
        assert forced_actions({}, [], "move") is None

    @patch("action_space.legal_mask")
    def test_single_legal_action_both_sides(self, mock_mask):
        """Both sides have exactly one legal action -> returns forced indices."""
        mock_mask.return_value = np.array([False, True, False, False, False])
        result = forced_actions({"p1": {}, "p2": {}}, ["p1", "p2"], "move")
        assert result == {"p1": 1, "p2": 1}

    @patch("action_space.legal_mask")
    def test_multiple_legal_actions_returns_none(self, mock_mask):
        """One side has multiple legal actions -> not forced."""
        mock_mask.return_value = np.array([True, True, False, False, False])
        assert forced_actions({"p1": {}, "p2": {}}, ["p1", "p2"], "move") is None

    @patch("action_space.legal_mask")
    def test_unilateral_single_action(self, mock_mask):
        """One side acting with one legal action -> forced."""
        mock_mask.return_value = np.array([False, False, True, False, False])
        result = forced_actions({"p1": {}}, ["p1"], "move")
        assert result == {"p1": 2}

    @patch("action_space.legal_mask")
    def test_unilateral_multiple_actions_returns_none(self, mock_mask):
        """One side acting with multiple legal actions -> not forced."""
        mock_mask.return_value = np.array([True, False, True, False, False])
        assert forced_actions({"p1": {}}, ["p1"], "move") is None


# ---------------------------------------------------------------------------
# Tests: _sample_action
# ---------------------------------------------------------------------------


class TestSampleAction:
    """Tests for temperature-scaled action sampling."""

    def test_temperature_1_preserves_distribution(self):
        """tau=1 should reproduce the input distribution over many samples."""
        strategy = {0: 0.7, 1: 0.2, 2: 0.1}
        rng = random.Random(42)
        counts = {0: 0, 1: 0, 2: 0}
        n = 10000

        for _ in range(n):
            idx = _sample_action(strategy, rng, temperature=1.0)
            counts[idx] += 1

        # Check approximate distribution (within 3% tolerance)
        assert abs(counts[0] / n - 0.7) < 0.03
        assert abs(counts[1] / n - 0.2) < 0.03
        assert abs(counts[2] / n - 0.1) < 0.03

    def test_temperature_near_zero_is_greedy(self):
        """tau->0 should always pick the highest-probability action."""
        strategy = {5: 0.1, 10: 0.8, 15: 0.1}
        rng = random.Random(123)

        for _ in range(100):
            idx = _sample_action(strategy, rng, temperature=1e-10)
            assert idx == 10

    def test_temperature_zero_is_greedy(self):
        """tau=0 should always pick the highest-probability action."""
        strategy = {0: 0.3, 1: 0.5, 2: 0.2}
        rng = random.Random(99)

        for _ in range(50):
            idx = _sample_action(strategy, rng, temperature=0.0)
            assert idx == 1

    def test_high_temperature_flattens_distribution(self):
        """High tau flattens toward uniform."""
        strategy = {0: 0.9, 1: 0.05, 2: 0.05}
        rng = random.Random(42)
        counts = {0: 0, 1: 0, 2: 0}
        n = 10000

        for _ in range(n):
            idx = _sample_action(strategy, rng, temperature=10.0)
            counts[idx] += 1

        # With tau=10, probabilities should be much more uniform
        # All actions should get at least 20% of samples
        assert counts[0] / n > 0.2
        assert counts[1] / n > 0.2
        assert counts[2] / n > 0.2

    def test_single_action_always_returns_it(self):
        """With a single action in strategy, always returns that action."""
        strategy = {42: 1.0}
        rng = random.Random(0)
        assert _sample_action(strategy, rng, temperature=1.0) == 42

    def test_deterministic_with_same_rng_seed(self):
        """Same RNG state produces same samples."""
        strategy = {0: 0.5, 1: 0.3, 2: 0.2}
        results_a = [_sample_action(strategy, random.Random(77), temperature=1.0) for _ in range(10)]
        results_b = [_sample_action(strategy, random.Random(77), temperature=1.0) for _ in range(10)]
        assert results_a == results_b


# ---------------------------------------------------------------------------
# Tests: CurriculumMatchupSource
# ---------------------------------------------------------------------------


class TestCurriculumMatchupSource:
    """Tests for the fixed-matchup curriculum source."""

    def test_always_returns_same_teams(self):
        """sample() should always return the same pair regardless of RNG."""
        team_a = [{"species": "Charizard"}]
        team_b = [{"species": "Venusaur"}]
        source = CurriculumMatchupSource(team_a, team_b)

        for seed in range(10):
            rng = random.Random(seed)
            a, b = source.sample(rng)
            assert a == team_a
            assert b == team_b


# ---------------------------------------------------------------------------
# Tests: SparsePolicy
# ---------------------------------------------------------------------------


class TestSparsePolicy:
    """Tests for the sparse policy container."""

    def test_creation(self):
        """Should store indices and probs as tuples."""
        sp = SparsePolicy(indices=(0, 5, 10), probs=(0.5, 0.3, 0.2))
        assert sp.indices == (0, 5, 10)
        assert sp.probs == (0.5, 0.3, 0.2)

    def test_frozen(self):
        """Frozen dataclass should be immutable."""
        sp = SparsePolicy(indices=(0,), probs=(1.0,))
        with pytest.raises(Exception):
            sp.indices = (1,)  # type: ignore


# ---------------------------------------------------------------------------
# Tests: run() generator with mocked dependencies
# ---------------------------------------------------------------------------


class TestRunGenerator:
    """Tests for the run() game loop using mocked SimClient/search."""

    @patch("self_play.search")
    @patch("self_play.encode")
    @patch("self_play.legal_mask")
    def test_max_decisions_aborts_game(self, mock_mask, mock_encode, mock_search):
        """Exceeding max_decisions should discard all tuples for that game."""
        # Configure mask to always have multiple legal actions (never forced)
        mask_arr = np.zeros(10, dtype=bool)
        mask_arr[0] = True
        mask_arr[1] = True
        mock_mask.return_value = mask_arr

        # Configure search result
        mock_result = MagicMock()
        mock_result.strategy = {"p1": {0: 0.5, 1: 0.5}, "p2": {0: 0.5, 1: 0.5}}
        mock_result.value = 0.1
        mock_search.return_value = mock_result

        # Configure encode to return a dummy ObsBundle
        mock_encode.return_value = MagicMock()

        # Create a mock SimClient that never reaches terminal
        sim = MagicMock()
        non_terminal_view = _make_view(
            phase="move", to_move=["p1", "p2"],
            legal={"p1": {}, "p2": {}}, terminal=False,
        )
        sim.new_battle.return_value = (1, non_terminal_view)
        # step always returns non-terminal
        step_result = MagicMock()
        step_result.child = 2
        step_result.view = non_terminal_view
        sim.step.return_value = step_result

        # Team source
        source = CurriculumMatchupSource([{"species": "A"}], [{"species": "B"}])
        config = SelfPlayConfig(max_decisions=3, num_games=1, master_seed=1)

        # Drain the generator — should yield nothing (game aborted)
        tuples = list(run(MagicMock(), source, sim, config))
        assert tuples == []

    @patch("self_play.search")
    @patch("self_play.encode")
    @patch("self_play.legal_mask")
    def test_z_stamping_on_terminal(self, mock_mask, mock_encode, mock_search):
        """Tuples should be stamped with the correct z from utility at terminal."""
        # First call: multiple legal (not forced), triggers search
        mask_multi = np.zeros(10, dtype=bool)
        mask_multi[0] = True
        mask_multi[1] = True
        # Second call sequence: terminal state
        mock_mask.return_value = mask_multi

        mock_result = MagicMock()
        mock_result.strategy = {"p1": {0: 1.0}, "p2": {0: 1.0}}
        mock_result.value = 0.5
        mock_search.return_value = mock_result

        mock_encode.return_value = MagicMock()

        # SimClient: first view is non-terminal, then step gives terminal
        sim = MagicMock()
        initial_view = _make_view(
            phase="move", to_move=["p1", "p2"],
            legal={"p1": {}, "p2": {}}, terminal=False,
        )
        terminal_view = _make_view(
            phase="terminal", to_move=[], terminal=True,
            utility={"p1": 1.0, "p2": -1.0},
        )
        sim.new_battle.return_value = (1, initial_view)
        step_result = MagicMock()
        step_result.child = 2
        step_result.view = terminal_view
        sim.step.return_value = step_result

        source = CurriculumMatchupSource([{"species": "A"}], [{"species": "B"}])
        config = SelfPlayConfig(num_games=1, master_seed=1)

        tuples = list(run(MagicMock(), source, sim, config))

        # Should yield 2 tuples (one per side, both had >=2 legal actions)
        assert len(tuples) == 2
        # Check z values
        p1_tuples = [t for t in tuples if t.meta.side == "p1"]
        p2_tuples = [t for t in tuples if t.meta.side == "p2"]
        assert len(p1_tuples) == 1
        assert len(p2_tuples) == 1
        assert p1_tuples[0].z == 1.0
        assert p2_tuples[0].z == -1.0

    @patch("self_play.search")
    @patch("self_play.encode")
    @patch("self_play.legal_mask")
    def test_value_negated_for_p2(self, mock_mask, mock_encode, mock_search):
        """p2's value target should be the negation of p1's search value."""
        mask_multi = np.zeros(10, dtype=bool)
        mask_multi[0] = True
        mask_multi[1] = True
        mock_mask.return_value = mask_multi

        mock_result = MagicMock()
        mock_result.strategy = {"p1": {0: 1.0}, "p2": {0: 1.0}}
        mock_result.value = 0.75
        mock_search.return_value = mock_result

        mock_encode.return_value = MagicMock()

        sim = MagicMock()
        initial_view = _make_view(
            phase="move", to_move=["p1", "p2"],
            legal={"p1": {}, "p2": {}}, terminal=False,
        )
        terminal_view = _make_view(
            phase="terminal", to_move=[], terminal=True,
            utility={"p1": 1.0, "p2": -1.0},
        )
        sim.new_battle.return_value = (1, initial_view)
        step_result = MagicMock()
        step_result.child = 2
        step_result.view = terminal_view
        sim.step.return_value = step_result

        source = CurriculumMatchupSource([{"species": "A"}], [{"species": "B"}])
        config = SelfPlayConfig(num_games=1, master_seed=1)

        tuples = list(run(MagicMock(), source, sim, config))

        p1_tuple = next(t for t in tuples if t.meta.side == "p1")
        p2_tuple = next(t for t in tuples if t.meta.side == "p2")
        assert p1_tuple.value == 0.75
        assert p2_tuple.value == -0.75

    @patch("self_play.search")
    @patch("self_play.encode")
    @patch("self_play.legal_mask")
    def test_forced_decision_emits_no_tuples(self, mock_mask, mock_encode, mock_search):
        """Forced decisions should produce zero tuples and skip search."""
        # Single legal action at index 1 → _is_forced detects forced decision
        forced_mask = np.zeros(5, dtype=bool)
        forced_mask[1] = True
        mock_mask.return_value = forced_mask

        sim = MagicMock()
        # Initial: forced decision
        forced_view = _make_view(
            phase="move", to_move=["p1", "p2"],
            legal={"p1": {}, "p2": {}}, terminal=False,
        )
        terminal_view = _make_view(
            phase="terminal", to_move=[], terminal=True,
            utility={"p1": 1.0, "p2": -1.0},
        )
        sim.new_battle.return_value = (1, forced_view)
        step_result = MagicMock()
        step_result.child = 2
        step_result.view = terminal_view
        sim.step.return_value = step_result

        source = CurriculumMatchupSource([{"species": "A"}], [{"species": "B"}])
        config = SelfPlayConfig(num_games=1, master_seed=1)

        tuples = list(run(MagicMock(), source, sim, config))

        # No tuples because the only decision was forced, then terminal
        assert len(tuples) == 0
        # search() should never have been called
        mock_search.assert_not_called()

    @patch("self_play.search")
    @patch("self_play.encode")
    @patch("self_play.legal_mask")
    def test_unilateral_emits_one_tuple(self, mock_mask, mock_encode, mock_search):
        """A unilateral decision (one side forced, one choosing) emits 1 tuple."""
        # Use distinct sentinel objects to differentiate p1 vs p2 requests
        p1_request = {"_side": "p1"}
        p2_request = {"_side": "p2"}

        def mask_side_effect(request, phase):
            if request is p1_request:
                return np.array([True, True, False])  # p1: 2 legal
            else:
                return np.array([False, True, False])  # p2: 1 legal

        mock_mask.side_effect = mask_side_effect

        mock_result = MagicMock()
        mock_result.strategy = {"p1": {0: 0.6, 1: 0.4}, "p2": {1: 1.0}}
        mock_result.value = 0.3
        mock_search.return_value = mock_result

        mock_encode.return_value = MagicMock()

        sim = MagicMock()
        # Unilateral: both in to_move but p2 has only 1 legal action
        unilateral_view = _make_view(
            phase="forceSwitch", to_move=["p1", "p2"],
            legal={"p1": p1_request, "p2": p2_request}, terminal=False,
        )
        terminal_view = _make_view(
            phase="terminal", to_move=[], terminal=True,
            utility={"p1": 1.0, "p2": -1.0},
        )
        sim.new_battle.return_value = (1, unilateral_view)
        step_result = MagicMock()
        step_result.child = 2
        step_result.view = terminal_view
        sim.step.return_value = step_result

        source = CurriculumMatchupSource([{"species": "A"}], [{"species": "B"}])
        config = SelfPlayConfig(num_games=1, master_seed=1)

        tuples = list(run(MagicMock(), source, sim, config))

        # Only p1 had >=2 legal actions, so only 1 tuple
        assert len(tuples) == 1
        assert tuples[0].meta.side == "p1"

    @patch("self_play.search")
    @patch("self_play.encode")
    @patch("self_play.legal_mask")
    def test_sim_error_discards_game(self, mock_mask, mock_encode, mock_search):
        """SimError mid-game should discard all tuples and continue."""
        from sim_client import SimError

        sim = MagicMock()
        sim.new_battle.side_effect = SimError("engine crashed")

        source = CurriculumMatchupSource([{"species": "A"}], [{"species": "B"}])
        config = SelfPlayConfig(num_games=2, master_seed=1)

        # Both games fail -> no tuples
        tuples = list(run(MagicMock(), source, sim, config))
        assert tuples == []

    @patch("self_play.search")
    @patch("self_play.encode")
    @patch("self_play.legal_mask")
    def test_sim_error_mid_game_releases_handle(self, mock_mask, mock_encode, mock_search):
        """Regression: SimError after new_battle must release the live handle."""
        from sim_client import SimError

        # Mask with multiple legal actions so search() is invoked
        mask_multi = np.zeros(10, dtype=bool)
        mask_multi[0] = True
        mask_multi[1] = True
        mock_mask.return_value = mask_multi

        # search() raises SimError to simulate a mid-game crash
        mock_search.side_effect = SimError("search exploded")

        sim = MagicMock()
        initial_view = _make_view(
            phase="move", to_move=["p1", "p2"],
            legal={"p1": {}, "p2": {}}, terminal=False,
        )
        sim.new_battle.return_value = (7, initial_view)

        source = CurriculumMatchupSource([{"species": "A"}], [{"species": "B"}])
        config = SelfPlayConfig(num_games=1, master_seed=1)

        tuples = list(run(MagicMock(), source, sim, config))

        # Game aborted — no tuples emitted
        assert tuples == []
        # The handle (7) from new_battle must have been released
        sim.release.assert_called_once_with(7)

    @patch("self_play.search")
    @patch("self_play.encode")
    @patch("self_play.legal_mask")
    def test_multiple_games(self, mock_mask, mock_encode, mock_search):
        """Running multiple games should yield tuples from all of them."""
        mask_multi = np.zeros(10, dtype=bool)
        mask_multi[0] = True
        mask_multi[1] = True
        mock_mask.return_value = mask_multi

        mock_result = MagicMock()
        mock_result.strategy = {"p1": {0: 1.0}, "p2": {0: 1.0}}
        mock_result.value = 0.0
        mock_search.return_value = mock_result

        mock_encode.return_value = MagicMock()

        sim = MagicMock()
        initial_view = _make_view(
            phase="move", to_move=["p1", "p2"],
            legal={"p1": {}, "p2": {}}, terminal=False,
        )
        terminal_view = _make_view(
            phase="terminal", to_move=[], terminal=True,
            utility={"p1": 0.0, "p2": 0.0},
        )
        sim.new_battle.return_value = (1, initial_view)
        step_result = MagicMock()
        step_result.child = 2
        step_result.view = terminal_view
        sim.step.return_value = step_result

        source = CurriculumMatchupSource([{"species": "A"}], [{"species": "B"}])
        config = SelfPlayConfig(num_games=3, master_seed=1)

        tuples = list(run(MagicMock(), source, sim, config))

        # Each game: 1 decision -> 2 tuples (p1 + p2) -> 3 games = 6 tuples
        assert len(tuples) == 6
        # All draws
        for t in tuples:
            assert t.z == 0.0

    @patch("self_play.search")
    @patch("self_play.encode")
    @patch("self_play.legal_mask")
    def test_meta_fields_populated(self, mock_mask, mock_encode, mock_search):
        """TupleMeta should contain correct generation, game_id, decision_idx, phase, side."""
        mask_multi = np.zeros(10, dtype=bool)
        mask_multi[0] = True
        mask_multi[1] = True
        mock_mask.return_value = mask_multi

        mock_result = MagicMock()
        mock_result.strategy = {"p1": {0: 1.0}, "p2": {0: 1.0}}
        mock_result.value = 0.0
        mock_search.return_value = mock_result

        mock_encode.return_value = MagicMock()

        sim = MagicMock()
        initial_view = _make_view(
            phase="move", to_move=["p1", "p2"],
            legal={"p1": {}, "p2": {}}, terminal=False,
        )
        terminal_view = _make_view(
            phase="terminal", to_move=[], terminal=True,
            utility={"p1": 1.0, "p2": -1.0},
        )
        sim.new_battle.return_value = (1, initial_view)
        step_result = MagicMock()
        step_result.child = 2
        step_result.view = terminal_view
        sim.step.return_value = step_result

        source = CurriculumMatchupSource([{"species": "A"}], [{"species": "B"}])
        config = SelfPlayConfig(num_games=1, master_seed=1, generation=5)

        tuples = list(run(MagicMock(), source, sim, config))

        for t in tuples:
            assert t.meta.generation == 5
            assert t.meta.game_id == 0
            assert t.meta.decision_idx == 0
            assert t.meta.phase == "move"
            assert t.meta.side in ("p1", "p2")

    @patch("self_play.search")
    @patch("self_play.encode")
    @patch("self_play.legal_mask")
    def test_retry_does_not_blacklist_per_side(self, mock_mask, mock_encode, mock_search):
        """Regression: a joint SimError must not blacklist individual per-side actions.

        Before the fix, a failing joint (A, B) would blacklist A for p1 AND B
        for p2 independently.  If p1 had only action A in sigma-bar, p1's action
        was incorrectly exhausted even though A was fine — only the joint was bad.
        The fix resamples fresh from sigma-bar each attempt instead of per-side
        blacklisting, so p1's action A remains available on retry.
        """
        from sim_client import SimError

        mask_multi = np.zeros(10, dtype=bool)
        mask_multi[0] = True
        mask_multi[1] = True
        mock_mask.return_value = mask_multi

        # p1 has ONE action (idx=0), p2 has TWO (idx=0, idx=1).
        # First joint (0, <any>) fails, but p1's action 0 must survive for retry.
        mock_result = MagicMock()
        mock_result.strategy = {"p1": {0: 1.0}, "p2": {0: 0.5, 1: 0.5}}
        mock_result.value = 0.0
        mock_search.return_value = mock_result
        mock_encode.return_value = MagicMock()

        sim = MagicMock()
        initial_view = _make_view(
            phase="move", to_move=["p1", "p2"],
            legal={"p1": {}, "p2": {}}, terminal=False,
        )
        terminal_view = _make_view(
            phase="terminal", to_move=[], terminal=True,
            utility={"p1": 1.0, "p2": -1.0},
        )
        sim.new_battle.return_value = (1, initial_view)

        # First step raises SimError; subsequent steps succeed (terminal).
        step_ok = MagicMock()
        step_ok.child = 2
        step_ok.view = terminal_view
        sim.step.side_effect = [SimError("bad joint"), step_ok]

        source = CurriculumMatchupSource([{"species": "A"}], [{"species": "B"}])
        config = SelfPlayConfig(num_games=1, master_seed=1)

        tuples = list(run(MagicMock(), source, sim, config))

        # Game should NOT abort — the retry succeeds after the first failure.
        assert len(tuples) == 2
        assert sim.step.call_count == 2


# ---------------------------------------------------------------------------
# Tests: _is_forced helper
# ---------------------------------------------------------------------------


class TestIsForced:
    """Tests for the inline forced-decision predicate using pre-computed masks."""

    def test_empty_masks_returns_none(self):
        """No masks (empty to_move) → not forced."""
        assert _is_forced({}) is None

    def test_all_single_action(self):
        """Both sides have exactly one legal action → forced."""
        masks = {
            "p1": np.array([False, True, False]),
            "p2": np.array([False, False, True]),
        }
        result = _is_forced(masks)
        assert result == {"p1": 1, "p2": 2}

    def test_one_side_multiple_actions(self):
        """One side has multiple legal actions → not forced."""
        masks = {
            "p1": np.array([True, True, False]),
            "p2": np.array([False, True, False]),
        }
        assert _is_forced(masks) is None

    def test_unilateral_single_action(self):
        """Single side with one legal action → forced."""
        masks = {"p1": np.array([False, False, True])}
        result = _is_forced(masks)
        assert result == {"p1": 2}


# ---------------------------------------------------------------------------
# Tests: empty to_move guard in search()
# ---------------------------------------------------------------------------


class TestSearchEmptyToMoveGuard:
    """search() must raise ValueError when called with empty to_move."""

    def test_raises_on_empty_to_move(self):
        """Empty to_move is not a decision point — should raise immediately."""
        from search import search, SearchConfig

        mock_view = MagicMock()
        mock_view.phase = "move"
        mock_view.to_move = []
        mock_view.legal = {}

        with pytest.raises(ValueError, match="empty to_move"):
            search(mock_view, MagicMock(), MagicMock(), from_handle=99, config=SearchConfig())


# ---------------------------------------------------------------------------
# Gap 4: _advance step+release contract
# ---------------------------------------------------------------------------


class TestAdvance:
    """Test the _advance helper's step+release contract."""

    def test_success_calls_step_then_release(self):
        """On success: step first, then release old handle, return (child, view)."""
        sim = MagicMock()
        child_view = MagicMock()
        sim.step.return_value = MagicMock(child=42, view=child_view)

        result = _advance(sim, handle=10, choices={"p1": "move 1"}, seed=[1, 2, 3, 4])

        assert result == (42, child_view)
        sim.step.assert_called_once_with(10, {"p1": "move 1"}, [1, 2, 3, 4])
        sim.release.assert_called_once_with(10)

    def test_simerror_propagates_before_release(self):
        """On SimError: exception propagates BEFORE release — old handle stays valid."""
        from sim_client import SimError

        sim = MagicMock()
        sim.step.side_effect = SimError("bad choice")

        with pytest.raises(SimError, match="bad choice"):
            _advance(sim, handle=10, choices={"p1": "move 1"}, seed=[1, 2, 3, 4])

        sim.step.assert_called_once()
        sim.release.assert_not_called()

    def test_release_receives_old_handle_not_child(self):
        """release() must be called with the OLD handle, not the child handle."""
        sim = MagicMock()
        sim.step.return_value = MagicMock(child=99, view=MagicMock())

        _advance(sim, handle=5, choices={}, seed=[0, 0, 0, 0])

        sim.release.assert_called_once_with(5)


# ---------------------------------------------------------------------------
# Gap 5: _sample_and_step helper branches
# ---------------------------------------------------------------------------


class TestSampleAndStep:
    """Test the _sample_and_step retry loop."""

    def test_empty_strategy_returns_none(self):
        """If a side's strategy is empty, returns None immediately."""
        sim = MagicMock()
        view = _make_view(to_move=["p1"], legal={"p1": {}})
        strategy = {"p1": {}}
        result = _sample_and_step(sim, 1, view, strategy, random.Random(0), 1.0)
        assert result is None
        sim.step.assert_not_called()

    def test_first_attempt_success(self):
        """Successful first attempt returns (child_handle, child_view)."""
        sim = MagicMock()
        child_view = _make_view(terminal=True)
        sim.step.return_value = MagicMock(child=42, view=child_view)

        view = _make_view(to_move=["p1"], legal={"p1": {}})
        strategy = {"p1": {0: 1.0}}

        result = _sample_and_step(sim, 1, view, strategy, random.Random(0), 1.0)

        assert result is not None
        assert result[0] == 42
        sim.step.assert_called_once()

    def test_all_retries_exhausted_returns_none(self):
        """10 SimErrors in a row -> returns None."""
        from sim_client import SimError

        sim = MagicMock()
        sim.step.side_effect = SimError("bad choice")

        view = _make_view(to_move=["p1"], legal={"p1": {}})
        strategy = {"p1": {0: 1.0}}

        result = _sample_and_step(sim, 1, view, strategy, random.Random(0), 1.0)

        assert result is None
        assert sim.step.call_count == 10

    def test_retry_then_success(self):
        """First 3 attempts fail, 4th succeeds."""
        from sim_client import SimError

        sim = MagicMock()
        child_view = _make_view(terminal=True)
        step_ok = MagicMock(child=99, view=child_view)
        sim.step.side_effect = [SimError("e1"), SimError("e2"), SimError("e3"), step_ok]

        view = _make_view(to_move=["p1"], legal={"p1": {}})
        strategy = {"p1": {0: 1.0}}

        result = _sample_and_step(sim, 1, view, strategy, random.Random(0), 1.0)

        assert result is not None
        assert result[0] == 99
        assert sim.step.call_count == 4


# ---------------------------------------------------------------------------
# Gap 6: run() empty-phase passthrough
# ---------------------------------------------------------------------------


class TestRunEmptyPhase:
    """Test run() handling of non-acting states (empty to_move / phase='none')."""

    @patch("self_play.search")
    @patch("self_play.encode")
    @patch("self_play.legal_mask")
    def test_empty_to_move_advances_without_search(self, mock_mask, mock_encode, mock_search):
        """When to_move is empty (non-terminal), advance with empty choices and continue."""
        mask_multi = np.zeros(10, dtype=bool)
        mask_multi[0] = True
        mask_multi[1] = True
        mock_mask.return_value = mask_multi

        mock_result = MagicMock()
        mock_result.strategy = {"p1": {0: 1.0}, "p2": {0: 1.0}}
        mock_result.value = 0.0
        mock_search.return_value = mock_result
        mock_encode.return_value = MagicMock()

        sim = MagicMock()
        # First view: empty to_move (non-terminal, e.g. between-turn processing)
        empty_phase_view = _make_view(phase="none", to_move=[], terminal=False)
        # After advancing past empty phase: genuine decision
        move_view = _make_view(
            phase="move", to_move=["p1", "p2"],
            legal={"p1": {}, "p2": {}}, terminal=False,
        )
        terminal_view = _make_view(
            phase="terminal", to_move=[], terminal=True,
            utility={"p1": 1.0, "p2": -1.0},
        )

        sim.new_battle.return_value = (1, empty_phase_view)
        # First step advances past empty phase; second step reaches terminal
        step1 = MagicMock(child=2, view=move_view)
        step2 = MagicMock(child=3, view=terminal_view)
        sim.step.side_effect = [step1, step2]

        source = CurriculumMatchupSource([{"species": "A"}], [{"species": "B"}])
        config = SelfPlayConfig(num_games=1, master_seed=1)

        tuples = list(run(MagicMock(), source, sim, config))

        # Should produce tuples from the genuine decision
        assert len(tuples) == 2
        # The first step call should have been with empty choices (empty phase)
        first_call_choices = sim.step.call_args_list[0][0][1]
        assert first_call_choices == {}


# ---------------------------------------------------------------------------
# Gap 7: run() handle release on normal terminal
# ---------------------------------------------------------------------------


class TestRunHandleRelease:
    """Verify handle lifecycle: release called after terminal."""

    @patch("self_play.search")
    @patch("self_play.encode")
    @patch("self_play.legal_mask")
    def test_release_called_on_terminal(self, mock_mask, mock_encode, mock_search):
        """After yielding all tuples, the terminal handle is released."""
        mask_multi = np.zeros(10, dtype=bool)
        mask_multi[0] = True
        mask_multi[1] = True
        mock_mask.return_value = mask_multi

        mock_result = MagicMock()
        mock_result.strategy = {"p1": {0: 1.0}, "p2": {0: 1.0}}
        mock_result.value = 0.0
        mock_search.return_value = mock_result
        mock_encode.return_value = MagicMock()

        sim = MagicMock()
        initial_view = _make_view(
            phase="move", to_move=["p1", "p2"],
            legal={"p1": {}, "p2": {}}, terminal=False,
        )
        terminal_view = _make_view(
            phase="terminal", to_move=[], terminal=True,
            utility={"p1": 1.0, "p2": -1.0},
        )
        sim.new_battle.return_value = (1, initial_view)
        step_result = MagicMock()
        step_result.child = 2
        step_result.view = terminal_view
        sim.step.return_value = step_result

        source = CurriculumMatchupSource([{"species": "A"}], [{"species": "B"}])
        config = SelfPlayConfig(num_games=1, master_seed=1)

        tuples = list(run(MagicMock(), source, sim, config))

        assert len(tuples) == 2
        # release should be called for old handle (via _advance) AND for terminal handle
        release_calls = [call[0][0] for call in sim.release.call_args_list]
        assert 1 in release_calls, "Old handle (1) should be released by _advance"
        assert 2 in release_calls, "Terminal handle (2) should be released after yield"


# ---------------------------------------------------------------------------
# Gap 16: run() with config=None default
# ---------------------------------------------------------------------------


class TestRunConfigDefault:
    """run() with config=None should use SelfPlayConfig defaults."""

    @patch("self_play.search")
    @patch("self_play.encode")
    @patch("self_play.legal_mask")
    def test_config_none_uses_defaults(self, mock_mask, mock_encode, mock_search):
        """Passing config=None should not crash and use default SelfPlayConfig."""
        mask_multi = np.zeros(10, dtype=bool)
        mask_multi[0] = True
        mask_multi[1] = True
        mock_mask.return_value = mask_multi

        mock_result = MagicMock()
        mock_result.strategy = {"p1": {0: 1.0}, "p2": {0: 1.0}}
        mock_result.value = 0.0
        mock_search.return_value = mock_result
        mock_encode.return_value = MagicMock()

        sim = MagicMock()
        terminal_view = _make_view(
            phase="terminal", to_move=[], terminal=True,
            utility={"p1": 0.0, "p2": 0.0},
        )
        sim.new_battle.return_value = (1, _make_view(
            phase="move", to_move=["p1", "p2"],
            legal={"p1": {}, "p2": {}}, terminal=False,
        ))
        sim.step.return_value = MagicMock(child=2, view=terminal_view)

        source = CurriculumMatchupSource([{"species": "A"}], [{"species": "B"}])

        # config=None explicitly — should default to SelfPlayConfig() with num_games=None (infinite)
        # We need to break out after one game since num_games=None = infinite
        gen = run(MagicMock(), source, sim, config=None)
        first_batch = []
        for t in gen:
            first_batch.append(t)
            if len(first_batch) >= 2:
                break

        assert len(first_batch) == 2


# ---------------------------------------------------------------------------
# Gap 17: _sample_action adversarial inputs
# ---------------------------------------------------------------------------


class TestSampleActionAdversarial:
    """Adversarial and boundary inputs for _sample_action."""

    def test_all_zero_probs_returns_valid_action(self):
        """All-zero probs should fall back to uniform (degenerate case)."""
        strategy = {0: 0.0, 1: 0.0, 2: 0.0}
        rng = random.Random(42)
        idx = _sample_action(strategy, rng, temperature=1.0)
        assert idx in strategy

    def test_near_zero_probs_returns_valid_action(self):
        """Very small probs (near-zero) should still produce valid samples."""
        strategy = {0: 1e-15, 1: 1e-15, 2: 1e-15}
        rng = random.Random(42)
        idx = _sample_action(strategy, rng, temperature=1.0)
        assert idx in strategy

    def test_equal_probs_all_sampled(self):
        """Equal probs should sample all actions over many trials."""
        strategy = {0: 0.5, 1: 0.5}
        rng = random.Random(42)
        seen = set()
        for _ in range(100):
            seen.add(_sample_action(strategy, rng, temperature=1.0))
        assert seen == {0, 1}

    def test_many_actions(self):
        """100 actions with uniform probs should not crash."""
        strategy = {i: 0.01 for i in range(100)}
        rng = random.Random(42)
        idx = _sample_action(strategy, rng, temperature=1.0)
        assert 0 <= idx < 100

    def test_single_zero_prob_with_temperature(self):
        """Single action at prob 0.0 with temperature scaling."""
        strategy = {0: 0.0}
        rng = random.Random(42)
        idx = _sample_action(strategy, rng, temperature=1.0)
        assert idx == 0


# ---------------------------------------------------------------------------
# Gap 18: _rng_seed output shape
# ---------------------------------------------------------------------------


class TestRngSeed:
    """Test _rng_seed returns correct format."""

    def test_returns_4_ints(self):
        rng = random.Random(42)
        seed = _rng_seed(rng)
        assert isinstance(seed, list)
        assert len(seed) == 4
        for s in seed:
            assert isinstance(s, int)

    def test_values_in_range(self):
        rng = random.Random(42)
        seed = _rng_seed(rng)
        for s in seed:
            assert 0 <= s <= 0xFFFF

    def test_deterministic(self):
        """Same RNG state produces same seeds."""
        seed_a = _rng_seed(random.Random(99))
        seed_b = _rng_seed(random.Random(99))
        assert seed_a == seed_b


# ---------------------------------------------------------------------------
# Gap 19: z defaults to 0.0 when utility is None or missing
# ---------------------------------------------------------------------------


class TestZDefaultOnMissingUtility:
    """Test z stamping edge cases when utility is None or missing keys."""

    @patch("self_play.search")
    @patch("self_play.encode")
    @patch("self_play.legal_mask")
    def test_none_utility_gives_zero_z(self, mock_mask, mock_encode, mock_search):
        """When view.utility is None, z should default to 0.0."""
        mask_multi = np.zeros(10, dtype=bool)
        mask_multi[0] = True
        mask_multi[1] = True
        mock_mask.return_value = mask_multi

        mock_result = MagicMock()
        mock_result.strategy = {"p1": {0: 1.0}, "p2": {0: 1.0}}
        mock_result.value = 0.0
        mock_search.return_value = mock_result
        mock_encode.return_value = MagicMock()

        sim = MagicMock()
        sim.new_battle.return_value = (1, _make_view(
            phase="move", to_move=["p1", "p2"],
            legal={"p1": {}, "p2": {}}, terminal=False,
        ))
        # Terminal view with utility=None
        terminal_view = _make_view(
            phase="terminal", to_move=[], terminal=True,
            utility=None,
        )
        sim.step.return_value = MagicMock(child=2, view=terminal_view)

        source = CurriculumMatchupSource([{"species": "A"}], [{"species": "B"}])
        config = SelfPlayConfig(num_games=1, master_seed=1)

        tuples = list(run(MagicMock(), source, sim, config))

        for t in tuples:
            assert t.z == 0.0

    @patch("self_play.search")
    @patch("self_play.encode")
    @patch("self_play.legal_mask")
    def test_missing_side_key_gives_zero_z(self, mock_mask, mock_encode, mock_search):
        """When utility dict is missing a side's key, z defaults to 0.0."""
        mask_multi = np.zeros(10, dtype=bool)
        mask_multi[0] = True
        mask_multi[1] = True
        mock_mask.return_value = mask_multi

        mock_result = MagicMock()
        mock_result.strategy = {"p1": {0: 1.0}, "p2": {0: 1.0}}
        mock_result.value = 0.0
        mock_search.return_value = mock_result
        mock_encode.return_value = MagicMock()

        sim = MagicMock()
        sim.new_battle.return_value = (1, _make_view(
            phase="move", to_move=["p1", "p2"],
            legal={"p1": {}, "p2": {}}, terminal=False,
        ))
        # Terminal view with only p1 in utility (p2 missing)
        terminal_view = _make_view(
            phase="terminal", to_move=[], terminal=True,
            utility={"p1": 1.0},
        )
        sim.step.return_value = MagicMock(child=2, view=terminal_view)

        source = CurriculumMatchupSource([{"species": "A"}], [{"species": "B"}])
        config = SelfPlayConfig(num_games=1, master_seed=1)

        tuples = list(run(MagicMock(), source, sim, config))

        p1_tuple = next(t for t in tuples if t.meta.side == "p1")
        p2_tuple = next(t for t in tuples if t.meta.side == "p2")
        assert p1_tuple.z == 1.0
        assert p2_tuple.z == 0.0


# ---------------------------------------------------------------------------
# Gap 20: TupleMeta and TrainingTuple frozen immutability
# ---------------------------------------------------------------------------


class TestFrozenImmutability:
    """Frozen dataclass fields should reject mutation."""

    def test_tuple_meta_frozen(self):
        meta = TupleMeta(generation=0, game_id=0, decision_idx=0, phase="move", side="p1")
        with pytest.raises(Exception):
            meta.generation = 1  # type: ignore

    def test_training_tuple_frozen(self):
        meta = TupleMeta(generation=0, game_id=0, decision_idx=0, phase="move", side="p1")
        policy = SparsePolicy(indices=(0,), probs=(1.0,))
        tt = TrainingTuple(beta=MagicMock(), value=0.5, policy=policy, z=1.0, meta=meta)
        with pytest.raises(Exception):
            tt.value = 0.0  # type: ignore
        with pytest.raises(Exception):
            tt.z = -1.0  # type: ignore
