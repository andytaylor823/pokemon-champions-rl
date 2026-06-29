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

from self_play import (
    CurriculumMatchupSource,
    SelfPlayConfig,
    SparsePolicy,
    TrainingTuple,
    TupleMeta,
    _PendingTuple,
    _forced_choices,
    _is_forced,
    _sample_action,
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
# Tests: _is_forced
# ---------------------------------------------------------------------------


class TestIsForced:
    """Tests for forced-decision detection."""

    def test_empty_to_move_is_not_forced(self):
        """Empty to_move means no one acts — not a forced decision."""
        view = _make_view(to_move=[])
        assert _is_forced(view) is False

    @patch("self_play.legal_mask")
    def test_single_legal_action_both_sides_is_forced(self, mock_mask):
        """Both sides have exactly one legal action -> forced."""
        mock_mask.return_value = np.array([False, True, False, False, False])
        view = _make_view(to_move=["p1", "p2"], legal={"p1": {}, "p2": {}})
        assert _is_forced(view) is True
        assert mock_mask.call_count == 2

    @patch("self_play.legal_mask")
    def test_multiple_legal_actions_not_forced(self, mock_mask):
        """One side has multiple legal actions -> not forced."""
        mock_mask.return_value = np.array([True, True, False, False, False])
        view = _make_view(to_move=["p1", "p2"], legal={"p1": {}, "p2": {}})
        assert _is_forced(view) is False

    @patch("self_play.legal_mask")
    def test_unilateral_single_action_is_forced(self, mock_mask):
        """One side acting with one legal action -> forced."""
        mock_mask.return_value = np.array([False, False, True, False, False])
        view = _make_view(to_move=["p1"], legal={"p1": {}})
        assert _is_forced(view) is True

    @patch("self_play.legal_mask")
    def test_unilateral_multiple_actions_not_forced(self, mock_mask):
        """One side acting with multiple legal actions -> not forced."""
        mock_mask.return_value = np.array([True, False, True, False, False])
        view = _make_view(to_move=["p1"], legal={"p1": {}})
        assert _is_forced(view) is False


# ---------------------------------------------------------------------------
# Tests: _forced_choices
# ---------------------------------------------------------------------------


class TestForcedChoices:
    """Tests for building choice strings from forced decisions."""

    @patch("self_play.action_to_choice_contextual")
    @patch("self_play.legal_mask")
    def test_returns_choice_for_each_side(self, mock_mask, mock_ctx_choice):
        """Should return one choice string per acting side."""
        # Mask with single legal action at index 2
        mock_mask.return_value = np.array([False, False, True, False])
        mock_ctx_choice.return_value = "team 1234"
        view = _make_view(to_move=["p1", "p2"], legal={"p1": {}, "p2": {}})

        choices = _forced_choices(view)
        assert choices == {"p1": "team 1234", "p2": "team 1234"}
        assert mock_ctx_choice.call_count == 2
        # Both calls should pass index 2 and the legal request
        mock_ctx_choice.assert_any_call(2, {})


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
        call_count = [0]

        def mask_side_effect(request, phase):
            call_count[0] += 1
            if call_count[0] <= 2:
                # First state: forced (single legal for both sides)
                return np.array([False, True, False])
            else:
                # After step: also forced but with different index
                return np.array([False, True, False])

        mock_mask.side_effect = mask_side_effect

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
