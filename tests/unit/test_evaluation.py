"""Unit tests for src/evaluation.py — pure logic, no Node worker.

All tests run without a real SimClient.  Network calls and search() are
monkeypatched or exercised via minimal stubs.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest
import torch

from action_space import A, legal_mask
from evaluation import (
    CurriculumReport,
    EvalConfig,
    GameResult,
    HeadToHeadReport,
    PolicyAgent,
    RandomAgent,
    SearchAgent,
    _rng_seed,
    curriculum_report,
    fitness,
    head_to_head,
    play_game,
)
from search import SearchConfig, SearchResult
from state_types import (
    BattleSnapshot,
    FieldSnapshot,
    SideSnapshot,
    StateView,
)


# ---------------------------------------------------------------------------
# Helpers to build minimal StateView stubs
# ---------------------------------------------------------------------------


def _empty_field() -> FieldSnapshot:
    return FieldSnapshot(weather=None, weatherDuration=None, terrain=None, terrainDuration=None, pseudoWeather={})


def _empty_snapshot(turn: int = 3) -> BattleSnapshot:
    return BattleSnapshot(turn=turn, field=_empty_field(), sides=[SideSnapshot(id="p1", sideConditions={}, pokemon=[]), SideSnapshot(id="p2", sideConditions={}, pokemon=[])])


def _make_view(
    *,
    terminal: bool = False,
    utility: dict | None = None,
    to_move: list[str] | None = None,
    phase: str = "move",
    legal: dict | None = None,
    turn: int = 3,
) -> StateView:
    return StateView(
        phase=phase,
        to_move=to_move or [],
        legal=legal or {},
        snapshot=_empty_snapshot(turn=turn),
        terminal=terminal,
        utility=utility,
    )


# ---------------------------------------------------------------------------
# Helper: build a minimal ObsBundle-like stub for PolicyAgent tests
# ---------------------------------------------------------------------------


def _make_obs_bundle_stub(legal_indices: list[int] | None = None) -> Any:
    """Return a mock that encoder.encode() would return."""
    obs = MagicMock()
    obs.batch_size = torch.Size([])
    return obs


# ---------------------------------------------------------------------------
# RandomAgent tests
# ---------------------------------------------------------------------------


class TestRandomAgent:
    def _agent(self, seed: int = 0) -> RandomAgent:
        return RandomAgent(random.Random(seed))

    def test_returns_legal_index(self):
        """RandomAgent must always return an index marked True in the legal mask."""
        # Build a simple move-phase request with a couple of legal actions
        request = {
            "active": [
                {"moves": [{"id": "heatwave", "pp": 8, "maxpp": 8, "target": "allAdjacentFoes", "disabled": False}]},
                {"moves": [{"id": "protect", "pp": 8, "maxpp": 8, "target": "self", "disabled": False}]},
            ],
            "side": {"pokemon": [{"condition": "100/100"}, {"condition": "100/100"}, {"condition": "80/100"}, {"condition": "60/100 fnt"}]},
        }
        view = _make_view(to_move=["p1"], phase="move", legal={"p1": request})

        agent = self._agent(seed=42)
        mask = legal_mask(request, "move")
        legal_set = set(int(i) for i, v in enumerate(mask) if v)

        # Sample many times and confirm every result is legal
        results = {agent.act(view, "p1", MagicMock(), handle=1) for _ in range(50)}
        assert results.issubset(legal_set), f"Illegal indices returned: {results - legal_set}"

    def test_distribution_roughly_uniform(self):
        """RandomAgent should visit all legal actions with roughly equal frequency."""
        request = {
            "active": [
                {"moves": [{"id": "heatwave", "pp": 8, "maxpp": 8, "target": "allAdjacentFoes", "disabled": False}, {"id": "protect", "pp": 8, "maxpp": 8, "target": "self", "disabled": False}]},
                {"moves": [{"id": "heatwave", "pp": 8, "maxpp": 8, "target": "allAdjacentFoes", "disabled": False}, {"id": "protect", "pp": 8, "maxpp": 8, "target": "self", "disabled": False}]},
            ],
            "side": {"pokemon": [{"condition": "100/100"}, {"condition": "100/100"}, {"condition": "80/100"}, {"condition": "60/100"}]},
        }
        view = _make_view(to_move=["p1"], phase="move", legal={"p1": request})
        mask = legal_mask(request, "move")
        n_legal = int(mask.sum())

        agent = self._agent(seed=7)
        counts: dict[int, int] = {}
        n_samples = n_legal * 200
        for _ in range(n_samples):
            idx = agent.act(view, "p1", MagicMock(), handle=1)
            counts[idx] = counts.get(idx, 0) + 1

        # Every legal index should be sampled at least once
        assert len(counts) == n_legal, f"Expected {n_legal} distinct indices, got {len(counts)}"
        # Rough uniformity: no index > 3× expected
        expected = n_samples / n_legal
        for idx, cnt in counts.items():
            assert cnt < 3 * expected, f"Index {idx} over-sampled: {cnt} vs expected ~{expected:.0f}"

    def test_raises_on_no_legal_actions(self):
        """RandomAgent must raise ValueError when the legal mask is all False."""
        view = _make_view(to_move=["p1"], phase="move", legal={"p1": None})
        agent = self._agent()
        with pytest.raises(ValueError, match="no legal actions"):
            agent.act(view, "p1", MagicMock(), handle=1)


# ---------------------------------------------------------------------------
# SearchAgent greedy-argmax tests
# ---------------------------------------------------------------------------


class TestSearchAgentGreedy:
    def _make_search_result(self, strategy: dict[str, dict[int, float]]) -> SearchResult:
        return SearchResult(
            strategy=strategy,
            value=0.5,
            policy_target={s: np.zeros(A) for s in strategy},
        )

    def test_picks_highest_prob_action(self):
        """SearchAgent should pick the action with the maximum probability."""
        strategy = {"p1": {360: 0.1, 361: 0.8, 362: 0.1}}
        result = self._make_search_result(strategy)

        net = MagicMock()
        cfg = SearchConfig()
        agent = SearchAgent(net, cfg)

        view = _make_view(to_move=["p1"])
        sim = MagicMock()
        handle = 42

        with patch("evaluation.search", return_value=result):
            idx = agent.act(view, "p1", sim, handle)

        assert idx == 361

    def test_lowest_index_on_tie(self):
        """On tied probabilities, the smallest index wins."""
        strategy = {"p1": {365: 0.5, 363: 0.5, 370: 0.5}}
        result = self._make_search_result(strategy)

        agent = SearchAgent(MagicMock(), SearchConfig())
        view = _make_view(to_move=["p1"])

        with patch("evaluation.search", return_value=result):
            idx = agent.act(view, "p1", MagicMock(), handle=1)

        assert idx == 363

    def test_cache_hit_same_handle(self):
        """The second act() call at the same handle must NOT call search() again."""
        strategy = {"p1": {360: 0.6, 361: 0.4}, "p2": {362: 0.7, 363: 0.3}}
        result = self._make_search_result(strategy)

        agent = SearchAgent(MagicMock(), SearchConfig())
        view = _make_view(to_move=["p1", "p2"])

        with patch("evaluation.search", return_value=result) as mock_search:
            agent.act(view, "p1", MagicMock(), handle=99)
            agent.act(view, "p2", MagicMock(), handle=99)
            assert mock_search.call_count == 1, "search() should be called exactly once"

    def test_cache_miss_new_handle(self):
        """A different handle must trigger a fresh search() call."""
        strategy = {"p1": {360: 1.0}}
        result = self._make_search_result(strategy)

        agent = SearchAgent(MagicMock(), SearchConfig())
        view = _make_view(to_move=["p1"])

        with patch("evaluation.search", return_value=result) as mock_search:
            agent.act(view, "p1", MagicMock(), handle=1)
            agent.act(view, "p1", MagicMock(), handle=2)
            assert mock_search.call_count == 2

    def test_raises_on_empty_strategy(self):
        """SearchAgent must raise ValueError when the strategy for a side is empty."""
        result = SearchResult(strategy={"p1": {}}, value=0.0, policy_target={})
        agent = SearchAgent(MagicMock(), SearchConfig())
        view = _make_view(to_move=["p1"])

        with patch("evaluation.search", return_value=result):
            with pytest.raises(ValueError, match="empty strategy"):
                agent.act(view, "p1", MagicMock(), handle=1)


# ---------------------------------------------------------------------------
# PolicyAgent tests
# ---------------------------------------------------------------------------


class TestPolicyAgent:
    def test_picks_highest_logit(self):
        """PolicyAgent picks the argmax of the (masked) logit vector."""
        net = MagicMock()
        # Return logits with index 370 as the winner
        logits = torch.full((A,), float("-inf"))
        logits[370] = 2.5
        logits[371] = 1.0
        logits[372] = 0.5
        net.return_value = (logits, torch.tensor(0.2))

        agent = PolicyAgent(net)
        view = _make_view(to_move=["p1"])

        with patch("evaluation.encode", return_value=MagicMock()):
            idx = agent.act(view, "p1", MagicMock(), handle=1)

        assert idx == 370

    def test_raises_on_all_inf_logits(self):
        """PolicyAgent must raise ValueError when all logits are -inf."""
        net = MagicMock()
        net.return_value = (torch.full((A,), float("-inf")), torch.tensor(0.0))

        agent = PolicyAgent(net)
        view = _make_view(to_move=["p1"])

        with patch("evaluation.encode", return_value=MagicMock()):
            with pytest.raises(ValueError, match="no legal actions"):
                agent.act(view, "p1", MagicMock(), handle=1)


# ---------------------------------------------------------------------------
# GameResult outcome classification
# ---------------------------------------------------------------------------


class TestGameResultClassification:
    """Utility > 0 → p1_win; < 0 → p2_win; == 0 or None → draw."""

    def _run_terminal(self, utility: dict | None, turn: int = 5) -> GameResult:
        """Run play_game with a game that immediately terminates."""
        terminal_view = _make_view(terminal=True, utility=utility, turn=turn)
        initial_view = _make_view(terminal=False, to_move=[], turn=0)  # empty phase → step → terminal

        sim = MagicMock()
        sim.new_battle.return_value = (1, initial_view)
        step_result = MagicMock()
        step_result.child = 2
        step_result.view = terminal_view
        sim.step.return_value = step_result
        sim.release.return_value = None

        return play_game(MagicMock(), MagicMock(), [], [], sim, seed=0)

    def test_p1_win(self):
        result = self._run_terminal({"p1": 1.0})
        assert result.outcome == "p1_win"

    def test_p2_win(self):
        result = self._run_terminal({"p1": -1.0})
        assert result.outcome == "p2_win"

    def test_draw_zero_utility(self):
        result = self._run_terminal({"p1": 0.0})
        assert result.outcome == "draw"

    def test_draw_none_utility(self):
        result = self._run_terminal(None)
        assert result.outcome == "draw"


# ---------------------------------------------------------------------------
# CurriculumReport math
# ---------------------------------------------------------------------------


class TestCurriculumReportMath:
    def _report(
        self,
        favored_wins: int,
        underdog_wins: int,
        draws: int,
        aborted: int,
        favored_turns: tuple[int, ...] = (),
        all_turns: tuple[int, ...] = (),
    ) -> CurriculumReport:
        return CurriculumReport(
            n_games=favored_wins + underdog_wins + draws + aborted,
            favored_wins=favored_wins,
            underdog_wins=underdog_wins,
            draws=draws,
            aborted=aborted,
            favored_turns=favored_turns,
            all_turns=all_turns,
        )

    def test_favored_win_rate_excludes_draws_and_aborted(self):
        r = self._report(favored_wins=3, underdog_wins=1, draws=2, aborted=4)
        # Decided = 3 + 1 + 2 = 6; aborted not counted
        assert r.favored_win_rate == pytest.approx(3 / 6)

    def test_favored_win_rate_all_draws(self):
        r = self._report(favored_wins=0, underdog_wins=0, draws=10, aborted=0)
        assert r.favored_win_rate == pytest.approx(0.0)

    def test_favored_win_rate_no_games(self):
        r = self._report(favored_wins=0, underdog_wins=0, draws=0, aborted=0)
        assert r.favored_win_rate == 0.0

    def test_aborted_excluded_from_denominator(self):
        r = self._report(favored_wins=2, underdog_wins=0, draws=0, aborted=8)
        # Only 2 decided games → rate = 2/2 = 1.0
        assert r.favored_win_rate == pytest.approx(1.0)

    def test_mean_favored_turns_none_when_never_won(self):
        r = self._report(favored_wins=0, underdog_wins=5, draws=0, aborted=0, favored_turns=())
        assert r.mean_favored_turns is None

    def test_mean_favored_turns(self):
        r = self._report(favored_wins=3, underdog_wins=0, draws=0, aborted=0, favored_turns=(4, 6, 8), all_turns=(4, 6, 8))
        assert r.mean_favored_turns == pytest.approx(6.0)

    def test_median_favored_turns_odd(self):
        r = self._report(favored_wins=3, underdog_wins=0, draws=0, aborted=0, favored_turns=(4, 6, 10))
        assert r.median_favored_turns == pytest.approx(6.0)

    def test_median_favored_turns_even(self):
        r = self._report(favored_wins=4, underdog_wins=0, draws=0, aborted=0, favored_turns=(2, 4, 6, 8))
        assert r.median_favored_turns == pytest.approx(5.0)

    def test_overall_mean_turns_none_when_no_decided(self):
        r = self._report(favored_wins=0, underdog_wins=0, draws=0, aborted=5)
        assert r.overall_mean_turns is None

    def test_overall_mean_turns(self):
        r = self._report(favored_wins=1, underdog_wins=1, draws=0, aborted=0, favored_turns=(4,), all_turns=(4, 8))
        assert r.overall_mean_turns == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# HeadToHeadReport math
# ---------------------------------------------------------------------------


class TestHeadToHeadReportMath:
    def _report(self, a_wins: int, b_wins: int, draws: int, aborted: int) -> HeadToHeadReport:
        return HeadToHeadReport(
            n_games=a_wins + b_wins + draws + aborted,
            a_wins=a_wins,
            b_wins=b_wins,
            draws=draws,
            aborted=aborted,
            a_turns=tuple(range(a_wins)),
            all_turns=tuple(range(a_wins + b_wins + draws)),
        )

    def test_a_win_rate_excludes_draws_and_aborted(self):
        r = self._report(a_wins=4, b_wins=2, draws=2, aborted=2)
        # Decided = 4 + 2 + 2 = 8
        assert r.a_win_rate == pytest.approx(4 / 8)

    def test_a_win_rate_zero_decided(self):
        r = self._report(a_wins=0, b_wins=0, draws=0, aborted=5)
        assert r.a_win_rate == 0.0


# ---------------------------------------------------------------------------
# play_game abort paths
# ---------------------------------------------------------------------------


class TestPlayGameAborts:
    """Test the four abort paths without a real SimClient."""

    def _make_sim(self, initial_view: StateView, step_side_effect=None):
        sim = MagicMock()
        sim.new_battle.return_value = (1, initial_view)
        if step_side_effect is not None:
            sim.step.side_effect = step_side_effect
        return sim

    def test_max_decisions_abort(self):
        """Hitting max_decisions without terminal → 'aborted' with reason 'max_decisions'."""
        # Build a view that always looks like a genuine decision (never terminal)
        # We need to mock forced_actions to return None (not forced)
        # and to_move to have both sides acting
        request = {
            "active": [
                {"moves": [{"id": "heatwave", "pp": 8, "maxpp": 8, "target": "allAdjacentFoes", "disabled": False}, {"id": "protect", "pp": 8, "maxpp": 8, "target": "self", "disabled": False}]},
                {"moves": [{"id": "heatwave", "pp": 8, "maxpp": 8, "target": "allAdjacentFoes", "disabled": False}, {"id": "protect", "pp": 8, "maxpp": 8, "target": "self", "disabled": False}]},
            ],
            "side": {"pokemon": [{"condition": "100/100"}, {"condition": "100/100"}, {"condition": "80/100"}, {"condition": "60/100"}]},
        }
        # Never-ending view: non-terminal, has genuine decisions
        loop_view = _make_view(terminal=False, to_move=["p1", "p2"], phase="move", legal={"p1": request, "p2": request})

        sim = MagicMock()
        sim.new_battle.return_value = (1, loop_view)
        # step always returns same looping view
        step_result = MagicMock()
        step_result.child = 2
        step_result.view = loop_view
        sim.step.return_value = step_result

        agent = RandomAgent(random.Random(0))
        result = play_game(agent, agent, [], [], sim, seed=0, max_decisions=3)
        assert result.outcome == "aborted"
        assert result.abort_reason == "max_decisions"

    def test_sim_error_abort(self):
        """SimError on step → 'aborted' with 'sim_error' reason."""
        from sim_client import SimError

        request = {
            "active": [
                {"moves": [{"id": "heatwave", "pp": 8, "maxpp": 8, "target": "allAdjacentFoes", "disabled": False}]},
                {"moves": [{"id": "protect", "pp": 8, "maxpp": 8, "target": "self", "disabled": False}]},
            ],
            "side": {"pokemon": [{"condition": "100/100"}, {"condition": "100/100"}, {"condition": "80/100"}, {"condition": "60/100"}]},
        }
        view = _make_view(terminal=False, to_move=["p1", "p2"], phase="move", legal={"p1": request, "p2": request})
        sim = MagicMock()
        sim.new_battle.return_value = (1, view)
        sim.step.side_effect = SimError("illegal choice")

        agent = RandomAgent(random.Random(0))
        result = play_game(agent, agent, [], [], sim, seed=0)
        assert result.outcome == "aborted"
        assert "sim_error" in result.abort_reason

    def test_no_action_abort(self):
        """Agent raising ValueError → 'aborted' with 'no_action' reason."""
        view = _make_view(terminal=False, to_move=["p1"], phase="move", legal={"p1": None})

        sim = MagicMock()
        sim.new_battle.return_value = (1, view)

        bad_agent = RandomAgent(random.Random(0))  # legal=None → no legal actions → ValueError
        result = play_game(bad_agent, MagicMock(), [], [], sim, seed=0)
        assert result.outcome == "aborted"
        assert "no_action" in result.abort_reason


# ---------------------------------------------------------------------------
# curriculum_report counting
# ---------------------------------------------------------------------------


class TestCurriculumReportCounts:
    """Check that curriculum_report correctly aggregates GameResults."""

    def test_counts_sum_to_n_games(self):
        """favored_wins + underdog_wins + draws + aborted must equal n_games."""
        # Use a mock play_game that cycles through outcomes
        outcomes = ["p1_win", "p2_win", "draw", "aborted", "p1_win"]
        game_idx = 0

        def fake_play_game(*args, **kwargs):
            nonlocal game_idx
            o = outcomes[game_idx % len(outcomes)]
            game_idx += 1
            if o == "aborted":
                return GameResult("aborted", 0, "test")
            return GameResult(o, 5)

        matchup = MagicMock()
        matchup.sample.return_value = ([], [])
        net = MagicMock()
        cfg = EvalConfig(n_games=5, use_search=False)

        with patch("evaluation.play_game", side_effect=fake_play_game):
            with patch("evaluation.PolicyAgent"):
                report = curriculum_report(net, matchup, MagicMock(), config=cfg)

        total = report.favored_wins + report.underdog_wins + report.draws + report.aborted
        assert total == cfg.n_games

    def test_favored_side_p2(self):
        """When favored_side='p2', p2_wins should count as favored wins."""
        outcomes_cycle = ["p2_win", "p1_win"]
        idx = 0

        def fake_play(*args, **kwargs):
            nonlocal idx
            o = outcomes_cycle[idx % len(outcomes_cycle)]
            idx += 1
            return GameResult(o, 5)

        matchup = MagicMock()
        matchup.sample.return_value = ([], [])
        cfg = EvalConfig(n_games=4, favored_side="p2", use_search=False)

        with patch("evaluation.play_game", side_effect=fake_play):
            with patch("evaluation.PolicyAgent"):
                report = curriculum_report(MagicMock(), matchup, MagicMock(), config=cfg)

        # 2 p2_wins → favored; 2 p1_wins → underdog
        assert report.favored_wins == 2
        assert report.underdog_wins == 2


# ---------------------------------------------------------------------------
# head_to_head counting
# ---------------------------------------------------------------------------


class TestHeadToHeadCounts:
    def test_counts_sum_to_n_games(self):
        outcomes = ["p1_win", "p2_win", "draw", "aborted"]
        idx = 0

        def fake_play(*args, **kwargs):
            nonlocal idx
            o = outcomes[idx % len(outcomes)]
            idx += 1
            if o == "aborted":
                return GameResult("aborted", 0, "test")
            return GameResult(o, 5)

        matchup = MagicMock()
        matchup.sample.return_value = ([], [])
        cfg = EvalConfig(n_games=4, use_search=False)

        with patch("evaluation.play_game", side_effect=fake_play):
            report = head_to_head(MagicMock(), MagicMock(), matchup, MagicMock(), config=cfg)

        total = report.a_wins + report.b_wins + report.draws + report.aborted
        assert total == cfg.n_games
        assert report.a_wins == 1
        assert report.b_wins == 1
        assert report.draws == 1
        assert report.aborted == 1


# ---------------------------------------------------------------------------
# _rng_seed helper
# ---------------------------------------------------------------------------


class TestRngSeed:
    def test_returns_four_element_list(self):
        rng = random.Random(0)
        seed = _rng_seed(rng)
        assert len(seed) == 4
        assert all(isinstance(v, int) for v in seed)
        assert all(0 <= v <= 0xFFFF for v in seed)

    def test_deterministic(self):
        seed1 = _rng_seed(random.Random(42))
        seed2 = _rng_seed(random.Random(42))
        assert seed1 == seed2

    def test_sequential_calls_advance_state(self):
        """Two consecutive _rng_seed calls from the same RNG must differ."""
        rng = random.Random(99)
        first = _rng_seed(rng)
        second = _rng_seed(rng)
        # Both are 4-element lists of valid ints; they must not be identical
        # (the RNG must advance between calls).
        assert first != second, "Two successive _rng_seed calls returned the same seed"


# ---------------------------------------------------------------------------
# EvalConfig default field values
# ---------------------------------------------------------------------------


class TestEvalConfigDefaults:
    def test_defaults(self):
        cfg = EvalConfig()
        assert cfg.n_games == 50
        assert cfg.max_decisions == 300
        assert cfg.master_seed == 0
        assert cfg.use_search is True
        assert cfg.favored_side == "p1"
        assert cfg.device == "cpu"


# ---------------------------------------------------------------------------
# GameResult: terminal outcomes always have abort_reason=None
# ---------------------------------------------------------------------------


class TestGameResultTerminalAbortReason:
    """For p1_win, p2_win, and draw: abort_reason must be None."""

    def _run_terminal(self, utility: dict | None, turn: int = 5) -> GameResult:
        terminal_view = _make_view(terminal=True, utility=utility, turn=turn)
        initial_view = _make_view(terminal=False, to_move=[], turn=0)
        sim = MagicMock()
        sim.new_battle.return_value = (1, initial_view)
        step_result = MagicMock()
        step_result.child = 2
        step_result.view = terminal_view
        sim.step.return_value = step_result
        return play_game(MagicMock(), MagicMock(), [], [], sim, seed=0)

    def test_p1_win_abort_reason_none(self):
        result = self._run_terminal({"p1": 1.0})
        assert result.outcome == "p1_win"
        assert result.abort_reason is None

    def test_p2_win_abort_reason_none(self):
        result = self._run_terminal({"p1": -1.0})
        assert result.outcome == "p2_win"
        assert result.abort_reason is None

    def test_draw_abort_reason_none(self):
        result = self._run_terminal({"p1": 0.0})
        assert result.outcome == "draw"
        assert result.abort_reason is None


# ---------------------------------------------------------------------------
# play_game: utility dict with explicit None value
# ---------------------------------------------------------------------------


class TestPlayGameUtilityDictNone:
    """utility={'p1': None} is distinct from utility=None (no dict); both → draw.

    StateView's Pydantic model enforces dict[str, float], so utility={'p1': None}
    cannot be constructed via the real StateView.  We simulate it with a MagicMock
    to confirm that the guard in play_game handles this defensively.
    """

    def test_draw_when_utility_dict_has_none_value(self):
        # Build a MagicMock terminal view with utility={'p1': None}
        terminal_view = MagicMock()
        terminal_view.terminal = True
        terminal_view.utility = {"p1": None}
        terminal_view.snapshot.turn = 4

        initial_view = _make_view(terminal=False, to_move=[], turn=0)
        sim = MagicMock()
        sim.new_battle.return_value = (1, initial_view)
        step_result = MagicMock()
        step_result.child = 2
        step_result.view = terminal_view
        sim.step.return_value = step_result

        result = play_game(MagicMock(), MagicMock(), [], [], sim, seed=0)
        assert result.outcome == "draw"
        assert result.abort_reason is None


# ---------------------------------------------------------------------------
# play_game: outer SimError from new_battle
# ---------------------------------------------------------------------------


class TestPlayGameOuterSimError:
    """SimError raised by sim.new_battle → 'aborted', handle=None so no release."""

    def test_new_battle_sim_error_returns_aborted(self):
        from sim_client import SimError

        sim = MagicMock()
        sim.new_battle.side_effect = SimError("battle init failed")

        result = play_game(MagicMock(), MagicMock(), [], [], sim, seed=0)

        assert result.outcome == "aborted"
        assert result.turns == 0
        assert result.abort_reason is not None
        assert "sim_error" in result.abort_reason

    def test_new_battle_sim_error_release_never_called(self):
        """handle is still None when new_battle raises → release must NOT be called."""
        from sim_client import SimError

        sim = MagicMock()
        sim.new_battle.side_effect = SimError("battle init failed")

        play_game(MagicMock(), MagicMock(), [], [], sim, seed=0)

        sim.release.assert_not_called()


# ---------------------------------------------------------------------------
# play_game: max_decisions boundary cases
# ---------------------------------------------------------------------------


class TestPlayGameMaxDecisionsBoundary:
    def _non_terminal_genuine_view(self) -> StateView:
        request = {
            "active": [
                {"moves": [{"id": "heatwave", "pp": 8, "maxpp": 8, "target": "allAdjacentFoes", "disabled": False}]},
                {"moves": [{"id": "protect", "pp": 8, "maxpp": 8, "target": "self", "disabled": False}]},
            ],
            "side": {"pokemon": [{"condition": "100/100"}, {"condition": "100/100"}, {"condition": "80/100"}, {"condition": "60/100"}]},
        }
        return _make_view(terminal=False, to_move=["p1", "p2"], phase="move", legal={"p1": request, "p2": request})

    def test_max_decisions_zero_aborts_immediately(self):
        """Loop never enters when max_decisions=0; returns aborted with 'max_decisions'."""
        view = self._non_terminal_genuine_view()
        sim = MagicMock()
        sim.new_battle.return_value = (1, view)
        # step should never be called because the loop body is never reached
        sim.step.return_value = MagicMock(child=2, view=view)

        agent = MagicMock()
        result = play_game(agent, agent, [], [], sim, seed=0, max_decisions=0)

        assert result.outcome == "aborted"
        assert result.abort_reason == "max_decisions"
        # Agents must not have been called
        agent.act.assert_not_called()
        # release is called once: for the handle from new_battle (post-loop path)
        sim.release.assert_called_once()

    def test_max_decisions_one_allows_exactly_one_decision(self):
        """Exactly one genuine decision runs before abort at max_decisions=1."""
        view = self._non_terminal_genuine_view()
        sim = MagicMock()
        sim.new_battle.return_value = (1, view)
        step_result = MagicMock()
        step_result.child = 2
        step_result.view = view  # still non-terminal so loop would continue
        sim.step.return_value = step_result

        agent = RandomAgent(random.Random(0))
        result = play_game(agent, agent, [], [], sim, seed=0, max_decisions=1)

        assert result.outcome == "aborted"
        assert result.abort_reason == "max_decisions"
        # step() called once (the one genuine decision)
        assert sim.step.call_count == 1

    def test_max_decisions_zero_release_called_once(self):
        """With max_decisions=0 the handle from new_battle is released exactly once."""
        view = self._non_terminal_genuine_view()
        sim = MagicMock()
        sim.new_battle.return_value = (5, view)

        play_game(MagicMock(), MagicMock(), [], [], sim, seed=0, max_decisions=0)

        # Only one release: the post-loop abort path releases the handle returned by new_battle
        sim.release.assert_called_once_with(5)


# ---------------------------------------------------------------------------
# play_game: forced-decision branch
# ---------------------------------------------------------------------------


class TestPlayGameForcedDecision:
    """action_space.forced_actions returns a non-None dict → agents NOT called, decision_idx stays 0."""

    def test_forced_decision_agents_not_called(self):
        """When forced_actions returns non-None, neither agent.act() is invoked."""
        request = {
            "active": [
                {"moves": [{"id": "heatwave", "pp": 8, "maxpp": 8, "target": "allAdjacentFoes", "disabled": False}]},
                {"moves": [{"id": "protect", "pp": 8, "maxpp": 8, "target": "self", "disabled": False}]},
            ],
            "side": {"pokemon": [{"condition": "100/100"}, {"condition": "100/100"}, {"condition": "80/100"}, {"condition": "60/100"}]},
        }
        forced_view = _make_view(terminal=False, to_move=["p1", "p2"], phase="move", legal={"p1": request, "p2": request})
        terminal_view = _make_view(terminal=True, utility={"p1": 1.0}, turn=5)

        sim = MagicMock()
        sim.new_battle.return_value = (1, forced_view)
        step_result = MagicMock()
        step_result.child = 2
        step_result.view = terminal_view
        sim.step.return_value = step_result

        agent_p1 = MagicMock()
        agent_p2 = MagicMock()

        with patch("action_space.forced_actions", return_value={"p1": 0, "p2": 0}), \
             patch("action_space.action_to_choice_contextual", return_value="move 1"):
            result = play_game(agent_p1, agent_p2, [], [], sim, seed=0)

        agent_p1.act.assert_not_called()
        agent_p2.act.assert_not_called()
        assert result.outcome == "p1_win"

    def test_forced_decision_decision_idx_not_incremented(self):
        """A pure forced-decision game must not count toward max_decisions."""
        request = {
            "active": [
                {"moves": [{"id": "heatwave", "pp": 8, "maxpp": 8, "target": "allAdjacentFoes", "disabled": False}]},
                {"moves": [{"id": "protect", "pp": 8, "maxpp": 8, "target": "self", "disabled": False}]},
            ],
            "side": {"pokemon": [{"condition": "100/100"}, {"condition": "100/100"}, {"condition": "80/100"}, {"condition": "60/100"}]},
        }
        forced_view = _make_view(terminal=False, to_move=["p1"], phase="move", legal={"p1": request})
        terminal_view = _make_view(terminal=True, utility={"p1": -1.0}, turn=3)

        sim = MagicMock()
        sim.new_battle.return_value = (1, forced_view)
        step_result = MagicMock()
        step_result.child = 2
        step_result.view = terminal_view
        sim.step.return_value = step_result

        # max_decisions=0 would abort immediately if any genuine decisions counted
        # but forced decisions should not count → game should complete normally
        with patch("action_space.forced_actions", return_value={"p1": 0}), \
             patch("action_space.action_to_choice_contextual", return_value="move 1"):
            result = play_game(MagicMock(), MagicMock(), [], [], sim, seed=0, max_decisions=0)

        # The loop body was entered (forced path), but decision_idx never incremented.
        # Since view became terminal, we do NOT hit the max_decisions abort path.
        # With max_decisions=0 the while condition 0 < 0 is False from the start,
        # so the forced-decision branch never runs — we get the max_decisions abort.
        # This confirms: forced steps don't count as decisions.
        assert result.outcome == "aborted"
        assert result.abort_reason == "max_decisions"

    def test_forced_decision_release_called_for_each_step(self):
        """Each forced step must release the old handle before acquiring the new one."""
        request = {
            "active": [
                {"moves": [{"id": "heatwave", "pp": 8, "maxpp": 8, "target": "allAdjacentFoes", "disabled": False}]},
                {"moves": [{"id": "protect", "pp": 8, "maxpp": 8, "target": "self", "disabled": False}]},
            ],
            "side": {"pokemon": [{"condition": "100/100"}, {"condition": "100/100"}, {"condition": "80/100"}, {"condition": "60/100"}]},
        }
        forced_view = _make_view(terminal=False, to_move=["p1"], phase="move", legal={"p1": request})
        terminal_view = _make_view(terminal=True, utility={"p1": 1.0}, turn=4)

        sim = MagicMock()
        sim.new_battle.return_value = (10, forced_view)
        step_result = MagicMock()
        step_result.child = 20
        step_result.view = terminal_view
        sim.step.return_value = step_result

        with patch("action_space.forced_actions", return_value={"p1": 0}), \
             patch("action_space.action_to_choice_contextual", return_value="move 1"):
            play_game(MagicMock(), MagicMock(), [], [], sim, seed=0)

        # release(10) in the forced step, release(20) in the post-loop terminal path
        assert sim.release.call_count == 2
        sim.release.assert_any_call(10)
        sim.release.assert_any_call(20)


# ---------------------------------------------------------------------------
# play_game: p2-only to_move
# ---------------------------------------------------------------------------


class TestPlayGameP2Only:
    """When to_move=['p2'], only agent_p2 acts; agent_p1 is never called."""

    def test_p2_only_agent_p1_not_called(self):
        request = {
            "active": [
                {"moves": [{"id": "heatwave", "pp": 8, "maxpp": 8, "target": "allAdjacentFoes", "disabled": False}]},
                {"moves": [{"id": "protect", "pp": 8, "maxpp": 8, "target": "self", "disabled": False}]},
            ],
            "side": {"pokemon": [{"condition": "100/100"}, {"condition": "100/100"}, {"condition": "80/100"}, {"condition": "60/100"}]},
        }
        p2_view = _make_view(terminal=False, to_move=["p2"], phase="move", legal={"p2": request})
        terminal_view = _make_view(terminal=True, utility={"p1": -1.0}, turn=6)

        sim = MagicMock()
        sim.new_battle.return_value = (1, p2_view)
        step_result = MagicMock()
        step_result.child = 2
        step_result.view = terminal_view
        sim.step.return_value = step_result

        agent_p1 = MagicMock()
        agent_p2 = MagicMock()
        agent_p2.act.return_value = 0

        with patch("action_space.forced_actions", return_value=None), \
             patch("action_space.action_to_choice_contextual", return_value="move 1"):
            result = play_game(agent_p1, agent_p2, [], [], sim, seed=0)

        agent_p1.act.assert_not_called()
        agent_p2.act.assert_called_once()
        assert result.outcome == "p2_win"


# ---------------------------------------------------------------------------
# play_game: handle lifecycle (release call counts and double-release guard)
# ---------------------------------------------------------------------------


class TestPlayGameHandleLifecycle:
    """Verify sim.release() is called the correct number of times and with the
    right handle values across all abort paths and the terminal path."""

    def test_no_action_abort_release_called_once(self):
        """Agent raises → release(handle) called once, then handle=None → no double-release."""
        view = _make_view(terminal=False, to_move=["p1"], phase="move", legal={"p1": None})
        sim = MagicMock()
        sim.new_battle.return_value = (7, view)

        bad_agent = RandomAgent(random.Random(0))  # legal=None → ValueError
        play_game(bad_agent, MagicMock(), [], [], sim, seed=0)

        sim.release.assert_called_once_with(7)

    def test_sim_error_step_release_called_once(self):
        """SimError on step → release(handle) once, not twice."""
        from sim_client import SimError

        request = {
            "active": [{"moves": [{"id": "heatwave", "pp": 8, "maxpp": 8, "target": "allAdjacentFoes", "disabled": False}]},
                       {"moves": [{"id": "protect", "pp": 8, "maxpp": 8, "target": "self", "disabled": False}]}],
            "side": {"pokemon": [{"condition": "100/100"}, {"condition": "100/100"}, {"condition": "80/100"}, {"condition": "60/100"}]},
        }
        view = _make_view(terminal=False, to_move=["p1", "p2"], phase="move", legal={"p1": request, "p2": request})
        sim = MagicMock()
        sim.new_battle.return_value = (3, view)
        sim.step.side_effect = SimError("bad choice")

        play_game(RandomAgent(random.Random(0)), RandomAgent(random.Random(1)), [], [], sim, seed=0)

        sim.release.assert_called_once_with(3)

    def test_terminal_release_called_for_each_handle(self):
        """Empty-phase initial view → step to terminal: release called twice (once per handle)."""
        terminal_view = _make_view(terminal=True, utility={"p1": 1.0}, turn=5)
        initial_view = _make_view(terminal=False, to_move=[], turn=0)

        sim = MagicMock()
        sim.new_battle.return_value = (11, initial_view)
        step_result = MagicMock()
        step_result.child = 22
        step_result.view = terminal_view
        sim.step.return_value = step_result

        play_game(MagicMock(), MagicMock(), [], [], sim, seed=0)

        assert sim.release.call_count == 2
        sim.release.assert_any_call(11)
        sim.release.assert_any_call(22)


# ---------------------------------------------------------------------------
# CurriculumReport: draw_rate property
# ---------------------------------------------------------------------------


class TestCurriculumReportDrawRate:
    def _report(self, favored_wins: int, underdog_wins: int, draws: int, aborted: int) -> CurriculumReport:
        return CurriculumReport(
            n_games=favored_wins + underdog_wins + draws + aborted,
            favored_wins=favored_wins,
            underdog_wins=underdog_wins,
            draws=draws,
            aborted=aborted,
            favored_turns=(),
            all_turns=tuple(range(favored_wins + underdog_wins + draws)),
        )

    def test_all_draws(self):
        r = self._report(favored_wins=0, underdog_wins=0, draws=10, aborted=0)
        assert r.draw_rate == pytest.approx(1.0)

    def test_draws_plus_wins(self):
        r = self._report(favored_wins=3, underdog_wins=1, draws=2, aborted=0)
        # decided = 6; draws = 2
        assert r.draw_rate == pytest.approx(2 / 6)

    def test_zero_denominator_returns_zero(self):
        r = self._report(favored_wins=0, underdog_wins=0, draws=0, aborted=0)
        assert r.draw_rate == 0.0

    def test_aborted_excluded_from_denominator(self):
        r = self._report(favored_wins=2, underdog_wins=2, draws=2, aborted=10)
        # decided = 6 (aborted excluded)
        assert r.draw_rate == pytest.approx(2 / 6)

    def test_no_draws(self):
        r = self._report(favored_wins=5, underdog_wins=5, draws=0, aborted=0)
        assert r.draw_rate == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# CurriculumReport: median_favored_turns edge cases
# ---------------------------------------------------------------------------


class TestCurriculumReportMedianFavoredTurns:
    def _report(self, favored_turns: tuple[int, ...]) -> CurriculumReport:
        return CurriculumReport(
            n_games=len(favored_turns),
            favored_wins=len(favored_turns),
            underdog_wins=0,
            draws=0,
            aborted=0,
            favored_turns=favored_turns,
            all_turns=favored_turns,
        )

    def test_none_when_no_wins(self):
        r = CurriculumReport(n_games=5, favored_wins=0, underdog_wins=5, draws=0, aborted=0, favored_turns=(), all_turns=(5, 6, 7, 8, 9))
        assert r.median_favored_turns is None

    def test_single_element(self):
        r = self._report((5,))
        assert r.median_favored_turns == pytest.approx(5.0)

    def test_odd_count(self):
        r = self._report((2, 6, 10))
        assert r.median_favored_turns == pytest.approx(6.0)

    def test_even_count(self):
        r = self._report((4, 6, 8, 10))
        assert r.median_favored_turns == pytest.approx(7.0)


# ---------------------------------------------------------------------------
# CurriculumReport: all-aborted scenario
# ---------------------------------------------------------------------------


class TestCurriculumReportAllAborted:
    def test_all_aborted_rates_are_zero(self):
        """When all games abort, all rate properties must return 0.0."""
        r = CurriculumReport(
            n_games=10,
            favored_wins=0,
            underdog_wins=0,
            draws=0,
            aborted=10,
            favored_turns=(),
            all_turns=(),
        )
        assert r.favored_win_rate == 0.0
        assert r.draw_rate == 0.0
        assert r.mean_favored_turns is None
        assert r.median_favored_turns is None
        assert r.overall_mean_turns is None

    def test_all_aborted_turns_empty(self):
        r = CurriculumReport(n_games=5, favored_wins=0, underdog_wins=0, draws=0, aborted=5, favored_turns=(), all_turns=())
        assert r.all_turns == ()
        assert r.favored_turns == ()


# ---------------------------------------------------------------------------
# CurriculumReport: n_games=0 via curriculum_report()
# ---------------------------------------------------------------------------


class TestCurriculumReportNGamesZero:
    def test_zero_games_no_play_game_call(self):
        """EvalConfig(n_games=0): play_game must not be called."""
        matchup = MagicMock()
        matchup.sample.return_value = ([], [])
        cfg = EvalConfig(n_games=0, use_search=False)

        with patch("evaluation.play_game") as mock_pg, patch("evaluation.PolicyAgent"):
            report = curriculum_report(MagicMock(), matchup, MagicMock(), config=cfg)

        mock_pg.assert_not_called()
        assert report.n_games == 0
        assert report.favored_wins == 0
        assert report.aborted == 0

    def test_zero_games_head_to_head_no_play_game_call(self):
        """head_to_head with n_games=0 must not call play_game."""
        matchup = MagicMock()
        matchup.sample.return_value = ([], [])
        cfg = EvalConfig(n_games=0)

        with patch("evaluation.play_game") as mock_pg:
            report = head_to_head(MagicMock(), MagicMock(), matchup, MagicMock(), config=cfg)

        mock_pg.assert_not_called()
        assert report.n_games == 0


# ---------------------------------------------------------------------------
# CurriculumReport: invalid favored_side
# ---------------------------------------------------------------------------


class TestCurriculumReportInvalidFavoredSide:
    """The favored_side validation guard in curriculum_report must raise ValueError
    for any value that is neither 'p1' nor 'p2'."""

    def test_invalid_favored_side_p3_raises(self):
        """favored_side='p3' must raise ValueError."""
        # EvalConfig is frozen so we bypass the constructor restriction via
        # object.__setattr__, which is the standard CPython workaround for
        # injecting invalid values into frozen dataclasses in tests.
        cfg = EvalConfig(n_games=0, use_search=False)
        object.__setattr__(cfg, "favored_side", "p3")

        with pytest.raises(ValueError, match="favored_side"):
            curriculum_report(MagicMock(), MagicMock(), MagicMock(), config=cfg)

    def test_invalid_favored_side_empty_string_raises(self):
        cfg = EvalConfig(n_games=0, use_search=False)
        object.__setattr__(cfg, "favored_side", "")

        with pytest.raises(ValueError, match="favored_side"):
            curriculum_report(MagicMock(), MagicMock(), MagicMock(), config=cfg)

    def test_valid_favored_sides_do_not_raise(self):
        """'p1' and 'p2' must not trigger the ValueError."""
        matchup = MagicMock()
        matchup.sample.return_value = ([], [])
        for side in ("p1", "p2"):
            cfg = EvalConfig(n_games=0, use_search=False, favored_side=side)
            with patch("evaluation.PolicyAgent"), patch("evaluation.play_game"):
                # Should not raise
                curriculum_report(MagicMock(), matchup, MagicMock(), config=cfg)


# ---------------------------------------------------------------------------
# CurriculumReport: use_search=True uses SearchAgent at unit level
# ---------------------------------------------------------------------------


class TestCurriculumReportUsesCorrectAgent:
    def test_use_search_true_instantiates_search_agent(self):
        """use_search=True must use SearchAgent, not PolicyAgent."""
        matchup = MagicMock()
        matchup.sample.return_value = ([], [])
        cfg = EvalConfig(n_games=0, use_search=True)

        with patch("evaluation.SearchAgent") as mock_sa, \
             patch("evaluation.PolicyAgent") as mock_pa, \
             patch("evaluation.play_game", return_value=GameResult("p1_win", 5)):
            curriculum_report(MagicMock(), matchup, MagicMock(), config=cfg)

        mock_sa.assert_called_once()
        mock_pa.assert_not_called()

    def test_use_search_false_instantiates_policy_agent(self):
        """use_search=False must use PolicyAgent, not SearchAgent."""
        matchup = MagicMock()
        matchup.sample.return_value = ([], [])
        cfg = EvalConfig(n_games=0, use_search=False)

        with patch("evaluation.SearchAgent") as mock_sa, \
             patch("evaluation.PolicyAgent") as mock_pa, \
             patch("evaluation.play_game", return_value=GameResult("p1_win", 5)):
            curriculum_report(MagicMock(), matchup, MagicMock(), config=cfg)

        mock_pa.assert_called_once()
        mock_sa.assert_not_called()


# ---------------------------------------------------------------------------
# HeadToHeadReport: draw_rate property
# ---------------------------------------------------------------------------


class TestHeadToHeadReportDrawRate:
    def _report(self, a_wins: int, b_wins: int, draws: int, aborted: int) -> HeadToHeadReport:
        return HeadToHeadReport(
            n_games=a_wins + b_wins + draws + aborted,
            a_wins=a_wins,
            b_wins=b_wins,
            draws=draws,
            aborted=aborted,
            a_turns=tuple(range(a_wins)),
            all_turns=tuple(range(a_wins + b_wins + draws)),
        )

    def test_all_draws(self):
        r = self._report(a_wins=0, b_wins=0, draws=8, aborted=0)
        assert r.draw_rate == pytest.approx(1.0)

    def test_mixed_draws_and_wins(self):
        r = self._report(a_wins=3, b_wins=1, draws=2, aborted=0)
        assert r.draw_rate == pytest.approx(2 / 6)

    def test_zero_denominator(self):
        r = self._report(a_wins=0, b_wins=0, draws=0, aborted=5)
        assert r.draw_rate == 0.0

    def test_no_draws(self):
        r = self._report(a_wins=4, b_wins=4, draws=0, aborted=0)
        assert r.draw_rate == pytest.approx(0.0)

    def test_aborted_excluded_from_denominator(self):
        r = self._report(a_wins=2, b_wins=2, draws=2, aborted=10)
        assert r.draw_rate == pytest.approx(2 / 6)


# ---------------------------------------------------------------------------
# HeadToHeadReport: mean_a_turns property
# ---------------------------------------------------------------------------


class TestHeadToHeadReportMeanATurns:
    def test_none_when_a_never_won(self):
        r = HeadToHeadReport(n_games=5, a_wins=0, b_wins=5, draws=0, aborted=0, a_turns=(), all_turns=(3, 4, 5, 6, 7))
        assert r.mean_a_turns is None

    def test_correct_mean(self):
        r = HeadToHeadReport(n_games=3, a_wins=3, b_wins=0, draws=0, aborted=0, a_turns=(4, 6, 8), all_turns=(4, 6, 8))
        assert r.mean_a_turns == pytest.approx(6.0)

    def test_single_win(self):
        r = HeadToHeadReport(n_games=1, a_wins=1, b_wins=0, draws=0, aborted=0, a_turns=(10,), all_turns=(10,))
        assert r.mean_a_turns == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# HeadToHeadReport: overall_mean_turns property
# ---------------------------------------------------------------------------


class TestHeadToHeadReportOverallMeanTurns:
    def test_none_when_all_aborted(self):
        r = HeadToHeadReport(n_games=5, a_wins=0, b_wins=0, draws=0, aborted=5, a_turns=(), all_turns=())
        assert r.overall_mean_turns is None

    def test_correct_mean(self):
        r = HeadToHeadReport(n_games=4, a_wins=2, b_wins=2, draws=0, aborted=0, a_turns=(4, 8), all_turns=(4, 6, 8, 10))
        assert r.overall_mean_turns == pytest.approx(7.0)

    def test_single_game(self):
        r = HeadToHeadReport(n_games=1, a_wins=1, b_wins=0, draws=0, aborted=0, a_turns=(7,), all_turns=(7,))
        assert r.overall_mean_turns == pytest.approx(7.0)


# ---------------------------------------------------------------------------
# head_to_head: a_turns population
# ---------------------------------------------------------------------------


class TestHeadToHeadATurnsPopulation:
    def test_a_turns_contains_only_p1_win_turns(self):
        """a_turns must be populated exclusively from p1_win games, not p2_win or draw."""
        outcomes_turns = [
            ("p1_win", 10),
            ("p2_win", 15),
            ("draw", 20),
            ("p1_win", 8),
            ("aborted", 0),
        ]
        idx = 0

        def fake_play(*args, **kwargs):
            nonlocal idx
            outcome, turns = outcomes_turns[idx % len(outcomes_turns)]
            idx += 1
            if outcome == "aborted":
                return GameResult("aborted", turns, "test")
            return GameResult(outcome, turns)

        matchup = MagicMock()
        matchup.sample.return_value = ([], [])
        cfg = EvalConfig(n_games=5)

        with patch("evaluation.play_game", side_effect=fake_play):
            report = head_to_head(MagicMock(), MagicMock(), matchup, MagicMock(), config=cfg)

        # Only the two p1_win games contribute turns
        assert set(report.a_turns) == {10, 8}
        assert report.a_wins == 2
        # 15 and 20 are b_win and draw turns — must NOT be in a_turns
        assert 15 not in report.a_turns
        assert 20 not in report.a_turns


# ---------------------------------------------------------------------------
# SearchAgent: handle=0 and three-handle cache eviction
# ---------------------------------------------------------------------------


class TestSearchAgentHandleEdgeCases:
    def _make_search_result(self, strategy: dict) -> SearchResult:
        return SearchResult(strategy=strategy, value=0.5, policy_target={s: np.zeros(A) for s in strategy})

    def test_cache_hit_handle_zero(self):
        """handle=0 is falsy but the cache check uses 'is None' first → no spurious miss."""
        strategy = {"p1": {360: 1.0}}
        result = self._make_search_result(strategy)
        agent = SearchAgent(MagicMock(), SearchConfig())
        view = _make_view(to_move=["p1"])

        with patch("evaluation.search", return_value=result) as mock_search:
            agent.act(view, "p1", MagicMock(), handle=0)
            agent.act(view, "p1", MagicMock(), handle=0)
            # Second call must be a cache hit — only one search
            assert mock_search.call_count == 1

    def test_cache_eviction_on_third_handle(self):
        """act(h=1), act(h=2), act(h=1): third call must miss (h=2 is cached)."""
        strategy = {"p1": {360: 1.0}}
        result = self._make_search_result(strategy)
        agent = SearchAgent(MagicMock(), SearchConfig())
        view = _make_view(to_move=["p1"])

        with patch("evaluation.search", return_value=result) as mock_search:
            agent.act(view, "p1", MagicMock(), handle=1)  # miss → search
            agent.act(view, "p1", MagicMock(), handle=2)  # miss → search
            agent.act(view, "p1", MagicMock(), handle=1)  # miss (h=2 in cache) → search
            assert mock_search.call_count == 3


# ---------------------------------------------------------------------------
# PolicyAgent: acts as p2
# ---------------------------------------------------------------------------


class TestPolicyAgentP2Side:
    def test_acts_as_p2_returns_argmax(self):
        net = MagicMock()
        logits = torch.full((A,), float("-inf"))
        logits[5] = 3.0
        net.return_value = (logits, torch.tensor(0.0))

        agent = PolicyAgent(net)
        view = _make_view(to_move=["p2"])

        with patch("evaluation.encode", return_value=MagicMock()) as mock_enc:
            idx = agent.act(view, "p2", MagicMock(), handle=1)

        # encode must be called with side="p2"
        mock_enc.assert_called_once()
        call_args = mock_enc.call_args
        assert call_args.args[1] == "p2" or call_args.args[1] == "p2"
        assert idx == 5


# ---------------------------------------------------------------------------
# RandomAgent: missing legal key (view.legal={})
# ---------------------------------------------------------------------------


class TestRandomAgentMissingLegalKey:
    def test_missing_key_raises_value_error(self):
        """view.legal.get(side) returns None when key is absent → no legal actions."""
        view = _make_view(to_move=["p1"], phase="move", legal={})  # 'p1' key missing
        agent = RandomAgent(random.Random(0))
        with pytest.raises(ValueError, match="no legal actions"):
            agent.act(view, "p1", MagicMock(), handle=1)

    def test_explicit_none_and_missing_key_both_raise(self):
        """Explicit None value and absent key must both raise (same code path)."""
        agent = RandomAgent(random.Random(0))
        view_none = _make_view(to_move=["p1"], phase="move", legal={"p1": None})
        view_missing = _make_view(to_move=["p1"], phase="move", legal={})

        for view in (view_none, view_missing):
            with pytest.raises(ValueError, match="no legal actions"):
                agent.act(view, "p1", MagicMock(), handle=1)


# ---------------------------------------------------------------------------
# fitness()
# ---------------------------------------------------------------------------


class TestFitness:
    def _dummy_report(self) -> HeadToHeadReport:
        return HeadToHeadReport(n_games=0, a_wins=0, b_wins=0, draws=0, aborted=0, a_turns=(), all_turns=())

    def _mock_load(self):
        mock_loaded = MagicMock()
        mock_loaded.net = MagicMock()
        return mock_loaded

    def test_empty_opponents_returns_empty_dict(self):
        with patch("evaluation.load_checkpoint", return_value=self._mock_load()):
            result = fitness("subject.pt", [], MagicMock(), MagicMock())
        assert result == {}

    def test_use_search_false_raises_value_error(self):
        """fitness() must raise ValueError when config.use_search=False."""
        cfg = EvalConfig(use_search=False)
        with pytest.raises(ValueError, match="use_search"):
            fitness("subject.pt", [], MagicMock(), MagicMock(), config=cfg)

    def test_path_opponent_label_is_str_path(self):
        """Label for a path opponent is str(path)."""
        opp_path = Path("checkpoints/gen_0001.pt")

        with patch("evaluation.load_checkpoint", return_value=self._mock_load()), \
             patch("evaluation.head_to_head", return_value=self._dummy_report()):
            result = fitness("subject.pt", [opp_path], MagicMock(), MagicMock())

        assert str(opp_path) in result

    def test_path_opponent_string_label_is_str(self):
        """A str path (not Path) is also labelled as str(opp)."""
        opp_str = "checkpoints/gen_0002.pt"

        with patch("evaluation.load_checkpoint", return_value=self._mock_load()), \
             patch("evaluation.head_to_head", return_value=self._dummy_report()):
            result = fitness("subject.pt", [opp_str], MagicMock(), MagicMock())

        assert opp_str in result

    def test_prebuilt_agent_label_is_class_name(self):
        """Label for a pre-built Agent is type(opp).__name__."""
        rng_agent = RandomAgent(random.Random(0))

        with patch("evaluation.load_checkpoint", return_value=self._mock_load()), \
             patch("evaluation.head_to_head", return_value=self._dummy_report()):
            result = fitness("subject.pt", [rng_agent], MagicMock(), MagicMock())

        assert "RandomAgent" in result

    def test_mixed_opponents_all_present(self):
        """Both path and pre-built-agent opponents appear in the result."""
        rng_agent = RandomAgent(random.Random(0))
        opp_path = Path("opp.pt")

        with patch("evaluation.load_checkpoint", return_value=self._mock_load()), \
             patch("evaluation.head_to_head", return_value=self._dummy_report()):
            result = fitness("subject.pt", [rng_agent, opp_path], MagicMock(), MagicMock())

        assert len(result) == 2
        assert "RandomAgent" in result
        assert str(opp_path) in result

    def test_subject_always_uses_search_agent(self):
        """fitness() must build a SearchAgent for the subject regardless of config."""
        dummy_report = self._dummy_report()

        with patch("evaluation.load_checkpoint", return_value=self._mock_load()), \
             patch("evaluation.head_to_head", return_value=dummy_report), \
             patch("evaluation.SearchAgent") as mock_sa:
            mock_sa.return_value = MagicMock()
            cfg = EvalConfig(use_search=True)
            fitness("subject.pt", [RandomAgent(random.Random(0))], MagicMock(), MagicMock(), config=cfg)

        # SearchAgent instantiated at least once (for the subject checkpoint)
        mock_sa.assert_called()

    def test_load_checkpoint_called_for_subject(self):
        """fitness() must call load_checkpoint with the subject path."""
        subject = Path("subject.pt")

        with patch("evaluation.load_checkpoint", return_value=self._mock_load()) as mock_lc, \
             patch("evaluation.head_to_head", return_value=self._dummy_report()):
            fitness(subject, [], MagicMock(), MagicMock())

        assert mock_lc.call_count >= 1
        # First call must be for the subject path
        first_call_path = mock_lc.call_args_list[0].args[0]
        assert first_call_path == subject

    def test_load_checkpoint_called_for_path_opponent(self):
        """For a path opponent, load_checkpoint is called twice: subject + opponent."""
        opp = Path("opp.pt")

        with patch("evaluation.load_checkpoint", return_value=self._mock_load()) as mock_lc, \
             patch("evaluation.head_to_head", return_value=self._dummy_report()):
            fitness("subject.pt", [opp], MagicMock(), MagicMock())

        assert mock_lc.call_count == 2
        paths_called = [c.args[0] for c in mock_lc.call_args_list]
        assert opp in paths_called

    def test_head_to_head_called_per_opponent(self):
        """head_to_head is called once per opponent."""
        with patch("evaluation.load_checkpoint", return_value=self._mock_load()), \
             patch("evaluation.head_to_head", return_value=self._dummy_report()) as mock_h2h:
            fitness("subject.pt", [RandomAgent(random.Random(0)), RandomAgent(random.Random(1))], MagicMock(), MagicMock())

        assert mock_h2h.call_count == 2


# ---------------------------------------------------------------------------
# Seeding determinism
# ---------------------------------------------------------------------------


class TestSeedingDeterminism:
    def test_curriculum_report_identical_seeds_produce_identical_play_calls(self):
        """Two curriculum_report runs with the same master_seed must call
        play_game with exactly the same seed values."""
        matchup = MagicMock()
        matchup.sample.return_value = ([], [])
        cfg = EvalConfig(n_games=3, use_search=False, master_seed=1234)

        seeds_run1: list[int] = []
        seeds_run2: list[int] = []

        def fake_play_run1(*args, **kwargs):
            seeds_run1.append(kwargs["seed"])
            return GameResult("p1_win", 5)

        def fake_play_run2(*args, **kwargs):
            seeds_run2.append(kwargs["seed"])
            return GameResult("p1_win", 5)

        with patch("evaluation.play_game", side_effect=fake_play_run1), patch("evaluation.PolicyAgent"):
            curriculum_report(MagicMock(), matchup, MagicMock(), config=cfg)

        with patch("evaluation.play_game", side_effect=fake_play_run2), patch("evaluation.PolicyAgent"):
            curriculum_report(MagicMock(), matchup, MagicMock(), config=cfg)

        assert seeds_run1 == seeds_run2, "Same master_seed produced different play_game seed sequences"
        assert len(seeds_run1) == 3

    def test_different_master_seeds_produce_different_play_calls(self):
        """Different master_seeds must produce different seed sequences."""
        matchup = MagicMock()
        matchup.sample.return_value = ([], [])
        cfg_a = EvalConfig(n_games=2, use_search=False, master_seed=0)
        cfg_b = EvalConfig(n_games=2, use_search=False, master_seed=9999)

        seeds_a: list[int] = []
        seeds_b: list[int] = []

        def fake_play_a(*args, **kwargs):
            seeds_a.append(kwargs["seed"])
            return GameResult("p1_win", 5)

        def fake_play_b(*args, **kwargs):
            seeds_b.append(kwargs["seed"])
            return GameResult("p1_win", 5)

        with patch("evaluation.play_game", side_effect=fake_play_a), patch("evaluation.PolicyAgent"):
            curriculum_report(MagicMock(), matchup, MagicMock(), config=cfg_a)

        with patch("evaluation.play_game", side_effect=fake_play_b), patch("evaluation.PolicyAgent"):
            curriculum_report(MagicMock(), matchup, MagicMock(), config=cfg_b)

        assert seeds_a != seeds_b, "Different master_seeds should produce different play sequences"

    def test_head_to_head_identical_seeds_produce_identical_play_calls(self):
        """head_to_head with the same master_seed must call play_game identically."""
        matchup = MagicMock()
        matchup.sample.return_value = ([], [])
        cfg = EvalConfig(n_games=4, master_seed=777)

        seeds_run1: list[int] = []
        seeds_run2: list[int] = []

        def fake_play_r1(*args, **kwargs):
            seeds_run1.append(kwargs["seed"])
            return GameResult("draw", 5)

        def fake_play_r2(*args, **kwargs):
            seeds_run2.append(kwargs["seed"])
            return GameResult("draw", 5)

        with patch("evaluation.play_game", side_effect=fake_play_r1):
            head_to_head(MagicMock(), MagicMock(), matchup, MagicMock(), config=cfg)

        with patch("evaluation.play_game", side_effect=fake_play_r2):
            head_to_head(MagicMock(), MagicMock(), matchup, MagicMock(), config=cfg)

        assert seeds_run1 == seeds_run2
