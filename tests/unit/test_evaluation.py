"""Unit tests for src/evaluation.py — pure logic, no Node worker.

All tests run without a real SimClient.  Network calls and search() are
monkeypatched or exercised via minimal stubs.
"""

from __future__ import annotations

import random
from dataclasses import replace
from typing import Any
from unittest.mock import MagicMock, patch

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
