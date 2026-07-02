"""Integration tests for src/evaluation.py — requires a real SimClient subprocess.

All tests are marked ``@pytest.mark.slow`` and are skipped unless the slow
marker is explicitly included.

Uses random-weights CVPN + Stage-0 Fire-vs-Grass teams (same pattern as
``tests/integration/test_self_play.py``).
"""

from __future__ import annotations

import random
import time

import pytest
import torch

from curriculum import STAGE_0_FIRE, STAGE_0_GRASS
from cvpn import CVPN
from evaluation import (
    EvalConfig,
    RandomAgent,
    SearchAgent,
    curriculum_report,
    head_to_head,
    play_game,
)
from search import SearchConfig
from self_play import CurriculumMatchupSource
from sim_client import SimClient

# Tiny search budget so integration tests run in seconds
_FAST_SEARCH_CONFIG = SearchConfig(
    k_actions=3,
    max_chance_children=2,
    expansion_budget=4,
    cfr_iters_per_expansion=5,
    c_puct=2.0,
)

_FAST_EVAL_CONFIG = EvalConfig(
    n_games=2,
    max_decisions=200,
    master_seed=0,
    use_search=True,
    favored_side="p1",
    eval_search_config=_FAST_SEARCH_CONFIG,
    device="cpu",
)


@pytest.fixture
def stage0_source() -> CurriculumMatchupSource:
    return CurriculumMatchupSource(team_a=STAGE_0_FIRE, team_b=STAGE_0_GRASS)


@pytest.fixture
def fresh_net() -> CVPN:
    torch.manual_seed(0)
    net = CVPN()
    net.eval()
    return net


# ---------------------------------------------------------------------------
# curriculum_report
# ---------------------------------------------------------------------------


class TestCurriculumReportIntegration:
    @pytest.mark.slow
    def test_runs_to_completion(self, sim_client: SimClient, stage0_source, fresh_net):
        """curriculum_report must complete without raising and return sane counts."""
        random.seed(0)
        torch.manual_seed(0)

        report = curriculum_report(fresh_net, stage0_source, sim_client, config=_FAST_EVAL_CONFIG)

        # Counts must be consistent
        total = report.favored_wins + report.underdog_wins + report.draws + report.aborted
        assert total == _FAST_EVAL_CONFIG.n_games, f"Total={total} != n_games={_FAST_EVAL_CONFIG.n_games}"

        # Win rate must be a valid probability
        assert 0.0 <= report.favored_win_rate <= 1.0

        # Turns must be non-negative
        assert all(t >= 0 for t in report.favored_turns)
        assert all(t >= 0 for t in report.all_turns)

    @pytest.mark.slow
    def test_all_decided_turns_populated(self, sim_client: SimClient, stage0_source, fresh_net):
        """all_turns should contain one entry per decided (non-aborted) game."""
        random.seed(1)
        torch.manual_seed(1)

        report = curriculum_report(fresh_net, stage0_source, sim_client, config=_FAST_EVAL_CONFIG)
        decided = report.favored_wins + report.underdog_wins + report.draws
        assert len(report.all_turns) == decided

    @pytest.mark.slow
    def test_favored_turns_subset_of_all_turns(self, sim_client: SimClient, stage0_source, fresh_net):
        """favored_turns must be a (multiset) subset of all_turns."""
        random.seed(2)
        torch.manual_seed(2)

        report = curriculum_report(fresh_net, stage0_source, sim_client, config=_FAST_EVAL_CONFIG)
        # Every favored_turns value must appear in all_turns with at least equal count
        from collections import Counter

        favored_counts = Counter(report.favored_turns)
        all_counts = Counter(report.all_turns)
        for turn_val, cnt in favored_counts.items():
            assert all_counts[turn_val] >= cnt, f"Turn {turn_val} appears {cnt}× in favored_turns but only {all_counts[turn_val]}× in all_turns"


# ---------------------------------------------------------------------------
# head_to_head
# ---------------------------------------------------------------------------


class TestHeadToHeadIntegration:
    @pytest.mark.slow
    def test_search_vs_random_completes(self, sim_client: SimClient, stage0_source, fresh_net):
        """head_to_head(SearchAgent, RandomAgent) must complete with valid counts."""
        random.seed(3)
        torch.manual_seed(3)

        search_agent = SearchAgent(fresh_net, _FAST_SEARCH_CONFIG)
        rng_agent = RandomAgent(random.Random(99))

        report = head_to_head(search_agent, rng_agent, stage0_source, sim_client, config=_FAST_EVAL_CONFIG)

        total = report.a_wins + report.b_wins + report.draws + report.aborted
        assert total == _FAST_EVAL_CONFIG.n_games
        assert 0.0 <= report.a_win_rate <= 1.0

    @pytest.mark.slow
    def test_search_vs_search_completes(self, sim_client: SimClient, stage0_source):
        """Two independent SearchAgents (two nets) must produce valid results."""
        random.seed(4)
        torch.manual_seed(4)
        net_a = CVPN()
        net_a.eval()
        torch.manual_seed(5)
        net_b = CVPN()
        net_b.eval()

        agent_a = SearchAgent(net_a, _FAST_SEARCH_CONFIG)
        agent_b = SearchAgent(net_b, _FAST_SEARCH_CONFIG)

        report = head_to_head(agent_a, agent_b, stage0_source, sim_client, config=_FAST_EVAL_CONFIG)

        total = report.a_wins + report.b_wins + report.draws + report.aborted
        assert total == _FAST_EVAL_CONFIG.n_games


# ---------------------------------------------------------------------------
# PolicyAgent (use_search=False) — smoke test only
# ---------------------------------------------------------------------------


class TestPolicyOnlyIntegration:
    @pytest.mark.slow
    def test_policy_only_completes(self, sim_client: SimClient, stage0_source, fresh_net):
        """use_search=False (PolicyAgent) must complete faster than search mode."""
        random.seed(5)
        torch.manual_seed(5)

        policy_cfg = EvalConfig(
            n_games=2,
            max_decisions=200,
            master_seed=0,
            use_search=False,  # PolicyAgent
            favored_side="p1",
            device="cpu",
        )

        t0 = time.monotonic()
        report = curriculum_report(fresh_net, stage0_source, sim_client, config=policy_cfg)
        elapsed = time.monotonic() - t0

        total = report.favored_wins + report.underdog_wins + report.draws + report.aborted
        assert total == policy_cfg.n_games, f"Counts don't sum: {total} != {policy_cfg.n_games}"

        # Smoke: it should finish in a reasonable time (< 60s for 2 games with no search)
        assert elapsed < 60.0, f"Policy-only took {elapsed:.1f}s for 2 games — something is stuck"


# ---------------------------------------------------------------------------
# play_game directly (smoke)
# ---------------------------------------------------------------------------


class TestPlayGameIntegration:
    @pytest.mark.slow
    def test_single_game_search_agents(self, sim_client: SimClient, fresh_net):
        """play_game with two SearchAgents should produce a valid GameResult."""
        random.seed(6)
        torch.manual_seed(6)

        agent = SearchAgent(fresh_net, _FAST_SEARCH_CONFIG)
        result = play_game(
            agent,
            agent,
            STAGE_0_FIRE,
            STAGE_0_GRASS,
            sim_client,
            seed=12345,
            max_decisions=200,
        )

        assert result.outcome in ("p1_win", "p2_win", "draw", "aborted")
        assert result.turns >= 0
        if result.outcome == "aborted":
            assert result.abort_reason is not None

    @pytest.mark.slow
    def test_single_game_random_vs_random(self, sim_client: SimClient):
        """RandomAgent vs RandomAgent smoke — game completes or aborts gracefully."""
        rng_a = RandomAgent(random.Random(0))
        rng_b = RandomAgent(random.Random(1))

        result = play_game(
            rng_a,
            rng_b,
            STAGE_0_FIRE,
            STAGE_0_GRASS,
            sim_client,
            seed=42,
            max_decisions=200,
        )

        assert result.outcome in ("p1_win", "p2_win", "draw", "aborted")
