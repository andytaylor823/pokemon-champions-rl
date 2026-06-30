"""Integration test for the self-play module.

Exercises the full self-play loop with a real SimClient subprocess and a CVPN
with random weights. Validates that a game runs to terminal and produces
well-formed TrainingTuples.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from action_space import A
from cvpn import CVPN
from search import SearchConfig
from self_play import (
    CurriculumMatchupSource,
    SelfPlayConfig,
    run,
)
from sim_client import SimClient
from training_types import SparsePolicy, TrainingTuple

# Use a very small search budget to keep the integration test fast
_FAST_SEARCH_CONFIG = SearchConfig(
    k_actions=3,
    max_chance_children=2,
    expansion_budget=4,
    cfr_iters_per_expansion=5,
    c_puct=2.0,
)


class TestSelfPlayFullGame:
    """Run a complete self-play game with real SimClient + random CVPN."""

    @pytest.mark.slow
    def test_game_runs_to_terminal_and_produces_tuples(
        self, sim_client: SimClient, team_a: list, team_b: list
    ):
        """A full game should produce at least one well-formed TrainingTuple."""
        random.seed(0)
        torch.manual_seed(0)
        net = CVPN()
        net.eval()

        source = CurriculumMatchupSource(team_a, team_b)
        config = SelfPlayConfig(
            temperature=1.0,
            max_decisions=200,
            master_seed=42,
            num_games=1,
            search_config=_FAST_SEARCH_CONFIG,
            generation=0,
        )

        tuples = list(run(net, source, sim_client, config))

        # Must produce at least some tuples (team preview alone is 1 decision = 2 tuples)
        assert len(tuples) >= 2, f"Expected at least 2 tuples, got {len(tuples)}"

        # Validate every tuple
        for t in tuples:
            assert isinstance(t, TrainingTuple)

            # z must be a valid game result
            assert t.z in (-1.0, 0.0, 1.0), f"Invalid z={t.z}"

            # Value should be bounded
            assert -1.0 <= t.value <= 1.0, f"Value {t.value} out of range"

            # Policy should be a valid SparsePolicy
            assert isinstance(t.policy, SparsePolicy)
            assert len(t.policy.indices) == len(t.policy.probs)
            if t.policy.probs:
                total_prob = sum(t.policy.probs)
                assert abs(total_prob - 1.0) < 1e-5, f"Policy sums to {total_prob}"
                # All probs should be non-negative
                assert all(p >= 0 for p in t.policy.probs)
                # All indices should be valid action indices
                assert all(0 <= idx < A for idx in t.policy.indices)

            # Meta should be populated
            assert t.meta.generation == 0
            assert t.meta.game_id == 0
            assert t.meta.decision_idx >= 0
            assert t.meta.phase in ("teamPreview", "move", "forceSwitch")
            assert t.meta.side in ("p1", "p2")

            # Beta should be a valid ObsBundle (TensorDict)
            assert "entities" in t.beta.keys()
            assert "action_mask" in t.beta.keys()
            assert t.beta["action_mask"].shape == (A,)

    @pytest.mark.slow
    def test_z_values_consistent_across_game(
        self, sim_client: SimClient, team_a: list, team_b: list
    ):
        """All tuples from one game should have z values consistent with a single outcome."""
        random.seed(1)
        torch.manual_seed(1)
        net = CVPN()
        net.eval()

        source = CurriculumMatchupSource(team_a, team_b)
        config = SelfPlayConfig(
            temperature=1.0,
            max_decisions=200,
            master_seed=99,
            num_games=1,
            search_config=_FAST_SEARCH_CONFIG,
            generation=0,
        )

        tuples = list(run(net, source, sim_client, config))
        assert len(tuples) > 0

        # All tuples from one game: p1's z and p2's z should be negations
        p1_z_values = {t.z for t in tuples if t.meta.side == "p1"}
        p2_z_values = {t.z for t in tuples if t.meta.side == "p2"}

        # All p1 tuples should share the same z (same game outcome)
        assert len(p1_z_values) == 1
        # All p2 tuples should share the same z
        assert len(p2_z_values) == 1

        # z values should be zero-sum
        p1_z = p1_z_values.pop()
        p2_z = p2_z_values.pop()
        assert p1_z + p2_z == 0.0 or (p1_z == 0.0 and p2_z == 0.0)

    @pytest.mark.slow
    def test_decision_indices_are_sequential(
        self, sim_client: SimClient, team_a: list, team_b: list
    ):
        """Decision indices should be sequential (0, 1, 2, ...) for each game."""
        random.seed(2)
        torch.manual_seed(2)
        net = CVPN()
        net.eval()

        source = CurriculumMatchupSource(team_a, team_b)
        config = SelfPlayConfig(
            temperature=1.0,
            max_decisions=200,
            master_seed=77,
            num_games=1,
            search_config=_FAST_SEARCH_CONFIG,
            generation=0,
        )

        tuples = list(run(net, source, sim_client, config))
        assert len(tuples) > 0

        # Collect decision indices (each decision can emit 1-2 tuples)
        decision_indices = sorted(set(t.meta.decision_idx for t in tuples))

        # Should start at 0 and be contiguous
        assert decision_indices[0] == 0
        for i in range(1, len(decision_indices)):
            assert decision_indices[i] == decision_indices[i - 1] + 1

    @pytest.mark.slow
    def test_team_preview_is_first_decision(
        self, sim_client: SimClient, team_a: list, team_b: list
    ):
        """The first decision (decision_idx=0) should be team preview."""
        random.seed(3)
        torch.manual_seed(3)
        net = CVPN()
        net.eval()

        source = CurriculumMatchupSource(team_a, team_b)
        config = SelfPlayConfig(
            temperature=1.0,
            max_decisions=200,
            master_seed=55,
            num_games=1,
            search_config=_FAST_SEARCH_CONFIG,
            generation=0,
        )

        tuples = list(run(net, source, sim_client, config))
        assert len(tuples) > 0

        # Find tuples at decision_idx=0
        first_decision_tuples = [t for t in tuples if t.meta.decision_idx == 0]
        assert len(first_decision_tuples) >= 1

        # All should be team preview phase
        for t in first_decision_tuples:
            assert t.meta.phase == "teamPreview"


class TestSelfPlayCurriculumStage0:
    """Run self-play with the Stage 0 curriculum teams (Fire vs Grass)."""

    @pytest.mark.slow
    def test_stage0_teams_produce_valid_game(self, sim_client: SimClient):
        """Stage 0 curriculum teams should produce a complete game."""
        from curriculum import STAGE_0_FIRE, STAGE_0_GRASS

        random.seed(10)
        torch.manual_seed(10)
        net = CVPN()
        net.eval()

        source = CurriculumMatchupSource(STAGE_0_FIRE, STAGE_0_GRASS)
        config = SelfPlayConfig(
            temperature=1.0,
            max_decisions=200,
            master_seed=123,
            num_games=1,
            search_config=_FAST_SEARCH_CONFIG,
            generation=0,
        )

        tuples = list(run(net, source, sim_client, config))

        # Stage 0 has 2 moves per Pokemon -> small action space
        # Should definitely produce tuples
        assert len(tuples) >= 2
        # All tuples well-formed
        for t in tuples:
            assert t.z in (-1.0, 0.0, 1.0)
            assert -1.0 <= t.value <= 1.0
            assert t.meta.side in ("p1", "p2")
