"""Integration test: SelfPlay -> ReplayBuffer -> sample pipeline.

Exercises the full data flow: run a short self-play game with a real SimClient
and random CVPN, add the resulting TrainingTuples to a ReplayBuffer, sample
them back, and verify structural integrity. Also tests save/load with
real-shaped data.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import pytest
import torch

from action_space import A
from cvpn import CVPN
from replay_buffer import ReplayBuffer
from search import SearchConfig
from self_play import CurriculumMatchupSource, SelfPlayConfig, run
from training_types import SparsePolicy, TrainingTuple

if TYPE_CHECKING:
    from sim_client import SimClient

# Minimal search budget — just enough to produce valid average strategies
_FAST_SEARCH_CONFIG = SearchConfig(
    k_actions=3,
    max_chance_children=2,
    expansion_budget=4,
    cfr_iters_per_expansion=5,
    c_puct=2.0,
)


class TestReplayBufferPipeline:
    """End-to-end: SelfPlay tuples flow into ReplayBuffer and survive sampling."""

    @pytest.mark.slow
    def test_selfplay_tuples_into_buffer_and_sample(
        self, sim_client: SimClient, team_a: list, team_b: list
    ):
        """Tuples produced by a real self-play game are structurally valid after
        being added to and sampled from a ReplayBuffer."""
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

        # Generate tuples from a real game
        tuples = list(run(net, source, sim_client, config))
        assert len(tuples) >= 2, f"Expected at least 2 tuples, got {len(tuples)}"

        # Feed into a ReplayBuffer
        buf = ReplayBuffer(capacity=1000, seed=7)
        buf.add(tuples)
        assert len(buf) == len(tuples)

        # Sample back and verify every tuple survived intact
        sampled = buf.sample(len(tuples))
        assert len(sampled) == len(tuples)

        for t in sampled:
            assert isinstance(t, TrainingTuple)

            # Beta must be a valid ObsBundle with real encoder shapes
            assert "entities" in t.beta
            assert "action_mask" in t.beta
            assert t.beta["action_mask"].shape == (A,)
            # Entity features should have the encoder's real width (not tiny synthetic)
            assert t.beta["entities"].ndim == 2
            assert t.beta["entities"].shape[0] > 0  # at least one entity token
            assert t.beta["entities"].shape[1] > 0  # at least one feature

            # Policy indices must be valid action indices within [0, A)
            assert isinstance(t.policy, SparsePolicy)
            assert len(t.policy.indices) == len(t.policy.probs)
            assert all(0 <= idx < A for idx in t.policy.indices)
            if t.policy.probs:
                assert abs(sum(t.policy.probs) - 1.0) < 1e-5

            # z must be a valid game result
            assert t.z in (-1.0, 0.0, 1.0)

            # Meta fields populated
            assert t.meta.generation == 0
            assert t.meta.side in ("p1", "p2")
            assert t.meta.phase in ("teamPreview", "move", "forceSwitch")

    @pytest.mark.slow
    def test_save_load_with_real_tuples(
        self, sim_client: SimClient, team_a: list, team_b: list, tmp_path
    ):
        """Real-shaped tuples survive a save/load round-trip through torch.save."""
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
        assert len(tuples) >= 2

        # Save and reload
        buf = ReplayBuffer(capacity=1000, seed=42)
        buf.add(tuples)
        path = tmp_path / "pipeline_buf.pt"
        buf.save(path)

        loaded = ReplayBuffer.load(path)
        assert len(loaded) == len(tuples)

        # Verify the loaded tuples have the same structure
        loaded_sample = loaded.sample(len(tuples))
        for t in loaded_sample:
            assert isinstance(t, TrainingTuple)
            # Entity token count varies (8 at team preview, 12 in battle) — just check shape is valid
            assert t.beta["entities"].ndim == 2
            assert t.beta["entities"].shape[0] > 0
            assert t.beta["entities"].shape[1] > 0
            assert t.beta["action_mask"].shape == (A,)
            assert t.z in (-1.0, 0.0, 1.0)
