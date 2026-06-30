"""Integration test for the outer-loop driver.

Runs a tiny real Stage-0 training run: real SimClient subprocess + a fresh CVPN +
real search, a couple of generations at a small budget. Asserts the loop generates
data, trains, publishes checkpoints paired with buffer snapshots, and never produces
a NaN loss. Not a convergence test — that is the manual smoke run / Evaluation's job.
"""

from __future__ import annotations

import math

import pytest
import torch

import curriculum
from checkpoint import load_checkpoint
from cvpn import CVPN
from replay_buffer import ReplayBuffer
from search import SearchConfig
from self_play import SelfPlayConfig
from sim_client import SimClient
from train_loop import TrainerConfig, TrainLoopConfig, run_training

# Minimal search budget — same shape as the self-play integration test.
_FAST_SEARCH = SearchConfig(k_actions=3, max_chance_children=2, expansion_budget=4, cfr_iters_per_expansion=5, c_puct=2.0)


@pytest.mark.slow
def test_two_generation_stage0_run(sim_client: SimClient, tmp_path):
    torch.manual_seed(0)
    net = CVPN()

    config = TrainLoopConfig(
        games_per_generation=1,
        train_steps_per_generation=2,
        warmup_threshold=1,
        checkpoint_interval=1,
        n_generations=2,
        buffer_capacity=500,
        checkpoint_dir=str(tmp_path),
        master_seed=42,
        trainer=TrainerConfig(batch_size=2, learning_rate=1e-3),
        self_play=SelfPlayConfig(temperature=1.0, max_decisions=200, search_config=_FAST_SEARCH),
    )

    logs = run_training(net, curriculum.STAGE_0, sim_client, config)

    # --- Two generations, each producing tuples ---
    assert len(logs) == 2
    assert all(log.n_tuples_generated > 0 for log in logs)

    # --- At least one generation trained, and every loss it logged is finite ---
    trained = [log for log in logs if log.trained]
    assert trained, "buffer never warmed up enough to train"
    for log in trained:
        assert math.isfinite(log.value_loss)
        assert math.isfinite(log.policy_loss)
        assert math.isfinite(log.total_loss)

    # --- Checkpoints + paired buffer snapshots landed on disk, one per generation ---
    for gen in (1, 2):
        assert (tmp_path / f"gen_{gen:04d}.pt").exists()
        assert (tmp_path / f"buffer_gen_{gen:04d}.pt").exists()

    # --- The published checkpoint reloads into a usable net at the right generation ---
    loaded = load_checkpoint(tmp_path / "gen_0002.pt")
    assert loaded.generation == 2
    assert isinstance(loaded.net, CVPN)

    # --- The paired buffer snapshot reloads (schema fingerprint matches) ---
    restored = ReplayBuffer.load(tmp_path / "buffer_gen_0002.pt")
    assert len(restored) > 0
