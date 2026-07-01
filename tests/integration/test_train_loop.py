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
from trainer import Trainer
from train_loop import TrainerConfig, TrainLoopConfig, resume_training, run_training

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

    # --- Gap 27: each generation should produce a realistic number of tuples ---
    for log in logs:
        assert log.n_tuples_generated >= 2, f"Gen {log.generation} produced only {log.n_tuples_generated} tuples"

    # --- At least one generation trained, and every loss it logged is finite ---
    trained = [log for log in logs if log.trained]
    assert trained, "buffer never warmed up enough to train"
    for log in trained:
        assert math.isfinite(log.value_loss)
        assert math.isfinite(log.policy_loss)
        assert math.isfinite(log.total_loss)

    # --- Gap 26: z_value_corr diagnostic should be populated (>= 2 tuples per gen) ---
    corrs = [log.z_value_corr for log in logs if log.z_value_corr is not None]
    assert corrs, "z_value_corr was None for every generation despite >= 2 tuples each"
    for c in corrs:
        assert math.isfinite(c)

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


@pytest.mark.slow
def test_resume_continues_from_checkpoint(sim_client: SimClient, tmp_path):
    """resume_training picks up from a checkpoint + paired buffer snapshot and drives the
    remaining generations, writing new per-generation checkpoints."""
    torch.manual_seed(0)
    net = CVPN()

    base = dict(
        games_per_generation=1,
        train_steps_per_generation=2,
        warmup_threshold=1,
        checkpoint_interval=1,
        buffer_capacity=500,
        checkpoint_dir=str(tmp_path),
        master_seed=42,
        trainer=TrainerConfig(batch_size=2, learning_rate=1e-3),
        self_play=SelfPlayConfig(temperature=1.0, max_decisions=200, search_config=_FAST_SEARCH),
    )

    # --- Fresh run to generation 2 (writes gen_0002.pt + buffer_gen_0002.pt) ---
    run_training(net, curriculum.STAGE_0, sim_client, TrainLoopConfig(n_generations=2, **base))

    # --- Resume to generation 4; only device is honored from this config, the learner
    #     knobs + buffer come from the checkpoint / snapshot. ---
    resume_cfg = {**base, "trainer": TrainerConfig(device="cpu")}
    resumed = resume_training(
        curriculum.STAGE_0,
        sim_client,
        TrainLoopConfig(n_generations=4, **resume_cfg),
        checkpoint_path=tmp_path / "gen_0002.pt",
        buffer_path=tmp_path / "buffer_gen_0002.pt",
    )

    # --- It continued from generation 3 (not 1) through 4 ---
    assert [log.generation for log in resumed] == [3, 4]
    for log in resumed:
        if log.trained:
            assert math.isfinite(log.total_loss)

    # --- New checkpoints landed for the resumed generations ---
    for gen in (3, 4):
        assert (tmp_path / f"gen_{gen:04d}.pt").exists()
        assert (tmp_path / f"buffer_gen_{gen:04d}.pt").exists()

    # --- The restored learner used the checkpoint's batch_size (2), not this config's default ---
    trainer, generation, _ = Trainer.resume(tmp_path / "gen_0004.pt")
    assert generation == 4
    assert trainer.config.batch_size == 2

    # --- Gap 28: grad norm from the resumed trainer's next step is finite ---
    buf = ReplayBuffer.load(tmp_path / "buffer_gen_0004.pt")
    step_log = trainer.train_step(buf.sample(2))
    assert step_log.grad_norm is not None and math.isfinite(step_log.grad_norm)
