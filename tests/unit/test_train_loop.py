"""Unit tests for the outer-loop training driver (train_loop.py).

Mocks self_play.run so these tests run in milliseconds without a SimClient
subprocess. Covers: _z_value_corr (4 cases), GenerationLog defaults,
TrainLoopConfig validation, warmup gating, checkpoint_interval > 1,
start_gen > n_generations (empty), resume-without-RNG warning, seed uniqueness
across generations, and the batch_size/warmup interaction.
"""

from __future__ import annotations

import logging
import math
from unittest.mock import MagicMock, patch

import pytest
import torch

from action_space import A
from cvpn import CVPN, CVPNConfig
from encoder import (
    ENTITY_FEATURE_DIM,
    FIELD_FEATURE_DIM,
    NUM_MOVE_SLOTS,
    SCALAR_FEATURE_DIM,
    SIDE_FEATURE_DIM,
)
from obs_bundle import make_obs_bundle
from search import SearchConfig
from self_play import SelfPlayConfig
from trainer import TrainerConfig
from training_types import SparsePolicy, TrainingTuple, TupleMeta
from train_loop import (
    GenerationLog,
    TrainLoopConfig,
    _z_value_corr,
    resume_training,
    run_training,
)

_SMALL = CVPNConfig(d_model=32, n_heads=2, n_layers=1, ffn_mult=2)

# Minimal search config (never actually used — self_play.run is mocked).
_DUMMY_SEARCH = SearchConfig(k_actions=3, max_chance_children=2, expansion_budget=4, cfr_iters_per_expansion=5, c_puct=2.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_bundle(legal: list[int], n_entities: int = 12):
    """A synthetic ObsBundle whose action_mask is True only at ``legal`` indices."""
    mask = torch.zeros(A, dtype=torch.bool)
    mask[legal] = True
    n = n_entities
    return make_obs_bundle(
        entities=torch.randn(n, ENTITY_FEATURE_DIM),
        species_ids=torch.zeros(n, dtype=torch.long),
        ability_ids=torch.zeros(n, dtype=torch.long),
        item_ids=torch.zeros(n, dtype=torch.long),
        move_ids=torch.zeros(n, NUM_MOVE_SLOTS, dtype=torch.long),
        belief_weight=torch.ones(n),
        slot_id=torch.zeros(n, dtype=torch.long),
        field=torch.randn(FIELD_FEATURE_DIM),
        sides=torch.randn(2, SIDE_FEATURE_DIM),
        scalars=torch.randn(SCALAR_FEATURE_DIM),
        action_mask=mask,
        padding_mask=torch.ones(n, dtype=torch.bool),
    )


def _fake_tuple(legal: list[int], value: float, z: float) -> TrainingTuple:
    """A tuple with uniform policy over ``legal`` and explicit z."""
    p = 1.0 / len(legal)
    policy = SparsePolicy(indices=tuple(legal), probs=tuple(p for _ in legal))
    meta = TupleMeta(generation=0, game_id=0, decision_idx=0, phase="move", side="p1")
    return TrainingTuple(beta=_fake_bundle(legal), value=value, policy=policy, z=z, meta=meta)


def _make_tuples(n: int) -> list[TrainingTuple]:
    """Generate ``n`` fake tuples with slightly varied value/z for non-degenerate diagnostics."""
    return [_fake_tuple([5, 6, 7], value=0.1 * i, z=0.1 * i + 0.05) for i in range(n)]


def _base_config(tmp_path, **overrides) -> TrainLoopConfig:
    """Build a TrainLoopConfig with sensible test defaults, optionally overriding fields."""
    defaults = dict(
        games_per_generation=1,
        train_steps_per_generation=2,
        warmup_threshold=1,
        checkpoint_interval=1,
        n_generations=2,
        buffer_capacity=500,
        checkpoint_dir=str(tmp_path),
        master_seed=42,
        trainer=TrainerConfig(batch_size=2, learning_rate=1e-3),
        self_play=SelfPlayConfig(temperature=1.0, max_decisions=200, search_config=_DUMMY_SEARCH),
    )
    defaults.update(overrides)
    return TrainLoopConfig(**defaults)


# ---------------------------------------------------------------------------
# Gap 14: _z_value_corr — normal positive correlation
# ---------------------------------------------------------------------------


def test_z_value_corr_positive_correlation():
    """When z and value trend together, the correlation should be near 1.0."""
    tuples = [_fake_tuple([5, 6], value=float(i), z=float(i)) for i in range(5)]
    r = _z_value_corr(tuples)
    assert r is not None
    assert r == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Gap 15: _z_value_corr — fewer than 2 tuples returns None
# ---------------------------------------------------------------------------


def test_z_value_corr_empty_returns_none():
    assert _z_value_corr([]) is None


def test_z_value_corr_single_tuple_returns_none():
    assert _z_value_corr([_fake_tuple([5, 6], value=0.5, z=0.5)]) is None


# ---------------------------------------------------------------------------
# Gap 16: _z_value_corr — zero variance returns None
# ---------------------------------------------------------------------------


def test_z_value_corr_zero_variance_z_returns_none():
    """All z identical, varying value — correlation is undefined."""
    tuples = [_fake_tuple([5, 6], value=float(i), z=1.0) for i in range(5)]
    assert _z_value_corr(tuples) is None


def test_z_value_corr_zero_variance_value_returns_none():
    """All value identical, varying z — correlation is undefined."""
    tuples = [_fake_tuple([5, 6], value=0.5, z=float(i)) for i in range(5)]
    assert _z_value_corr(tuples) is None


# ---------------------------------------------------------------------------
# Gap 17: _z_value_corr — perfect correlation returns 1.0
# ---------------------------------------------------------------------------


def test_z_value_corr_perfect_positive():
    tuples = [_fake_tuple([5, 6], value=0.2 * i, z=0.2 * i) for i in range(10)]
    r = _z_value_corr(tuples)
    assert r is not None
    assert r == pytest.approx(1.0, abs=1e-6)


def test_z_value_corr_perfect_negative():
    """Anti-correlated z and value should produce r ~ -1.0."""
    tuples = [_fake_tuple([5, 6], value=float(i), z=-float(i)) for i in range(10)]
    r = _z_value_corr(tuples)
    assert r is not None
    assert r == pytest.approx(-1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Gap 18: GenerationLog defaults and trained=False path
# ---------------------------------------------------------------------------


def test_generation_log_defaults():
    log = GenerationLog(generation=1, n_tuples_generated=5, buffer_size=5)
    assert log.trained is False
    assert log.value_loss is None
    assert log.policy_loss is None
    assert log.total_loss is None
    assert log.z_value_corr is None


# ---------------------------------------------------------------------------
# Gap 19: TrainLoopConfig validation rejects illegal values
# ---------------------------------------------------------------------------


def test_train_loop_config_rejects_zero_games():
    with pytest.raises(Exception):
        TrainLoopConfig(games_per_generation=0, checkpoint_dir="/tmp/x")


def test_train_loop_config_rejects_negative_generations():
    with pytest.raises(Exception):
        TrainLoopConfig(n_generations=-1, checkpoint_dir="/tmp/x")


def test_train_loop_config_rejects_zero_buffer():
    with pytest.raises(Exception):
        TrainLoopConfig(buffer_capacity=0, checkpoint_dir="/tmp/x")


def test_train_loop_config_rejects_zero_train_steps():
    with pytest.raises(Exception):
        TrainLoopConfig(train_steps_per_generation=0, checkpoint_dir="/tmp/x")


# ---------------------------------------------------------------------------
# Gap 20: warmup gating — buffer below threshold skips training
# ---------------------------------------------------------------------------


@patch("train_loop.self_play")
def test_warmup_gating_skips_training(mock_sp, tmp_path):
    """With warmup_threshold=100 and only 3 tuples per generation, training never fires."""
    mock_sp.run.return_value = iter(_make_tuples(3))
    torch.manual_seed(0)
    net = CVPN(_SMALL)
    config = _base_config(tmp_path, warmup_threshold=100, n_generations=2)
    sim = MagicMock()
    matchup = MagicMock()

    logs = run_training(net, matchup, sim, config)

    assert len(logs) == 2
    assert all(not log.trained for log in logs)
    assert all(log.value_loss is None for log in logs)


# ---------------------------------------------------------------------------
# Gap 21: _drive with checkpoint_interval > 1
# ---------------------------------------------------------------------------


@patch("train_loop.self_play")
def test_checkpoint_interval_greater_than_one(mock_sp, tmp_path):
    """With checkpoint_interval=2 and 4 generations, only gen 2 and 4 checkpoints exist."""
    mock_sp.run.return_value = iter(_make_tuples(5))
    torch.manual_seed(0)
    net = CVPN(_SMALL)
    config = _base_config(tmp_path, checkpoint_interval=2, n_generations=4, warmup_threshold=1)
    sim = MagicMock()
    matchup = MagicMock()

    # Each call to self_play.run needs to return fresh tuples.
    mock_sp.run.side_effect = [iter(_make_tuples(5)) for _ in range(4)]
    run_training(net, matchup, sim, config)

    # Only generations 2 and 4 should have checkpoints.
    assert (tmp_path / "gen_0002.pt").exists()
    assert (tmp_path / "gen_0004.pt").exists()
    assert not (tmp_path / "gen_0001.pt").exists()
    assert not (tmp_path / "gen_0003.pt").exists()

    # Same for paired buffer snapshots.
    assert (tmp_path / "buffer_gen_0002.pt").exists()
    assert (tmp_path / "buffer_gen_0004.pt").exists()
    assert not (tmp_path / "buffer_gen_0001.pt").exists()
    assert not (tmp_path / "buffer_gen_0003.pt").exists()


# ---------------------------------------------------------------------------
# Gap 22: start_gen > n_generations returns empty logs
# ---------------------------------------------------------------------------


@patch("train_loop.self_play")
def test_resume_past_end_returns_empty(mock_sp, tmp_path):
    """resume_training returns [] when the checkpoint generation >= n_generations."""
    # Run 2 generations first to produce a checkpoint.
    mock_sp.run.side_effect = [iter(_make_tuples(5)) for _ in range(2)]
    torch.manual_seed(0)
    net = CVPN(_SMALL)
    config = _base_config(tmp_path, n_generations=2)
    sim = MagicMock()
    matchup = MagicMock()
    run_training(net, matchup, sim, config)

    # Resume with n_generations=2, checkpoint at gen 2 → start_gen=3 > 2 → empty.
    resume_config = _base_config(tmp_path, n_generations=2)
    logs = resume_training(
        matchup, sim, resume_config,
        checkpoint_path=tmp_path / "gen_0002.pt",
        buffer_path=tmp_path / "buffer_gen_0002.pt",
    )
    assert logs == []


# ---------------------------------------------------------------------------
# Gap 23: resume_training without RNG state in checkpoint (warning path)
# ---------------------------------------------------------------------------


@patch("train_loop.self_play")
def test_resume_without_rng_state_warns(mock_sp, tmp_path, caplog):
    """When the checkpoint carries no loop RNG state, a warning is logged but the run proceeds."""
    # Run 1 generation to produce a checkpoint.
    mock_sp.run.side_effect = [iter(_make_tuples(5)) for _ in range(5)]
    torch.manual_seed(0)
    net = CVPN(_SMALL)
    config = _base_config(tmp_path, n_generations=1)
    sim = MagicMock()
    matchup = MagicMock()
    run_training(net, matchup, sim, config)

    # Tamper the checkpoint to remove rng_state.
    ckpt_path = tmp_path / "gen_0001.pt"
    payload = torch.load(ckpt_path, weights_only=False)
    payload["rng_state"] = None
    torch.save(payload, ckpt_path)

    resume_config = _base_config(tmp_path, n_generations=2)
    with caplog.at_level(logging.WARNING, logger="train_loop"):
        logs = resume_training(
            matchup, sim, resume_config,
            checkpoint_path=ckpt_path,
            buffer_path=tmp_path / "buffer_gen_0001.pt",
        )

    assert len(logs) == 1
    assert logs[0].generation == 2
    assert any("loop RNG state" in msg for msg in caplog.messages)


# ---------------------------------------------------------------------------
# Gap 24: per-generation seeds differ
# ---------------------------------------------------------------------------


@patch("train_loop.self_play")
def test_per_generation_seeds_differ(mock_sp, tmp_path):
    """Each generation should receive a different master_seed via the per-generation RNG."""
    captured_seeds: list[int] = []

    def _capture_run(net, matchup_source, sim, sp_config):
        captured_seeds.append(sp_config.master_seed)
        return iter(_make_tuples(5))

    mock_sp.run.side_effect = _capture_run

    torch.manual_seed(0)
    net = CVPN(_SMALL)
    config = _base_config(tmp_path, n_generations=4, warmup_threshold=1)
    sim = MagicMock()
    matchup = MagicMock()
    run_training(net, matchup, sim, config)

    assert len(captured_seeds) == 4
    # All seeds should be distinct.
    assert len(set(captured_seeds)) == 4, f"seeds were not unique: {captured_seeds}"


# ---------------------------------------------------------------------------
# Gap 25: batch_size gate interacts with warmup
# ---------------------------------------------------------------------------


@patch("train_loop.self_play")
def test_batch_size_gate_prevents_training_below_batch_size(mock_sp, tmp_path):
    """warmup_threshold=2 is met, but batch_size=10 prevents training with only 7 tuples."""
    mock_sp.run.side_effect = [iter(_make_tuples(7))]
    torch.manual_seed(0)
    net = CVPN(_SMALL)
    config = _base_config(
        tmp_path,
        warmup_threshold=2,
        n_generations=1,
        trainer=TrainerConfig(batch_size=10, learning_rate=1e-3),
    )
    sim = MagicMock()
    matchup = MagicMock()

    logs = run_training(net, matchup, sim, config)

    assert len(logs) == 1
    # Buffer has 7 tuples (>= warmup 2), but < batch_size 10 → no training.
    assert not logs[0].trained
    assert logs[0].buffer_size == 7
