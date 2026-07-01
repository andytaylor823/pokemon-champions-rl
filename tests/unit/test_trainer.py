"""Unit tests for the Trainer (the outer-loop learner).

No SimClient or real games needed — synthetic ObsBundles with controlled action masks
exercise the loss, the masked cross-entropy (the 0×-inf NaN guard), the AdamW wiring,
and save/resume. The masking guard is the headline test: most of the A actions are
illegal (-inf logits), and training must stay finite.
"""

from __future__ import annotations

import math

import pytest
import torch
from pydantic import ValidationError

from action_space import A
from checkpoint import save_checkpoint
from cvpn import CVPN, CVPNConfig
from encoder import (
    ENTITY_FEATURE_DIM,
    FIELD_FEATURE_DIM,
    NUM_MOVE_SLOTS,
    SCALAR_FEATURE_DIM,
    SIDE_FEATURE_DIM,
)
from obs_bundle import make_obs_bundle
from trainer import Trainer, TrainerConfig, _densify_policy
from training_types import SparsePolicy, TrainingTuple, TupleMeta

_SMALL = CVPNConfig(d_model=32, n_heads=2, n_layers=1, ffn_mult=2, dropout=0.0)


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


def _fake_tuple(legal: list[int], value: float, z: float | None = None) -> TrainingTuple:
    """A tuple whose σ̄ is uniform over ``legal`` (always a subset of the legal set)."""
    p = 1.0 / len(legal)
    policy = SparsePolicy(indices=tuple(legal), probs=tuple(p for _ in legal))
    meta = TupleMeta(generation=0, game_id=0, decision_idx=0, phase="move", side="p1")
    return TrainingTuple(beta=_fake_bundle(legal), value=value, policy=policy, z=value if z is None else z, meta=meta)


def test_train_step_is_finite_with_mostly_illegal_actions():
    """The headline guard: with only a few legal actions (most logits -inf), the
    losses and gradients must stay finite — no 0×-inf=NaN leak."""
    torch.manual_seed(0)
    trainer = Trainer(CVPN(_SMALL), TrainerConfig(batch_size=4))
    # Only 3 legal actions out of A (>1000) → the vast majority of logits are -inf.
    batch = [_fake_tuple([5, 6, 7], value=0.2), _fake_tuple([5, 6, 7], value=-0.3), _fake_tuple([10, 11], value=0.5), _fake_tuple([10, 11], value=0.0)]

    log = trainer.train_step(batch)

    assert math.isfinite(log.value_loss)
    assert math.isfinite(log.policy_loss)
    assert math.isfinite(log.total_loss)
    assert log.grad_norm is not None and math.isfinite(log.grad_norm)
    # Every parameter gradient must be finite (no NaN poisoning).
    for p in trainer.net.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()


def test_train_step_reduces_loss_on_fixed_batch():
    """Overfitting a fixed batch should drive the total loss down — the learner learns."""
    torch.manual_seed(0)
    trainer = Trainer(CVPN(_SMALL), TrainerConfig(learning_rate=1e-2, batch_size=6))
    batch = [_fake_tuple([5, 6, 7], value=0.3) for _ in range(3)] + [_fake_tuple([10, 11], value=-0.4) for _ in range(3)]

    first = trainer.train_step(batch).total_loss
    last = first
    for _ in range(60):
        last = trainer.train_step(batch).total_loss

    assert last < first


def test_value_loss_targets_search_value_not_z():
    """With policy_weight=0, value targets 0.7, and z=-1, the value loss must collapse
    toward 0 — proving the regression target is ``value`` (search-refined), not ``z``."""
    torch.manual_seed(0)
    trainer = Trainer(CVPN(_SMALL), TrainerConfig(learning_rate=1e-2, value_weight=1.0, policy_weight=0.0, batch_size=4))
    batch = [_fake_tuple([5, 6, 7], value=0.7, z=-1.0) for _ in range(4)]

    last = None
    for _ in range(150):
        last = trainer.train_step(batch).value_loss

    assert last < 0.01  # v_hat converged to ~0.7; had it chased z=-1 the loss would stay large


def test_optimizer_is_adamw_with_configured_hyperparams():
    cfg = TrainerConfig(learning_rate=3e-4, weight_decay=1e-2)
    trainer = Trainer(CVPN(_SMALL), cfg)
    assert isinstance(trainer.optimizer, torch.optim.AdamW)
    group = trainer.optimizer.param_groups[0]
    assert group["lr"] == pytest.approx(3e-4)
    assert group["weight_decay"] == pytest.approx(1e-2)


def test_empty_batch_raises():
    trainer = Trainer(CVPN(_SMALL))
    with pytest.raises(ValueError, match="non-empty"):
        trainer.train_step([])


def test_save_resume_round_trip(tmp_path):
    """Resume reconstructs identical weights, the generation, and the TrainerConfig."""
    torch.manual_seed(0)
    trainer = Trainer(CVPN(_SMALL), TrainerConfig(learning_rate=5e-3, batch_size=4))
    batch = [_fake_tuple([5, 6, 7], value=0.2) for _ in range(4)]
    for _ in range(5):
        trainer.train_step(batch)

    path = tmp_path / "ckpt.pt"
    trainer.save(path, generation=4, rng_state={"loop_rng": [1, 2, 3]})

    resumed, generation, rng_state = Trainer.resume(path)
    assert generation == 4
    assert rng_state == {"loop_rng": [1, 2, 3]}
    assert resumed.config.learning_rate == pytest.approx(5e-3)
    for k, v in trainer.net.state_dict().items():
        assert torch.equal(v, resumed.net.state_dict()[k])


# ---------------------------------------------------------------------------
# Gap 4: _densify_policy — direct unit test with edge-case indices
# ---------------------------------------------------------------------------


def test_densify_policy_boundary_indices():
    """Index 0 and A-1 (last action) are placed correctly in the dense tensor."""
    t0 = _fake_tuple([0, A - 1], value=0.0)
    dense = _densify_policy([t0], device="cpu")
    assert dense.shape == (1, A)
    assert dense[0, 0] == pytest.approx(0.5)
    assert dense[0, A - 1] == pytest.approx(0.5)
    # All other entries must be zero.
    mask = torch.ones(A, dtype=torch.bool)
    mask[[0, A - 1]] = False
    assert (dense[0, mask] == 0).all()


def test_densify_policy_single_action():
    """A policy with a single legal action gets probability 1.0 at that index."""
    t = TrainingTuple(
        beta=_fake_bundle([5]),
        value=0.0,
        policy=SparsePolicy(indices=(5,), probs=(1.0,)),
        z=0.0,
        meta=TupleMeta(generation=0, game_id=0, decision_idx=0, phase="move", side="p1"),
    )
    dense = _densify_policy([t], device="cpu")
    assert dense[0, 5] == pytest.approx(1.0)
    assert dense[0].sum() == pytest.approx(1.0)


def test_densify_policy_row_sums_to_one():
    """Each row of the densified tensor should sum to 1.0 for a valid policy."""
    batch = [_fake_tuple([5, 6, 7], value=0.0) for _ in range(4)]
    dense = _densify_policy(batch, device="cpu")
    for i in range(4):
        assert dense[i].sum() == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Gap 5: _densify_policy with empty SparsePolicy
# ---------------------------------------------------------------------------


def test_densify_policy_empty_policy():
    """An empty SparsePolicy produces an all-zero row (the guard fires)."""
    t = TrainingTuple(
        beta=_fake_bundle([5, 6]),
        value=0.0,
        policy=SparsePolicy(indices=(), probs=()),
        z=0.0,
        meta=TupleMeta(generation=0, game_id=0, decision_idx=0, phase="move", side="p1"),
    )
    dense = _densify_policy([t], device="cpu")
    assert dense[0].sum() == 0.0


# ---------------------------------------------------------------------------
# Gap 6: train_step with batch size = 1
# ---------------------------------------------------------------------------


def test_train_step_batch_size_one():
    """A single-tuple batch produces finite losses without shape errors."""
    torch.manual_seed(0)
    trainer = Trainer(CVPN(_SMALL), TrainerConfig(batch_size=1))
    batch = [_fake_tuple([5, 6, 7], value=0.3)]
    log = trainer.train_step(batch)
    assert math.isfinite(log.total_loss)
    assert math.isfinite(log.value_loss)
    assert math.isfinite(log.policy_loss)


# ---------------------------------------------------------------------------
# Gap 7: train_step with weight_decay > 0
# ---------------------------------------------------------------------------


def test_train_step_with_weight_decay_is_finite_and_learns():
    """With weight_decay=0.01 the optimizer applies L2 and the loss still decreases."""
    torch.manual_seed(0)
    trainer = Trainer(CVPN(_SMALL), TrainerConfig(learning_rate=1e-2, weight_decay=0.01, batch_size=4))
    batch = [_fake_tuple([5, 6, 7], value=0.3) for _ in range(4)]

    first = trainer.train_step(batch).total_loss
    for _ in range(40):
        last = trainer.train_step(batch).total_loss
    assert math.isfinite(last)
    assert last < first


# ---------------------------------------------------------------------------
# Gap 8: train_step with value_weight=0 (policy-only mode)
# ---------------------------------------------------------------------------


def test_policy_only_mode():
    """With value_weight=0, total_loss == policy_loss and the value head does not learn."""
    torch.manual_seed(0)
    trainer = Trainer(CVPN(_SMALL), TrainerConfig(learning_rate=1e-2, value_weight=0.0, policy_weight=1.0, batch_size=4))
    batch = [_fake_tuple([5, 6, 7], value=0.8) for _ in range(4)]

    log = trainer.train_step(batch)
    assert log.total_loss == pytest.approx(log.policy_loss)

    # After many steps, value loss should NOT collapse (value head is not being trained).
    for _ in range(100):
        log = trainer.train_step(batch)
    assert log.value_loss > 0.1


# ---------------------------------------------------------------------------
# Gap 9: TrainerConfig Pydantic validation rejects illegal values
# ---------------------------------------------------------------------------


def test_trainer_config_rejects_negative_lr():
    with pytest.raises(ValidationError):
        TrainerConfig(learning_rate=-1)


def test_trainer_config_rejects_zero_batch_size():
    with pytest.raises(ValidationError):
        TrainerConfig(batch_size=0)


def test_trainer_config_rejects_negative_value_weight():
    with pytest.raises(ValidationError):
        TrainerConfig(value_weight=-1)


def test_trainer_config_rejects_negative_weight_decay():
    with pytest.raises(ValidationError):
        TrainerConfig(weight_decay=-0.01)


# ---------------------------------------------------------------------------
# Gap 10: Trainer.resume from a lean (no-optimizer) checkpoint
# ---------------------------------------------------------------------------


def test_resume_from_lean_checkpoint(tmp_path):
    """Resuming from a weights-only checkpoint (no optimizer state) should succeed
    with a freshly initialized optimizer."""
    torch.manual_seed(0)
    net = CVPN(_SMALL)
    path = tmp_path / "lean.pt"
    # Save without optimizer or trainer_config.
    save_checkpoint(path, net=net, generation=3)

    trainer, gen, rng = Trainer.resume(path)
    assert gen == 3
    assert rng is None
    # The trainer should have a working optimizer (freshly initialized).
    assert isinstance(trainer.optimizer, torch.optim.AdamW)
    # Should be able to train without crashing.
    batch = [_fake_tuple([5, 6, 7], value=0.2) for _ in range(2)]
    log = trainer.train_step(batch)
    assert math.isfinite(log.total_loss)


# ---------------------------------------------------------------------------
# Gap 11: Trainer.resume device override
# ---------------------------------------------------------------------------


def test_resume_device_override(tmp_path):
    """The resumed trainer's config.device matches the requested device."""
    torch.manual_seed(0)
    trainer = Trainer(CVPN(_SMALL), TrainerConfig(device="cpu"))
    path = tmp_path / "ckpt.pt"
    trainer.save(path, generation=1)

    resumed, _, _ = Trainer.resume(path, device="cpu")
    assert resumed.config.device == "cpu"
    # All parameters should be on the requested device.
    for p in resumed.net.parameters():
        assert p.device == torch.device("cpu")


# ---------------------------------------------------------------------------
# Gap 13: TrainStepLog.grad_norm is positive
# ---------------------------------------------------------------------------


def test_grad_norm_is_positive():
    """After a train_step the grad_norm should be positive (not just finite)."""
    torch.manual_seed(0)
    trainer = Trainer(CVPN(_SMALL), TrainerConfig(batch_size=4))
    batch = [_fake_tuple([5, 6, 7], value=0.2) for _ in range(4)]
    log = trainer.train_step(batch)
    assert log.grad_norm is not None
    assert log.grad_norm > 0


# ---------------------------------------------------------------------------
# Architectural observation B: adversarial / boundary inputs
# ---------------------------------------------------------------------------


def _adversarial_bundle(legal: list[int], entities: torch.Tensor, n_entities: int | None = None):
    """Build an ObsBundle with caller-controlled entity features."""
    n = n_entities or entities.shape[0]
    mask = torch.zeros(A, dtype=torch.bool)
    mask[legal] = True
    return make_obs_bundle(
        entities=entities,
        species_ids=torch.zeros(n, dtype=torch.long),
        ability_ids=torch.zeros(n, dtype=torch.long),
        item_ids=torch.zeros(n, dtype=torch.long),
        move_ids=torch.zeros(n, NUM_MOVE_SLOTS, dtype=torch.long),
        belief_weight=torch.ones(n),
        slot_id=torch.zeros(n, dtype=torch.long),
        field=torch.zeros(FIELD_FEATURE_DIM),
        sides=torch.zeros(2, SIDE_FEATURE_DIM),
        scalars=torch.zeros(SCALAR_FEATURE_DIM),
        action_mask=mask,
        padding_mask=torch.ones(n, dtype=torch.bool),
    )


def _adversarial_tuple(legal: list[int], entities: torch.Tensor) -> TrainingTuple:
    p = 1.0 / len(legal)
    policy = SparsePolicy(indices=tuple(legal), probs=tuple(p for _ in legal))
    meta = TupleMeta(generation=0, game_id=0, decision_idx=0, phase="move", side="p1")
    return TrainingTuple(
        beta=_adversarial_bundle(legal, entities),
        value=0.0, policy=policy, z=0.0, meta=meta,
    )


def test_all_zero_features_produce_finite_loss():
    """All-zero entity features (maximally uninformative input) must not NaN."""
    torch.manual_seed(0)
    trainer = Trainer(CVPN(_SMALL), TrainerConfig(batch_size=2))
    ents = torch.zeros(12, ENTITY_FEATURE_DIM)
    batch = [_adversarial_tuple([5, 6, 7], ents) for _ in range(2)]
    log = trainer.train_step(batch)
    assert math.isfinite(log.total_loss)
    for p in trainer.net.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()


def test_single_entity_produces_finite_loss():
    """N=1 (minimum-size entity sequence) must not crash or NaN."""
    torch.manual_seed(0)
    trainer = Trainer(CVPN(_SMALL), TrainerConfig(batch_size=2))
    ents = torch.randn(1, ENTITY_FEATURE_DIM)
    batch = [_adversarial_tuple([5, 6], ents) for _ in range(2)]
    log = trainer.train_step(batch)
    assert math.isfinite(log.total_loss)


def test_extreme_magnitude_features_produce_finite_loss():
    """Entity features at 1e6 scale must still produce finite losses (LayerNorm absorbs)."""
    torch.manual_seed(0)
    trainer = Trainer(CVPN(_SMALL), TrainerConfig(batch_size=2))
    ents = torch.full((12, ENTITY_FEATURE_DIM), 1e6)
    batch = [_adversarial_tuple([5, 6, 7], ents) for _ in range(2)]
    log = trainer.train_step(batch)
    assert math.isfinite(log.total_loss)


def test_mixed_entity_counts_in_batch():
    """Batch with varying N (1 and 12 entities) — collation pads correctly and loss is finite."""
    torch.manual_seed(0)
    trainer = Trainer(CVPN(_SMALL), TrainerConfig(batch_size=2))
    batch = [
        _adversarial_tuple([5, 6], torch.randn(1, ENTITY_FEATURE_DIM)),
        _adversarial_tuple([5, 6], torch.randn(12, ENTITY_FEATURE_DIM)),
    ]
    log = trainer.train_step(batch)
    assert math.isfinite(log.total_loss)


# ---------------------------------------------------------------------------
# Architectural observation C: optimizer state continuity after resume
# ---------------------------------------------------------------------------


def test_optimizer_state_continuity_after_resume(tmp_path):
    """Save/resume preserves the AdamW momentum buffers (exp_avg, exp_avg_sq, step)
    exactly, so the next gradient step uses the correct adaptive learning rate.

    We verify this by comparing the optimizer state tensors directly (not the weights
    after further training, which would diverge due to dropout stochasticity).
    """
    cfg = TrainerConfig(learning_rate=1e-2, batch_size=4)

    torch.manual_seed(99)
    batch = [_fake_tuple([5, 6, 7], value=0.3) for _ in range(4)]

    torch.manual_seed(0)
    trainer = Trainer(CVPN(_SMALL), cfg)
    for _ in range(5):
        trainer.train_step(batch)

    # Snapshot the optimizer state BEFORE saving.
    pre_save_state = trainer.optimizer.state_dict()

    ckpt_path = tmp_path / "mid.pt"
    trainer.save(ckpt_path, generation=5)
    resumed, gen, _ = Trainer.resume(ckpt_path, device="cpu")
    assert gen == 5

    post_load_state = resumed.optimizer.state_dict()

    # Every parameter's AdamW buffers (exp_avg, exp_avg_sq, step) must match exactly.
    assert len(pre_save_state["state"]) == len(post_load_state["state"])
    for param_id in pre_save_state["state"]:
        orig = pre_save_state["state"][param_id]
        rest = post_load_state["state"][param_id]
        assert orig["step"] == rest["step"], f"step mismatch for param {param_id}"
        assert torch.equal(orig["exp_avg"], rest["exp_avg"]), f"exp_avg mismatch for param {param_id}"
        assert torch.equal(orig["exp_avg_sq"], rest["exp_avg_sq"]), f"exp_avg_sq mismatch for param {param_id}"

    # Additionally: the resumed trainer can take a step without NaN/crash.
    log = resumed.train_step(batch)
    assert math.isfinite(log.total_loss)
