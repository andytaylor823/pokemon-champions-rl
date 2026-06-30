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
from trainer import Trainer, TrainerConfig
from training_types import SparsePolicy, TrainingTuple, TupleMeta

_SMALL = CVPNConfig(d_model=32, n_heads=2, n_layers=1, ffn_mult=2)


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
