"""Trainer — Phase-1 outer-loop learner.

Owns the CVPN + optimizer and does exactly one thing: turn a minibatch of
``TrainingTuple``s into a gradient step. It collates the stored ``ObsBundle``s,
densifies the sparse average strategy (sigma-bar), stacks the search-refined value
targets, computes the combined loss, and steps AdamW. The outer-loop *loop* (generate ->
fill buffer -> train -> checkpoint) lives in the separate driver (``train_loop.py``);
keeping the Trainer a stateful learner -- not the loop -- makes it unit-testable on a fake
batch and lets the later concurrent generation/training split wrap the driver without
touching the learner.

Loss:  ``L = value_weight * MSE(v_hat, v_search) + policy_weight * CE(pi_hat, sigma-bar)``
(+ AdamW decoupled weight decay for the architecture's ``lambda * ||theta||^2``, default
off). The value target is the *search-refined* value (bootstrapping), never the game
result ``z`` (``docs/plans/self-play.md`` section 4, vibes 8.4).

Design: ``docs/plans/trainer.md``; ``docs/architecture/repo-architecture.md`` §3.8.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from pydantic import BaseModel, Field

from action_space import A
from checkpoint import load_checkpoint, save_checkpoint
from obs_bundle import collate_obs_bundles

if TYPE_CHECKING:
    from pathlib import Path

    from cvpn import CVPN
    from training_types import TrainingTuple


class TrainerConfig(BaseModel):
    """Learner knobs (the loop knobs live in ``TrainLoopConfig``)."""

    learning_rate: float = Field(default=1e-3, gt=0, description="AdamW learning rate.")
    weight_decay: float = Field(default=0.0, ge=0, description="AdamW decoupled L2 (the architecture's lambda*||theta||^2). Default off; turn on only if a stage overfits.")
    batch_size: int = Field(default=64, gt=0, description="Minibatch size per gradient step (the driver passes buffer.sample(batch_size)).")
    value_weight: float = Field(default=1.0, ge=0, description="Weight on the MSE value loss.")
    policy_weight: float = Field(default=1.0, ge=0, description="Weight on the cross-entropy policy loss.")
    device: str = Field(default="cpu", description="Torch device for the net and batches (cpu / mps / cuda).")


class TrainStepLog(BaseModel):
    """Losses from one gradient step (mean over the minibatch)."""

    value_loss: float
    policy_loss: float
    total_loss: float
    grad_norm: float | None = None


def _densify_policy(tuples: list[TrainingTuple], device: str) -> torch.Tensor:
    """Scatter each tuple's sparse sigma-bar into a dense ``[n, A]`` target (0 on unlisted actions).

    The unlisted actions are exactly those sigma-bar assigns no mass -- including every
    illegal action -- so the dense target is 0 on all illegal entries, which is what makes
    the masked cross-entropy (see :meth:`Trainer.train_step`) provably signal-free there.
    """
    dense = torch.zeros(len(tuples), A, dtype=torch.float32, device=device)
    for i, t in enumerate(tuples):
        if t.policy.indices:
            dense[i, list(t.policy.indices)] = torch.tensor(t.policy.probs, dtype=torch.float32, device=device)
    return dense


class Trainer:
    """Stateful learner: holds the CVPN + AdamW optimizer; does one gradient step at a time."""

    def __init__(self, net: CVPN, config: TrainerConfig | None = None) -> None:
        self.config = config or TrainerConfig()
        self.net = net.to(self.config.device)
        self.optimizer = torch.optim.AdamW(
            self.net.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def train_step(self, tuples: list[TrainingTuple]) -> TrainStepLog:
        """Run one gradient step on a minibatch of training tuples.

        Owns all batching (``replay-buffer.md`` section 4): collate beta via
        ``collate_obs_bundles``, densify sigma-bar -> ``[n, A]``, stack the search-refined
        value targets -> ``[n]``, move to device, forward, compute loss, backward, step.
        """
        if not tuples:
            raise ValueError("train_step requires a non-empty minibatch")

        device = self.config.device
        batch = collate_obs_bundles([t.beta for t in tuples]).to(device)
        value_target = torch.tensor([t.value for t in tuples], dtype=torch.float32, device=device)
        sigma_bar = _densify_policy(tuples, device)

        self.net.train()
        policy_logits, v_hat = self.net(batch)  # [n, A] (illegal = -inf), [n] in [-1, 1]

        # Value loss: MSE toward the search-refined value (bootstrapping), never z.
        value_loss = nn.functional.mse_loss(v_hat, value_target)

        # Policy loss: masked soft-target cross-entropy.
        # log_softmax leaves illegal entries at -inf; replace those log-probs with 0
        # BEFORE the multiply so the 0*(-inf)=NaN float artifact can't fire. sigma-bar is
        # already 0 on illegal actions, so 0*0=0 -- the fill value is arbitrary and the
        # gradient is exactly (softmax - sigma-bar) on legal logits and 0 on illegal ones
        # (provably signal-free). Every emitted tuple has >=2 legal actions (forced decisions
        # are skipped), so the log_softmax normalizer is always finite. Reuses the same
        # action_mask the net used.
        logp = torch.log_softmax(policy_logits, dim=-1)
        logp = logp.masked_fill(~batch["action_mask"], 0.0)
        policy_loss = -(sigma_bar * logp).sum(dim=-1).mean()

        total = self.config.value_weight * value_loss + self.config.policy_weight * policy_loss

        self.optimizer.zero_grad()
        total.backward()
        # Measure the gradient norm without clipping (max_norm=inf is a no-op clip that
        # still returns the total norm) — a cheap NaN/explosion diagnostic for the logs.
        grad_norm = torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=float("inf"))
        self.optimizer.step()

        return TrainStepLog(
            value_loss=value_loss.item(),
            policy_loss=policy_loss.item(),
            total_loss=total.item(),
            grad_norm=float(grad_norm),
        )

    def save(self, path: str | Path, generation: int, *, rng_state: dict | None = None) -> None:
        """Publish a checkpoint (weights + config + optimizer + optional RNG state)."""
        save_checkpoint(
            path,
            net=self.net,
            generation=generation,
            optimizer=self.optimizer,
            trainer_config=self.config.model_dump(),
            rng_state=rng_state,
        )

    @classmethod
    def resume(cls, path: str | Path, *, device: str = "cpu") -> tuple[Trainer, int, dict | None]:
        """Reconstruct a Trainer from a checkpoint for exact mid-stage resume.

        Restores the net weights, the saved ``TrainerConfig`` (with ``device`` overridden
        to the requested one), and the optimizer state. Returns ``(trainer, generation,
        rng_state)`` — the driver restores its own RNG from ``rng_state``.
        """
        loaded = load_checkpoint(path, map_location=device)
        config = TrainerConfig(**loaded.trainer_config) if loaded.trainer_config else TrainerConfig()
        config = config.model_copy(update={"device": device})
        trainer = cls(loaded.net, config)
        if loaded.optimizer_state_dict is not None:
            trainer.optimizer.load_state_dict(loaded.optimizer_state_dict)
        return trainer, loaded.generation, loaded.rng_state
