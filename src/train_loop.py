"""Outer-loop training driver — the thin orchestrator.

Runs the generation loop: for each generation, play ``games_per_generation`` self-play
games into the ReplayBuffer, gate on the warmup threshold, do ``train_steps_per_generation``
gradient steps via the Trainer, and (every ``checkpoint_interval`` generations) publish a
checkpoint paired with a buffer snapshot at the same generation.

The Trainer is the *learner*; this driver is the *conductor*. It is deliberately thin so
the later multi-process generation/training split (``replay-buffer.md`` §7) can wrap it
without touching the Trainer. One *generation* = G games → M gradient steps → one checkpoint.

Design: ``docs/plans/trainer.md``; ``docs/plans/self-play.md`` §7; ``docs/plans/replay-buffer.md`` §5.
"""

from __future__ import annotations

import logging
import random
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

import self_play
from replay_buffer import ReplayBuffer
from self_play import SelfPlayConfig
from trainer import Trainer, TrainerConfig

if TYPE_CHECKING:
    from cvpn import CVPN
    from self_play import MatchupSource
    from sim_client import SimClient
    from training_types import TrainingTuple

logger = logging.getLogger(__name__)


class TrainLoopConfig(BaseModel):
    """Loop knobs (the learner knobs live in ``TrainerConfig``).

    Composes a ``TrainerConfig`` and a ``SelfPlayConfig``; the driver overrides the
    self-play config's ``num_games``/``generation``/``master_seed`` per generation so each
    generation plays a fresh set of games.
    """

    # SelfPlayConfig / SearchConfig are frozen dataclasses, not pydantic models.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    games_per_generation: int = Field(default=50, gt=0, description="G: self-play games played into the buffer each generation.")
    train_steps_per_generation: int = Field(default=200, gt=0, description="M: gradient steps each generation (once warmed up).")
    warmup_threshold: int = Field(default=1000, ge=0, description="Minimum buffer fill before the first gradient step (owned here, not by the buffer).")
    checkpoint_interval: int = Field(default=1, gt=0, description="Generations between checkpoint + paired buffer snapshots.")
    n_generations: int = Field(default=100, gt=0, description="Total outer-loop generations.")
    buffer_capacity: int = Field(default=100_000, gt=0, description="ReplayBuffer capacity in tuples (a fresh buffer is built per stage).")
    checkpoint_dir: str = Field(description="Directory for gen_NNNN.pt checkpoints and paired buffer snapshots.")
    master_seed: int = Field(default=42, description="Seeds the per-generation game seeds + the buffer sampler.")
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    self_play: SelfPlayConfig = Field(default_factory=SelfPlayConfig)


class GenerationLog(BaseModel):
    """Per-generation summary (mean losses, buffer state, diagnostics)."""

    generation: int
    n_tuples_generated: int
    buffer_size: int
    trained: bool = False
    value_loss: float | None = None
    policy_loss: float | None = None
    total_loss: float | None = None
    # Diagnostic only: correlation between the side label z and the search value target.
    # z is never a training target in Phase 1 (vibes 8.4); this is the door to a future blend.
    z_value_corr: float | None = None


def _z_value_corr(tuples: list[TrainingTuple]) -> float | None:
    """Pearson correlation between z and the search value across a generation's tuples.

    Returns ``None`` when undefined (fewer than 2 tuples, or zero variance in either).
    """
    if len(tuples) < 2:
        return None
    z = np.array([t.z for t in tuples], dtype=np.float64)
    v = np.array([t.value for t in tuples], dtype=np.float64)
    if z.std() == 0 or v.std() == 0:
        return None
    return float(np.corrcoef(z, v)[0, 1])


def run_training(
    net: CVPN,
    matchup_source: MatchupSource,
    sim: SimClient,
    config: TrainLoopConfig,
) -> list[GenerationLog]:
    """Drive a curriculum stage end to end; return the per-generation logs.

    A fresh ReplayBuffer is built here (no cross-stage bleed — ``replay-buffer.md`` §12).
    The caller owns the ``net`` (fresh for from-scratch, or a loaded checkpoint for
    warm-start) and the ``sim`` (its Node worker lifecycle).
    """
    trainer = Trainer(net, config.trainer)
    buffer = ReplayBuffer(config.buffer_capacity, seed=config.master_seed)
    ckpt_dir = Path(config.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    loop_rng = random.Random(config.master_seed)
    logs: list[GenerationLog] = []

    for gen in range(1, config.n_generations + 1):
        # --- Generate G games (fresh per-generation seed so generations differ) ---
        gen_seed = loop_rng.randint(0, 2**32 - 1)
        sp_config = replace(
            config.self_play,
            num_games=config.games_per_generation,
            generation=gen,
            master_seed=gen_seed,
        )
        new_tuples = list(self_play.run(net, matchup_source, sim, sp_config))
        buffer.add(new_tuples)

        log = GenerationLog(
            generation=gen,
            n_tuples_generated=len(new_tuples),
            buffer_size=len(buffer),
            z_value_corr=_z_value_corr(new_tuples),
        )

        # --- Train if the buffer is warmed up and holds a full batch ---
        batch_size = config.trainer.batch_size
        if len(buffer) >= config.warmup_threshold and len(buffer) >= batch_size:
            v_sum = p_sum = t_sum = 0.0
            for _ in range(config.train_steps_per_generation):
                step = trainer.train_step(buffer.sample(batch_size))
                v_sum += step.value_loss
                p_sum += step.policy_loss
                t_sum += step.total_loss
            steps = config.train_steps_per_generation
            log.trained = True
            log.value_loss = v_sum / steps
            log.policy_loss = p_sum / steps
            log.total_loss = t_sum / steps

        # --- Publish checkpoint + paired buffer snapshot (same generation) ---
        if gen % config.checkpoint_interval == 0:
            trainer.save(ckpt_dir / f"gen_{gen:04d}.pt", generation=gen, rng_state={"loop_rng": loop_rng.getstate()})
            buffer.save(ckpt_dir / f"buffer_gen_{gen:04d}.pt")

        logs.append(log)
        logger.info(
            "Gen %d: +%d tuples, buffer=%d, trained=%s, total_loss=%s",
            gen,
            log.n_tuples_generated,
            log.buffer_size,
            log.trained,
            None if log.total_loss is None else round(log.total_loss, 4),
        )

    return logs
