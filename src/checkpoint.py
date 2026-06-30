"""Checkpoint persistence — the neutral seam for saving/loading CVPN snapshots.

A *checkpoint* is a published network snapshot: the weights plus the ``CVPNConfig``
needed to rebuild the net, plus (optionally) optimizer + RNG state for exact resume.
Written by the Trainer once per generation; consumed by SelfPlay (warm-start) and
Evaluation (head-to-head play against frozen prior checkpoints).

This lives in its own module — *not* inside ``trainer`` — so ``self_play`` and
``evaluation`` can load checkpoints without importing the learner (the same neutral-
ground reasoning that put ``TrainingTuple`` in ``training_types``).

Design: ``docs/plans/trainer.md``; ``docs/architecture/repo-architecture.md`` §3.8, §4
(the checkpoint store seam that decouples Trainer from SelfPlay).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import torch

from cvpn import CVPN, CVPNConfig

if TYPE_CHECKING:
    from pathlib import Path

# Bump only when the on-disk checkpoint *payload* shape changes.
CHECKPOINT_FORMAT_VERSION = 1


class CheckpointFormatError(RuntimeError):
    """Raised by :func:`load_checkpoint` on a format-version mismatch or a corrupted file.

    Like the ReplayBuffer's schema guard, we refuse to load a mismatched file rather
    than silently reconstruct a net from a payload the current code no longer understands.
    """


@dataclass(frozen=True)
class LoadedCheckpoint:
    """The reconstructed contents of a checkpoint file.

    ``net`` is always rebuilt (from the stored ``model_config``) and weight-loaded. The
    remaining fields are populated only if they were saved: resume needs all of them;
    warm-start / Evaluation read ``net`` and ignore the rest.
    """

    net: CVPN
    generation: int
    optimizer_state_dict: dict | None
    trainer_config: dict | None
    rng_state: dict | None


def save_checkpoint(
    path: str | Path,
    *,
    net: CVPN,
    generation: int,
    optimizer: torch.optim.Optimizer | None = None,
    trainer_config: dict | None = None,
    rng_state: dict | None = None,
) -> None:
    """Persist a CVPN snapshot via ``torch.save``.

    ``model_config`` is the net's ``CVPNConfig`` flattened with ``dataclasses.asdict`` —
    a plain dict, so the file never pickles the ``CVPNConfig`` class (forward-compatible:
    reload rebuilds it via ``CVPNConfig(**d)``). ``optimizer``/``rng_state`` are saved only
    when provided, so a lean "published" checkpoint and a full "resume" checkpoint share
    one format (the extra keys are simply ``None`` when omitted).
    """
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "generation": int(generation),
        "model_config": asdict(net.config),
        "model_state_dict": net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "trainer_config": trainer_config,
        "rng_state": rng_state,
    }
    torch.save(payload, path)


def load_checkpoint(path: str | Path, *, map_location: str = "cpu") -> LoadedCheckpoint:
    """Reconstruct a checkpoint, rebuilding the CVPN from its stored config.

    Raises :class:`CheckpointFormatError` on a format-version mismatch or missing required
    keys — it never silently loads a file the current code cannot interpret.

    .. warning::
        Uses ``weights_only=False`` (pickle-based) because the payload holds Python
        objects (the config dict, RNG state). Only load files you produced yourself.
    """
    payload = torch.load(path, weights_only=False, map_location=map_location)

    saved_format = payload.get("format_version")
    if saved_format != CHECKPOINT_FORMAT_VERSION:
        raise CheckpointFormatError(f"checkpoint format_version {saved_format!r} != current {CHECKPOINT_FORMAT_VERSION}; refusing to load.")

    required = ("generation", "model_config", "model_state_dict")
    missing = [k for k in required if k not in payload]
    if missing:
        raise CheckpointFormatError(f"corrupted or incomplete checkpoint file: missing key(s) {missing}")

    net = CVPN(CVPNConfig(**payload["model_config"]))
    net.load_state_dict(payload["model_state_dict"])
    net.to(map_location)
    return LoadedCheckpoint(
        net=net,
        generation=int(payload["generation"]),
        optimizer_state_dict=payload.get("optimizer_state_dict"),
        trainer_config=payload.get("trainer_config"),
        rng_state=payload.get("rng_state"),
    )
