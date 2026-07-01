"""Manual entrypoint for resuming an interrupted Phase-1 outer-loop run.

The sibling of ``scripts/train.py``: instead of starting a stage from scratch, this
reconstructs the Trainer (weights + AdamW state + saved config), the ReplayBuffer, and
the loop RNG from a checkpoint + its paired buffer snapshot, then continues from the next
generation. Because the optimizer and RNG are restored, the continuation picks up exactly
where the original left off rather than warm-restarting.

The two files are the pair the driver writes together each checkpointed generation:
``gen_NNNN.pt`` and ``buffer_gen_NNNN.pt``. Pass the checkpoint; the paired buffer path is
derived from it unless ``--buffer`` overrides.

Usage (from the repo root, with the venv active):

    python scripts/train_resume.py --checkpoint checkpoints/smoke/gen_0010.pt \
        --stage 0 --generations 20

Only the loop knobs (``--generations``, ``--games-per-gen``, ``--train-steps``,
``--warmup``, cadence, self-play) and ``--device`` are honored — the learner knobs
(learning rate, batch size, weight decay) and buffer capacity come from the checkpoint.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

# src-layout: make the project modules importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import curriculum  # noqa: E402
from search import SearchConfig  # noqa: E402
from self_play import SelfPlayConfig  # noqa: E402
from sim_client import SimClient  # noqa: E402
from trainer import TrainerConfig  # noqa: E402
from train_loop import TrainLoopConfig, resume_training  # noqa: E402

_STAGES = {0: curriculum.STAGE_0, 1: curriculum.STAGE_1}


def _paired_buffer_path(checkpoint: Path) -> Path:
    """Derive the paired buffer snapshot for a checkpoint (``gen_NNNN.pt`` -> ``buffer_gen_NNNN.pt``)."""
    return checkpoint.parent / f"buffer_{checkpoint.name}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Resume a Phase-1 outer-loop run from a checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a gen_NNNN.pt checkpoint to resume from.")
    parser.add_argument("--buffer", type=str, default=None, help="Paired buffer snapshot (default: derived from --checkpoint).")
    parser.add_argument("--stage", type=int, default=0, choices=sorted(_STAGES), help="Validation-curriculum stage (must match the original run).")
    parser.add_argument("--generations", type=int, default=20, help="Total generations to run *to* (the loop stops at this generation).")
    parser.add_argument("--games-per-gen", type=int, default=8)
    parser.add_argument("--train-steps", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    checkpoint = Path(args.checkpoint)
    buffer_path = Path(args.buffer) if args.buffer else _paired_buffer_path(checkpoint)

    # A small search budget keeps the smoke run quick (mirrors train.py). Search settings
    # are not part of the checkpoint, so they are re-specified here.
    search_config = SearchConfig(k_actions=3, max_chance_children=2, expansion_budget=6, cfr_iters_per_expansion=8, c_puct=2.0)
    config = TrainLoopConfig(
        games_per_generation=args.games_per_gen,
        train_steps_per_generation=args.train_steps,
        warmup_threshold=args.warmup,
        n_generations=args.generations,
        checkpoint_dir=str(checkpoint.parent),
        master_seed=args.seed,
        # Only device is honored on resume; the rest of the learner config is restored
        # from the checkpoint.
        trainer=TrainerConfig(device=args.device),
        self_play=SelfPlayConfig(temperature=1.0, max_decisions=200, search_config=search_config),
    )

    sim = SimClient(inherit_stderr=True)
    try:
        logs = resume_training(_STAGES[args.stage], sim, config, checkpoint_path=checkpoint, buffer_path=buffer_path)
    finally:
        sim.close()

    if not logs:
        print(f"Nothing to do: checkpoint is already at or past generation {args.generations}.")
        return

    print("\n=== Resumed training summary ===")
    for log in logs:
        if log.trained:
            print(f"Gen {log.generation:3d}: total={log.total_loss:.4f} (v={log.value_loss:.4f} p={log.policy_loss:.4f}) buffer={log.buffer_size} z~v={log.z_value_corr}")
        else:
            print(f"Gen {log.generation:3d}: warming up (buffer={log.buffer_size})")


if __name__ == "__main__":
    main()
