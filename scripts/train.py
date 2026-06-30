"""Manual smoke / training entrypoint for the Phase-1 outer loop.

Wires a fresh CVPN + a real SimClient + a validation-curriculum stage into the
``train_loop`` driver, then prints per-generation losses. This is the empirical
feasibility check (`docs/plans/self-play.md` §15): watch policy/value loss fall and
the favored (Fire) side start winning on Stage 0. Full convergence metrics belong to
the future Evaluation module.

Usage (from the repo root, with the venv active):

    python scripts/train.py --stage 0 --generations 20 --games-per-gen 8 \
        --train-steps 50 --warmup 50 --batch-size 32 --buffer-capacity 5000

Defaults are sized for a quick smoke run, not a real training run.
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
from cvpn import CVPN  # noqa: E402
from search import SearchConfig  # noqa: E402
from self_play import SelfPlayConfig  # noqa: E402
from sim_client import SimClient  # noqa: E402
from train_loop import TrainLoopConfig, TrainerConfig, run_training  # noqa: E402

_STAGES = {0: curriculum.STAGE_0, 1: curriculum.STAGE_1}


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase-1 outer-loop smoke trainer.")
    parser.add_argument("--stage", type=int, default=0, choices=sorted(_STAGES), help="Validation-curriculum stage.")
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--games-per-gen", type=int, default=8)
    parser.add_argument("--train-steps", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--buffer-capacity", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/smoke")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    torch.manual_seed(args.seed)
    net = CVPN()

    # A small search budget keeps the smoke run quick (mirrors the integration test).
    search_config = SearchConfig(k_actions=3, max_chance_children=2, expansion_budget=6, cfr_iters_per_expansion=8, c_puct=2.0)
    config = TrainLoopConfig(
        games_per_generation=args.games_per_gen,
        train_steps_per_generation=args.train_steps,
        warmup_threshold=args.warmup,
        n_generations=args.generations,
        buffer_capacity=args.buffer_capacity,
        checkpoint_dir=args.checkpoint_dir,
        master_seed=args.seed,
        trainer=TrainerConfig(learning_rate=args.lr, batch_size=args.batch_size, device=args.device),
        self_play=SelfPlayConfig(temperature=1.0, max_decisions=200, search_config=search_config),
    )

    sim = SimClient(inherit_stderr=True)
    try:
        logs = run_training(net, _STAGES[args.stage], sim, config)
    finally:
        sim.close()

    print("\n=== Training summary ===")
    for log in logs:
        if log.trained:
            print(f"Gen {log.generation:3d}: total={log.total_loss:.4f} (v={log.value_loss:.4f} p={log.policy_loss:.4f}) buffer={log.buffer_size} z~v={log.z_value_corr}")
        else:
            print(f"Gen {log.generation:3d}: warming up (buffer={log.buffer_size})")


if __name__ == "__main__":
    main()
