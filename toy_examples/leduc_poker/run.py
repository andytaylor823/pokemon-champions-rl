"""
Entry point: run the full GT-CFR self-play training loop on Leduc Hold'em.

Usage:
    python -m toy_examples.leduc_poker [options]

Examples:
    # Quick test run (~2 min)
    python -m toy_examples.leduc_poker --generations 10 --games-per-gen 20

    # Medium run (~15 min, shows convergence trend)
    python -m toy_examples.leduc_poker --generations 50 --games-per-gen 60

    # Full run (~1 hour, clear convergence)
    python -m toy_examples.leduc_poker --generations 150 --games-per-gen 100
"""

from __future__ import annotations

import argparse
import os
import random
import time

import numpy as np

from toy_examples.leduc_poker.self_play import SelfPlayTrainer, TrainingConfig
from toy_examples.leduc_poker.visualize import (
    plot_exploitability,
    plot_strategy_evolution,
    plot_tournament_heatmap,
    plot_loss_curves,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GT-CFR self-play training on Leduc Hold'em (two-bet-size variant)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Outer loop
    parser.add_argument("--generations", type=int, default=50, help="Number of training generations")
    parser.add_argument("--games-per-gen", type=int, default=60, help="Self-play games per generation")
    # Inner loop
    parser.add_argument("--search-iterations", type=int, default=50, help="CFR+ iterations per search")
    parser.add_argument("--c-puct", type=float, default=2.0, help="PUCT exploration constant")
    parser.add_argument("--expansion-interval", type=int, default=5, help="Tree expansion frequency")
    # Training
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size")
    parser.add_argument("--train-steps", type=int, default=200, help="Gradient steps per generation")
    parser.add_argument("--buffer-size", type=int, default=50000, help="Replay buffer capacity")
    # Evaluation
    parser.add_argument("--eval-interval", type=int, default=10, help="Evaluate every N generations")
    parser.add_argument("--tournament", action="store_true", help="Run tournament between checkpoints")
    parser.add_argument("--tournament-games", type=int, default=500, help="Games per tournament matchup")
    # Output
    parser.add_argument("--verbose", action="store_true", help="Print detailed generation logs")
    parser.add_argument("--no-plots", action="store_true", help="Skip visualization plots")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save plots")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        import torch
        torch.manual_seed(args.seed)

    # Configure training
    config = TrainingConfig(
        n_generations=args.generations,
        games_per_generation=args.games_per_gen,
        search_iterations=args.search_iterations,
        c_puct=args.c_puct,
        expansion_interval=args.expansion_interval,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        train_steps_per_gen=args.train_steps,
        buffer_capacity=args.buffer_size,
        eval_interval=args.eval_interval,
        verbose=args.verbose,
    )

    # Print configuration
    print("=" * 70)
    print("GT-CFR Self-Play Training on Leduc Hold'em (Two-Bet-Size Variant)")
    print("=" * 70)
    print(f"  Generations:         {config.n_generations}")
    print(f"  Games/gen:           {config.games_per_generation}")
    print(f"  Search iterations:   {config.search_iterations}")
    print(f"  PUCT constant:       {config.c_puct}")
    print(f"  Learning rate:       {config.learning_rate}")
    print(f"  Batch size:          {config.batch_size}")
    print(f"  Buffer capacity:     {config.buffer_capacity}")
    print("  Info sets:           ~7920 (3960 per player)")
    print("  Actions per node:    2-4")
    if args.seed is not None:
        print(f"  Random seed:         {args.seed}")
    print("=" * 70)
    print()

    # Run training
    start_time = time.time()
    trainer = SelfPlayTrainer(config)
    logs = trainer.train()
    elapsed = time.time() - start_time

    # Summary
    print()
    print("=" * 70)
    print(f"Training complete in {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print("=" * 70)

    # Final exploitability
    final_expl = [log.exploitability for log in logs if log.exploitability is not None]
    if final_expl:
        print(f"  Initial exploitability:  {final_expl[0]:.4f}")
        print(f"  Final exploitability:    {final_expl[-1]:.4f}")
        print(f"  Reduction:               {final_expl[0] - final_expl[-1]:.4f}")
        print("  Nash equilibrium target: 0.0000")

    # Final strategy at key info sets
    print()
    print("Final learned strategy (selected info sets):")
    print("-" * 60)
    final_strategy = trainer.extract_strategy()
    key_info_sets = [
        "K:?:", "Q:?:", "J:?:",
        "K:K:check,check", "Q:Q:check,check", "J:J:check,check",
        "K:?:bet_big", "Q:?:bet_big", "J:?:bet_big",
    ]
    for info_key in key_info_sets:
        if info_key in final_strategy:
            probs = final_strategy[info_key]
            actions_str = ", ".join(f"{a}={p:.3f}" for a, p in probs.items())
            print(f"  {info_key:30s}  {actions_str}")

    # Tournament
    if args.tournament and len(trainer.checkpoints) >= 2:
        print()
        print("Running generation tournament...")
        win_matrix = trainer.play_tournament(n_games=args.tournament_games)
        gens = [cp["generation"] for cp in trainer.checkpoints]
        print("Win rate matrix (row = P1, col = P2):")
        header = "     " + " ".join(f"G{g:3d}" for g in gens)
        print(header)
        for i, g in enumerate(gens):
            row = f"G{g:3d} " + " ".join(f"{win_matrix[i, j]:.2f}" for j in range(len(gens)))
            print(row)

    # Visualization
    if not args.no_plots:
        output_dir = args.output_dir or "toy_examples/leduc_poker/output"
        os.makedirs(output_dir, exist_ok=True)

        print(f"\nSaving plots to {output_dir}/")

        plot_exploitability(logs, save_path=os.path.join(output_dir, "exploitability.png"))
        plot_strategy_evolution(logs, save_path=os.path.join(output_dir, "strategy_evolution.png"))
        plot_loss_curves(logs, save_path=os.path.join(output_dir, "loss_curves.png"))

        if args.tournament and len(trainer.checkpoints) >= 2:
            gens = [cp["generation"] for cp in trainer.checkpoints]
            plot_tournament_heatmap(
                win_matrix, gens,
                save_path=os.path.join(output_dir, "tournament_heatmap.png"),
            )

        print("Done! Check the output directory for plots.")


if __name__ == "__main__":
    main()
