"""
Entry point: run the full GT-CFR self-play training loop on Kuhn Poker.

Usage:
    python -m toy_examples.kuhn_poker.run [options]

Examples:
    # Quick run (few minutes)
    python -m toy_examples.kuhn_poker.run --generations 20 --games-per-gen 100

    # Full run (shows clear convergence)
    python -m toy_examples.kuhn_poker.run --generations 80 --games-per-gen 300

    # Verbose with tournament
    python -m toy_examples.kuhn_poker.run --generations 50 --tournament --verbose
"""

from __future__ import annotations

import argparse
import os
import random
import time

import numpy as np

from toy_examples.kuhn_poker.self_play import SelfPlayTrainer, TrainingConfig
from toy_examples.kuhn_poker.visualize import (
    plot_exploitability,
    plot_strategy_evolution,
    plot_tournament_heatmap,
    plot_loss_curves,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GT-CFR self-play training on Kuhn Poker",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Outer loop
    parser.add_argument("--generations", type=int, default=50, help="Number of training generations")
    parser.add_argument("--games-per-gen", type=int, default=200, help="Self-play games per generation")
    # Inner loop
    parser.add_argument("--search-iterations", type=int, default=100, help="CFR+ iterations per search")
    parser.add_argument("--c-puct", type=float, default=2.0, help="PUCT exploration constant")
    parser.add_argument("--expansion-interval", type=int, default=10, help="Tree expansion frequency")
    # Training
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    parser.add_argument("--train-steps", type=int, default=100, help="Gradient steps per generation")
    parser.add_argument("--buffer-size", type=int, default=10000, help="Replay buffer capacity")
    # Evaluation
    parser.add_argument("--eval-interval", type=int, default=5, help="Evaluate every N generations")
    parser.add_argument("--tournament", action="store_true", help="Run tournament between checkpoints")
    parser.add_argument("--tournament-games", type=int, default=2000, help="Games per tournament matchup")
    # Output
    parser.add_argument("--verbose", action="store_true", help="Print detailed generation logs")
    parser.add_argument("--no-plots", action="store_true", help="Skip visualization plots")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save plots and logs")
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

    # Configure the training
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
    print("GT-CFR Self-Play Training on Kuhn Poker")
    print("=" * 70)
    print(f"  Generations:        {config.n_generations}")
    print(f"  Games/gen:          {config.games_per_generation}")
    print(f"  Search iterations:  {config.search_iterations}")
    print(f"  PUCT constant:      {config.c_puct}")
    print(f"  Learning rate:      {config.learning_rate}")
    print(f"  Batch size:         {config.batch_size}")
    print(f"  Buffer capacity:    {config.buffer_capacity}")
    if args.seed is not None:
        print(f"  Random seed:        {args.seed}")
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
    print(f"Training complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print("=" * 70)

    # Final exploitability
    final_expl = [log.exploitability for log in logs if log.exploitability is not None]
    if final_expl:
        print(f"  Initial exploitability:  {final_expl[0]:.4f}")
        print(f"  Final exploitability:    {final_expl[-1]:.4f}")
        print(f"  Reduction:               {final_expl[0] - final_expl[-1]:.4f}")
        print("  Nash equilibrium target: 0.0000")
        print(f"  Kuhn game value (P1):    {-1/18:.4f}")

    # Final strategy
    print()
    print("Final learned strategy (key info sets):")
    print("-" * 50)
    final_strategy = trainer.extract_strategy()
    key_info_sets = ["K:", "Q:", "J:", "K:check,bet", "Q:check,bet", "J:check,bet",
                     "K:bet", "K:check", "Q:bet", "Q:check", "J:bet", "J:check"]
    for info_key in key_info_sets:
        if info_key in final_strategy:
            probs = final_strategy[info_key]
            actions_str = ", ".join(f"{a}={p:.3f}" for a, p in probs.items())
            print(f"  {info_key:15s}  {actions_str}")

    # Known Nash reference
    print()
    print("Nash equilibrium reference (alpha=1/3):")
    print("-" * 50)
    nash_ref = {
        "K:": "bet=1.000",
        "Q:": "bet=0.000, check=1.000",
        "J:": "bet=0.333, check=0.667",
        "K:bet": "call=1.000",
        "Q:bet": "call=0.333, fold=0.667",
        "J:bet": "call=0.000, fold=1.000",
        "K:check": "bet=1.000",
        "Q:check": "bet=0.000, check=1.000",
        "J:check": "bet=0.333, check=0.667",
    }
    for info_key, ref in nash_ref.items():
        print(f"  {info_key:15s}  {ref}")

    # Tournament
    if args.tournament and len(trainer.checkpoints) >= 2:
        print()
        print("Running generation tournament...")
        win_matrix = trainer.play_tournament(n_games=args.tournament_games)
        print("Win rate matrix (row = P1, col = P2):")
        gens = [cp["generation"] for cp in trainer.checkpoints]
        # Print header
        header = "     " + " ".join(f"G{g:3d}" for g in gens)
        print(header)
        for i, g in enumerate(gens):
            row = f"G{g:3d} " + " ".join(f"{win_matrix[i,j]:.2f}" for j in range(len(gens)))
            print(row)

    # Visualization
    if not args.no_plots:
        output_dir = args.output_dir or "toy_examples/kuhn_poker/output"
        os.makedirs(output_dir, exist_ok=True)

        print(f"\nSaving plots to {output_dir}/")

        # Exploitability curve
        plot_exploitability(logs, save_path=os.path.join(output_dir, "exploitability.png"))

        # Strategy evolution
        plot_strategy_evolution(logs, save_path=os.path.join(output_dir, "strategy_evolution.png"))

        # Loss curves
        plot_loss_curves(logs, save_path=os.path.join(output_dir, "loss_curves.png"))

        # Tournament heatmap (if run)
        if args.tournament and len(trainer.checkpoints) >= 2:
            gens = [cp["generation"] for cp in trainer.checkpoints]
            plot_tournament_heatmap(
                win_matrix, gens,
                save_path=os.path.join(output_dir, "tournament_heatmap.png"),
            )

        print("Done! Check the output directory for plots.")


if __name__ == "__main__":
    main()
