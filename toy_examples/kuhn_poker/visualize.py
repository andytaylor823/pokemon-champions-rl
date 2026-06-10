"""
Visualization for GT-CFR training on Kuhn Poker.

Produces plots showing:
  - Exploitability convergence over generations
  - Strategy evolution at key info sets
  - Generation tournament heatmap (later gens beat earlier ones)
  - Training loss curves
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from toy_examples.kuhn_poker.self_play import GenerationLog


def plot_exploitability(
    logs: list[GenerationLog],
    save_path: str | None = None,
    show: bool = False,
) -> None:
    """
    Plot exploitability over training generations (log scale y-axis).
    Shows the agent progressing from random play toward Nash equilibrium.
    """
    import matplotlib.pyplot as plt

    # Extract generations with exploitability measurements
    gens = [log.generation for log in logs if log.exploitability is not None]
    expls = [log.exploitability for log in logs if log.exploitability is not None]

    if not gens:
        print("No exploitability data to plot.")
        return

    _, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(gens, expls, "b-o", markersize=4, linewidth=1.5, label="GT-CFR agent")

    # Reference lines
    ax.axhline(y=0.0, color="green", linestyle="--", alpha=0.7, label="Nash (exploitability=0)")
    if expls[0] > 0:
        ax.axhline(y=expls[0], color="red", linestyle=":", alpha=0.5, label=f"Initial ({expls[0]:.3f})")

    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Exploitability (log scale)", fontsize=12)
    ax.set_title("GT-CFR Training: Exploitability Convergence", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    if show:
        plt.show()
    plt.close()


def plot_strategy_evolution(
    logs: list[GenerationLog],
    save_path: str | None = None,
    show: bool = False,
) -> None:
    """
    Plot how the strategy at key info sets evolves over training.
    Shows convergence toward Nash equilibrium action frequencies.
    """
    import matplotlib.pyplot as plt

    # Key info sets to track with their Nash reference values
    # Format: (info_set_key, action_to_plot, nash_value, label)
    tracked = [
        ("K:", "bet", 1.0, "P1 King: bet freq"),
        ("J:", "bet", 1.0 / 3.0, "P1 Jack: bet (bluff) freq"),
        ("J:bet", "call", 0.0, "P2 Jack facing bet: call freq"),
        ("J:check", "bet", 1.0 / 3.0, "P2 Jack after check: bet freq"),
    ]

    # Collect data from strategy snapshots
    gens_with_data = [log.generation for log in logs if log.strategy_snapshot]
    if not gens_with_data:
        print("No strategy snapshot data to plot.")
        return

    _, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for idx, (info_key, action, nash_val, label) in enumerate(tracked):
        ax = axes[idx]
        gens = []
        values = []

        for log in logs:
            if log.strategy_snapshot and info_key in log.strategy_snapshot:
                probs = log.strategy_snapshot[info_key]
                if action in probs:
                    gens.append(log.generation)
                    values.append(probs[action])

        if gens:
            ax.plot(gens, values, "b-o", markersize=3, linewidth=1.5, label="Learned")
            ax.axhline(y=nash_val, color="green", linestyle="--", linewidth=2,
                       label=f"Nash ({nash_val:.3f})")

        ax.set_xlabel("Generation")
        ax.set_ylabel(f"P({action})")
        ax.set_title(label, fontsize=11)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Strategy Evolution at Key Info Sets", fontsize=14, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    if show:
        plt.show()
    plt.close()


def plot_tournament_heatmap(
    win_matrix: np.ndarray,
    generations: list[int],
    save_path: str | None = None,
    show: bool = False,
) -> None:
    """
    Plot a heatmap of win rates between generation checkpoints.
    Later generations should dominate earlier ones (upper-left > 0.5, lower-right < 0.5).
    """
    import matplotlib.pyplot as plt

    n = len(generations)
    _, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(win_matrix, cmap="RdYlGn", vmin=0.3, vmax=0.7, aspect="auto")

    # Label axes with generation numbers
    ax.set_xticks(range(n))
    ax.set_xticklabels([f"G{g}" for g in generations], rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"G{g}" for g in generations])

    # Add text annotations
    for i in range(n):
        for j in range(n):
            color = "white" if abs(win_matrix[i, j] - 0.5) > 0.15 else "black"
            ax.text(j, i, f"{win_matrix[i, j]:.2f}", ha="center", va="center",
                    color=color, fontsize=8)

    ax.set_xlabel("Opponent (P2)", fontsize=12)
    ax.set_ylabel("Agent (P1)", fontsize=12)
    ax.set_title("Generation Tournament: Win Rate Heatmap\n(row plays P1 vs column as P2)",
                 fontsize=12)

    plt.colorbar(im, ax=ax, label="Win Rate")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    if show:
        plt.show()
    plt.close()


def plot_loss_curves(
    logs: list[GenerationLog],
    save_path: str | None = None,
    show: bool = False,
) -> None:
    """Plot training loss (value + policy) over generations."""
    import matplotlib.pyplot as plt

    gens = [log.generation for log in logs]
    value_losses = [log.value_loss for log in logs]
    policy_losses = [log.policy_loss for log in logs]
    total_losses = [log.total_loss for log in logs]

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Combined loss
    ax1.plot(gens, total_losses, "k-", linewidth=1.5, label="Total")
    ax1.plot(gens, value_losses, "r-", linewidth=1, alpha=0.7, label="Value (MSE)")
    ax1.plot(gens, policy_losses, "b-", linewidth=1, alpha=0.7, label="Policy (CE)")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Smoothed (rolling average)
    window = max(1, len(gens) // 10)
    if len(gens) > window:
        smoothed_total = np.convolve(total_losses, np.ones(window) / window, mode="valid")
        smoothed_gens = gens[window - 1:]
        ax2.plot(smoothed_gens, smoothed_total, "k-", linewidth=2, label=f"Smoothed (w={window})")
        ax2.plot(gens, total_losses, "k-", linewidth=0.5, alpha=0.3, label="Raw")
    else:
        ax2.plot(gens, total_losses, "k-", linewidth=1.5)
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Loss")
    ax2.set_title("Training Loss (Smoothed)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    if show:
        plt.show()
    plt.close()
