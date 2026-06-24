"""Strategy extraction from the GT-CFR search tree.

Converts raw strategy sums and regret tables into the average strategy
(sigma-bar) and full [A] policy-target vectors for training.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from action_space import A

if TYPE_CHECKING:
    from search.types import TurnNode


def extract_average_strategy(node: TurnNode, side: str) -> dict[int, float]:
    """Extract the average strategy sigma-bar for one side at a node.

    Returns a dict mapping canonical action indices to probabilities.
    Both sides are guaranteed to have a valid InfoSet after expansion.
    """
    info = node.info[side]
    if len(info.actions) == 0:
        return {}

    total = info.strategy_sum.sum()

    if total > 0:
        probs = info.strategy_sum / total
    else:
        # No iterations ran — fall back to uniform over top-k
        k = len(info.actions)
        probs = np.ones(k) / k

    result: dict[int, float] = {}
    for idx, action in enumerate(info.actions):
        if probs[idx] > 0:
            result[action] = float(probs[idx])
    return result


def build_policy_target(node: TurnNode, side: str) -> np.ndarray:
    """Build the full [A] policy target vector from the average strategy.

    Mass on the top-k support, zeros elsewhere.
    """
    target = np.zeros(A, dtype=np.float32)
    strategy = extract_average_strategy(node, side)
    for action_idx, prob in strategy.items():
        target[action_idx] = prob
    return target
