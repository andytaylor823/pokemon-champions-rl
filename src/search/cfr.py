"""CFR+ regret-update traversals for the GT-CFR search tree.

Implements regret matching+ and recursive CFR+ updates over TurnNode/ChanceNode
trees. Recurses through ChanceOutcome.node references (set during PUCT expansion)
to traverse the full depth of the search tree.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from search.types import TurnNode


# ---------------------------------------------------------------------------
# CFR+ helpers
# ---------------------------------------------------------------------------


def regret_matching(cumulative_regret: np.ndarray) -> np.ndarray:
    """Derive current strategy from cumulative regrets via regret matching+.

    Returns uniform distribution if no positive regret exists.
    """
    positive = np.maximum(cumulative_regret, 0.0)
    total = positive.sum()
    if total > 0:
        return positive / total
    # Uniform fallback when all regrets are non-positive
    return np.ones_like(cumulative_regret) / len(cumulative_regret)


def cfr_update_recursive(node: TurnNode, iteration: int) -> float:
    """Recursively run CFR+ update through the tree (depth-first).

    Both sides always have valid InfoSets (guaranteed by _expand_turn_node's
    centralized no-op padding), so no conditional fallbacks are needed. For
    expanded children below ChanceNodes, recurses through outcome.node; for
    frontier leaves and terminals, uses outcome.leaf_value_p1.

    Returns p1's counterfactual value at this node.
    """
    info_p1 = node.info["p1"]
    info_p2 = node.info["p2"]

    sigma_p1 = regret_matching(info_p1.regret)
    sigma_p2 = regret_matching(info_p2.regret)

    k_p1 = len(info_p1.actions)
    k_p2 = len(info_p2.actions)
    cfv_grid = np.zeros((k_p1, k_p2))

    # Fill the CFV grid: recurse into expanded children, use cached values otherwise
    for (i, j), chance in node.grid.items():
        if chance.children:
            child_values = []
            for outcome in chance.children:
                child_val = cfr_update_recursive(outcome.node, iteration) if outcome.node is not None else outcome.leaf_value_p1
                child_values.append(child_val)
            # Uniform-weighted average over sampled chance outcomes
            cfv_grid[i, j] = sum(child_values) / len(child_values)
        else:
            cfv_grid[i, j] = 0.0

    # Marginalize to per-action counterfactual values
    v_p1_actions = cfv_grid @ sigma_p2  # shape [k_p1]
    v_p2_actions = -(cfv_grid.T @ sigma_p1)  # shape [k_p2], negated for p2's perspective

    # Node value from p1's perspective
    node_value_p1 = float(sigma_p1 @ v_p1_actions)

    # Update p1 regrets and strategy sum (single-action no-op sides compute zero instant regret)
    instant_regret_p1 = v_p1_actions - node_value_p1
    info_p1.regret = np.maximum(0.0, info_p1.regret + instant_regret_p1)
    info_p1.strategy_sum += iteration * sigma_p1

    # Update p2 regrets and strategy sum
    node_value_p2 = float(sigma_p2 @ v_p2_actions)
    instant_regret_p2 = v_p2_actions - node_value_p2
    info_p2.regret = np.maximum(0.0, info_p2.regret + instant_regret_p2)
    info_p2.strategy_sum += iteration * sigma_p2

    return node_value_p1


def tree_value(node: TurnNode) -> float:
    """p1's counterfactual value under the current regret-matched strategy.

    Read-only: no regret, strategy_sum, or visit mutation.  Uses the same
    marginalization as cfr_update_recursive but skips the three write lines.
    """
    info_p1 = node.info["p1"]
    info_p2 = node.info["p2"]

    sigma_p1 = regret_matching(info_p1.regret)
    sigma_p2 = regret_matching(info_p2.regret)

    k_p1 = len(info_p1.actions)
    k_p2 = len(info_p2.actions)
    cfv_grid = np.zeros((k_p1, k_p2))

    for (i, j), chance in node.grid.items():
        if chance.children:
            child_values = [tree_value(o.node) if o.node is not None else o.leaf_value_p1 for o in chance.children]
            cfv_grid[i, j] = sum(child_values) / len(child_values)

    v_p1_actions = cfv_grid @ sigma_p2
    return float(sigma_p1 @ v_p1_actions)
