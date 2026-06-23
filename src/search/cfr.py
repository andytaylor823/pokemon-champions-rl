"""CFR+ regret-update traversals for the GT-CFR search tree.

Implements regret matching+ and both flat and recursive CFR+ updates over
TurnNode/ChanceNode trees. Owns the module-level _expanded_children registry
that maps SimClient handles to expanded TurnNodes during a single search.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from search.types import TurnNode

# Module-level registry mapping handles to expanded TurnNodes (set per-search)
_expanded_children: dict[int, TurnNode] = {}


# ---------------------------------------------------------------------------
# CFR+ helpers
# ---------------------------------------------------------------------------


def _regret_matching(cumulative_regret: np.ndarray) -> np.ndarray:
    """Derive current strategy from cumulative regrets via regret matching+.

    Returns uniform distribution if no positive regret exists.
    """
    positive = np.maximum(cumulative_regret, 0.0)
    total = positive.sum()
    if total > 0:
        return positive / total
    # Uniform fallback
    return np.ones_like(cumulative_regret) / len(cumulative_regret)


def _cfr_update(node: TurnNode, iteration: int) -> float:
    """Run one CFR+ traversal over the frozen tree rooted at this TurnNode.

    Updates both players' regret tables simultaneously (no alternation).
    Returns the node's CFV from p1's perspective.
    """
    sides = node.to_move
    k = {s: len(node.actions[s]) for s in sides}

    # Current strategy per side from regret matching+
    sigma: dict[str, np.ndarray] = {}
    for s in sides:
        sigma[s] = _regret_matching(node.cumulative_regret[s])

    # For unilateral nodes, the non-acting side has a single no-op action with mass 1
    all_sides = ["p1", "p2"]
    for s in all_sides:
        if s not in sigma:
            sigma[s] = np.array([1.0])

    # Compute the CFV matrix for the grid cells
    k_p1 = k.get("p1", 1)
    k_p2 = k.get("p2", 1)
    cfv_grid = np.zeros((k_p1, k_p2))

    for (i, j), chance in node.grid.items():
        cfv_grid[i, j] = chance.value_p1

    # Marginalize to per-action CFVs for each player
    # v_p1(a1) = sum_a2 sigma_p2(a2) * CFV(a1, a2)
    sigma_p2 = sigma.get("p2", np.array([1.0]))
    sigma_p1 = sigma.get("p1", np.array([1.0]))

    v_p1_actions = cfv_grid @ sigma_p2  # shape [k_p1]
    v_p2_actions = -(cfv_grid.T @ sigma_p1)  # shape [k_p2], negated for p2's perspective

    # Node value from p1's perspective
    node_value_p1 = float(sigma_p1 @ v_p1_actions)

    # Update regrets and strategy sums for each acting side
    if "p1" in node.cumulative_regret:
        v_p1_node = float(sigma_p1 @ v_p1_actions)
        instant_regret_p1 = v_p1_actions - v_p1_node
        # CFR+ clipping: cumulative regret floored at 0
        node.cumulative_regret["p1"] = np.maximum(0.0, node.cumulative_regret["p1"] + instant_regret_p1)
        # Linear-weighted strategy sum accumulation
        node.strategy_sum["p1"] += iteration * sigma["p1"]
        node.visit_counts["p1"] += 1

    if "p2" in node.cumulative_regret:
        v_p2_node = float(sigma_p2 @ v_p2_actions)
        instant_regret_p2 = v_p2_actions - v_p2_node
        node.cumulative_regret["p2"] = np.maximum(0.0, node.cumulative_regret["p2"] + instant_regret_p2)
        node.strategy_sum["p2"] += iteration * sigma["p2"]
        node.visit_counts["p2"] += 1

    return node_value_p1


def _cfr_update_recursive(node: TurnNode, iteration: int) -> float:
    """Recursively run CFR+ update through the tree (depth-first).

    For expanded children below ChanceNodes, recurse into them; for frontier
    leaves and terminals, use cached values. Returns p1's CFV at this node.
    """
    sides = node.to_move
    k = {s: len(node.actions[s]) for s in sides}

    # Current strategy per side
    sigma: dict[str, np.ndarray] = {}
    for s in sides:
        sigma[s] = _regret_matching(node.cumulative_regret[s])

    all_sides = ["p1", "p2"]
    for s in all_sides:
        if s not in sigma:
            sigma[s] = np.array([1.0])

    k_p1 = k.get("p1", 1)
    k_p2 = k.get("p2", 1)
    cfv_grid = np.zeros((k_p1, k_p2))

    # Fill the CFV grid from ChanceNode children
    for (i, j), chance in node.grid.items():
        # ChanceNode's value is the uniform average of its outcome children
        # Each child may itself be an expanded TurnNode (recurse) or a leaf (use cached value)
        if chance.children:
            child_values = []
            for child_handle, cached_val in chance.children:
                # Check if this child has been expanded into a TurnNode
                child_node = _expanded_children.get(child_handle)
                child_val = _cfr_update_recursive(child_node, iteration) if child_node is not None else cached_val
                child_values.append(child_val)
            cfv_grid[i, j] = sum(child_values) / len(child_values)
        else:
            cfv_grid[i, j] = 0.0

    # Marginalize to per-action CFVs
    sigma_p2 = sigma.get("p2", np.array([1.0]))
    sigma_p1 = sigma.get("p1", np.array([1.0]))

    v_p1_actions = cfv_grid @ sigma_p2
    v_p2_actions = -(cfv_grid.T @ sigma_p1)

    node_value_p1 = float(sigma_p1 @ v_p1_actions)

    # Update regrets and strategy sums
    if "p1" in node.cumulative_regret:
        v_p1_node = float(sigma_p1 @ v_p1_actions)
        instant_regret_p1 = v_p1_actions - v_p1_node
        node.cumulative_regret["p1"] = np.maximum(0.0, node.cumulative_regret["p1"] + instant_regret_p1)
        node.strategy_sum["p1"] += iteration * sigma["p1"]
        node.visit_counts["p1"] += 1

    if "p2" in node.cumulative_regret:
        v_p2_node = float(sigma_p2 @ v_p2_actions)
        instant_regret_p2 = v_p2_actions - v_p2_node
        node.cumulative_regret["p2"] = np.maximum(0.0, node.cumulative_regret["p2"] + instant_regret_p2)
        node.strategy_sum["p2"] += iteration * sigma["p2"]
        node.visit_counts["p2"] += 1

    return node_value_p1
