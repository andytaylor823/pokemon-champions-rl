"""PUCT-guided tree expansion for GT-CFR search.

Combines PUCT score computation, SimClient stepping (to grow the tree), and
batched CVPN evaluation of new leaf states. Also contains the top-level
puct_expand_one loop that walks down from the root selecting cells to expand.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np
import torch

from action_space import index_to_choice_string, legal_mask
from encoder import encode
from obs_bundle import ObsBundle, collate_obs_bundles
from search.cfr import regret_matching
from search.types import ChanceNode, ChanceOutcome, InfoSet, SearchConfig, TurnNode
from sim_client import SimError

if TYPE_CHECKING:
    from cvpn import CVPN
    from sim_client import SimClient, StepResult
    from state_types import StateView


# ---------------------------------------------------------------------------
# PUCT scoring
# ---------------------------------------------------------------------------


def puct_scores(node: TurnNode, side: str, config: SearchConfig) -> np.ndarray:
    """Compute PUCT scores for one side's actions at this node.

    score(I, a) = sigma^t(I, a) + c_puct * P(I, a) * sqrt(sum_b N(I, b)) / (1 + N(I, a))
    """
    info = node.info[side]
    sigma = regret_matching(info.regret)
    visits = info.visits.astype(np.float64)
    total_visits = visits.sum()
    # PUCT exploration term: prior-weighted, visit-decaying bonus
    exploration = info.prior * (np.sqrt(total_visits) / (1.0 + visits))
    return sigma + config.c_puct * exploration


def puct_select_cell(node: TurnNode, config: SearchConfig) -> tuple[int, int] | None:
    """Select the grid cell to expand/descend into via PUCT scores.

    Returns (p1_pos, p2_pos) index into the top-k grid, or None if fully expanded.
    """
    info_p1 = node.info.get("p1")
    info_p2 = node.info.get("p2")
    if info_p1 is None or info_p2 is None:
        return None

    k_p1 = len(info_p1.actions)
    k_p2 = len(info_p2.actions)

    if k_p1 == 0 or k_p2 == 0:
        return None

    # Both sides always have valid InfoSets after expansion
    scores_p1 = puct_scores(node, "p1", config)
    scores_p2 = puct_scores(node, "p2", config)

    # Joint score = outer product; pick the cell with highest combined score
    joint_scores = np.outer(scores_p1, scores_p2)

    # Flatten, sort descending, pick the first cell that can be widened, deepened, or descended
    flat_order = np.argsort(joint_scores.ravel())[::-1]
    for flat_idx in flat_order:
        i = int(flat_idx // k_p2)
        j = int(flat_idx % k_p2)
        chance = node.grid.get((i, j))
        if chance is None:
            # Cell failed during initial expansion (SimError) — permanently dead
            continue
        # Widen: ChanceNode needs more outcome children
        if len(chance.children) < config.max_chance_children:
            return (i, j)
        # Full ChanceNode — eligible for deepening (frontier leaves / unexpanded
        # children) OR descent (all outcomes are expanded TurnNodes). Either way
        # puct_expand_one can make progress, so return the cell unconditionally.
        return (i, j)
    return None


# ---------------------------------------------------------------------------
# Expansion helpers
# ---------------------------------------------------------------------------


def _generate_seed() -> list[int]:
    """Generate a fresh random 4-element seed for SimClient reseeding."""
    return [random.randint(0, 2**31 - 1) for _ in range(4)]


def _terminal_utility_p1(view: StateView) -> float:
    """Extract p1's terminal utility from a StateView, defaulting to 0.0."""
    return view.utility.get("p1", 0.0) if view.utility else 0.0


def _cell_choices(node: TurnNode, i: int, j: int) -> dict[str, str]:
    """Build the SimClient choice dict for grid cell (i, j).

    Only includes acting sides — the non-acting side's placeholder action is
    not sent to SimClient.
    """
    choices: dict[str, str] = {}
    for side, idx in (("p1", i), ("p2", j)):
        if side in node.to_move:
            choices[side] = index_to_choice_string(node.info[side].actions[idx])
    return choices


# ---------------------------------------------------------------------------
# Expansion (CVPN batch evaluation + SimClient steps)
# ---------------------------------------------------------------------------


def expand_turn_node(
    node: TurnNode,
    sim: SimClient,
    net: CVPN,
    config: SearchConfig,
) -> None:
    """Expand a TurnNode: compute policy priors, select top-k, step all grid cells.

    Performs one batched CVPN forward for 2 + k^2 token bundles. Guarantees
    both sides have a valid InfoSet after expansion: the non-acting side in a
    unilateral node gets a single no-op action, so CFR/PUCT code can assume
    both p1 and p2 entries exist without conditional fallbacks.
    """
    view = node.view
    sides = node.to_move

    # --- Encode both perspectives for policy priors ---
    obs_bundles: list[ObsBundle] = []
    for s in sides:
        obs = encode(view, s)
        obs_bundles.append(obs)

    # Stage 1: Get policy priors via CVPN
    with torch.no_grad():
        prior_batch = collate_obs_bundles(obs_bundles)
        prior_logits, _ = net(prior_batch)  # [num_sides, A]

    # Extract top-k per acting side and build InfoSet
    for idx, s in enumerate(sides):
        logits = prior_logits[idx]  # [A]
        # Apply legal mask to get valid probabilities
        mask = torch.from_numpy(legal_mask(view.legal.get(s), view.phase)).bool()
        masked_logits = logits.masked_fill(~mask, float("-inf"))
        probs = torch.softmax(masked_logits, dim=0)

        # Select top-k legal actions
        k = min(config.k_actions, int(mask.sum().item()))
        if k == 0:
            node.info[s] = InfoSet.empty()
            continue

        topk_values, topk_indices = torch.topk(probs, k)
        action_indices = topk_indices.numpy().tolist()

        # Normalize the prior over the top-k support
        prior_probs = topk_values.numpy()
        prior_probs = prior_probs / prior_probs.sum()

        node.info[s] = InfoSet.from_actions(action_indices, prior_probs)

    # Non-acting sides get a single-action no-op InfoSet
    for s in ("p1", "p2"):
        if s not in sides:
            node.info[s] = InfoSet.noop()

    k_p1 = len(node.info["p1"].actions)
    k_p2 = len(node.info["p2"].actions)

    if k_p1 == 0 or k_p2 == 0:
        node.expanded = True
        return

    # --- Stage 2: Step all k x k grid cells through SimClient ---
    grid_results: list[tuple[int, int, StepResult]] = []
    for i in range(k_p1):
        for j in range(k_p2):
            choices = _cell_choices(node, i, j)

            # Step with a fresh seed to sample a chance outcome.
            # Guard against mask/engine discrepancies (e.g. target types the mask
            # marks legal but Showdown rejects at strictChoices validation).
            seed = _generate_seed()
            try:
                result = sim.step(node.handle, choices, seed)
            except SimError:
                continue
            grid_results.append((i, j, result))

    # --- Stage 3: Batch-evaluate all resulting states with CVPN ---
    if not grid_results:
        # All cells failed (mask/engine discrepancy) — mark as expanded but empty
        node.expanded = True
        return

    child_bundles: list[ObsBundle] = []
    child_info: list[tuple[int, int, int, StateView]] = []  # (i, j, child_handle, child_view)

    for i, j, result in grid_results:
        child_view = result.view
        if child_view.terminal:
            # Terminal — use real payoff, no CVPN needed
            utility_p1 = _terminal_utility_p1(child_view)
            chance = ChanceNode(children=[ChanceOutcome(handle=result.child, leaf_value_p1=utility_p1)])
            node.grid[(i, j)] = chance
        else:
            # Non-terminal — encode for CVPN value evaluation (from p1 perspective)
            obs = encode(child_view, "p1")
            child_bundles.append(obs)
            child_info.append((i, j, result.child, child_view))

    # Batch forward for non-terminal children
    if child_bundles:
        with torch.no_grad():
            batched = collate_obs_bundles(child_bundles)
            _, values = net(batched)  # [num_children]

        for idx, (i, j, child_handle, _child_view) in enumerate(child_info):
            value_p1 = float(values[idx].item())
            chance = ChanceNode(children=[ChanceOutcome(handle=child_handle, leaf_value_p1=value_p1)])
            node.grid[(i, j)] = chance

    node.expanded = True


def _expand_chance_child(
    node: TurnNode,
    cell: tuple[int, int],
    sim: SimClient,
    net: CVPN,
    config: SearchConfig,
) -> None:
    """Grow a ChanceNode by adding one more outcome world (up to K children)."""
    i, j = cell
    chance = node.grid.get(cell)
    if chance is None or len(chance.children) >= config.max_chance_children:
        return

    # Build the choice for this cell and step with a new seed
    choices = _cell_choices(node, i, j)
    seed = _generate_seed()
    try:
        result = sim.step(node.handle, choices, seed)
    except SimError:
        return
    child_view = result.view

    if child_view.terminal:
        chance.children.append(ChanceOutcome(handle=result.child, leaf_value_p1=_terminal_utility_p1(child_view)))
    else:
        # CVPN evaluation for the new world
        with torch.no_grad():
            obs = encode(child_view, "p1")
            _, value = net(obs)
            value_p1 = float(value.item())
        chance.children.append(ChanceOutcome(handle=result.child, leaf_value_p1=value_p1))


def _expand_child_turn_node(
    outcome: ChanceOutcome,
    sim: SimClient,
    net: CVPN,
    config: SearchConfig,
) -> TurnNode | None:
    """Expand a frontier leaf into a full TurnNode (if it's a decision point).

    Sets outcome.node in-place so CFR+ and PUCT can traverse the child directly
    without an external registry.
    """
    child_view = sim.view(outcome.handle)

    if child_view.terminal:
        return None

    # No acting sides means nothing to decide (shouldn't happen for non-terminals)
    if not child_view.to_move:
        return None

    child_node = TurnNode(
        handle=outcome.handle,
        view=child_view,
        to_move=child_view.to_move,
    )
    expand_turn_node(child_node, sim, net, config)
    # Wire the child into the tree so CFR+ recurses through it directly
    outcome.node = child_node
    return child_node


# ---------------------------------------------------------------------------
# PUCT-guided tree growth
# ---------------------------------------------------------------------------


def puct_expand_one(
    root: TurnNode,
    sim: SimClient,
    net: CVPN,
    config: SearchConfig,
) -> bool:
    """Perform one PUCT-guided expansion step, descending arbitrarily deep.

    Walks down from root following PUCT scores. At each level:
    - If the selected cell has room for more chance children → widen it.
    - If the cell is full and has frontier leaves → deepen (expand into TurnNode).
    - If the cell is full and all outcomes are expanded → descend into the best child.

    Returns True if a node was expanded (widen or deepen), False if tree is saturated.
    """
    node = root
    depth = 0
    max_depth = 50  # safety bound to prevent infinite loops

    while depth < max_depth:
        cell = puct_select_cell(node, config)
        if cell is None:
            return False  # tree is fully expanded at this node

        chance = node.grid.get(cell)

        if chance is None:
            # Dead cell (SimError during expansion) — select shouldn't return these
            return False

        # Widen: ChanceNode needs more outcome children
        if len(chance.children) < config.max_chance_children:
            _expand_chance_child(node, cell, sim, net, config)
            return True

        # Full ChanceNode — find a frontier leaf to deepen, or an expanded child to descend
        expanded_child = None
        unexpanded_outcome: ChanceOutcome | None = None
        for outcome in chance.children:
            if outcome.node is None:
                unexpanded_outcome = outcome
                break
            if not outcome.node.expanded:
                unexpanded_outcome = outcome
                break
            # Fully expanded — potential descent target
            if expanded_child is None:
                expanded_child = outcome.node

        if unexpanded_outcome is not None:
            # Deepen: expand this frontier leaf into a TurnNode
            new_node = _expand_child_turn_node(unexpanded_outcome, sim, net, config)
            return new_node is not None

        if expanded_child is not None:
            # Descend: move down one level and repeat
            node = expanded_child
            depth += 1
        else:
            return False

    return False
