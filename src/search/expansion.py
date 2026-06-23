"""PUCT-guided tree expansion for GT-CFR search.

Combines PUCT score computation, SimClient stepping (to grow the tree), and
batched CVPN evaluation of new leaf states. Also contains the top-level
_puct_expand_one loop that walks down from the root selecting cells to expand.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np
import torch

from action_space import index_to_choice_string, legal_mask
from encoder import encode
from obs_bundle import ObsBundle, collate_obs_bundles
from search.cfr import _expanded_children, _regret_matching
from search.types import ChanceNode, NodeKind, SearchConfig, TurnNode
from sim_client import SimError

if TYPE_CHECKING:
    from cvpn import CVPN
    from sim_client import SimClient, StepResult
    from state_types import StateView


# ---------------------------------------------------------------------------
# PUCT scoring
# ---------------------------------------------------------------------------


def _puct_scores(node: TurnNode, side: str) -> np.ndarray:
    """Compute PUCT scores for one side's actions at this node.

    score(I, a) = sigma^t(I, a) + c_puct * P(I, a) * sqrt(sum_b N(I, b)) / (1 + N(I, a))
    """
    sigma = _regret_matching(node.cumulative_regret[side])
    prior = node.policy_prior[side]
    visits = node.visit_counts[side].astype(np.float64)
    total_visits = visits.sum()
    exploration = prior * (np.sqrt(total_visits) / (1.0 + visits))
    return sigma + node._c_puct * exploration  # type: ignore[attr-defined]


def _puct_scores_with_config(node: TurnNode, side: str, config: SearchConfig) -> np.ndarray:
    """Compute PUCT scores using the config's c_puct value."""
    sigma = _regret_matching(node.cumulative_regret[side])
    prior = node.policy_prior[side]
    visits = node.visit_counts[side].astype(np.float64)
    total_visits = visits.sum()
    exploration = prior * (np.sqrt(total_visits) / (1.0 + visits))
    return sigma + config.c_puct * exploration


def _puct_select_cell(node: TurnNode, config: SearchConfig) -> tuple[int, int] | None:
    """Select the grid cell to expand/descend into via PUCT scores.

    Returns (p1_pos, p2_pos) index into the top-k grid, or None if fully expanded.
    """
    # Compute PUCT scores for each side and pick the joint cell with highest combined score
    sides = node.to_move
    k_p1 = len(node.actions.get("p1", []))
    k_p2 = len(node.actions.get("p2", []))

    scores_p1 = _puct_scores_with_config(node, "p1", config) if "p1" in sides and k_p1 > 0 else np.array([1.0])
    scores_p2 = _puct_scores_with_config(node, "p2", config) if "p2" in sides and k_p2 > 0 else np.array([1.0])

    # Joint score = outer product; pick the cell with highest combined score
    joint_scores = np.outer(scores_p1, scores_p2)

    # Flatten, sort descending, pick the first cell that has room for expansion
    flat_order = np.argsort(joint_scores.ravel())[::-1]
    for flat_idx in flat_order:
        i = int(flat_idx // k_p2)
        j = int(flat_idx % k_p2)
        chance = node.grid.get((i, j))
        if chance is None:
            return (i, j)
        # ChanceNode exists — check if it needs more outcome children
        if len(chance.children) < config.max_chance_children:
            return (i, j)
        # Check if any child below is an unexpanded TurnNode candidate
        for child_handle, _ in chance.children:
            child_node = _expanded_children.get(child_handle)
            if child_node is not None and not child_node.expanded:
                return (i, j)
    return None


# ---------------------------------------------------------------------------
# Expansion (CVPN batch evaluation + SimClient steps)
# ---------------------------------------------------------------------------


def _generate_seed() -> list[int]:
    """Generate a fresh random 4-element seed for SimClient reseeding."""
    return [random.randint(0, 2**31 - 1) for _ in range(4)]


def _expand_turn_node(
    node: TurnNode,
    sim: SimClient,
    net: CVPN,
    session: int,
    config: SearchConfig,
) -> None:
    """Expand a TurnNode: compute policy priors, select top-k, step all grid cells.

    Performs one batched CVPN forward for 2 + k^2 token bundles.
    """
    view = node.view
    sides = node.to_move

    # --- Encode both perspectives for policy priors ---
    obs_bundles: list[ObsBundle] = []
    bundle_purposes: list[str] = []  # track what each bundle is for

    for s in sides:
        obs = encode(view, s)
        obs_bundles.append(obs)
        bundle_purposes.append(f"prior_{s}")

    # Stage 1: Get policy priors via CVPN
    with torch.no_grad():
        prior_batch = collate_obs_bundles(obs_bundles)
        prior_logits, _ = net(prior_batch)  # [num_sides, A]

    # Extract top-k per side
    for idx, s in enumerate(sides):
        logits = prior_logits[idx]  # [A]
        # Apply legal mask to get valid probabilities
        mask = torch.from_numpy(legal_mask(view.legal.get(s), view.phase)).bool()
        masked_logits = logits.masked_fill(~mask, float("-inf"))
        probs = torch.softmax(masked_logits, dim=0)

        # Select top-k legal actions
        k = min(config.k_actions, int(mask.sum().item()))
        if k == 0:
            # No legal actions for this side (shouldn't happen in a non-terminal)
            node.actions[s] = []
            node.policy_prior[s] = np.array([])
            node.cumulative_regret[s] = np.array([])
            node.strategy_sum[s] = np.array([])
            node.visit_counts[s] = np.array([], dtype=np.int64)
            continue

        topk_values, topk_indices = torch.topk(probs, k)
        action_indices = topk_indices.numpy().tolist()

        # Normalize the prior over the top-k support
        prior_probs = topk_values.numpy()
        prior_probs = prior_probs / prior_probs.sum()

        node.actions[s] = action_indices
        node.policy_prior[s] = prior_probs
        node.cumulative_regret[s] = np.zeros(k, dtype=np.float64)
        node.strategy_sum[s] = np.zeros(k, dtype=np.float64)
        node.visit_counts[s] = np.zeros(k, dtype=np.int64)

    # Handle unilateral nodes — the non-acting side gets a single no-op
    for s in ["p1", "p2"]:
        if s not in sides:
            node.actions[s] = [0]  # placeholder single action
            node.policy_prior[s] = np.array([1.0])
            node.cumulative_regret[s] = np.zeros(1, dtype=np.float64)
            node.strategy_sum[s] = np.zeros(1, dtype=np.float64)
            node.visit_counts[s] = np.zeros(1, dtype=np.int64)

    k_p1 = len(node.actions["p1"])
    k_p2 = len(node.actions["p2"])

    if k_p1 == 0 or k_p2 == 0:
        node.expanded = True
        return

    # --- Stage 2: Step all k x k grid cells through SimClient ---
    grid_results: list[tuple[int, int, StepResult]] = []
    for i in range(k_p1):
        for j in range(k_p2):
            # Build the choice dict for this cell
            choices: dict[str, str] = {}
            if "p1" in sides:
                choices["p1"] = index_to_choice_string(node.actions["p1"][i])
            if "p2" in sides:
                choices["p2"] = index_to_choice_string(node.actions["p2"][j])

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
            utility_p1 = child_view.utility.get("p1", 0.0) if child_view.utility else 0.0
            chance = ChanceNode(children=[(result.child, utility_p1)])
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
            chance = ChanceNode(children=[(child_handle, value_p1)])
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
    sides = node.to_move
    choices: dict[str, str] = {}
    if "p1" in sides:
        choices["p1"] = index_to_choice_string(node.actions["p1"][i])
    if "p2" in sides:
        choices["p2"] = index_to_choice_string(node.actions["p2"][j])

    seed = _generate_seed()
    try:
        result = sim.step(node.handle, choices, seed)
    except SimError:
        return
    child_view = result.view

    if child_view.terminal:
        utility_p1 = child_view.utility.get("p1", 0.0) if child_view.utility else 0.0
        chance.children.append((result.child, utility_p1))
    else:
        # CVPN evaluation for the new world
        with torch.no_grad():
            obs = encode(child_view, "p1")
            _, value = net(obs)
            value_p1 = float(value.item())
        chance.children.append((result.child, value_p1))


def _expand_child_turn_node(
    child_handle: int,
    sim: SimClient,
    net: CVPN,
    session: int,
    config: SearchConfig,
) -> TurnNode | None:
    """Expand a frontier leaf into a full TurnNode (if it's a decision point)."""
    child_view = sim.view(child_handle)

    if child_view.terminal:
        return None

    # Determine node kind based on to_move
    if len(child_view.to_move) == 2:
        kind = NodeKind.TURN
    elif len(child_view.to_move) == 1:
        kind = NodeKind.UNILATERAL
    else:
        return None

    child_node = TurnNode(
        handle=child_handle,
        view=child_view,
        kind=kind,
        to_move=child_view.to_move,
    )
    _expand_turn_node(child_node, sim, net, session, config)
    _expanded_children[child_handle] = child_node
    return child_node


# ---------------------------------------------------------------------------
# PUCT-guided tree growth
# ---------------------------------------------------------------------------


def _puct_expand_one(
    root: TurnNode,
    sim: SimClient,
    net: CVPN,
    session: int,
    config: SearchConfig,
) -> bool:
    """Perform one PUCT-guided expansion step. Returns True if a node was expanded."""
    # Walk down from root following PUCT scores to find an expandable frontier
    node = root
    depth = 0
    max_depth = 50  # safety bound to prevent infinite loops

    while depth < max_depth:
        cell = _puct_select_cell(node, config)
        if cell is None:
            return False  # tree is fully expanded at this node

        chance = node.grid.get(cell)

        if chance is None:
            # This shouldn't happen after initial expansion, but handle gracefully
            return False

        # Check if this ChanceNode needs more outcome children
        if len(chance.children) < config.max_chance_children:
            _expand_chance_child(node, cell, sim, net, config)
            return True

        # Descend into the best child of this ChanceNode
        # Pick the child with the most "need" for expansion (least explored)
        expanded_child = None
        unexpanded_handle = None
        for child_handle, _ in chance.children:
            child_node = _expanded_children.get(child_handle)
            if child_node is None:
                # This is a frontier leaf — expand it
                unexpanded_handle = child_handle
                break
            if not child_node.expanded:
                unexpanded_handle = child_handle
                break
            # Already expanded — potential descent target
            if expanded_child is None:
                expanded_child = child_node

        if unexpanded_handle is not None:
            # Expand this frontier leaf into a TurnNode
            new_node = _expand_child_turn_node(unexpanded_handle, sim, net, session, config)
            return new_node is not None

        if expanded_child is not None:
            # Descend deeper into the tree
            node = expanded_child
            depth += 1
        else:
            return False

    return False
