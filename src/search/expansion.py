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

from action_space import action_to_choice_contextual, forced_actions
from encoder import encode
from obs_bundle import ObsBundle, collate_obs_bundles
from search.cfr import regret_matching
from search.types import ChanceNode, ChanceOutcome, InfoSet, SearchConfig, TurnNode
from sim_client import SimError

if TYPE_CHECKING:
    from cvpn import CVPN
    from sim_client import SimClient
    from state_types import StateView

# Absolute depth cap for puct_expand_one descent to prevent runaway loops.
# In practice the expansion_budget exhausts long before this is reached.
_MAX_DESCENT_DEPTH = 50


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
    """Select the highest-PUCT-scored grid cell that puct_expand_one can act on.

    Returns (p1_pos, p2_pos) index into the top-k grid, or None if the grid
    is empty.  The widen-vs-deepen-vs-descend decision lives entirely in
    puct_expand_one; this function just picks the best live cell.
    """
    info_p1, info_p2 = node.info["p1"], node.info["p2"]
    if not info_p1.actions or not info_p2.actions:
        return None

    joint = np.outer(puct_scores(node, "p1", config), puct_scores(node, "p2", config))
    k_p2 = len(info_p2.actions)
    for flat_idx in np.argsort(joint.ravel())[::-1]:
        i, j = divmod(int(flat_idx), k_p2)
        if (i, j) in node.grid:
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

    Uses the contextual builder so spread/self/charging moves get their targets
    stripped and slot-specific ally targets are resolved correctly.
    """
    choices: dict[str, str] = {}
    for side, idx in (("p1", i), ("p2", j)):
        if side in node.to_move:
            legal_req = node.view.legal.get(side) if node.view else None
            choices[side] = action_to_choice_contextual(
                node.info[side].actions[idx], legal_req
            )
    return choices


# ---------------------------------------------------------------------------
# Forced-node collapse helpers
# ---------------------------------------------------------------------------

# Safety cap for chained forced decisions (e.g. multi-faint forced switches).
# In practice these chains are 1–2 steps; the cap prevents pathological loops.
_MAX_FORCED_CHAIN = 20


def _collapse_forced(
    sim: SimClient,
    handle: int,
    view: StateView,
) -> tuple[int, StateView]:
    """Step through a chain of forced decisions to the next genuine decision or terminal.

    A forced decision (every acting side has exactly one legal action) carries
    no strategic choice, so it is collapsed transparently — no CVPN forward,
    no regret tables, no tree node (self-play.md §2.2, search.md §8).

    Returns (final_handle, final_view) at the first genuine decision, terminal,
    or non-acting state encountered.
    """
    for _ in range(_MAX_FORCED_CHAIN):
        if view.terminal:
            break
        forced = forced_actions(view.legal, view.to_move, view.phase)
        if forced is None:
            break
        choices = {
            s: action_to_choice_contextual(idx, view.legal.get(s))
            for s, idx in forced.items()
        }
        seed = _generate_seed()
        try:
            result = sim.step(handle, choices, seed)
        except SimError:
            break
        handle = result.child
        view = result.view
    return handle, view


# ---------------------------------------------------------------------------
# Single-state CVPN helpers
# ---------------------------------------------------------------------------


def cvpn_value(net: CVPN, view: StateView, perspective: str = "p1") -> float:
    """Evaluate a single state with CVPN and return the scalar value."""
    with torch.no_grad():
        _, value = net(encode(view, perspective))
    return float(value.item())


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
        # CVPN already masked illegal logits to -inf; softmax directly
        probs = torch.softmax(prior_logits[idx], dim=0)

        # Top-k count from the action_mask baked into the ObsBundle by the encoder
        k = min(config.k_actions, int(obs_bundles[idx]["action_mask"].sum().item()))
        if k == 0:
            node.info[s] = InfoSet.empty()
            continue

        topk_values, topk_indices = torch.topk(probs, k)
        action_indices = topk_indices.detach().cpu().numpy().tolist()

        # Normalize the prior over the top-k support
        prior_probs = topk_values.detach().cpu().numpy()
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
    # Each cell is stepped, then forced decisions are collapsed transparently
    # so the CVPN only evaluates genuine decision points or terminals.
    cell_results: list[tuple[int, int, int, StateView]] = []
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
            # Collapse any forced decisions after the step (no CVPN, no regret tables)
            child_handle, child_view = _collapse_forced(sim, result.child, result.view)
            cell_results.append((i, j, child_handle, child_view))

    # --- Stage 3: Batch-evaluate all resulting states with CVPN ---
    if not cell_results:
        # All cells failed (mask/engine discrepancy) — mark as expanded but empty
        node.expanded = True
        return

    child_bundles: list[ObsBundle] = []
    child_info: list[tuple[int, int, int, StateView]] = []  # (i, j, child_handle, child_view)

    for i, j, child_handle, child_view in cell_results:
        if child_view.terminal:
            # Terminal — use real payoff, no CVPN needed
            utility_p1 = _terminal_utility_p1(child_view)
            chance = ChanceNode(children=[ChanceOutcome(handle=child_handle, leaf_value_p1=utility_p1)])
            node.grid[(i, j)] = chance
        else:
            # Non-terminal genuine decision — encode for CVPN value evaluation
            obs = encode(child_view, "p1")
            child_bundles.append(obs)
            child_info.append((i, j, child_handle, child_view))

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
    # Collapse any forced decisions after the step
    child_handle, child_view = _collapse_forced(sim, result.child, result.view)

    if child_view.terminal:
        chance.children.append(ChanceOutcome(handle=child_handle, leaf_value_p1=_terminal_utility_p1(child_view)))
    else:
        value_p1 = cvpn_value(net, child_view)
        chance.children.append(ChanceOutcome(handle=child_handle, leaf_value_p1=value_p1))


def _expand_child_turn_node(
    outcome: ChanceOutcome,
    sim: SimClient,
    net: CVPN,
    config: SearchConfig,
) -> TurnNode | None:
    """Expand a frontier leaf into a full TurnNode (if it's a decision point).

    If the child state is a forced decision (every acting side has one legal
    action), collapses through it transparently to the next genuine decision or
    terminal — no degenerate TurnNode is created. Sets outcome.node in-place so
    CFR+ and PUCT can traverse the child directly without an external registry.
    """
    child_view = sim.view(outcome.handle)

    if child_view.terminal:
        return None

    if not child_view.to_move:
        return None

    # Collapse through forced decisions to the next genuine decision/terminal
    handle, child_view = _collapse_forced(sim, outcome.handle, child_view)

    if child_view.terminal:
        # Forced chain ended at a terminal — update cached value, no TurnNode
        outcome.leaf_value_p1 = _terminal_utility_p1(child_view)
        return None

    if not child_view.to_move:
        return None

    child_node = TurnNode(
        handle=handle,
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

    while depth < _MAX_DESCENT_DEPTH:
        cell = puct_select_cell(node, config)
        if cell is None:
            return False

        i, j = cell
        chance = node.grid[cell]  # puct_select_cell only returns live cells

        # Track which actions were selected so PUCT exploration decays per action
        node.info["p1"].visits[i] += 1
        node.info["p2"].visits[j] += 1

        # Widen: ChanceNode needs more outcome children
        if len(chance.children) < config.max_chance_children:
            _expand_chance_child(node, cell, sim, net, config)
            return True

        # Full ChanceNode — find a frontier leaf to deepen, or an expanded child to descend
        expanded_child = None
        for outcome in chance.children:
            if outcome.node is None or not outcome.node.expanded:
                new_node = _expand_child_turn_node(outcome, sim, net, config)
                if new_node is not None:
                    return True
                # Terminal or empty to_move — skip to next outcome
                continue
            if expanded_child is None:
                expanded_child = outcome.node

        if expanded_child is not None:
            node = expanded_child
            depth += 1
        else:
            return False

    return False
