"""
GT-CFR inner-loop search for Leduc Hold'em.

Implements GT-CFR (Player of Games, Schmid et al. 2021) adapted for Leduc:
  1. Build a search tree over the PUBLIC state (community card + action history)
  2. Run CFR+ traversals over all 30 consistent deals simultaneously
  3. At chance nodes (community card), average over possible outcomes per deal
  4. At leaves, use CVPN value estimates; at terminals, use real payoffs
  5. After the budget is spent, output the average strategy and refined CFVs

KEY ADDITION vs Kuhn: the search tree contains CHANCE NODES for the community
card deal between rounds. At a chance node, the tree branches into one child per
possible community rank (J, Q, K). During CFR traversal with a specific deal,
each branch is weighted by the probability of that community rank given the
cards already dealt to the players.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np
import torch

from toy_examples.leduc_poker.game import (
    RANKS,
    DECK_SIZE,
    SUITS_PER_RANK,
    PLAYER_1,
    PLAYER_2,
    CHANCE,
    ANTE,
    CHECK,
    BET_SMALL,
    BET_BIG,
    FOLD,
    CALL,
    RAISE_SMALL,
    RAISE_BIG,
    ROUND1_BET_SIZES,
    ROUND2_BET_SIZES,
    LeducState,
    card_rank,
    card_rank_index,
    _split_rounds,
)
from toy_examples.leduc_poker.network import (
    LeducCVPN,
    encode_public_state,
    get_policy_for_info_set,
    get_value_for_info_set,
)


class NodeType(Enum):
    """Type of node in the search tree."""
    PLAYER = auto()
    CHANCE = auto()
    TERMINAL = auto()


@dataclass
class SearchNode:
    """
    A node in the GT-CFR search tree, representing a PUBLIC state.
    The tree is card-agnostic; cards only matter during CFR traversal.
    """

    # The public state this node represents
    history: tuple[str, ...]
    # Community card rank at this node (None = not yet dealt, or a rank string)
    community_rank: str | None
    # Node type
    node_type: NodeType = NodeType.PLAYER
    # Children: action_or_rank -> SearchNode
    children: dict[str, "SearchNode"] = field(default_factory=dict)
    # Whether this node has been expanded
    expanded: bool = False
    # CFR+ cumulative regrets per info set: {info_key: {action: regret}}
    cumulative_regret: dict[str, dict[str, float]] = field(default_factory=dict)
    # Strategy sum for average strategy: {info_key: {action: sum}}
    strategy_sum: dict[str, dict[str, float]] = field(default_factory=dict)
    # Visit count per info set per action: {info_key: {action: count}}
    visit_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    # Cached CVPN policy prior: {info_key: {action: probability}}
    policy_prior: dict[str, dict[str, float]] = field(default_factory=dict)
    # Cached CVPN value estimates: {rank: value}
    nn_values: dict[str, float] = field(default_factory=dict)
    # Pot at this node (for utility computation)
    pot: tuple[int, int] = (ANTE, ANTE)


# ---------------------------------------------------------------------------
# Game-logic helpers operating on (history, community_rank, pot) tuples
# ---------------------------------------------------------------------------

def _build_state(
    history: tuple[str, ...],
    community_rank: str | None,
    pot: tuple[int, int],
    p1_card: int,
    p2_card: int,
) -> LeducState:
    """Build a concrete LeducState from search node info + specific deal."""
    comm_id: int | None = None
    if community_rank is not None:
        # Pick a concrete card id for the community rank that doesn't collide
        target_rank_idx = RANKS.index(community_rank)
        for s in range(SUITS_PER_RANK):
            candidate = target_rank_idx * SUITS_PER_RANK + s
            if candidate != p1_card and candidate != p2_card:
                comm_id = candidate
                break
    return LeducState(cards=(p1_card, p2_card), community_card=comm_id, history=history, pot=pot)


def _is_terminal(history: tuple[str, ...], community_rank: str | None) -> bool:
    """Check if the game is over at this public state."""
    dummy = _build_state(history, community_rank, (ANTE, ANTE), 0, 2)
    return dummy.is_terminal()


def _is_chance(history: tuple[str, ...], community_rank: str | None) -> bool:
    """Check if this public state is a community card chance node."""
    if community_rank is not None:
        return False
    dummy = _build_state(history, None, (ANTE, ANTE), 0, 2)
    return dummy.is_chance_node()


def _current_player(history: tuple[str, ...], community_rank: str | None) -> int:
    """Determine which player acts at this public state."""
    dummy = _build_state(history, community_rank, (ANTE, ANTE), 0, 2)
    if dummy.is_terminal():
        raise ValueError("No player at terminal")
    if dummy.is_chance_node():
        return CHANCE
    return dummy.current_player()


def _legal_actions(history: tuple[str, ...], community_rank: str | None) -> list[str]:
    """Get legal actions at this public state."""
    dummy = _build_state(history, community_rank, (ANTE, ANTE), 0, 2)
    if dummy.is_terminal() or dummy.is_chance_node():
        return []
    return dummy.legal_actions()


def _terminal_utility(
    history: tuple[str, ...],
    community_rank: str | None,
    pot: tuple[int, int],
    p1_card: int,
    p2_card: int,
) -> float:
    """
    Compute P1's payoff at a terminal node for a specific deal.
    Computed directly from ranks and pot rather than building a full LeducState,
    to handle edge cases where both players hold the same rank as the community.
    """
    p1_committed, p2_committed = pot
    total_pot = p1_committed + p2_committed

    # Fold: the player who folded loses their committed chips
    if history and history[-1] == FOLD:
        # Determine who folded based on action count within current round
        r1, r2 = _split_rounds(history)
        in_round2 = community_rank is not None and len(r2) > 0
        round_actions = r2 if in_round2 else r1
        # Folder is the player who just acted (odd index = P2, even = P1)
        folder = PLAYER_1 if len(round_actions) % 2 == 1 else PLAYER_2
        if folder == PLAYER_1:
            return float(-p1_committed)  # P1 folded, loses investment
        else:
            return float(total_pot - p1_committed)  # P2 folded, P1 wins

    # Showdown: compare ranks
    p1_rank = card_rank_index(p1_card)
    p2_rank = card_rank_index(p2_card)
    comm_rank_idx = RANKS.index(community_rank) if community_rank is not None else -1

    p1_pair = p1_rank == comm_rank_idx
    p2_pair = p2_rank == comm_rank_idx

    if p1_pair and not p2_pair:
        winner = PLAYER_1
    elif p2_pair and not p1_pair:
        winner = PLAYER_2
    elif p1_rank > p2_rank:
        winner = PLAYER_1
    elif p2_rank > p1_rank:
        winner = PLAYER_2
    else:
        winner = PLAYER_1  # tie-break (same rank, no pair)

    if winner == PLAYER_1:
        return float(total_pot - p1_committed)
    else:
        return float(-p1_committed)


def _info_set_key(rank: str, community_rank: str | None, history: tuple[str, ...]) -> str:
    """Construct the info set key for a player holding `rank` at this public state."""
    comm_str = community_rank if community_rank is not None else "?"
    history_str = ",".join(history) if history else ""
    return f"{rank}:{comm_str}:{history_str}"


def _compute_pot(
    history: tuple[str, ...],
    community_rank: str | None,
    p1_card: int,
    p2_card: int,
) -> tuple[int, int]:
    """Replay the history to compute the pot at a given node."""
    state = LeducState(cards=(p1_card, p2_card))
    for action in history:
        if state.is_chance_node():
            # Need to apply community card first
            target = RANKS.index(community_rank) * SUITS_PER_RANK
            for s in range(SUITS_PER_RANK):
                candidate = RANKS.index(community_rank) * SUITS_PER_RANK + s
                if candidate != p1_card and candidate != p2_card:
                    state = state.apply_chance(candidate)
                    break
        state = state.apply_action(action)
    return state.pot


def _community_chance_probs(p1_card: int, p2_card: int) -> dict[str, float]:
    """
    Probability of each community RANK given a specific deal.
    The 4 remaining cards are equally likely; group by rank.
    """
    remaining = [c for c in range(DECK_SIZE) if c != p1_card and c != p2_card]
    total = len(remaining)
    rank_counts: dict[str, int] = {}
    for c in remaining:
        r = card_rank(c)
        rank_counts[r] = rank_counts.get(r, 0) + 1
    return {r: count / total for r, count in rank_counts.items()}


# ---------------------------------------------------------------------------
# CFR+ helpers
# ---------------------------------------------------------------------------

def _get_current_strategy(
    node: SearchNode, info_key: str, actions: list[str]
) -> dict[str, float]:
    """Derive current strategy from cumulative regrets via regret matching+."""
    if info_key not in node.cumulative_regret:
        return {a: 1.0 / len(actions) for a in actions}

    regrets = node.cumulative_regret[info_key]
    positive_sum = sum(max(0.0, regrets.get(a, 0.0)) for a in actions)

    if positive_sum > 0:
        return {a: max(0.0, regrets.get(a, 0.0)) / positive_sum for a in actions}
    return {a: 1.0 / len(actions) for a in actions}


def _evaluate_node(node: SearchNode, net: LeducCVPN) -> None:
    """
    Evaluate a PLAYER node using the CVPN. Caches policy priors and value
    estimates for all possible private ranks.
    """
    history = node.history
    community_rank = node.community_rank
    acting_player = _current_player(history, community_rank)
    actions = _legal_actions(history, community_rank)

    # Build a dummy state for encoding
    dummy = _build_state(history, community_rank, node.pot, 0, 2)
    encoded = encode_public_state(dummy, acting_player)
    x = torch.tensor(encoded, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        policy_logits, values = net(x)

    # Cache priors and values for each possible private rank
    for rank in RANKS:
        info_key = _info_set_key(rank, community_rank, history)
        node.policy_prior[info_key] = get_policy_for_info_set(policy_logits[0], rank, actions)
        node.nn_values[rank] = get_value_for_info_set(values[0], rank)


def _expand_node(node: SearchNode, net: LeducCVPN) -> None:
    """Expand a node: create children for all legal actions or chance outcomes."""
    if node.node_type == NodeType.TERMINAL:
        return

    if node.node_type == NodeType.CHANCE:
        # Chance node: create one child per possible community rank
        for rank in RANKS:
            child_history = node.history
            child = SearchNode(
                history=child_history,
                community_rank=rank,
                pot=node.pot,
            )
            # Determine child node type
            if _is_terminal(child_history, rank):
                child.node_type = NodeType.TERMINAL
            else:
                child.node_type = NodeType.PLAYER
            node.children[rank] = child
        node.expanded = True
        return

    # Player node: create children for each legal action
    actions = _legal_actions(node.history, node.community_rank)
    _evaluate_node(node, net)

    for action in actions:
        child_history = node.history + (action,)
        # Compute child pot using a concrete deal (pot doesn't depend on specific cards
        # for action transitions, only on the action and current pot)
        child_pot = _updated_pot_from_action(node.pot, action, node.history, node.community_rank)

        # Determine child type
        if _is_terminal(child_history, node.community_rank):
            child_type = NodeType.TERMINAL
        elif _is_chance(child_history, node.community_rank):
            child_type = NodeType.CHANCE
        else:
            child_type = NodeType.PLAYER

        node.children[action] = SearchNode(
            history=child_history,
            community_rank=node.community_rank,
            pot=child_pot,
            node_type=child_type,
        )

    node.expanded = True


def _updated_pot_from_action(
    pot: tuple[int, int],
    action: str,
    history: tuple[str, ...],
    community_rank: str | None,
) -> tuple[int, int]:
    """Compute new pot after a player action without building a full state."""
    p1, p2 = pot
    # Determine who is acting
    player = _current_player(history, community_rank)
    # Determine round for bet sizes
    current_round = 2 if community_rank is not None else 1
    bet_sizes = ROUND1_BET_SIZES if current_round == 1 else ROUND2_BET_SIZES
    small_inc, big_inc = bet_sizes[0], bet_sizes[-1]

    if action in (CHECK, FOLD):
        return (p1, p2)
    elif action == CALL:
        if player == PLAYER_1:
            return (p2, p2)
        else:
            return (p1, p1)
    elif action in (BET_SMALL, RAISE_SMALL):
        opp = p2 if player == PLAYER_1 else p1
        new_val = opp + small_inc
        return (new_val, p2) if player == PLAYER_1 else (p1, new_val)
    elif action in (BET_BIG, RAISE_BIG):
        opp = p2 if player == PLAYER_1 else p1
        new_val = opp + big_inc
        return (new_val, p2) if player == PLAYER_1 else (p1, new_val)
    else:
        raise ValueError(f"Unknown action: {action}")


def _expand_tree_fully(node: SearchNode, net: LeducCVPN, max_depth: int = 20) -> None:
    """Recursively expand all non-terminal nodes."""
    if max_depth <= 0 or node.node_type == NodeType.TERMINAL:
        return
    if not node.expanded:
        _expand_node(node, net)
    for child in node.children.values():
        _expand_tree_fully(child, net, max_depth - 1)


# ---------------------------------------------------------------------------
# Core CFR+ traversal
# ---------------------------------------------------------------------------

def _cfr_traverse(
    node: SearchNode,
    net: LeducCVPN,
    traversing_player: int,
    p1_card: int,
    p2_card: int,
    reach_p1: float,
    reach_p2: float,
    iteration: int,
) -> float:
    """
    CFR+ traversal for a specific deal (p1_card, p2_card).
    Returns the counterfactual value for the traversing player.
    """
    # --- Terminal node: real payoff ---
    if node.node_type == NodeType.TERMINAL:
        payoff = _terminal_utility(
            node.history, node.community_rank, node.pot, p1_card, p2_card,
        )
        return payoff if traversing_player == PLAYER_1 else -payoff

    # --- Chance node: average over community card outcomes ---
    if node.node_type == NodeType.CHANCE:
        if not node.expanded:
            _expand_node(node, net)
        # Get probability of each community rank for this specific deal
        rank_probs = _community_chance_probs(p1_card, p2_card)
        value = 0.0
        for rank, prob in rank_probs.items():
            if prob > 0 and rank in node.children:
                child = node.children[rank]
                # Reach probabilities unchanged through chance nodes
                child_val = _cfr_traverse(
                    child, net, traversing_player, p1_card, p2_card,
                    reach_p1, reach_p2, iteration,
                )
                value += prob * child_val
        return value

    # --- Unexpanded leaf: use CVPN value estimate ---
    if not node.expanded:
        if not node.nn_values:
            _evaluate_node(node, net)
        rank = card_rank(p1_card if traversing_player == PLAYER_1 else p2_card)
        return node.nn_values.get(rank, 0.0)

    # --- Player node: CFR+ update ---
    acting_player = _current_player(node.history, node.community_rank)
    actions = _legal_actions(node.history, node.community_rank)
    acting_rank = card_rank(p1_card if acting_player == PLAYER_1 else p2_card)
    info_key = _info_set_key(acting_rank, node.community_rank, node.history)

    # Current strategy from regret matching+
    strategy = _get_current_strategy(node, info_key, actions)

    # Compute value of each action and weighted node value
    action_values: dict[str, float] = {}
    node_value = 0.0

    for action in actions:
        child = node.children[action]
        # Update reach for the acting player
        if acting_player == PLAYER_1:
            child_val = _cfr_traverse(
                child, net, traversing_player, p1_card, p2_card,
                reach_p1 * strategy[action], reach_p2, iteration,
            )
        else:
            child_val = _cfr_traverse(
                child, net, traversing_player, p1_card, p2_card,
                reach_p1, reach_p2 * strategy[action], iteration,
            )
        action_values[action] = child_val
        node_value += strategy[action] * child_val

    # Update regrets only at traversing player's info sets
    if acting_player == traversing_player:
        # Initialize accumulators if needed
        if info_key not in node.cumulative_regret:
            node.cumulative_regret[info_key] = {a: 0.0 for a in actions}
        if info_key not in node.strategy_sum:
            node.strategy_sum[info_key] = {a: 0.0 for a in actions}
        if info_key not in node.visit_counts:
            node.visit_counts[info_key] = {a: 0 for a in actions}

        # Counterfactual reach: opponent's contribution
        cf_reach = reach_p2 if traversing_player == PLAYER_1 else reach_p1

        # Cumulative regrets with CFR+ clipping (floor at 0)
        for action in actions:
            instant_regret = action_values[action] - node_value
            node.cumulative_regret[info_key][action] = max(
                0.0,
                node.cumulative_regret[info_key][action] + cf_reach * instant_regret,
            )

        # Accumulate strategy sum (linear weighting for CFR+)
        player_reach = reach_p1 if traversing_player == PLAYER_1 else reach_p2
        for action in actions:
            node.strategy_sum[info_key][action] += iteration * player_reach * strategy[action]
            node.visit_counts[info_key][action] += 1

    return node_value


# ---------------------------------------------------------------------------
# Main search entry point
# ---------------------------------------------------------------------------

def gt_cfr_search(
    root_state: LeducState,
    net: LeducCVPN,
    n_iterations: int = 200,
    c_puct: float = 2.0,
    expansion_interval: int = 10,
    full_expand: bool = True,
) -> tuple[dict[str, dict[str, float]], dict[str, float], dict[str, dict[str, float]]]:
    """
    Run GT-CFR search from a root state.

    Args:
        root_state: the current game state
        net: the CVPN for leaf evaluation and policy priors
        n_iterations: CFR+ iteration budget
        c_puct: exploration constant for PUCT expansion (incremental mode)
        expansion_interval: expand one node every N iters (incremental mode)
        full_expand: fully expand the tree upfront

    Returns:
        sigma_bar: average strategy at root {info_key: {action: prob}}
        cfvs: search-refined CFVs {rank: value} for the acting player
        full_strategy: average strategy at ALL tree nodes
    """
    history = root_state.history
    community_rank = (
        card_rank(root_state.community_card) if root_state.community_card is not None else None
    )

    # Determine root node type
    if root_state.is_terminal():
        return {}, {}, {}
    if root_state.is_chance_node():
        root_type = NodeType.CHANCE
    else:
        root_type = NodeType.PLAYER

    # Build search tree
    root = SearchNode(
        history=history,
        community_rank=community_rank,
        pot=root_state.pot,
        node_type=root_type,
    )

    if full_expand:
        _expand_tree_fully(root, net)
    else:
        _expand_node(root, net)

    # Enumerate all 30 consistent deals
    consistent_deals = [
        (p1, p2)
        for p1 in range(DECK_SIZE)
        for p2 in range(DECK_SIZE)
        if p1 != p2
    ]

    # Run CFR+ iterations
    for t in range(1, n_iterations + 1):
        for p1_card, p2_card in consistent_deals:
            _cfr_traverse(root, net, PLAYER_1, p1_card, p2_card, 1.0, 1.0, t)
            _cfr_traverse(root, net, PLAYER_2, p1_card, p2_card, 1.0, 1.0, t)

        # Incremental expansion
        if not full_expand and t % expansion_interval == 0 and t < n_iterations:
            _puct_expand(root, net, c_puct)

    # Extract outputs
    if root_type == NodeType.CHANCE:
        # At chance root, no player strategy to extract
        sigma_bar: dict[str, dict[str, float]] = {}
        cfvs: dict[str, float] = {}
    else:
        acting_player = _current_player(history, community_rank)
        sigma_bar = _extract_average_strategy(root, acting_player)
        cfvs = _extract_cfvs(root, acting_player, consistent_deals)

    full_strategy = extract_full_tree_strategy(root)
    return sigma_bar, cfvs, full_strategy


# ---------------------------------------------------------------------------
# PUCT expansion (incremental mode)
# ---------------------------------------------------------------------------

def _puct_expand(node: SearchNode, net: LeducCVPN, c_puct: float) -> None:
    """PUCT-guided expansion: walk the tree, expand first unexpanded leaf."""
    current = node

    while current.expanded and current.node_type == NodeType.PLAYER:
        actions = _legal_actions(current.history, current.community_rank)
        if not actions:
            break

        best_action = None
        best_score = float("-inf")

        for action in actions:
            total_score = 0.0
            count = 0
            for rank in RANKS:
                info_key = _info_set_key(rank, current.community_rank, current.history)
                strategy = _get_current_strategy(current, info_key, actions)
                prior = current.policy_prior.get(info_key, {}).get(action, 1.0 / len(actions))
                n_action = current.visit_counts.get(info_key, {}).get(action, 0)
                n_total = sum(
                    current.visit_counts.get(info_key, {}).get(a, 0) for a in actions
                )
                exploit = strategy[action]
                explore = c_puct * prior * np.sqrt(max(1, n_total)) / (1 + n_action)
                total_score += exploit + explore
                count += 1

            avg = total_score / count if count > 0 else 0.0
            if avg > best_score:
                best_score = avg
                best_action = action

        current = current.children[best_action]

    # Handle chance nodes by picking a random child
    if current.expanded and current.node_type == NodeType.CHANCE:
        for child in current.children.values():
            if not child.expanded and child.node_type != NodeType.TERMINAL:
                current = child
                break

    if current.node_type != NodeType.TERMINAL and not current.expanded:
        _expand_node(current, net)


# ---------------------------------------------------------------------------
# Strategy extraction
# ---------------------------------------------------------------------------

def _extract_average_strategy(
    root: SearchNode, _acting_player: int
) -> dict[str, dict[str, float]]:
    """Extract the average strategy (sigma_bar) from the root node only."""
    sigma_bar: dict[str, dict[str, float]] = {}
    actions = _legal_actions(root.history, root.community_rank)

    for rank in RANKS:
        info_key = _info_set_key(rank, root.community_rank, root.history)
        if info_key in root.strategy_sum:
            sums = root.strategy_sum[info_key]
            total = sum(sums.values())
            if total > 0:
                sigma_bar[info_key] = {a: sums[a] / total for a in actions}
            else:
                sigma_bar[info_key] = {a: 1.0 / len(actions) for a in actions}
        else:
            sigma_bar[info_key] = {a: 1.0 / len(actions) for a in actions}

    return sigma_bar


def extract_full_tree_strategy(root: SearchNode) -> dict[str, dict[str, float]]:
    """Recursively extract average strategy from ALL nodes in the search tree."""
    strategy: dict[str, dict[str, float]] = {}
    _collect_strategies(root, strategy)
    return strategy


def _collect_strategies(
    node: SearchNode, strategy: dict[str, dict[str, float]]
) -> None:
    """Recursively collect average strategies from all expanded nodes."""
    if node.node_type == NodeType.TERMINAL:
        return
    if not node.expanded:
        return

    if node.node_type == NodeType.PLAYER:
        actions = _legal_actions(node.history, node.community_rank)
        for rank in RANKS:
            info_key = _info_set_key(rank, node.community_rank, node.history)
            if info_key in node.strategy_sum:
                sums = node.strategy_sum[info_key]
                total = sum(sums.values())
                if total > 0:
                    strategy[info_key] = {a: sums[a] / total for a in actions}
                else:
                    strategy[info_key] = {a: 1.0 / len(actions) for a in actions}
            else:
                strategy[info_key] = {a: 1.0 / len(actions) for a in actions}

    for child in node.children.values():
        _collect_strategies(child, strategy)


def _extract_cfvs(
    root: SearchNode,
    acting_player: int,
    consistent_deals: list[tuple[int, int]],
) -> dict[str, float]:
    """
    Extract search-refined counterfactual values at the root.
    For each rank the acting player could hold, compute the expected value
    under the converged strategy (weighted over opponent's possible cards).
    """
    cfvs: dict[str, float] = {}
    actions = _legal_actions(root.history, root.community_rank)

    for rank in RANKS:
        info_key = _info_set_key(rank, root.community_rank, root.history)
        strategy = _get_current_strategy(root, info_key, actions)

        total_value = 0.0
        n_deals = 0

        for p1_card, p2_card in consistent_deals:
            acting_card = p1_card if acting_player == PLAYER_1 else p2_card
            if card_rank(acting_card) != rank:
                continue

            # Value under current strategy
            value = 0.0
            for action in actions:
                child = root.children.get(action)
                if child is not None:
                    child_val = _node_value(child, acting_player, p1_card, p2_card)
                    value += strategy[action] * child_val
            total_value += value
            n_deals += 1

        cfvs[rank] = total_value / max(1, n_deals)

    return cfvs


def _node_value(
    node: SearchNode,
    eval_player: int,
    p1_card: int,
    p2_card: int,
) -> float:
    """Recursively compute value at a node for eval_player under current strategies."""
    if node.node_type == NodeType.TERMINAL:
        payoff = _terminal_utility(
            node.history, node.community_rank, node.pot, p1_card, p2_card,
        )
        return payoff if eval_player == PLAYER_1 else -payoff

    if node.node_type == NodeType.CHANCE:
        if not node.expanded:
            return 0.0
        rank_probs = _community_chance_probs(p1_card, p2_card)
        value = 0.0
        for rank, prob in rank_probs.items():
            if prob > 0 and rank in node.children:
                value += prob * _node_value(node.children[rank], eval_player, p1_card, p2_card)
        return value

    if not node.expanded:
        if not node.nn_values:
            return 0.0
        rank = card_rank(p1_card if eval_player == PLAYER_1 else p2_card)
        return node.nn_values.get(rank, 0.0)

    acting_player = _current_player(node.history, node.community_rank)
    actions = _legal_actions(node.history, node.community_rank)
    acting_rank = card_rank(p1_card if acting_player == PLAYER_1 else p2_card)
    info_key = _info_set_key(acting_rank, node.community_rank, node.history)
    strategy = _get_current_strategy(node, info_key, actions)

    value = 0.0
    for action in actions:
        child = node.children[action]
        value += strategy[action] * _node_value(child, eval_player, p1_card, p2_card)
    return value
