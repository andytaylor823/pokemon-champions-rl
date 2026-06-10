"""
GT-CFR inner-loop search for Kuhn Poker.

Implements the Player of Games search algorithm (Schmid et al. 2021):
  1. Build a card-agnostic search tree over the public action history
  2. Run CFR+ traversals over all consistent deals simultaneously
  3. At terminals use real payoffs; at unexpanded leaves use CVPN estimates
  4. Output the time-averaged strategy (sigma_bar) and search-refined CFVs

The search tree is EPHEMERAL -- built fresh for one decision, then discarded.
The tree is indexed by action history (public state), NOT by specific cards.
This prevents strategy fusion: the search never pretends to know hidden info.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations

import numpy as np
import torch

from toy_examples.kuhn_poker.game import (
    CARDS,
    PLAYER_1,
    PLAYER_2,
    KuhnState,
    is_terminal,
    current_player,
    legal_actions,
    terminal_utility,
    make_info_set_key,
)
from toy_examples.kuhn_poker.network import (
    KuhnCVPN,
    encode_public_state,
    get_policy_for_info_set,
    get_value_for_info_set,
)

# ---------------------------------------------------------------------------
# Search tree node
# ---------------------------------------------------------------------------


@dataclass
class SearchNode:
    """
    One node in the ephemeral search tree, representing a public state.
    Stores CFR+ accumulators (regrets, strategy sums) for every info set
    (card + history combination) that passes through it.
    """

    history: tuple[str, ...]
    children: dict[str, "SearchNode"] = field(default_factory=dict)
    expanded: bool = False
    # Per-info-set CFR+ state
    cumulative_regret: dict[str, dict[str, float]] = field(default_factory=dict)
    strategy_sum: dict[str, dict[str, float]] = field(default_factory=dict)
    visit_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    # Cached CVPN outputs (populated on expansion)
    policy_prior: dict[str, dict[str, float]] = field(default_factory=dict)
    nn_values: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# CFR+ regret matching
# ---------------------------------------------------------------------------


def _get_current_strategy(node: SearchNode, info_key: str, actions: list[str]) -> dict[str, float]:
    """Derive the current iteration's strategy from positive cumulative regrets."""
    if info_key not in node.cumulative_regret:
        return {a: 1.0 / len(actions) for a in actions}

    regrets = node.cumulative_regret[info_key]
    positive_sum = sum(max(0.0, regrets.get(a, 0.0)) for a in actions)

    if positive_sum > 0:
        return {a: max(0.0, regrets.get(a, 0.0)) / positive_sum for a in actions}
    return {a: 1.0 / len(actions) for a in actions}


# ---------------------------------------------------------------------------
# Tree expansion (CVPN evaluation)
# ---------------------------------------------------------------------------


def _evaluate_node(node: SearchNode, net: KuhnCVPN) -> None:
    """Query the CVPN and cache policy priors + value estimates for all private states."""
    history = node.history
    acting = current_player(history)
    actions = legal_actions(history)

    encoded = encode_public_state(history, acting)
    x = torch.tensor(encoded, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        policy_logits, values = net(x)

    for card in CARDS:
        info_key = make_info_set_key(card, history)
        node.policy_prior[info_key] = get_policy_for_info_set(policy_logits[0], card, actions)
        node.nn_values[card] = get_value_for_info_set(values[0], card)


def _expand_node(node: SearchNode, net: KuhnCVPN) -> None:
    """Expand a node: create children for all legal actions, evaluate with CVPN."""
    if is_terminal(node.history):
        return
    _evaluate_node(node, net)
    for action in legal_actions(node.history):
        node.children[action] = SearchNode(history=node.history + (action,))
    node.expanded = True


def _expand_tree_fully(node: SearchNode, net: KuhnCVPN, max_depth: int = 10) -> None:
    """Recursively expand all non-terminal nodes (for small games like Kuhn)."""
    if max_depth <= 0 or is_terminal(node.history):
        return
    if not node.expanded:
        _expand_node(node, net)
    for child in node.children.values():
        _expand_tree_fully(child, net, max_depth - 1)


# ---------------------------------------------------------------------------
# CFR+ traversal
# ---------------------------------------------------------------------------


def _cfr_traverse(
    node: SearchNode,
    net: KuhnCVPN,
    traversing_player: int,
    p1_card: str,
    p2_card: str,
    reach_p1: float,
    reach_p2: float,
    iteration: int,
) -> float:
    """
    One CFR+ traversal for a specific deal. Returns the counterfactual value
    for traversing_player. Updates regrets only at traversing_player's info sets.
    """
    history = node.history

    # Terminal: real payoff
    if is_terminal(history):
        payoff = terminal_utility(history, p1_card, p2_card)
        return payoff if traversing_player == PLAYER_1 else -payoff

    # Unexpanded leaf: CVPN value estimate
    if not node.expanded:
        if not node.nn_values:
            _evaluate_node(node, net)
        card = p1_card if traversing_player == PLAYER_1 else p2_card
        return node.nn_values.get(card, 0.0)

    acting = current_player(history)
    actions = legal_actions(history)
    card = p1_card if acting == PLAYER_1 else p2_card
    info_key = make_info_set_key(card, history)

    strategy = _get_current_strategy(node, info_key, actions)

    # Recurse into each action, tracking reach probabilities
    action_values: dict[str, float] = {}
    node_value = 0.0
    for action in actions:
        if acting == PLAYER_1:
            child_val = _cfr_traverse(
                node.children[action], net, traversing_player,
                p1_card, p2_card, reach_p1 * strategy[action], reach_p2, iteration,
            )
        else:
            child_val = _cfr_traverse(
                node.children[action], net, traversing_player,
                p1_card, p2_card, reach_p1, reach_p2 * strategy[action], iteration,
            )
        action_values[action] = child_val
        node_value += strategy[action] * child_val

    # Update regrets and strategy sums at the traversing player's nodes
    if acting == traversing_player:
        if info_key not in node.cumulative_regret:
            node.cumulative_regret[info_key] = {a: 0.0 for a in actions}
        if info_key not in node.strategy_sum:
            node.strategy_sum[info_key] = {a: 0.0 for a in actions}
        if info_key not in node.visit_counts:
            node.visit_counts[info_key] = {a: 0 for a in actions}

        # Counterfactual reach = opponent's reach contribution
        cf_reach = reach_p2 if traversing_player == PLAYER_1 else reach_p1

        for action in actions:
            # CFR+ regret clipping: floor at zero
            instant_regret = action_values[action] - node_value
            node.cumulative_regret[info_key][action] = max(
                0.0, node.cumulative_regret[info_key][action] + cf_reach * instant_regret
            )

        # Linear-weighted strategy accumulation (CFR+ averaging)
        player_reach = reach_p1 if traversing_player == PLAYER_1 else reach_p2
        for action in actions:
            node.strategy_sum[info_key][action] += iteration * player_reach * strategy[action]
            node.visit_counts[info_key][action] += 1

    return node_value


# ---------------------------------------------------------------------------
# PUCT-guided incremental expansion (for larger games)
# ---------------------------------------------------------------------------


def _puct_expand(node: SearchNode, net: KuhnCVPN, c_puct: float) -> None:
    """Walk the tree via PUCT scores and expand the first unexpanded leaf."""
    current = node
    while current.expanded and not is_terminal(current.history):
        actions = legal_actions(current.history)
        if not actions:
            break

        best_action = None
        best_score = float("-inf")
        for action in actions:
            score = 0.0
            for card in CARDS:
                info_key = make_info_set_key(card, current.history)
                strat = _get_current_strategy(current, info_key, actions)
                prior = current.policy_prior.get(info_key, {}).get(action, 1.0 / len(actions))
                n_a = current.visit_counts.get(info_key, {}).get(action, 0)
                n_total = sum(current.visit_counts.get(info_key, {}).get(a, 0) for a in actions)
                score += strat[action] + c_puct * prior * np.sqrt(max(1, n_total)) / (1 + n_a)
            avg_score = score / len(CARDS)
            if avg_score > best_score:
                best_score = avg_score
                best_action = action

        current = current.children[best_action]

    if not is_terminal(current.history) and not current.expanded:
        _expand_node(current, net)


# ---------------------------------------------------------------------------
# Output extraction
# ---------------------------------------------------------------------------


def _extract_average_strategy(root: SearchNode) -> dict[str, dict[str, float]]:
    """Extract sigma_bar (time-weighted average strategy) at the root node only."""
    sigma_bar: dict[str, dict[str, float]] = {}
    actions = legal_actions(root.history)

    for card in CARDS:
        info_key = make_info_set_key(card, root.history)
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
    """Recursively extract sigma_bar from ALL expanded nodes in the tree."""
    strategy: dict[str, dict[str, float]] = {}
    _collect_strategies(root, strategy)
    return strategy


def _collect_strategies(node: SearchNode, strategy: dict[str, dict[str, float]]) -> None:
    if is_terminal(node.history) or not node.expanded:
        return
    actions = legal_actions(node.history)
    for card in CARDS:
        info_key = make_info_set_key(card, node.history)
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
    consistent_deals: list[tuple[str, str]],
) -> dict[str, float]:
    """Compute search-refined CFVs at the root under the converged strategy."""
    cfvs: dict[str, float] = {}
    actions = legal_actions(root.history)

    for card in CARDS:
        info_key = make_info_set_key(card, root.history)
        strategy = _get_current_strategy(root, info_key, actions)

        total_value = 0.0
        n_opponents = 0
        for p1_card, p2_card in consistent_deals:
            acting_card = p1_card if acting_player == PLAYER_1 else p2_card
            if acting_card != card:
                continue
            value = 0.0
            for action in actions:
                child = root.children.get(action)
                child_val = _node_value(child, acting_player, p1_card, p2_card) if child else 0.0
                value += strategy[action] * child_val
            total_value += value
            n_opponents += 1

        cfvs[card] = total_value / max(1, n_opponents)

    return cfvs


def _node_value(node: SearchNode, eval_player: int, p1_card: str, p2_card: str) -> float:
    """Recursively compute the value at a node under current strategies."""
    history = node.history

    if is_terminal(history):
        payoff = terminal_utility(history, p1_card, p2_card)
        return payoff if eval_player == PLAYER_1 else -payoff

    if not node.expanded:
        if not node.nn_values:
            return 0.0
        card = p1_card if eval_player == PLAYER_1 else p2_card
        return node.nn_values.get(card, 0.0)

    acting = current_player(history)
    actions = legal_actions(history)
    card = p1_card if acting == PLAYER_1 else p2_card
    info_key = make_info_set_key(card, history)
    strategy = _get_current_strategy(node, info_key, actions)

    return sum(strategy[a] * _node_value(node.children[a], eval_player, p1_card, p2_card)
               for a in actions)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def gt_cfr_search(
    root_state: KuhnState,
    net: KuhnCVPN,
    n_iterations: int = 100,
    c_puct: float = 2.0,
    expansion_interval: int = 10,
    full_expand: bool = True,
) -> tuple[dict[str, dict[str, float]], dict[str, float], dict[str, dict[str, float]]]:
    """
    Run GT-CFR search from a root state.

    Args:
        root_state: provides the action history context
        net: CVPN for leaf evaluation and policy priors
        n_iterations: CFR+ iteration budget
        c_puct: PUCT exploration constant (incremental mode only)
        expansion_interval: expand one leaf every N iterations (incremental mode)
        full_expand: expand entire tree upfront (recommended for small games)

    Returns:
        sigma_bar: average strategy at root {info_set_key: {action: prob}}
        cfvs: search-refined CFVs {card: value}
        full_strategy: average strategy for ALL info sets in the tree
    """
    history = root_state.history
    if is_terminal(history):
        return {}, {}, {}

    root = SearchNode(history=history)
    if full_expand:
        _expand_tree_fully(root, net)
    else:
        _expand_node(root, net)

    acting_player = current_player(history)
    consistent_deals = list(permutations(CARDS, 2))

    for t in range(1, n_iterations + 1):
        for p1_card, p2_card in consistent_deals:
            _cfr_traverse(root, net, PLAYER_1, p1_card, p2_card, 1.0, 1.0, t)
            _cfr_traverse(root, net, PLAYER_2, p1_card, p2_card, 1.0, 1.0, t)

        if not full_expand and t % expansion_interval == 0 and t < n_iterations:
            _puct_expand(root, net, c_puct)

    sigma_bar = _extract_average_strategy(root)
    cfvs = _extract_cfvs(root, acting_player, consistent_deals)
    full_strategy = extract_full_tree_strategy(root)

    return sigma_bar, cfvs, full_strategy
