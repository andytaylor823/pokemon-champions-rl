"""
Kuhn Poker game rules and state representation.

Kuhn Poker: 3-card deck {J, Q, K} (J<Q<K). Each player antes 1 chip and receives
one card. Player 1 acts first (bet or check). The game tree has 12 information sets
(6 per player), each with 2 legal actions.

This module exposes two layers:
  1. Pure-history functions (is_terminal, current_player, legal_actions, terminal_utility)
     that operate on raw action tuples. These are the canonical game rules used by both
     the search module and the state class.
  2. KuhnState -- an immutable convenience wrapper for full-game simulation (used by
     exploitability computation and self-play game rollouts).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CARDS = ("J", "Q", "K")

BET = "bet"
CHECK = "check"
CALL = "call"
FOLD = "fold"

PLAYER_1 = 0
PLAYER_2 = 1
CHANCE = -1

# All five terminal action sequences
_TERMINALS = frozenset([
    (BET, CALL), (BET, FOLD),
    (CHECK, CHECK),
    (CHECK, BET, CALL), (CHECK, BET, FOLD),
])

# ---------------------------------------------------------------------------
# Pure-history game rules (the single source of truth)
# ---------------------------------------------------------------------------


def is_terminal(history: tuple[str, ...]) -> bool:
    """True if the action sequence represents a finished game."""
    return history in _TERMINALS


def current_player(history: tuple[str, ...]) -> int:
    """Which player acts next at a non-terminal history (PLAYER_1 or PLAYER_2)."""
    if len(history) == 0:
        return PLAYER_1
    if len(history) == 1:
        return PLAYER_2
    if history == (CHECK, BET):
        return PLAYER_1
    raise ValueError(f"No acting player for terminal/invalid history: {history}")


def legal_actions(history: tuple[str, ...]) -> list[str]:
    """Legal actions available at a non-terminal history."""
    if is_terminal(history):
        return []
    if len(history) == 0:
        return [BET, CHECK]
    if len(history) == 1:
        return [CALL, FOLD] if history[0] == BET else [BET, CHECK]
    if history == (CHECK, BET):
        return [CALL, FOLD]
    return []


def terminal_utility(history: tuple[str, ...], p1_card: str, p2_card: str) -> float:
    """
    P1's payoff at a terminal history for a specific deal.
    Negate for P2's payoff (zero-sum).
    """
    p1_wins = CARDS.index(p1_card) > CARDS.index(p2_card)

    if history == (BET, FOLD):
        return 1.0
    if history == (CHECK, BET, FOLD):
        return -1.0
    if history == (CHECK, CHECK):
        return 1.0 if p1_wins else -1.0
    if history in ((BET, CALL), (CHECK, BET, CALL)):
        return 2.0 if p1_wins else -2.0
    raise ValueError(f"Not a terminal history: {history}")


def make_info_set_key(card: str, history: tuple[str, ...]) -> str:
    """
    Build the info-set key: the acting player's card + the public action history.
    Format: "K:" (root), "J:bet" (P2 facing a bet), "Q:check,bet" (P1 facing re-bet).
    """
    return f"{card}:{','.join(history)}" if history else f"{card}:"


def parse_info_set_key(key: str) -> tuple[str, tuple[str, ...]]:
    """Inverse of make_info_set_key. Returns (card, history_tuple)."""
    card, history_str = key.split(":")
    history = tuple(history_str.split(",")) if history_str else ()
    return card, history


# ---------------------------------------------------------------------------
# KuhnState -- immutable convenience wrapper
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KuhnState:
    """
    Immutable full-information game state (specific deal + action history).
    Used for game simulation (exploitability, self-play rollouts) where we
    need card-specific payoffs and info-set keys.
    """

    cards: tuple[str, str]
    history: tuple[str, ...] = ()

    def is_terminal(self) -> bool:
        return is_terminal(self.history)

    def current_player(self) -> int:
        return current_player(self.history)

    def legal_actions(self) -> list[str]:
        return legal_actions(self.history)

    def apply_action(self, action: str) -> KuhnState:
        return KuhnState(cards=self.cards, history=self.history + (action,))

    def terminal_utility(self, player: int) -> float:
        """Payoff for `player` at a terminal state."""
        p1_payoff = terminal_utility(self.history, self.cards[0], self.cards[1])
        return p1_payoff if player == PLAYER_1 else -p1_payoff

    def info_set_key(self) -> str:
        """Info-set key for the currently acting player."""
        player = self.current_player()
        card = self.cards[player]
        return make_info_set_key(card, self.history)


# ---------------------------------------------------------------------------
# Chance-node helpers
# ---------------------------------------------------------------------------


def all_deals() -> list[KuhnState]:
    """All 6 possible deals (P(3,2) ordered pairs from {J, Q, K})."""
    return [KuhnState(cards=(p1, p2)) for p1, p2 in permutations(CARDS, 2)]


def chance_probability() -> float:
    """Uniform probability per deal: 1/6."""
    return 1.0 / 6.0


# ---------------------------------------------------------------------------
# Info-set enumeration
# ---------------------------------------------------------------------------


def all_info_set_keys() -> dict[int, list[str]]:
    """Enumerate all 12 info-set keys (6 per player), sorted."""
    info_sets: dict[int, set[str]] = {PLAYER_1: set(), PLAYER_2: set()}

    def _traverse(state: KuhnState) -> None:
        if state.is_terminal():
            return
        player = state.current_player()
        info_sets[player].add(state.info_set_key())
        for action in state.legal_actions():
            _traverse(state.apply_action(action))

    for deal in all_deals():
        _traverse(deal)

    return {p: sorted(keys) for p, keys in info_sets.items()}
