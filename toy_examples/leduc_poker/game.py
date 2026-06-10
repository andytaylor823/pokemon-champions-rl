"""
Leduc Hold'em (two-bet-size variant) game environment for GT-CFR.

Leduc Hold'em: 6-card deck {J, Q, K} x 2 suits. Each player antes 1 chip and
receives one private card. After a round of betting, one community card is dealt
face-up, followed by a second round of betting. At showdown, a pair (private card
matches community rank) beats no pair; otherwise the higher private card wins.

Two-bet-size variant: each round offers two bet increments (small and big),
giving 3-4 legal actions per decision node instead of standard Leduc's 2-3.

States are immutable -- apply_action returns a new state.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Deck configuration
# ---------------------------------------------------------------------------

# Card ranks (higher index = stronger)
RANKS = ("J", "Q", "K")
SUITS_PER_RANK = 2

# Build the full deck as (rank, suit_index) pairs; each card has a unique int id
# Deck: [(J,0), (J,1), (Q,0), (Q,1), (K,0), (K,1)]  -> ids 0..5
DECK_SIZE = len(RANKS) * SUITS_PER_RANK


def card_rank(card_id: int) -> str:
    """Return the rank string for a card id."""
    return RANKS[card_id // SUITS_PER_RANK]


def card_rank_index(card_id: int) -> int:
    """Return the rank index (0=J, 1=Q, 2=K) for a card id."""
    return card_id // SUITS_PER_RANK


# ---------------------------------------------------------------------------
# Betting configuration (two-bet-size variant)
# ---------------------------------------------------------------------------

# Bet increment options per round (chips added when betting / raising)
ROUND1_BET_SIZES: tuple[int, ...] = (2, 4)
ROUND2_BET_SIZES: tuple[int, ...] = (4, 8)
MAX_RAISES_PER_ROUND = 2
ANTE = 1

# ---------------------------------------------------------------------------
# Action labels
# ---------------------------------------------------------------------------

CHECK = "check"
BET_SMALL = "bet_small"
BET_BIG = "bet_big"
FOLD = "fold"
CALL = "call"
RAISE_SMALL = "raise_small"
RAISE_BIG = "raise_big"

# All possible action strings (used for encoding)
ALL_ACTIONS = (CHECK, BET_SMALL, BET_BIG, FOLD, CALL, RAISE_SMALL, RAISE_BIG)

# Player indices
PLAYER_1 = 0
PLAYER_2 = 1
CHANCE = -1


# ---------------------------------------------------------------------------
# Betting-round helper: parses a flat action history into round structure
# ---------------------------------------------------------------------------

def _split_rounds(history: tuple[str, ...]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """
    Split a flat history into (round1_actions, round2_actions).
    Round 1 ends when both players have settled (last action is CHECK or CALL
    and each player has acted at least once), or when someone folds.
    """
    # Walk through round 1 until it completes
    r1_actions: list[str] = []
    for i, action in enumerate(history):
        r1_actions.append(action)
        if _round_over(tuple(r1_actions)):
            # Everything after this belongs to round 2
            return tuple(r1_actions), tuple(history[i + 1:])
    # Round 1 is still in progress (no round 2 yet)
    return tuple(r1_actions), ()


def _round_over(round_actions: tuple[str, ...]) -> bool:
    """Check if a single round's action sequence is complete."""
    if not round_actions:
        return False
    # Fold ends the whole game (and thus the round)
    if round_actions[-1] == FOLD:
        return True
    # Check-check ends round 1 opener
    if len(round_actions) >= 2 and round_actions[-1] == CHECK and round_actions[-2] == CHECK:
        return True
    # Any sequence ending in CALL after at least one bet/raise completes the round
    if round_actions[-1] == CALL and _has_bet(round_actions):
        return True
    return False


def _has_bet(round_actions: tuple[str, ...]) -> bool:
    """True if the round contains at least one bet or raise."""
    return any(a in (BET_SMALL, BET_BIG, RAISE_SMALL, RAISE_BIG) for a in round_actions)


def _count_raises(round_actions: tuple[str, ...]) -> int:
    """Count the number of raises (not counting the initial bet) in a round."""
    return sum(1 for a in round_actions if a in (RAISE_SMALL, RAISE_BIG))


def _current_bet_increment(round_actions: tuple[str, ...]) -> int:
    """Return the chip increment of the last bet/raise in this round (0 if none)."""
    for a in reversed(round_actions):
        if a == BET_SMALL or a == RAISE_SMALL:
            return _small_bet_for_round_actions(round_actions)
        if a == BET_BIG or a == RAISE_BIG:
            return _big_bet_for_round_actions(round_actions)
    return 0


def _small_bet_for_round_actions(round_actions: tuple[str, ...]) -> int:
    """Cannot determine round from actions alone; caller must use the state's round."""
    raise NotImplementedError("Use LeducState methods instead")


def _big_bet_for_round_actions(round_actions: tuple[str, ...]) -> int:
    raise NotImplementedError("Use LeducState methods instead")


# ---------------------------------------------------------------------------
# LeducState
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LeducState:
    """
    Immutable representation of a Leduc Hold'em game state.

    Cards are integer ids into the 6-card deck.
    """

    # Private cards: (player1_card_id, player2_card_id); None before deal
    cards: tuple[int, int] | None = None
    # Community card id; None until dealt between rounds
    community_card: int | None = None
    # Flat sequence of player actions (both rounds, chronological)
    history: tuple[str, ...] = ()
    # Chips committed by each player (P1, P2); starts at (ANTE, ANTE)
    pot: tuple[int, int] = (ANTE, ANTE)

    # ------------------------------------------------------------------
    # Phase / terminal detection
    # ------------------------------------------------------------------

    def is_terminal(self) -> bool:
        """Check if the game has ended (fold or showdown after round 2)."""
        if self.cards is None:
            return False
        # Fold ends the game immediately
        if self.history and self.history[-1] == FOLD:
            return True
        # Showdown requires community card to have been dealt
        if self.community_card is None:
            return False
        # Both rounds must be complete for showdown
        r1, r2 = self.round_split()
        if not _round_over(r1):
            return False
        return len(r2) > 0 and _round_over(r2)

    def is_chance_node(self) -> bool:
        """True when round 1 is complete but community card hasn't been dealt."""
        if self.cards is None:
            return False
        if self.community_card is not None:
            return False
        r1, _ = self.round_split()
        # Round 1 ended normally (not by fold) → chance node
        return _round_over(r1) and (not r1 or r1[-1] != FOLD)

    def game_round(self) -> int:
        """Return current round (1 or 2). Round 2 starts after community card is dealt."""
        if self.community_card is not None:
            return 2
        return 1

    def round_split(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Split history into (round1_actions, round2_actions)."""
        return _split_rounds(self.history)

    # ------------------------------------------------------------------
    # Current player
    # ------------------------------------------------------------------

    def current_player(self) -> int:
        """Return PLAYER_1, PLAYER_2, or CHANCE."""
        if self.cards is None:
            return CHANCE
        if self.is_terminal():
            raise ValueError("No current player at terminal state")
        if self.is_chance_node():
            return CHANCE
        # Determine who acts based on the current round's action count
        current_round = self.game_round()
        if current_round == 1:
            round_actions = self.round_split()[0]
        else:
            round_actions = self.round_split()[1]
        # P1 acts first each round; players alternate
        return PLAYER_1 if len(round_actions) % 2 == 0 else PLAYER_2

    # ------------------------------------------------------------------
    # Legal actions
    # ------------------------------------------------------------------

    def legal_actions(self) -> list[str]:
        """Return the list of legal actions at this state."""
        if self.is_terminal():
            return []
        if self.cards is None:
            raise ValueError("Use all_deals() for initial chance node")
        if self.is_chance_node():
            raise ValueError("Use community_outcomes() for community chance node")

        current_round = self.game_round()
        bet_sizes = ROUND1_BET_SIZES if current_round == 1 else ROUND2_BET_SIZES
        round_actions = self.round_split()[0] if current_round == 1 else self.round_split()[1]

        # Has anyone bet/raised this round?
        has_aggression = _has_bet(round_actions)

        if not has_aggression:
            # Unopened: check or bet (small / big)
            actions = [CHECK]
            if len(bet_sizes) >= 1:
                actions.append(BET_SMALL)
            if len(bet_sizes) >= 2:
                actions.append(BET_BIG)
            return actions

        # Facing a bet/raise
        raises_so_far = _count_raises(round_actions)
        can_raise = raises_so_far < MAX_RAISES_PER_ROUND

        actions = [FOLD, CALL]
        if can_raise:
            if len(bet_sizes) >= 1:
                actions.append(RAISE_SMALL)
            if len(bet_sizes) >= 2:
                actions.append(RAISE_BIG)
        return actions

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def apply_action(self, action: str) -> LeducState:
        """Return a new state after applying a player action."""
        assert action in self.legal_actions(), (
            f"Illegal action '{action}' at history {self.history}"
        )
        new_pot = self._updated_pot(action)
        return LeducState(
            cards=self.cards,
            community_card=self.community_card,
            history=self.history + (action,),
            pot=new_pot,
        )

    def apply_chance(self, community_card_id: int) -> LeducState:
        """Deal the community card (chance transition between rounds)."""
        assert self.is_chance_node(), "Not a chance node"
        assert self.cards is not None
        assert community_card_id not in self.cards, (
            f"Community card {community_card_id} already dealt to a player"
        )
        return LeducState(
            cards=self.cards,
            community_card=community_card_id,
            history=self.history,
            pot=self.pot,
        )

    def _updated_pot(self, action: str) -> tuple[int, int]:
        """Compute new pot after a player action."""
        p1, p2 = self.pot
        player = self.current_player()
        current_round = self.game_round()
        bet_sizes = ROUND1_BET_SIZES if current_round == 1 else ROUND2_BET_SIZES
        small_inc, big_inc = (bet_sizes[0], bet_sizes[-1])

        if action == CHECK:
            return (p1, p2)
        elif action == FOLD:
            return (p1, p2)
        elif action == CALL:
            # Match opponent's commitment
            if player == PLAYER_1:
                return (p2, p2)  # P1 matches P2
            else:
                return (p1, p1)  # P2 matches P1
        elif action in (BET_SMALL, RAISE_SMALL):
            # Add small increment on top of matching
            opponent_pot = p2 if player == PLAYER_1 else p1
            new_commitment = opponent_pot + small_inc
            if player == PLAYER_1:
                return (new_commitment, p2)
            else:
                return (p1, new_commitment)
        elif action in (BET_BIG, RAISE_BIG):
            # Add big increment on top of matching
            opponent_pot = p2 if player == PLAYER_1 else p1
            new_commitment = opponent_pot + big_inc
            if player == PLAYER_1:
                return (new_commitment, p2)
            else:
                return (p1, new_commitment)
        else:
            raise ValueError(f"Unknown action: {action}")

    # ------------------------------------------------------------------
    # Terminal utility
    # ------------------------------------------------------------------

    def terminal_utility(self, player: int) -> float:
        """
        Return the payoff for `player` at a terminal node.
        Payoff = chips won minus chips invested.
        """
        assert self.is_terminal(), "Not a terminal state"
        assert self.cards is not None

        p1_committed, p2_committed = self.pot
        total_pot = p1_committed + p2_committed

        # Fold: last actor folded, the other player wins
        if self.history[-1] == FOLD:
            # Who folded? The player who was about to act
            r1, r2 = self.round_split()
            current_round = 2 if self.community_card is not None and _round_over(r1) else 1
            round_actions = r1 if current_round == 1 else r2
            folder = PLAYER_1 if len(round_actions) % 2 == 1 else PLAYER_2
            winner = PLAYER_2 if folder == PLAYER_1 else PLAYER_1
        else:
            # Showdown
            winner = self._showdown_winner()

        # Compute payoff: winner gets the whole pot minus what they put in
        if player == winner:
            return float(total_pot - self.pot[player])
        else:
            return float(-self.pot[player])

    def _showdown_winner(self) -> int:
        """Determine winner at showdown (pair beats no pair, else higher card)."""
        assert self.cards is not None and self.community_card is not None

        p1_rank = card_rank_index(self.cards[0])
        p2_rank = card_rank_index(self.cards[1])
        comm_rank = card_rank_index(self.community_card)

        p1_pair = p1_rank == comm_rank
        p2_pair = p2_rank == comm_rank

        if p1_pair and not p2_pair:
            return PLAYER_1
        if p2_pair and not p1_pair:
            return PLAYER_2
        # Both pair or neither: higher private card wins
        if p1_rank > p2_rank:
            return PLAYER_1
        if p2_rank > p1_rank:
            return PLAYER_2
        # Exact tie (same rank, no pair differentiation) — split pot effectively
        # This shouldn't happen since if both pair they have the same rank card
        # which is impossible (only 2 copies per rank, both are held).
        # If neither pairs, equal rank means a true tie.
        return PLAYER_1  # arbitrary tie-break (won't affect EV in expectation)

    # ------------------------------------------------------------------
    # Information set keys
    # ------------------------------------------------------------------

    def info_set_key(self) -> str:
        """
        Info set key for the current player.
        Format: "<private_rank>:<community_rank_or_?>:<action_history>"
        """
        assert self.cards is not None, "No info set at initial chance node"
        player = self.current_player()
        private_rank = card_rank(self.cards[player])
        comm_str = card_rank(self.community_card) if self.community_card is not None else "?"
        history_str = ",".join(self.history) if self.history else ""
        return f"{private_rank}:{comm_str}:{history_str}"

    def public_state_key(self) -> str:
        """
        Public state key (visible to both players).
        Format: "<community_rank_or_?>:<action_history>"
        """
        comm_str = card_rank(self.community_card) if self.community_card is not None else "?"
        history_str = ",".join(self.history) if self.history else ""
        return f"{comm_str}:{history_str}"


# ---------------------------------------------------------------------------
# Deal and community card enumeration
# ---------------------------------------------------------------------------

def all_deals() -> list[LeducState]:
    """
    Enumerate all ordered deals (p1_card, p2_card) from the 6-card deck.
    P(6,2) = 30 deals, each with probability 1/30.
    """
    deals = []
    for p1 in range(DECK_SIZE):
        for p2 in range(DECK_SIZE):
            if p1 != p2:
                deals.append(LeducState(cards=(p1, p2)))
    return deals


def chance_probability() -> float:
    """Probability of any specific ordered deal: 1 / P(6,2) = 1/30."""
    return 1.0 / (DECK_SIZE * (DECK_SIZE - 1))


def community_outcomes(state: LeducState) -> list[tuple[int, float]]:
    """
    Return (community_card_id, probability) pairs for the chance node.
    Each remaining card is equally likely.
    """
    assert state.is_chance_node()
    assert state.cards is not None
    remaining = [c for c in range(DECK_SIZE) if c not in state.cards]
    prob = 1.0 / len(remaining)
    return [(c, prob) for c in remaining]


def community_rank_outcomes(state: LeducState) -> list[tuple[str, float, list[int]]]:
    """
    Group community outcomes by rank for the search tree (which is rank-indexed).
    Returns (rank, probability, [card_ids]) for each possible community rank.
    """
    assert state.is_chance_node()
    assert state.cards is not None
    remaining = [c for c in range(DECK_SIZE) if c not in state.cards]
    # Group by rank
    rank_groups: dict[str, list[int]] = {}
    for c in remaining:
        r = card_rank(c)
        rank_groups.setdefault(r, []).append(c)
    total = len(remaining)
    return [(rank, len(ids) / total, ids) for rank, ids in rank_groups.items()]


# ---------------------------------------------------------------------------
# Enumeration helpers (cached for performance)
# ---------------------------------------------------------------------------

# Cache: {player: [info_set_key, ...]} and {info_set_key: [actions]}
_info_set_cache: dict[int, list[str]] | None = None
_actions_cache: dict[str, list[str]] = {}


def _build_info_set_cache() -> None:
    """Build the info set and actions caches via a single full tree traversal."""
    global _info_set_cache
    info_sets: dict[int, set[str]] = {PLAYER_1: set(), PLAYER_2: set()}

    def _traverse(state: LeducState) -> None:
        if state.is_terminal():
            return
        if state.is_chance_node():
            for comm_id, _ in community_outcomes(state):
                _traverse(state.apply_chance(comm_id))
            return
        player = state.current_player()
        key = state.info_set_key()
        if key not in _actions_cache:
            _actions_cache[key] = state.legal_actions()
        info_sets[player].add(key)
        for action in state.legal_actions():
            _traverse(state.apply_action(action))

    for deal in all_deals():
        _traverse(deal)

    _info_set_cache = {p: sorted(keys) for p, keys in info_sets.items()}


def all_info_set_keys() -> dict[int, list[str]]:
    """
    Enumerate all information set keys in the game.
    Results are cached after the first call (~3s to compute, then instant).
    """
    if _info_set_cache is None:
        _build_info_set_cache()
    return _info_set_cache  # type: ignore[return-value]


def actions_at_info_set(info_set_key: str) -> list[str]:
    """Return the legal actions at an info set (cached from tree traversal)."""
    if _info_set_cache is None:
        _build_info_set_cache()
    return _actions_cache.get(info_set_key, [])
