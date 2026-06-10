"""
CVPN (Counterfactual Value + Policy Network) for Kuhn Poker.

Maps a public belief state to:
  - Policy head: per-private-state action logits (guides PUCT exploration)
  - Value head: per-private-state CFVs (leaf evaluation)

Architecture: tiny MLP (2 hidden layers, 64 units) -- deliberately over-sized for
Kuhn to mirror how the real Pokemon system will work at scale.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from toy_examples.kuhn_poker.game import (
    CARDS,
    BET,
    CHECK,
    CALL,
    FOLD,
)

# ---------------------------------------------------------------------------
# Encoding constants
# ---------------------------------------------------------------------------

NUM_ACTION_TYPES = 4  # bet, check, call, fold
MAX_HISTORY_LEN = 3
NUM_CARDS = len(CARDS)
# Input: action history one-hot (3*4=12) + acting player one-hot (2) + belief (3) = 17
INPUT_DIM = MAX_HISTORY_LEN * NUM_ACTION_TYPES + 2 + NUM_CARDS
NUM_ACTIONS = 2  # max legal actions at any Kuhn info set

ACTION_TO_IDX = {BET: 0, CHECK: 1, CALL: 2, FOLD: 3}

# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------


class KuhnCVPN(nn.Module):
    """
    Counterfactual Value + Policy Network.

    Input:  [batch, INPUT_DIM] encoded public belief state
    Output: policy_logits [batch, NUM_CARDS, NUM_ACTIONS], values [batch, NUM_CARDS]
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(INPUT_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_dim, NUM_CARDS * NUM_ACTIONS)
        self.value_head = nn.Linear(hidden_dim, NUM_CARDS)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.shared(x)
        policy_logits = self.policy_head(h).view(-1, NUM_CARDS, NUM_ACTIONS)
        # tanh bounds raw output to [-1,1], then scale to Kuhn payoff range [-2,2]
        values = torch.tanh(self.value_head(h)) * 2.0
        return policy_logits, values


# ---------------------------------------------------------------------------
# State encoding (operates on raw history tuples -- no KuhnState dependency)
# ---------------------------------------------------------------------------


def encode_public_state(history: tuple[str, ...], acting_player: int) -> np.ndarray:
    """
    Encode a public state (action history + acting player) into a flat feature vector.

    Only uses public information -- the acting player's private card is NOT encoded
    (the vector output handles per-card predictions).
    """
    features = np.zeros(INPUT_DIM, dtype=np.float32)

    # One-hot encode each action in the history sequence
    for i, action in enumerate(history):
        if i < MAX_HISTORY_LEN:
            features[i * NUM_ACTION_TYPES + ACTION_TO_IDX[action]] = 1.0

    # One-hot acting player
    player_offset = MAX_HISTORY_LEN * NUM_ACTION_TYPES
    features[player_offset + acting_player] = 1.0

    # Uniform belief (Kuhn's public state never reveals cards)
    belief_offset = player_offset + 2
    features[belief_offset: belief_offset + NUM_CARDS] = 1.0 / NUM_CARDS

    return features


# ---------------------------------------------------------------------------
# Output extraction helpers
# ---------------------------------------------------------------------------


def get_policy_for_info_set(
    policy_logits: torch.Tensor,
    card: str,
    actions: list[str],
) -> dict[str, float]:
    """
    Extract action probabilities for a specific private card from policy logits.
    Applies softmax over the legal action slots.
    """
    card_idx = CARDS.index(card)
    logits = policy_logits[card_idx]

    # Mask to only the legal action slots (positions 0..len(actions)-1)
    masked_logits = torch.full_like(logits, float("-inf"))
    for i in range(min(len(actions), NUM_ACTIONS)):
        masked_logits[i] = logits[i]

    probs = F.softmax(masked_logits, dim=0)
    return {actions[i]: probs[i].item() for i in range(min(len(actions), NUM_ACTIONS))}


def get_value_for_info_set(values: torch.Tensor, card: str) -> float:
    """Extract the CFV for a specific private card from the value head output."""
    return values[CARDS.index(card)].item()
