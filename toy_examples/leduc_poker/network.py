"""
CVPN (Counterfactual Value + Policy Network) for Leduc Hold'em.

Maps a public belief state (action history + community card + opponent belief)
to:
  - Policy head: action probabilities per possible private card rank (guides search)
  - Value head: one counterfactual value per possible private rank (leaf evaluation)

Architecture: 2-layer MLP with 128 hidden units. Input encodes:
  - Action history: one-hot per slot (up to 12 actions x 7 action types = 84 dims)
  - Community card: 3-dim rank one-hot + 1 binary dealt flag = 4 dims
  - Acting player: 2-dim one-hot
  - Belief over opponent's rank: 3 floats
  Total INPUT_DIM = 93
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from toy_examples.leduc_poker.game import (
    RANKS,
    ALL_ACTIONS,
    LeducState,
    card_rank_index,
)

# ---------------------------------------------------------------------------
# Encoding constants
# ---------------------------------------------------------------------------

# 7 distinct action types
NUM_ACTION_TYPES = len(ALL_ACTIONS)
# Buffer for max action sequence across both rounds (2 rounds x ~5 actions each + margin)
MAX_HISTORY_LEN = 12
# Number of card ranks
NUM_RANKS = len(RANKS)

# Input: history one-hots + community card + acting player + belief
INPUT_DIM = MAX_HISTORY_LEN * NUM_ACTION_TYPES + (NUM_RANKS + 1) + 2 + NUM_RANKS  # 84+4+2+3 = 93

# Output dimensions
NUM_PRIVATE_STATES = NUM_RANKS  # one row per rank the acting player could hold
NUM_ACTIONS = 4  # max legal actions at any Leduc decision node

# Maps action strings to one-hot indices for history encoding
ACTION_TO_IDX = {a: i for i, a in enumerate(ALL_ACTIONS)}

# Max payoff for value head scaling (rough upper bound on chips won/lost)
MAX_PAYOFF = 25.0


class LeducCVPN(nn.Module):
    """
    Counterfactual Value + Policy Network for Leduc Hold'em.

    Input: encoded public belief state [batch, INPUT_DIM]
    Output:
      - policy_logits: [batch, NUM_PRIVATE_STATES, NUM_ACTIONS]
      - values: [batch, NUM_PRIVATE_STATES] (bounded to +/- MAX_PAYOFF)
    """

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        # Shared backbone (deeper than Kuhn's to handle richer input)
        self.shared = nn.Sequential(
            nn.Linear(INPUT_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # Policy head: logits for each private rank's action choice
        self.policy_head = nn.Linear(hidden_dim, NUM_PRIVATE_STATES * NUM_ACTIONS)
        # Value head: one CFV per private rank
        self.value_head = nn.Linear(hidden_dim, NUM_PRIVATE_STATES)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Args:
            x: [batch, INPUT_DIM] encoded public belief state
        Returns:
            policy_logits: [batch, NUM_PRIVATE_STATES, NUM_ACTIONS]
            values: [batch, NUM_PRIVATE_STATES] bounded to [-MAX_PAYOFF, MAX_PAYOFF]
        """
        # Shared representation
        h = self.shared(x)
        # Policy: reshape to per-private-state action logits
        policy_logits = self.policy_head(h).view(-1, NUM_PRIVATE_STATES, NUM_ACTIONS)
        # Value: tanh scales to payoff range
        values = torch.tanh(self.value_head(h)) * MAX_PAYOFF
        return policy_logits, values


# ---------------------------------------------------------------------------
# State encoding
# ---------------------------------------------------------------------------

def encode_public_state(state: LeducState, acting_player: int) -> np.ndarray:
    """
    Encode a public state into a feature vector for the CVPN.

    The encoding uses ONLY public information:
    - Action history (one-hot per slot)
    - Community card rank (one-hot + dealt flag)
    - Acting player (one-hot)
    - Belief over opponent's rank (uniform by default)

    The acting player's private card is NOT encoded -- the network outputs a
    vector over all possible private ranks, and the correct row is selected at
    readout time.
    """
    features = np.zeros(INPUT_DIM, dtype=np.float32)
    offset = 0

    # --- Action history: one-hot per slot ---
    for i, action in enumerate(state.history):
        if i < MAX_HISTORY_LEN:
            features[offset + i * NUM_ACTION_TYPES + ACTION_TO_IDX[action]] = 1.0
    offset += MAX_HISTORY_LEN * NUM_ACTION_TYPES  # 84

    # --- Community card: rank one-hot (3) + dealt flag (1) ---
    if state.community_card is not None:
        rank_idx = card_rank_index(state.community_card)
        features[offset + rank_idx] = 1.0
        features[offset + NUM_RANKS] = 1.0  # dealt flag
    offset += NUM_RANKS + 1  # 4

    # --- Acting player: one-hot (2) ---
    features[offset + acting_player] = 1.0
    offset += 2

    # --- Belief over opponent's rank: uniform (3) ---
    features[offset: offset + NUM_RANKS] = 1.0 / NUM_RANKS
    offset += NUM_RANKS

    assert offset == INPUT_DIM, f"Encoding offset {offset} != INPUT_DIM {INPUT_DIM}"
    return features


def encode_public_state_with_belief(
    state: LeducState,
    acting_player: int,
    belief: np.ndarray,
) -> np.ndarray:
    """
    Encode public state with a specific belief vector over opponent's rank.
    The belief vector should have length NUM_RANKS and sum to 1.
    """
    features = encode_public_state(state, acting_player)
    # Overwrite the belief portion
    belief_offset = MAX_HISTORY_LEN * NUM_ACTION_TYPES + (NUM_RANKS + 1) + 2
    features[belief_offset: belief_offset + NUM_RANKS] = belief
    return features


# ---------------------------------------------------------------------------
# Policy / value readout helpers
# ---------------------------------------------------------------------------

def get_policy_for_info_set(
    policy_logits: torch.Tensor,
    rank: str,
    legal_actions: list[str],
) -> dict[str, float]:
    """
    Extract action probabilities for a specific info set from network output.

    Args:
        policy_logits: [NUM_PRIVATE_STATES, NUM_ACTIONS] from one forward pass
        rank: the private card rank for this info set (J, Q, or K)
        legal_actions: legal actions at this info set (length 2-4)

    Returns:
        dict mapping action -> probability
    """
    rank_idx = RANKS.index(rank)
    # Get logits for this private rank
    logits = policy_logits[rank_idx]  # [NUM_ACTIONS]

    # Mask illegal action slots to -inf
    masked_logits = torch.full_like(logits, float("-inf"))
    for i, action in enumerate(legal_actions):
        if i < NUM_ACTIONS:
            masked_logits[i] = logits[i]

    # Softmax to get probabilities
    probs = F.softmax(masked_logits, dim=0)

    # Build result dict
    result = {}
    for i, action in enumerate(legal_actions):
        if i < NUM_ACTIONS:
            result[action] = probs[i].item()
    return result


def get_value_for_info_set(values: torch.Tensor, rank: str) -> float:
    """
    Extract the CFV for a specific private rank from the network output.

    Args:
        values: [NUM_PRIVATE_STATES] values from one forward pass
        rank: the private card rank (J, Q, or K)

    Returns:
        The counterfactual value estimate
    """
    rank_idx = RANKS.index(rank)
    return values[rank_idx].item()
