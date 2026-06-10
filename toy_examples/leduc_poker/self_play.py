"""
Outer training loop for GT-CFR on Leduc Hold'em.

Orchestrates:
  1. Self-play games where both players use GT-CFR search (guided by CVPN)
  2. Collection of training tuples (public_state, search_cfvs, search_sigma_bar)
  3. Neural network training on a replay buffer
  4. Periodic exploitability measurement and strategy logging

The cycle: better net -> better search -> better training data -> better net.

NOTE: Uses incremental PUCT expansion (full_expand=False) for search because
the Leduc tree is too large for full expansion at every decision point during
self-play. Full expansion is reserved for the exploitability module's tabular
solver which pre-caches trees.
"""

from __future__ import annotations

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pydantic import BaseModel, Field

from toy_examples.leduc_poker.game import (
    RANKS,
    SUITS_PER_RANK,
    PLAYER_1,
    PLAYER_2,
    LeducState,
    all_deals,
    all_info_set_keys,
    actions_at_info_set,
    community_outcomes,
    card_rank,
)
from toy_examples.leduc_poker.network import (
    LeducCVPN,
    encode_public_state,
    get_policy_for_info_set,
    NUM_ACTIONS,
    NUM_PRIVATE_STATES,
)
from toy_examples.leduc_poker.gt_cfr_search import gt_cfr_search
from toy_examples.leduc_poker.exploitability import compute_exploitability, Strategy


class TrainingTuple(BaseModel):
    """
    One training sample from a GT-CFR search at a single decision point.
    The outer loop collects these and trains the CVPN to predict the search outputs.
    """

    model_config = {"arbitrary_types_allowed": True}

    public_state: np.ndarray = Field(
        description="Encoded public belief state vector. Shape: [INPUT_DIM]."
    )
    acting_player: int = Field(
        description="Which player is acting (0 = P1, 1 = P2)."
    )
    target_values: np.ndarray = Field(
        description="Search-refined counterfactual values per private rank. Shape: [NUM_PRIVATE_STATES]."
    )
    target_policy: np.ndarray = Field(
        description="Search average strategy per private rank. Shape: [NUM_PRIVATE_STATES, NUM_ACTIONS]."
    )


class TrainingConfig(BaseModel):
    """
    Configuration for the GT-CFR self-play training loop.

    Tuned for Leduc Hold'em (much larger than Kuhn):
    - More generations needed for convergence (~50-200+)
    - Larger replay buffer for diverse data
    - Incremental search (PUCT) since full tree expansion is too expensive
    """

    # --- Outer loop ---
    n_generations: int = Field(
        default=100,
        description="Total training generations. Leduc needs 50-200+ vs Kuhn's ~30."
    )
    games_per_generation: int = Field(
        default=60,
        description="Self-play games per generation. Each game yields ~4-8 training tuples."
    )

    # --- Inner loop: GT-CFR search ---
    search_iterations: int = Field(
        default=50,
        description="CFR+ iterations per search. Incremental expansion means each iter "
        "is fast (~1ms), so 50 iters is plenty for rough policy improvement."
    )
    c_puct: float = Field(
        default=2.0,
        description="PUCT exploration constant for incremental tree expansion."
    )
    expansion_interval: int = Field(
        default=5,
        description="Expand one tree node every N CFR+ iterations."
    )

    # --- Network training ---
    learning_rate: float = Field(
        default=5e-4,
        description="Adam learning rate. Lower than Kuhn's 1e-3 for stability."
    )
    batch_size: int = Field(
        default=128,
        description="Minibatch size for gradient steps."
    )
    train_steps_per_gen: int = Field(
        default=200,
        description="Gradient steps per generation."
    )

    # --- Replay buffer ---
    buffer_capacity: int = Field(
        default=50000,
        description="Max training tuples stored (FIFO eviction)."
    )

    # --- Evaluation ---
    eval_interval: int = Field(
        default=10,
        description="Compute exploitability every N generations."
    )
    verbose: bool = Field(default=True, description="Print per-generation summaries.")

    # --- Loss weighting ---
    value_weight: float = Field(default=1.0, description="Value loss multiplier.")
    policy_weight: float = Field(default=1.0, description="Policy loss multiplier.")


class GenerationLog(BaseModel):
    """Logged metrics for one training generation."""

    generation: int = Field(description="Generation number (0 = random network).")
    exploitability: float | None = Field(
        default=None, description="Exploitability of raw network policy (0 = Nash)."
    )
    value_loss: float = Field(default=0.0, description="Avg MSE value loss.")
    policy_loss: float = Field(default=0.0, description="Avg cross-entropy policy loss.")
    total_loss: float = Field(default=0.0, description="Weighted combined loss.")
    strategy_snapshot: dict[str, dict[str, float]] = Field(
        default_factory=dict, description="Strategy at key info sets for visualization."
    )


# Key info sets to track for visualization
_KEY_INFO_SETS = [
    "K:?:",         # K opening, no community
    "J:?:",         # J opening, no community
    "K:K:check,check",  # K with K community, round 2 opening
    "J:J:check,check",  # J with J community (pair), round 2 opening
]


class SelfPlayTrainer:
    """
    Orchestrates the GT-CFR self-play training loop for Leduc Hold'em.
    """

    def __init__(self, config: TrainingConfig | None = None):
        self.config = config or TrainingConfig()
        # Initialize network with random weights
        self.net = LeducCVPN()
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.config.learning_rate)
        # Replay buffer (FIFO)
        self.replay_buffer: deque[TrainingTuple] = deque(maxlen=self.config.buffer_capacity)
        # Training history
        self.generation_logs: list[GenerationLog] = []
        # Saved checkpoints for tournament play
        self.checkpoints: list[dict] = []

    def train(self) -> list[GenerationLog]:
        """Run the full training loop. Returns generation logs."""
        # Gen 0: random network baseline
        initial_strategy = self.extract_strategy()
        initial_expl = compute_exploitability(initial_strategy)
        self.checkpoints.append({
            "generation": 0,
            "state_dict": {k: v.clone() for k, v in self.net.state_dict().items()},
            "strategy": initial_strategy,
        })
        gen0_log = GenerationLog(generation=0, exploitability=initial_expl)
        gen0_log.strategy_snapshot = self._snapshot_strategy(initial_strategy)
        self.generation_logs.append(gen0_log)
        if self.config.verbose:
            self._print_generation_log(gen0_log)

        for gen in range(1, self.config.n_generations + 1):
            log = GenerationLog(generation=gen)

            # Phase 1: Self-play data generation
            new_tuples = self._self_play_generation()
            self.replay_buffer.extend(new_tuples)

            # Phase 2: Train network on replay buffer
            if len(self.replay_buffer) >= self.config.batch_size:
                losses = self._train_network()
                log.value_loss = losses["value"]
                log.policy_loss = losses["policy"]
                log.total_loss = losses["total"]

            # Phase 3: Periodic exploitability evaluation
            if gen % self.config.eval_interval == 0 or gen == 1:
                strategy = self.extract_strategy()
                log.exploitability = compute_exploitability(strategy)
                log.strategy_snapshot = self._snapshot_strategy(strategy)
                self.checkpoints.append({
                    "generation": gen,
                    "state_dict": {k: v.clone() for k, v in self.net.state_dict().items()},
                    "strategy": strategy,
                })

            self.generation_logs.append(log)
            if self.config.verbose:
                self._print_generation_log(log)

        return self.generation_logs

    def _self_play_generation(self) -> list[TrainingTuple]:
        """Play self-play games and collect training tuples."""
        tuples: list[TrainingTuple] = []

        for _ in range(self.config.games_per_generation):
            deal = random.choice(all_deals())
            game_tuples = self._play_game(deal)
            tuples.extend(game_tuples)

        return tuples

    def _play_game(self, initial_state: LeducState) -> list[TrainingTuple]:
        """
        Play one game using GT-CFR search at each decision point.
        Handles chance nodes (community card) by sampling.
        """
        tuples: list[TrainingTuple] = []
        state = initial_state

        while not state.is_terminal():
            # Chance node: sample community card
            if state.is_chance_node():
                outcomes = community_outcomes(state)
                cards, probs = zip(*outcomes)
                chosen_card = random.choices(cards, weights=probs, k=1)[0]
                state = state.apply_chance(chosen_card)
                continue

            acting_player = state.current_player()

            # Run GT-CFR search (incremental expansion for speed)
            sigma_bar, cfvs, _ = gt_cfr_search(
                root_state=state,
                net=self.net,
                n_iterations=self.config.search_iterations,
                c_puct=self.config.c_puct,
                expansion_interval=self.config.expansion_interval,
                full_expand=False,
            )

            # Create training tuple
            training_tuple = self._make_training_tuple(state, acting_player, sigma_bar, cfvs)
            tuples.append(training_tuple)

            # Sample action from search's average strategy for the actual card
            info_key = state.info_set_key()
            if info_key in sigma_bar:
                action_probs = sigma_bar[info_key]
            else:
                actions = state.legal_actions()
                action_probs = {a: 1.0 / len(actions) for a in actions}

            actions = list(action_probs.keys())
            probs = [action_probs[a] for a in actions]
            chosen_action = random.choices(actions, weights=probs, k=1)[0]
            state = state.apply_action(chosen_action)

        return tuples

    def _make_training_tuple(
        self,
        state: LeducState,
        acting_player: int,
        sigma_bar: dict[str, dict[str, float]],
        cfvs: dict[str, float],
    ) -> TrainingTuple:
        """Convert search outputs into a training tuple."""
        public_state = encode_public_state(state, acting_player)

        # Target values: one CFV per rank
        target_values = np.zeros(NUM_PRIVATE_STATES, dtype=np.float32)
        for i, rank in enumerate(RANKS):
            target_values[i] = cfvs.get(rank, 0.0)

        # Target policy: [NUM_PRIVATE_STATES, NUM_ACTIONS]
        target_policy = np.zeros((NUM_PRIVATE_STATES, NUM_ACTIONS), dtype=np.float32)
        actions = state.legal_actions()
        comm_str = card_rank(state.community_card) if state.community_card is not None else "?"
        history_str = ",".join(state.history) if state.history else ""

        for i, rank in enumerate(RANKS):
            info_key = f"{rank}:{comm_str}:{history_str}"
            if info_key in sigma_bar:
                for j, action in enumerate(actions):
                    if j < NUM_ACTIONS:
                        target_policy[i, j] = sigma_bar[info_key].get(action, 0.0)
            else:
                for j in range(min(len(actions), NUM_ACTIONS)):
                    target_policy[i, j] = 1.0 / len(actions)

        return TrainingTuple(
            public_state=public_state,
            acting_player=acting_player,
            target_values=target_values,
            target_policy=target_policy,
        )

    def _train_network(self) -> dict[str, float]:
        """Train network on minibatches from the replay buffer."""
        self.net.train()
        total_value_loss = 0.0
        total_policy_loss = 0.0
        total_loss_sum = 0.0
        n_steps = 0

        for _ in range(self.config.train_steps_per_gen):
            batch = random.sample(
                list(self.replay_buffer),
                min(self.config.batch_size, len(self.replay_buffer)),
            )

            # Prepare tensors
            states = torch.tensor(
                np.array([t.public_state for t in batch]), dtype=torch.float32
            )
            target_values = torch.tensor(
                np.array([t.target_values for t in batch]), dtype=torch.float32
            )
            target_policies = torch.tensor(
                np.array([t.target_policy for t in batch]), dtype=torch.float32
            )

            # Forward pass
            pred_policy_logits, pred_values = self.net(states)

            # Value loss: MSE
            value_loss = nn.functional.mse_loss(pred_values, target_values)

            # Policy loss: soft cross-entropy
            pred_flat = pred_policy_logits.view(-1, NUM_ACTIONS)
            target_flat = target_policies.view(-1, NUM_ACTIONS)
            log_probs = torch.log_softmax(pred_flat, dim=-1)
            policy_loss = -(target_flat * log_probs).sum(dim=-1).mean()

            # Combined loss
            loss = (
                self.config.value_weight * value_loss
                + self.config.policy_weight * policy_loss
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_value_loss += value_loss.item()
            total_policy_loss += policy_loss.item()
            total_loss_sum += loss.item()
            n_steps += 1

        self.net.eval()

        return {
            "value": total_value_loss / max(1, n_steps),
            "policy": total_policy_loss / max(1, n_steps),
            "total": total_loss_sum / max(1, n_steps),
        }

    def extract_strategy(self) -> Strategy:
        """Extract the network's raw policy (no search) at every info set."""
        return self._extract_network_policy()

    def _extract_network_policy(self) -> Strategy:
        """Query CVPN policy head for every info set (no search)."""
        self.net.eval()
        strategy: Strategy = {}

        for player in (PLAYER_1, PLAYER_2):
            for info_key in all_info_set_keys()[player]:
                parts = info_key.split(":")
                rank = parts[0]
                comm_str = parts[1]
                history_str = parts[2] if len(parts) > 2 else ""
                history = tuple(history_str.split(",")) if history_str else ()

                # Build dummy state for encoding
                private_id = RANKS.index(rank) * SUITS_PER_RANK
                opp_id = ((RANKS.index(rank) + 1) % len(RANKS)) * SUITS_PER_RANK
                comm_id: int | None = None
                if comm_str != "?":
                    comm_rank_idx = RANKS.index(comm_str)
                    for s in range(SUITS_PER_RANK):
                        candidate = comm_rank_idx * SUITS_PER_RANK + s
                        if candidate != private_id and candidate != opp_id:
                            comm_id = candidate
                            break

                dummy_state = LeducState(
                    cards=(private_id, opp_id), community_card=comm_id, history=history
                )

                # Forward pass
                encoded = encode_public_state(dummy_state, player)
                x = torch.tensor(encoded, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    policy_logits, _ = self.net(x)

                actions = actions_at_info_set(info_key)
                if actions:
                    policy = get_policy_for_info_set(policy_logits[0], rank, actions)
                    strategy[info_key] = policy

        return strategy

    def _snapshot_strategy(self, strategy: Strategy) -> dict[str, dict[str, float]]:
        """Extract strategy values at key info sets for logging."""
        return {k: v for k, v in strategy.items() if k in _KEY_INFO_SETS}

    def _print_generation_log(self, log: GenerationLog) -> None:
        """Print a one-line summary of a generation."""
        parts = [f"Gen {log.generation:3d}"]
        if log.exploitability is not None:
            parts.append(f"expl={log.exploitability:.4f}")
        parts.append(f"loss={log.total_loss:.4f} (v={log.value_loss:.4f} p={log.policy_loss:.4f})")
        if log.strategy_snapshot:
            for key in _KEY_INFO_SETS[:2]:
                if key in log.strategy_snapshot:
                    probs = log.strategy_snapshot[key]
                    first_action = list(probs.keys())[0]
                    parts.append(f"{key}{first_action}={probs[first_action]:.2f}")
        print(" | ".join(parts))

    def play_tournament(self, n_games: int = 500) -> np.ndarray:
        """
        Round-robin tournament between saved checkpoints.
        Returns win_matrix[i, j] = win rate of checkpoint i (P1) vs j (P2).
        """
        n = len(self.checkpoints)
        if n < 2:
            return np.zeros((n, n))

        results = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i == j:
                    results[i, j] = 0.5
                    continue
                strat_i = self.checkpoints[i]["strategy"]
                strat_j = self.checkpoints[j]["strategy"]
                wins_i = 0
                for _ in range(n_games):
                    deal = random.choice(all_deals())
                    result = self._simulate_game(deal, strat_i, strat_j)
                    if result > 0:
                        wins_i += 1
                    elif result == 0:
                        wins_i += 0.5
                results[i, j] = wins_i / n_games

        return results

    def _simulate_game(
        self, state: LeducState, p1_strategy: Strategy, p2_strategy: Strategy
    ) -> float:
        """Simulate one game using fixed strategies. Returns P1's payoff."""
        while not state.is_terminal():
            # Handle chance node
            if state.is_chance_node():
                outcomes = community_outcomes(state)
                cards, probs = zip(*outcomes)
                chosen = random.choices(cards, weights=probs, k=1)[0]
                state = state.apply_chance(chosen)
                continue

            acting_player = state.current_player()
            info_key = state.info_set_key()

            if acting_player == PLAYER_1:
                action_probs = p1_strategy.get(info_key, None)
            else:
                action_probs = p2_strategy.get(info_key, None)

            if action_probs is None:
                actions = state.legal_actions()
                action_probs = {a: 1.0 / len(actions) for a in actions}

            actions = list(action_probs.keys())
            probs = [action_probs[a] for a in actions]
            chosen = random.choices(actions, weights=probs, k=1)[0]
            state = state.apply_action(chosen)

        return state.terminal_utility(PLAYER_1)
