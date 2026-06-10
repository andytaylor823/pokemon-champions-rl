"""
Outer training loop for GT-CFR on Kuhn Poker.

This module orchestrates:
  1. Self-play games where both players use GT-CFR search (guided by the current CVPN)
  2. Collection of training tuples (public_state, search_cfvs, search_sigma_bar)
  3. Neural network training on a replay buffer
  4. Periodic exploitability measurement and strategy logging

The cycle: better net -> better search -> better training data -> better net.
"""

from __future__ import annotations

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pydantic import BaseModel, Field

from toy_examples.kuhn_poker.game import (
    CARDS,
    PLAYER_1,
    PLAYER_2,
    KuhnState,
    all_deals,
    all_info_set_keys,
    legal_actions,
    make_info_set_key,
    parse_info_set_key,
)
from toy_examples.kuhn_poker.network import (
    KuhnCVPN,
    encode_public_state,
    get_policy_for_info_set,
    NUM_ACTIONS,
    NUM_CARDS,
)
from toy_examples.kuhn_poker.gt_cfr_search import gt_cfr_search
from toy_examples.kuhn_poker.exploitability import compute_exploitability, Strategy


class TrainingTuple(BaseModel):
    """
    One training sample emitted by a GT-CFR search at a single decision point.

    The outer loop collects these into a replay buffer and trains the CVPN to
    predict target_values (value head) and target_policy (policy head).
    """

    model_config = {"arbitrary_types_allowed": True}

    public_state: np.ndarray = Field(
        description="Encoded public belief state vector -- the network input. "
        "Shape: [INPUT_DIM]. Contains action history one-hots, acting player, and belief."
    )
    acting_player: int = Field(
        description="Which player is acting at this decision point (0 = P1, 1 = P2). "
        "Needed to interpret which dimension of the network output corresponds to whom."
    )
    target_values: np.ndarray = Field(
        description="Search-refined counterfactual values, one per possible private card "
        "(J, Q, K). Shape: [NUM_CARDS=3]. These are the VALUE HEAD training targets -- "
        "the search's answer to 'how good is this position for each card I could hold?'"
    )
    target_policy: np.ndarray = Field(
        description="Search-refined average strategy (sigma_bar) per private card. "
        "Shape: [NUM_CARDS=3, NUM_ACTIONS=2]. These are the POLICY HEAD training targets -- "
        "the search's answer to 'what should I do with each possible card?'"
    )


class TrainingConfig(BaseModel):
    """
    Configuration for the GT-CFR self-play training loop.

    The training has two nested loops:
      - OUTER LOOP (generations): play self-play games, collect data, train the NN.
        One generation = play N games + do M gradient steps on the replay buffer.
      - INNER LOOP (search): at each decision point within a game, run GT-CFR search
        for a fixed iteration budget to produce a refined strategy + value estimates.

    The NN learns to approximate what the search produces, so future searches start
    from a better prior and converge faster -- the virtuous cycle.
    """

    # --- Outer loop: how many generations and how much data per generation ---
    n_generations: int = Field(
        default=30,
        description="Total number of outer-loop training generations. Each generation "
        "plays games, trains the network, and optionally evaluates exploitability."
    )
    games_per_generation: int = Field(
        default=50,
        description="Number of full self-play games per generation. Each game produces "
        "2-3 training tuples (one per decision point). More games = more diverse data "
        "but slower generations."
    )

    # --- Inner loop: GT-CFR search budget at each decision point ---
    search_iterations: int = Field(
        default=100,
        description="Number of CFR+ iterations to run per search invocation. Higher = "
        "better search output (closer to Nash) but more expensive. With full tree "
        "expansion on Kuhn, ~200 iterations gives near-perfect convergence."
    )
    c_puct: float = Field(
        default=2.0,
        description="Exploration constant for PUCT-guided tree expansion. Controls the "
        "trade-off between exploiting the current strategy (low c) and exploring "
        "actions the NN policy recommends (high c). Only matters when full_expand=False."
    )
    expansion_interval: int = Field(
        default=10,
        description="Expand one new tree node every this many CFR+ iterations. Only "
        "active when using incremental expansion (full_expand=False). Lower = faster "
        "tree growth but fewer CFR iterations between expansions."
    )

    # --- Network training hyperparameters ---
    learning_rate: float = Field(
        default=1e-3,
        description="Adam optimizer learning rate. The network is small (64-unit MLP), "
        "so 1e-3 works well. Lower (5e-4) can help with stability on longer runs."
    )
    batch_size: int = Field(
        default=64,
        description="Minibatch size for gradient steps. Sampled uniformly from the "
        "replay buffer. Smaller batches = more noise but more updates per epoch."
    )
    train_steps_per_gen: int = Field(
        default=100,
        description="Number of gradient steps (minibatch updates) per generation. "
        "The network sees batch_size * train_steps_per_gen samples each generation "
        "(with replacement from the buffer)."
    )

    # --- Replay buffer ---
    buffer_capacity: int = Field(
        default=10000,
        description="Maximum number of training tuples stored. Oldest tuples are evicted "
        "when full (FIFO). Larger buffers smooth learning but retain stale data from "
        "early (bad) checkpoints longer."
    )

    # --- Evaluation and logging ---
    eval_interval: int = Field(
        default=5,
        description="Compute exploitability every N generations. Exploitability requires "
        "a full game-tree traversal so it's moderately expensive -- don't set to 1 for "
        "long runs."
    )
    verbose: bool = Field(
        default=True,
        description="Print per-generation training summaries (exploitability, loss, "
        "key strategy values) to stdout."
    )

    # --- Loss weighting ---
    value_weight: float = Field(
        default=1.0,
        description="Multiplier on the value loss (MSE on CFVs) in the combined "
        "training objective. Increase to prioritize accurate value predictions over "
        "policy accuracy."
    )
    policy_weight: float = Field(
        default=1.0,
        description="Multiplier on the policy loss (cross-entropy on sigma_bar) in the "
        "combined training objective. Increase to prioritize policy distillation over "
        "value accuracy."
    )


class GenerationLog(BaseModel):
    """Logged metrics for one generation of training."""

    generation: int = Field(
        description="Generation number (0 = initial random network, before any training)."
    )
    exploitability: float | None = Field(
        default=None,
        description="How exploitable the network's raw policy is (0 = Nash equilibrium). "
        "Only measured every eval_interval generations."
    )
    value_loss: float = Field(
        default=0.0,
        description="Average MSE loss on the value head this generation (how well the "
        "network predicts the search's CFVs)."
    )
    policy_loss: float = Field(
        default=0.0,
        description="Average cross-entropy loss on the policy head this generation (how "
        "well the network replicates the search's average strategy)."
    )
    total_loss: float = Field(
        default=0.0,
        description="Weighted sum of value_loss and policy_loss (the actual gradient target)."
    )
    strategy_snapshot: dict[str, dict[str, float]] = Field(
        default_factory=dict,
        description="Strategy at a few key info sets (K:, J:, J:bet, etc.) for "
        "visualization of how the learned policy evolves over time."
    )


class SelfPlayTrainer:
    """
    Orchestrates the GT-CFR self-play training loop.

    The trainer maintains:
      - The CVPN (neural network)
      - A replay buffer of training tuples
      - Generation logs for visualization
    """

    def __init__(self, config: TrainingConfig | None = None):
        self.config = config or TrainingConfig()
        # Initialize the network with random weights
        self.net = KuhnCVPN()
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.config.learning_rate)
        # Replay buffer (FIFO)
        self.replay_buffer: deque[TrainingTuple] = deque(maxlen=self.config.buffer_capacity)
        # Training history
        self.generation_logs: list[GenerationLog] = []
        # Saved checkpoints (strategy snapshots for tournament play)
        self.checkpoints: list[dict] = []

    def train(self) -> list[GenerationLog]:
        """
        Run the full training loop for n_generations.
        Returns the generation logs.
        """
        # Save gen-0 checkpoint (random network, before any training)
        initial_strategy = self.extract_strategy()
        initial_expl = compute_exploitability(initial_strategy)
        self.checkpoints.append({
            "generation": 0,
            "state_dict": {k: v.clone() for k, v in self.net.state_dict().items()},
            "strategy": initial_strategy,
        })
        gen0_log = GenerationLog(generation=0, exploitability=initial_expl)
        gen0_log.strategy_snapshot = {
            k: v for k, v in initial_strategy.items()
            if k in ("K:", "Q:", "J:", "K:bet", "J:bet", "J:check")
        }
        self.generation_logs.append(gen0_log)
        if self.config.verbose:
            self._print_generation_log(gen0_log)

        for gen in range(1, self.config.n_generations + 1):
            log = GenerationLog(generation=gen)

            # Phase 1: Self-play data generation
            new_tuples = self._self_play_generation()
            self.replay_buffer.extend(new_tuples)

            # Phase 2: Train the network on replay buffer
            if len(self.replay_buffer) >= self.config.batch_size:
                losses = self._train_network()
                log.value_loss = losses["value"]
                log.policy_loss = losses["policy"]
                log.total_loss = losses["total"]

            # Phase 3: Evaluate exploitability periodically
            if gen % self.config.eval_interval == 0 or gen == 1:
                strategy = self.extract_strategy()
                log.exploitability = compute_exploitability(strategy)
                log.strategy_snapshot = {
                    k: v for k, v in strategy.items()
                    if k in ("K:", "Q:", "J:", "K:bet", "J:bet", "J:check")
                }
                # Save checkpoint
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
        """
        Play self-play games and collect training tuples.
        Each game: deal cards, then at each decision point run GT-CFR search.
        """
        tuples: list[TrainingTuple] = []

        for _ in range(self.config.games_per_generation):
            # Sample a random deal
            deal = random.choice(all_deals())
            # Play the game, collecting training data at each decision point
            game_tuples = self._play_game(deal)
            tuples.extend(game_tuples)

        return tuples

    def _play_game(self, initial_state: KuhnState) -> list[TrainingTuple]:
        """
        Play one game from a deal, using GT-CFR search at each decision point.
        Returns training tuples collected during the game.
        """
        tuples: list[TrainingTuple] = []
        state = initial_state

        while not state.is_terminal():
            acting_player = state.current_player()

            # Run GT-CFR search at this decision point
            sigma_bar, cfvs, _ = gt_cfr_search(
                root_state=state,
                net=self.net,
                n_iterations=self.config.search_iterations,
                c_puct=self.config.c_puct,
                expansion_interval=self.config.expansion_interval,
            )

            # Create training tuple from search results
            training_tuple = self._make_training_tuple(state, acting_player, sigma_bar, cfvs)
            tuples.append(training_tuple)

            # Choose an action from the search's average strategy for the actual card
            info_key = state.info_set_key()
            if info_key in sigma_bar:
                action_probs = sigma_bar[info_key]
            else:
                # Fallback to uniform
                actions = state.legal_actions()
                action_probs = {a: 1.0 / len(actions) for a in actions}

            # Sample action from the strategy
            actions = list(action_probs.keys())
            probs = [action_probs[a] for a in actions]
            chosen_action = random.choices(actions, weights=probs, k=1)[0]

            # Apply the action
            state = state.apply_action(chosen_action)

        return tuples

    def _make_training_tuple(
        self,
        state: KuhnState,
        acting_player: int,
        sigma_bar: dict[str, dict[str, float]],
        cfvs: dict[str, float],
    ) -> TrainingTuple:
        """Convert search outputs into a training tuple."""
        # Encode the public state (only history + player needed)
        public_state = encode_public_state(state.history, acting_player)

        # Build target values array: one CFV per card
        target_values = np.zeros(NUM_CARDS, dtype=np.float32)
        for i, card in enumerate(CARDS):
            target_values[i] = cfvs.get(card, 0.0)

        # Build target policy array: [NUM_CARDS, NUM_ACTIONS]
        target_policy = np.zeros((NUM_CARDS, NUM_ACTIONS), dtype=np.float32)
        actions = state.legal_actions()
        for i, card in enumerate(CARDS):
            info_key = make_info_set_key(card, state.history)
            if info_key in sigma_bar:
                for j, action in enumerate(actions):
                    if j < NUM_ACTIONS:
                        target_policy[i, j] = sigma_bar[info_key].get(action, 0.0)
            else:
                # Uniform fallback
                for j in range(min(len(actions), NUM_ACTIONS)):
                    target_policy[i, j] = 1.0 / len(actions)

        return TrainingTuple(
            public_state=public_state,
            acting_player=acting_player,
            target_values=target_values,
            target_policy=target_policy,
        )

    def _train_network(self) -> dict[str, float]:
        """
        Train the network on minibatches from the replay buffer.
        Loss = MSE(predicted_values, target_values) + CE(predicted_policy, target_policy)
        """
        self.net.train()
        total_value_loss = 0.0
        total_policy_loss = 0.0
        total_loss_sum = 0.0
        n_steps = 0

        for _ in range(self.config.train_steps_per_gen):
            # Sample a minibatch
            batch = random.sample(
                list(self.replay_buffer),
                min(self.config.batch_size, len(self.replay_buffer)),
            )

            # Prepare tensors
            states = torch.tensor(np.array([t.public_state for t in batch]), dtype=torch.float32)
            target_values = torch.tensor(np.array([t.target_values for t in batch]), dtype=torch.float32)
            target_policies = torch.tensor(np.array([t.target_policy for t in batch]), dtype=torch.float32)

            # Forward pass
            pred_policy_logits, pred_values = self.net(states)

            # Value loss: MSE between predicted and search-refined CFVs
            value_loss = nn.functional.mse_loss(pred_values, target_values)

            # Policy loss: cross-entropy between predicted policy and search average strategy
            # Reshape for cross-entropy: [batch * NUM_CARDS, NUM_ACTIONS]
            pred_flat = pred_policy_logits.view(-1, NUM_ACTIONS)
            target_flat = target_policies.view(-1, NUM_ACTIONS)
            # Use soft cross-entropy (target is a distribution, not one-hot)
            log_probs = torch.log_softmax(pred_flat, dim=-1)
            policy_loss = -(target_flat * log_probs).sum(dim=-1).mean()

            # Combined loss
            loss = (
                self.config.value_weight * value_loss
                + self.config.policy_weight * policy_loss
            )

            # Backward pass
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
        """
        Extract the current network's RAW policy as a strategy (no search).
        This shows what the network has actually learned -- exploitability of
        this strategy decreasing over generations is the core learning signal.
        """
        return self._extract_network_policy()

    def _extract_network_policy(self) -> Strategy:
        """
        Query the CVPN's policy head directly for every info set.
        No search involved -- this is the network's "snap judgment."
        """
        self.net.eval()
        strategy: Strategy = {}

        for player in (PLAYER_1, PLAYER_2):
            for info_key in all_info_set_keys()[player]:
                card, history = parse_info_set_key(info_key)

                # Forward pass on the public state
                encoded = encode_public_state(history, player)
                x = torch.tensor(encoded, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    policy_logits, _ = self.net(x)

                # Extract policy for this specific private card
                actions = legal_actions(history)
                policy = get_policy_for_info_set(policy_logits[0], card, actions)
                strategy[info_key] = policy

        return strategy

    def _print_generation_log(self, log: GenerationLog) -> None:
        """Print a summary of one generation's training."""
        parts = [f"Gen {log.generation:3d}"]
        if log.exploitability is not None:
            parts.append(f"expl={log.exploitability:.4f}")
        parts.append(f"loss={log.total_loss:.4f} (v={log.value_loss:.4f} p={log.policy_loss:.4f})")
        if log.strategy_snapshot:
            # Show a few key strategies
            for key in ("K:", "J:", "J:bet"):
                if key in log.strategy_snapshot:
                    probs = log.strategy_snapshot[key]
                    first_action = list(probs.keys())[0]
                    parts.append(f"{key}→{first_action}={probs[first_action]:.2f}")
        print(" | ".join(parts))

    def play_tournament(self, n_games: int = 1000) -> np.ndarray:
        """
        Play a round-robin tournament between all saved checkpoints.
        Returns a win-rate matrix: result[i, j] = win rate of checkpoint i vs j.
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
                # Load strategies for both checkpoints
                strat_i = self.checkpoints[i]["strategy"]
                strat_j = self.checkpoints[j]["strategy"]
                # Play games: checkpoint i as P1, checkpoint j as P2
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
        self, state: KuhnState, p1_strategy: Strategy, p2_strategy: Strategy
    ) -> float:
        """Simulate one game using fixed strategies. Returns P1's payoff."""
        while not state.is_terminal():
            acting_player = state.current_player()
            info_key = state.info_set_key()

            # Pick the appropriate strategy
            if acting_player == PLAYER_1:
                action_probs = p1_strategy.get(info_key, None)
            else:
                action_probs = p2_strategy.get(info_key, None)

            if action_probs is None:
                # Fallback to uniform
                actions = state.legal_actions()
                action_probs = {a: 1.0 / len(actions) for a in actions}

            # Sample an action
            actions = list(action_probs.keys())
            probs = [action_probs[a] for a in actions]
            chosen = random.choices(actions, weights=probs, k=1)[0]
            state = state.apply_action(chosen)

        return state.terminal_utility(PLAYER_1)
