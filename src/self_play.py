"""SelfPlay — Phase-1 outer-loop data generator.

Plays full self-play games where every genuine decision is one search() call,
skips forced decisions, samples actions from the average strategy (sigma-bar),
and yields TrainingTuples for the (future) ReplayBuffer / Trainer.

This module is a pure consumer/orchestrator: it reimplements no game rules
(SimClient is the sole source of truth) and no search logic (it calls search()).
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Iterator, Protocol

import numpy as np

from action_space import (
    action_to_choice_contextual,
    legal_mask,
)
from cvpn import CVPN
from encoder import encode
from obs_bundle import ObsBundle
from search import SearchConfig, search
from sim_client import SimClient, SimError
from state_types import StateView

logger = logging.getLogger(__name__)

# Max retry attempts when sim.step raises SimError (targeting discrepancy)
_MAX_STEP_RETRIES = 10


# ---------------------------------------------------------------------------
# Data types (frozen dataclasses per repo convention for hot-path internals)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SparsePolicy:
    """Average strategy stored sparse: action indices and their probabilities."""

    indices: tuple[int, ...]
    probs: tuple[float, ...]


@dataclass(frozen=True)
class TupleMeta:
    """Provenance metadata for a training tuple."""

    generation: int
    game_id: int
    decision_idx: int
    phase: str
    side: str


@dataclass(frozen=True)
class TrainingTuple:
    """One training sample: the stable boundary between inner and outer loops.

    Analogous to AlphaZero's (state, pi, z) but with bootstrapped search value
    as the primary value target and the game result z as a side label.
    """

    beta: ObsBundle
    value: float
    policy: SparsePolicy
    z: float
    meta: TupleMeta


@dataclass(frozen=True)
class SelfPlayConfig:
    """Configuration for the self-play data generation loop."""

    temperature: float = 1.0
    max_decisions: int = 500
    master_seed: int = 42
    num_games: int | None = None  # None = infinite
    search_config: SearchConfig | None = None
    generation: int = 0


# ---------------------------------------------------------------------------
# MatchupSource protocol and implementations
# ---------------------------------------------------------------------------


class MatchupSource(Protocol):
    """Protocol for team matchup providers."""

    def sample(self, rng: random.Random) -> tuple[list[dict], list[dict]]:
        """Return (team_a, team_b) as legal team dicts for new_battle."""
        ...


class CurriculumMatchupSource:
    """Returns a fixed matchup — used for validation curriculum stages."""

    def __init__(self, team_a: list[dict], team_b: list[dict]) -> None:
        self._team_a = team_a
        self._team_b = team_b

    def sample(self, rng: random.Random) -> tuple[list[dict], list[dict]]:
        return self._team_a, self._team_b


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _is_forced(view: StateView) -> bool:
    """Check if every acting side has exactly one legal action (no real choice)."""
    if not view.to_move:
        return False
    for side in view.to_move:
        mask = legal_mask(view.legal.get(side), view.phase)
        if mask.sum() != 1:
            return False
    return True


def _forced_choices(view: StateView) -> dict[str, str]:
    """Build the choice dict for a forced decision (single legal action per side)."""
    choices: dict[str, str] = {}
    for side in view.to_move:
        mask = legal_mask(view.legal.get(side), view.phase)
        action_idx = int(np.argmax(mask))
        choices[side] = action_to_choice_contextual(action_idx, view.legal.get(side))
    return choices


def _sample_action(
    strategy: dict[int, float], rng: random.Random, temperature: float
) -> int:
    """Sample one action index from sparse sigma-bar with temperature scaling.

    Temperature rescaling: p_i' = p_i^(1/tau), then renormalize.
    tau=1.0 samples exactly from sigma-bar; tau->0 is greedy/argmax.
    """
    indices = list(strategy.keys())
    probs = np.array([strategy[i] for i in indices], dtype=np.float64)

    if temperature <= 0 or temperature < 1e-8:
        # Greedy: pick the highest-probability action
        return indices[int(np.argmax(probs))]

    if temperature != 1.0:
        # Rescale by temperature
        probs = probs ** (1.0 / temperature)

    # Renormalize (handles potential floating-point drift)
    total = probs.sum()
    if total > 0:
        probs = probs / total
    else:
        # Degenerate: uniform fallback
        probs = np.ones_like(probs) / len(probs)

    # Sample using random.Random for reproducibility
    r = rng.random()
    cumulative = 0.0
    for i, p in enumerate(probs):
        cumulative += p
        if r < cumulative:
            return indices[i]
    # Numerical safety: return last action
    return indices[-1]


def _rng_seed(rng: random.Random) -> list[int]:
    """Generate a 4-element PRNG seed list from the given RNG."""
    return [rng.randint(0, 0xFFFF) for _ in range(4)]


# ---------------------------------------------------------------------------
# Mutable buffer entry (z stamped after terminal)
# ---------------------------------------------------------------------------


@dataclass
class _PendingTuple:
    """Mutable intermediate before z is known. Converted to TrainingTuple at flush."""

    beta: ObsBundle
    value: float
    policy: SparsePolicy
    meta: TupleMeta


# ---------------------------------------------------------------------------
# Core generator
# ---------------------------------------------------------------------------


def run(
    net: CVPN,
    matchup_source: MatchupSource,
    sim: SimClient,
    config: SelfPlayConfig | None = None,
) -> Iterator[TrainingTuple]:
    """Play self-play games and yield training tuples.

    Each game is played to terminal (or aborted on max_decisions / SimError).
    Tuples are buffered per-game and flushed with the terminal result z stamped
    on each. The generator yields one game's worth of tuples at a time.

    Args:
        net: The CVPN network (used only via search()).
        matchup_source: Provides team matchups for each game.
        sim: SimClient instance (caller owns lifecycle).
        config: Self-play configuration knobs.
    """
    if config is None:
        config = SelfPlayConfig()

    # Master RNG produces per-game seeds for reproducibility
    master_rng = random.Random(config.master_seed)
    game_count = 0

    while config.num_games is None or game_count < config.num_games:
        game_id = game_count
        game_count += 1

        # Per-game RNG derived from master
        game_seed = master_rng.randint(0, 2**32 - 1)
        game_rng = random.Random(game_seed)

        # Get teams for this game
        team_a, team_b = matchup_source.sample(game_rng)

        # Per-game buffer: pending tuples awaiting terminal z
        pending: list[_PendingTuple] = []
        decision_idx = 0
        aborted = False

        try:
            # Start a new battle
            battle_seed = _rng_seed(game_rng)
            handle, view = sim.new_battle(team_a, team_b, seed=battle_seed)

            while not view.terminal:
                # Safety cap: abort if too many decisions
                if decision_idx >= config.max_decisions:
                    logger.warning(
                        "Game %d aborted: exceeded max_decisions=%d",
                        game_id,
                        config.max_decisions,
                    )
                    aborted = True
                    sim.release(handle)
                    break

                # Defensive edge: non-terminal with phase "none" / empty to_move
                if not view.to_move or view.phase == "none":
                    step_seed = _rng_seed(game_rng)
                    res = sim.step(handle, {}, seed=step_seed)
                    sim.release(handle)
                    handle, view = res.child, res.view
                    continue

                # Forced-decision skip: no search, no CVPN, no tuple
                if _is_forced(view):
                    choices = _forced_choices(view)
                    step_seed = _rng_seed(game_rng)
                    res = sim.step(handle, choices, seed=step_seed)
                    sim.release(handle)
                    handle, view = res.child, res.view
                    continue

                # Genuine decision: run search
                result = search(
                    view, sim, net, from_handle=handle, config=config.search_config
                )

                # Buffer one tuple per side with >=2 legal actions
                for side in view.to_move:
                    mask = legal_mask(view.legal.get(side), view.phase)
                    if mask.sum() < 2:
                        continue

                    # Encode the observation from this side's perspective
                    beta = encode(view, side)

                    # Value from this side's perspective
                    value = result.value if side == "p1" else -result.value

                    # Sparse policy from sigma-bar
                    side_strategy = result.strategy.get(side, {})
                    policy = SparsePolicy(
                        indices=tuple(side_strategy.keys()),
                        probs=tuple(side_strategy.values()),
                    )

                    meta = TupleMeta(
                        generation=config.generation,
                        game_id=game_id,
                        decision_idx=decision_idx,
                        phase=view.phase,
                        side=side,
                    )

                    pending.append(
                        _PendingTuple(beta=beta, value=value, policy=policy, meta=meta)
                    )

                # Sample actions for both sides and advance. Retry on SimError
                # (mask/engine targeting discrepancy — search's expand_turn_node
                # catches the same per-cell; some strategy actions may be invalid).
                # Track failed action indices so retries sample different ones.
                stepped = False
                failed_actions: dict[str, set[int]] = {
                    s: set() for s in view.to_move
                }
                for _attempt in range(_MAX_STEP_RETRIES):
                    choices: dict[str, str] = {}
                    sampled_indices: dict[str, int] = {}
                    for side in view.to_move:
                        side_strategy = result.strategy.get(side, {})
                        # Exclude previously-failed actions
                        filtered = {
                            k: v
                            for k, v in side_strategy.items()
                            if k not in failed_actions[side]
                        }
                        if filtered:
                            action_idx = _sample_action(
                                filtered, game_rng, config.temperature
                            )
                        else:
                            # All strategy actions exhausted; try legal mask
                            mask = legal_mask(view.legal.get(side), view.phase)
                            for bad in failed_actions[side]:
                                if bad < len(mask):
                                    mask[bad] = False
                            legal_indices = np.flatnonzero(mask)
                            if len(legal_indices) > 0:
                                action_idx = int(game_rng.choice(legal_indices))
                            else:
                                # True fallback: pick any legal action
                                full_mask = legal_mask(
                                    view.legal.get(side), view.phase
                                )
                                action_idx = int(np.argmax(full_mask))
                        sampled_indices[side] = action_idx
                        choices[side] = action_to_choice_contextual(
                            action_idx, view.legal.get(side)
                        )

                    step_seed = _rng_seed(game_rng)
                    try:
                        res = sim.step(handle, choices, seed=step_seed)
                        sim.release(handle)
                        handle, view = res.child, res.view
                        stepped = True
                        break
                    except SimError as e:
                        logger.debug(
                            "Decision %d attempt %d: SimError=%s choices=%s",
                            decision_idx, _attempt, str(e), choices,
                        )
                        # Mark sampled actions as failed for next retry
                        for side, idx in sampled_indices.items():
                            failed_actions[side].add(idx)
                        continue

                if not stepped:
                    logger.warning(
                        "Game %d aborted: failed to step after %d retries",
                        game_id,
                        _MAX_STEP_RETRIES,
                    )
                    aborted = True
                    sim.release(handle)
                    break

                decision_idx += 1

            # Terminal reached: stamp z and yield all buffered tuples
            if not aborted:
                utility = view.utility or {}
                for pt in pending:
                    z = utility.get(pt.meta.side, 0.0)
                    yield TrainingTuple(
                        beta=pt.beta,
                        value=pt.value,
                        policy=pt.policy,
                        z=z,
                        meta=pt.meta,
                    )
                # Release terminal handle
                sim.release(handle)

        except SimError as e:
            logger.warning("Game %d aborted due to SimError: %s", game_id, e)
            # Discard all pending tuples for this game
            continue
