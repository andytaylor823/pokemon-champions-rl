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

from action_space import action_to_choice_contextual, legal_mask
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


def _advance(
    sim: SimClient, handle: int, choices: dict[str, str], seed: list[int]
) -> tuple[int, StateView]:
    """Step the battle forward and release the old handle.

    Centralizes the step+release contract: callers never need to manage
    handle lifecycle manually.  On SimError the exception propagates
    *before* release, so the old handle stays valid for retry or cleanup.
    """
    res = sim.step(handle, choices, seed)
    sim.release(handle)
    return res.child, res.view


def _is_forced(masks: dict[str, np.ndarray]) -> dict[str, int] | None:
    """Return {side: sole_legal_idx} if every side has exactly one legal action.

    Same semantics as action_space.forced_actions but operates on pre-computed
    masks to avoid redundant legal_mask calls.
    """
    if not masks:
        return None
    out: dict[str, int] = {}
    for s, m in masks.items():
        if int(m.sum()) != 1:
            return None
        out[s] = int(np.argmax(m))
    return out


def _sample_and_step(
    sim: SimClient,
    handle: int,
    view: StateView,
    strategy: dict[str, dict[int, float]],
    game_rng: random.Random,
    temperature: float,
) -> tuple[int, StateView] | None:
    """Sample actions from sigma-bar and advance, retrying on mask/engine discrepancy.

    Each retry resamples fresh from sigma-bar — the RNG naturally produces
    different joints, so no per-side or per-joint blacklist is needed.
    Mask/engine targeting gaps are rare (expansion handles the same
    discrepancy by dropping the cell).

    Returns (new_handle, new_view) on success, or None if all retries fail.
    """
    for _attempt in range(_MAX_STEP_RETRIES):
        choices: dict[str, str] = {}
        for side in view.to_move:
            side_strategy = strategy.get(side, {})
            if not side_strategy:
                return None
            action_idx = _sample_action(side_strategy, game_rng, temperature)
            choices[side] = action_to_choice_contextual(
                action_idx, view.legal.get(side)
            )

        step_seed = _rng_seed(game_rng)
        try:
            return _advance(sim, handle, choices, step_seed)
        except SimError as e:
            logger.debug(
                "Step attempt %d: SimError=%s choices=%s",
                _attempt, str(e), choices,
            )

    return None


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
        handle: int | None = None

        try:
            # Start a new battle
            battle_seed = _rng_seed(game_rng)
            handle, view = sim.new_battle(team_a, team_b, seed=battle_seed)

            while not view.terminal:
                # --- Cap: abort runaway games ---
                if decision_idx >= config.max_decisions:
                    logger.warning(
                        "Game %d aborted: exceeded max_decisions=%d",
                        game_id,
                        config.max_decisions,
                    )
                    aborted = True
                    sim.release(handle)
                    break

                # --- Empty phase: non-terminal with no acting sides ---
                if not view.to_move or view.phase == "none":
                    handle, view = _advance(sim, handle, {}, _rng_seed(game_rng))
                    continue

                # Compute masks once for this decision point — used by the
                # forced check and the buffering loop, avoiding redundant
                # legal_mask calls.
                masks = {
                    s: legal_mask(view.legal.get(s), view.phase)
                    for s in view.to_move
                }

                # --- Forced-decision skip ---
                forced = _is_forced(masks)
                if forced is not None:
                    choices = {
                        s: action_to_choice_contextual(idx, view.legal.get(s))
                        for s, idx in forced.items()
                    }
                    handle, view = _advance(sim, handle, choices, _rng_seed(game_rng))
                    continue

                # --- Genuine decision: run search ---
                result = search(
                    view, sim, net, from_handle=handle, config=config.search_config
                )

                # Buffer one tuple per side with >=2 legal actions
                for side in view.to_move:
                    if masks[side].sum() < 2:
                        continue
                    beta = encode(view, side)
                    value = result.value if side == "p1" else -result.value
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

                # --- Sample from sigma-bar and step (with retry) ---
                advance_result = _sample_and_step(
                    sim, handle, view, result.strategy,
                    game_rng, config.temperature,
                )
                if advance_result is None:
                    logger.warning(
                        "Game %d aborted: failed to step after %d retries",
                        game_id,
                        _MAX_STEP_RETRIES,
                    )
                    aborted = True
                    sim.release(handle)
                    break

                handle, view = advance_result
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
                sim.release(handle)

        except SimError as e:
            logger.warning("Game %d aborted due to SimError: %s", game_id, e)
            if handle is not None:
                sim.release(handle)
            continue
