"""Evaluation module — Phase-1 progress measurement for the GT-CFR training loop.

Implements three metrics:
  1. ``curriculum_report`` — one net driving both sides of a lopsided matchup,
     tracking favored-team win-rate and win speed across checkpoints.
  2. ``head_to_head`` — two different checkpoints playing each other.
  3. ``fitness`` — convenience wrapper that runs head-to-head vs a list of
     opponents and returns a dict of HeadToHeadReport.

All measurement runs through the public seams already used by SelfPlay and
Search. This module is read-only — it owns no SimClient lifecycle and
reimplements no game rules or search logic.

Design: docs/plans/evaluation.md
"""

from __future__ import annotations

import contextlib
import logging
import random
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

import torch

import action_space
from checkpoint import load_checkpoint
from encoder import encode
from search import SearchConfig, SearchResult, search
from sim_client import SimError

if TYPE_CHECKING:
    from cvpn import CVPN
    from self_play import MatchupSource
    from sim_client import SimClient
    from state_types import StateView

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Seeding helper (mirrors self_play._rng_seed; defined locally to avoid
# importing a private symbol)
# ---------------------------------------------------------------------------


def _rng_seed(rng: random.Random) -> list[int]:
    """Generate a 4-element PRNG seed list from the given RNG."""
    return [rng.randint(0, 0xFFFF) for _ in range(4)]


# ---------------------------------------------------------------------------
# Agent abstractions
# ---------------------------------------------------------------------------


class Agent(Protocol):
    """Per-side move chooser used exclusively in Evaluation.

    Each call returns one canonical action index (0 .. A-1) for ``side`` at
    the current decision point.  The caller (``play_game``) maps the index to
    a Showdown choice string before stepping the engine.
    """

    def act(self, view: StateView, side: str, sim: SimClient, handle: int) -> int:
        """Return a canonical action index for ``side``."""
        ...


class SearchAgent:
    """Greedy agent backed by a full GT-CFR search.

    Holds a **1-slot cache** keyed by battle handle so that a single
    ``SearchAgent`` instance used on *both* sides triggers only **one search
    per turn**: the first ``act(handle=h)`` call runs the search and caches the
    ``SearchResult``; the second call at the same ``h`` (for the other side)
    reads ``result.strategy[side]`` from the cached result.

    When ``agent_a`` and ``agent_b`` are *different* ``SearchAgent`` objects
    (head-to-head mode), each runs its own independent search — they happen to
    search from the same handle, but neither sees the other's cached result.

    Greedy tie-break: lowest action index among all max-probability actions.
    """

    def __init__(self, net: CVPN, config: SearchConfig) -> None:
        self._net = net
        self._config = config
        # 1-slot cache: (handle, SearchResult) or None
        self._cached: tuple[int, SearchResult] | None = None

    def act(self, view: StateView, side: str, sim: SimClient, handle: int) -> int:
        if self._cached is None or self._cached[0] != handle:
            result = search(view, sim, self._net, from_handle=handle, config=self._config)
            self._cached = (handle, result)
        else:
            result = self._cached[1]

        strategy = result.strategy.get(side, {})
        if not strategy:
            raise ValueError(f"SearchAgent: empty strategy for side {side} (empty legal mask?)")

        # Greedy: max probability, lowest index on ties
        return min(strategy.keys(), key=lambda k: (-strategy[k], k))


class PolicyAgent:
    """Greedy agent using only the CVPN policy head (no tree search).

    Fast mode: one network forward pass per decision, no SimClient clones.
    The CVPN masks illegal actions to ``-inf``, so ``argmax`` is always legal.
    """

    def __init__(self, net: CVPN) -> None:
        self._net = net

    def act(self, view: StateView, side: str, sim: SimClient, handle: int) -> int:
        obs = encode(view, side)
        device = next(self._net.parameters()).device
        obs = obs.to(device)
        with torch.no_grad():
            logits, _ = self._net(obs)  # [A], -inf at illegal positions
        if torch.all(logits == float("-inf")):
            raise ValueError(f"PolicyAgent: no legal actions for side {side}")
        return int(torch.argmax(logits).item())


class RandomAgent:
    """Uniform-random agent -- no network, no search.

    Picks uniformly from the legal mask.  Used as an exploitability floor
    when comparing checkpoints against a random-play baseline.
    """

    def __init__(self, rng: random.Random) -> None:
        self._rng = rng

    def act(self, view: StateView, side: str, sim: SimClient, handle: int) -> int:
        mask = action_space.legal_mask(view.legal.get(side), view.phase)
        legal_indices = [int(i) for i, v in enumerate(mask) if v]
        if not legal_indices:
            raise ValueError(f"RandomAgent: no legal actions for side {side}")
        return self._rng.choice(legal_indices)


# ---------------------------------------------------------------------------
# GameResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GameResult:
    """Outcome of one evaluated game.

    ``turns`` is ``StateView.snapshot.turn`` at the terminal state, or the
    last known turn number if the game was aborted.  Aborted games carry a
    human-readable ``abort_reason`` for diagnostics.
    """

    outcome: Literal["p1_win", "p2_win", "draw", "aborted"]
    turns: int
    abort_reason: str | None = None


# ---------------------------------------------------------------------------
# Game runner
# ---------------------------------------------------------------------------


def play_game(
    agent_p1: Agent,
    agent_p2: Agent,
    team_a: list[dict],
    team_b: list[dict],
    sim: SimClient,
    *,
    seed: int,
    max_decisions: int = 300,
    on_terminal: "((SimClient, int) -> None) | None" = None,
) -> GameResult:
    """Play one complete game and return a ``GameResult``.

    ``team_a`` maps to p1, ``team_b`` to p2 — matching ``sim.new_battle`` and
    ``self_play.run``'s convention.

    The loop mirrors ``self_play.run``'s battle walk (lines 262-354) minus all
    tuple/z buffering:

    * Empty phases advance the engine without calling any agent.
    * Forced decisions (every acting side has exactly one legal action) are
      auto-stepped without calling any agent and without counting toward
      ``max_decisions``.
    * Genuine decisions call each acting agent, build the joint choice, step,
      and increment ``decision_idx``.

    Abort paths (all return ``GameResult("aborted", ...)``):

    * ``max_decisions`` genuine decisions reached without terminal.
    * ``SimError`` on ``sim.step`` — engine rejected the choice.
    * An agent raises (e.g. empty strategy / no legal actions → ``"no_action"``).

    If ``on_terminal`` is provided, it is called with (sim, handle) immediately
    before the final release on a successful terminal. Use this to extract the
    protocol log or other final-state data without duplicating the game loop.

    The function always releases the current handle before returning.
    """
    agents: dict[str, Agent] = {"p1": agent_p1, "p2": agent_p2}
    game_rng = random.Random(seed)
    battle_seed = _rng_seed(game_rng)

    handle: int | None = None
    turns_so_far: int = 0

    try:
        handle, view = sim.new_battle(team_a, team_b, seed=battle_seed)
        decision_idx = 0

        while not view.terminal and decision_idx < max_decisions:
            turns_so_far = view.snapshot.turn

            # --- Empty phase: no acting sides ---
            if not view.to_move or view.phase == "none":
                step_seed = _rng_seed(game_rng)
                res = sim.step(handle, {}, step_seed)
                sim.release(handle)
                handle, view = res.child, res.view
                continue

            # --- Forced decision: auto-step, no agent, no counter ---
            forced = action_space.forced_actions(view.legal, view.to_move, view.phase)
            if forced is not None:
                choices = {s: action_space.action_to_choice_contextual(idx, view.legal.get(s)) for s, idx in forced.items()}
                step_seed = _rng_seed(game_rng)
                res = sim.step(handle, choices, step_seed)
                sim.release(handle)
                handle, view = res.child, res.view
                continue

            # --- Genuine decision: call agents, then step ---
            choices: dict[str, str] = {}
            for side in view.to_move:
                try:
                    idx = agents[side].act(view, side, sim, handle)
                except Exception as exc:
                    sim.release(handle)
                    handle = None
                    return GameResult("aborted", turns_so_far, f"no_action: {exc}")
                choices[side] = action_space.action_to_choice_contextual(idx, view.legal.get(side))

            step_seed = _rng_seed(game_rng)
            try:
                res = sim.step(handle, choices, step_seed)
            except SimError as exc:
                sim.release(handle)
                handle = None
                logger.debug("play_game: SimError after genuine decision: %s (choices=%s)", exc, choices)
                return GameResult("aborted", turns_so_far, f"sim_error: {exc}")

            sim.release(handle)
            handle, view = res.child, res.view
            decision_idx += 1

        # --- Post-loop: classify ---
        final_turns = view.snapshot.turn

        if not view.terminal:
            # max_decisions exhausted
            sim.release(handle)
            handle = None
            return GameResult("aborted", final_turns, "max_decisions")

        # Terminal: let callers extract data (e.g. protocol log) before release
        if on_terminal is not None:
            on_terminal(sim, handle)
        sim.release(handle)
        handle = None
        utility = view.utility
        if utility is None:
            return GameResult("draw", final_turns)
        p1_util = utility.get("p1", 0.0)
        if p1_util is None:
            return GameResult("draw", final_turns)
        if p1_util > 0:
            return GameResult("p1_win", final_turns)
        if p1_util < 0:
            return GameResult("p2_win", final_turns)
        return GameResult("draw", final_turns)

    except SimError as exc:
        logger.warning("play_game: top-level SimError: %s", exc)
        if handle is not None:
            with contextlib.suppress(SimError):
                sim.release(handle)
        return GameResult("aborted", turns_so_far, f"sim_error: {exc}")


# ---------------------------------------------------------------------------
# Report types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CurriculumReport:
    """Aggregate results from a single-net curriculum eval run.

    One net drives **both** sides of a fixed matchup.  The favored side is
    whichever team the caller designates (``EvalConfig.favored_side``, default
    ``"p1"`` = team_a = Fire in Stage 0).

    Aborted games are excluded from every rate denominator and reported
    separately as a health count.
    """

    n_games: int
    favored_wins: int
    underdog_wins: int
    draws: int
    aborted: int
    favored_turns: tuple[int, ...]  # snapshot.turn for favored-side wins
    all_turns: tuple[int, ...]  # snapshot.turn for all decided (non-aborted) games

    @property
    def favored_win_rate(self) -> float:
        """Fraction of decided games (wins + losses + draws) won by the favored side."""
        decided = self.favored_wins + self.underdog_wins + self.draws
        if decided == 0:
            return 0.0
        return self.favored_wins / decided

    @property
    def draw_rate(self) -> float:
        """Fraction of decided games that ended in a draw."""
        decided = self.favored_wins + self.underdog_wins + self.draws
        if decided == 0:
            return 0.0
        return self.draws / decided

    @property
    def mean_favored_turns(self) -> float | None:
        """Mean engine turns for games the favored side won.  None if none won."""
        if not self.favored_turns:
            return None
        return statistics.mean(self.favored_turns)

    @property
    def median_favored_turns(self) -> float | None:
        """Median engine turns for games the favored side won.  None if none won."""
        if not self.favored_turns:
            return None
        return statistics.median(self.favored_turns)

    @property
    def overall_mean_turns(self) -> float | None:
        """Mean engine turns across all decided games.  None if no decided games."""
        if not self.all_turns:
            return None
        return statistics.mean(self.all_turns)


@dataclass(frozen=True)
class HeadToHeadReport:
    """Aggregate results from a head-to-head eval between two different agents.

    ``agent_a`` drives p1 / team_a; ``agent_b`` drives p2 / team_b.  No
    side-swap is performed — results are legible without it.

    Aborted games excluded from every rate denominator.
    """

    n_games: int
    a_wins: int  # p1 / team_a wins
    b_wins: int  # p2 / team_b wins
    draws: int
    aborted: int
    a_turns: tuple[int, ...]  # snapshot.turn for a-wins
    all_turns: tuple[int, ...]  # snapshot.turn for all decided games

    @property
    def a_win_rate(self) -> float:
        """Fraction of decided games (wins + losses + draws) won by agent_a."""
        decided = self.a_wins + self.b_wins + self.draws
        if decided == 0:
            return 0.0
        return self.a_wins / decided

    @property
    def draw_rate(self) -> float:
        """Fraction of decided games that ended in a draw."""
        decided = self.a_wins + self.b_wins + self.draws
        if decided == 0:
            return 0.0
        return self.draws / decided

    @property
    def mean_a_turns(self) -> float | None:
        """Mean engine turns for games agent_a won.  None if agent_a never won."""
        if not self.a_turns:
            return None
        return statistics.mean(self.a_turns)

    @property
    def overall_mean_turns(self) -> float | None:
        """Mean engine turns across all decided games.  None if no decided games."""
        if not self.all_turns:
            return None
        return statistics.mean(self.all_turns)


# ---------------------------------------------------------------------------
# Eval configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalConfig:
    """Knobs for an evaluation run.

    ``eval_search_config`` defaults to a budget lighter than the training
    default (``SearchConfig``).  Comparisons must use the same budget for both
    checkpoints to be meaningful.

    ``use_search=False`` switches to ``PolicyAgent`` (no tree, single forward
    pass per decision) — much faster for a quick sanity check.
    """

    n_games: int = 50
    max_decisions: int = 300
    master_seed: int = 0
    use_search: bool = True
    favored_side: Literal["p1", "p2"] = "p1"  # curriculum: team_a (Fire) is p1
    eval_search_config: SearchConfig = field(
        default_factory=lambda: SearchConfig(
            k_actions=6,
            max_chance_children=3,
            expansion_budget=8,
            cfr_iters_per_expansion=5,
            c_puct=2.0,
        )
    )
    device: str = "cpu"


# ---------------------------------------------------------------------------
# Metric entry points
# ---------------------------------------------------------------------------


def curriculum_report(
    net: CVPN,
    matchup: MatchupSource,
    sim: SimClient,
    *,
    config: EvalConfig | None = None,
) -> CurriculumReport:
    """Measure favored-team win-rate and win speed for ``net`` on ``matchup``.

    A **single** agent instance is used for both sides so that the search
    cache fires: one search per turn for simultaneous decisions.  The favored
    side is set by ``config.favored_side`` (default ``"p1"``).

    ``n_games`` games are played from seeds derived from ``config.master_seed``.
    Aborted games count toward ``aborted`` but are excluded from all rates.
    """
    if config is None:
        config = EvalConfig()

    if config.favored_side not in ("p1", "p2"):
        raise ValueError(f"favored_side must be 'p1' or 'p2', got {config.favored_side!r}")

    if config.use_search:
        agent: Agent = SearchAgent(net, config.eval_search_config)
    else:
        agent = PolicyAgent(net)

    master_rng = random.Random(config.master_seed)

    favored_wins = 0
    underdog_wins = 0
    draws = 0
    aborted = 0
    favored_turns_list: list[int] = []
    all_turns_list: list[int] = []

    for _ in range(config.n_games):
        game_seed = master_rng.randint(0, 2**32 - 1)
        game_rng = random.Random(game_seed)
        team_a, team_b = matchup.sample(game_rng)
        play_seed = game_rng.randint(0, 2**32 - 1)

        result = play_game(
            agent,
            agent,  # same object → 1 search/turn via handle cache
            team_a,
            team_b,
            sim,
            seed=play_seed,
            max_decisions=config.max_decisions,
        )

        if result.outcome == "aborted":
            aborted += 1
            logger.debug("curriculum_report: game aborted (%s)", result.abort_reason)
            continue

        all_turns_list.append(result.turns)

        favored_won = result.outcome == f"{config.favored_side}_win"
        if favored_won:
            favored_wins += 1
            favored_turns_list.append(result.turns)
        elif result.outcome == "draw":
            draws += 1
        else:
            underdog_wins += 1

    return CurriculumReport(
        n_games=config.n_games,
        favored_wins=favored_wins,
        underdog_wins=underdog_wins,
        draws=draws,
        aborted=aborted,
        favored_turns=tuple(favored_turns_list),
        all_turns=tuple(all_turns_list),
    )


def head_to_head(
    agent_a: Agent,
    agent_b: Agent,
    matchup: MatchupSource,
    sim: SimClient,
    *,
    config: EvalConfig | None = None,
) -> HeadToHeadReport:
    """Measure win-rate between two different agents on ``matchup``.

    ``agent_a`` drives p1 / team_a; ``agent_b`` drives p2 / team_b.  No
    side-swap is performed (keeps results legible; see docs/vibes-decisions.md
    §13.1).

    Each agent runs its own independent search per turn — two searches/turn
    total for two ``SearchAgent`` instances.
    """
    if config is None:
        config = EvalConfig()

    master_rng = random.Random(config.master_seed)

    a_wins = 0
    b_wins = 0
    draws = 0
    aborted = 0
    a_turns_list: list[int] = []
    all_turns_list: list[int] = []

    for _ in range(config.n_games):
        game_seed = master_rng.randint(0, 2**32 - 1)
        game_rng = random.Random(game_seed)
        team_a, team_b = matchup.sample(game_rng)
        play_seed = game_rng.randint(0, 2**32 - 1)

        result = play_game(
            agent_a,
            agent_b,
            team_a,
            team_b,
            sim,
            seed=play_seed,
            max_decisions=config.max_decisions,
        )

        if result.outcome == "aborted":
            aborted += 1
            logger.debug("head_to_head: game aborted (%s)", result.abort_reason)
            continue

        all_turns_list.append(result.turns)

        if result.outcome == "p1_win":
            a_wins += 1
            a_turns_list.append(result.turns)
        elif result.outcome == "p2_win":
            b_wins += 1
        else:
            draws += 1

    return HeadToHeadReport(
        n_games=config.n_games,
        a_wins=a_wins,
        b_wins=b_wins,
        draws=draws,
        aborted=aborted,
        a_turns=tuple(a_turns_list),
        all_turns=tuple(all_turns_list),
    )


def fitness(
    checkpoint_path: str | Path,
    opponents: list[Agent | str | Path],
    matchup: MatchupSource,
    sim: SimClient,
    *,
    config: EvalConfig | None = None,
) -> dict[str, HeadToHeadReport]:
    """Load a checkpoint and run head-to-head against each opponent.

    ``opponents`` entries may be:
    * A pre-built ``Agent`` instance (e.g. ``RandomAgent``).
    * A path string / ``Path`` to another checkpoint — its ``SearchAgent`` is
      built with the same ``config.eval_search_config``.

    Returns a dict mapping a label (opponent path string or class name) to a
    ``HeadToHeadReport``.  This satisfies the ``fitness(checkpoint) -> metric``
    shape from ``docs/architecture/repo-architecture.md`` §3.8.

    Note: this entry point always uses ``SearchAgent`` for the subject
    checkpoint and all checkpoint-based opponents, regardless of
    ``config.use_search``.
    """
    if config is None:
        config = EvalConfig()

    # fitness() always uses SearchAgent for checkpoint-based agents.
    # config.use_search=False is meaningless here and almost certainly a
    # caller mistake — raise early rather than silently ignoring the flag.
    if not config.use_search:
        raise ValueError(
            "fitness() always uses SearchAgent for checkpoint-based agents; "
            "config.use_search=False has no effect and is not supported. "
            "Pass config.use_search=True (the default) or omit config entirely."
        )

    loaded = load_checkpoint(checkpoint_path, map_location=config.device)
    net = loaded.net
    net.eval()
    agent_a = SearchAgent(net, config.eval_search_config)

    results: dict[str, HeadToHeadReport] = {}
    for opp in opponents:
        if isinstance(opp, (str, Path)):
            opp_loaded = load_checkpoint(opp, map_location=config.device)
            opp_net = opp_loaded.net
            opp_net.eval()
            opp_agent: Agent = SearchAgent(opp_net, config.eval_search_config)
            label = str(opp)
        else:
            opp_agent = opp
            label = type(opp).__name__

        results[label] = head_to_head(agent_a, opp_agent, matchup, sim, config=config)

    return results
