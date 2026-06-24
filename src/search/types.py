"""Data structures for the GT-CFR search tree.

Contains all node types, config, and result containers used across the search
package. No algorithmic logic lives here — pure data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from state_types import Side, StateView


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SearchConfig:
    """Tuning knobs for the GT-CFR search. Defaults are vibes-based starting guesses."""

    k_actions: int = 6  # top-k joint actions per player (vibes 9.7)
    max_chance_children: int = 5  # chance fan-out cap K (vibes 9.8)
    expansion_budget: int = 24  # PUCT expansions per search (vibes 9.2)
    cfr_iters_per_expansion: int = 10  # CFR+ updates between expansions (vibes 9.10)
    c_puct: float = 2.0  # PUCT exploration constant (vibes 9.10)


@dataclass(frozen=True)
class SearchResult:
    """Output of one search() call — the stable inner/outer loop boundary."""

    strategy: dict[Side, dict[int, float]]  # sigma-bar per side over top-k action indices
    value: float  # search-refined CFV for p1 (negate for p2)
    policy_target: dict[Side, np.ndarray]  # full [A] vector per side (mass on top-k, zeros elsewhere)


# ---------------------------------------------------------------------------
# Node types (internal, mutable — hold ephemeral regret tables)
# ---------------------------------------------------------------------------


@dataclass
class InfoSet:
    """One player's info-set state at a TurnNode: actions, prior, regret, strategy-sum, visits.

    Encapsulates the five parallel arrays that were previously spread across
    separate dicts on TurnNode.
    """

    actions: list[int]  # top-k canonical action indices
    prior: np.ndarray  # CVPN policy prior over top-k (probabilities, sums to 1)
    regret: np.ndarray  # CFR+ cumulative regret [k]
    strategy_sum: np.ndarray  # iteration-weighted strategy accumulator [k]
    visits: np.ndarray  # per-action visit counts [k] (int64)

    @staticmethod
    def from_actions(actions: list[int], prior: np.ndarray) -> InfoSet:
        """Create an InfoSet with zeroed regret/strategy/visits from an action list and prior."""
        k = len(actions)
        return InfoSet(
            actions=actions,
            prior=prior,
            regret=np.zeros(k, dtype=np.float64),
            strategy_sum=np.zeros(k, dtype=np.float64),
            visits=np.zeros(k, dtype=np.int64),
        )

    @staticmethod
    def noop() -> InfoSet:
        """Create a single-action no-op InfoSet for non-acting sides."""
        return InfoSet.from_actions([0], np.array([1.0]))

    @staticmethod
    def empty() -> InfoSet:
        """Create an empty InfoSet (no legal actions — shouldn't happen for non-terminals)."""
        return InfoSet.from_actions([], np.array([]))


@dataclass
class ChanceOutcome:
    """One sampled outcome world under a joint action cell.

    Starts as a frontier leaf (node=None, value from CVPN or terminal utility).
    When expanded during PUCT tree growth, node is set in-place to the child
    TurnNode so CFR+ can recurse through it directly — no external registry needed.
    """

    handle: int  # SimClient handle for this outcome world
    leaf_value_p1: float  # CVPN value or terminal utility — used until expanded
    node: TurnNode | None = None  # set in-place when this outcome becomes a decision node


@dataclass
class TurnNode:
    """A simultaneous-move decision node with a k x k action grid.

    Holds two info sets (one per player) via the `info` dict. Both sides always
    have a valid InfoSet after expansion: the non-acting side in a unilateral
    node gets a single no-op action, so CFR/PUCT code can assume both p1 and p2
    entries exist without conditional fallbacks.
    """

    handle: int  # SimClient handle for this state
    view: StateView

    # Which sides must act (["p1", "p2"] for simultaneous, one side for unilateral)
    to_move: list[str] = field(default_factory=list)

    # Per-side info set: actions, prior, regret, strategy_sum, visits
    info: dict[str, InfoSet] = field(default_factory=dict)

    # Grid of ChanceNode children indexed by (p1_action_pos, p2_action_pos)
    # where pos is the position within the top-k array (0..k-1)
    grid: dict[tuple[int, int], ChanceNode] = field(default_factory=dict)

    # Whether this node has been expanded
    expanded: bool = False


@dataclass
class ChanceNode:
    """A chance node representing one joint cell (a1, a2) with sampled outcomes.

    Holds up to K outcome worlds with uniform weights. CFV = uniform average.
    """

    children: list[ChanceOutcome] = field(default_factory=list)
