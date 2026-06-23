"""Data structures for the GT-CFR search tree.

Contains all node types, config, and result containers used across the search
package. No algorithmic logic beyond ChanceNode.value_p1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

    from state_types import StateView

Side = str  # "p1" | "p2"


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

    strategy: dict[str, dict[int, float]]  # sigma-bar per side over top-k action indices
    value: float  # search-refined CFV for p1 (negate for p2)
    policy_target: dict[str, np.ndarray]  # full [A] vector per side (mass on top-k, zeros elsewhere)


# ---------------------------------------------------------------------------
# Node types (internal, mutable — hold ephemeral regret tables)
# ---------------------------------------------------------------------------


class NodeKind(Enum):
    """Discriminant for search tree nodes."""

    TURN = auto()  # both players to move (simultaneous)
    UNILATERAL = auto()  # one player to move (forced switch)
    CHANCE = auto()  # stochastic outcome of a joint action
    TERMINAL = auto()  # game over — real payoff available


@dataclass
class TurnNode:
    """A simultaneous-move decision node with a k x k action grid.

    Holds two info sets (one per player), each with regret/strategy-sum tables.
    Children are ChanceNodes indexed by (action_idx_p1, action_idx_p2).
    """

    handle: int  # SimClient handle for this state
    view: StateView
    kind: NodeKind  # TURN or UNILATERAL

    # Which sides must act (["p1", "p2"] for TURN, one side for UNILATERAL)
    to_move: list[str] = field(default_factory=list)

    # Top-k action indices per side (set at expansion time)
    actions: dict[str, list[int]] = field(default_factory=dict)

    # Cached CVPN policy prior per side: {side: ndarray[k] probabilities}
    policy_prior: dict[str, np.ndarray] = field(default_factory=dict)

    # CFR+ cumulative regret per side: {side: ndarray[k]}
    cumulative_regret: dict[str, np.ndarray] = field(default_factory=dict)

    # Strategy sum for average strategy: {side: ndarray[k]}
    strategy_sum: dict[str, np.ndarray] = field(default_factory=dict)

    # Visit count per side per action: {side: ndarray[k] int}
    visit_counts: dict[str, np.ndarray] = field(default_factory=dict)

    # Grid of ChanceNode children indexed by (p1_action_pos, p2_action_pos)
    # where pos is the position within the top-k array (0..k-1)
    grid: dict[tuple[int, int], ChanceNode] = field(default_factory=dict)

    # Whether this node has been expanded
    expanded: bool = False


@dataclass
class ChanceNode:
    """A chance node representing one joint cell (a1, a2) with sampled outcomes.

    Holds up to K outcome worlds with uniform weights. CFV = weighted average.
    """

    # Outcome children: list of (handle, cached_value_p1) tuples
    children: list[tuple[int, float]] = field(default_factory=list)

    @property
    def value_p1(self) -> float:
        """Uniform-weighted average of child values (p1 perspective)."""
        if not self.children:
            return 0.0
        return sum(v for _, v in self.children) / len(self.children)
