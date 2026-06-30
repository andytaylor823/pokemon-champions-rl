"""Training-tuple seam types — the stable boundary between the inner and outer loops.

`TrainingTuple` is the GT-CFR analogue of AlphaZero's (state, policy, outcome): the data
each inner-loop search emits and the outer loop (ReplayBuffer -> Trainer) consumes. These
types live in their own module — not inside the producer (``self_play.py``) — so that
``self_play``, ``replay_buffer``, and ``trainer`` all import them from neutral ground.

See ``docs/plans/replay-buffer.md`` §8 and ``docs/architecture/repo-architecture.md`` §3.7.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from obs_bundle import ObsBundle


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
