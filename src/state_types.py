# ruff: noqa: N815
# Field names are camelCase to match the TypeScript/JSON wire format from types.ts.
"""Python-side contract types mirroring sim/src/types.ts.

These Pydantic models are the single Python source of truth for the
snapshot/view boundary. SimClient validates every worker response against
them so a missing or renamed key raises ValidationError instead of silently
producing zeros in the encoder.

Keep in strict 1:1 correspondence with the TypeScript interfaces in
``sim/src/types.ts``. When a field is added or renamed on the TS side,
update the matching model here.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel


class MoveSnapshot(BaseModel):
    """Per-slot move state (mirrors TS ``MoveSnapshot``)."""

    id: str
    pp: int
    maxpp: int
    disabled: bool


class StatusStateSnapshot(BaseModel):
    """Toxic/sleep counters attached to a status condition."""

    stage: int | None = None
    time: int | None = None


class VolatileDetail(BaseModel):
    """Per-volatile metadata (duration, timer, substitute HP, counter)."""

    duration: int | None = None
    time: int | None = None
    hp: int | None = None
    counter: int | None = None


class PokemonSnapshot(BaseModel):
    """Full pokemon state (mirrors TS ``PokemonSnapshot``)."""

    species: str | None
    nature: str | None
    level: int
    gender: str
    hp: int
    maxhp: int
    fainted: bool
    status: str | None
    statusState: StatusStateSnapshot
    ability: str | None
    item: str | None
    lastItem: str | None
    active: bool
    position: int
    activeTurns: int
    teraType: str | None
    terastallized: str | None
    stats: dict[str, int]
    boosts: dict[str, int]
    moves: list[MoveSnapshot]
    volatiles: list[str]
    volatileDetails: dict[str, VolatileDetail]


class SideConditionSnapshot(BaseModel):
    """Duration + layer count for a single side condition."""

    duration: int | None = None
    layers: int | None = None


class SideSnapshot(BaseModel):
    """Per-side state (mirrors TS ``SideSnapshot``)."""

    id: str
    sideConditions: dict[str, SideConditionSnapshot]
    pokemon: list[PokemonSnapshot]


class PseudoWeatherEntry(BaseModel):
    """Duration wrapper for a pseudo-weather (trick room, gravity, etc.)."""

    duration: int | None = None


class FieldSnapshot(BaseModel):
    """Global field state (mirrors TS ``FieldSnapshot``)."""

    weather: str | None
    weatherDuration: int | None
    terrain: str | None
    terrainDuration: int | None
    pseudoWeather: dict[str, PseudoWeatherEntry]


class BattleSnapshot(BaseModel):
    """Top-level battle snapshot (mirrors TS ``BattleSnapshot``)."""

    turn: int
    field: FieldSnapshot
    sides: list[SideSnapshot]


Side = Literal["p1", "p2"]


class StateView(BaseModel):
    """The omniscient view returned by new_battle / step / view.

    ``legal`` stays as an untyped dict — it's the raw Showdown request
    object consumed by ``action_space.py``, not the encoder.
    """

    phase: str
    to_move: list[str]
    legal: dict[str, Any]
    snapshot: BattleSnapshot
    terminal: bool
    utility: dict[str, float] | None
