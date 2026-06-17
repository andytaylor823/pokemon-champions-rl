"""Unit tests for state_types.py — Pydantic contract models for the TS/Python boundary.

These models are the firewall between the sim-worker (TypeScript) and the encoder
(Python). If they drift from the wire format, the encoder silently produces garbage.
Tests here verify parsing, validation, defaults, and round-trips without needing the
live engine subprocess.
"""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError

from state_types import (
    BattleSnapshot,
    FieldSnapshot,
    MoveSnapshot,
    PokemonSnapshot,
    PseudoWeatherEntry,
    SideConditionSnapshot,
    SideSnapshot,
    StateView,
    StatusStateSnapshot,
    VolatileDetail,
)

# ---------------------------------------------------------------------------
# Load the real fixture for positive-case parsing
# ---------------------------------------------------------------------------

_FIXTURE_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "real_snapshot.json"
with open(_FIXTURE_PATH, encoding="utf-8") as _f:
    _REAL = json.load(_f)


# ---------------------------------------------------------------------------
# Fixture parsing — real wire data validates cleanly
# ---------------------------------------------------------------------------


class TestFixtureParsing:
    """Real sim-worker output must parse without error against our models."""

    def test_team_preview_parses(self):
        view = StateView.model_validate(_REAL["team_preview"])
        assert view.phase == "teamPreview"
        assert view.terminal is False
        assert len(view.snapshot.sides) == 2
        assert len(view.snapshot.sides[0].pokemon) == 6
        assert len(view.snapshot.sides[1].pokemon) == 6

    def test_move_phase_parses(self):
        view = StateView.model_validate(_REAL["move_phase"])
        assert view.phase in ("move", "forceSwitch")
        assert len(view.snapshot.sides) == 2

    def test_all_pokemon_have_required_fields(self):
        view = StateView.model_validate(_REAL["team_preview"])
        for side in view.snapshot.sides:
            for mon in side.pokemon:
                assert mon.species is not None
                assert mon.nature is not None
                assert isinstance(mon.level, int)
                assert isinstance(mon.hp, int)
                assert isinstance(mon.maxhp, int)
                assert isinstance(mon.moves, list)
                assert len(mon.moves) > 0

    def test_move_phase_pokemon_have_stats(self):
        view = StateView.model_validate(_REAL["move_phase"])
        for side in view.snapshot.sides:
            for mon in side.pokemon:
                assert "hp" in mon.stats or mon.fainted

    def test_fixture_round_trips(self):
        """model_validate -> model_dump -> model_validate produces identical result."""
        view = StateView.model_validate(_REAL["team_preview"])
        dumped = view.model_dump()
        view2 = StateView.model_validate(dumped)
        assert view == view2


# ---------------------------------------------------------------------------
# Required field rejection
# ---------------------------------------------------------------------------


class TestRequiredFieldRejection:
    """Missing required fields must raise ValidationError."""

    def test_pokemon_missing_species(self):
        data = deepcopy(_REAL["team_preview"]["snapshot"]["sides"][0]["pokemon"][0])
        del data["species"]
        with pytest.raises(ValidationError):
            PokemonSnapshot.model_validate(data)

    def test_pokemon_missing_hp(self):
        data = deepcopy(_REAL["team_preview"]["snapshot"]["sides"][0]["pokemon"][0])
        del data["hp"]
        with pytest.raises(ValidationError):
            PokemonSnapshot.model_validate(data)

    def test_pokemon_missing_moves(self):
        data = deepcopy(_REAL["team_preview"]["snapshot"]["sides"][0]["pokemon"][0])
        del data["moves"]
        with pytest.raises(ValidationError):
            PokemonSnapshot.model_validate(data)

    def test_pokemon_missing_nature(self):
        data = deepcopy(_REAL["team_preview"]["snapshot"]["sides"][0]["pokemon"][0])
        del data["nature"]
        with pytest.raises(ValidationError):
            PokemonSnapshot.model_validate(data)

    def test_battle_snapshot_missing_turn(self):
        data = deepcopy(_REAL["team_preview"]["snapshot"])
        del data["turn"]
        with pytest.raises(ValidationError):
            BattleSnapshot.model_validate(data)

    def test_battle_snapshot_missing_field(self):
        data = deepcopy(_REAL["team_preview"]["snapshot"])
        del data["field"]
        with pytest.raises(ValidationError):
            BattleSnapshot.model_validate(data)

    def test_field_snapshot_missing_weather(self):
        with pytest.raises(ValidationError):
            FieldSnapshot.model_validate({"terrain": None, "terrainDuration": None, "pseudoWeather": {}})

    def test_move_snapshot_missing_id(self):
        with pytest.raises(ValidationError):
            MoveSnapshot.model_validate({"pp": 10, "maxpp": 10, "disabled": False})

    def test_move_snapshot_missing_pp(self):
        with pytest.raises(ValidationError):
            MoveSnapshot.model_validate({"id": "heatwave", "maxpp": 10, "disabled": False})

    def test_state_view_missing_phase(self):
        data = deepcopy(_REAL["team_preview"])
        del data["phase"]
        with pytest.raises(ValidationError):
            StateView.model_validate(data)

    def test_state_view_missing_snapshot(self):
        data = deepcopy(_REAL["team_preview"])
        del data["snapshot"]
        with pytest.raises(ValidationError):
            StateView.model_validate(data)


# ---------------------------------------------------------------------------
# Nested validation — invalid inner structures rejected
# ---------------------------------------------------------------------------


class TestNestedValidation:
    """Invalid inner types should be caught by Pydantic validators."""

    def test_pokemon_with_invalid_move_type(self):
        data = deepcopy(_REAL["team_preview"]["snapshot"]["sides"][0]["pokemon"][0])
        data["moves"] = ["heatwave", "protect"]  # strings instead of MoveSnapshot dicts
        with pytest.raises(ValidationError):
            PokemonSnapshot.model_validate(data)

    def test_pokemon_with_move_missing_maxpp(self):
        data = deepcopy(_REAL["team_preview"]["snapshot"]["sides"][0]["pokemon"][0])
        data["moves"] = [{"id": "heatwave", "pp": 10, "disabled": False}]  # missing maxpp
        with pytest.raises(ValidationError):
            PokemonSnapshot.model_validate(data)

    def test_pokemon_with_non_int_hp(self):
        data = deepcopy(_REAL["team_preview"]["snapshot"]["sides"][0]["pokemon"][0])
        data["hp"] = "full"
        with pytest.raises(ValidationError):
            PokemonSnapshot.model_validate(data)

    def test_side_snapshot_invalid_pokemon_list(self):
        with pytest.raises(ValidationError):
            SideSnapshot.model_validate({"id": "p1", "sideConditions": {}, "pokemon": "not_a_list"})


# ---------------------------------------------------------------------------
# Optional field defaults
# ---------------------------------------------------------------------------


class TestOptionalFieldDefaults:
    """Optional fields with defaults should produce correct values when omitted."""

    def test_side_condition_snapshot_defaults(self):
        sc = SideConditionSnapshot()
        assert sc.duration is None
        assert sc.layers is None

    def test_side_condition_snapshot_explicit_values(self):
        sc = SideConditionSnapshot(duration=5, layers=2)
        assert sc.duration == 5
        assert sc.layers == 2

    def test_status_state_snapshot_defaults(self):
        ss = StatusStateSnapshot()
        assert ss.stage is None
        assert ss.time is None

    def test_status_state_snapshot_toxic_counter(self):
        ss = StatusStateSnapshot(stage=3, time=None)
        assert ss.stage == 3

    def test_status_state_snapshot_sleep_counter(self):
        ss = StatusStateSnapshot(stage=None, time=2)
        assert ss.time == 2

    def test_volatile_detail_all_defaults(self):
        vd = VolatileDetail()
        assert vd.duration is None
        assert vd.time is None
        assert vd.hp is None
        assert vd.counter is None

    def test_volatile_detail_explicit_values(self):
        vd = VolatileDetail(duration=5, time=2, hp=100, counter=3)
        assert vd.duration == 5
        assert vd.time == 2
        assert vd.hp == 100
        assert vd.counter == 3

    def test_pseudo_weather_entry_default(self):
        pwe = PseudoWeatherEntry()
        assert pwe.duration is None

    def test_pseudo_weather_entry_explicit(self):
        pwe = PseudoWeatherEntry(duration=5)
        assert pwe.duration == 5

    def test_pseudo_weather_duration_zero_vs_none(self):
        """Duration=0 (expired) and None (permanent) are distinguishable."""
        expired = PseudoWeatherEntry(duration=0)
        permanent = PseudoWeatherEntry(duration=None)
        assert expired.duration == 0
        assert permanent.duration is None
        assert expired != permanent


# ---------------------------------------------------------------------------
# Round-trip tests for all model types
# ---------------------------------------------------------------------------


class TestModelRoundTrips:
    """model_dump -> model_validate produces identical instances."""

    def test_move_snapshot_round_trip(self):
        m = MoveSnapshot(id="heatwave", pp=10, maxpp=12, disabled=True)
        assert MoveSnapshot.model_validate(m.model_dump()) == m

    def test_status_state_round_trip(self):
        ss = StatusStateSnapshot(stage=5, time=3)
        assert StatusStateSnapshot.model_validate(ss.model_dump()) == ss

    def test_volatile_detail_round_trip(self):
        vd = VolatileDetail(duration=3, hp=100, counter=2, time=1)
        assert VolatileDetail.model_validate(vd.model_dump()) == vd

    def test_side_condition_round_trip(self):
        sc = SideConditionSnapshot(duration=4, layers=2)
        assert SideConditionSnapshot.model_validate(sc.model_dump()) == sc

    def test_field_snapshot_round_trip(self):
        f = FieldSnapshot(
            weather="raindance", weatherDuration=5,
            terrain="electricterrain", terrainDuration=4,
            pseudoWeather={"trickroom": PseudoWeatherEntry(duration=3)},
        )
        assert FieldSnapshot.model_validate(f.model_dump()) == f

    def test_pokemon_snapshot_round_trip(self):
        data = _REAL["team_preview"]["snapshot"]["sides"][0]["pokemon"][0]
        mon = PokemonSnapshot.model_validate(data)
        assert PokemonSnapshot.model_validate(mon.model_dump()) == mon

    def test_state_view_round_trip(self):
        view = StateView.model_validate(_REAL["team_preview"])
        assert StateView.model_validate(view.model_dump()) == view

    def test_state_view_move_phase_round_trip(self):
        view = StateView.model_validate(_REAL["move_phase"])
        assert StateView.model_validate(view.model_dump()) == view
