"""Unit tests for the encoder module — feature encoding helpers and full encode().

Fixtures are derived from a real worker snapshot (tests/fixtures/real_snapshot.json)
so their shape can never drift from the actual wire format. The _minimal_mon factory
loads and mutates the ground-truth JSON rather than hand-building dicts.

Sub-encoder tests assert on the sub-encoder outputs directly (short fixed-width
vectors), eliminating fragile hand-derived offset arithmetic.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from action_space import A
from encoder import (
    ENTITY_FEATURE_DIM,
    FIELD_FEATURE_DIM,
    NUM_NATURES,
    NUM_STATUS,
    SCALAR_FEATURE_DIM,
    SIDE_FEATURE_DIM,
    _boost_stages,
    _encode_field,
    _encode_move_ids,
    _encode_pokemon_features,
    _encode_scalars,
    _encode_side,
    _gravity,
    _hp_fraction,
    _move_pp_flags,
    _nature_onehot,
    _norm_stats,
    _phase_onehot,
    _side_hazards,
    _side_screen,
    _slot_flags,
    _status_onehot,
    _terrain_onehot_dur,
    _trick_room,
    _turn_norm,
    _volatile_counters,
    _weather_onehot_dur,
    _whose_decision,
    encode,
)
from state_types import (
    BattleSnapshot,
    FieldSnapshot,
    MoveSnapshot,
    PokemonSnapshot,
    PseudoWeatherEntry,
    SideConditionSnapshot,
    SideSnapshot,
    StateView,
)

# ---------------------------------------------------------------------------
# Load the ground-truth fixture captured from a real worker
# ---------------------------------------------------------------------------

_FIXTURE_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "real_snapshot.json"
with open(_FIXTURE_PATH, encoding="utf-8") as _f:
    _REAL = json.load(_f)

# The first pokemon from the real team-preview snapshot — canonical wire shape
_REAL_MON_DICT = _REAL["team_preview"]["snapshot"]["sides"][0]["pokemon"][0]


# ---------------------------------------------------------------------------
# Helpers: minimal data factories (derived from real snapshot)
# ---------------------------------------------------------------------------


def _minimal_mon(**overrides) -> PokemonSnapshot:
    """Create a PokemonSnapshot from the real fixture, with optional overrides."""
    data = {**_REAL_MON_DICT, **overrides}
    return PokemonSnapshot.model_validate(data)


def _minimal_field(**overrides) -> FieldSnapshot:
    """Create a FieldSnapshot with defaults matching an empty field."""
    base = {"weather": None, "weatherDuration": None, "terrain": None, "terrainDuration": None, "pseudoWeather": {}}
    base.update(overrides)
    return FieldSnapshot.model_validate(base)


def _minimal_side(pokemon: list[PokemonSnapshot] | None = None, side_conditions: dict | None = None, side_id: str = "p1") -> SideSnapshot:
    """Create a SideSnapshot with optional overrides."""
    if pokemon is None:
        pokemon = [_minimal_mon() for _ in range(6)]
    conds = side_conditions or {}
    return SideSnapshot(id=side_id, sideConditions=conds, pokemon=pokemon)


def _minimal_view(
    my_pokemon: list[PokemonSnapshot] | None = None,
    opp_pokemon: list[PokemonSnapshot] | None = None,
    field: FieldSnapshot | None = None,
    my_side_conds: dict | None = None,
    opp_side_conds: dict | None = None,
    phase: str = "move",
    turn: int = 1,
    to_move: list | None = None,
) -> StateView:
    """Build a minimal StateView for testing encode()."""
    if my_pokemon is None:
        my_pokemon = [_minimal_mon() for _ in range(6)]
    if opp_pokemon is None:
        opp_pokemon = [_minimal_mon(species="venusaur", ability="chlorophyll") for _ in range(6)]
    if field is None:
        field = _minimal_field()
    if to_move is None:
        to_move = ["p1"]

    return StateView(
        phase=phase,
        terminal=False,
        to_move=to_move,
        snapshot=BattleSnapshot(
            turn=turn,
            sides=[
                _minimal_side(my_pokemon, my_side_conds, "p1"),
                _minimal_side(opp_pokemon, opp_side_conds, "p2"),
            ],
            field=field,
        ),
        legal={
            "p1": {"active": [], "side": {"pokemon": []}},
            "p2": {"active": [], "side": {"pokemon": []}},
        },
        utility=None,
    )


# ---------------------------------------------------------------------------
# Entity sub-encoder tests
# ---------------------------------------------------------------------------


class TestHpFraction:
    """Test _hp_fraction sub-encoder."""

    def test_half_hp(self):
        vec = _hp_fraction(_minimal_mon(hp=50, maxhp=100))
        assert vec.shape == (1,)
        assert vec[0].item() == pytest.approx(0.5)

    def test_full_hp(self):
        vec = _hp_fraction(_minimal_mon(hp=200, maxhp=200))
        assert vec[0].item() == pytest.approx(1.0)

    def test_zero_hp(self):
        vec = _hp_fraction(_minimal_mon(hp=0, maxhp=200))
        assert vec[0].item() == pytest.approx(0.0)


class TestNormStats:
    """Test _norm_stats sub-encoder."""

    def test_shape(self):
        vec = _norm_stats(_minimal_mon())
        assert vec.shape == (6,)

    def test_known_stat(self):
        vec = _norm_stats(_minimal_mon(stats={"hp": 200, "atk": 120, "def": 90, "spa": 150, "spd": 100, "spe": 130}))
        # hp stat = 200, normalized by MAX_STAT=200 -> 1.0
        assert vec[0].item() == pytest.approx(1.0)


class TestBoostStages:
    """Test _boost_stages sub-encoder."""

    def test_shape(self):
        vec = _boost_stages(_minimal_mon())
        assert vec.shape == (7,)

    def test_max_boost(self):
        vec = _boost_stages(_minimal_mon(boosts={"atk": 6, "def": -6, "spa": 0, "spd": 0, "spe": 0, "accuracy": 0, "evasion": 0}))
        assert vec[0].item() == pytest.approx(1.0)      # atk +6/6
        assert vec[1].item() == pytest.approx(-1.0)      # def -6/6


class TestStatusOnehot:
    """Test _status_onehot sub-encoder."""

    def test_shape(self):
        vec = _status_onehot(_minimal_mon())
        assert vec.shape == (NUM_STATUS,)

    def test_burn(self):
        vec = _status_onehot(_minimal_mon(status="brn"))
        assert vec[0].item() == 1.0
        assert vec[NUM_STATUS - 1].item() == 0.0

    def test_none(self):
        vec = _status_onehot(_minimal_mon(status=None))
        assert vec[NUM_STATUS - 1].item() == 1.0
        for i in range(NUM_STATUS - 1):
            assert vec[i].item() == 0.0


class TestNatureOnehot:
    """Test _nature_onehot sub-encoder."""

    def test_shape(self):
        vec = _nature_onehot(_minimal_mon())
        assert vec.shape == (NUM_NATURES,)

    def test_exactly_one_hot(self):
        vec = _nature_onehot(_minimal_mon(nature="Adamant"))
        assert vec.sum().item() == pytest.approx(1.0)

    def test_from_real_fixture(self):
        """Nature from the real fixture should produce exactly one hot bit."""
        vec = _nature_onehot(_minimal_mon())
        assert vec.sum().item() == pytest.approx(1.0)


class TestMovePpFlags:
    """Test _move_pp_flags sub-encoder."""

    def test_shape_full_moveset(self):
        mon = _minimal_mon()
        vec = _move_pp_flags(mon)
        assert vec.shape == (8,)

    def test_disabled_flag(self):
        mon = _minimal_mon(moves=[
            {"id": "heatwave", "pp": 10, "maxpp": 10, "disabled": True},
            {"id": "protect", "pp": 16, "maxpp": 16, "disabled": False},
            {"id": "airslash", "pp": 15, "maxpp": 15, "disabled": False},
            {"id": "solarbeam", "pp": 10, "maxpp": 10, "disabled": False},
        ])
        vec = _move_pp_flags(mon)
        assert vec[1].item() == 1.0   # move 0 disabled
        assert vec[3].item() == 0.0   # move 1 not disabled


class TestVolatileCounters:
    """Test _volatile_counters sub-encoder."""

    def test_shape(self):
        vec = _volatile_counters(_minimal_mon())
        assert vec.shape == (3,)

    def test_all_zero_for_default(self):
        vec = _volatile_counters(_minimal_mon())
        assert vec.sum().item() == pytest.approx(0.0)


class TestSlotFlags:
    """Test _slot_flags sub-encoder."""

    def test_shape(self):
        vec = _slot_flags(_minimal_mon(), is_opponent=False)
        assert vec.shape == (8,)

    def test_active_left(self):
        vec = _slot_flags(_minimal_mon(active=True, position=0, fainted=False), is_opponent=False)
        # is_active=1, is_bench=0, is_fainted=0, item_consumed=0, pos=(1,0,0), side=0
        assert vec[0].item() == 1.0   # is_active
        assert vec[4].item() == 1.0   # active-left
        assert vec[5].item() == 0.0   # active-right
        assert vec[6].item() == 0.0   # bench

    def test_active_right(self):
        vec = _slot_flags(_minimal_mon(active=True, position=1, fainted=False), is_opponent=False)
        assert vec[4].item() == 0.0   # active-left
        assert vec[5].item() == 1.0   # active-right
        assert vec[6].item() == 0.0   # bench

    def test_bench(self):
        vec = _slot_flags(_minimal_mon(active=False, position=2, fainted=False), is_opponent=False)
        assert vec[0].item() == 0.0   # not active
        assert vec[1].item() == 1.0   # is_bench
        assert vec[4].item() == 0.0   # active-left
        assert vec[5].item() == 0.0   # active-right
        assert vec[6].item() == 1.0   # bench

    def test_opponent_side_flag(self):
        vec_mine = _slot_flags(_minimal_mon(), is_opponent=False)
        vec_opp = _slot_flags(_minimal_mon(), is_opponent=True)
        assert vec_mine[7].item() == 0.0
        assert vec_opp[7].item() == 1.0


# ---------------------------------------------------------------------------
# _encode_pokemon_features (composite)
# ---------------------------------------------------------------------------


class TestEncodePokemonFeatures:
    """Test the full entity feature vector is correctly assembled."""

    def test_output_shape(self):
        feats = _encode_pokemon_features(_minimal_mon())
        assert feats.shape == (ENTITY_FEATURE_DIM,)

    def test_hp_fraction_is_first(self):
        feats = _encode_pokemon_features(_minimal_mon(hp=50, maxhp=100))
        assert feats[0].item() == pytest.approx(0.5)

    def test_opponent_side_flag_is_last(self):
        feats_mine = _encode_pokemon_features(_minimal_mon(), is_opponent=False)
        feats_opp = _encode_pokemon_features(_minimal_mon(), is_opponent=True)
        assert feats_mine[-1].item() == 0.0
        assert feats_opp[-1].item() == 1.0


# ---------------------------------------------------------------------------
# _encode_move_ids
# ---------------------------------------------------------------------------


class TestEncodeMoveIds:
    """Test move ID encoding."""

    def test_four_moves(self):
        moves = [MoveSnapshot(id="heatwave", pp=10, maxpp=10, disabled=False),
                 MoveSnapshot(id="protect", pp=16, maxpp=16, disabled=False),
                 MoveSnapshot(id="airslash", pp=15, maxpp=15, disabled=False),
                 MoveSnapshot(id="solarbeam", pp=10, maxpp=10, disabled=False)]
        ids = _encode_move_ids(moves)
        assert ids.shape == (4,)
        assert (ids > 0).all()

    def test_fewer_than_four_moves_padded_with_zero(self):
        moves = [MoveSnapshot(id="heatwave", pp=10, maxpp=10, disabled=False),
                 MoveSnapshot(id="protect", pp=16, maxpp=16, disabled=False)]
        ids = _encode_move_ids(moves)
        assert ids[0].item() > 0
        assert ids[1].item() > 0
        assert ids[2].item() == 0
        assert ids[3].item() == 0

    def test_empty_moves(self):
        ids = _encode_move_ids([])
        assert ids.shape == (4,)
        assert (ids == 0).all()


# ---------------------------------------------------------------------------
# Field sub-encoder tests
# ---------------------------------------------------------------------------


class TestWeatherOnehotDur:
    """Test _weather_onehot_dur sub-encoder."""

    def test_shape(self):
        vec = _weather_onehot_dur(_minimal_field())
        assert vec.shape == (5,)

    def test_empty_all_zeros(self):
        vec = _weather_onehot_dur(_minimal_field())
        assert (vec == 0).all()

    def test_rain(self):
        vec = _weather_onehot_dur(_minimal_field(weather="raindance", weatherDuration=5))
        assert vec[0].item() == 1.0
        assert vec[4].item() == pytest.approx(5 / 20)

    def test_sun(self):
        vec = _weather_onehot_dur(_minimal_field(weather="sunnyday", weatherDuration=3))
        assert vec[1].item() == 1.0


class TestTerrainOnehotDur:
    """Test _terrain_onehot_dur sub-encoder."""

    def test_shape(self):
        vec = _terrain_onehot_dur(_minimal_field())
        assert vec.shape == (5,)

    def test_electric(self):
        vec = _terrain_onehot_dur(_minimal_field(terrain="electricterrain", terrainDuration=4))
        assert vec[0].item() == 1.0


class TestTrickRoom:
    """Test _trick_room sub-encoder."""

    def test_shape(self):
        vec = _trick_room(_minimal_field())
        assert vec.shape == (2,)

    def test_active(self):
        field = _minimal_field(pseudoWeather={"trickroom": PseudoWeatherEntry(duration=3)})
        vec = _trick_room(field)
        assert vec[0].item() == 1.0
        assert vec[1].item() == pytest.approx(3 / 20)

    def test_inactive(self):
        vec = _trick_room(_minimal_field())
        assert vec[0].item() == 0.0
        assert vec[1].item() == 0.0


class TestGravity:
    """Test _gravity sub-encoder."""

    def test_shape(self):
        vec = _gravity(_minimal_field())
        assert vec.shape == (1,)

    def test_inactive(self):
        assert _gravity(_minimal_field())[0].item() == 0.0


class TestEncodeField:
    """Test the composite _encode_field function."""

    def test_output_shape(self):
        feats = _encode_field(_minimal_field())
        assert feats.shape == (FIELD_FEATURE_DIM,)

    def test_empty_field_all_zeros(self):
        feats = _encode_field(_minimal_field())
        assert (feats == 0).all()


# ---------------------------------------------------------------------------
# Side sub-encoder tests
# ---------------------------------------------------------------------------


class TestSideScreen:
    """Test _side_screen sub-encoder."""

    def test_shape(self):
        vec = _side_screen("tailwind", {})
        assert vec.shape == (2,)

    def test_tailwind_active(self):
        conds = {"tailwind": SideConditionSnapshot(duration=4, layers=None)}
        vec = _side_screen("tailwind", conds)
        assert vec[0].item() == 1.0
        assert vec[1].item() == pytest.approx(4 / 20)

    def test_absent(self):
        vec = _side_screen("tailwind", {})
        assert vec[0].item() == 0.0
        assert vec[1].item() == 0.0


class TestSideHazards:
    """Test _side_hazards sub-encoder."""

    def test_shape(self):
        vec = _side_hazards({})
        assert vec.shape == (2,)

    def test_stealth_rock(self):
        conds = {"stealthrock": SideConditionSnapshot(duration=None, layers=None)}
        vec = _side_hazards(conds)
        assert vec[0].item() == 1.0

    def test_spikes_layers(self):
        conds = {"spikes": SideConditionSnapshot(duration=None, layers=2)}
        vec = _side_hazards(conds)
        assert vec[1].item() == pytest.approx(2 / 3)


class TestEncodeSide:
    """Test the composite _encode_side function."""

    def test_output_shape(self):
        feats = _encode_side(_minimal_side())
        assert feats.shape == (SIDE_FEATURE_DIM,)

    def test_empty_conditions_all_zeros(self):
        feats = _encode_side(_minimal_side())
        assert (feats == 0).all()


# ---------------------------------------------------------------------------
# Scalar sub-encoder tests
# ---------------------------------------------------------------------------


class TestTurnNorm:
    """Test _turn_norm sub-encoder."""

    def test_shape(self):
        view = _minimal_view(turn=5)
        vec = _turn_norm(view.snapshot)
        assert vec.shape == (1,)

    def test_value(self):
        view = _minimal_view(turn=10)
        vec = _turn_norm(view.snapshot)
        assert vec[0].item() == pytest.approx(10 / 20)


class TestPhaseOnehot:
    """Test _phase_onehot sub-encoder."""

    def test_shape(self):
        view = _minimal_view(phase="move")
        vec = _phase_onehot(view)
        assert vec.shape == (4,)

    def test_move_phase(self):
        view = _minimal_view(phase="move")
        vec = _phase_onehot(view)
        assert vec[0].item() == 0.0  # teamPreview
        assert vec[1].item() == 1.0  # move
        assert vec[2].item() == 0.0  # forceSwitch
        assert vec[3].item() == 0.0  # terminal

    def test_team_preview(self):
        view = _minimal_view(phase="teamPreview")
        vec = _phase_onehot(view)
        assert vec[0].item() == 1.0


class TestWhoseDecision:
    """Test _whose_decision sub-encoder."""

    def test_shape(self):
        view = _minimal_view(to_move=["p1"])
        vec = _whose_decision(view, "p1")
        assert vec.shape == (2,)

    def test_both_acting(self):
        view = _minimal_view(to_move=["p1", "p2"])
        vec = _whose_decision(view, "p1")
        assert vec[0].item() == 1.0
        assert vec[1].item() == 1.0

    def test_only_opponent(self):
        view = _minimal_view(to_move=["p2"])
        vec = _whose_decision(view, "p1")
        assert vec[0].item() == 0.0
        assert vec[1].item() == 1.0


class TestEncodeScalars:
    """Test the composite _encode_scalars function."""

    def test_output_shape(self):
        view = _minimal_view(turn=5)
        feats = _encode_scalars(view.snapshot, view, "p1")
        assert feats.shape == (SCALAR_FEATURE_DIM,)


# ---------------------------------------------------------------------------
# Full encode()
# ---------------------------------------------------------------------------


class TestEncode:
    """Test the top-level encode() function with synthetic views."""

    def test_output_shapes_phase1(self):
        view = _minimal_view()
        obs = encode(view, perspective="p1")
        assert obs["entities"].shape == (12, ENTITY_FEATURE_DIM)
        assert obs["ids", "species"].shape == (12,)
        assert obs["ids", "moves"].shape == (12, 4)
        assert obs["action_mask"].shape == (A,)
        assert obs["field"].shape == (FIELD_FEATURE_DIM,)
        assert obs["sides"].shape == (2, SIDE_FEATURE_DIM)
        assert obs["scalars"].shape == (SCALAR_FEATURE_DIM,)
        assert obs["padding_mask"].shape == (12,)

    def test_padding_mask_all_true_in_phase1(self):
        view = _minimal_view()
        obs = encode(view, perspective="p1")
        assert obs["padding_mask"].all()

    def test_belief_weight_all_one_in_phase1(self):
        view = _minimal_view()
        obs = encode(view, perspective="p1")
        assert (obs["belief_weight"] == 1.0).all()

    def test_perspective_flip_slot_ids(self):
        view = _minimal_view()
        obs_p1 = encode(view, perspective="p1")
        obs_p2 = encode(view, perspective="p2")
        assert (obs_p1["slot_id"][:6] == 0).all()
        assert (obs_p1["slot_id"][6:] == 1).all()
        assert (obs_p2["slot_id"][:6] == 0).all()
        assert (obs_p2["slot_id"][6:] == 1).all()

    def test_species_ids_encoded(self):
        view = _minimal_view()
        obs = encode(view, perspective="p1")
        assert (obs["ids", "species"] > 0).all()

    def test_team_preview_mask_when_phase_is_team_preview(self):
        view = _minimal_view(phase="teamPreview")
        obs = encode(view, perspective="p1")
        mask_sum = obs["action_mask"].sum().item()
        assert mask_sum == 360
