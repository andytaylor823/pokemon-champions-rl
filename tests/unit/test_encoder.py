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


# ===========================================================================
# EXHAUSTIVE BATTLE-STATE VARIATION TESTS
# ===========================================================================


# ---------------------------------------------------------------------------
# Status conditions — every possible status
# ---------------------------------------------------------------------------


class TestStatusOnehotExhaustive:
    """Every status condition must produce the correct one-hot index."""

    @pytest.mark.parametrize("status,expected_idx", [
        ("brn", 0),
        ("par", 1),
        ("slp", 2),
        ("frz", 3),
        ("tox", 4),
        ("psn", 5),
    ])
    def test_each_status(self, status, expected_idx):
        vec = _status_onehot(_minimal_mon(status=status))
        assert vec[expected_idx].item() == 1.0
        assert vec.sum().item() == pytest.approx(1.0)

    def test_none_status(self):
        vec = _status_onehot(_minimal_mon(status=None))
        assert vec[NUM_STATUS - 1].item() == 1.0
        assert vec.sum().item() == pytest.approx(1.0)

    def test_empty_string_status(self):
        vec = _status_onehot(_minimal_mon(status=""))
        assert vec[NUM_STATUS - 1].item() == 1.0

    def test_unknown_status_falls_to_none(self):
        """Non-major status (e.g. confusion is a volatile, not a status)."""
        vec = _status_onehot(_minimal_mon(status="confusion"))
        assert vec[NUM_STATUS - 1].item() == 1.0

    def test_status_state_toxic_counter_accessible(self):
        """Verify statusState with toxic counter parses and is accessible."""
        mon = _minimal_mon(statusState={"stage": 3, "time": None}, status="tox")
        assert mon.statusState.stage == 3
        vec = _status_onehot(mon)
        assert vec[4].item() == 1.0  # tox = index 4

    def test_status_state_sleep_counter_accessible(self):
        """Verify statusState with sleep counter parses."""
        mon = _minimal_mon(statusState={"stage": None, "time": 2}, status="slp")
        assert mon.statusState.time == 2
        vec = _status_onehot(mon)
        assert vec[2].item() == 1.0  # slp = index 2


# ---------------------------------------------------------------------------
# Volatile counters — substitute, stall, active turns, and many others
# ---------------------------------------------------------------------------


class TestVolatileCountersExhaustive:
    """Test every volatile state the encoder reads from, plus many it ignores."""

    def test_substitute_half_hp(self):
        mon = _minimal_mon(volatileDetails={"substitute": {"hp": 100}})
        vec = _volatile_counters(mon)
        assert vec[0].item() == pytest.approx(100 / 200)

    def test_substitute_full_hp(self):
        mon = _minimal_mon(volatileDetails={"substitute": {"hp": 200}})
        vec = _volatile_counters(mon)
        assert vec[0].item() == pytest.approx(1.0)

    def test_substitute_zero_hp(self):
        mon = _minimal_mon(volatileDetails={"substitute": {"hp": 0}})
        vec = _volatile_counters(mon)
        assert vec[0].item() == pytest.approx(0.0)

    def test_substitute_absent(self):
        mon = _minimal_mon(volatileDetails={})
        vec = _volatile_counters(mon)
        assert vec[0].item() == pytest.approx(0.0)

    def test_stall_counter_1(self):
        mon = _minimal_mon(volatileDetails={"stall": {"counter": 1}})
        vec = _volatile_counters(mon)
        assert vec[1].item() == pytest.approx(1 / 6)

    def test_stall_counter_max(self):
        mon = _minimal_mon(volatileDetails={"stall": {"counter": 6}})
        vec = _volatile_counters(mon)
        assert vec[1].item() == pytest.approx(1.0)

    def test_stall_counter_zero(self):
        mon = _minimal_mon(volatileDetails={"stall": {"counter": 0}})
        vec = _volatile_counters(mon)
        assert vec[1].item() == pytest.approx(0.0)

    def test_stall_counter_absent(self):
        mon = _minimal_mon(volatileDetails={})
        vec = _volatile_counters(mon)
        assert vec[1].item() == pytest.approx(0.0)

    @pytest.mark.parametrize("turns,expected", [
        (0, 0.0),
        (1, 1 / 20),
        (5, 5 / 20),
        (10, 10 / 20),
        (20, 1.0),
    ])
    def test_active_turns(self, turns, expected):
        mon = _minimal_mon(activeTurns=turns)
        vec = _volatile_counters(mon)
        assert vec[2].item() == pytest.approx(expected)

    def test_substitute_and_stall_simultaneously(self):
        mon = _minimal_mon(
            volatileDetails={"substitute": {"hp": 50}, "stall": {"counter": 3}},
            activeTurns=4,
        )
        vec = _volatile_counters(mon)
        assert vec[0].item() == pytest.approx(50 / 200)
        assert vec[1].item() == pytest.approx(3 / 6)
        assert vec[2].item() == pytest.approx(4 / 20)


class TestVolatilesDoNotCrashEncoder:
    """Volatiles the engine emits but the encoder ignores must not crash encoding."""

    @pytest.mark.parametrize("volatile_name,detail", [
        ("encore", {"duration": 3}),
        ("perishsong", {"duration": 2}),
        ("protect", {"duration": 1}),
        ("taunt", {"duration": 3}),
        ("disable", {"duration": 4}),
        ("confusion", {"time": 3}),
        ("leechseed", {}),
        ("yawn", {"time": 1}),
        ("flinch", {}),
        ("torment", {}),
        ("attract", {}),
        ("imprison", {}),
        ("healblock", {"duration": 5}),
        ("embargo", {"duration": 5}),
    ])
    def test_single_volatile_does_not_crash(self, volatile_name, detail):
        mon = _minimal_mon(
            volatiles=[volatile_name],
            volatileDetails={volatile_name: detail},
        )
        feats = _encode_pokemon_features(mon)
        assert feats.shape == (ENTITY_FEATURE_DIM,)

    def test_many_volatiles_simultaneously(self):
        details = {
            "substitute": {"hp": 80},
            "stall": {"counter": 2},
            "encore": {"duration": 3},
            "taunt": {"duration": 2},
            "perishsong": {"duration": 1},
            "confusion": {"time": 2},
        }
        mon = _minimal_mon(
            volatiles=list(details.keys()),
            volatileDetails=details,
            activeTurns=5,
        )
        feats = _encode_pokemon_features(mon)
        assert feats.shape == (ENTITY_FEATURE_DIM,)
        # Verify the features the encoder actually extracts are correct
        vec = _volatile_counters(mon)
        assert vec[0].item() == pytest.approx(80 / 200)
        assert vec[1].item() == pytest.approx(2 / 6)
        assert vec[2].item() == pytest.approx(5 / 20)


# ---------------------------------------------------------------------------
# Slot flags — every positional state
# ---------------------------------------------------------------------------


class TestSlotFlagsExhaustive:
    """Every distinct positional state a pokemon can occupy in doubles."""

    def test_active_left_mine(self):
        vec = _slot_flags(_minimal_mon(active=True, position=0, fainted=False), is_opponent=False)
        assert vec[0].item() == 1.0   # is_active
        assert vec[1].item() == 0.0   # is_bench
        assert vec[2].item() == 0.0   # is_fainted
        assert vec[4].item() == 1.0   # pos_left
        assert vec[5].item() == 0.0   # pos_right
        assert vec[6].item() == 0.0   # pos_bench
        assert vec[7].item() == 0.0   # side_flag (mine)

    def test_active_right_mine(self):
        vec = _slot_flags(_minimal_mon(active=True, position=1, fainted=False), is_opponent=False)
        assert vec[0].item() == 1.0
        assert vec[4].item() == 0.0   # pos_left
        assert vec[5].item() == 1.0   # pos_right
        assert vec[6].item() == 0.0   # pos_bench

    def test_bench_mon(self):
        vec = _slot_flags(_minimal_mon(active=False, position=2, fainted=False), is_opponent=False)
        assert vec[0].item() == 0.0   # not active
        assert vec[1].item() == 1.0   # is_bench
        assert vec[2].item() == 0.0   # not fainted
        assert vec[4].item() == 0.0
        assert vec[5].item() == 0.0
        assert vec[6].item() == 1.0   # pos_bench

    def test_fainted_mon(self):
        vec = _slot_flags(_minimal_mon(active=False, position=0, fainted=True), is_opponent=False)
        assert vec[0].item() == 0.0   # not active
        assert vec[1].item() == 0.0   # not bench (fainted overrides)
        assert vec[2].item() == 1.0   # is_fainted
        assert vec[6].item() == 1.0   # pos_bench (not in active slot)

    def test_item_consumed(self):
        vec = _slot_flags(
            _minimal_mon(item=None, lastItem="sitrusberry", active=True, position=0),
            is_opponent=False,
        )
        assert vec[3].item() == 1.0   # item_consumed

    def test_item_never_held(self):
        vec = _slot_flags(
            _minimal_mon(item=None, lastItem=None),
            is_opponent=False,
        )
        assert vec[3].item() == 0.0   # not consumed (never had one)

    def test_item_still_held(self):
        vec = _slot_flags(
            _minimal_mon(item="choicescarf", lastItem=None),
            is_opponent=False,
        )
        assert vec[3].item() == 0.0   # still holding it

    def test_opponent_flag(self):
        vec_mine = _slot_flags(_minimal_mon(active=True, position=0), is_opponent=False)
        vec_opp = _slot_flags(_minimal_mon(active=True, position=0), is_opponent=True)
        assert vec_mine[7].item() == 0.0
        assert vec_opp[7].item() == 1.0

    def test_fainted_at_position_zero(self):
        """Fainted mon with position=0 — should not get active-left flag."""
        vec = _slot_flags(_minimal_mon(active=False, position=0, fainted=True), is_opponent=False)
        assert vec[0].item() == 0.0   # not active
        assert vec[2].item() == 1.0   # is_fainted
        assert vec[4].item() == 0.0   # NOT active-left (fainted overrides position)


# ---------------------------------------------------------------------------
# Field conditions — weather, terrain, pseudo-weather (EXHAUSTIVE)
# ---------------------------------------------------------------------------


class TestWeatherExhaustive:
    """Every weather condition with correct index and duration encoding."""

    @pytest.mark.parametrize("weather,expected_idx", [
        ("raindance", 0),
        ("sunnyday", 1),
        ("sandstorm", 2),
        ("snow", 3),
        ("hail", 3),  # alias for snow
    ])
    def test_each_weather(self, weather, expected_idx):
        vec = _weather_onehot_dur(_minimal_field(weather=weather, weatherDuration=5))
        assert vec[expected_idx].item() == 1.0
        assert vec.sum().item() == pytest.approx(1.0 + 5 / 20)

    def test_no_weather(self):
        vec = _weather_onehot_dur(_minimal_field())
        assert (vec == 0).all()

    def test_unknown_weather_no_hot_bit(self):
        vec = _weather_onehot_dur(_minimal_field(weather="deltastream", weatherDuration=5))
        # No hot bit for unknown weather, but duration still encoded
        assert vec[:4].sum().item() == 0.0
        assert vec[4].item() == pytest.approx(5 / 20)

    def test_duration_zero(self):
        vec = _weather_onehot_dur(_minimal_field(weather="raindance", weatherDuration=0))
        assert vec[0].item() == 1.0
        assert vec[4].item() == 0.0

    def test_duration_none_permanent(self):
        vec = _weather_onehot_dur(_minimal_field(weather="raindance", weatherDuration=None))
        assert vec[0].item() == 1.0
        assert vec[4].item() == 0.0  # None treated as 0


class TestTerrainExhaustive:
    """Every terrain condition with correct index and duration encoding."""

    @pytest.mark.parametrize("terrain,expected_idx", [
        ("electricterrain", 0),
        ("grassyterrain", 1),
        ("mistyterrain", 2),
        ("psychicterrain", 3),
    ])
    def test_each_terrain(self, terrain, expected_idx):
        vec = _terrain_onehot_dur(_minimal_field(terrain=terrain, terrainDuration=4))
        assert vec[expected_idx].item() == 1.0
        assert vec[4].item() == pytest.approx(4 / 20)

    def test_no_terrain(self):
        vec = _terrain_onehot_dur(_minimal_field())
        assert (vec == 0).all()

    def test_unknown_terrain_no_hot_bit(self):
        vec = _terrain_onehot_dur(_minimal_field(terrain="weirdterrain", terrainDuration=3))
        assert vec[:4].sum().item() == 0.0
        assert vec[4].item() == pytest.approx(3 / 20)


class TestTrickRoomExhaustive:
    """Trick Room active flag + duration."""

    @pytest.mark.parametrize("duration,expected_dur", [
        (5, 5 / 20),
        (3, 3 / 20),
        (1, 1 / 20),
        (0, 0.0),
    ])
    def test_active_with_duration(self, duration, expected_dur):
        field = _minimal_field(pseudoWeather={"trickroom": PseudoWeatherEntry(duration=duration)})
        vec = _trick_room(field)
        assert vec[0].item() == 1.0
        assert vec[1].item() == pytest.approx(expected_dur)

    def test_trick_room_duration_none(self):
        field = _minimal_field(pseudoWeather={"trickroom": PseudoWeatherEntry(duration=None)})
        vec = _trick_room(field)
        assert vec[0].item() == 1.0
        assert vec[1].item() == 0.0


class TestGravityExhaustive:
    """Gravity active flag."""

    def test_gravity_active(self):
        field = _minimal_field(pseudoWeather={"gravity": PseudoWeatherEntry(duration=5)})
        vec = _gravity(field)
        assert vec[0].item() == 1.0

    def test_gravity_inactive(self):
        vec = _gravity(_minimal_field())
        assert vec[0].item() == 0.0


class TestPseudoWeatherCombinations:
    """Multiple pseudo-weathers simultaneously."""

    def test_trick_room_and_gravity_both_active(self):
        field = _minimal_field(pseudoWeather={
            "trickroom": PseudoWeatherEntry(duration=3),
            "gravity": PseudoWeatherEntry(duration=5),
        })
        tr_vec = _trick_room(field)
        g_vec = _gravity(field)
        assert tr_vec[0].item() == 1.0
        assert g_vec[0].item() == 1.0

    def test_unknown_pseudo_weather_does_not_affect_encoded_ones(self):
        field = _minimal_field(pseudoWeather={
            "trickroom": PseudoWeatherEntry(duration=3),
            "magicroom": PseudoWeatherEntry(duration=5),
        })
        tr_vec = _trick_room(field)
        g_vec = _gravity(field)
        assert tr_vec[0].item() == 1.0
        assert g_vec[0].item() == 0.0  # gravity not present


class TestFieldComposite:
    """Full field encoding with everything active simultaneously."""

    def test_all_field_conditions_active(self):
        field = _minimal_field(
            weather="sunnyday", weatherDuration=5,
            terrain="grassyterrain", terrainDuration=4,
            pseudoWeather={
                "trickroom": PseudoWeatherEntry(duration=3),
                "gravity": PseudoWeatherEntry(duration=2),
            },
        )
        feats = _encode_field(field)
        assert feats.shape == (FIELD_FEATURE_DIM,)
        # Weather: sun at index 1
        assert feats[1].item() == 1.0
        # Terrain: grassy at offset 5 + index 1 = 6
        assert feats[6].item() == 1.0
        # Trick room active flag at offset 10
        assert feats[10].item() == 1.0
        # Gravity active flag at offset 12
        assert feats[12].item() == 1.0


# ---------------------------------------------------------------------------
# Side conditions — screens, hazards, combinations
# ---------------------------------------------------------------------------


class TestSideScreensExhaustive:
    """Every screen tested individually with various durations."""

    @pytest.mark.parametrize("screen_name", [
        "tailwind", "reflect", "lightscreen", "auroraveil",
    ])
    def test_screen_active(self, screen_name):
        conds = {screen_name: SideConditionSnapshot(duration=4, layers=None)}
        vec = _side_screen(screen_name, conds)
        assert vec[0].item() == 1.0
        assert vec[1].item() == pytest.approx(4 / 20)

    @pytest.mark.parametrize("screen_name", [
        "tailwind", "reflect", "lightscreen", "auroraveil",
    ])
    def test_screen_absent(self, screen_name):
        vec = _side_screen(screen_name, {})
        assert vec[0].item() == 0.0
        assert vec[1].item() == 0.0

    @pytest.mark.parametrize("duration,expected", [
        (1, 1 / 20),
        (3, 3 / 20),
        (5, 5 / 20),
        (8, 8 / 20),
    ])
    def test_tailwind_various_durations(self, duration, expected):
        conds = {"tailwind": SideConditionSnapshot(duration=duration, layers=None)}
        vec = _side_screen("tailwind", conds)
        assert vec[1].item() == pytest.approx(expected)

    @pytest.mark.parametrize("duration", [1, 3, 5, 8])
    def test_reflect_various_durations(self, duration):
        conds = {"reflect": SideConditionSnapshot(duration=duration, layers=None)}
        vec = _side_screen("reflect", conds)
        assert vec[0].item() == 1.0
        assert vec[1].item() == pytest.approx(duration / 20)

    @pytest.mark.parametrize("duration", [1, 3, 5, 8])
    def test_light_screen_various_durations(self, duration):
        conds = {"lightscreen": SideConditionSnapshot(duration=duration, layers=None)}
        vec = _side_screen("lightscreen", conds)
        assert vec[0].item() == 1.0
        assert vec[1].item() == pytest.approx(duration / 20)

    @pytest.mark.parametrize("duration", [1, 3, 5, 8])
    def test_aurora_veil_various_durations(self, duration):
        conds = {"auroraveil": SideConditionSnapshot(duration=duration, layers=None)}
        vec = _side_screen("auroraveil", conds)
        assert vec[0].item() == 1.0
        assert vec[1].item() == pytest.approx(duration / 20)


class TestSideHazardsExhaustive:
    """Stealth Rock (binary) and Spikes (layers 1/2/3)."""

    def test_no_hazards(self):
        vec = _side_hazards({})
        assert vec[0].item() == 0.0
        assert vec[1].item() == 0.0

    def test_stealth_rock_present(self):
        conds = {"stealthrock": SideConditionSnapshot(duration=None, layers=None)}
        vec = _side_hazards(conds)
        assert vec[0].item() == 1.0

    def test_stealth_rock_absent(self):
        conds = {"spikes": SideConditionSnapshot(duration=None, layers=2)}
        vec = _side_hazards(conds)
        assert vec[0].item() == 0.0  # no stealth rock

    @pytest.mark.parametrize("layers,expected", [
        (1, 1 / 3),
        (2, 2 / 3),
        (3, 1.0),
    ])
    def test_spikes_layers(self, layers, expected):
        conds = {"spikes": SideConditionSnapshot(duration=None, layers=layers)}
        vec = _side_hazards(conds)
        assert vec[1].item() == pytest.approx(expected)

    def test_both_hazards(self):
        conds = {
            "stealthrock": SideConditionSnapshot(duration=None, layers=None),
            "spikes": SideConditionSnapshot(duration=None, layers=3),
        }
        vec = _side_hazards(conds)
        assert vec[0].item() == 1.0
        assert vec[1].item() == pytest.approx(1.0)


class TestSideConditionCombinations:
    """Multiple side conditions co-occurring."""

    def test_reflect_and_light_screen(self):
        """Both screens active simultaneously with different durations."""
        conds = {
            "reflect": SideConditionSnapshot(duration=3, layers=None),
            "lightscreen": SideConditionSnapshot(duration=5, layers=None),
        }
        side = _minimal_side(side_conditions=conds)
        feats = _encode_side(side)
        assert feats.shape == (SIDE_FEATURE_DIM,)
        # Tailwind absent (offset 0-1)
        assert feats[0].item() == 0.0
        # Reflect active (offset 2-3)
        assert feats[2].item() == 1.0
        assert feats[3].item() == pytest.approx(3 / 20)
        # Light Screen active (offset 4-5)
        assert feats[4].item() == 1.0
        assert feats[5].item() == pytest.approx(5 / 20)

    def test_aurora_veil_and_reflect(self):
        """Aurora Veil + Reflect both present (edge case)."""
        conds = {
            "auroraveil": SideConditionSnapshot(duration=5, layers=None),
            "reflect": SideConditionSnapshot(duration=3, layers=None),
        }
        side = _minimal_side(side_conditions=conds)
        feats = _encode_side(side)
        # Reflect at offset 2-3
        assert feats[2].item() == 1.0
        # Aurora Veil at offset 6-7
        assert feats[6].item() == 1.0
        assert feats[7].item() == pytest.approx(5 / 20)

    def test_all_screens_plus_all_hazards(self):
        """Kitchen sink: every side condition active."""
        conds = {
            "tailwind": SideConditionSnapshot(duration=4, layers=None),
            "reflect": SideConditionSnapshot(duration=5, layers=None),
            "lightscreen": SideConditionSnapshot(duration=3, layers=None),
            "auroraveil": SideConditionSnapshot(duration=5, layers=None),
            "stealthrock": SideConditionSnapshot(duration=None, layers=None),
            "spikes": SideConditionSnapshot(duration=None, layers=2),
        }
        side = _minimal_side(side_conditions=conds)
        feats = _encode_side(side)
        assert feats.shape == (SIDE_FEATURE_DIM,)
        # Every active flag should be 1.0
        assert feats[0].item() == 1.0   # tailwind active
        assert feats[2].item() == 1.0   # reflect active
        assert feats[4].item() == 1.0   # lightscreen active
        assert feats[6].item() == 1.0   # aurora veil active
        assert feats[8].item() == 1.0   # stealth rock
        assert feats[9].item() == pytest.approx(2 / 3)  # spikes 2 layers

    def test_unknown_side_condition_does_not_crash(self):
        """Engine-emitted conditions not in our encoder (toxic spikes, sticky web)."""
        conds = {
            "toxicspikes": SideConditionSnapshot(duration=None, layers=1),
            "stickyweb": SideConditionSnapshot(duration=None, layers=1),
            "tailwind": SideConditionSnapshot(duration=4, layers=None),
        }
        side = _minimal_side(side_conditions=conds)
        feats = _encode_side(side)
        assert feats.shape == (SIDE_FEATURE_DIM,)
        # Tailwind still encodes correctly despite extra conditions
        assert feats[0].item() == 1.0


# ---------------------------------------------------------------------------
# Nature — all 25 + edge cases
# ---------------------------------------------------------------------------


class TestNatureOnehotExhaustive:
    """Every nature produces exactly one hot bit; unknowns produce zeros."""

    @pytest.mark.parametrize("nature", [
        "Adamant", "Bashful", "Bold", "Brave", "Calm", "Careful", "Docile",
        "Gentle", "Hardy", "Hasty", "Impish", "Jolly", "Lax", "Lonely",
        "Mild", "Modest", "Naive", "Naughty", "Quiet", "Quirky", "Rash",
        "Relaxed", "Sassy", "Serious", "Timid",
    ])
    def test_known_nature_one_hot(self, nature):
        vec = _nature_onehot(_minimal_mon(nature=nature))
        assert vec.sum().item() == pytest.approx(1.0)
        assert vec.shape == (NUM_NATURES,)

    def test_unknown_nature_all_zeros(self):
        vec = _nature_onehot(_minimal_mon(nature="MadeUpNature"))
        assert vec.sum().item() == pytest.approx(0.0)

    def test_none_nature_all_zeros(self):
        vec = _nature_onehot(_minimal_mon(nature=None))
        assert vec.sum().item() == pytest.approx(0.0)

    def test_empty_string_nature_all_zeros(self):
        vec = _nature_onehot(_minimal_mon(nature=""))
        assert vec.sum().item() == pytest.approx(0.0)

    def test_different_natures_produce_different_indices(self):
        vec_adamant = _nature_onehot(_minimal_mon(nature="Adamant"))
        vec_timid = _nature_onehot(_minimal_mon(nature="Timid"))
        assert vec_adamant.argmax().item() != vec_timid.argmax().item()


# ---------------------------------------------------------------------------
# Move PP flags — all variations
# ---------------------------------------------------------------------------


class TestMovePpFlagsExhaustive:
    """Every move-state variation: full PP, partial, zero, disabled, missing."""

    def test_four_moves_full_pp(self):
        mon = _minimal_mon(moves=[
            {"id": "heatwave", "pp": 10, "maxpp": 10, "disabled": False},
            {"id": "protect", "pp": 16, "maxpp": 16, "disabled": False},
            {"id": "airslash", "pp": 15, "maxpp": 15, "disabled": False},
            {"id": "solarbeam", "pp": 10, "maxpp": 10, "disabled": False},
        ])
        vec = _move_pp_flags(mon)
        for i in range(4):
            assert vec[i * 2].item() == pytest.approx(1.0)    # pp_fraction
            assert vec[i * 2 + 1].item() == 0.0              # not disabled

    def test_partial_pp(self):
        mon = _minimal_mon(moves=[
            {"id": "heatwave", "pp": 3, "maxpp": 10, "disabled": False},
            {"id": "protect", "pp": 8, "maxpp": 16, "disabled": False},
            {"id": "airslash", "pp": 0, "maxpp": 15, "disabled": False},
            {"id": "solarbeam", "pp": 10, "maxpp": 10, "disabled": False},
        ])
        vec = _move_pp_flags(mon)
        assert vec[0].item() == pytest.approx(3 / 10)
        assert vec[2].item() == pytest.approx(8 / 16)
        assert vec[4].item() == pytest.approx(0.0)   # 0 PP
        assert vec[6].item() == pytest.approx(1.0)

    def test_all_disabled(self):
        mon = _minimal_mon(moves=[
            {"id": "heatwave", "pp": 10, "maxpp": 10, "disabled": True},
            {"id": "protect", "pp": 16, "maxpp": 16, "disabled": True},
            {"id": "airslash", "pp": 15, "maxpp": 15, "disabled": True},
            {"id": "solarbeam", "pp": 10, "maxpp": 10, "disabled": True},
        ])
        vec = _move_pp_flags(mon)
        for i in range(4):
            assert vec[i * 2 + 1].item() == 1.0  # all disabled

    def test_two_moves_only(self):
        mon = _minimal_mon(moves=[
            {"id": "heatwave", "pp": 10, "maxpp": 10, "disabled": False},
            {"id": "protect", "pp": 16, "maxpp": 16, "disabled": False},
        ])
        vec = _move_pp_flags(mon)
        assert vec[0].item() == pytest.approx(1.0)
        assert vec[2].item() == pytest.approx(1.0)
        # Slots 2-3 are zero
        assert vec[4].item() == 0.0
        assert vec[5].item() == 0.0
        assert vec[6].item() == 0.0
        assert vec[7].item() == 0.0

    def test_one_move_only(self):
        mon = _minimal_mon(moves=[
            {"id": "struggle", "pp": 1, "maxpp": 1, "disabled": False},
        ])
        vec = _move_pp_flags(mon)
        assert vec[0].item() == pytest.approx(1.0)
        for i in range(1, 4):
            assert vec[i * 2].item() == 0.0
            assert vec[i * 2 + 1].item() == 0.0

    def test_zero_moves(self):
        mon = _minimal_mon(moves=[])
        vec = _move_pp_flags(mon)
        assert (vec == 0).all()

    def test_maxpp_zero_no_division_error(self):
        """maxpp=0 edge case — should not crash (uses max(maxpp, 1))."""
        mon = _minimal_mon(moves=[
            {"id": "heatwave", "pp": 0, "maxpp": 0, "disabled": False},
            {"id": "protect", "pp": 5, "maxpp": 16, "disabled": False},
            {"id": "airslash", "pp": 15, "maxpp": 15, "disabled": False},
            {"id": "solarbeam", "pp": 10, "maxpp": 10, "disabled": False},
        ])
        vec = _move_pp_flags(mon)
        assert vec[0].item() == pytest.approx(0.0)  # 0 / max(0,1) = 0


# ---------------------------------------------------------------------------
# Boost stages — boundary values
# ---------------------------------------------------------------------------


class TestBoostStagesExhaustive:
    """Stat stages at boundaries and mixed values."""

    def test_all_zeros(self):
        vec = _boost_stages(_minimal_mon(boosts={"atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0, "accuracy": 0, "evasion": 0}))
        assert (vec == 0).all()

    def test_all_max_positive(self):
        vec = _boost_stages(_minimal_mon(boosts={"atk": 6, "def": 6, "spa": 6, "spd": 6, "spe": 6, "accuracy": 6, "evasion": 6}))
        for i in range(7):
            assert vec[i].item() == pytest.approx(1.0)

    def test_all_max_negative(self):
        vec = _boost_stages(_minimal_mon(boosts={"atk": -6, "def": -6, "spa": -6, "spd": -6, "spe": -6, "accuracy": -6, "evasion": -6}))
        for i in range(7):
            assert vec[i].item() == pytest.approx(-1.0)

    def test_mixed_boosts(self):
        vec = _boost_stages(_minimal_mon(boosts={"atk": 6, "def": -6, "spa": 3, "spd": -2, "spe": 0, "accuracy": 1, "evasion": -1}))
        assert vec[0].item() == pytest.approx(1.0)
        assert vec[1].item() == pytest.approx(-1.0)
        assert vec[2].item() == pytest.approx(0.5)
        assert vec[3].item() == pytest.approx(-2 / 6)
        assert vec[4].item() == pytest.approx(0.0)
        assert vec[5].item() == pytest.approx(1 / 6)
        assert vec[6].item() == pytest.approx(-1 / 6)

    def test_missing_boost_keys(self):
        """Partial dict — missing keys should default to 0."""
        vec = _boost_stages(_minimal_mon(boosts={"atk": 2}))
        assert vec[0].item() == pytest.approx(2 / 6)
        assert vec[1].item() == pytest.approx(0.0)  # def missing


# ---------------------------------------------------------------------------
# Full encode() — phase variations and perspective
# ---------------------------------------------------------------------------


class TestEncodePhaseVariations:
    """Test encode() with every phase and edge case."""

    def test_terminal_phase_onehot(self):
        view = _minimal_view(phase="terminal")
        obs = encode(view, perspective="p1")
        # Phase one-hot is in scalars: turn(1) + phase(4) + whose_decision(2)
        scalars = obs["scalars"]
        assert scalars[4].item() == 1.0  # terminal is index 3 in phase, offset by turn(1)

    def test_force_switch_phase_onehot(self):
        view = _minimal_view(phase="forceSwitch")
        obs = encode(view, perspective="p1")
        scalars = obs["scalars"]
        assert scalars[3].item() == 1.0  # forceSwitch is index 2

    def test_perspective_not_in_legal_produces_empty_mask(self):
        view = _minimal_view(phase="move")
        # Remove p1 from legal
        view.legal = {"p2": {"active": [], "side": {"pokemon": []}}}
        obs = encode(view, perspective="p1")
        assert obs["action_mask"].sum().item() == 0

    def test_entity_feature_dim_value(self):
        """Sentinel-derived ENTITY_FEATURE_DIM matches expected sum of sub-encoder widths."""
        # 1(hp) + 6(stats) + 7(boosts) + 7(status) + 25(nature) + 8(moves) + 3(volatile) + 8(flags) = 65
        assert ENTITY_FEATURE_DIM == 65

    def test_field_feature_dim_value(self):
        """FIELD_FEATURE_DIM = 5(weather) + 5(terrain) + 2(trick_room) + 1(gravity) = 13."""
        assert FIELD_FEATURE_DIM == 13

    def test_side_feature_dim_value(self):
        """SIDE_FEATURE_DIM = 2*4(screens) + 2(hazards) + 1(mega) = 11."""
        assert SIDE_FEATURE_DIM == 11

    def test_scalar_feature_dim_value(self):
        """SCALAR_FEATURE_DIM = 1(turn) + 4(phase) + 2(whose) = 7."""
        assert SCALAR_FEATURE_DIM == 7

    def test_encode_with_empty_legal_dict(self):
        view = _minimal_view(phase="move")
        view.legal = {}
        obs = encode(view, perspective="p1")
        assert obs["action_mask"].sum().item() == 0


# ---------------------------------------------------------------------------
# Kitchen sink — everything active at once
# ---------------------------------------------------------------------------


class TestKitchenSink:
    """Full encode with every feature active simultaneously."""

    def test_full_state_encodes_without_error(self):
        """A pokemon with status + volatiles + boosts + partial PP on a field
        with weather + terrain + trick room + gravity + screens + hazards."""
        damaged_mon = _minimal_mon(
            hp=50, maxhp=200,
            status="tox",
            statusState={"stage": 3, "time": None},
            boosts={"atk": 2, "def": -1, "spa": 0, "spd": 0, "spe": 4, "accuracy": 0, "evasion": -2},
            active=True, position=0, fainted=False,
            item=None, lastItem="sitrusberry",
            activeTurns=5,
            volatiles=["substitute", "stall", "taunt"],
            volatileDetails={
                "substitute": {"hp": 80},
                "stall": {"counter": 2},
                "taunt": {"duration": 3},
            },
            moves=[
                {"id": "heatwave", "pp": 3, "maxpp": 10, "disabled": False},
                {"id": "protect", "pp": 0, "maxpp": 16, "disabled": True},
                {"id": "airslash", "pp": 15, "maxpp": 15, "disabled": False},
                {"id": "solarbeam", "pp": 10, "maxpp": 10, "disabled": False},
            ],
        )
        view = _minimal_view(
            my_pokemon=[damaged_mon] + [_minimal_mon() for _ in range(5)],
            field=_minimal_field(
                weather="sunnyday", weatherDuration=5,
                terrain="grassyterrain", terrainDuration=4,
                pseudoWeather={
                    "trickroom": PseudoWeatherEntry(duration=3),
                    "gravity": PseudoWeatherEntry(duration=2),
                },
            ),
            my_side_conds={
                "tailwind": SideConditionSnapshot(duration=4, layers=None),
                "reflect": SideConditionSnapshot(duration=5, layers=None),
                "lightscreen": SideConditionSnapshot(duration=3, layers=None),
                "stealthrock": SideConditionSnapshot(duration=None, layers=None),
                "spikes": SideConditionSnapshot(duration=None, layers=2),
            },
            phase="move",
            turn=8,
            to_move=["p1", "p2"],
        )
        obs = encode(view, perspective="p1")

        # Basic shape checks
        assert obs["entities"].shape == (12, ENTITY_FEATURE_DIM)
        assert obs["field"].shape == (FIELD_FEATURE_DIM,)
        assert obs["sides"].shape == (2, SIDE_FEATURE_DIM)
        assert obs["scalars"].shape == (SCALAR_FEATURE_DIM,)

        # Verify the damaged mon's HP fraction
        assert obs["entities"][0, 0].item() == pytest.approx(50 / 200)
