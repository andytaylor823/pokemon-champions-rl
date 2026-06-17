"""Unit tests for the encoder module — feature encoding helpers and full encode().

Fixtures are derived from a real worker snapshot (tests/fixtures/real_snapshot.json)
so their shape can never drift from the actual wire format. The _minimal_mon factory
loads and mutates the ground-truth JSON rather than hand-building dicts.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from action_space import A
from encoder import (
    ENTITY_FEATURE_DIM,
    FIELD_FEATURE_DIM,
    NUM_BOOSTS,
    NUM_NATURES,
    NUM_STATS,
    NUM_STATUS,
    SCALAR_FEATURE_DIM,
    SIDE_FEATURE_DIM,
    _encode_field,
    _encode_move_ids,
    _encode_pokemon_features,
    _encode_scalars,
    _encode_side,
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
# _encode_pokemon_features
# ---------------------------------------------------------------------------


class TestEncodePokemonFeatures:
    """Test per-pokemon feature encoding."""

    def test_output_shape(self):
        mon = _minimal_mon()
        feats = _encode_pokemon_features(mon)
        assert feats.shape == (ENTITY_FEATURE_DIM,)

    def test_hp_fraction(self):
        mon = _minimal_mon(hp=50, maxhp=100)
        feats = _encode_pokemon_features(mon)
        assert feats[0].item() == pytest.approx(0.5)

    def test_hp_fraction_full(self):
        mon = _minimal_mon(hp=200, maxhp=200)
        feats = _encode_pokemon_features(mon)
        assert feats[0].item() == pytest.approx(1.0)

    def test_hp_fraction_zero(self):
        mon = _minimal_mon(hp=0, maxhp=200)
        feats = _encode_pokemon_features(mon)
        assert feats[0].item() == pytest.approx(0.0)

    def test_stats_normalized(self):
        mon = _minimal_mon(stats={"hp": 200, "atk": 120, "def": 90, "spa": 150, "spd": 100, "spe": 130})
        feats = _encode_pokemon_features(mon)
        # hp stat = 200, normalized by MAX_STAT=200 -> 1.0
        assert feats[1].item() == pytest.approx(1.0)

    def test_boosts_normalized(self):
        mon = _minimal_mon(boosts={"atk": 6, "def": -6, "spa": 0, "spd": 0, "spe": 0, "accuracy": 0, "evasion": 0})
        feats = _encode_pokemon_features(mon)
        boost_start = 1 + NUM_STATS
        assert feats[boost_start].item() == pytest.approx(1.0)      # atk +6 / 6
        assert feats[boost_start + 1].item() == pytest.approx(-1.0)  # def -6 / 6

    def test_status_burn_onehot(self):
        mon = _minimal_mon(status="brn")
        feats = _encode_pokemon_features(mon)
        status_start = 1 + NUM_STATS + NUM_BOOSTS
        assert feats[status_start].item() == 1.0
        assert feats[status_start + NUM_STATUS - 1].item() == 0.0

    def test_status_none_onehot(self):
        mon = _minimal_mon(status=None)
        feats = _encode_pokemon_features(mon)
        status_start = 1 + NUM_STATS + NUM_BOOSTS
        # "none" is the last status slot
        assert feats[status_start + NUM_STATUS - 1].item() == 1.0
        for i in range(NUM_STATUS - 1):
            assert feats[status_start + i].item() == 0.0

    def test_nature_onehot(self):
        mon = _minimal_mon(nature="Adamant")
        feats = _encode_pokemon_features(mon)
        nature_start = 1 + NUM_STATS + NUM_BOOSTS + NUM_STATUS
        nature_slice = feats[nature_start : nature_start + NUM_NATURES]
        assert nature_slice.sum().item() == pytest.approx(1.0)

    def test_nature_from_real_fixture(self):
        """Nature from the real fixture should produce exactly one hot bit."""
        mon = _minimal_mon()  # uses real fixture's nature (e.g. "Timid")
        feats = _encode_pokemon_features(mon)
        nature_start = 1 + NUM_STATS + NUM_BOOSTS + NUM_STATUS
        nature_slice = feats[nature_start : nature_start + NUM_NATURES]
        assert nature_slice.sum().item() == pytest.approx(1.0)

    def test_position_active_left(self):
        """Engine position 0 + active=True -> active-left one-hot slot."""
        mon = _minimal_mon(active=True, position=0, fainted=False)
        feats = _encode_pokemon_features(mon)
        # Physical slot starts at: 1+6+7+7+25+8+3 + 4 flags = idx for slot one-hot
        # is_active(1) + is_bench(1) + is_fainted(1) + item_consumed(1) = 4 flags before slot
        flags_start = 1 + NUM_STATS + NUM_BOOSTS + NUM_STATUS + NUM_NATURES + 8 + 3
        slot_start = flags_start + 4
        assert feats[slot_start].item() == 1.0      # active-left
        assert feats[slot_start + 1].item() == 0.0  # active-right
        assert feats[slot_start + 2].item() == 0.0  # bench

    def test_position_active_right(self):
        """Engine position 1 + active=True -> active-right one-hot slot."""
        mon = _minimal_mon(active=True, position=1, fainted=False)
        feats = _encode_pokemon_features(mon)
        flags_start = 1 + NUM_STATS + NUM_BOOSTS + NUM_STATUS + NUM_NATURES + 8 + 3
        slot_start = flags_start + 4
        assert feats[slot_start].item() == 0.0      # active-left
        assert feats[slot_start + 1].item() == 1.0  # active-right
        assert feats[slot_start + 2].item() == 0.0  # bench

    def test_position_bench(self):
        """Inactive mon -> bench one-hot slot."""
        mon = _minimal_mon(active=False, position=2, fainted=False)
        feats = _encode_pokemon_features(mon)
        flags_start = 1 + NUM_STATS + NUM_BOOSTS + NUM_STATUS + NUM_NATURES + 8 + 3
        slot_start = flags_start + 4
        assert feats[slot_start].item() == 0.0      # active-left
        assert feats[slot_start + 1].item() == 0.0  # active-right
        assert feats[slot_start + 2].item() == 1.0  # bench

    def test_opponent_side_flag(self):
        mon = _minimal_mon()
        feats_mine = _encode_pokemon_features(mon, is_opponent=False)
        feats_opp = _encode_pokemon_features(mon, is_opponent=True)
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
# _encode_field
# ---------------------------------------------------------------------------


class TestEncodeField:
    """Test field feature encoding."""

    def test_output_shape(self):
        feats = _encode_field(_minimal_field())
        assert feats.shape == (FIELD_FEATURE_DIM,)

    def test_empty_field_all_zeros(self):
        feats = _encode_field(_minimal_field())
        assert (feats == 0).all()

    def test_rain_weather(self):
        # Engine emits lowercase status IDs
        feats = _encode_field(_minimal_field(weather="raindance", weatherDuration=5))
        assert feats[0].item() == 1.0
        assert feats[5].item() == pytest.approx(5 / 20)

    def test_sun_weather(self):
        feats = _encode_field(_minimal_field(weather="sunnyday", weatherDuration=3))
        assert feats[1].item() == 1.0

    def test_terrain_electric(self):
        feats = _encode_field(_minimal_field(terrain="electricterrain", terrainDuration=4))
        terrain_start = 6
        assert feats[terrain_start].item() == 1.0

    def test_trick_room(self):
        field = _minimal_field(pseudoWeather={"trickroom": PseudoWeatherEntry(duration=3)})
        feats = _encode_field(field)
        tr_start = 12
        assert feats[tr_start].item() == 1.0
        assert feats[tr_start + 1].item() == pytest.approx(3 / 20)


# ---------------------------------------------------------------------------
# _encode_side
# ---------------------------------------------------------------------------


class TestEncodeSide:
    """Test per-side feature encoding."""

    def test_output_shape(self):
        feats = _encode_side(_minimal_side())
        assert feats.shape == (SIDE_FEATURE_DIM,)

    def test_empty_conditions_all_zeros(self):
        feats = _encode_side(_minimal_side())
        assert (feats == 0).all()

    def test_tailwind(self):
        conds = {"tailwind": SideConditionSnapshot(duration=4, layers=None)}
        feats = _encode_side(_minimal_side(side_conditions=conds))
        assert feats[0].item() == 1.0
        assert feats[1].item() == pytest.approx(4 / 20)

    def test_reflect(self):
        conds = {"reflect": SideConditionSnapshot(duration=3, layers=None)}
        feats = _encode_side(_minimal_side(side_conditions=conds))
        assert feats[2].item() == 1.0
        assert feats[3].item() == pytest.approx(3 / 20)

    def test_light_screen(self):
        conds = {"lightscreen": SideConditionSnapshot(duration=5, layers=None)}
        feats = _encode_side(_minimal_side(side_conditions=conds))
        assert feats[4].item() == 1.0
        assert feats[5].item() == pytest.approx(5 / 20)

    def test_stealth_rock(self):
        conds = {"stealthrock": SideConditionSnapshot(duration=None, layers=None)}
        feats = _encode_side(_minimal_side(side_conditions=conds))
        assert feats[8].item() == 1.0

    def test_spikes_layers(self):
        """Spikes reads .layers (not .duration) — the bug this whole fix addresses."""
        conds = {"spikes": SideConditionSnapshot(duration=None, layers=2)}
        feats = _encode_side(_minimal_side(side_conditions=conds))
        assert feats[9].item() == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# _encode_scalars
# ---------------------------------------------------------------------------


class TestEncodeScalars:
    """Test global scalar encoding."""

    def test_output_shape(self):
        view = _minimal_view(turn=5)
        feats = _encode_scalars(view.snapshot, view, "p1")
        assert feats.shape == (SCALAR_FEATURE_DIM,)

    def test_turn_normalization(self):
        view = _minimal_view(turn=10)
        feats = _encode_scalars(view.snapshot, view, "p1")
        assert feats[0].item() == pytest.approx(10 / 20)

    def test_phase_onehot_move(self):
        view = _minimal_view(phase="move")
        feats = _encode_scalars(view.snapshot, view, "p1")
        assert feats[1].item() == 0.0  # teamPreview
        assert feats[2].item() == 1.0  # move
        assert feats[3].item() == 0.0  # forceSwitch
        assert feats[4].item() == 0.0  # terminal

    def test_phase_onehot_team_preview(self):
        view = _minimal_view(phase="teamPreview")
        feats = _encode_scalars(view.snapshot, view, "p1")
        assert feats[1].item() == 1.0

    def test_whose_decision(self):
        view = _minimal_view(to_move=["p1", "p2"])
        feats = _encode_scalars(view.snapshot, view, "p1")
        assert feats[5].item() == 1.0
        assert feats[6].item() == 1.0

    def test_whose_decision_only_opponent(self):
        view = _minimal_view(to_move=["p2"])
        feats = _encode_scalars(view.snapshot, view, "p1")
        assert feats[5].item() == 0.0
        assert feats[6].item() == 1.0


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
