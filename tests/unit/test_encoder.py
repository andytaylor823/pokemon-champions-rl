"""Unit tests for the encoder module — feature encoding helpers and full encode()."""
from __future__ import annotations

import pytest
import torch

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
from action_space import A


# ---------------------------------------------------------------------------
# Helpers: minimal data factories
# ---------------------------------------------------------------------------


def _minimal_mon(**overrides) -> dict:
    """Create a minimal pokemon dict with sane defaults."""
    base = {
        "species": "Charizard",
        "ability": "Blaze",
        "item": "Charizardite Y",
        "hp": 100,
        "maxhp": 200,
        "stats": {"hp": 200, "atk": 120, "def": 90, "spa": 150, "spd": 100, "spe": 130},
        "boosts": {},
        "status": None,
        "nature": "Timid",
        "moves": [
            {"id": "heatwave", "pp": 10, "maxpp": 10, "disabled": False},
            {"id": "protect", "pp": 16, "maxpp": 16, "disabled": False},
            {"id": "airslash", "pp": 15, "maxpp": 15, "disabled": False},
            {"id": "solarbeam", "pp": 10, "maxpp": 10, "disabled": False},
        ],
        "volatileDetails": {},
        "active": True,
        "fainted": False,
        "position": 1,
        "activeTurns": 0,
        "lastItem": None,
    }
    base.update(overrides)
    return base


def _minimal_view(
    my_pokemon: list | None = None,
    opp_pokemon: list | None = None,
    field: dict | None = None,
    my_side_conds: dict | None = None,
    opp_side_conds: dict | None = None,
    phase: str = "move",
    turn: int = 1,
    to_move: list | None = None,
) -> dict:
    """Build a minimal StateView dict for testing encode()."""
    if my_pokemon is None:
        my_pokemon = [_minimal_mon() for _ in range(6)]
    if opp_pokemon is None:
        opp_pokemon = [_minimal_mon(species="Venusaur", ability="Chlorophyll") for _ in range(6)]
    if field is None:
        field = {}
    if to_move is None:
        to_move = ["p1"]

    return {
        "phase": phase,
        "terminal": False,
        "to_move": to_move,
        "snapshot": {
            "turn": turn,
            "sides": [
                {"id": "p1", "pokemon": my_pokemon, "sideConditions": my_side_conds or {}},
                {"id": "p2", "pokemon": opp_pokemon, "sideConditions": opp_side_conds or {}},
            ],
            "field": field,
        },
        "legal": {
            "p1": {"active": [], "side": {"pokemon": []}},
            "p2": {"active": [], "side": {"pokemon": []}},
        },
    }


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
        # HP fraction is the first feature
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
        mon = _minimal_mon()
        feats = _encode_pokemon_features(mon)
        # Stats start at index 1 (after hp_fraction)
        # hp stat = 200, normalized by MAX_STAT=200 → 1.0
        assert feats[1].item() == pytest.approx(1.0)

    def test_boosts_normalized(self):
        mon = _minimal_mon(boosts={"atk": 6, "def": -6})
        feats = _encode_pokemon_features(mon)
        # Boosts start at index 1 + NUM_STATS = 7
        boost_start = 1 + NUM_STATS
        assert feats[boost_start].item() == pytest.approx(1.0)   # atk boost +6 / 6
        assert feats[boost_start + 1].item() == pytest.approx(-1.0)  # def boost -6 / 6

    def test_status_burn_onehot(self):
        mon = _minimal_mon(status="brn")
        feats = _encode_pokemon_features(mon)
        status_start = 1 + NUM_STATS + NUM_BOOSTS
        # brn is index 0 in status map
        assert feats[status_start].item() == 1.0
        # "none" slot (last) should be 0
        assert feats[status_start + NUM_STATUS - 1].item() == 0.0

    def test_status_none_onehot(self):
        mon = _minimal_mon(status=None)
        feats = _encode_pokemon_features(mon)
        status_start = 1 + NUM_STATS + NUM_BOOSTS
        # "none" is the last status slot
        assert feats[status_start + NUM_STATUS - 1].item() == 1.0
        # All other status slots should be 0
        for i in range(NUM_STATUS - 1):
            assert feats[status_start + i].item() == 0.0

    def test_nature_onehot(self):
        mon = _minimal_mon(nature="Adamant")
        feats = _encode_pokemon_features(mon)
        nature_start = 1 + NUM_STATS + NUM_BOOSTS + NUM_STATUS
        # Exactly one slot should be 1.0
        nature_slice = feats[nature_start : nature_start + NUM_NATURES]
        assert nature_slice.sum().item() == pytest.approx(1.0)

    def test_opponent_side_flag(self):
        mon = _minimal_mon()
        # Mine: side_flag = 0
        feats_mine = _encode_pokemon_features(mon, is_opponent=False)
        # Opponent: side_flag = 1
        feats_opp = _encode_pokemon_features(mon, is_opponent=True)
        # Side flag is the last feature
        assert feats_mine[-1].item() == 0.0
        assert feats_opp[-1].item() == 1.0


# ---------------------------------------------------------------------------
# _encode_move_ids
# ---------------------------------------------------------------------------


class TestEncodeMoveIds:
    """Test move ID encoding."""

    def test_four_moves(self):
        moves = [{"id": "heatwave"}, {"id": "protect"}, {"id": "airslash"}, {"id": "solarbeam"}]
        ids = _encode_move_ids(moves)
        assert ids.shape == (4,)
        # Known moves should have non-zero IDs
        assert (ids > 0).all()

    def test_fewer_than_four_moves_padded_with_zero(self):
        moves = [{"id": "heatwave"}, {"id": "protect"}]
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
        feats = _encode_field({})
        assert feats.shape == (FIELD_FEATURE_DIM,)

    def test_empty_field_all_zeros(self):
        feats = _encode_field({})
        assert (feats == 0).all()

    def test_rain_weather(self):
        feats = _encode_field({"weather": "RainDance", "weatherDuration": 5})
        # Rain is index 0 in weather one-hot
        assert feats[0].item() == 1.0
        # Duration normalized by MAX_TURNS=20
        assert feats[5].item() == pytest.approx(5 / 20)

    def test_sun_weather(self):
        feats = _encode_field({"weather": "SunnyDay", "weatherDuration": 3})
        # Sun is index 1
        assert feats[1].item() == 1.0

    def test_terrain_electric(self):
        feats = _encode_field({"terrain": "electricterrain", "terrainDuration": 4})
        # Terrain one-hot starts after weather (5) + duration (1) = index 6
        terrain_start = 6
        assert feats[terrain_start].item() == 1.0

    def test_trick_room(self):
        feats = _encode_field({"pseudoWeather": {"trickroom": {"duration": 3}}})
        # TR starts after weather(5)+dur(1)+terrain(5)+dur(1) = index 12
        tr_start = 12
        assert feats[tr_start].item() == 1.0
        assert feats[tr_start + 1].item() == pytest.approx(3 / 20)


# ---------------------------------------------------------------------------
# _encode_side
# ---------------------------------------------------------------------------


class TestEncodeSide:
    """Test per-side feature encoding."""

    def test_output_shape(self):
        feats = _encode_side({"sideConditions": {}})
        assert feats.shape == (SIDE_FEATURE_DIM,)

    def test_empty_conditions_all_zeros(self):
        feats = _encode_side({"sideConditions": {}})
        assert (feats == 0).all()

    def test_tailwind(self):
        feats = _encode_side({"sideConditions": {"tailwind": 4}})
        assert feats[0].item() == 1.0  # tailwind active
        assert feats[1].item() == pytest.approx(4 / 20)  # tailwind duration

    def test_reflect(self):
        feats = _encode_side({"sideConditions": {"reflect": 3}})
        assert feats[2].item() == 1.0  # reflect active
        assert feats[3].item() == pytest.approx(3 / 20)

    def test_light_screen(self):
        feats = _encode_side({"sideConditions": {"lightscreen": 5}})
        assert feats[4].item() == 1.0
        assert feats[5].item() == pytest.approx(5 / 20)

    def test_stealth_rock(self):
        feats = _encode_side({"sideConditions": {"stealthrock": 1}})
        assert feats[8].item() == 1.0

    def test_spikes_normalized(self):
        feats = _encode_side({"sideConditions": {"spikes": 2}})
        assert feats[9].item() == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# _encode_scalars
# ---------------------------------------------------------------------------


class TestEncodeScalars:
    """Test global scalar encoding."""

    def test_output_shape(self):
        view = _minimal_view(turn=5)
        feats = _encode_scalars(view["snapshot"], view, "p1")
        assert feats.shape == (SCALAR_FEATURE_DIM,)

    def test_turn_normalization(self):
        view = _minimal_view(turn=10)
        feats = _encode_scalars(view["snapshot"], view, "p1")
        # Turn normalized by MAX_TURNS=20
        assert feats[0].item() == pytest.approx(10 / 20)

    def test_phase_onehot_move(self):
        view = _minimal_view(phase="move")
        feats = _encode_scalars(view["snapshot"], view, "p1")
        # Phase one-hot starts at index 1: move is index 1 in _PHASE_MAP
        assert feats[1].item() == 0.0  # teamPreview
        assert feats[2].item() == 1.0  # move
        assert feats[3].item() == 0.0  # forceSwitch
        assert feats[4].item() == 0.0  # terminal

    def test_phase_onehot_team_preview(self):
        view = _minimal_view(phase="teamPreview")
        feats = _encode_scalars(view["snapshot"], view, "p1")
        assert feats[1].item() == 1.0  # teamPreview

    def test_whose_decision(self):
        view = _minimal_view(to_move=["p1", "p2"])
        feats = _encode_scalars(view["snapshot"], view, "p1")
        # "am I acting" at index 5, "is opponent acting" at index 6
        assert feats[5].item() == 1.0
        assert feats[6].item() == 1.0

    def test_whose_decision_only_opponent(self):
        view = _minimal_view(to_move=["p2"])
        feats = _encode_scalars(view["snapshot"], view, "p1")
        assert feats[5].item() == 0.0  # p1 not acting
        assert feats[6].item() == 1.0  # p2 acting


# ---------------------------------------------------------------------------
# Full encode()
# ---------------------------------------------------------------------------


class TestEncode:
    """Test the top-level encode() function with synthetic views."""

    def test_output_shapes_phase1(self):
        view = _minimal_view()
        obs = encode(view, perspective="p1")
        # Phase 1: 6 my + 6 opp = 12 entity tokens
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
        # For p1: first 6 tokens have slot_id=0, last 6 have slot_id=1
        assert (obs_p1["slot_id"][:6] == 0).all()
        assert (obs_p1["slot_id"][6:] == 1).all()
        # For p2: swapped — p2's team is "my" so first 6 are slot_id=0
        assert (obs_p2["slot_id"][:6] == 0).all()
        assert (obs_p2["slot_id"][6:] == 1).all()

    def test_species_ids_encoded(self):
        view = _minimal_view()
        obs = encode(view, perspective="p1")
        # All tokens should have non-zero species IDs (Charizard and Venusaur)
        assert (obs["ids", "species"] > 0).all()

    def test_team_preview_mask_when_phase_is_team_preview(self):
        view = _minimal_view(phase="teamPreview")
        obs = encode(view, perspective="p1")
        # In teamPreview, exactly 360 entries should be True
        mask_sum = obs["action_mask"].sum().item()
        assert mask_sum == 360
