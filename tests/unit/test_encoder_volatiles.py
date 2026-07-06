"""Volatile encoder tests — exhaustive coverage of _volatile_counters sub-encoder.

Extracted from test_encoder.py to keep file sizes manageable. Tests the binary
volatile tuple, special counters (substitute, stall, perish song), and verifies
that unencoded volatiles don't crash the encoder.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from encoder import (
    BINARY_VOLATILE_OFFSET,
    BINARY_VOLATILES,
    ENTITY_FEATURE_DIM,
    _encode_pokemon_features,
    _volatile_counters,
)
from state_types import PokemonSnapshot

# Load the same real fixture used by test_encoder.py
_FIXTURE_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "real_snapshot.json"
with open(_FIXTURE_PATH, encoding="utf-8") as _f:
    _REAL = json.load(_f)
_REAL_MON_DICT = _REAL["team_preview"]["snapshot"]["sides"][0]["pokemon"][0]


def _minimal_mon(**overrides) -> PokemonSnapshot:
    """Create a PokemonSnapshot from the real fixture, with optional overrides."""
    data = {**_REAL_MON_DICT, **overrides}
    return PokemonSnapshot.model_validate(data)


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
        assert vec[3].item() == pytest.approx(0.0)

    def test_yawn_active(self):
        mon = _minimal_mon(
            volatiles=["yawn"],
            volatileDetails={"yawn": {"time": 1}},
        )
        vec = _volatile_counters(mon)
        assert vec[3].item() == pytest.approx(1.0)

    def test_yawn_absent(self):
        mon = _minimal_mon(volatiles=[], volatileDetails={})
        vec = _volatile_counters(mon)
        assert vec[3].item() == pytest.approx(0.0)

    def test_yawn_with_other_volatiles(self):
        mon = _minimal_mon(
            volatiles=["substitute", "stall", "yawn"],
            volatileDetails={"substitute": {"hp": 80}, "stall": {"counter": 2}, "yawn": {"time": 1}},
            activeTurns=3,
        )
        vec = _volatile_counters(mon)
        assert vec[0].item() == pytest.approx(80 / 200)
        assert vec[1].item() == pytest.approx(2 / 6)
        assert vec[2].item() == pytest.approx(3 / 20)
        assert vec[3].item() == pytest.approx(1.0)

    def test_flash_fire_active(self):
        mon = _minimal_mon(
            volatiles=["flashfire"],
            volatileDetails={"flashfire": {}},
        )
        vec = _volatile_counters(mon)
        assert vec[4].item() == pytest.approx(1.0)

    def test_flash_fire_absent(self):
        mon = _minimal_mon(volatiles=[], volatileDetails={})
        vec = _volatile_counters(mon)
        assert vec[4].item() == pytest.approx(0.0)

    def test_flash_fire_with_other_volatiles(self):
        mon = _minimal_mon(
            volatiles=["substitute", "flashfire", "yawn"],
            volatileDetails={"substitute": {"hp": 60}, "flashfire": {}, "yawn": {"time": 1}},
            activeTurns=2,
        )
        vec = _volatile_counters(mon)
        assert vec[0].item() == pytest.approx(60 / 200)
        assert vec[1].item() == pytest.approx(0.0)
        assert vec[2].item() == pytest.approx(2 / 20)
        assert vec[3].item() == pytest.approx(1.0)
        assert vec[4].item() == pytest.approx(1.0)

    # --- Binary volatiles (data-driven from BINARY_VOLATILES) ---

    @pytest.mark.parametrize("volatile_name", BINARY_VOLATILES)
    def test_binary_volatile_active(self, volatile_name):
        """Each binary volatile should be 1.0 at its canonical index when present."""
        idx = BINARY_VOLATILE_OFFSET + BINARY_VOLATILES.index(volatile_name)
        mon = _minimal_mon(
            volatiles=[volatile_name],
            volatileDetails={volatile_name: {}},
        )
        vec = _volatile_counters(mon)
        assert vec[idx].item() == pytest.approx(1.0)

    @pytest.mark.parametrize("volatile_name", BINARY_VOLATILES)
    def test_binary_volatile_absent(self, volatile_name):
        """Each binary volatile should be 0.0 at its canonical index when absent."""
        idx = BINARY_VOLATILE_OFFSET + BINARY_VOLATILES.index(volatile_name)
        mon = _minimal_mon(volatiles=[], volatileDetails={})
        vec = _volatile_counters(mon)
        assert vec[idx].item() == pytest.approx(0.0)

    # --- Perish Song counter (index 5 — special normalized counter) ---

    @pytest.mark.parametrize("duration,expected", [
        (3, 1.0),
        (2, 2 / 3),
        (1, 1 / 3),
        (0, 0.0),
    ])
    def test_perishsong_counter(self, duration, expected):
        mon = _minimal_mon(
            volatiles=["perishsong"],
            volatileDetails={"perishsong": {"duration": duration}},
        )
        vec = _volatile_counters(mon)
        assert vec[5].item() == pytest.approx(expected)

    def test_perishsong_absent(self):
        mon = _minimal_mon(volatiles=[], volatileDetails={})
        vec = _volatile_counters(mon)
        assert vec[5].item() == pytest.approx(0.0)

    # --- Combined multi-volatile test ---

    def test_all_new_volatiles_simultaneously(self):
        all_vols = list(BINARY_VOLATILES) + ["perishsong"]
        details = {v: {} for v in all_vols}
        details["perishsong"] = {"duration": 2}
        mon = _minimal_mon(volatiles=all_vols, volatileDetails=details)
        vec = _volatile_counters(mon)
        assert vec.shape == (29,)
        # Perish song counter at index 5
        assert vec[5].item() == pytest.approx(2 / 3)
        # All binary flags should be 1.0 (indices BINARY_VOLATILE_OFFSET onward)
        for i in range(BINARY_VOLATILE_OFFSET, BINARY_VOLATILE_OFFSET + len(BINARY_VOLATILES)):
            assert vec[i].item() == pytest.approx(1.0), f"index {i} should be 1.0"


class TestVolatilesDoNotCrashEncoder:
    """Volatiles the engine emits but the encoder ignores must not crash encoding."""

    @pytest.mark.parametrize("volatile_name,detail", [
        # Tier 3: single-turn / mid-resolution (intentionally not encoded; see volatile-encoding-audit.md §1)
        ("protect", {"duration": 1}),
        ("flinch", {}),
        ("helpinghand", {}),
        ("followme", {}),
        ("ragepowder", {}),
        ("spotlight", {}),
        ("banefulbunker", {}),
        ("kingsshield", {}),
        ("silktrap", {}),
        ("obstruct", {}),
        ("burningbulwark", {}),
        ("spikyshield", {}),
        ("endure", {}),
        ("focuspunch", {}),
        ("shelltrap", {}),
        ("beakblast", {}),
        ("roost", {}),
        ("powder", {}),
        ("electrify", {}),
        ("dragoncheer", {}),
        ("laserfocus", {}),
        # Omitted Tier 2 (not common enough in Champions meta)
        ("attract", {}),
        ("curse", {}),
        ("ingrain", {}),
        ("aquaring", {}),
        ("telekinesis", {"duration": 3}),
        ("embargo", {"duration": 5}),
        ("syrupbomb", {"duration": 3}),
        ("tarshot", {}),
        ("gastroacid", {}),
        ("choicelock", {}),
        ("commanding", {}),
        ("commanded", {}),
        # Not worth encoding (niche / not in Champions dex / internal markers)
        ("allyswitch", {}),
        ("bide", {}),
        ("counter", {}),
        ("mirrorcoat", {}),
        ("mefirst", {}),
        ("pursuit", {}),
        ("fling", {}),
        ("snatch", {}),
        ("magiccoat", {}),
        ("furycutter", {"hit": 2}),
        ("iceball", {"hit": 1}),
        ("rollout", {"hit": 1}),
        ("chillyreception", {}),
        ("sparklingaria", {}),
        ("defensecurl", {}),
        ("minimize", {}),
        ("foresight", {}),
        ("miracleeye", {}),
        ("stockpile", {"layers": 2}),
        ("grudge", {}),
        ("destinybond", {}),
        ("octolock", {}),
        ("rage", {}),
        ("uproar", {"duration": 3}),
        ("glaiverush", {}),
        ("lockon", {}),
        ("truant", {}),
        ("zenmode", {}),
        ("powershift", {}),
        ("powertrick", {}),
        ("nightmare", {}),
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
            "yawn": {"time": 1},
        }
        mon = _minimal_mon(
            volatiles=list(details.keys()),
            volatileDetails=details,
            activeTurns=5,
        )
        feats = _encode_pokemon_features(mon)
        assert feats.shape == (ENTITY_FEATURE_DIM,)
        vec = _volatile_counters(mon)
        assert vec[0].item() == pytest.approx(80 / 200)   # substitute HP
        assert vec[1].item() == pytest.approx(2 / 6)      # stall counter
        assert vec[2].item() == pytest.approx(5 / 20)     # active turns
        assert vec[3].item() == pytest.approx(1.0)         # yawn
        assert vec[5].item() == pytest.approx(1 / 3)      # perishsong duration=1
        # Binary flags via data-driven offsets
        encore_idx = BINARY_VOLATILE_OFFSET + BINARY_VOLATILES.index("encore")
        taunt_idx = BINARY_VOLATILE_OFFSET + BINARY_VOLATILES.index("taunt")
        confusion_idx = BINARY_VOLATILE_OFFSET + BINARY_VOLATILES.index("confusion")
        assert vec[encore_idx].item() == pytest.approx(1.0)
        assert vec[taunt_idx].item() == pytest.approx(1.0)
        assert vec[confusion_idx].item() == pytest.approx(1.0)
