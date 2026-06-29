"""Unit tests for the curriculum module — team definitions and helpers."""
from __future__ import annotations

import copy
import subprocess
import sys
from pathlib import Path

import pytest

# Project root — two levels up from tests/unit/
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)

from curriculum import (
    STAGE_0,
    STAGE_0_FIRE,
    STAGE_0_GRASS,
    STAGE_1,
    STAGE_1_FIRE,
    STAGE_1_GRASS,
    _FIRE_BASE,
    _GRASS_BASE,
    _with_moves,
)
from self_play import CurriculumMatchupSource


# ---------------------------------------------------------------------------
# Gap 15: _with_moves helper
# ---------------------------------------------------------------------------


class TestWithMoves:
    """Tests for the _with_moves team construction helper."""

    def test_correct_number_of_mons(self):
        """Output should have the same number of mons as the base."""
        moves = [["Move A", "Move B"]] * len(_FIRE_BASE)
        result = _with_moves(_FIRE_BASE, moves)
        assert len(result) == len(_FIRE_BASE)

    def test_strict_length_mismatch_raises(self):
        """Mismatched lengths should raise (strict=True in zip)."""
        with pytest.raises(ValueError):
            _with_moves(_FIRE_BASE, [["Move A"]] * 3)

    def test_base_dicts_not_mutated(self):
        """_with_moves should not modify the base list in-place."""
        base_copy = copy.deepcopy(_FIRE_BASE)
        _with_moves(_FIRE_BASE, [["Move A", "Move B"]] * len(_FIRE_BASE))
        assert _FIRE_BASE == base_copy

    def test_output_includes_moves(self):
        """Each output dict should contain the 'moves' key."""
        moves = [["Heat Wave", "Protect"]] * len(_FIRE_BASE)
        result = _with_moves(_FIRE_BASE, moves)
        for mon in result:
            assert "moves" in mon
            assert mon["moves"] == ["Heat Wave", "Protect"]

    def test_output_preserves_base_fields(self):
        """Base fields (species, item, etc.) should survive the merge."""
        moves = [["A"]] * len(_FIRE_BASE)
        result = _with_moves(_FIRE_BASE, moves)
        for base, out in zip(_FIRE_BASE, result):
            assert out["species"] == base["species"]
            assert out["item"] == base["item"]


# ---------------------------------------------------------------------------
# Gap 14: Curriculum team structure validation
# ---------------------------------------------------------------------------


class TestCurriculumTeamStructure:
    """Validate curriculum team definitions without requiring the sim subprocess."""

    def test_stage0_fire_has_6_mons(self):
        assert len(STAGE_0_FIRE) == 6

    def test_stage0_grass_has_6_mons(self):
        assert len(STAGE_0_GRASS) == 6

    def test_stage1_fire_has_6_mons(self):
        assert len(STAGE_1_FIRE) == 6

    def test_stage1_grass_has_6_mons(self):
        assert len(STAGE_1_GRASS) == 6

    def test_stage0_has_2_moves_each(self):
        for mon in STAGE_0_FIRE:
            assert len(mon["moves"]) == 2, f"{mon['species']} should have 2 moves"
        for mon in STAGE_0_GRASS:
            assert len(mon["moves"]) == 2, f"{mon['species']} should have 2 moves"

    def test_stage1_has_4_moves_each(self):
        for mon in STAGE_1_FIRE:
            assert len(mon["moves"]) == 4, f"{mon['species']} should have 4 moves"
        for mon in STAGE_1_GRASS:
            assert len(mon["moves"]) == 4, f"{mon['species']} should have 4 moves"

    def test_stat_points_sum_to_66(self):
        """Each mon's stat points should sum to exactly 66."""
        for team_name, team in [
            ("STAGE_0_FIRE", STAGE_0_FIRE),
            ("STAGE_0_GRASS", STAGE_0_GRASS),
            ("STAGE_1_FIRE", STAGE_1_FIRE),
            ("STAGE_1_GRASS", STAGE_1_GRASS),
        ]:
            for mon in team:
                total = sum(mon["statPoints"].values())
                assert total == 66, f"{team_name} {mon['species']} has {total} stat points, expected 66"

    def test_no_duplicate_species(self):
        """Species clause: no duplicate species within a team."""
        for team_name, team in [
            ("STAGE_0_FIRE", STAGE_0_FIRE),
            ("STAGE_0_GRASS", STAGE_0_GRASS),
            ("STAGE_1_FIRE", STAGE_1_FIRE),
            ("STAGE_1_GRASS", STAGE_1_GRASS),
        ]:
            species = [m["species"] for m in team]
            assert len(species) == len(set(species)), f"{team_name} has duplicate species"

    def test_no_duplicate_items(self):
        """Item clause: no duplicate items within a team."""
        for team_name, team in [
            ("STAGE_0_FIRE", STAGE_0_FIRE),
            ("STAGE_0_GRASS", STAGE_0_GRASS),
            ("STAGE_1_FIRE", STAGE_1_FIRE),
            ("STAGE_1_GRASS", STAGE_1_GRASS),
        ]:
            items = [m["item"] for m in team]
            assert len(items) == len(set(items)), f"{team_name} has duplicate items"

    def test_stage0_and_stage1_same_species(self):
        """Stage 0 and Stage 1 should have the same species (only moves differ)."""
        fire_0_species = [m["species"] for m in STAGE_0_FIRE]
        fire_1_species = [m["species"] for m in STAGE_1_FIRE]
        assert fire_0_species == fire_1_species

        grass_0_species = [m["species"] for m in STAGE_0_GRASS]
        grass_1_species = [m["species"] for m in STAGE_1_GRASS]
        assert grass_0_species == grass_1_species

    def test_pre_built_matchup_sources(self):
        """STAGE_0 and STAGE_1 should be CurriculumMatchupSource instances."""
        assert isinstance(STAGE_0, CurriculumMatchupSource)
        assert isinstance(STAGE_1, CurriculumMatchupSource)

    def test_all_mons_have_required_keys(self):
        """Each mon dict should have species, item, ability, nature, statPoints, moves."""
        required = {"species", "item", "ability", "nature", "statPoints", "moves"}
        for team in [STAGE_0_FIRE, STAGE_0_GRASS, STAGE_1_FIRE, STAGE_1_GRASS]:
            for mon in team:
                missing = required - set(mon.keys())
                assert not missing, f"{mon.get('species', '?')} missing keys: {missing}"


class TestCurriculumTeamLegality:
    """Validate curriculum teams against the legality checker."""

    @staticmethod
    def _team_to_paste(team: list[dict]) -> str:
        """Convert a team dict list to Showdown paste format for the legality checker."""
        lines = []
        for mon in team:
            lines.append(f"{mon['species']} @ {mon['item']}")
            lines.append(f"Ability: {mon['ability']}")
            lines.append(f"{mon['nature']} Nature")
            for move in mon["moves"]:
                lines.append(f"- {move}")
            lines.append("")
        return "\n".join(lines)

    @pytest.mark.parametrize("team_name,team", [
        ("Stage 0 Fire", STAGE_0_FIRE),
        ("Stage 0 Grass", STAGE_0_GRASS),
        ("Stage 1 Fire", STAGE_1_FIRE),
        ("Stage 1 Grass", STAGE_1_GRASS),
    ])
    def test_team_passes_legality_check(self, team_name, team):
        """Each curriculum team should pass the legality checker."""
        paste = self._team_to_paste(team)
        result = subprocess.run(
            [sys.executable, "scripts/check_team_legality.py"],
            input=paste,
            capture_output=True,
            text=True,
            cwd=_PROJECT_ROOT,
        )
        assert result.returncode == 0, (
            f"{team_name} failed legality check:\n{result.stdout}\n{result.stderr}"
        )
