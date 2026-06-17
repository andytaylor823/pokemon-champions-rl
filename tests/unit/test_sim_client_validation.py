"""Unit tests for SimClient validation boundary — no subprocess required.

These test the Pydantic models (StepResult, StateView) that validate
wire data from the worker. They don't need the live engine.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from sim_client import SimClient, StepResult
from state_types import StateView

# Load real fixture for positive-case testing
_FIXTURE_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "real_snapshot.json"
with open(_FIXTURE_PATH, encoding="utf-8") as _f:
    _REAL = json.load(_f)


# ---------------------------------------------------------------------------
# StepResult validation
# ---------------------------------------------------------------------------


class TestStepResultValidation:
    """StepResult must reject invalid data from the worker."""

    def test_valid_step_result(self):
        """A well-formed StepResult parses without error."""
        data = {
            "child": 42,
            "view": _REAL["move_phase"],
            "outcome": ["|-weather|RainDance|[from] ability: Drizzle"],
        }
        result = StepResult.model_validate(data)
        assert result.child == 42
        assert result.view.phase in ("move", "forceSwitch")
        assert isinstance(result.outcome, list)

    def test_child_must_be_int(self):
        data = {
            "child": "not_an_int",
            "view": _REAL["move_phase"],
            "outcome": [],
        }
        with pytest.raises(ValidationError):
            StepResult.model_validate(data)

    def test_missing_view_raises(self):
        data = {"child": 42, "outcome": []}
        with pytest.raises(ValidationError):
            StepResult.model_validate(data)

    def test_missing_child_raises(self):
        data = {"view": _REAL["move_phase"], "outcome": []}
        with pytest.raises(ValidationError):
            StepResult.model_validate(data)

    def test_missing_outcome_raises(self):
        data = {"child": 42, "view": _REAL["move_phase"]}
        with pytest.raises(ValidationError):
            StepResult.model_validate(data)

    def test_invalid_view_nested_raises(self):
        """A StepResult with a malformed view dict raises ValidationError."""
        data = {
            "child": 42,
            "view": {"phase": "move"},  # missing required fields
            "outcome": [],
        }
        with pytest.raises(ValidationError):
            StepResult.model_validate(data)


# ---------------------------------------------------------------------------
# SimClient._parse_view validation
# ---------------------------------------------------------------------------


class TestParseViewValidation:
    """SimClient._parse_view must raise on invalid wire data."""

    def test_valid_team_preview(self):
        view = SimClient._parse_view(_REAL["team_preview"])
        assert view.phase == "teamPreview"
        assert isinstance(view, StateView)

    def test_valid_move_phase(self):
        view = SimClient._parse_view(_REAL["move_phase"])
        assert view.phase in ("move", "forceSwitch")

    def test_empty_dict_raises(self):
        with pytest.raises(ValidationError):
            SimClient._parse_view({})

    def test_missing_snapshot_raises(self):
        data = {"phase": "move", "to_move": ["p1"], "legal": {}, "terminal": False, "utility": None}
        with pytest.raises(ValidationError):
            SimClient._parse_view(data)

    def test_missing_phase_raises(self):
        data = {
            "to_move": ["p1"],
            "legal": {},
            "snapshot": _REAL["team_preview"]["snapshot"],
            "terminal": False,
            "utility": None,
        }
        with pytest.raises(ValidationError):
            SimClient._parse_view(data)

    def test_invalid_pokemon_in_snapshot_raises(self):
        """A snapshot with a malformed pokemon entry raises ValidationError."""
        import copy
        data = copy.deepcopy(_REAL["team_preview"])
        # Corrupt first pokemon: remove required field
        del data["snapshot"]["sides"][0]["pokemon"][0]["hp"]
        with pytest.raises(ValidationError):
            SimClient._parse_view(data)
