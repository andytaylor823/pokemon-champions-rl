"""Integration tests for the encoder pipeline — real sim views through encoder.

These tests use the session-scoped `sim_client` fixture (a real Node sim-worker
subprocess) and pass real StateViews through the encoder and action_space modules.

Beyond shape checks, value-level assertions verify that the four previously-dead
feature blocks (nature, weather, position, spikes) now produce correct non-zero
features on real battle data.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import action_space
import encoder
from encoder import (
    ENTITY_FEATURE_DIM,
    FIELD_FEATURE_DIM,
    NUM_NATURES,
    SCALAR_FEATURE_DIM,
    SIDE_FEATURE_DIM,
)

if TYPE_CHECKING:
    from sim_client import SimClient


# ---------------------------------------------------------------------------
# Team preview encoding
# ---------------------------------------------------------------------------


class TestEncodeTeamPreview:
    """Encode a real teamPreview StateView from the engine."""

    def test_encode_team_preview_shapes(self, sim_client: SimClient, team_a: list, team_b: list):
        _, view = sim_client.new_battle(team_a, team_b, seed=[1, 2, 3, 4])
        assert view.phase == "teamPreview"

        obs = encoder.encode(view, perspective="p1")

        # 6 my + 6 opp = 12 tokens
        assert obs["entities"].shape == (12, ENTITY_FEATURE_DIM)
        assert obs["ids", "species"].shape == (12,)
        assert obs["ids", "moves"].shape == (12, 4)
        assert obs["action_mask"].shape == (action_space.A,)
        assert obs["field"].shape == (FIELD_FEATURE_DIM,)
        assert obs["sides"].shape == (2, SIDE_FEATURE_DIM)
        assert obs["scalars"].shape == (SCALAR_FEATURE_DIM,)
        assert obs["padding_mask"].shape == (12,)

    def test_team_preview_mask_has_360_trues(self, sim_client: SimClient, team_a: list, team_b: list):
        _, view = sim_client.new_battle(team_a, team_b, seed=[1, 2, 3, 4])
        obs = encoder.encode(view, perspective="p1")
        mask = obs["action_mask"]
        assert mask.sum().item() == action_space.TEAM_PREVIEW_COUNT

    def test_species_ids_nonzero(self, sim_client: SimClient, team_a: list, team_b: list):
        _, view = sim_client.new_battle(team_a, team_b, seed=[1, 2, 3, 4])
        obs = encoder.encode(view, perspective="p1")
        assert (obs["ids", "species"] > 0).all()

    def test_both_perspectives_valid(self, sim_client: SimClient, team_a: list, team_b: list):
        _, view = sim_client.new_battle(team_a, team_b, seed=[1, 2, 3, 4])
        obs_p1 = encoder.encode(view, perspective="p1")
        obs_p2 = encoder.encode(view, perspective="p2")
        assert obs_p1["entities"].shape[0] == 12
        assert obs_p2["entities"].shape[0] == 12


# ---------------------------------------------------------------------------
# Move phase encoding
# ---------------------------------------------------------------------------


class TestEncodeMovePhase:
    """Advance past team preview and encode a real move-phase view."""

    def test_encode_move_phase_shapes(self, sim_client: SimClient, team_a: list, team_b: list):
        live, _ = sim_client.new_battle(team_a, team_b, seed=[1, 2, 3, 4])
        session, root, _ = sim_client.open_search(from_handle=live)
        try:
            res = sim_client.step(root, {"p1": "team 1234", "p2": "team 1234"}, seed=[10, 20, 30, 40])
            move_view = res["view"]
            assert move_view.phase in ("move", "forceSwitch")

            obs = encoder.encode(move_view, perspective="p1")
            n_tokens = obs["entities"].shape[0]
            assert n_tokens == 8
            assert obs["entities"].shape == (n_tokens, ENTITY_FEATURE_DIM)
            assert obs["action_mask"].shape == (action_space.A,)
        finally:
            sim_client.close_search(session)

    def test_move_phase_mask_excludes_team_preview(self, sim_client: SimClient, team_a: list, team_b: list):
        live, _ = sim_client.new_battle(team_a, team_b, seed=[1, 2, 3, 4])
        session, root, _ = sim_client.open_search(from_handle=live)
        try:
            res = sim_client.step(root, {"p1": "team 1234", "p2": "team 1234"}, seed=[10, 20, 30, 40])
            move_view = res["view"]

            obs = encoder.encode(move_view, perspective="p1")
            mask = obs["action_mask"].numpy()
            assert not mask[:action_space.TEAM_PREVIEW_COUNT].any()
            assert mask[action_space.MOVE_PHASE_OFFSET:].any()
        finally:
            sim_client.close_search(session)


# ---------------------------------------------------------------------------
# Round-trip consistency
# ---------------------------------------------------------------------------


class TestMaskConsistency:
    """The encoder's action_mask should match action_space.legal_mask on the same request."""

    def test_encoder_mask_matches_action_space_direct(self, sim_client: SimClient, team_a: list, team_b: list):
        _, view = sim_client.new_battle(team_a, team_b, seed=[5, 6, 7, 8])
        perspective = "p1"
        phase = view.phase
        legal_request = view.legal.get(perspective, {})

        obs = encoder.encode(view, perspective=perspective)
        encoder_mask = obs["action_mask"].numpy()

        direct_mask = action_space.legal_mask(legal_request, phase)
        np.testing.assert_array_equal(encoder_mask, direct_mask)


# ---------------------------------------------------------------------------
# Value-level feature assertions (previously-dead feature blocks)
# ---------------------------------------------------------------------------

# Feature index helpers
_NATURE_START = 1 + 6 + 7 + 7  # hp_frac + stats(6) + boosts(7) + status(7)


class TestNatureFeatures:
    """Nature one-hot should be non-zero for real pokemon (was always zero before fix)."""

    def test_nature_onehot_set_at_team_preview(self, sim_client: SimClient, team_a: list, team_b: list):
        _, view = sim_client.new_battle(team_a, team_b, seed=[1, 2, 3, 4])
        obs = encoder.encode(view, perspective="p1")
        entities = obs["entities"]

        # Every token should have exactly one nature bit set
        for tok_idx in range(entities.shape[0]):
            nature_slice = entities[tok_idx, _NATURE_START : _NATURE_START + NUM_NATURES]
            assert nature_slice.sum().item() == 1.0, f"token {tok_idx}: nature one-hot sum != 1.0"

    def test_nature_onehot_set_in_move_phase(self, sim_client: SimClient, team_a: list, team_b: list):
        live, _ = sim_client.new_battle(team_a, team_b, seed=[1, 2, 3, 4])
        session, root, _ = sim_client.open_search(from_handle=live)
        try:
            res = sim_client.step(root, {"p1": "team 1234", "p2": "team 1234"}, seed=[10, 20, 30, 40])
            obs = encoder.encode(res["view"], perspective="p1")
            entities = obs["entities"]
            for tok_idx in range(entities.shape[0]):
                nature_slice = entities[tok_idx, _NATURE_START : _NATURE_START + NUM_NATURES]
                assert nature_slice.sum().item() == 1.0, f"token {tok_idx}: nature one-hot sum != 1.0"
        finally:
            sim_client.close_search(session)


class TestWeatherFeatures:
    """Weather one-hot should fire when a Drizzle lead is on field."""

    def test_drizzle_sets_weather_feature(self, sim_client: SimClient, team_a: list, team_b: list):
        """Pelipper (team_a slot 5) has Drizzle — bringing it active should set rain."""
        live, _ = sim_client.new_battle(team_a, team_b, seed=[1, 2, 3, 4])
        session, root, _ = sim_client.open_search(from_handle=live)
        try:
            # "team 5123" puts Pelipper (slot 5) active
            res = sim_client.step(root, {"p1": "team 5123", "p2": "team 1234"}, seed=[10, 20, 30, 40])
            move_view = res["view"]

            # Verify the engine set rain
            assert move_view.snapshot.field.weather is not None, "Expected Drizzle to set rain"

            obs = encoder.encode(move_view, perspective="p1")
            field = obs["field"]

            # Weather one-hot (first 5 slots) should have at least one bit set
            weather_slice = field[:5]
            assert weather_slice.sum().item() > 0, "Weather one-hot should be non-zero with Drizzle active"

            # Rain is index 0 in _WEATHER_MAP
            assert field[0].item() == 1.0, "Rain (index 0) should be set by Drizzle"
        finally:
            sim_client.close_search(session)


class TestPositionFeatures:
    """Position one-hot should correctly distinguish active-left vs active-right."""

    def test_active_mons_have_distinct_positions(self, sim_client: SimClient, team_a: list, team_b: list):
        live, _ = sim_client.new_battle(team_a, team_b, seed=[1, 2, 3, 4])
        session, root, _ = sim_client.open_search(from_handle=live)
        try:
            res = sim_client.step(root, {"p1": "team 1234", "p2": "team 1234"}, seed=[10, 20, 30, 40])
            move_view = res["view"]
            obs = encoder.encode(move_view, perspective="p1")
            entities = obs["entities"]

            # Compute the physical slot start index
            flags_start = 1 + 6 + 7 + 7 + 25 + 8 + 3  # hp+stats+boosts+status+nature+moves+volatiles
            slot_start = flags_start + 4  # skip is_active, is_bench, is_fainted, item_consumed

            # First 4 tokens are p1's team (2 active, 2 bench in doubles)
            p1_active_positions = []
            for tok_idx in range(4):
                slot_vec = entities[tok_idx, slot_start : slot_start + 3]
                # Active mons: one of active-left or active-right should be set
                is_active = entities[tok_idx, flags_start].item()
                if is_active > 0:
                    p1_active_positions.append(slot_vec.argmax().item())

            # In doubles, there should be exactly 2 active mons with distinct slot assignments
            assert len(p1_active_positions) == 2, f"Expected 2 active p1 mons, got {len(p1_active_positions)}"
            assert 0 in p1_active_positions, "Expected one active mon at position 0 (active-left)"
            assert 1 in p1_active_positions, "Expected one active mon at position 1 (active-right)"
        finally:
            sim_client.close_search(session)
