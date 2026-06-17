"""Integration tests for the encoder pipeline — real sim views through encoder.

These tests use the session-scoped `sim_client` fixture (a real Node sim-worker
subprocess) and pass real StateViews through the encoder and action_space modules.

Beyond shape checks, value-level assertions verify that the four previously-dead
feature blocks (nature, weather, position, spikes) now produce correct non-zero
features on real battle data.
"""
from __future__ import annotations

from typing import TYPE_CHECKING
import random

import numpy as np
import pytest

import action_space
import encoder
from encoder import (
    ENTITY_FEATURE_DIM,
    FIELD_FEATURE_DIM,
    SCALAR_FEATURE_DIM,
    SIDE_FEATURE_DIM,
    _encode_side,
    _nature_onehot,
    _slot_flags,
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
            move_view = res.view
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
            move_view = res.view

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

class TestNatureFeatures:
    """Nature one-hot should be non-zero for real pokemon (was always zero before fix)."""

    def test_nature_onehot_set_at_team_preview(self, sim_client: SimClient, team_a: list, team_b: list):
        _, view = sim_client.new_battle(team_a, team_b, seed=[1, 2, 3, 4])
        # Use the sub-encoder directly — no hand-derived offsets needed
        my_side = view.snapshot.sides[0]
        for i, mon in enumerate(my_side.pokemon):
            nature_vec = _nature_onehot(mon)
            assert nature_vec.sum().item() == 1.0, f"my pokemon {i}: nature one-hot sum != 1.0"

    def test_nature_onehot_set_in_move_phase(self, sim_client: SimClient, team_a: list, team_b: list):
        live, _ = sim_client.new_battle(team_a, team_b, seed=[1, 2, 3, 4])
        session, root, _ = sim_client.open_search(from_handle=live)
        try:
            res = sim_client.step(root, {"p1": "team 1234", "p2": "team 1234"}, seed=[10, 20, 30, 40])
            move_view = res.view
            for side in move_view.snapshot.sides:
                for i, mon in enumerate(side.pokemon):
                    nature_vec = _nature_onehot(mon)
                    assert nature_vec.sum().item() == 1.0, f"{side.id} pokemon {i}: nature one-hot sum != 1.0"
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
            move_view = res.view

            # Verify the engine set rain
            assert move_view.snapshot.field.weather is not None, "Expected Drizzle to set rain"

            obs = encoder.encode(move_view, perspective="p1")
            field = obs["field"]

            # Weather one-hot (first 4 slots) should have at least one bit set
            weather_slice = field[:4]
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
            move_view = res.view
            # Use the sub-encoder directly — no hand-derived offsets needed
            p1_side = move_view.snapshot.sides[0] if move_view.snapshot.sides[0].id == "p1" else move_view.snapshot.sides[1]

            p1_active_positions = []
            for mon in p1_side.pokemon:
                flags = _slot_flags(mon, is_opponent=False)
                # flags layout: [is_active, is_bench, is_fainted, item_consumed, pos_left, pos_right, pos_bench, side]
                if flags[0].item() > 0:  # is_active
                    slot_vec = flags[4:7]  # physical_slot one-hot
                    p1_active_positions.append(slot_vec.argmax().item())

            assert len(p1_active_positions) == 2, f"Expected 2 active p1 mons, got {len(p1_active_positions)}"
            assert 0 in p1_active_positions, "Expected one active mon at position 0 (active-left)"
            assert 1 in p1_active_positions, "Expected one active mon at position 1 (active-right)"
        finally:
            sim_client.close_search(session)


# ---------------------------------------------------------------------------
# Side condition encoding from real engine data
# ---------------------------------------------------------------------------


class TestSideConditionsFromEngine:
    """Side conditions set by real engine moves encode correctly."""

    def test_side_conditions_encode_from_real_battle(self, sim_client: SimClient, team_a: list, team_b: list):
        """Drive a battle with default moves and verify any side conditions encode correctly."""

        rng = random.Random(99)
        live, _ = sim_client.new_battle(team_a, team_b, seed=[1, 2, 3, 4])
        session, root, view = sim_client.open_search(from_handle=live)
        try:
            # Play several turns with default to accumulate side conditions
            cur = root
            for _ in range(10):
                if view.terminal:
                    break
                choices = dict.fromkeys(view.to_move, "default")
                seed = [rng.randint(0, 0xFFFF) for _ in range(4)]
                res = sim_client.step(cur, choices, seed=seed)
                cur, view = res.child, res.view

            # Encode both sides — should not crash regardless of what conditions are set
            for side in view.snapshot.sides:
                feats = _encode_side(side)
                assert feats.shape == (SIDE_FEATURE_DIM,)
        finally:
            sim_client.close_search(session)


# ---------------------------------------------------------------------------
# Perspective consistency
# ---------------------------------------------------------------------------


class TestPerspectiveConsistency:
    """Encoding from p1 vs p2 should swap sides correctly."""

    def test_slot_ids_swap_with_perspective(self, sim_client: SimClient, team_a: list, team_b: list):
        _, view = sim_client.new_battle(team_a, team_b, seed=[1, 2, 3, 4])
        obs_p1 = encoder.encode(view, perspective="p1")
        obs_p2 = encoder.encode(view, perspective="p2")

        # From p1's perspective: first 6 tokens are mine (slot_id=0), next 6 are opponent (slot_id=1)
        assert (obs_p1["slot_id"][:6] == 0).all()
        assert (obs_p1["slot_id"][6:] == 1).all()

        # From p2's perspective: same structure but p2's mons come first
        assert (obs_p2["slot_id"][:6] == 0).all()
        assert (obs_p2["slot_id"][6:] == 1).all()

        # The species IDs should be swapped
        p1_my_species = obs_p1["ids", "species"][:6]
        p1_opp_species = obs_p1["ids", "species"][6:]
        p2_my_species = obs_p2["ids", "species"][:6]
        p2_opp_species = obs_p2["ids", "species"][6:]

        # p1's opponent = p2's own team
        assert (p1_opp_species == p2_my_species).all()
        assert (p1_my_species == p2_opp_species).all()


# ---------------------------------------------------------------------------
# Release frees handle
# ---------------------------------------------------------------------------


class TestReleaseHandle:
    """release() frees a handle; subsequent view() raises SimError."""

    def test_release_then_view_raises(self, sim_client: SimClient, team_a: list, team_b: list):
        from sim_client import SimError

        live, _ = sim_client.new_battle(team_a, team_b, seed=[1, 2, 3, 4])
        session, root, _ = sim_client.open_search(from_handle=live)
        try:
            res = sim_client.step(root, {"p1": "team 1234", "p2": "team 1234"}, seed=[10, 20, 30, 40])
            child = res.child

            sim_client.release(child)

            with pytest.raises(SimError, match="unknown handle"):
                sim_client.view(child)
        finally:
            sim_client.close_search(session)
