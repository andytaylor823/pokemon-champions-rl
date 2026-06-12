"""Integration tests for the encoder pipeline — real sim views through encoder.

These tests use the session-scoped `sim_client` fixture (a real Node sim-worker
subprocess) and pass real StateViews through the encoder and action_space modules.
"""
from __future__ import annotations

import numpy as np

import action_space
import encoder
from encoder import ENTITY_FEATURE_DIM, FIELD_FEATURE_DIM, SCALAR_FEATURE_DIM, SIDE_FEATURE_DIM
from sim_client import SimClient


# ---------------------------------------------------------------------------
# Team preview encoding
# ---------------------------------------------------------------------------


class TestEncodeTeamPreview:
    """Encode a real teamPreview StateView from the engine."""

    def test_encode_team_preview_shapes(self, sim_client: SimClient, team_a: list, team_b: list):
        handle, view = sim_client.new_battle(team_a, team_b, seed=[1, 2, 3, 4])
        assert view["phase"] == "teamPreview"

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
        # All 12 tokens should have known species
        assert (obs["ids", "species"] > 0).all()

    def test_both_perspectives_valid(self, sim_client: SimClient, team_a: list, team_b: list):
        _, view = sim_client.new_battle(team_a, team_b, seed=[1, 2, 3, 4])
        obs_p1 = encoder.encode(view, perspective="p1")
        obs_p2 = encoder.encode(view, perspective="p2")
        # Both should produce valid 12-token observations
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
            # Advance past team preview
            res = sim_client.step(root, {"p1": "team 1234", "p2": "team 1234"}, seed=[10, 20, 30, 40])
            move_view = res["view"]
            assert move_view["phase"] in ("move", "forceSwitch")

            obs = encoder.encode(move_view, perspective="p1")
            # After team selection: 4 brought per side = 8 tokens
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
            # Team preview range should be all false in move phase
            assert not mask[:action_space.TEAM_PREVIEW_COUNT].any()
            # Some move-phase actions should be legal
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
        phase = view["phase"]
        legal_request = view.get("legal", {}).get(perspective, {})

        # Encoder-produced mask
        obs = encoder.encode(view, perspective=perspective)
        encoder_mask = obs["action_mask"].numpy()

        # Direct action_space mask
        direct_mask = action_space.legal_mask(legal_request, phase)

        np.testing.assert_array_equal(encoder_mask, direct_mask)
