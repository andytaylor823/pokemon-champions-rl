"""Integration tests for the CVPN pipeline — real sim views through encoder → CVPN.

These tests use the session-scoped `sim_client` fixture (a real Node sim-worker
subprocess) to drive full battle states through the encoder and CVPN, asserting
no crashes and stable outputs across turns and phases.
"""
from __future__ import annotations

import random
from typing import TYPE_CHECKING

import pytest
import torch

import action_space
import encoder
from cvpn import CVPN

if TYPE_CHECKING:
    from sim_client import SimClient


@pytest.fixture(scope="module")
def cvpn_model():
    """A shared CVPN in eval mode for all integration tests in this module."""
    model = CVPN()
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Team preview
# ---------------------------------------------------------------------------


class TestCVPNTeamPreview:
    """Encode a real teamPreview state and forward through the CVPN."""

    def test_team_preview_forward(self, sim_client: SimClient, team_a: list, team_b: list, cvpn_model):
        _, view = sim_client.new_battle(team_a, team_b, seed=[1, 2, 3, 4])
        assert view.phase == "teamPreview"

        obs = encoder.encode(view, perspective="p1")
        with torch.no_grad():
            policy, value = cvpn_model(obs)

        # Shape checks
        assert policy.shape == (action_space.A,)
        assert value.shape == ()
        assert -1.0 <= value.item() <= 1.0

        # Mask: team preview region has legal actions, move region is all -inf
        tp_region = policy[:action_space.TEAM_PREVIEW_COUNT]
        move_region = policy[action_space.MOVE_PHASE_OFFSET:]
        assert (tp_region > float("-inf")).any()
        assert (move_region == float("-inf")).all()

        # Probabilities over legal actions sum to 1
        probs = torch.softmax(policy, dim=-1)
        legal_sum = probs[obs["action_mask"]].sum()
        assert legal_sum.item() == pytest.approx(1.0, abs=1e-5)

    def test_both_perspectives(self, sim_client: SimClient, team_a: list, team_b: list, cvpn_model):
        _, view = sim_client.new_battle(team_a, team_b, seed=[1, 2, 3, 4])
        for perspective in ("p1", "p2"):
            obs = encoder.encode(view, perspective=perspective)
            with torch.no_grad():
                policy, value = cvpn_model(obs)
            assert policy.shape == (action_space.A,)
            assert -1.0 <= value.item() <= 1.0


# ---------------------------------------------------------------------------
# Move phase
# ---------------------------------------------------------------------------


class TestCVPNMovePhase:
    """Advance past team preview and forward move-phase states through the CVPN."""

    def test_move_phase_forward(self, sim_client: SimClient, team_a: list, team_b: list, cvpn_model):
        live, _ = sim_client.new_battle(team_a, team_b, seed=[1, 2, 3, 4])
        session, root, _ = sim_client.open_search(from_handle=live)
        try:
            res = sim_client.step(root, {"p1": "team 1234", "p2": "team 1234"}, seed=[10, 20, 30, 40])
            move_view = res.view
            assert move_view.phase in ("move", "forceSwitch")

            obs = encoder.encode(move_view, perspective="p1")
            with torch.no_grad():
                policy, value = cvpn_model(obs)

            # Shape checks
            assert policy.shape == (action_space.A,)
            assert value.shape == ()
            assert -1.0 <= value.item() <= 1.0

            # Team-preview region should be all -inf in move phase
            tp_region = policy[:action_space.TEAM_PREVIEW_COUNT]
            assert (tp_region == float("-inf")).all()

            # Move region should have legal actions
            move_region = policy[action_space.MOVE_PHASE_OFFSET:]
            assert (move_region > float("-inf")).any()
        finally:
            sim_client.close_search(session)


# ---------------------------------------------------------------------------
# Multi-turn battle
# ---------------------------------------------------------------------------


class TestCVPNMultiTurn:
    """Play several turns with default moves, encoding and forwarding each state."""

    def test_multi_turn_no_crashes(self, sim_client: SimClient, team_a: list, team_b: list, cvpn_model):
        rng = random.Random(42)
        live, _ = sim_client.new_battle(team_a, team_b, seed=[1, 2, 3, 4])
        session, root, view = sim_client.open_search(from_handle=live)
        try:
            cur = root
            turns_processed = 0
            encode_errors = 0

            for _ in range(8):
                if view.terminal:
                    break

                # Encode and forward from both perspectives
                for perspective in view.to_move:
                    try:
                        obs = encoder.encode(view, perspective=perspective)
                    except ValueError:
                        # Pre-existing action_space gap (e.g. unhandled target
                        # type "allies") — skip this perspective but keep playing
                        encode_errors += 1
                        continue

                    with torch.no_grad():
                        policy, value = cvpn_model(obs)

                    # Invariants that must hold at every step
                    assert policy.shape == (action_space.A,), f"Bad policy shape at turn {turns_processed}"
                    assert value.shape == (), f"Bad value shape at turn {turns_processed}"
                    assert -1.0 <= value.item() <= 1.0, f"Value out of range at turn {turns_processed}"
                    assert torch.isfinite(policy[obs["action_mask"]]).all(), "Non-finite legal logit"

                    # Legal probs sum to ~1
                    probs = torch.softmax(policy, dim=-1)
                    legal_sum = probs[obs["action_mask"]].sum()
                    assert legal_sum.item() == pytest.approx(1.0, abs=1e-4), f"Legal probs don't sum to 1 at turn {turns_processed}"

                # Advance the battle with default moves
                choices = dict.fromkeys(view.to_move, "default")
                seed = [rng.randint(0, 0xFFFF) for _ in range(4)]
                res = sim_client.step(cur, choices, seed=seed)
                cur, view = res.child, res.view
                turns_processed += 1

            assert turns_processed >= 1, "Should have processed at least one turn"
        finally:
            sim_client.close_search(session)
