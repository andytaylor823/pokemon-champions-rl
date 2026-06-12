"""Integration tests for SimClient — the Python side of the battle-engine seam.

Each test uses the session-scoped `sim_client` fixture (a real Node sim-worker
subprocess) plus the shared `team_a` / `team_b` fixtures from conftest.
"""
from __future__ import annotations

import random

import pytest

from sim_client import SimClient, SimError


def _rng_seed(rng: random.Random) -> list[int]:
    """Generate a 4-element PRNG seed list."""
    return [rng.randint(0, 0xFFFF) for _ in range(4)]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSimClientLifecycle:
    """Worker startup, new_battle, and shutdown."""

    def test_new_battle_returns_handle_and_view(self, sim_client: SimClient, team_a: list, team_b: list):
        # Create a battle and verify the returned handle and view
        handle, view = sim_client.new_battle(team_a, team_b, seed=[1, 2, 3, 4])
        assert isinstance(handle, int)
        assert view["phase"] == "teamPreview"
        assert view["terminal"] is False

    def test_view_returns_state(self, sim_client: SimClient, team_a: list, team_b: list):
        # View an existing handle
        handle, _ = sim_client.new_battle(team_a, team_b, seed=[5, 6, 7, 8])
        view = sim_client.view(handle)
        assert view["phase"] == "teamPreview"
        assert len(view["snapshot"]["sides"]) == 2

    def test_stats_counts_a_new_battle(self, sim_client: SimClient, team_a: list, team_b: list):
        # Order-independent: a new battle bumps the live handle count by exactly one.
        before = sim_client.stats()["handles"]
        sim_client.new_battle(team_a, team_b, seed=[1, 2, 3, 4])
        after = sim_client.stats()["handles"]
        assert isinstance(after, int)
        assert after == before + 1


class TestSearchSession:
    """open_search / step / close_search lifecycle."""

    def test_full_game_to_terminal(self, sim_client: SimClient, team_a: list, team_b: list):
        """Drive a full game with auto-choices and verify terminal state."""
        rng = random.Random(0)
        # Create the live battle
        live, _ = sim_client.new_battle(team_a, team_b, seed=[1, 2, 3, 4])

        # Open a search session cloned from the live battle
        session, root, view = sim_client.open_search(from_handle=live)
        assert isinstance(session, int)
        assert isinstance(root, int)
        assert view["phase"] == "teamPreview"

        # Play the game to terminal using "default" auto-choices
        cur, steps = root, 0
        while not view["terminal"] and steps < 200:
            choices = {side: "default" for side in view["to_move"]}
            res = sim_client.step(cur, choices, seed=_rng_seed(rng))
            cur, view = res["child"], res["view"]
            steps += 1

        # Verify the game terminated properly
        assert view["terminal"], f"Game did not terminate after {steps} steps"
        assert view["utility"] in (
            {"p1": 1, "p2": -1},
            {"p1": -1, "p2": 1},
            {"p1": 0, "p2": 0},
        )

    def test_close_search_frees_handles(self, sim_client: SimClient, team_a: list, team_b: list):
        """close_search frees all clones; the live battle survives."""
        rng = random.Random(42)
        live, _ = sim_client.new_battle(team_a, team_b, seed=[1, 2, 3, 4])
        session, root, view = sim_client.open_search(from_handle=live)

        # Step a few times to accumulate handles
        cur = root
        for _ in range(3):
            if view["terminal"]:
                break
            choices = {side: "default" for side in view["to_move"]}
            res = sim_client.step(cur, choices, seed=_rng_seed(rng))
            cur, view = res["child"], res["view"]

        # Record handle count before cleanup
        before = sim_client.stats()["handles"]
        freed = sim_client.close_search(session)
        after = sim_client.stats()["handles"]

        assert freed >= 2, f"Expected at least 2 freed handles, got {freed}"
        assert after < before

        # The live battle should still be accessible
        live_view = sim_client.view(live)
        assert not live_view["terminal"]
        assert live_view["phase"] == "teamPreview"

    def test_live_battle_untouched_by_search(self, sim_client: SimClient, team_a: list, team_b: list):
        """The live battle handle remains in teamPreview even after search clones advance."""
        live, _ = sim_client.new_battle(team_a, team_b, seed=[1, 2, 3, 4])
        session, root, _ = sim_client.open_search(from_handle=live)

        # Advance the search clone past team preview
        sim_client.step(root, {"p1": "team 1234", "p2": "team 1234"}, seed=[10, 20, 30, 40])

        # Live battle should be untouched
        live_view = sim_client.view(live)
        assert live_view["phase"] == "teamPreview"

        sim_client.close_search(session)


class TestErrorHandling:
    """SimClient error propagation from the worker."""

    def test_unknown_handle_raises(self, sim_client: SimClient):
        """Viewing a nonexistent handle raises SimError."""
        with pytest.raises(SimError, match="unknown handle"):
            sim_client.view(999999)
