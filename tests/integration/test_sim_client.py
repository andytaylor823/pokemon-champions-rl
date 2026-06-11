"""Integration tests for SimClient — the Python side of the battle-engine seam.

Extracted from SimClient._self_test(). Each test uses the session-scoped
sim_client fixture which spawns a real Node sim-worker subprocess.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

# Ensure src/ is importable
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from sim_client import SimClient, SimError  # type: ignore[import-untyped]

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

TEAM_A = [
    {"species": "Charizard", "item": "Charizardite Y", "ability": "Blaze",
     "moves": ["Heat Wave", "Protect", "Air Slash", "Solar Beam"],
     "nature": "Timid", "statPoints": {"hp": 2, "atk": 0, "def": 0, "spa": 32, "spd": 0, "spe": 32}},
    {"species": "Venusaur", "item": "Lum Berry", "ability": "Chlorophyll",
     "moves": ["Protect", "Sleep Powder", "Giga Drain", "Sludge Bomb"],
     "nature": "Modest", "statPoints": {"hp": 2, "atk": 0, "def": 0, "spa": 32, "spd": 0, "spe": 32}},
    {"species": "Garchomp", "item": "Choice Scarf", "ability": "Rough Skin",
     "moves": ["Earthquake", "Dragon Claw", "Rock Slide", "Protect"],
     "nature": "Jolly", "statPoints": {"hp": 2, "atk": 32, "def": 0, "spa": 0, "spd": 0, "spe": 32}},
    {"species": "Whimsicott", "item": "Mental Herb", "ability": "Prankster",
     "moves": ["Tailwind", "Helping Hand", "Encore", "Protect"],
     "nature": "Timid", "statPoints": {"hp": 32, "atk": 0, "def": 2, "spa": 0, "spd": 0, "spe": 32}},
    {"species": "Pelipper", "item": "Wacan Berry", "ability": "Drizzle",
     "moves": ["Hydro Pump", "Hurricane", "Tailwind", "Protect"],
     "nature": "Bold", "statPoints": {"hp": 32, "atk": 0, "def": 32, "spa": 0, "spd": 2, "spe": 0}},
    {"species": "Incineroar", "item": "Sitrus Berry", "ability": "Intimidate",
     "moves": ["Flare Blitz", "Darkest Lariat", "Fake Out", "Parting Shot"],
     "nature": "Adamant", "statPoints": {"hp": 32, "atk": 32, "def": 0, "spa": 0, "spd": 2, "spe": 0}},
]

TEAM_B = [
    {"species": "Corviknight", "item": "Leftovers", "ability": "Pressure",
     "moves": ["Brave Bird", "Tailwind", "Iron Defense", "Roost"],
     "nature": "Careful", "statPoints": {"hp": 32, "atk": 0, "def": 2, "spa": 0, "spd": 32, "spe": 0}},
    {"species": "Meganium", "item": "Meganiumite", "ability": "Overgrow",
     "moves": ["Body Press", "Light Screen", "Reflect", "Synthesis"],
     "nature": "Bold", "statPoints": {"hp": 32, "atk": 0, "def": 32, "spa": 0, "spd": 2, "spe": 0}},
    {"species": "Sinistcha", "item": "Focus Sash", "ability": "Hospitality",
     "moves": ["Matcha Gotcha", "Rage Powder", "Trick Room", "Life Dew"],
     "nature": "Bold", "statPoints": {"hp": 32, "atk": 0, "def": 32, "spa": 0, "spd": 2, "spe": 0}},
    {"species": "Kingambit", "item": "Chople Berry", "ability": "Defiant",
     "moves": ["Kowtow Cleave", "Swords Dance", "Iron Defense", "Sucker Punch"],
     "nature": "Adamant", "statPoints": {"hp": 32, "atk": 32, "def": 2, "spa": 0, "spd": 0, "spe": 0}},
    {"species": "Meowstic", "item": "Kasib Berry", "ability": "Prankster",
     "moves": ["Psychic", "Light Screen", "Reflect", "Helping Hand"],
     "nature": "Timid", "statPoints": {"hp": 32, "atk": 0, "def": 0, "spa": 2, "spd": 0, "spe": 32}},
    {"species": "Talonflame", "item": "Sharp Beak", "ability": "Gale Wings",
     "moves": ["Brave Bird", "Roost", "Feather Dance", "Bulk Up"],
     "nature": "Jolly", "statPoints": {"hp": 32, "atk": 0, "def": 2, "spa": 0, "spd": 0, "spe": 32}},
]


def _rng_seed(rng: random.Random) -> list[int]:
    """Generate a 4-element PRNG seed list."""
    return [rng.randint(0, 0xFFFF) for _ in range(4)]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSimClientLifecycle:
    """Worker startup, new_battle, and shutdown."""

    def test_new_battle_returns_handle_and_view(self, sim_client: SimClient):
        # Create a battle and verify the returned handle and view
        handle, view = sim_client.new_battle(TEAM_A, TEAM_B, seed=[1, 2, 3, 4])
        assert isinstance(handle, int)
        assert view["phase"] == "teamPreview"
        assert view["terminal"] is False

    def test_view_returns_state(self, sim_client: SimClient):
        # View an existing handle
        handle, _ = sim_client.new_battle(TEAM_A, TEAM_B, seed=[5, 6, 7, 8])
        view = sim_client.view(handle)
        assert view["phase"] == "teamPreview"
        assert len(view["snapshot"]["sides"]) == 2

    def test_stats_reports_handles(self, sim_client: SimClient):
        # Stats should report at least 1 handle (from earlier tests in session)
        stats = sim_client.stats()
        assert isinstance(stats["handles"], int)
        assert stats["handles"] >= 1


class TestSearchSession:
    """open_search / step / close_search lifecycle."""

    def test_full_game_to_terminal(self, sim_client: SimClient):
        """Drive a full game with auto-choices and verify terminal state."""
        rng = random.Random(0)
        # Create the live battle
        live, _ = sim_client.new_battle(TEAM_A, TEAM_B, seed=[1, 2, 3, 4])

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

    def test_close_search_frees_handles(self, sim_client: SimClient):
        """close_search frees all clones; the live battle survives."""
        rng = random.Random(42)
        live, _ = sim_client.new_battle(TEAM_A, TEAM_B, seed=[1, 2, 3, 4])
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

    def test_live_battle_untouched_by_search(self, sim_client: SimClient):
        """The live battle handle remains in teamPreview even after search clones advance."""
        live, _ = sim_client.new_battle(TEAM_A, TEAM_B, seed=[1, 2, 3, 4])
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
