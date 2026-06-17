"""Capture a real StateView snapshot from the sim-worker for use as a test fixture.

Produces two snapshots:
  - team_preview: the initial state before any choices
  - move_phase: after advancing past team preview into the first move phase

Both are written as JSON to tests/fixtures/real_snapshot.json with the exact
shape the worker emits. Unit tests should derive their _minimal_mon / _minimal_view
from this fixture so test shapes can never drift from the wire format.

Usage:
    python scripts/capture_snapshot_fixture.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add src to path so we can import SimClient
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sim_client import SimClient

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

FIXTURE_PATH = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "real_snapshot.json"


def main() -> None:
    with SimClient() as sc:
        # Capture team preview snapshot
        handle, team_preview_view = sc.new_battle(TEAM_A, TEAM_B, seed=[1, 2, 3, 4])

        # Step past team preview into move phase
        session, root, _ = sc.open_search(from_handle=handle)
        res = sc.step(root, {"p1": "team 1234", "p2": "team 1234"}, seed=[10, 20, 30, 40])
        move_phase_view = res["view"]
        sc.close_search(session)

    fixture = {
        "team_preview": team_preview_view.model_dump(),
        "move_phase": move_phase_view.model_dump(),
    }

    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FIXTURE_PATH, "w", encoding="utf-8") as f:
        json.dump(fixture, f, indent=2)

    print(f"Fixture written to {FIXTURE_PATH}")
    print(f"  team_preview: {len(team_preview_view.snapshot.sides[0].pokemon)} p1 mons, "
          f"{len(team_preview_view.snapshot.sides[1].pokemon)} p2 mons")
    print(f"  move_phase: {len(move_phase_view.snapshot.sides[0].pokemon)} p1 mons, "
          f"{len(move_phase_view.snapshot.sides[1].pokemon)} p2 mons")

    # Quick sanity: print first mon's nature and position
    mon0 = team_preview_view.snapshot.sides[0].pokemon[0]
    print(f"  First mon: species={mon0.species}, nature={mon0.nature}, position={mon0.position}")


if __name__ == "__main__":
    main()
