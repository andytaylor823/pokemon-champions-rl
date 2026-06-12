"""Shared pytest fixtures.

The `sim_client` fixture starts a real Node sim-worker subprocess and yields a
connected SimClient. It is session-scoped (one worker per entire test run).

`team_a` / `team_b` are the shared Champions Reg M-A teams used across the
integration tests.
"""
from __future__ import annotations

import pytest

from sim_client import SimClient


@pytest.fixture(scope="session")
def sim_client():
    """Spawn a sim-worker subprocess, yield a connected SimClient, clean up."""
    # inherit_stderr=True so worker startup errors are visible in test output
    client = SimClient(inherit_stderr=True)
    yield client
    client.close()


# --- Shared team fixtures (Champions Reg M-A) ------------------------------

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


@pytest.fixture
def team_a() -> list[dict]:
    return TEAM_A


@pytest.fixture
def team_b() -> list[dict]:
    return TEAM_B
