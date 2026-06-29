"""Validation curriculum — hand-crafted matchups for staged Phase-1 correctness proofs.

Each stage is a fixed MatchupSource that tests the architecture's ability to
discover the known-correct solution at a given complexity level.

Stage 0: 6 Fire-types (2 moves each) vs 6 Grass-types (2 moves each).
         Minimal action space, obvious type advantage. Fire should win almost always.
Stage 1: Same teams but with 4 moves each (larger per-slot action space).

Ref: docs/plans/self-play.md §6.2
"""

from __future__ import annotations

from self_play import CurriculumMatchupSource

# ---------------------------------------------------------------------------
# Stage 0 — all-Fire (2 moves) vs all-Grass (2 moves)
# Proves: the architecture can learn *anything*. Fire has a massive type edge;
# Heat Wave (spread, super effective) dominates. Humanly-verifiable correct play.
# ---------------------------------------------------------------------------

STAGE_0_FIRE = [
    {
        "species": "Charizard",
        "item": "Charcoal",
        "ability": "Blaze",
        "moves": ["Heat Wave", "Protect"],
        "nature": "Timid",
        "statPoints": {"hp": 2, "atk": 0, "def": 0, "spa": 32, "spd": 0, "spe": 32},
    },
    {
        "species": "Arcanine",
        "item": "Sitrus Berry",
        "ability": "Intimidate",
        "moves": ["Heat Wave", "Protect"],
        "nature": "Timid",
        "statPoints": {"hp": 32, "atk": 0, "def": 0, "spa": 2, "spd": 0, "spe": 32},
    },
    {
        "species": "Chandelure",
        "item": "Focus Sash",
        "ability": "Flash Fire",
        "moves": ["Heat Wave", "Protect"],
        "nature": "Timid",
        "statPoints": {"hp": 2, "atk": 0, "def": 0, "spa": 32, "spd": 0, "spe": 32},
    },
    {
        "species": "Torkoal",
        "item": "Chesto Berry",
        "ability": "Drought",
        "moves": ["Heat Wave", "Protect"],
        "nature": "Modest",
        "statPoints": {"hp": 32, "atk": 0, "def": 32, "spa": 2, "spd": 0, "spe": 0},
    },
    {
        "species": "Incineroar",
        "item": "Lum Berry",
        "ability": "Intimidate",
        "moves": ["Flare Blitz", "Protect"],
        "nature": "Adamant",
        "statPoints": {"hp": 32, "atk": 32, "def": 0, "spa": 0, "spd": 2, "spe": 0},
    },
    {
        "species": "Talonflame",
        "item": "Sharp Beak",
        "ability": "Gale Wings",
        "moves": ["Flare Blitz", "Protect"],
        "nature": "Jolly",
        "statPoints": {"hp": 2, "atk": 32, "def": 0, "spa": 0, "spd": 0, "spe": 32},
    },
]

STAGE_0_GRASS = [
    {
        "species": "Venusaur",
        "item": "Miracle Seed",
        "ability": "Overgrow",
        "moves": ["Energy Ball", "Protect"],
        "nature": "Modest",
        "statPoints": {"hp": 2, "atk": 0, "def": 0, "spa": 32, "spd": 0, "spe": 32},
    },
    {
        "species": "Meganium",
        "item": "Sitrus Berry",
        "ability": "Overgrow",
        "moves": ["Energy Ball", "Protect"],
        "nature": "Bold",
        "statPoints": {"hp": 32, "atk": 0, "def": 32, "spa": 0, "spd": 2, "spe": 0},
    },
    {
        "species": "Abomasnow",
        "item": "Focus Sash",
        "ability": "Snow Warning",
        "moves": ["Energy Ball", "Protect"],
        "nature": "Modest",
        "statPoints": {"hp": 2, "atk": 0, "def": 0, "spa": 32, "spd": 0, "spe": 32},
    },
    {
        "species": "Chesnaught",
        "item": "Chesto Berry",
        "ability": "Bulletproof",
        "moves": ["Energy Ball", "Protect"],
        "nature": "Bold",
        "statPoints": {"hp": 32, "atk": 0, "def": 32, "spa": 0, "spd": 2, "spe": 0},
    },
    {
        "species": "Appletun",
        "item": "Lum Berry",
        "ability": "Thick Fat",
        "moves": ["Energy Ball", "Protect"],
        "nature": "Modest",
        "statPoints": {"hp": 32, "atk": 0, "def": 0, "spa": 32, "spd": 2, "spe": 0},
    },
    {
        "species": "Torterra",
        "item": "Chople Berry",
        "ability": "Overgrow",
        "moves": ["Energy Ball", "Protect"],
        "nature": "Bold",
        "statPoints": {"hp": 32, "atk": 0, "def": 32, "spa": 0, "spd": 2, "spe": 0},
    },
]


# ---------------------------------------------------------------------------
# Stage 1 — same species, 4 moves each (larger action space)
# ---------------------------------------------------------------------------

STAGE_1_FIRE = [
    {
        "species": "Charizard",
        "item": "Charcoal",
        "ability": "Blaze",
        "moves": ["Heat Wave", "Protect", "Flamethrower", "Overheat"],
        "nature": "Timid",
        "statPoints": {"hp": 2, "atk": 0, "def": 0, "spa": 32, "spd": 0, "spe": 32},
    },
    {
        "species": "Arcanine",
        "item": "Sitrus Berry",
        "ability": "Intimidate",
        "moves": ["Heat Wave", "Protect", "Flamethrower", "Flare Blitz"],
        "nature": "Timid",
        "statPoints": {"hp": 32, "atk": 0, "def": 0, "spa": 2, "spd": 0, "spe": 32},
    },
    {
        "species": "Chandelure",
        "item": "Focus Sash",
        "ability": "Flash Fire",
        "moves": ["Heat Wave", "Protect", "Flamethrower", "Overheat"],
        "nature": "Timid",
        "statPoints": {"hp": 2, "atk": 0, "def": 0, "spa": 32, "spd": 0, "spe": 32},
    },
    {
        "species": "Torkoal",
        "item": "Chesto Berry",
        "ability": "Drought",
        "moves": ["Heat Wave", "Protect", "Flamethrower", "Overheat"],
        "nature": "Modest",
        "statPoints": {"hp": 32, "atk": 0, "def": 32, "spa": 2, "spd": 0, "spe": 0},
    },
    {
        "species": "Incineroar",
        "item": "Lum Berry",
        "ability": "Intimidate",
        "moves": ["Flare Blitz", "Protect", "Flamethrower", "Heat Wave"],
        "nature": "Adamant",
        "statPoints": {"hp": 32, "atk": 32, "def": 0, "spa": 0, "spd": 2, "spe": 0},
    },
    {
        "species": "Talonflame",
        "item": "Sharp Beak",
        "ability": "Gale Wings",
        "moves": ["Flare Blitz", "Protect", "Heat Wave", "Flamethrower"],
        "nature": "Jolly",
        "statPoints": {"hp": 2, "atk": 32, "def": 0, "spa": 0, "spd": 0, "spe": 32},
    },
]

STAGE_1_GRASS = [
    {
        "species": "Venusaur",
        "item": "Miracle Seed",
        "ability": "Overgrow",
        "moves": ["Energy Ball", "Protect", "Giga Drain", "Leaf Storm"],
        "nature": "Modest",
        "statPoints": {"hp": 2, "atk": 0, "def": 0, "spa": 32, "spd": 0, "spe": 32},
    },
    {
        "species": "Meganium",
        "item": "Sitrus Berry",
        "ability": "Overgrow",
        "moves": ["Energy Ball", "Protect", "Giga Drain", "Leaf Storm"],
        "nature": "Bold",
        "statPoints": {"hp": 32, "atk": 0, "def": 32, "spa": 0, "spd": 2, "spe": 0},
    },
    {
        "species": "Abomasnow",
        "item": "Focus Sash",
        "ability": "Snow Warning",
        "moves": ["Energy Ball", "Protect", "Giga Drain", "Leaf Storm"],
        "nature": "Modest",
        "statPoints": {"hp": 2, "atk": 0, "def": 0, "spa": 32, "spd": 0, "spe": 32},
    },
    {
        "species": "Chesnaught",
        "item": "Chesto Berry",
        "ability": "Bulletproof",
        "moves": ["Energy Ball", "Protect", "Giga Drain", "Leaf Storm"],
        "nature": "Bold",
        "statPoints": {"hp": 32, "atk": 0, "def": 32, "spa": 0, "spd": 2, "spe": 0},
    },
    {
        "species": "Appletun",
        "item": "Lum Berry",
        "ability": "Thick Fat",
        "moves": ["Energy Ball", "Protect", "Giga Drain", "Leaf Storm"],
        "nature": "Modest",
        "statPoints": {"hp": 32, "atk": 0, "def": 0, "spa": 32, "spd": 2, "spe": 0},
    },
    {
        "species": "Torterra",
        "item": "Chople Berry",
        "ability": "Overgrow",
        "moves": ["Energy Ball", "Protect", "Giga Drain", "Leaf Storm"],
        "nature": "Bold",
        "statPoints": {"hp": 32, "atk": 0, "def": 32, "spa": 0, "spd": 2, "spe": 0},
    },
]


# ---------------------------------------------------------------------------
# Pre-built MatchupSource instances
# ---------------------------------------------------------------------------

STAGE_0 = CurriculumMatchupSource(team_a=STAGE_0_FIRE, team_b=STAGE_0_GRASS)
STAGE_1 = CurriculumMatchupSource(team_a=STAGE_1_FIRE, team_b=STAGE_1_GRASS)
