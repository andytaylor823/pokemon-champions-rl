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


def _with_moves(base: list[dict], moves_per_mon: list[list[str]]) -> list[dict]:
    """Attach a per-mon move list to a shared base definition."""
    return [{**mon, "moves": moves} for mon, moves in zip(base, moves_per_mon, strict=True)]


# ---------------------------------------------------------------------------
# Fire team — base stats shared across stages, moves vary
# ---------------------------------------------------------------------------

_FIRE_BASE = [
    {"species": "Charizard", "item": "Charcoal", "ability": "Blaze", "nature": "Timid", "statPoints": {"hp": 2, "atk": 0, "def": 0, "spa": 32, "spd": 0, "spe": 32}},
    {"species": "Arcanine", "item": "Sitrus Berry", "ability": "Intimidate", "nature": "Timid", "statPoints": {"hp": 32, "atk": 0, "def": 0, "spa": 2, "spd": 0, "spe": 32}},
    {"species": "Chandelure", "item": "Focus Sash", "ability": "Flash Fire", "nature": "Timid", "statPoints": {"hp": 2, "atk": 0, "def": 0, "spa": 32, "spd": 0, "spe": 32}},
    {"species": "Torkoal", "item": "Chesto Berry", "ability": "Drought", "nature": "Modest", "statPoints": {"hp": 32, "atk": 0, "def": 32, "spa": 2, "spd": 0, "spe": 0}},
    {"species": "Incineroar", "item": "Lum Berry", "ability": "Intimidate", "nature": "Adamant", "statPoints": {"hp": 32, "atk": 32, "def": 0, "spa": 0, "spd": 2, "spe": 0}},
    {"species": "Talonflame", "item": "Sharp Beak", "ability": "Gale Wings", "nature": "Jolly", "statPoints": {"hp": 2, "atk": 32, "def": 0, "spa": 0, "spd": 0, "spe": 32}},
]

# ---------------------------------------------------------------------------
# Grass team — base stats shared across stages, moves vary
# ---------------------------------------------------------------------------

_GRASS_BASE = [
    {"species": "Venusaur", "item": "Miracle Seed", "ability": "Overgrow", "nature": "Modest", "statPoints": {"hp": 2, "atk": 0, "def": 0, "spa": 32, "spd": 0, "spe": 32}},
    {"species": "Meganium", "item": "Sitrus Berry", "ability": "Overgrow", "nature": "Bold", "statPoints": {"hp": 32, "atk": 0, "def": 32, "spa": 0, "spd": 2, "spe": 0}},
    {"species": "Abomasnow", "item": "Focus Sash", "ability": "Snow Warning", "nature": "Modest", "statPoints": {"hp": 2, "atk": 0, "def": 0, "spa": 32, "spd": 0, "spe": 32}},
    {"species": "Chesnaught", "item": "Chesto Berry", "ability": "Bulletproof", "nature": "Bold", "statPoints": {"hp": 32, "atk": 0, "def": 32, "spa": 0, "spd": 2, "spe": 0}},
    {"species": "Appletun", "item": "Lum Berry", "ability": "Thick Fat", "nature": "Modest", "statPoints": {"hp": 32, "atk": 0, "def": 0, "spa": 32, "spd": 2, "spe": 0}},
    {"species": "Torterra", "item": "Chople Berry", "ability": "Overgrow", "nature": "Bold", "statPoints": {"hp": 32, "atk": 0, "def": 32, "spa": 0, "spd": 2, "spe": 0}},
]

# ---------------------------------------------------------------------------
# Stage 0 — 2 moves each (minimal action space)
# Proves: the architecture can learn *anything*. Fire has a massive type edge;
# Heat Wave (spread, super effective) dominates. Humanly-verifiable correct play.
# ---------------------------------------------------------------------------

STAGE_0_FIRE = _with_moves(
    _FIRE_BASE,
    [
        ["Heat Wave", "Roar"],
        ["Heat Wave", "Rest"],
        ["Heat Wave", "Haze"],
        ["Heat Wave", "Stealth Rock"],
        ["Flare Blitz", "Fake Out"],
        ["Flare Blitz", "Tailwind"],
    ],
)

STAGE_0_GRASS = _with_moves(
    _GRASS_BASE,
    [
        ["Sleep Powder", "Sludge Bomb"],
        ["Energy Ball", "Ancient Power"],
        ["Energy Ball", "Earth Power"],
        ["Energy Ball", "Close Combat"],
        ["Energy Ball", "High Horsepower"],
        ["Energy Ball", "High Horsepower"],
    ],
)

# ---------------------------------------------------------------------------
# Stage 1 — 4 moves each (larger action space, same base teams)
# ---------------------------------------------------------------------------

STAGE_1_FIRE = _with_moves(
    _FIRE_BASE,
    [
        ["Heat Wave", "Protect", "Flamethrower", "Overheat"],
        ["Heat Wave", "Protect", "Flamethrower", "Flare Blitz"],
        ["Heat Wave", "Protect", "Flamethrower", "Overheat"],
        ["Heat Wave", "Protect", "Flamethrower", "Overheat"],
        ["Flare Blitz", "Protect", "Flamethrower", "Heat Wave"],
        ["Flare Blitz", "Protect", "Heat Wave", "Flamethrower"],
    ],
)

STAGE_1_GRASS = _with_moves(
    _GRASS_BASE,
    [
        ["Energy Ball", "Protect", "Giga Drain", "Leaf Storm"],
        ["Energy Ball", "Protect", "Giga Drain", "Leaf Storm"],
        ["Energy Ball", "Protect", "Giga Drain", "Leaf Storm"],
        ["Energy Ball", "Protect", "Giga Drain", "Leaf Storm"],
        ["Energy Ball", "Protect", "Giga Drain", "Leaf Storm"],
        ["Energy Ball", "Protect", "Giga Drain", "Leaf Storm"],
    ],
)


# ---------------------------------------------------------------------------
# Pre-built MatchupSource instances
# ---------------------------------------------------------------------------

STAGE_0 = CurriculumMatchupSource(team_a=STAGE_0_FIRE, team_b=STAGE_0_GRASS)
STAGE_1 = CurriculumMatchupSource(team_a=STAGE_1_FIRE, team_b=STAGE_1_GRASS)
