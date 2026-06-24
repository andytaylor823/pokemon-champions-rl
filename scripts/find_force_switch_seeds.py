"""One-off script: find battle seeds that reliably produce a forceSwitch phase.

Plays battles with an aggressive action-selection heuristic (prefer foe-targeting
attack moves) and reports which seeds hit forceSwitch and on which turn.

Run from repo root:
    python scripts/find_force_switch_seeds.py
"""
from __future__ import annotations

import random
import sys
import time

import numpy as np

sys.path.insert(0, "src")

from action_space import (
    ACTIONS_PER_SLOT,
    MOVE_PHASE_OFFSET,
    MoveAction,
    _index_to_slot_action,
    index_to_choice_string,
    legal_mask,
)
from sim_client import SimClient, SimError

# ---- Teams from tests/conftest.py -------------------------------------------

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
    return [rng.randint(0, 0xFFFF) for _ in range(4)]


def _pick_aggressive(mask: np.ndarray, rng: random.Random) -> int:
    """Pick a legal action biased toward foe-targeting attack moves.

    Scoring per slot: +2 for a MoveAction, +1 extra if targeting a foe.
    Joint score = sum of both slots. Ties broken randomly.
    """
    legal_indices = np.flatnonzero(mask)
    best_score = -1
    best_actions: list[int] = []

    for idx in legal_indices:
        if idx < MOVE_PHASE_OFFSET:
            # Team preview action — shouldn't appear here but handle gracefully
            score = 0
        else:
            joint_idx = idx - MOVE_PHASE_OFFSET
            s1 = joint_idx // ACTIONS_PER_SLOT
            s2 = joint_idx % ACTIONS_PER_SLOT
            score = 0
            for si in (s1, s2):
                action = _index_to_slot_action(si)
                if isinstance(action, MoveAction):
                    score += 2
                    if action.target in (1, 2):
                        score += 1

        if score > best_score:
            best_score = score
            best_actions = [idx]
        elif score == best_score:
            best_actions.append(idx)

    return rng.choice(best_actions)


def try_seed(sc: SimClient, battle_seed: list[int], max_turns: int = 50) -> tuple[str, int]:
    """Play one battle aggressively. Returns (outcome, turn).

    outcome is one of: "forceSwitch", "terminal", "timeout", "error".
    """
    rng = random.Random(battle_seed[0])

    handle, _ = sc.new_battle(TEAM_A, TEAM_B, seed=battle_seed)
    res = sc.step(handle, {"p1": "team 1234", "p2": "team 1234"}, seed=_rng_seed(rng))
    cur_h = res.child
    cur_v = res.view

    for turn in range(max_turns):
        if cur_v.terminal:
            return "terminal", turn
        if cur_v.phase == "forceSwitch":
            return "forceSwitch", turn

        masks = {s: legal_mask(cur_v.legal.get(s), cur_v.phase) for s in cur_v.to_move}
        stepped = False
        for _ in range(10):
            choices = {}
            for s in cur_v.to_move:
                pick = _pick_aggressive(masks[s], rng)
                choices[s] = index_to_choice_string(pick)
            try:
                res = sc.step(cur_h, choices, seed=_rng_seed(rng))
                cur_h = res.child
                cur_v = res.view
                stepped = True
                break
            except SimError:
                continue
        if not stepped:
            return "error", turn

    return "timeout", max_turns


def main() -> None:
    n = 100
    print(f"Scanning {n} seeds for forceSwitch (aggressive play)...")
    t0 = time.time()

    sc = SimClient(inherit_stderr=False)
    hits: list[tuple[list[int], int]] = []

    for base in range(n):
        seed = [base, base + 1, base + 2, base + 3]
        outcome, turn = try_seed(sc, seed)
        tag = f"seed={seed}  →  {outcome} @ turn {turn}"
        if outcome == "forceSwitch":
            hits.append((seed, turn))
            print(f"  ✓ {tag}")
        else:
            print(f"    {tag}")

    sc.close()
    elapsed = time.time() - t0

    print(f"\n{'='*60}")
    print(f"Done in {elapsed:.1f}s.  {len(hits)}/{n} seeds produced forceSwitch.\n")
    if hits:
        hits.sort(key=lambda x: x[1])
        print("Best candidates (earliest forceSwitch):")
        for seed, turn in hits[:10]:
            print(f"  seed={seed}  turn={turn}")


if __name__ == "__main__":
    main()
