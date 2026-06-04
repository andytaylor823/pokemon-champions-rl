#!/usr/bin/env python3
"""Validate a Pokemon team against Champions Regulation M-A legal lists.

Sources of truth:
  - Species: data/legal/pokemon.txt
  - Items:   data/legal/items.txt
  - Moves:   data/legal/learnsets.json (per-species; moves.txt is ignored)

Usage:
  # Check a single Pokemon
  python scripts/check_team_legality.py \
    --pokemon Charizard --item "Charizardite Y" \
    --moves "Heat Wave,Protect,Air Slash,Solar Beam"

  # Check a full team from Showdown paste format (stdin)
  python scripts/check_team_legality.py < team.txt

  # Check a full team from a file
  python scripts/check_team_legality.py --file team.txt

Showdown paste format example:
  Charizard @ Charizardite Y
  Ability: Blaze
  Level: 50
  EVs: 4 HP / 252 SpA / 252 Spe
  Timid Nature
  - Heat Wave
  - Protect
  - Air Slash
  - Solar Beam
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "legal"


def _load_list(path: Path) -> frozenset[str]:
    names: list[str] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                names.append(stripped)
    return frozenset(names)


def _load_learnsets(path: Path) -> dict[str, list[str]]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


LEGAL_POKEMON = _load_list(DATA_DIR / "pokemon.txt")
LEGAL_ITEMS = _load_list(DATA_DIR / "items.txt")
LEARNSETS: dict[str, list[str]] = _load_learnsets(DATA_DIR / "learnsets.json")

_NORM_POKEMON = {n.lower(): n for n in LEGAL_POKEMON}
_NORM_ITEMS = {n.lower(): n for n in LEGAL_ITEMS}
_NORM_LEARNSETS = {k: {m.lower(): m for m in v} for k, v in LEARNSETS.items()}

_FORM_SUFFIXES = (
    "-mega-x", "-mega-y", "-mega", "-blade", "-shield",
    "-hisui", "-alola", "-galar", "-bloodmoon", "-dusk",
    "-wash", "-heat", "-mow", "-frost",
)


def base_species(name: str) -> str:
    low = name.lower()
    for suffix in _FORM_SUFFIXES:
        if low.endswith(suffix):
            return low[: -len(suffix)]
    return low


def learnset_key(species: str) -> str | None:
    low = species.lower().replace(" ", "").replace("-", "")
    for key in _NORM_LEARNSETS:
        if key.replace("-", "") == low:
            return key
    base = base_species(species)
    normalized_base = base.replace(" ", "").replace("-", "")
    for key in _NORM_LEARNSETS:
        if key.replace("-", "") == normalized_base:
            return key
    return None


def check_pokemon(species: str, item: str, moves: list[str]) -> list[str]:
    errors: list[str] = []

    base = base_species(species)
    if base not in _NORM_POKEMON:
        errors.append(f"ILLEGAL species: {species!r}")

    if item and item.lower() not in _NORM_ITEMS:
        errors.append(f"ILLEGAL item: {item!r}")

    lk = learnset_key(species)
    if lk is None:
        errors.append(f"WARNING: no learnset found for {species!r}, cannot validate moves")
    else:
        moveset = _NORM_LEARNSETS[lk]
        for move in moves:
            if move.lower() not in moveset:
                errors.append(f"ILLEGAL move: {move!r} (not in {species}'s learnset)")

    return errors


def parse_showdown_paste(text: str) -> list[dict]:
    team: list[dict] = []
    current: dict | None = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if current:
                team.append(current)
                current = None
            continue

        if line.startswith("- "):
            if current:
                current["moves"].append(line[2:].strip())
            continue

        if line.startswith("Ability:") or line.startswith("Level:"):
            continue
        if "Nature" in line and not line.startswith("- "):
            continue
        if line.startswith("EVs:") or line.startswith("IVs:"):
            continue
        if line.startswith("Shiny:") or line.startswith("Happiness:"):
            continue

        if "@" in line:
            parts = line.split("@", 1)
            species = parts[0].strip()
            item = parts[1].strip()
            # Strip nickname if present (format: "Nickname (Species)")
            if "(" in species and ")" in species:
                species = species.split("(")[1].split(")")[0].strip()
            current = {"species": species, "item": item, "moves": []}
        elif current is None:
            current = {"species": line, "item": "", "moves": []}

    if current:
        team.append(current)

    return team


def check_team(team: list[dict]) -> bool:
    all_ok = True

    species_seen: list[str] = []
    items_seen: list[str] = []

    for i, mon in enumerate(team, 1):
        species = mon["species"]
        item = mon.get("item", "")
        moves = mon.get("moves", [])

        errors = check_pokemon(species, item, moves)
        status = "PASS" if not errors else "FAIL"

        moves_str = ", ".join(moves) if moves else "(none)"
        print(f"  [{status}] {species} @ {item or '(no item)'} | {moves_str}")
        for err in errors:
            print(f"         {err}")
            all_ok = False

        species_seen.append(base_species(species))
        if item:
            items_seen.append(item.lower())

    duped_species = [s for s in set(species_seen) if species_seen.count(s) > 1]
    if duped_species:
        print(f"\n  [FAIL] Species clause violated: {', '.join(duped_species)}")
        all_ok = False

    duped_items = [it for it in set(items_seen) if items_seen.count(it) > 1]
    if duped_items:
        print(f"\n  [FAIL] Item clause violated: {', '.join(duped_items)}")
        all_ok = False

    return all_ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Check team legality for Champions Reg M-A")
    parser.add_argument("--pokemon", help="Single Pokemon species to check")
    parser.add_argument("--item", default="", help="Held item")
    parser.add_argument("--moves", default="", help="Comma-separated move list")
    parser.add_argument("--file", help="Path to Showdown paste file")
    args = parser.parse_args()

    if args.pokemon:
        moves = [m.strip() for m in args.moves.split(",") if m.strip()]
        errors = check_pokemon(args.pokemon, args.item, moves)
        moves_str = ", ".join(moves) if moves else "(none)"
        status = "PASS" if not errors else "FAIL"
        print(f"[{status}] {args.pokemon} @ {args.item or '(no item)'} | {moves_str}")
        for err in errors:
            print(f"       {err}")
        sys.exit(0 if not errors else 1)

    if args.file:
        text = Path(args.file).read_text(encoding="utf-8")
    elif not sys.stdin.isatty():
        text = sys.stdin.read()
    else:
        parser.print_help()
        sys.exit(2)

    team = parse_showdown_paste(text)
    if not team:
        print("No Pokemon found in input.")
        sys.exit(2)

    print(f"Checking {len(team)} Pokemon:\n")
    ok = check_team(team)
    print()
    if ok:
        print("ALL CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
