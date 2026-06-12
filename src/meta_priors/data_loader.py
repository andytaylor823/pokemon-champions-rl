"""Load and parse tournament team data from the Limitless API."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from meta_priors.legality import (
    TOURNAMENTS_META_FILENAME,
    fetch_standings_raw,
    parse_tournament_id,
    validate_standings,
)

_TOURNAMENTS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "tournaments"

logger = logging.getLogger(__name__)

_MEGA_SUFFIX = re.compile(r"-mega(?:-[a-z])?$")

_MINOR_WORDS = frozenset(
    {"to", "of", "the", "a", "an", "in", "on", "at", "for", "and", "or", "but", "nor"}
)


def _normalize_name(name: str) -> str:
    """Capitalize every word except minor words (to, of, ...) in non-initial position.

    Handles inconsistent casing from the Limitless API (e.g. "intimidate" →
    "Intimidate") while preserving correct names like "Zero to Hero".
    """
    if not name:
        return name
    words = name.split()
    return " ".join(
        word if (i > 0 and word.lower() in _MINOR_WORDS) else word[0].upper() + word[1:]
        for i, word in enumerate(words)
        if word
    )


@dataclass(frozen=True)
class PokemonSet:  # pylint: disable=too-many-instance-attributes
    species: str
    item: str
    ability: str
    moves: frozenset[str]
    teammates: tuple[str, ...]
    player: str
    placing: Optional[int]
    games_played: int = 1


@dataclass
class Team:
    player: str
    placing: Optional[int]
    pokemon: list[PokemonSet] = field(default_factory=list)


def _base_species(species_id: str) -> str:
    """Normalize mega-form species IDs to their base form.

    "charizard-mega-x" → "charizard", "floette-mega" → "floette",
    while "meganium" (which contains "mega" in its name) is unchanged.
    """
    return _MEGA_SUFFIX.sub("", species_id)


def parse_standings(
    raw: list[dict],
) -> tuple[list[Team], dict[str, list[PokemonSet]]]:
    """Parse raw API standings into Teams and a per-species index.

    Mega-form species IDs (e.g. "charizard-mega-y") are collapsed to
    their base form so all builds for a species cluster together.

    Returns (teams, species_index) where species_index maps species id
    to every PokemonSet observed for that species across all teams.
    """
    teams: list[Team] = []
    species_index: dict[str, list[PokemonSet]] = {}

    for entry in raw:
        decklist = entry.get("decklist")
        if not decklist:
            continue

        player = entry["player"]
        placing = entry.get("placing")
        record = entry.get("record", {})
        games_played = (
            record.get("wins", 0) + record.get("losses", 0) + record.get("ties", 0)
        ) or 1
        species_ids = [_base_species(mon["id"]) for mon in decklist]

        team = Team(player=player, placing=placing)

        for mon, base_id in zip(decklist, species_ids):
            teammates = tuple(s for s in species_ids if s != base_id)
            pset = PokemonSet(
                species=base_id,
                item=_normalize_name(mon.get("item", "")),
                ability=_normalize_name(mon.get("ability", "")),
                moves=frozenset(
                    _normalize_name(m) for m in mon.get("attacks", [])
                ),
                teammates=teammates,
                player=player,
                placing=placing,
                games_played=games_played,
            )
            team.pokemon.append(pset)
            species_index.setdefault(pset.species, []).append(pset)

        teams.append(team)

    return teams, species_index


def load_tournament(
    url_or_id: str = "69cdcda5d478313a15a39666",
) -> tuple[list[Team], dict[str, list[PokemonSet]]]:
    """High-level loader: fetch/cache then parse.

    Accepts a Limitless tournament URL or bare hex ID.  The underlying
    ``fetch_standings_raw`` caches under a human-readable filename
    derived from the tournament name.
    """
    tournament_id = parse_tournament_id(url_or_id)
    raw = fetch_standings_raw(tournament_id)

    report = validate_standings(raw)
    if not report.clean:
        report.log_warnings()

    return parse_standings(raw)


def load_all_tournaments() -> tuple[list[Team], dict[str, list[PokemonSet]]]:
    """Load every cached tournament JSON in the tournaments directory.

    Returns merged (teams, species_index) across all files.
    """
    all_teams: list[Team] = []
    merged_index: dict[str, list[PokemonSet]] = {}

    if not _TOURNAMENTS_DIR.is_dir():
        logger.warning("Tournaments directory not found: %s", _TOURNAMENTS_DIR)
        return all_teams, merged_index

    files = sorted(
        p for p in _TOURNAMENTS_DIR.glob("*.json")
        if p.name != TOURNAMENTS_META_FILENAME
    )
    if not files:
        logger.warning("No tournament JSON files in %s", _TOURNAMENTS_DIR)
        return all_teams, merged_index

    for path in files:
        try:
            with open(path, encoding="utf-8") as fh:
                raw = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Skipping %s: %s", path.name, exc)
            continue

        teams, species_index = parse_standings(raw)
        all_teams.extend(teams)
        for species, sets in species_index.items():
            merged_index.setdefault(species, []).extend(sets)
        logger.info("Loaded %d teams from %s", len(teams), path.name)

    logger.info(
        "Loaded %d total teams across %d tournament(s)",
        len(all_teams),
        len(files),
    )
    return all_teams, merged_index
