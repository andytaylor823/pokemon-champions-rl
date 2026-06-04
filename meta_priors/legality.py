"""Legality validation for Pokemon Champions Regulation M-A.

Legal lists stored in data/legal/*.txt (one name per line, # comments).
Per-Pokemon learnsets stored in data/legal/learnsets.json (scraped from
pokemon-zone.com).
"""

from __future__ import annotations

import json
import logging
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from difflib import get_close_matches
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_LEGAL_DIR = _DATA_DIR / "legal"
_CACHE_DIR = _DATA_DIR / "tournaments"

_LIMITLESS_URL_RE = re.compile(
    r"play\.limitlesstcg\.com/(?:api/)?tournament(?:s)?/([a-f0-9]+)"
)
_LIMITLESS_API = "https://play.limitlesstcg.com/api"
_LIMITLESS_TOURNAMENT_URL = "https://play.limitlesstcg.com/tournament"
_UNSAFE_FILENAME_RE = re.compile(r"[^\w\s\-]")
_WHITESPACE_RE = re.compile(r"\s+")
TOURNAMENTS_META_FILENAME = "_tournaments_meta.json"

# ---------------------------------------------------------------------------
# File loader
# ---------------------------------------------------------------------------


def _load_list(path: Path) -> frozenset[str]:
    """Read a text file of names (one per line, # comments, blank lines ok)."""
    names: list[str] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                names.append(stripped)
    return frozenset(names)


def _load_learnsets(path: Path) -> dict[str, frozenset[str]]:
    """Load per-Pokemon learnsets from a JSON file.

    Keys are Showdown-style species IDs (e.g. "rotom-wash").
    Values are frozensets of move names in Title Case.
    """
    if not path.exists():
        logger.warning("Learnsets file not found: %s", path)
        return {}
    with open(path, encoding="utf-8") as fh:
        raw: dict[str, list[str]] = json.load(fh)
    return {k: frozenset(v) for k, v in raw.items()}


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

_SKIP_ITEMS = frozenset({"", "no item", "nothing", "none"})

_FORM_SUFFIXES = (
    "-mega-x",
    "-mega-y",
    "-mega",
    "-blade",
    "-shield",
    "-hisui",
    "-alola",
    "-galar",
    "-paldea-aqua",
    "-paldea-blaze",
    "-paldea-combat",
    "-bloodmoon",
    "-dusk",
    "-midnight",
    "-wash",
    "-heat",
    "-mow",
    "-frost",
    "-fan",
)


def _normalize(name: str) -> str:
    """Lowercase and strip non-alphanumeric chars for fuzzy comparison."""
    return "".join(c for c in name.lower() if c.isalnum())


def _base_species(species_id: str) -> str:
    """Strip Showdown form suffixes to get the base species."""
    for suffix in _FORM_SUFFIXES:
        if species_id.endswith(suffix):
            return species_id[: -len(suffix)]
    return species_id


# ---------------------------------------------------------------------------
# Legal sets loaded from data/legal/*.txt and learnsets.json
# ---------------------------------------------------------------------------

LEGAL_POKEMON: frozenset[str] = _load_list(_LEGAL_DIR / "pokemon.txt")
LEGAL_ITEMS: frozenset[str] = _load_list(_LEGAL_DIR / "items.txt")
LEGAL_MOVES: frozenset[str] = _load_list(_LEGAL_DIR / "moves.txt")
LEARNSETS: dict[str, frozenset[str]] = _load_learnsets(
    _LEGAL_DIR / "learnsets.json"
)

_POKEMON_LOOKUP: dict[str, str] = {_normalize(n): n for n in LEGAL_POKEMON}
_ITEM_LOOKUP: dict[str, str] = {_normalize(n): n for n in LEGAL_ITEMS}
_MOVE_LOOKUP: dict[str, str] = {_normalize(n): n for n in LEGAL_MOVES}
_LEARNSET_NORM: dict[str, dict[str, str]] = {
    species: {_normalize(m): m for m in moves}
    for species, moves in LEARNSETS.items()
}


# ---------------------------------------------------------------------------
# Report dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LegalityViolation:
    player: str
    species: str
    category: str  # "pokemon", "item", or "move"
    value: str
    suggestion: str


@dataclass
class LegalityReport:
    violations: list[LegalityViolation] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return len(self.violations) == 0

    def summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for v in self.violations:
            counts[v.category] = counts.get(v.category, 0) + 1
        return counts

    def log_warnings(self) -> None:
        if self.clean:
            logger.info("Legality check passed — no violations found.")
            return

        counts = self.summary()
        parts = [f"{n} {cat}" for cat, n in sorted(counts.items())]
        logger.warning(
            "Legality check found %d violation(s): %s",
            len(self.violations),
            ", ".join(parts),
        )

        for v in self.violations:
            hint = f" (did you mean {v.suggestion!r}?)" if v.suggestion else ""
            logger.warning(
                "  [%s] illegal %s on %s: %r%s",
                v.player,
                v.category,
                v.species,
                v.value,
                hint,
            )


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------


def _suggest(name: str, lookup: dict[str, str]) -> str:
    """Return the closest legal name, or empty string if nothing is close."""
    matches = get_close_matches(
        _normalize(name), lookup.keys(), n=1, cutoff=0.6
    )
    return lookup[matches[0]] if matches else ""


def is_legal_pokemon(species_id: str) -> bool:
    return _normalize(_base_species(species_id)) in _POKEMON_LOOKUP


def is_legal_item(item: str) -> bool:
    if item.lower().strip() in _SKIP_ITEMS:
        return True
    return _normalize(item) in _ITEM_LOOKUP


_BATTLE_ONLY_SUFFIXES = (
    "-mega-x", "-mega-y", "-mega", "-blade", "-shield",
)
_WARNED_SPECIES: set[str] = set()


def _resolve_learnset_key(species_id: str) -> str | None:
    """Map a Showdown species ID to a learnsets.json key.

    Tries the full ID first, then strips battle-only suffixes to fall back
    to the base species.  Returns None if no learnset is available.
    """
    if species_id in _LEARNSET_NORM:
        return species_id
    for suffix in _BATTLE_ONLY_SUFFIXES:
        if species_id.endswith(suffix):
            base = species_id[: -len(suffix)]
            if base in _LEARNSET_NORM:
                return base
    if species_id not in _WARNED_SPECIES:
        _WARNED_SPECIES.add(species_id)
        logger.debug("No learnset found for %r — move check skipped", species_id)
    return None


def is_legal_move(move: str, species: str = "") -> bool:
    """Check whether *move* is legal on *species*.

    When a per-Pokemon learnset is available the check is precise.
    Falls back to the global moves list when learnsets are missing.
    """
    if species:
        key = _resolve_learnset_key(species)
        if key is not None:
            return _normalize(move) in _LEARNSET_NORM[key]
    return _normalize(move) in _MOVE_LOOKUP


# ---------------------------------------------------------------------------
# Full dataset validation
# ---------------------------------------------------------------------------


def validate_standings(raw: list[dict]) -> LegalityReport:
    """Scan raw Limitless API standings for illegal pokemon, items, or moves.

    Works on the raw JSON list before it is parsed into dataclasses, so it
    can be called as early as possible in the loading pipeline.
    """
    violations: list[LegalityViolation] = []

    for entry in raw:
        decklist = entry.get("decklist")
        if not decklist:
            continue

        player = entry.get("player", "unknown")

        for mon in decklist:
            species = mon.get("id", "")

            if not is_legal_pokemon(species):
                violations.append(
                    LegalityViolation(
                        player=player,
                        species=species,
                        category="pokemon",
                        value=species,
                        suggestion=_suggest(
                            _base_species(species), _POKEMON_LOOKUP
                        ),
                    )
                )

            item = mon.get("item", "")
            if not is_legal_item(item):
                violations.append(
                    LegalityViolation(
                        player=player,
                        species=species,
                        category="item",
                        value=item,
                        suggestion=_suggest(item, _ITEM_LOOKUP),
                    )
                )

            for move in mon.get("attacks", []):
                if move and not is_legal_move(move, species):
                    violations.append(
                        LegalityViolation(
                            player=player,
                            species=species,
                            category="move",
                            value=move,
                            suggestion=_suggest(move, _MOVE_LOOKUP),
                        )
                    )

    return LegalityReport(violations=violations)


# ---------------------------------------------------------------------------
# URL / tournament helpers
# ---------------------------------------------------------------------------


def parse_tournament_id(url_or_id: str) -> str:
    """Extract a Limitless tournament ID from a URL or pass through a bare ID.

    Accepts formats like:
      https://play.limitlesstcg.com/tournament/69cdcda5d478313a15a39666/standings
      https://play.limitlesstcg.com/api/tournaments/69cdcda5d478313a15a39666/standings
      69cdcda5d478313a15a39666
    """
    match = _LIMITLESS_URL_RE.search(url_or_id)
    if match:
        return match.group(1)
    stripped = url_or_id.strip().strip("/")
    if re.fullmatch(r"[a-f0-9]{20,}", stripped):
        return stripped
    raise ValueError(f"Cannot extract tournament ID from: {url_or_id!r}")


def _sanitize_filename(name: str) -> str:
    """Turn a tournament name into a safe, readable filename slug."""
    cleaned = _UNSAFE_FILENAME_RE.sub(" ", name)
    cleaned = _WHITESPACE_RE.sub("-", cleaned.strip())
    cleaned = re.sub(r"-{2,}", "-", cleaned)
    cleaned = cleaned.strip("-").lower()
    return cleaned or "unknown-tournament"


def fetch_tournament_details(tournament_id: str) -> dict | None:
    """Fetch tournament details from the Limitless API.

    Returns the full details dict, or None on failure.
    """
    try:
        resp = requests.get(
            f"{_LIMITLESS_API}/tournaments/{tournament_id}/details",
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()
    except (requests.RequestException, ValueError):
        logger.debug("Could not fetch tournament details for %s", tournament_id)
        return None


def fetch_tournament_name(tournament_id: str) -> str | None:
    """Fetch the human-readable tournament name from the Limitless API.

    Returns None if the details endpoint is unavailable or missing a name.
    """
    details = fetch_tournament_details(tournament_id)
    return details.get("name") if details else None


def _load_tournaments_meta() -> dict[str, dict]:
    """Read the tournaments meta file, keyed by tournament ID."""
    meta_path = _CACHE_DIR / TOURNAMENTS_META_FILENAME
    if not meta_path.exists():
        return {}
    with open(meta_path, encoding="utf-8") as fh:
        return json.load(fh)


def _save_tournaments_meta(meta: dict[str, dict]) -> None:
    """Write the tournaments meta file (sorted by tournament date)."""
    meta_path = _CACHE_DIR / TOURNAMENTS_META_FILENAME
    sorted_meta = dict(
        sorted(meta.items(), key=lambda kv: kv[1].get("date", ""))
    )
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(sorted_meta, fh, indent=2, ensure_ascii=False)
    logger.info("Updated tournament metadata: %s", meta_path.name)


def _update_tournaments_meta(
    tournament_id: str, details: dict | None
) -> None:
    """Add or update an entry in the tournaments meta file."""
    meta = _load_tournaments_meta()
    name = details.get("name", "") if details else ""
    date = details.get("date", "") if details else ""
    meta[tournament_id] = {
        "name": name,
        "date": date,
        "downloaded": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "url": f"{_LIMITLESS_TOURNAMENT_URL}/{tournament_id}/standings",
        "tournament_id": tournament_id,
    }
    _save_tournaments_meta(meta)


def fetch_standings_raw(
    tournament_id: str, *, cache: bool = True
) -> list[dict]:
    """Fetch standings JSON from the Limitless API, with optional disk cache.

    When caching is enabled, the file is named after the tournament's
    human-readable name (fetched from the details endpoint).  Falls back
    to the raw hex ID if the name is unavailable.  Also updates the
    tournaments meta file with tournament info.
    """
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    existing = _find_cached(tournament_id)
    if cache and existing is not None:
        cache_path, data = existing
        logger.debug("Cache hit: %s", cache_path.name)
        return data

    resp = requests.get(
        f"{_LIMITLESS_API}/tournaments/{tournament_id}/standings", timeout=30
    )
    resp.raise_for_status()
    data = resp.json()

    if cache:
        details = fetch_tournament_details(tournament_id)
        name = details.get("name") if details else None
        slug = _sanitize_filename(name) if name else tournament_id
        cache_path = _CACHE_DIR / f"{slug}.json"
        with open(cache_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        logger.info("Cached standings to %s", cache_path.name)
        _update_tournaments_meta(tournament_id, details)

    return data


def _find_cached(tournament_id: str) -> tuple[Path, list[dict]] | None:
    """Look for a cached standings file for *tournament_id*.

    Checks the legacy ``{id}.json`` filename first, then tries the
    friendly name derived from the tournament details endpoint.
    """
    legacy = _CACHE_DIR / f"{tournament_id}.json"
    if legacy.exists():
        with open(legacy, encoding="utf-8") as fh:
            return legacy, json.load(fh)

    name = fetch_tournament_name(tournament_id)
    if name:
        friendly = _CACHE_DIR / f"{_sanitize_filename(name)}.json"
        if friendly.exists():
            with open(friendly, encoding="utf-8") as fh:
                return friendly, json.load(fh)

    return None


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def _print_report(report: LegalityReport) -> None:
    """Pretty-print the legality report to stdout."""
    if report.clean:
        print("Legality check passed — no violations found.")
        return

    counts = report.summary()
    total = len(report.violations)
    parts = [f"{n} {cat}" for cat, n in sorted(counts.items())]
    print(f"\n{'='*60}")
    print(f"  LEGALITY REPORT — {total} violation(s): {', '.join(parts)}")
    print(f"{'='*60}\n")

    for category in ("pokemon", "item", "move"):
        cat_violations = [
            v for v in report.violations if v.category == category
        ]
        if not cat_violations:
            continue

        print(f"--- Illegal {category} ({len(cat_violations)}) ---")
        for v in cat_violations:
            hint = f"  ->  {v.suggestion}" if v.suggestion else ""
            print(f"  [{v.player}] {v.species}: {v.value}{hint}")
        print()


def main(argv: list[str] | None = None) -> None:
    """CLI: check legality for a Limitless tournament URL or ID."""
    args = argv if argv is not None else sys.argv[1:]
    if not args:
        print(
            "Usage: python -m meta_priors.check_legality <tournament-url-or-id>"
        )
        sys.exit(2)

    url_or_id = args[0]
    tournament_id = parse_tournament_id(url_or_id)

    name = fetch_tournament_name(tournament_id)
    label = f"{name} ({tournament_id})" if name else tournament_id
    print(f"Fetching standings for {label} ...")

    raw = fetch_standings_raw(tournament_id)
    entries_with_teams = sum(
        1 for e in raw if e.get("decklist")
    )
    print(f"Loaded {len(raw)} entries ({entries_with_teams} with decklists).")

    report = validate_standings(raw)
    _print_report(report)

    if not report.clean:
        sys.exit(1)
