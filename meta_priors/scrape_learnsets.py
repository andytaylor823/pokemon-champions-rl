"""Scrape per-Pokemon learnsets from pokemon-zone.com for Champions format.

Usage:
    python -m meta_priors.scrape_learnsets
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import requests

_BASE_URL = "https://www.pokemon-zone.com/champions/pokemon"
_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_LEGAL_DIR = _DATA_DIR / "legal"
_OUTPUT_PATH = _LEGAL_DIR / "learnsets.json"

_REQUEST_DELAY = 2.0
_MAX_RETRIES = 3

_MOVE_LINK_RE = re.compile(
    r'<a\s+href="/champions/moves/[^"]+/"[^>]*>([^<]+)</a>'
)
_FORM_LINK_RE = re.compile(
    r'href="/champions/pokemon/([^"]+)/"'
)

# Pokemon-zone slug suffix -> Showdown ID suffix.
# None = skip (shares base learnset: megas, battle-only, cosmetic).
_SLUG_SUFFIX_TO_SHOWDOWN: dict[str, str | None] = {
    "alolan-form": "-alola",
    "hisuian-form": "-hisui",
    "galarian-form": "-galar",
    "paldean-form-combat-breed": "-paldea-combat",
    "paldean-form-blaze-breed": "-paldea-blaze",
    "paldean-form-aqua-breed": "-paldea-aqua",
    "wash-rotom": "-wash",
    "heat-rotom": "-heat",
    "mow-rotom": "-mow",
    "frost-rotom": "-frost",
    "fan-rotom": "-fan",
    "midnight-form": "-midnight",
    "dusk-form": "-dusk",
    "female": "-f",
}

# Forms that share the base learnset exactly (cosmetic or battle-only).
_SKIP_FORM_PATTERNS: tuple[str, ...] = (
    "mega-",
    "blade-forme",
    "busted-form",
    "hangry-mode",
    "hero-form",
    "antique-form",
    "masterpiece-form",
    "family-of-four",
    "ruby-cream", "lemon-cream", "matcha-cream", "mint-cream",
    "salted-cream", "caramel-swirl", "ruby-swirl", "rainbow-swirl",
    "blue-flower", "orange-flower", "white-flower", "yellow-flower",
    "dandy-trim", "debutante-trim", "diamond-trim", "heart-trim",
    "kabuki-trim", "la-reine-trim", "matron-trim", "pharaoh-trim", "star-trim",
    "jumbo-variety", "large-variety", "small-variety",
    "rainy-form", "snowy-form", "sunny-form",
    "-pattern",
)

# Forms that won't be discovered via "Forms:" links on the base page.
_KNOWN_FORM_SLUGS: dict[str, list[str]] = {
    "rotom": [
        "rotom-wash-rotom",
        "rotom-heat-rotom",
        "rotom-mow-rotom",
        "rotom-frost-rotom",
        "rotom-fan-rotom",
    ],
}

_SESSION = requests.Session()
_SESSION.headers.update(
    {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/125.0.0.0 Safari/537.36"
        ),
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;"
            "q=0.9,image/webp,*/*;q=0.8"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
    }
)


def _species_to_slug(species: str) -> str:
    """Convert a pokemon.txt species name to a pokemon-zone URL slug."""
    return species.lower().replace(". ", "-").replace(" ", "-")


def _fetch_page(slug: str) -> str | None:
    """Fetch a pokemon-zone Pokemon page with retries and backoff."""
    url = f"{_BASE_URL}/{slug}/"
    delay = _REQUEST_DELAY
    for attempt in range(1, _MAX_RETRIES + 1):
        time.sleep(delay)
        try:
            resp = _SESSION.get(url, timeout=30)
        except requests.RequestException as exc:
            print(f"  [attempt {attempt}] network error for {slug}: {exc}")
            delay *= 2
            continue

        if resp.status_code == 200:
            return resp.text
        if resp.status_code == 404:
            print(f"  404 for {slug} — skipping")
            return None
        if resp.status_code == 403:
            print(f"  [attempt {attempt}] 403 for {slug} — backing off")
            delay *= 2
            continue
        print(f"  [attempt {attempt}] HTTP {resp.status_code} for {slug}")
        delay *= 2

    print(f"  FAILED after {_MAX_RETRIES} attempts: {slug}")
    return None


def _parse_moves(html: str) -> list[str]:
    """Extract learnable move names from the Learnable Moves section."""
    start = html.find("Learnable Moves")
    if start < 0:
        return []
    section = html[start:]
    return sorted(set(_MOVE_LINK_RE.findall(section)))


def _parse_form_slugs(html: str, base_slug: str) -> list[str]:
    """Extract form page slugs from the 'Forms:' section of a page.

    Returns slugs for non-mega alternate forms only.
    """
    forms_idx = html.find("Forms:")
    if forms_idx < 0:
        return []

    section_end = html.find("</div>", forms_idx)
    section = html[forms_idx : section_end if section_end > 0 else forms_idx + 2000]
    raw_slugs = _FORM_LINK_RE.findall(section)

    result = []
    for slug in raw_slugs:
        if slug == base_slug:
            continue
        suffix = slug[len(base_slug) + 1 :] if slug.startswith(base_slug + "-") else ""
        if not suffix:
            continue
        if any(suffix.startswith(p) or suffix.endswith(p) for p in _SKIP_FORM_PATTERNS):
            continue
        result.append(slug)
    return result


def _slug_to_showdown_id(base_slug: str, form_slug: str) -> str | None:
    """Map a pokemon-zone form slug to a Showdown-style species ID.

    Returns None if the form should be skipped (e.g. mega evolutions).
    """
    suffix = form_slug[len(base_slug) + 1 :]
    for pz_suffix, sd_suffix in _SLUG_SUFFIX_TO_SHOWDOWN.items():
        if suffix == pz_suffix:
            if sd_suffix is None:
                return None
            return base_slug + sd_suffix
    print(f"  Unknown form suffix '{suffix}' for {form_slug} — using as-is")
    return base_slug + "-" + suffix


def _load_species_list() -> list[str]:
    """Load base species from pokemon.txt."""
    names: list[str] = []
    with open(_LEGAL_DIR / "pokemon.txt", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                names.append(stripped)
    return names


def scrape_all() -> dict[str, list[str]]:
    """Scrape learnsets for all legal Pokemon and their alternate forms."""
    species_list = _load_species_list()
    learnsets: dict[str, list[str]] = {}
    failures: list[str] = []

    print(f"Scraping learnsets for {len(species_list)} base species...\n")

    for i, species in enumerate(species_list, 1):
        base_slug = _species_to_slug(species)
        print(f"[{i}/{len(species_list)}] {species} ({base_slug})")

        html = _fetch_page(base_slug)
        if html is None:
            failures.append(base_slug)
            continue

        moves = _parse_moves(html)
        learnsets[base_slug] = moves
        print(f"  -> {len(moves)} moves")

        form_slugs = _parse_form_slugs(html, base_slug)
        if base_slug in _KNOWN_FORM_SLUGS:
            for known in _KNOWN_FORM_SLUGS[base_slug]:
                if known not in form_slugs:
                    form_slugs.append(known)

        for form_slug in form_slugs:
            sd_id = _slug_to_showdown_id(base_slug, form_slug)
            if sd_id is None:
                continue

            print(f"  Form: {form_slug} -> {sd_id}")
            form_html = _fetch_page(form_slug)
            if form_html is None:
                failures.append(form_slug)
                continue

            form_moves = _parse_moves(form_html)
            learnsets[sd_id] = form_moves
            print(f"  -> {len(form_moves)} moves")

    print(f"\n{'='*60}")
    print(f"Done. {len(learnsets)} learnsets scraped.")
    if failures:
        print(f"{len(failures)} failures: {', '.join(failures)}")
    else:
        print("No failures.")

    return learnsets


def main() -> None:
    """Scrape and save learnsets."""
    learnsets = scrape_all()
    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_OUTPUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(learnsets, fh, indent=2, sort_keys=True)
    print(f"\nSaved to {_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
