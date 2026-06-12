"""Download Pokemon sprites from the Showdown CDN into data/sprites/."""

from __future__ import annotations

import re
import time
from pathlib import Path

import requests

_LEGAL_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "legal"
_SPRITES_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "sprites"

_SHOWDOWN_DEX_URL = (
    "https://play.pokemonshowdown.com/sprites/dex/{showdown_id}.png"
)


def _to_showdown_id(name: str) -> str:
    """Convert a species name to the Showdown sprite ID format.

    Showdown IDs are lowercase alphanumeric only — all punctuation, spaces,
    and hyphens are stripped.
    """
    return re.sub(r"[^a-z0-9]", "", name.lower())


def download_sprites(overwrite: bool = False) -> None:
    _SPRITES_DIR.mkdir(parents=True, exist_ok=True)

    lines = (_LEGAL_DIR / "pokemon.txt").read_text().splitlines()
    species = [
        line.strip()
        for line in lines
        if line.strip() and not line.strip().startswith("#")
    ]

    total = len(species)
    downloaded = 0
    skipped = 0
    failed: list[str] = []

    for i, name in enumerate(species, 1):
        sid = _to_showdown_id(name)
        dest = _SPRITES_DIR / f"{sid}.png"

        if dest.exists() and not overwrite:
            skipped += 1
            continue

        url = _SHOWDOWN_DEX_URL.format(showdown_id=sid)
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            downloaded += 1
            print(f"[{i}/{total}] {name} -> {sid}.png ({len(resp.content)} bytes)")
        except requests.RequestException as exc:
            failed.append(f"{name} ({sid}): {exc}")
            print(f"[{i}/{total}] FAILED {name}: {exc}")

        time.sleep(0.05)

    print(f"\nDone: {downloaded} downloaded, {skipped} skipped, {len(failed)} failed")
    if failed:
        print("Failures:")
        for f in failed:
            print(f"  {f}")


if __name__ == "__main__":
    download_sprites()
