"""Vocabulary module — bidirectional ID maps for the encoder.

Loads species, items, moves, abilities, and natures from `data/legal/` and
provides stable integer IDs for the CVPN embedding tables. Index 0 is reserved
for <UNK>/<NONE> (unknown species, no item, empty move slot, etc.).

Keys are normalized to Showdown-style lowercase IDs (no spaces, no hyphens,
all lowercase) — e.g. "heatwave", "charizard", "choicescarf".
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "legal"

# All 25 Pokemon natures (alphabetical)
_NATURES = [
    "Adamant",
    "Bashful",
    "Bold",
    "Brave",
    "Calm",
    "Careful",
    "Docile",
    "Gentle",
    "Hardy",
    "Hasty",
    "Impish",
    "Jolly",
    "Lax",
    "Lonely",
    "Mild",
    "Modest",
    "Naive",
    "Naughty",
    "Quiet",
    "Quirky",
    "Rash",
    "Relaxed",
    "Sassy",
    "Serious",
    "Timid",
]


def _to_showdown_id(name: str) -> str:
    """Convert a display name to Showdown-style ID (lowercase, no spaces/punctuation)."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _load_txt_vocab(path: Path) -> dict[str, int]:
    """Load a newline-separated text file into a name_to_id dict (0 = UNK)."""
    entries: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Skip comments and blank lines
            if not line or line.startswith("#"):
                continue
            entries.append(_to_showdown_id(line))
    # Sort for deterministic IDs, index 0 reserved for UNK/NONE
    entries.sort()
    return {name: idx + 1 for idx, name in enumerate(entries)}


def _load_moves_vocab(learnsets_path: Path) -> dict[str, int]:
    """Extract the union of all legal moves from learnsets.json."""
    with open(learnsets_path, encoding="utf-8") as f:
        learnsets: dict[str, list[str]] = json.load(f)
    # Collect unique moves across all species
    all_moves: set[str] = set()
    for moves in learnsets.values():
        for move in moves:
            all_moves.add(_to_showdown_id(move))
    # Sort for deterministic IDs, index 0 reserved for UNK/NONE
    sorted_moves = sorted(all_moves)
    return {name: idx + 1 for idx, name in enumerate(sorted_moves)}


class Vocab:
    """Bidirectional vocabulary: name <-> integer ID."""

    def __init__(self, name_to_id: dict[str, int], label: str = "") -> None:
        self._name_to_id = name_to_id
        self._id_to_name = {v: k for k, v in name_to_id.items()}
        self.label = label

    @property
    def size(self) -> int:
        """Total vocab size including the reserved UNK slot at index 0."""
        return len(self._name_to_id) + 1

    def encode(self, name: str) -> int:
        """Map a name to its integer ID. Returns 0 (UNK) if not found."""
        return self._name_to_id.get(_to_showdown_id(name), 0)

    def decode(self, idx: int) -> str:
        """Map an integer ID back to its name. Returns '<UNK>' for index 0 or unknown."""
        return self._id_to_name.get(idx, "<UNK>")

    def __contains__(self, name: str) -> bool:
        return _to_showdown_id(name) in self._name_to_id

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        return f"Vocab(label={self.label!r}, size={self.size})"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict (for debugging / inspection)."""
        return {"label": self.label, "size": self.size, "entries": dict(self._name_to_id)}


# --- Module-level singletons (loaded once on import) -------------------------

SPECIES_VOCAB = Vocab(_load_txt_vocab(DATA_DIR / "pokemon.txt"), label="species")
ITEM_VOCAB = Vocab(_load_txt_vocab(DATA_DIR / "items.txt"), label="items")
MOVE_VOCAB = Vocab(_load_moves_vocab(DATA_DIR / "learnsets.json"), label="moves")
ABILITY_VOCAB = Vocab(_load_txt_vocab(DATA_DIR / "abilities.txt"), label="abilities")
NATURE_VOCAB = Vocab(
    {_to_showdown_id(n): idx + 1 for idx, n in enumerate(sorted(_NATURES))},
    label="natures",
)
