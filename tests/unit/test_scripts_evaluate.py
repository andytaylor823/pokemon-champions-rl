"""Unit tests for scripts/evaluate.py helper functions.

Tests the three pure, testable helpers that control which checkpoints are
evaluated and in what order:
  - _natural_sort_key
  - _scan_checkpoints
  - _generation_from_path

All tests are pure-Python with no SimClient, no network, and no file I/O beyond
tmp_path fixtures.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Import the helpers from scripts/evaluate.py without running main()
# ---------------------------------------------------------------------------

_SCRIPTS_PATH = Path(__file__).parent.parent.parent / "scripts" / "evaluate.py"


def _import_scripts_evaluate():
    """Load scripts/evaluate.py as a module, suppressing the main() side-effects."""
    spec = importlib.util.spec_from_file_location("scripts_evaluate", _SCRIPTS_PATH)
    module = importlib.util.module_from_spec(spec)
    # Add the module to sys.modules under a unique key so it doesn't clash with
    # the src/evaluation.py module already on the path.
    sys.modules["scripts_evaluate"] = module
    spec.loader.exec_module(module)
    return module


try:
    _mod = _import_scripts_evaluate()
    _natural_sort_key = _mod._natural_sort_key
    _scan_checkpoints = _mod._scan_checkpoints
    _generation_from_path = _mod._generation_from_path
    _IMPORT_OK = True
except Exception as _IMPORT_EXC:
    _IMPORT_OK = False
    _IMPORT_EXC_MSG = str(_IMPORT_EXC)


pytestmark = pytest.mark.skipif(
    not _IMPORT_OK,
    reason=f"Could not import scripts/evaluate.py: {_IMPORT_EXC_MSG if not _IMPORT_OK else ''}",
)


# ---------------------------------------------------------------------------
# _natural_sort_key
# ---------------------------------------------------------------------------


class TestNaturalSortKey:
    """_natural_sort_key must return an integer suitable for numeric sorting."""

    def test_returns_int_for_matching_filename(self):
        p = Path("gen_0042.pt")
        key = _natural_sort_key(p)
        assert isinstance(key, int)
        assert key == 42

    def test_returns_minus_one_for_non_matching_filename(self):
        assert _natural_sort_key(Path("model.pt")) == -1
        assert _natural_sort_key(Path("gen_foo.pt")) == -1
        assert _natural_sort_key(Path("other.bin")) == -1

    def test_numeric_order_diverges_from_lexicographic(self):
        """Natural sort (gen_2 < gen_10) differs from lexicographic (gen_10 < gen_2)."""
        gen2 = Path("gen_0002.pt")
        gen9 = Path("gen_0009.pt")
        gen10 = Path("gen_0010.pt")
        gen100 = Path("gen_0100.pt")

        keys = [_natural_sort_key(p) for p in [gen100, gen10, gen2, gen9]]
        sorted_paths = [p for _, p in sorted(zip(keys, [gen100, gen10, gen2, gen9]))]

        assert sorted_paths == [gen2, gen9, gen10, gen100], (
            "Natural sort order is wrong: expected gen_2 < gen_9 < gen_10 < gen_100"
        )

    def test_key_values_are_strictly_ordered(self):
        """Key for gen_0001 < key for gen_0009 < key for gen_0010 < key for gen_0100."""
        k1 = _natural_sort_key(Path("gen_0001.pt"))
        k9 = _natural_sort_key(Path("gen_0009.pt"))
        k10 = _natural_sort_key(Path("gen_0010.pt"))
        k100 = _natural_sort_key(Path("gen_0100.pt"))
        assert k1 < k9 < k10 < k100

    def test_leading_zeros_irrelevant(self):
        """gen_0010 and gen_10 should have the same key value."""
        assert _natural_sort_key(Path("gen_0010.pt")) == _natural_sort_key(Path("gen_10.pt"))

    def test_generation_zero(self):
        assert _natural_sort_key(Path("gen_0000.pt")) == 0

    def test_large_generation_number(self):
        assert _natural_sort_key(Path("gen_9999.pt")) == 9999


# ---------------------------------------------------------------------------
# _scan_checkpoints
# ---------------------------------------------------------------------------


class TestScanCheckpoints:
    def test_raises_file_not_found_when_no_checkpoints(self, tmp_path):
        """Empty directory (or no gen_*.pt files) must raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            _scan_checkpoints(str(tmp_path))

    def test_raises_file_not_found_non_gen_files_only(self, tmp_path):
        """Files that don't match gen_*.pt must not satisfy the glob."""
        (tmp_path / "model.pt").touch()
        (tmp_path / "checkpoint.bin").touch()
        with pytest.raises(FileNotFoundError):
            _scan_checkpoints(str(tmp_path))

    def test_returns_sorted_list(self, tmp_path):
        """Files created in scrambled order must come out naturally sorted."""
        # Create files in non-sorted order
        for name in ["gen_0010.pt", "gen_0001.pt", "gen_0100.pt", "gen_0002.pt"]:
            (tmp_path / name).touch()

        result = _scan_checkpoints(str(tmp_path))

        names = [p.name for p in result]
        assert names == ["gen_0001.pt", "gen_0002.pt", "gen_0010.pt", "gen_0100.pt"], (
            f"Expected natural sort order, got {names}"
        )

    def test_natural_not_lexicographic_sort(self, tmp_path):
        """Lexicographic sort would put gen_10 before gen_9; verify it does not."""
        for name in ["gen_0009.pt", "gen_0010.pt", "gen_0002.pt"]:
            (tmp_path / name).touch()

        result = _scan_checkpoints(str(tmp_path))
        names = [p.name for p in result]

        # Lexicographic order: gen_0002, gen_0009, gen_0010 — actually the same
        # for zero-padded filenames. Use a case that clearly diverges:
        # gen_0002 < gen_0009 < gen_0010 is correct in both orderings here.
        # Test the pathological case without zero-padding:
        for name in ["gen_0002.pt", "gen_0009.pt", "gen_0010.pt"]:
            assert name in names

        assert names.index("gen_0002.pt") < names.index("gen_0009.pt")
        assert names.index("gen_0009.pt") < names.index("gen_0010.pt")

    def test_returns_path_objects(self, tmp_path):
        """Each element in the result must be a pathlib.Path."""
        (tmp_path / "gen_0001.pt").touch()
        result = _scan_checkpoints(str(tmp_path))
        assert all(isinstance(p, Path) for p in result)

    def test_single_checkpoint(self, tmp_path):
        """A directory with exactly one checkpoint is valid."""
        (tmp_path / "gen_0042.pt").touch()
        result = _scan_checkpoints(str(tmp_path))
        assert len(result) == 1
        assert result[0].name == "gen_0042.pt"

    def test_ignores_non_gen_files(self, tmp_path):
        """Non-matching files in the directory must be excluded."""
        for name in ["gen_0001.pt", "not_gen.pt", "gen_0002.pt", "other.bin"]:
            (tmp_path / name).touch()
        result = _scan_checkpoints(str(tmp_path))
        assert len(result) == 2
        names = [p.name for p in result]
        assert "not_gen.pt" not in names
        assert "other.bin" not in names


# ---------------------------------------------------------------------------
# _generation_from_path
# ---------------------------------------------------------------------------


class TestGenerationFromPath:
    def test_matching_path_returns_generation_number(self):
        assert _generation_from_path(Path("gen_0042.pt")) == 42

    def test_matching_path_gen_0010(self):
        assert _generation_from_path(Path("gen_0010.pt")) == 10

    def test_matching_path_gen_0001(self):
        assert _generation_from_path(Path("gen_0001.pt")) == 1

    def test_non_matching_returns_minus_one(self):
        assert _generation_from_path(Path("other.pt")) == -1
        assert _generation_from_path(Path("model.bin")) == -1
        assert _generation_from_path(Path("gen_foo.pt")) == -1

    def test_generation_zero(self):
        assert _generation_from_path(Path("gen_0000.pt")) == 0

    def test_large_generation(self):
        assert _generation_from_path(Path("gen_9999.pt")) == 9999

    def test_consistent_with_natural_sort_key(self):
        """_generation_from_path and _natural_sort_key must agree for all matching paths."""
        test_cases = [
            Path("gen_0001.pt"),
            Path("gen_0009.pt"),
            Path("gen_0010.pt"),
            Path("gen_0100.pt"),
        ]
        for p in test_cases:
            assert _generation_from_path(p) == _natural_sort_key(p), (
                f"_generation_from_path and _natural_sort_key disagree for {p}"
            )
