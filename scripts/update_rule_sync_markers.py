"""Bump `last_synced` frontmatter in all cursor rules and plans docs.

Usage:
    python scripts/update_rule_sync_markers.py          # sets last_synced to HEAD
    python scripts/update_rule_sync_markers.py abc1234  # sets to explicit hash
    python scripts/update_rule_sync_markers.py --dry-run  # preview without writing
"""
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

# Scope: only the two sets of files the update-docs skill audits
REPO_ROOT = Path(__file__).resolve().parent.parent
RULE_GLOBS = [
    REPO_ROOT / ".cursor" / "rules",
    REPO_ROOT / "docs" / "plans",
]
EXTENSIONS = {".mdc", ".md"}

# Matches a `last_synced: <value>` line inside YAML frontmatter
SYNC_RE = re.compile(r"^(last_synced:\s*)(\S+)", re.MULTILINE)
# Matches the closing `---` of YAML frontmatter (the second one)
FRONTMATTER_END_RE = re.compile(r"^---\s*$", re.MULTILINE)


def find_files() -> list[Path]:
    """Collect all rule/plan files in scope."""
    files: list[Path] = []
    for root_dir in RULE_GLOBS:
        if root_dir.is_dir():
            for p in sorted(root_dir.rglob("*")):
                if p.is_file() and p.suffix in EXTENSIONS:
                    files.append(p)
    return files


def get_head_sha() -> str:
    """Return the short SHA of the current HEAD commit."""
    return subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()


def update_file(path: Path, target_sha: str, *, dry_run: bool = False) -> str:
    """Update or insert `last_synced` in a file's YAML frontmatter.

    Returns a status string: 'updated', 'already-current', 'inserted', or 'no-frontmatter'.
    """
    text = path.read_text()

    # Check whether the file has YAML frontmatter (starts with ---)
    if not text.startswith("---"):
        return "no-frontmatter"

    # Try to replace an existing last_synced line
    match = SYNC_RE.search(text)
    if match:
        if match.group(2) == target_sha:
            return "already-current"
        new_text = SYNC_RE.sub(rf"\g<1>{target_sha}", text, count=1)
        if not dry_run:
            path.write_text(new_text)
        return "updated"

    # No last_synced line — insert one before the closing ---
    # Find the second --- (end of frontmatter)
    frontmatter_hits = list(FRONTMATTER_END_RE.finditer(text))
    if len(frontmatter_hits) < 2:
        return "no-frontmatter"

    # Insert just before the closing ---
    insert_pos = frontmatter_hits[1].start()
    new_text = text[:insert_pos] + f"last_synced: {target_sha}\n" + text[insert_pos:]
    if not dry_run:
        path.write_text(new_text)
    return "inserted"


def main() -> None:
    parser = argparse.ArgumentParser(description="Update last_synced markers in rules and plans")
    parser.add_argument("sha", nargs="?", default=None, help="Target SHA (defaults to HEAD)")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without writing")
    args = parser.parse_args()

    target = args.sha or get_head_sha()
    files = find_files()

    if not files:
        print("No files found in scope.")
        return

    print(f"Target SHA: {target}  ({'DRY RUN' if args.dry_run else 'live'})")
    print(f"Files in scope: {len(files)}\n")

    for f in files:
        rel = f.relative_to(REPO_ROOT)
        status = update_file(f, target, dry_run=args.dry_run)
        # Color-code output for readability
        symbol = {"updated": "~", "inserted": "+", "already-current": "=", "no-frontmatter": "!"}[status]
        print(f"  [{symbol}] {rel}  ({status})")


if __name__ == "__main__":
    main()
