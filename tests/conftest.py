"""Shared pytest fixtures.

The `sim_client` fixture starts a real Node sim-worker subprocess and yields a
connected SimClient. It is session-scoped (one worker per test run) and marked
as `integration` so it only runs when explicitly requested:

    pytest -m integration
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure the repo's src/ is importable without editable install
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))


@pytest.fixture(scope="session")
def sim_client():
    """Spawn a sim-worker subprocess, yield a connected SimClient, clean up."""
    # Import here so non-integration tests don't need the sim_client module
    from sim_client import SimClient  # type: ignore[import-untyped]

    # inherit_stderr=True so worker startup errors are visible in test output
    client = SimClient(inherit_stderr=True)
    yield client
    client.close()
