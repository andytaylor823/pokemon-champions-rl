"""SimClient — Python side of the Pokémon Showdown battle-engine seam.

Owns a persistent Node `sim-worker.ts` subprocess and talks to it over
line-delimited JSON on stdin/stdout. Heavy `Battle` objects live in Node; here
we hold opaque integer handles. See docs/architecture/repo-architecture.md §3.1 and the
SimClient plan.

Decisions realised:
  - one Node subprocess per SimClient (one per self-play worker)
  - engine-native boundary: structured snapshots + Showdown choice strings
  - immutable step: step() returns a fresh child handle, parent untouched
  - mandatory seed: step re-seeds the clone (search samples chance via seeds)
  - scratchpad lifetime: open_search()/close_search() bracket a decision

Integration tests live in tests/integration/test_sim_client.py.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
SIM_DIR = REPO_ROOT / "sim"
DEFAULT_FORMAT = "gen9championsvgc2026regma"

Side = str  # "p1" | "p2"
Seed = list[int]  # [int, int, int, int]


class SimError(RuntimeError):
    """Raised when the worker returns ok=false (e.g. an illegal choice)."""


class SimClient:
    def __init__(
        self,
        format_id: str = DEFAULT_FORMAT,
        sim_dir: Path = SIM_DIR,
        inherit_stderr: bool = False,
    ) -> None:
        self._proc = subprocess.Popen(
            ["npx", "tsx", "src/sim-worker.ts"],
            cwd=str(sim_dir),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None if inherit_stderr else subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._id = 0
        self.config(format_id)

    # --- transport ---------------------------------------------------------
    def _rpc(self, cmd: str, **args: Any) -> dict:
        if self._proc.poll() is not None:
            raise SimError(f"worker exited (code {self._proc.returncode})")
        self._id += 1
        msg = {"id": self._id, "cmd": cmd, **args}
        if self._proc.stdin is None or self._proc.stdout is None:
            raise SimError("worker stdin/stdout pipes are not available")
        self._proc.stdin.write(json.dumps(msg) + "\n")
        self._proc.stdin.flush()
        line = self._proc.stdout.readline()
        if not line:
            raise SimError("worker closed the pipe (no response)")
        resp = json.loads(line)
        if resp.get("id") != self._id:
            raise SimError(f"response id mismatch: sent {self._id}, got {resp.get('id')}")
        if not resp.get("ok"):
            raise SimError(resp.get("error", "unknown worker error"))
        return resp

    # --- commands ----------------------------------------------------------
    def config(self, format_id: str) -> str:
        return self._rpc("config", format_id=format_id)["format_id"]

    def new_battle(self, team_a: list, team_b: list, seed: Optional[Seed] = None) -> tuple[int, dict]:
        r = self._rpc("new_battle", team_a=team_a, team_b=team_b, seed=seed)
        return r["handle"], r["view"]

    def open_search(self, from_handle: Optional[int] = None) -> tuple[int, Optional[int], Optional[dict]]:
        r = self._rpc("open_search", **({"from": from_handle} if from_handle is not None else {}))
        return r["session"], r["root"], r["root_view"]

    def step(self, handle: int, choices: dict[Side, str], seed: Seed) -> dict:
        return self._rpc("step", handle=handle, choices=choices, seed=seed)

    def view(self, handle: int) -> dict:
        return self._rpc("view", handle=handle)["view"]

    def release(self, handle: int) -> None:
        self._rpc("release", handle=handle)

    def close_search(self, session: int) -> int:
        return self._rpc("close_search", session=session)["freed"]

    def stats(self) -> dict:
        return self._rpc("stats")

    def close(self) -> None:
        try:
            self._rpc("close")
        except SimError:
            pass
        try:
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._proc.kill()

    def __enter__(self) -> "SimClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
