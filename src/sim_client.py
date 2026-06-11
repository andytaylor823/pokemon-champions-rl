"""SimClient — Python side of the Pokémon Showdown battle-engine seam.

Owns a persistent Node `sim-worker.ts` subprocess and talks to it over
line-delimited JSON on stdin/stdout. Heavy `Battle` objects live in Node; here
we hold opaque integer handles. See docs/repo-architecture.md §3.1 and the
SimClient plan.

Decisions realised:
  - one Node subprocess per SimClient (one per self-play worker)
  - engine-native boundary: structured snapshots + Showdown choice strings
  - immutable step: step() returns a fresh child handle, parent untouched
  - mandatory seed: step re-seeds the clone (search samples chance via seeds)
  - scratchpad lifetime: open_search()/close_search() bracket a decision

Run the built-in integration self-test:
    python src/sim_client.py
"""
from __future__ import annotations

import json
import random
import subprocess
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
SIM_DIR = REPO_ROOT / "sim"
DEFAULT_FORMAT = "gen9championsvgc2026regma"

Side = str  # "p1" | "p2"
Seed = list  # [int, int, int, int]


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
        assert self._proc.stdin is not None and self._proc.stdout is not None
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


# ---------------------------------------------------------------------------
# Integration self-test: drive a full Champions doubles game to terminal,
# re-seeding every step, then verify scratchpad cleanup frees every clone.
# Uses "default" (auto-pick) choices — legal-move/targeting generation is the
# separate Python action module's job, not SimClient's.
# ---------------------------------------------------------------------------
TEAM_A = [
    {"species": "Charizard", "item": "Charizardite Y", "ability": "Blaze",
     "moves": ["Heat Wave", "Protect", "Air Slash", "Solar Beam"],
     "nature": "Timid", "statPoints": {"hp": 2, "atk": 0, "def": 0, "spa": 32, "spd": 0, "spe": 32}},
    {"species": "Venusaur", "item": "Lum Berry", "ability": "Chlorophyll",
     "moves": ["Protect", "Sleep Powder", "Giga Drain", "Sludge Bomb"],
     "nature": "Modest", "statPoints": {"hp": 2, "atk": 0, "def": 0, "spa": 32, "spd": 0, "spe": 32}},
    {"species": "Garchomp", "item": "Choice Scarf", "ability": "Rough Skin",
     "moves": ["Earthquake", "Dragon Claw", "Rock Slide", "Protect"],
     "nature": "Jolly", "statPoints": {"hp": 2, "atk": 32, "def": 0, "spa": 0, "spd": 0, "spe": 32}},
    {"species": "Whimsicott", "item": "Mental Herb", "ability": "Prankster",
     "moves": ["Tailwind", "Helping Hand", "Encore", "Protect"],
     "nature": "Timid", "statPoints": {"hp": 32, "atk": 0, "def": 2, "spa": 0, "spd": 0, "spe": 32}},
    {"species": "Pelipper", "item": "Wacan Berry", "ability": "Drizzle",
     "moves": ["Hydro Pump", "Hurricane", "Tailwind", "Protect"],
     "nature": "Bold", "statPoints": {"hp": 32, "atk": 0, "def": 32, "spa": 0, "spd": 2, "spe": 0}},
    {"species": "Incineroar", "item": "Sitrus Berry", "ability": "Intimidate",
     "moves": ["Flare Blitz", "Darkest Lariat", "Fake Out", "Parting Shot"],
     "nature": "Adamant", "statPoints": {"hp": 32, "atk": 32, "def": 0, "spa": 0, "spd": 2, "spe": 0}},
]
TEAM_B = [
    {"species": "Corviknight", "item": "Leftovers", "ability": "Pressure",
     "moves": ["Brave Bird", "Tailwind", "Iron Defense", "Roost"],
     "nature": "Careful", "statPoints": {"hp": 32, "atk": 0, "def": 2, "spa": 0, "spd": 32, "spe": 0}},
    {"species": "Meganium", "item": "Meganiumite", "ability": "Overgrow",
     "moves": ["Body Press", "Light Screen", "Reflect", "Synthesis"],
     "nature": "Bold", "statPoints": {"hp": 32, "atk": 0, "def": 32, "spa": 0, "spd": 2, "spe": 0}},
    {"species": "Sinistcha", "item": "Focus Sash", "ability": "Hospitality",
     "moves": ["Matcha Gotcha", "Rage Powder", "Trick Room", "Life Dew"],
     "nature": "Bold", "statPoints": {"hp": 32, "atk": 0, "def": 32, "spa": 0, "spd": 2, "spe": 0}},
    {"species": "Kingambit", "item": "Chople Berry", "ability": "Defiant",
     "moves": ["Kowtow Cleave", "Swords Dance", "Iron Defense", "Sucker Punch"],
     "nature": "Adamant", "statPoints": {"hp": 32, "atk": 32, "def": 2, "spa": 0, "spd": 0, "spe": 0}},
    {"species": "Meowstic", "item": "Kasib Berry", "ability": "Prankster",
     "moves": ["Psychic", "Light Screen", "Reflect", "Helping Hand"],
     "nature": "Timid", "statPoints": {"hp": 32, "atk": 0, "def": 0, "spa": 2, "spd": 0, "spe": 32}},
    {"species": "Talonflame", "item": "Sharp Beak", "ability": "Gale Wings",
     "moves": ["Brave Bird", "Roost", "Feather Dance", "Bulk Up"],
     "nature": "Jolly", "statPoints": {"hp": 32, "atk": 0, "def": 2, "spa": 0, "spd": 0, "spe": 32}},
]


def _rng_seed(rng: random.Random) -> Seed:
    return [rng.randint(0, 0xFFFF) for _ in range(4)]


def _self_test() -> None:
    rng = random.Random(0)
    print("=== SimClient integration self-test ===")
    with SimClient(inherit_stderr=True) as sc:
        live, _ = sc.new_battle(TEAM_A, TEAM_B, seed=[1, 2, 3, 4])
        print(f"[new_battle] live handle={live}")

        session, root, view = sc.open_search(from_handle=live)
        print(f"[open_search] session={session} root={root} phase={view['phase']}")

        cur, steps = root, 0
        while not view["terminal"] and steps < 200:
            choices = {side: "default" for side in view["to_move"]}
            res = sc.step(cur, choices, seed=_rng_seed(rng))
            cur, view = res["child"], res["view"]
            steps += 1
        print(f"[play] reached phase={view['phase']} after {steps} steps; "
              f"turn={view['snapshot']['turn']} utility={view['utility']}")

        assert view["terminal"], "game did not terminate"
        assert view["utility"] in ({"p1": 1, "p2": -1}, {"p1": -1, "p2": 1}, {"p1": 0, "p2": 0})

        before = sc.stats()["handles"]
        freed = sc.close_search(session)
        after = sc.stats()["handles"]
        print(f"[cleanup] handles {before} -> {after} (freed {freed}); live battle survives")
        assert after == 1, f"expected only the live battle to remain, got {after}"

        # The live battle is untouched by the search and still steppable.
        live_view = sc.view(live)
        assert not live_view["terminal"] and live_view["phase"] == "teamPreview"
    print("SELF-TEST PASSED")


if __name__ == "__main__":
    _self_test()
