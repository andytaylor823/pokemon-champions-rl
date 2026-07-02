---
alwaysApply: false
paths: sim/src/**/*.ts
---

# Worker stdio JSON-RPC

One JSON request per stdin line; one JSON response per stdout line.
Request: `{ id, cmd, ...args }`. Response: `{ id, ok: true, ...result }` or `{ id, ok: false, error }`.
`src/sim_client.py` is strictly request→response (one stdout line per call) and asserts the echoed `id` matches.

## Commands
`config` · `new_battle` · `open_search` · `step` · `view` · `release` · `close_search` · `stats` · `close`

## Shapes
- `new_battle{team_a, team_b, seed}` → `{handle, view}`
- `open_search{from?}` → `{session, root, root_view}`  (clone the live battle into a search-owned root)
- `step{handle, choices:{p1?,p2?}, seed}` → `{child, view, outcome}`  (`outcome` = protocol log lines added this step, for the search to bucket)
- `view{handle}` → `{view}` · `release{handle}` · `close_search{session}` → `{freed}` · `stats` → `{handles, sessions}`

`view` is a **StateView**: `{ phase, to_move, legal, snapshot, terminal, utility }` (engine-native; see `simclient/showdown-engine-api--1.md`).

## Adding a command (keep both sides in sync)
1. Add a `case "name":` in `dispatch()` (`sim-worker.ts`) returning a plain JSON-safe object.
2. Add the matching method on `SimClient` (`src/sim_client.py`) via `self._rpc("name", ...)`.
3. The Python client mirrors the worker 1:1 — never let them drift.
