---
last_synced: a39ccc9
watches:
  - sim/src/
  - src/sim_client.py
---

# SimClient — Implementation Plan

> Module ref: `docs/architecture/repo-architecture.md` §3.1

## Status

Partly built. `sim/src/sim-worker.ts` and `src/sim_client.py` exist with passing test suites.
Snapshot extended with volatile details, status counters, activeTurns, lastItem, side condition
layers, and field duration counters.

## Remaining work

- **Structured chance outcome** — `step` currently returns raw protocol-log delta; needs structured "what randomness fired" for Search chance-bucketing.
