# SimClient — Implementation Plan

> Module ref: `docs/architecture/repo-architecture.md` §3.1

## Status

Partly built. `sim/src/sim-worker.ts` and `src/sim_client.py` exist with passing test suites.

## Remaining work

- **Snapshot completeness** — v1 snapshot drops volatile/status turn-counters, Protect counter, and Substitute HP (needed by Encoder token anatomy).
- **Structured chance outcome** — `step` currently returns raw protocol-log delta; needs structured "what randomness fired" for Search chance-bucketing.
