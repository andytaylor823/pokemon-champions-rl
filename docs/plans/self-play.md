---
last_synced: 3c3811f
watches:
  - src/
  - docs/architecture/repo-architecture.md
---

# SelfPlay — Implementation Plan

> Module ref: `docs/architecture/repo-architecture.md` §3.7

## Status

Not yet started.

## Scope

Outer-loop data generation. Play full self-play games (each decision = one Search call), emit training tuples `(β, v_search, σ̄)` into the replay buffer.
