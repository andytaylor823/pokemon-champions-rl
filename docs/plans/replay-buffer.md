---
last_synced: 3c3811f
watches:
  - src/
  - docs/architecture/repo-architecture.md
---

# ReplayBuffer — Implementation Plan

> Module ref: `docs/architecture/repo-architecture.md` §3.8

## Status

Not yet started.

## Scope

Fixed-capacity FIFO sliding window. Interface: `add(tuples)`, `sample(batch)`. Stalest tuples age out naturally.
