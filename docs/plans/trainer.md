---
last_synced: 3c3811f
watches:
  - src/
  - docs/architecture/repo-architecture.md
---

# Trainer — Implementation Plan

> Module ref: `docs/architecture/repo-architecture.md` §3.8

## Status

Not yet started.

## Scope

Minibatch SGD: `L = ‖v̂−v_search‖² + CE(π̂, σ̄) + λ‖θ‖²`. Publishes checkpoints consumed by SelfPlay workers. Training and generation run concurrently.
