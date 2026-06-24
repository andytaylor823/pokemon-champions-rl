---
last_synced: a39ccc9
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

## Deferred adaptive compute (decided in the CVPN grilling; see `cvpn.md`)

- **KataGo-style Playout Cap Randomization (deferred):** run most self-play decisions at a *small*
  search budget to finish games fast, and a sampled fraction (~25%) at the *full* budget — emitting
  policy targets (σ̄) *only* from the full searches, while every move still yields a value target.
  This resolves the value-vs-policy tension and guards against the genuinely sticky failure mode
  (under-searched policy targets biasing the net across generations). **Deferred** until self-play
  throughput is the measured bottleneck. Ref: KataGo arXiv:1902.10565. (The per-decision
  single-legal-move skip *is* planned now — see `search.md`.)
