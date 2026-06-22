---
last_synced: 3c3811f
watches:
  - src/
  - docs/architecture/
---

# Search — Implementation Plan

> Module ref: `docs/architecture/repo-architecture.md` §3.5
> Design refs: `docs/architecture/gt-cfr-theory.md`, `docs/architecture/search-nn-interface.md`

## Status

Not yet started.

## Scope

The project's primary seam. Inner-loop planner producing a near-optimal strategy at one decision point. Phase 1: MCTS/PUCT. Phase 4: GT-CFR (CFR+ + PUCT expansion + vector leaves).

## Adaptive compute (decided in the CVPN grilling; see `cvpn.md`)

- **Single-legal-move skip (build):** when the legal *canonical* action set has size 1, bypass
  search entirely and emit that action with policy mass 1.0 (value from a single CVPN eval if the
  training target needs it). Composes with the planned action-space canonicalization
  (`repo-architecture.md` §6): the double-faint "bring both back" case collapses to one canonical
  action and is auto-skipped.
- **All other early-stopping is DEFERRED** — including a σ̄-convergence early-stop (abort once the
  root average strategy stops changing between CFR passes, KL < ε; the GT-CFR analog of Lc0's
  move-preserving "smart pruning") and any dynamic per-turn time budget. Defer until search cost
  is a measured bottleneck; the *exact* σ̄-convergence stop carries no strategic risk, whereas
  heuristic pruning does and is unrecoverable in the sense that under-searched nodes bias the
  training targets baked into the net.
- **TODO (research before building the 60 s-turn time manager):** how AlphaGo allocated thinking
  time across moves in the Lee-Sedol match (optimized as a win-rate-vs-budget problem), and Lc0's
  curve-based time management. Refs: AlphaZero arXiv:1712.01815 (fixed 800-sim self-play budget,
  no adaptive stop), KataGo arXiv:1902.10565, lczero.org/blog/2018/09/time-management.
