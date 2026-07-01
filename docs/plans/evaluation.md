---
last_synced: a39ccc9
watches:
  - src/
  - docs/architecture/repo-architecture.md
---

# Evaluation — Implementation Plan

> Module ref: `docs/architecture/repo-architecture.md` §3.8

## Status

Not yet started.

## Scope

Progress signal via relative metrics: head-to-head win-rate vs prior checkpoints (Elo ladder), and later approximate/local best-response. Exact exploitability is infeasible for Pokémon.

> **Dependency now satisfied.** The checkpoint loader Evaluation needs to play frozen prior nets is built: `checkpoint.load_checkpoint(path) -> LoadedCheckpoint` (rebuilds the `CVPN` from its stored config + weights) in `src/checkpoint.py`. It lives in a neutral module specifically so Evaluation can load checkpoints without importing the Trainer (`docs/plans/trainer.md` §3). Greedy play (τ→0; `self-play.md` §3) is used for measurement, not the τ=1 data-generation sampling.

## Phase-1 curriculum metric (from the SelfPlay design)

> Added from the SelfPlay grilling — see `docs/plans/self-play.md` §6.2 and `docs/vibes-decisions.md` §8.8.

Phase-1 feasibility is demonstrated against a **validation curriculum** of fixed, hand-crafted
matchups of increasing complexity (Stage 0 = all-Fire vs all-Grass / 2 moves … Stage 3 = two
balanced teams). Evaluation owns the **success metric** for each stage:

- **Favored-team win-rate on a fixed matchup.** On a stage with a known-favored team, a trained
  agent driving *both* sides should make the favored side win **far more often** than the
  underdog — and on Stage 0–1 the play should match the humanly-verifiable correct strategy
  (exploit the type advantage). Operationalize "far more often" as a win-rate threshold per stage.
- This complements head-to-head win-rate vs prior checkpoints (the agent self-improves into beating
  earlier versions of itself) — both are relative metrics; neither needs exploitability.
- Greedy play (τ→0; `self-play.md` §3) is used here for measurement, **not** the τ=1 sampling used
  during data generation.
