# BeliefModel — Implementation Plan

> Module ref: `docs/architecture/repo-architecture.md` §3.4
> Design refs: `docs/architecture/state-encoding.md` §7,§9; `docs/architecture/search-nn-interface.md` §5,§6

## Status

Legacy clustering pipeline exists (`src/meta_priors/`). Target conditional sampler not started.

## Scope

Supply candidate opponent sets + sample joint deals for Search/MCCFR. Main open problem: smoothing/backoff for sparse conditioning.
