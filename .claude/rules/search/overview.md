---
alwaysApply: false
paths: src/search/**/*.py
---

# Search Package — GT-CFR inner loop

The `search/` package implements the GT-CFR inner-loop search for one decision point. It builds an ephemeral tree, runs CFR+ interleaved with PUCT-guided expansion, and outputs an average strategy + value. The tree is discarded after every call.

## Entry point

```python
search(view, sim, net, from_handle, config=None) → SearchResult
```

- `view`: Current `StateView` at the decision point.
- `sim`: `SimClient` instance for cloning/stepping.
- `net`: `CVPN` for policy priors and leaf evaluation.
- `from_handle`: Battle handle to clone from (live or existing clone).
- `config`: `SearchConfig` (defaults: `k_actions=6`, `max_chance_children=5`, `expansion_budget=24`, `cfr_iters_per_expansion=10`, `c_puct=2.0`).

## SearchResult (the inner/outer loop boundary)

| Field | Type | Description |
|-------|------|-------------|
| `strategy` | `dict[Side, dict[int, float]]` | Average strategy (sigma-bar) per side — action index → probability |
| `value` | `float` | p1's counterfactual value (negate for p2) |
| `policy_target` | `dict[Side, np.ndarray]` | Full `[A]` vector per side (mass on top-k, zeros elsewhere) |

## The GT-CFR loop

1. **Expand root** — CVPN provides policy priors; top-k actions selected per side; all k x k grid cells stepped through SimClient and leaf-evaluated.
2. **Repeat `expansion_budget` times:**
   - Phase A: Run `cfr_iters_per_expansion` CFR+ updates over the frozen tree.
   - Phase B: One PUCT-guided expansion (widen a ChanceNode or deepen a frontier leaf).
3. **Extract results** — average strategy from `strategy_sum`, value from `tree_value`.

## Forced-node collapse

**Root:** `search()` raises `ValueError` if called on a forced root (every acting side has exactly one legal action). Callers must filter forced decisions before invoking search. The canonical predicate is `forced_actions()` in `action_space.py`.

**In-tree:** During expansion (grid cell eval, ChanceNode widening, frontier deepening), forced decisions are collapsed transparently via `_collapse_forced`. The helper chains through forced actions — stepping with fresh seeds — until reaching a genuine decision, terminal, or non-acting state. Only the final state is CVPN-evaluated. No degenerate TurnNodes with 1×1 grids are created.

## Session lifecycle

`search()` calls `sim.open_search(from_handle)` at entry and `sim.close_search(session)` in a `finally` block. All cloned handles belong to the session and are freed together.

## Submodules

| Module | Responsibility |
|--------|---------------|
| `types.py` | Data containers (`SearchConfig`, `SearchResult`, `TurnNode`, `ChanceNode`, `InfoSet`, `ChanceOutcome`) |
| `core.py` | `search()` orchestration + forced-root guard |
| `cfr.py` | `regret_matching`, `cfr_update_recursive`, `tree_value` |
| `expansion.py` | `expand_turn_node`, `puct_expand_one`, `puct_scores`, `puct_select_cell`, `cvpn_value`, `_collapse_forced` |
| `strategy.py` | `extract_average_strategy`, `build_policy_target` |

## Phase 1 pins

- Belief = delta (one world per chance sample, no MCCFR).
- Scalar CVPN value head only.
- No opponent belief-range sampling.

See `search/tree-invariants.md` (node guarantees), `search/cfr-algorithm--1.md` (update mechanics), `docs/architecture/gt-cfr-theory.md` section 10.2.
