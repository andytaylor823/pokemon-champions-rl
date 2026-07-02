---
alwaysApply: false
paths: src/search/expansion.py
---

# CFR+ and PUCT Algorithm Mechanics

## Regret matching+ (`regret_matching`)

```
sigma(a) = max(0, R(a)) / sum_b max(0, R(b))
```

If all regrets are non-positive (or zero), falls back to **uniform** over all actions. Handles edge cases: empty arrays (return empty), inf-valued regrets (concentrate on inf actions).

This is the CFR+ variant — regrets are clamped to non-negative on write (`np.maximum(0, ...)`) so they never accumulate deeply negative.

## CFR+ recursive update (`cfr_update_recursive`)

Depth-first traversal of the tree. At each `TurnNode`:

1. Compute current strategy for both sides via `regret_matching`.
2. Fill a `cfv_grid[k_p1, k_p2]` — for each grid cell, either recurse into expanded `ChanceOutcome.node` children or use cached `leaf_value_p1`. Chance outcomes are **uniform-averaged**.
3. Marginalize to per-action CFVs: `v_p1 = grid @ sigma_p2`, `v_p2 = -(grid^T @ sigma_p1)`.
4. Compute node value: `sigma_p1 @ v_p1`.
5. **Write regrets**: `R(a) = max(0, R(a) + instant_regret(a))` (CFR+ clamp).
6. **Write strategy_sum**: `strategy_sum += iteration * sigma` (linear iteration-weighting).

Returns p1's counterfactual value at this node.

## Read-only value query (`tree_value`)

Same marginalization as `cfr_update_recursive` but **skips the three write lines** (regret, strategy_sum, visits). Used after the search loop converges to read p1's value without mutating the tree.

## PUCT scoring (`puct_scores`)

$$score(a) = \sigma^t(a) + c_{puct} \cdot P(a) \cdot \frac{\sqrt{\sum_b N(b)}}{1 + N(a)}$$

- `sigma^t(a)` = current regret-matched strategy (exploitation).
- `P(a)` = CVPN policy prior over top-k (exploration guidance).
- `N(a)` = per-action visit count (exploration decay).

`puct_select_cell` takes the outer product of p1 and p2 PUCT scores to pick the highest-scored grid cell.

## PUCT expansion (`puct_expand_one`)

Descends from the root following PUCT-selected cells. At each level, one of three actions:

| Condition | Action |
|-----------|--------|
| ChanceNode has `< max_chance_children` | **Widen** — step with a new seed, add another `ChanceOutcome` |
| ChanceNode is full, has frontier leaves (`outcome.node is None`) | **Deepen** — expand the leaf into a new `TurnNode` |
| ChanceNode is full, all outcomes expanded | **Descend** — recurse into the best expanded child |

Visits are incremented on the selected actions at each level (so PUCT scores decay with exploration). A depth cap (`_MAX_DESCENT_DEPTH = 50`) prevents runaway loops.

Returns `True` if a node was expanded, `False` if the tree is fully saturated.

See `search/overview.md` (loop structure), `search/tree-invariants.md` (node guarantees), `docs/architecture/gt-cfr-theory.md`.
