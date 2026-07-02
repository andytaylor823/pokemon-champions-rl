---
alwaysApply: false
paths: src/search/**/*.py
---

# Search Tree Invariants (do not break)

## TurnNode guarantees

- **Both sides always have a valid InfoSet.** After `expand_turn_node`, `node.info["p1"]` and `node.info["p2"]` both exist. The non-acting side in a unilateral decision gets `InfoSet.noop()` (single action, prior = [1.0]). CFR and PUCT code can unconditionally access both without fallback checks.
- **`expanded = True`** means the node's info sets are populated and the grid has been filled (possibly empty if all cells failed). Do not read `info` or `grid` on an unexpanded node.

## InfoSet structure

`InfoSet` holds five parallel arrays of length `k` (top-k actions):
- `actions` — canonical action indices (into the `[A]` space)
- `prior` — CVPN policy prior, normalized over top-k support
- `regret` — CFR+ cumulative regret (clamped non-negative)
- `strategy_sum` — iteration-weighted strategy accumulator
- `visits` — per-action visit counts (for PUCT decay)

Factory methods: `InfoSet.from_actions(actions, prior)`, `InfoSet.noop()`, `InfoSet.empty()`.

## Grid indexing

The `k x k` grid on `TurnNode` is indexed by **position within the top-k array** `(0..k-1)`, NOT canonical action indices. To recover the canonical action index: `node.info[side].actions[pos]`.

## ChanceOutcome wiring

`ChanceOutcome.node` starts as `None` (frontier leaf). When `puct_expand_one` deepens a leaf, it sets `outcome.node = child_TurnNode` **in place**. This is the sole mechanism for tree growth — there is no external node registry or lookup table. CFR+ recurses through `outcome.node` directly.

## Handle and session lifecycle

- Every handle created during search belongs to the session returned by `open_search`.
- `step` is **immutable**: it clones the parent into a new child handle. The parent remains steppable with different choices/seeds.
- `close_search(session)` frees all handles in the session at once. The live battle handle (which `from_handle` points to) is never freed by search.
- If search raises, the `finally` block in `core.py` still calls `close_search` — no handle leaks.

## Mutability rules

- `InfoSet.regret`, `strategy_sum`, `visits` are **mutated in-place** during CFR+ updates and PUCT expansion. These arrays are the mutable working state of the search.
- `TurnNode.grid` entries and `ChanceNode.children` are **append-only** during the expansion budget. Never remove or reorder existing children.
- `ChanceOutcome.leaf_value_p1` is written once at creation and read thereafter.

See `search/overview.md` (package entry point), `search/cfr-algorithm--1.md` (update mechanics).
