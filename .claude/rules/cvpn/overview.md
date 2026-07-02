---
alwaysApply: false
paths: src/cvpn.py
---

# CVPN — Counterfactual Value + Policy Network

Phase 1 Transformer that consumes an `ObsBundle` and produces policy logits + a scalar value. See `docs/plans/cvpn.md` for the full design spec.

## Architecture

**Token sequence** (input to Transformer):
`[CLS, field, side_p1, side_p2, meta, entity_1, ..., entity_N]`

- 5 fixed global prefix tokens (projected from field/side/scalar features + learnable CLS).
- N entity tokens (12 in Phase 1) — each built from continuous features + embedded species/ability/item/move-summary.

**Backbone**: Pre-norm `TransformerEncoder` (GELU, batch_first, no positional encoding). Defaults: `d_model=128`, `n_heads=4`, `n_layers=3`.

**MoveAttentionPool (D3)**: A single learnable query attends to 4 per-entity move embeddings, producing one `d_move`-width summary. Captures move synergies (e.g. Protect + Fake Out).

## Two heads

| Head | Input | Output | Description |
|------|-------|--------|-------------|
| Policy | CLS token | `[A]` logits | Phase-split: `tp_head` (team preview region) + `mp_head` (move phase region), assembled into one `[A]` vector. Illegal actions masked to `-inf`. |
| Value | CLS token | scalar in `[-1, 1]` | `Linear → tanh`. p1's expected utility. |

## Forward contract

```python
CVPN.forward(obs: ObsBundle) → (policy_logits: Tensor, value: Tensor)
```

- Accepts **unbatched** (`batch_size=[]`) or **batched** (`batch_size=[B]`) ObsBundle.
- `policy_logits`: `[A]` or `[B, A]` — softmax-ready (illegal = `-inf`).
- `value`: scalar `[]` or `[B]` — in `[-1, 1]`.

## CVPNConfig

All dimension fields derive from `encoder` and `action_space` module constants — **never hard-code** vocab sizes, feature dims, or action counts. Embedding table sizes come from `vocab.py` singletons.

## Phase 1 pin

- Scalar value head only (no vector CFV head for per-action counterfactual values).
- Phase 4 adds a vector CFV head (`[k]` values) without changing the forward signature for callers that only need policy + scalar value.

See `encoding/tensor-contract--1.md` (input schema), `encoding/overview--1.md` (pipeline), `action-space/overview.md` (A constant).
