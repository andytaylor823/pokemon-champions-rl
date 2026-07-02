---
alwaysApply: false
paths: src/state_types.py
---

# Encoding Pipeline — StateView to ObsBundle

Four modules form the encoding pipeline between SimClient and the CVPN:

| Module | Role |
|--------|------|
| `state_types.py` | Pydantic wire-format models (validated on every SimClient response) |
| `vocab.py` | Stable integer IDs for categorical embeddings (species, items, moves, abilities, natures) |
| `encoder.py` | Translates a `StateView` + perspective into an `ObsBundle` tensor bundle |
| `obs_bundle.py` | TensorDict contract between encoder and CVPN; factory + collation |

## Data flow

```
SimClient.view(handle) → StateView (Pydantic-validated)
                              ↓
                    encoder.encode(view, perspective, belief?)
                              ↓
                         ObsBundle (TensorDict)
                              ↓
                        CVPN.forward(obs)
```

## Entry point

```python
encode(view: StateView, perspective: str, belief: dict | None = None) → ObsBundle
```

- `perspective`: `"p1"` or `"p2"` — reorders sides so the acting player's team comes first.
- `belief`: Phase 4 placeholder (always `None` in Phase 1).

## Vocab singletons

`SPECIES_VOCAB`, `ITEM_VOCAB`, `MOVE_VOCAB`, `ABILITY_VOCAB`, `NATURE_VOCAB` — loaded once at import from `data/legal/`. Index 0 is always reserved for UNK/NONE. Keys are Showdown-style lowercase IDs (no spaces, no hyphens).

## Phase 1 pins (current)

- **Omniscient view** — both sides' full state visible (no hidden info).
- **12 fixed entity tokens** — 6 per side, no variable-length opponent candidates.
- **All `belief_weight = 1.0`** — no uncertainty weighting.

## Phase 4 extensions (designed-in, not yet active)

- Variable-length opponent candidate tokens from the BeliefModel.
- `belief_weight < 1.0` for uncertain candidates.
- Encoder signature (`belief` param) already accommodates this.

## Dimension constants (sentinel-derived, never hard-coded)

`ENTITY_FEATURE_DIM`, `FIELD_FEATURE_DIM`, `SIDE_FEATURE_DIM`, `SCALAR_FEATURE_DIM` — computed at module load from sentinel encoding calls. `CVPNConfig` reads these directly.

See `tensor-contract.mdc` (ObsBundle schema), `state-types.mdc` (wire model reference), `docs/architecture/state-encoding.md`.
