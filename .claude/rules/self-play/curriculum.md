---
alwaysApply: false
paths: src/curriculum.py
---

# Validation Curriculum

`curriculum.py` defines hand-crafted fixed matchups for staged Phase-1 correctness proofs. Each stage tests the architecture's ability to discover the known-correct solution at a given complexity level.

## Stages

| Stage | Matchup | What it proves |
|-------|---------|----------------|
| **0** | 6 Fire-types (2 moves each) vs 6 Grass-types (2 moves each) | Smallest action space, obvious type advantage. Fire should win almost always. The architecture can learn *anything*. |
| **1** | Same teams, 4 moves each | Larger per-slot action space; still an obvious type edge. |
| **2** (planned) | Two "normal" teams, one expert-favored | Realistic complexity; favored side should win far more often. |
| **3** (planned) | Two balanced normal teams | Full Phase-1 complexity; converges to sensible mixed play. |

## Module exports

- `STAGE_0_FIRE`, `STAGE_0_GRASS` — Stage 0 team data (`list[dict]`)
- `STAGE_1_FIRE`, `STAGE_1_GRASS` — Stage 1 team data
- `STAGE_0` — pre-built `CurriculumMatchupSource` for Stage 0
- `STAGE_1` — pre-built `CurriculumMatchupSource` for Stage 1

## Team data format

Each team is a `list[dict]` with keys matching what `SimClient.new_battle` expects:

```python
{"species": str, "item": str, "ability": str,
 "nature": str, "statPoints": dict[str, int], "moves": list[str]}
```

Teams share a `_*_BASE` definition (species/item/ability/nature/statPoints) across stages; per-stage move lists are attached via `_with_moves`.

## Per-stage correctness proofs

**From-scratch is the primary proof**: random init -> the architecture discovers the solution at this complexity, with no carryover from prior stages. Each stage is an independent, unconfounded experiment.

**Warm-start from the prior stage** is a side experiment measuring how much the curriculum accelerates learning.

## Adding new stages

1. Define base team dicts (or reuse existing bases).
2. Attach move lists via `_with_moves`.
3. Create a `CurriculumMatchupSource` instance.
4. Validate legality: `echo '<paste>' | python scripts/check_team_legality.py`.

See `docs/plans/self-play.md` §6.2, `self-play/overview.md` (MatchupSource protocol).
