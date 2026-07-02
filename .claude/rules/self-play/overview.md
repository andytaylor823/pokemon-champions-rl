---
alwaysApply: false
paths: src/self_play.py
---

# SelfPlay — Phase-1 outer-loop data generator

`self_play.py` plays full self-play games where every genuine decision is one `search()` call, skips forced decisions, samples from the average strategy, and yields `TrainingTuple`s. It is a pure **consumer/orchestrator**: it reimplements no game rules (SimClient is the sole source of truth) and no search logic (it calls `search()`).

## Entry point

```python
run(net, matchup_source, sim, config=None) -> Iterator[TrainingTuple]
```

A generator that yields one game's worth of tuples at a time. `sim` is caller-owned (SelfPlay never calls `sim.close()`).

## The game loop

```
matchup_source.sample() -> new_battle -> loop:
  empty to_move / phase "none"  -> step with empty choices
  forced decision               -> step immediately (no search, no CVPN, no tuple)
  genuine decision              -> search() -> buffer tuple(s) -> sample from σ̄ -> step
-> terminal: stamp z on all pending tuples, yield, release handle
```

### Forced-decision skip

Every acting side has exactly one legal action (e.g. forced switch to your only remaining Pokemon, Choice-locked slot with one target). Detected by `_is_forced()` on pre-computed masks. Chains of forced decisions auto-resolve in a loop. **No search, no CVPN forward, no tuple emitted.** This is distinct from the in-tree `_collapse_forced` in `expansion.py` which collapses forced nodes during search.

### Genuine decisions

1. `result = search(view, sim, net, from_handle=handle, config=search_config)`
2. Buffer one `_PendingTuple` per side with >=2 legal actions (see `self-play/training-tuple.md`)
3. Sample one action per side from `result.strategy[side]` via `_sample_action` (temperature-scaled)
4. Translate to choice strings via `action_to_choice_contextual`
5. Advance with `_advance` (step + release old handle)

## Handle lifecycle

SelfPlay owns the **live** handle (the real, advancing game). Each `step` returns a new child handle; the old is `release()`d immediately via `_advance`. Search opens its own session from `from_handle` and closes it internally — it never touches the live timeline.

## SimError retry

`_sample_and_step` retries up to `_MAX_STEP_RETRIES` (10) on `SimError`. Each retry **resamples fresh** from sigma-bar — no per-side or per-joint blacklist. The RNG naturally produces different joints. If all retries fail, the game is aborted and all pending tuples discarded.

## Termination and abort

- **Terminal:** stamp z from `view.utility` on all pending tuples, yield, release handle.
- **`max_decisions` exceeded:** abort, discard all pending tuples (never fabricate an outcome).
- **SimError from `new_battle`:** abort that game, continue to the next.
- **Empty `to_move` / phase `"none"`:** step with empty choices (engine phase transition).

## Seeding and reproducibility

Master seed (`SelfPlayConfig.master_seed`) -> master RNG -> per-game seed -> per-game RNG. Each step seed, battle seed, and action sample draws from the per-game RNG. Same master seed + same net = same games.

## MatchupSource protocol

```python
class MatchupSource(Protocol):
    def sample(self, rng: random.Random) -> tuple[list[dict], list[dict]]: ...
```

Returns `(team_a, team_b)` as `list[dict]` for `new_battle`. `CurriculumMatchupSource` returns a fixed pair.

## SelfPlayConfig

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `temperature` | `float` | 1.0 | Action sampling temperature (1.0 = exact sigma-bar; ->0 = greedy) |
| `max_decisions` | `int` | 500 | Safety cap; abort + discard on exceed |
| `master_seed` | `int` | 42 | Top-level reproducibility seed |
| `num_games` | `int \| None` | None | None = infinite |
| `search_config` | `SearchConfig \| None` | None | Passed to `search()` |
| `generation` | `int` | 0 | Checkpoint generation tag for `TupleMeta` |

See `self-play/training-tuple.md` (data contract), `self-play/curriculum.md` (validation matchups), `search/overview.md` (inner loop), `action-space/overview.md` (`forced_actions`, `action_to_choice_contextual`), `docs/plans/self-play.md`.
