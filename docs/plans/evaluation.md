---
last_synced: a39ccc9
watches:
  - src/evaluation.py
  - docs/architecture/repo-architecture.md
---

# Evaluation — Implementation Plan

> Module ref: `docs/architecture/repo-architecture.md` §3.8

## Status

**Built.** `src/evaluation.py` + `scripts/evaluate.py` landed in the same PR that
completed Phase-1 data generation.  See §Verification for the manual end-to-end
validation plan.

## Scope

Phase-1 progress signal: two legibility metrics (favored-team win-rate, win speed)
tracked across generations via a single-net curriculum run, plus a secondary
head-to-head tool for comparing two different checkpoints.  Full Elo ladder and
approximate/local best-response are deferred.

## Architecture

`src/evaluation.py` is a **read-only measurement module** — it owns no SimClient
lifecycle, reimplements no game rules, and reimplements no search logic.  It is a
pure orchestrator over public seams that are already used by SelfPlay and Search.

### Agent abstraction

Three implementations of the ``Agent`` protocol (all in `src/evaluation.py`):

| Agent | Net? | Search? | Notes |
|---|---|---|---|
| `SearchAgent(net, config)` | ✓ | ✓ GT-CFR | 1-slot handle cache: one `search()` per turn when used on both sides |
| `PolicyAgent(net)` | ✓ | ✗ | Single forward pass; fast mode; CVPN masks illegal logits to `-inf` |
| `RandomAgent(rng)` | ✗ | ✗ | Uniform over legal mask; random-move floor |

The **1-slot handle cache** on `SearchAgent` is the key design: when the *same*
agent object is used for both p1 and p2 (curriculum mode), the first `act(handle=h)`
call runs search and caches the `SearchResult`; the second call (other side, same
handle, before stepping) reuses it.  Effect: **one search per turn** for the
headline curriculum metric.  Two distinct agent objects (head-to-head) each run
their own independent search — two searches per turn.

### Game runner: `play_game`

Mirrors `self_play.run`'s battle loop (lines 262–354) minus all
tuple/z buffering:

1. Start battle with `sim.new_battle(team_a, team_b, seed=battle_seed)`.
2. Loop while `not view.terminal and decision_idx < max_decisions`:
   - **Empty phase** (`not view.to_move or view.phase == "none"`): step `{}`.
   - **Forced decision** (`action_space.forced_actions(...)` is not None):
     auto-step without calling any agent; do NOT increment `decision_idx`.
   - **Genuine decision**: call agents, build joint choice, step, increment counter.
3. Classify terminal from `view.utility["p1"]`: `>0 → p1_win`, `<0 → p2_win`,
   `0 / None → draw`.
4. **Abort paths** (always `sim.release(handle)` before returning):
   - `max_decisions` hit without terminal.
   - `SimError` on `sim.step` (greedy agents don't retry — would pick the same move).
   - Agent raises (empty strategy / no legal actions → `"no_action"`).

### Reports (frozen dataclasses — pure data)

`CurriculumReport`:
- `favored_win_rate` = `favored_wins / (favored_wins + underdog_wins + draws)`.
  Aborted games excluded from every denominator; count surfaced via `aborted`.
- `mean_favored_turns`, `median_favored_turns` over favored-side wins.
- `overall_mean_turns` over all decided games.

`HeadToHeadReport`: same structure for `a_win_rate`, etc.

### Config: `EvalConfig`

Lighter eval search budget than training default by design — comparisons must use
the same budget for both checkpoints.

```python
EvalConfig(
    n_games=50,
    max_decisions=300,
    master_seed=0,
    use_search=True,           # False → PolicyAgent fast mode
    favored_side="p1",         # curriculum: team_a (Fire) is p1
    eval_search_config=SearchConfig(
        k_actions=6, max_chance_children=3,
        expansion_budget=8, cfr_iters_per_expansion=5, c_puct=2.0),
    device="cpu",
)
```

### Entry points

```python
curriculum_report(net, matchup, sim, *, config) -> CurriculumReport
head_to_head(agent_a, agent_b, matchup, sim, *, config) -> HeadToHeadReport
fitness(checkpoint_path, opponents, matchup, sim, *, config) -> dict[label, HeadToHeadReport]
```

`fitness` satisfies the `fitness(checkpoint) -> metric` shape from
`docs/architecture/repo-architecture.md` §3.8.

## Seeding & determinism

`master_seed` seeds a `random.Random` that emits per-game seeds.  Inside
`play_game`, a fresh `random.Random(game_seed)` governs the battle seed and all
per-step reseeds — identical to `self_play.run`'s seeding chain.

`SearchAgent` eval is **not** bit-reproducible even at a fixed seed because
`search`'s internal expansion RNG is global (`src/search/expansion.py:77-79`).
`PolicyAgent` mode IS deterministic given the battle seeds.  Both are acceptable;
the deliverable is a statistical trend, not a bit-exact number.

## Decisions locked in this session (see docs/vibes-decisions.md §13)

- **Scope:** curriculum metric + head-to-head + random floor behind one runner + CLI. (Q1)
- **Head-to-head mechanic:** two independent searches per turn, each net plays only its
  own side, **no side-swap**.  Results stay legible without swap. (Q2)
- **Eval speed:** configurable lighter budget + policy-only fast mode; **decoupled from
  training** — `self_play.run` keeps its full search_config untouched. (Q3)
- **Win speed scoring:** `snapshot.turn` at terminal; mean + median over favored wins +
  overall mean.  Draws counted separately, excluded from win-rate.  Aborts excluded from
  denominator, reported as health count. (Q4)
- **Entry point:** standalone `src/evaluation.py` + `scripts/evaluate.py`.
  `train_loop.py` unchanged.  Full Elo ladder deferred. (Q5)

## Phase-1 curriculum metric (unchanged from original plan)

Phase-1 feasibility is demonstrated against a **validation curriculum** of fixed,
hand-crafted matchups of increasing complexity (Stage 0 = all-Fire vs all-Grass / 2
moves … Stage 3 = two balanced teams).  Evaluation owns the **success metric**:

- **Favored-team win-rate on a fixed matchup.** On Stage 0 a trained agent driving
  *both* sides should make the favored side (Fire) win **far more often** than the
  underdog — and the play should match the humanly-verifiable correct strategy (exploit
  the type advantage).
- **Win speed** — turns to terminal should drop as the agent learns to close out games
  efficiently.
- This complements head-to-head win-rate vs prior checkpoints (self-improvement into
  beating earlier versions).
- Greedy play (τ→0) is used for measurement, **not** the τ=1 sampling used during
  data generation.

## Verification

### 1. Unit tests (`tests/unit/test_evaluation.py`, no worker)
- `RandomAgent` only returns legal indices; distribution roughly uniform.
- `SearchAgent`/`PolicyAgent` greedy argmax picks max-prob index, lowest index on ties.
- Handle cache: same handle → one `search()` call; new handle → new call.
- Report math: `favored_win_rate` excludes draws+aborts from denominator; mean/median
  only over favored wins; aborted never enters a rate.
- Outcome classification from `utility`: `>0→p1_win`, `<0→p2_win`, `0/None→draw`.
- Abort paths: `max_decisions`, `sim_error`, `no_action`.

### 2. Integration tests (`tests/integration/test_evaluation.py`, `@pytest.mark.slow`)
- `curriculum_report` returns with counts summing to `n_games`; `0 ≤ favored_win_rate ≤ 1`.
- `head_to_head(SearchAgent, RandomAgent)` completes with counts summing.
- `use_search=False` (PolicyAgent) completes and is faster (smoke only).

### 3. End-to-end manual (empirical-validation preference)
```bash
# Train a few generations
python scripts/train.py --stage 0 --generations 10 --games-per-gen 8 \
    --train-steps 50 --checkpoint-dir /tmp/eval_test

# Evaluate all checkpoints
python scripts/evaluate.py \
    --checkpoint-dir /tmp/eval_test --stage 0 --n-games 30 \
    --out eval.csv

# Fast policy-only pass
python scripts/evaluate.py --checkpoint-dir /tmp/eval_test --policy-only --n-games 10

# With random floor
python scripts/evaluate.py --checkpoint-dir /tmp/eval_test --vs-random --n-games 20
```

Expect `favored_win_rate` to trend upward and `mean_favored_turns` to trend downward
across generations as the improvement signal.

## Files

| File | Role |
|---|---|
| `src/evaluation.py` | Library: Agent protocol + three implementations; `GameResult`; `play_game`; `CurriculumReport`, `HeadToHeadReport`, `EvalConfig`; `curriculum_report`, `head_to_head`, `fitness`. |
| `scripts/evaluate.py` | CLI: scans a checkpoint dir → per-generation table + CSV/JSON. |
| `tests/unit/test_evaluation.py` | Pure-logic unit tests (no Node worker). |
| `tests/integration/test_evaluation.py` | `@pytest.mark.slow` integration tests with real `sim_client` fixture. |

## Out of scope

- Full Elo/rating ladder and persistence (reports are structured so ratings can be
  computed later from the raw win/loss counts).
- Wiring eval into `train_loop.py` (eval runs offline; driver stays thin).
- Approximate/local best-response and Phase-4 belief/vector-head metrics.
- Side-swapping in head-to-head (explicitly declined — keeps results legible).
