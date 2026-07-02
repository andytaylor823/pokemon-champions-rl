# Project Overview — Pokemon Champions RL Agent

## Architecture: GT-CFR Self-Play with Neural Guidance

The agent uses **GT-CFR** (Game-Theoretic CFR, a.k.a. Player of Games) guided by the **CVPN** (Counterfactual Value + Policy Network), trained via self-play:

1. **CVPN** with two heads:
   - **Policy head**: outputs a prior over the joint action space (guides PUCT expansion in search)
   - **Value head**: predicts the expected game value (scalar in Phase 1; vector of CFVs in Phase 4)
2. **GT-CFR search** runs at every decision point: builds an ephemeral tree, interleaves PUCT expansion with CFR+ regret updates, and produces a mixed average strategy superior to the raw policy prior.
3. **Training loop**: self-play games generate `(state, average_strategy, search_value)` tuples. The CVPN is trained via supervised learning to match Search's improved strategy and value estimates.

The network improves → search produces better strategies → generates better training data → network improves further.

### Why GT-CFR, Not MCTS

Naive MCTS fails in Pokemon due to **strategy fusion** — even with perfect information, simultaneous moves mean neither player sees the other's choice before committing. Each turn is a matrix game whose solution is a **mixed** average strategy, not a single best move. This is exactly the regime CFR+ handles and minimax MCTS does not.

The project uses **GT-CFR / Player of Games** directly — the MCTS pit-stop named in early milestones was skipped because:
- Simultaneous moves make even perfect-info Pokemon a genuine imperfect-information game (step thinking has no pure-strategy solution)
- The CVPN and reference implementations (poker toys) are already GT-CFR-shaped; building MCTS first would be throwaway work

Determinized MCTS (sampling opponent states and running independent searches) is explicitly **not** used — it suffers from strategy fusion and produces exploitable strategies.

### Nash Hardening (Future)

Once the agent is strong, push toward unexploitable mixed strategies using ideas from R-NaD (Regularized Nash Dynamics), so the agent is robust against both strong and weak ladder opponents.

## Milestones

1. **Environment**: connect to Pokemon Showdown sim, encode battle state as tensors, mask illegal actions
2. **State cloning**: `pokemon-showdown` engine integration with `State.serializeBattle()` / `State.deserializeBattle()` for mid-battle forking
3. **GT-CFR search**: search over cloned states using CFR+ with PUCT expansion, producing mixed average strategies at each decision point
4. **CVPN**: policy + value heads, called from search for leaf evaluation and expansion priors
5. **Self-play training loop**: search generates training data, CVPN learns, repeat
6. **Imperfect info (the north star)**: un-pin the Phase-1 degenerate values — belief ranges over opponent private state, MCCFR world sampling, vector CFV head, meta priors supplying candidate sets
7. **Nash hardening**: mixed strategies, robust to exploitation

> **Sequencing philosophy (rebalanced toward shipping).** Phase 1 builds GT-CFR directly in the perfect-information regime — the degenerate case of the Phase-4 north star (same algorithm, same network, with imperfect-information machinery pinned to trivial values). The goal is to (1) prove the infrastructure works end to end (simulate games, encode state, mask actions, search, emit training tuples) and (2) show feasibility — a from-random agent self-improving into a semi-coherent battler that **reliably beats earlier versions of itself** (even while "cheating" with perfect info). The upgrade to Phase 4 is un-pinning those values, not a rewrite. See `docs/architecture/gt-cfr-theory.md` §14.

## Action Space (Per Turn in Doubles)

Each active Pokemon can:
- Use one of up to 4 moves, targeting either opponent slot (or ally, for certain moves)
- Switch to one of the Pokemon in the back (up to 2 back slots)

Both active Pokemon choose simultaneously, and both players submit choices simultaneously. Joint action space per turn is roughly **~100 combinations** (up to ~10 options per Pokemon, combined for 2 active Pokemon, minus illegal pairs like both switching to the same back slot).

## Imperfect Information

During a battle, the agent cannot see:
- Opponent's full movesets (only moves revealed by usage)
- Opponent's held items (until revealed by effect)
- Opponent's stat point distribution
- Opponent's ability (sometimes — some reveal on entry)
- Which 4 Pokemon the opponent selected from their 6 (known only as they appear)

## Simulation Engine

- `pokemon-showdown` (full Smogon repo, TypeScript) is the battle engine — supports state save/restore via `State.serializeBattle()` / `State.deserializeBattle()` (see `showdown-engine-api.md`)
- The `SimClient` seam (`src/sim_client.py` ↔ `sim/src/sim-worker.ts`) is the only boundary between Python and the engine
- Search/training Python code will call into the engine via SimClient's clone/step/view protocol
- **Do NOT reimplement battle rules** — always use Showdown's engine as source of truth

See `agent/theory-reference.md` (algorithmic theory pointer), `agent/paper-index.md` (Player of Games paper reference).

## Python Coding Conventions

- **Prefer `@dataclass(frozen=True)` over Pydantic `BaseModel`** for internal data structures. Pydantic's runtime type-checking adds measurable overhead on hot paths (search nodes, mask construction, tensor encoding). Reserve Pydantic for **wire boundaries** where untrusted external data needs validation (e.g. `SimClient` parsing raw JSON from the worker, `StepResult`, `StateView`). For everything else — action models, internal structs, config objects — use frozen dataclasses or `NamedTuple`.
- This notably disagrees with the user's personal cursor rule to prefer BaseModels over dataclasses. This repo-specific rule should be considered to override the personal rule.
