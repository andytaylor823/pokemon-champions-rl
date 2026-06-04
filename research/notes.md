# Research Notes

## Algorithms Under Consideration

### PPO (Proximal Policy Optimization)
- **What**: Policy gradient method with clipped objective for stable training
- **Role**: Phase 1 workhorse; actor-critic that maps observations → actions
- **Strengths**: Handles imperfect info naturally (maps observations, not
  full state), stable, well-supported in libraries
- **Weaknesses**: No explicit planning / lookahead — learns reactive
  patterns only. Can struggle with multi-turn tactical reasoning
  (stalling Tailwind, counting field turns) without massive training
- **Key paper**: Schulman et al., 2017 — https://arxiv.org/abs/1707.06347

### AlphaZero (+ adaptations for imperfect info)
- **What**: Neural network + MCTS self-play loop
- **Role**: Phase 2 — adds multi-turn planning via search
- **Problem**: Designed for perfect information. Naive application to
  Pokemon causes **strategy fusion** — simulated opponent "knows" your
  hidden items/moves, corrupting the search tree. Overestimates threats,
  underestimates bluffs.
- **Solution**: Information Set MCTS (determinize opponent state from
  domain priors, run MCTS, average across samples) or Player of Games
  approach (public belief states + counterfactual regret minimization)
- **Key paper (AlphaZero)**: Silver et al., 2018 — https://arxiv.org/abs/1712.01815
- **Key paper (Player of Games)**: Schmid et al., 2021 — https://arxiv.org/abs/2112.03178

### R-NaD (Regularized Nash Dynamics)
- **What**: RL algorithm that converges toward Nash equilibrium instead of
  a fixed exploitable strategy
- **Role**: Phase 3 — harden the agent against exploitation
- **Why it matters**: PPO self-play can collapse to one strategy. On
  ladder, even though individual opponents can't "learn" you, the
  metagame can. R-NaD produces mixed strategies (randomized leads, varied
  play) that are robust against any opponent.
- **Key paper**: Perolat et al., 2022 — https://arxiv.org/abs/2206.15378
- **Also see**: DeepNash (Stratego) — same authors

## The Imperfect Information Problem (Detail)

### Strategy Fusion
When MCTS rolls out a game with full state visibility, the simulated
opponent benefits from information they shouldn't have. Example:

- Your Garchomp has Focus Sash (hidden). In MCTS rollout, the opponent
  "sees" the Sash trigger and optimally double-targets. Real opponent
  would be surprised. The backed-up value at the root is corrupted.

This affects both directions:
- Opponent "knows" your hidden info → overestimates threats to you
- You "know" the sampled opponent set → underestimates your uncertainty

### Information Set MCTS (Determinization)
Sample plausible opponent states from meta priors, run MCTS as if
perfect info, average results. Domain expertise directly improves the
quality of the prior distribution.

### Player of Games Approach
Uses public belief states (what's common knowledge) and counterfactual
regret minimization at search nodes. Avoids strategy fusion by reasoning
over information sets rather than specific states.

## Simulation Engine Options

### Pokemon Showdown Sim (@pkmn/sim) — PREFERRED for rollouts
- TypeScript package, extracted from Showdown's source
- `Battle.toJSON()` / `Battle.fromJSON()` confirmed working for
  state save/restore (confirmed by Zarel, Showdown creator)
- Perfectly correct: IS the official battle engine
- Supports doubles, VGC, all gens
- npm: https://www.npmjs.com/package/@pkmn/sim
- GitHub issue on cloning: smogon/pokemon-showdown#8105

### poke-engine (pmariglia) — NOT suitable
- Rust engine with Python bindings, built for searching Pokemon states
- Has MCTS, expectiminimax, state serialization, apply/undo
- **Singles only, gens 4-8. No doubles/VGC. Not fully correct.**
- Useful as architectural reference, not as our engine
- https://github.com/pmariglia/poke-engine

### poke-env — For real games only
- Python ↔ Showdown WebSocket. Good for playing on ladder / self-play.
- Cannot save/restore battle states (issue #190, never implemented)
- Use for: PPO training (Phase 1), playing real games
- NOT usable for MCTS rollouts
- https://github.com/hsahovic/poke-env

### Planned Architecture
- **Real games**: poke-env → Showdown server (Python)
- **MCTS rollouts**: @pkmn/sim in Node.js, called from Python
- **RL training**: PyTorch (Python)
- MCTS search loop should run in JS to avoid Python↔JS bridge
  per state clone. Cross to Python only for neural net leaf evaluation.

## Data Sources for Priors

### Smogon Usage Stats (automated, recurring)
- https://www.smogon.com/stats/ — monthly dumps
- Move usage, item usage, ability usage, teammate correlations
- Stratified by rating — can filter to high-elo only
- Gives marginal probabilities (e.g. "72% of Incineroar carry Fake Out")
  but not full 4-move set distributions

### Showdown Replay Ladder (semi-automated)
- Users upload replays; can filter by elo
- Pro: regular stream of current data, follows meta trends
- Con: can't see full sets (only moves/items revealed in battle),
  bias toward same uploaders

### Tournament Team Sheets (manual, POC first)
- Community tournaments collect full team lists from all entrants
- Pro: complete sets (moves, items, EVs, abilities, tera types)
- Con: inconsistent cadence, spread across organizers
- **Starting here for POC** — one tournament's worth of full team data

## VGC-Specific Observations

- Action space per turn: ~100 joint actions (2 Pokemon × ~10 actions
  each, minus illegal combos). Very manageable.
- Team preview: C(6,4) × lead orderings. Small combinatorial space.
- Turn count: Games typically 6-12 turns. Short horizon = search is
  feasible.
- Field conditions (weather, terrain, Tailwind, screens) are turn-counted
  and hugely impactful. Explicit planning over these is a major advantage
  of search-based approaches over pure PPO.
