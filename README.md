# Pokemon VGC RL Agent

A reinforcement learning agent for competitive Pokemon VGC (Video Game Championships) double battles.

## Format

- **VGC Rules**: Bring 6, choose 4, double battle
- **Ladder-style**: Single game per opponent, no adaptation across games

## Architecture (Planned)

### Phase 1 — PPO with Self-Play
Get a working agent that plays VGC via Pokemon Showdown. Learn the
plumbing: state encoding, action masking, reward shaping, environment
integration.

### Phase 2 — Search (AlphaZero-style with Determinization)
Layer on multi-turn lookahead. Use domain priors for opponent set
prediction (Information Set MCTS / Player of Games approach) to handle
imperfect information without strategy fusion.

### Phase 3 — Nash Hardening (R-NaD ideas)
Push the agent toward mixed/unexploitable strategies so it's robust
against both strong and weak opponents on ladder.

## Key Challenges

1. **Imperfect information** — hidden opponent movesets, items, EVs,
   abilities. Naive MCTS leaks information via strategy fusion.
2. **Simultaneous moves** — both players select actions at the same
   time; game tree branches on both players' choices each turn.
3. **Team preview** — choosing 4 from 6 is a combinatorial selection
   problem (separate policy or heuristic).
4. **Reward shaping** — win/loss is sparse; intermediate signals
   (damage, KOs, field control) likely needed early on.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
