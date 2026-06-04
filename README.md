# Pokemon Champions RL Agent

A reinforcement learning agent for competitive Pokemon Champions
(Switch, 2026) — the official VGC platform for the 2026 World
Championships.

## Target Format

- **Game**: Pokemon Champions (Regulation M-A Doubles)
- **Rules**: Bring 6, pick 4, double battle, level 50
- **Dex**: ~195 fully-evolved Pokemon, 59 Mega Evolutions
- **Gimmick**: Mega Evolution only (Tera not available in Reg M-A)
- **Stat system**: No IVs (all 31), Stat Points (66 total, 32 max/stat)
- **Ladder-style**: Single game per opponent, no adaptation across games

## Architecture

AlphaZero-style self-play with MCTS, adapted for imperfect information:

1. **Neural network** (policy + value heads) guides MCTS at each decision
2. **MCTS** searches over cloned battle states via `@pkmn/sim`
3. **Self-play** generates training data; network learns to match MCTS's improved policy
4. **Determinization** handles imperfect info — sample opponent sets from meta priors
5. **Nash hardening** (future) — push toward unexploitable mixed strategies via R-NaD ideas

## Key Challenges

1. **Imperfect information** — hidden opponent movesets, items, stat
   points, abilities. Naive MCTS leaks info via strategy fusion.
2. **Simultaneous moves** — both players select actions at the same
   time; game tree branches on both players' choices each turn.
3. **Team preview** — choosing 4 from 6 is a combinatorial selection
   problem (separate policy or heuristic).
4. **Reward shaping** — win/loss is sparse; intermediate signals
   (damage, KOs, field control) likely needed early on.
5. **Mega decisions** — when to Mega and on which Pokemon. One-shot
   irreversible decision with huge impact.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```
