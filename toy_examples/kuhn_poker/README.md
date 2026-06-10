# GT-CFR on Kuhn Poker

A pedagogical implementation of **Growing-Tree Counterfactual Regret Minimization** (GT-CFR / Player of Games) applied to Kuhn Poker — the smallest non-trivial imperfect-information game.

The goal is to watch a neural network learn to play near-optimally through self-play, starting from zero game knowledge. Despite Kuhn Poker being solved (Nash equilibrium is known), we treat it as an unknown game and let the algorithm discover the equilibrium on its own.

## What This Demonstrates

1. **The GT-CFR inner loop**: at each decision point, build an ephemeral search tree over the public action space, run CFR+ iterations over all consistent private states, and produce a refined strategy + counterfactual values.

2. **The self-play outer loop**: play games using the search, collect training data, and teach the neural network to approximate the search output — creating the virtuous cycle where a better network leads to better search leads to better data.

3. **Measurable progress**: exploitability (distance from Nash equilibrium) drops from ~0.93 (random play) to <0.2 within 10–20 generations, and continues toward zero with more compute.

## Architecture

```
game.py             ← Game rules (single source of truth for both state class and search)
network.py          ← CVPN: policy head (action priors) + value head (leaf CFVs)
gt_cfr_search.py    ← Inner-loop GT-CFR search (CFR+ on an ephemeral tree)
self_play.py        ← Outer-loop trainer (self-play → replay buffer → NN updates)
exploitability.py   ← Exact exploitability computation via best-response
visualize.py        ← Matplotlib plots (exploitability curve, strategy evolution, etc.)
run.py              ← CLI entry point
```

### Design Decisions

- **`game.py` exposes two layers**: pure-history functions (`is_terminal`, `current_player`, `legal_actions`, `terminal_utility`) that operate on raw action tuples, and `KuhnState` — an immutable wrapper for full-game simulation. The search module imports the pure functions directly; the self-play module uses `KuhnState` for convenience.

- **The search tree is card-agnostic**: nodes are indexed by public action history only. The CFR traversal iterates over all 6 possible deals at each iteration, accumulating regrets per info set (card + history). This prevents strategy fusion — the search never pretends to know hidden information.

- **The network encodes only public information**: the feature vector contains the action history and acting player, not the private card. The output is vectorized over all 3 possible cards the player could hold (policy: [3, 2] logits; value: [3] CFVs).

- **Exploitability measures the raw network policy** (no search), so the curve reflects actual learning rather than just the search algorithm's inherent convergence.

## Quick Start

```bash
# From the project root:
python -m toy_examples.kuhn_poker.run --generations 20 --games-per-gen 100 --seed 42

# Full run with tournament and plots:
python -m toy_examples.kuhn_poker.run --generations 80 --games-per-gen 300 --tournament --verbose

# Quick sanity check:
python -m toy_examples.kuhn_poker.run --generations 5 --games-per-gen 30 --no-plots
```

Plots are saved to `toy_examples/kuhn_poker/output/` by default.

## Key Hyperparameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `--generations` | 50 | Outer loop iterations |
| `--games-per-gen` | 200 | Self-play games per generation (×2–3 training tuples each) |
| `--search-iterations` | 100 | CFR+ iterations per search (higher = closer to Nash per decision) |
| `--lr` | 1e-3 | Adam learning rate |
| `--batch-size` | 64 | Minibatch size from replay buffer |
| `--train-steps` | 100 | Gradient steps per generation |
| `--eval-interval` | 5 | Measure exploitability every N generations |

## Kuhn Poker Rules

- **Deck**: {J, Q, K} (3 cards, J < Q < K)
- **Setup**: both players ante 1 chip, each dealt one card
- **Player 1** acts first: **bet** (raise 1 chip) or **check**
- **Player 2** responds:
  - If P1 bet: **call** (match the bet) or **fold** (forfeit ante)
  - If P1 checked: **bet** or **check**
- If P2 bets after P1 checked, P1 gets a final decision: **call** or **fold**
- **Showdown**: higher card wins the pot

The Nash equilibrium has exploitability 0 and game value −1/18 for Player 1.

## Relationship to the Pokemon VGC Agent

This toy example validates the same algorithmic machinery that will power the full Pokemon agent (see `docs/gt-cfr-theory.md`):

- **CVPN architecture** → scales to a Transformer for Pokemon state encoding
- **Ephemeral search tree + CFR+** → same algorithm, larger action space
- **Self-play training loop** → same outer loop, more sophisticated data pipeline
- **Exploitability measurement** → replaced by Elo / head-to-head win rate at scale

The key difference at scale: Pokemon has too many info sets to fully expand the tree, so the agent relies on incremental PUCT expansion and the network's policy prior to focus search on promising branches.
