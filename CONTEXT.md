# Pokemon Champions RL Agent

A reinforcement learning agent for Pokemon Champions VGC doubles, using search + self-play to learn unexploitable strategies under imperfect information.

## Language

### Game Structure

**Info set**:
Everything one player knows at a decision point — the public state plus that player's own private state. You can't tell apart the different histories inside your info set, so you have to play one strategy that handles all of them.
_Avoid_: State, game state (too vague — say which level you mean)
_See_: `docs/architecture/gt-cfr-theory.md` §5.2, §15; `docs/research/article_summary_2.md` "Info Sets Are Not Time-Indexed"

**History**:
The full ground truth of the game at one node — both players' private info, every action and chance outcome so far. A single point in the game tree. "History" and "node" mean the same thing in extensive-form game theory.
_Avoid_: State (ambiguous), game record
_See_: `docs/architecture/gt-cfr-theory.md` §5.2; `docs/research/article_summary_3.md` "Counterfactual Value, Histories, and Utilities"

**Public state**:
Everything both players have observed — revealed Pokemon, revealed moves, items revealed by effect, HP, status, stat stages, field conditions with turn counters, Mega availability. Common knowledge only.
_Avoid_: Board state (this isn't chess)
_See_: `docs/architecture/gt-cfr-theory.md` §13, §15; `docs/architecture/state-encoding.md` §7

**Private state**:
What one player knows that the other doesn't — your unrevealed back-row Pokemon, hidden items, unshown moves, stat-point spreads.
_Avoid_: Hidden state (sounds like RNN state), secret info
_See_: `docs/architecture/gt-cfr-theory.md` §13; `docs/architecture/search-nn-interface.md` §4.2

**Belief range**:
A probability distribution over the opponent's possible private states. Supplied by meta priors, reduced to top-K concrete candidate sets per species. The belief is what connects "I don't know what they have" to "but here's what's likely."
_Avoid_: Belief state (too easily confused with public state)
_See_: `docs/architecture/gt-cfr-theory.md` §13; `docs/architecture/state-encoding.md` §7.4

**Candidate**:
One fully specified concrete set for an opponent Pokemon slot — species, moves, item, ability, stats. Nothing probabilistic inside it; probability lives in the belief weight on the candidate.
_Avoid_: Archetype (being retired — too fuzzy, subjective boundaries), hypothetical set
_See_: `docs/architecture/state-encoding.md` §7.1–7.2; `docs/architecture/search-nn-interface.md` §5.5

**Belief weight**:
The probability that a specific candidate is the real opponent set. A float in [0, 1]; sums to 1 across candidates for a slot.
_Avoid_: Prior (overloaded with policy prior), probability (too generic)
_See_: `docs/architecture/state-encoding.md` §4.1, §7.1

**Support**:
The set of candidates with non-zero belief weight. "Live support" means the support after conditioning on observations — it shrinks as the opponent reveals info.
_Avoid_: Candidate pool, candidate list
_See_: `docs/architecture/state-encoding.md` §7.3

**World**:
One specific assignment of the opponent's hidden information across all slots — which candidate set each opponent Pokemon is running and which 4 they brought. The thing you sample in MCCFR.
_Avoid_: Deal, scenario, sample (too generic)
_See_: `docs/architecture/search-nn-interface.md` §5.4–5.5; `docs/research/article_summary_4.md` "GT-CFR Operationally on a Pokemon Turn"

**Strategy fusion**:
The failure mode of determinized MCTS where the search "cheats" by playing differently in different sampled worlds, as if it knows the hidden info. In each world the search finds the per-world optimum, but you can only play one action — you don't know which world you're in. The backed-up Q values are inflated relative to reality.
_Avoid_: Information leakage (different concept)
_See_: `docs/architecture/gt-cfr-theory.md` §4; `docs/research/article_summary_7.md` "Strategy Fusion — Concrete Example"

### Algorithm Concepts

**Two loops**:
The inner loop (search at one decision) and the outer loop (training across self-play). Almost every confusion about "convergence" or "iteration" dissolves when you ask which loop.
_Avoid_: Training loop (ambiguous — could mean either)
_See_: `docs/architecture/gt-cfr-theory.md` §2; `docs/research/article_summary_6.md` "The Two Loops"

**Inner loop**:
The search at one decision point. Build a tree, run CFR iterations on it, read off a move, throw the whole tree and regret table away. Fixed budget (N iterations or M milliseconds), not full convergence.
_Avoid_: Search loop (OK as a synonym, but "inner loop" is the canonical term)
_See_: `docs/architecture/gt-cfr-theory.md` §2; `docs/architecture/search-nn-interface.md` §1–2

**Outer loop**:
Network training across self-play games. Play thousands of games (each using the inner loop at every decision), collect training targets, update the network. Measured by exploitability or head-to-head win rate.
_Avoid_: Training loop (say "outer loop" to be precise)
_See_: `docs/architecture/gt-cfr-theory.md` §12; `docs/research/article_summary_7.md` "Part 1"

**Regret**:
At one info set over many CFR iterations, how much better you would have done if you'd always picked a particular action instead of what your strategy actually said. Positive regret means you under-used that action.
_Avoid_: Loss (different concept), penalty
_See_: `docs/architecture/gt-cfr-theory.md` §6.1; `docs/research/article_summary.md` "Regret: The Foundational Concept" (RPS worked example)

**Cumulative regret**:
The running total of instantaneous regrets for an action at an info set across CFR iterations. The regret table maps (info set, action) to this scalar.
_Avoid_: Total regret (fine technically but uncommon)
_See_: `docs/architecture/gt-cfr-theory.md` §6.2; `docs/research/article_summary.md` "Regret Matching"

**Regret matching**:
The algorithm that turns regrets into a strategy — play each action with probability proportional to its positive cumulative regret. No tunable exploration constant; the convergence guarantees come from the math.
_Avoid_: Regret minimization (that's the broader goal, not this specific algorithm)
_See_: `docs/architecture/gt-cfr-theory.md` §6.2–6.3; `docs/research/article_summary.md` "Regret Matching" (RPS continuation)

**Regret table**:
A mapping from (info set, action) to cumulative regret. One per player. Ephemeral — built fresh for each inner-loop search and thrown away after.
_Avoid_: Q-table (different thing), value table
_See_: `docs/architecture/gt-cfr-theory.md` §10.4; `docs/research/article_summary_3.md` "What Is 'Strategy' in CFR"

**Counterfactual value (CFV)**:
The expected payoff of taking action a at info set I, weighted across all indistinguishable histories by how likely each is (given everyone else's strategies). The whole weighted sum, not the individual weights.
_Avoid_: Value (too generic — say CFV or counterfactual value), Q-value (MCTS concept)
_See_: `docs/architecture/gt-cfr-theory.md` §7.1 (the trap); `docs/research/article_summary_7.md` "Part 2: The two heads"

**Counterfactual reach probability**:
The probability of arriving at a history if the acting player had tried to get there with certainty — factor out their own contributions, keep chance and opponent. A weight inside the CFV sum, not a value. The network never outputs this.
_Avoid_: Reach probability (ambiguous — could include the player's own), counterfactual value (the reach is a weight, the value is the sum)
_See_: `docs/architecture/gt-cfr-theory.md` §5.3; `docs/research/article_summary_2.md` "Reach Probabilities — Concrete Decomposition" (Kuhn arithmetic)

**Average strategy**:
The weighted average of strategies across all CFR iterations — later iterations weighted more (linear weighting in CFR+). This is the search's output, the thing you sample your move from. The current iterate oscillates; only the average converges to Nash.
_Avoid_: Policy (overloaded), strategy (without "average" — the current iterate is also a strategy)
_See_: `docs/architecture/gt-cfr-theory.md` §10.2; `docs/research/article_summary_5.md` "PUCT for Expansion, Regret Matching for Strategy"

**Nash equilibrium**:
A strategy that is unexploitable — no opponent can profit by deviating. The game value is unique but the set of Nash strategies can be a range (like Kuhn poker's alpha parameter). Different training runs may find different equilibria; they're equally unexploitable but may exploit weak opponents differently.
_Avoid_: Optimal play (implies a single best strategy; Nash is a set)
_See_: `docs/architecture/gt-cfr-theory.md` §5.4; `docs/research/article_summary_3.md` "Variable Nash Equilibrium"

**Step thinking**:
The cycle of best responses — "they'll Protect, so I should Tailwind; but they'll expect Tailwind, so they'll attack; but I expect that, so I'll Protect..." This cycle has no pure-strategy solution. CFR resolves it by converging to a mixed strategy with calibrated randomization.
_Avoid_: Leveling, yomi
_See_: `docs/research/article_summary_5.md` "How GT-CFR Handles Step Thinkers"

### Architecture Components

**CVPN**:
Counterfactual Value-and-Policy Network. One network with a shared backbone and two heads. The GT-CFR analogue of AlphaZero's neural network.
_Avoid_: The network, the model, the NN (too generic when discussing multiple network designs like Deep CFR)
_See_: `docs/architecture/gt-cfr-theory.md` §10.1; `docs/research/article_summary_7.md` "Part 2: The two heads"

**Policy head**:
The CVPN output that predicts a distribution over actions at an info set. Used as the prior P in the PUCT expansion formula. This is the search's input — a cheap initial guess.
_Avoid_: Prior (OK as a synonym but be explicit when distinguishing from the average strategy)
_See_: `docs/architecture/gt-cfr-theory.md` §10.1; `docs/architecture/state-encoding.md` §6.3

**Value head**:
The CVPN output that predicts counterfactual values at search leaves. In GT-CFR this is a vector — one CFV per theoretical world. In AlphaZero's Phase 1 it's a single scalar.
_Avoid_: Evaluation function (implies a handcrafted heuristic)
_See_: `docs/architecture/gt-cfr-theory.md` §10.1, §10.3; `docs/architecture/state-encoding.md` §6.3

**Policy prior vs average strategy**:
Two different distributions over actions that share the word "policy." The prior P is the CVPN's one-shot guess (search input). The average strategy σ-bar is the refined output of the search (what you actually play). Over training generations, P converges toward σ-bar — that's the distillation.
_Avoid_: Conflating these two. If in doubt, say "prior" or "average strategy," never just "policy."
_See_: `docs/research/article_summary_7.md` synonym map at the end; `docs/architecture/gt-cfr-theory.md` §10.1

**Token**:
A fixed-length vector representing one entity (a Pokemon, the field, or the CLS summary). The fundamental input unit of the Transformer. Each token is self-contained — cross-entity reasoning happens through attention, not by stuffing other Pokemon's info into the token.
_Avoid_: Feature vector (technically correct but misses the Transformer-specific meaning)
_See_: `docs/architecture/state-encoding.md` §3, §4

**CLS token**:
A learnable summary vector with no input data. Accumulates a holistic battle summary via attention across all real tokens. Read by the policy head (and value head in Phase 1). Not the random initialization — it's overwritten layer by layer.
_Avoid_: Classification token (misleading — we're not classifying)
_See_: `docs/architecture/state-encoding.md` §6.1–6.2

**School A vs School B**:
Two approaches to hidden information in the network. School A feeds all candidates simultaneously with belief weights and gets a vector of CFVs in one pass. School B feeds one sampled world at a time and gets a scalar. School A is the target for GT-CFR; School B is what Phase 1 does implicitly.
_Avoid_: Using these labels without defining them — they're project-internal jargon
_See_: `docs/architecture/state-encoding.md` §11.4

**Three-tier split**:
The proposed Phase 4 architecture that splits computation into (1) the expensive Transformer backbone run once per node expansion, (2) the policy head run once per expansion, and (3) the cheap value-head MLP run once per sampled world. The backbone dominates cost; the per-world value calls are negligible.
_Avoid_: Three-phase (conflicts with the build phases)
_See_: `docs/architecture/search-nn-interface.md` §7

### Search Mechanics

**GT-CFR**:
Growing-Tree CFR. The search algorithm that grows a tree on the fly (like MCTS) but runs CFR+ updates on it (not UCB backups), producing a mixed strategy sound under imperfect information. The destination algorithm for this project.
_Avoid_: CFR (vanilla CFR traverses the full tree), MCTS (different search paradigm)
_See_: `docs/architecture/gt-cfr-theory.md` §10; `docs/research/article_summary.md` "GT-CFR (Growing-Tree CFR)"

**PUCT**:
Predictor + UCT. The selection formula used during tree expansion that combines exploitation (current CFR strategy from the regret table) with exploration (policy prior + visit-count bonus). Same formula shape as AlphaZero, but the Q-analogue is the regret-derived strategy.
_Avoid_: UCB (PUCT is the specific variant with neural-network priors)
_See_: `docs/architecture/gt-cfr-theory.md` §3.2, §10.2; `docs/research/article_summary_7.md` "The UCB Equation"

**Expansion**:
Growing the search tree by adding new nodes at a chosen leaf and querying the CVPN for values and priors. The CVPN is called once per expansion — that's where the NN cost lives.
_Avoid_: Simulation (MCTS term for a different operation)
_See_: `docs/architecture/search-nn-interface.md` §1–2; `docs/architecture/gt-cfr-theory.md` §10.2

**CFR+ update**:
The regret-update phase of GT-CFR. Traverse the current tree; at each info set compute counterfactual values, update cumulative regrets with non-negative clipping, derive the next iteration's strategy via regret matching, and accumulate the average strategy.
_Avoid_: Backup (MCTS term)
_See_: `docs/architecture/gt-cfr-theory.md` §7.3, §10.2; `docs/research/article_summary_6.md` "The Centerpiece: GT-CFR on Kuhn"

**Chance bucketing**:
Discretizing the near-continuous damage/accuracy/crit outcomes into a small number of representative buckets (~3-5) per chance node. Within a non-crit damage range, 68% HP vs 65% HP almost never flips what you should do; the critical boundaries are KO thresholds and status effects.
_Avoid_: Chance abstraction (same idea, less intuitive)
_See_: `docs/architecture/search-nn-interface.md` §8.1

**MCCFR**:
Monte Carlo CFR. Instead of enumerating all opponent configurations per CFR iteration, sample a world from the belief distribution and do the traversal on that one concrete world. The regret estimate is unbiased; just noisier.
_Avoid_: Monte Carlo sampling (too generic)
_See_: `docs/architecture/search-nn-interface.md` §5.4; `docs/architecture/gt-cfr-theory.md` §9

### Training Pipeline

**Replay buffer**:
A fixed-capacity sliding window (FIFO) storing training tuples from self-play searches. Newest data enters, oldest (from weak early checkpoints) is evicted. Not everything forever, not everything discarded each generation.
_Avoid_: Experience replay (DQN term with different connotations), memory
_See_: `docs/architecture/gt-cfr-theory.md` §12; `docs/research/article_summary_6.md` "How does the training actually work"

**Training tuple**:
The data emitted by each inner-loop search: (encoded public belief state β, search-refined CFVs, average strategy σ-bar). The GT-CFR analogue of AlphaZero's (state, MCTS policy, outcome).
_Avoid_: Sample, example (too generic)
_See_: `docs/architecture/gt-cfr-theory.md` §12; `docs/research/article_summary_7.md` "Part 1"

**Bootstrapping**:
Using the network's own estimates at search leaves to generate training targets that are better than the raw network, because search = network + terminals + the CFR improvement operator. The grounding ultimately traces back to real ±1 terminal payoffs; the network lets you truncate search at manageable depth.
_Avoid_: Self-referential training (sounds circular; it's not, because terminals provide the anchor)
_See_: `docs/research/article_summary_6.md` "Why train the next net using V_θ's own leaf values?" and "The Centerpiece" step 2; `docs/architecture/gt-cfr-theory.md` §11

**Exploitability**:
How much a best-response opponent could beat your current strategy by. Computed by training a best responder against your frozen strategy. As the agent improves, exploitability approaches zero. The outer loop's primary convergence metric.
_Avoid_: Loss (training loss is a different metric), weakness
_See_: `docs/architecture/gt-cfr-theory.md` §2; `docs/research/article_summary_5.md` "Deep CFR — How the Network Learns Regret"

**Generation**:
One outer-loop cycle: play G self-play games into the replay buffer, do M gradient steps, then publish one checkpoint. The unit of training progress and checkpoint cadence.
_Avoid_: Epoch (implies a full pass over a fixed dataset; the buffer is a sliding window), iteration (overloaded with inner-loop CFR iterations)
_See_: `docs/plans/trainer.md` §4; `docs/vibes-decisions.md` §12.4

**Training driver**:
The thin orchestrator that runs the generation loop — pumps SelfPlay output into the replay buffer, gates the warmup threshold, calls the Trainer, and pairs each checkpoint with a buffer snapshot. Distinct from the Trainer, which is only the learner (net + optimizer + one gradient step).
_Avoid_: Trainer (the Trainer is just the learner), orchestrator (fine as a synonym; "driver" is canonical)
_See_: `docs/plans/trainer.md` §2, §4; `docs/vibes-decisions.md` §12.1

**Checkpoint**:
A published network snapshot: the weights plus the CVPNConfig needed to rebuild the net, plus optimizer + RNG state for exact resume. Written by the Trainer once per generation; consumed by SelfPlay (warm-start) and Evaluation (head-to-head against frozen prior checkpoints).
_Avoid_: Model save; snapshot ("snapshot" is the replay buffer's persisted contents)
_See_: `docs/plans/trainer.md` §3; `docs/architecture/repo-architecture.md` §4; `docs/vibes-decisions.md` §12.2

**Warmup threshold**:
The minimum number of tuples in the replay buffer before the first gradient step. Owned by the training driver, not the buffer — the buffer only enforces "don't sample more than you hold."
_Avoid_: Burn-in
_See_: `docs/plans/trainer.md` §4; `docs/plans/replay-buffer.md` §3; `docs/vibes-decisions.md` §8.11

### Build Phases

**Phase 1 (pit-stop)**:
Perfect-info AlphaZero. Both teams fully revealed. MCTS + scalar-value NN + self-play. Validates all the hard engineering — sim integration, encoding, NN, training loop, action masking. The hardest engineering phase.
_Avoid_: Calling this "the baseline" (it's infrastructure validation, not the real product)
_See_: `docs/architecture/gt-cfr-theory.md` §14; `docs/architecture/state-encoding.md` §10

**Phase 4 (north star)**:
GT-CFR + CVPN with imperfect information. Belief-weighted candidate tokens, per-candidate CFV vector, meta priors supplying the support. The real product. Should be measurably less exploitable than determinized MCTS.
_Avoid_: Calling this "the endgame" (Phase 5 refinement follows)
_See_: `docs/architecture/gt-cfr-theory.md` §14; `docs/architecture/state-encoding.md` §10

**Validation curriculum**:
The staged sequence of fixed, hand-crafted matchups of increasing complexity used to prove the architecture finds the known-correct solution at each level before adding complexity. Begins with the absolute simplest case (all-Fire vs all-Grass, two moves each) and climbs toward two balanced teams. The spine of how Phase-1 feasibility is demonstrated.
_Avoid_: Curriculum learning (that's a training-acceleration technique; this is a validation methodology), test suite
_See_: `docs/plans/self-play.md` §6.2; `docs/plans/evaluation.md`

**Curriculum stage**:
One matchup in the validation curriculum, with a humanly-verifiable correct outcome (e.g. "the favored team wins far more often"). Each stage is trained from scratch as an independent correctness proof, and optionally warm-started from the previous stage as a side experiment.
_Avoid_: Phase (a build phase is the whole system regime; a stage is one curriculum matchup)
_See_: `docs/plans/self-play.md` §6.2

### Pokemon Domain

**Joint action**:
One complete turn submission — a move-and-target for each of your two active Pokemon, or a switch for one or both. The agent's action space is these combinations (~100 effective per turn after masking). One joint-action per player; two simultaneous joint-actions together comprise a standard Pokemon turn.
_Avoid_: Move (that's one Pokemon's action, not the pair), action (ambiguous without "joint")
_See_: `docs/architecture/gt-cfr-theory.md` §13; `docs/vibes-decisions.md` §6

**Simultaneous moves**:
Both players submit their joint actions without seeing the other's choice. Encoded in the search tree by routing all of your action branches into the same opponent info set — the opponent's strategy can't depend on which action you picked.
_Avoid_: Concurrent actions (sounds like threading)
_See_: `docs/architecture/gt-cfr-theory.md` §13; `docs/research/article_summary_4.md` "Crucial Detail: Simultaneous Moves and Info Sets"

**Forced switch**:
A unilateral decision after a Pokemon faints mid-turn. Breaks the simultaneous pattern — only the player who lost a Pokemon acts. A single-player decision node in the tree.
_Avoid_: Free switch (different concept — pivot moves give free switches)
_See_: `docs/architecture/gt-cfr-theory.md` §13; `docs/research/article_summary_4.md` "What the Pokemon Tree Actually Looks Like"

**Forced decision**:
A state where every acting side has exactly one legal action — pure game ceremony with no strategic choice (e.g. a forced switch to your only remaining Pokemon, or a Choice-locked slot with one legal target). Its value is fully determined by its successors, so it is skipped entirely: no search, no network evaluation, no training signal. Distinct from a forced switch, which can still be a genuine decision (which of several Pokemon to bring).
_Avoid_: Forced switch (a forced switch is only a forced *decision* when a single legal target remains)
_See_: `docs/plans/self-play.md` §2.2; `docs/plans/search.md` §8

**Matchup**:
A pairing of two concrete teams for one game. Supplied by a MatchupSource; in Phase 1 drawn from the fixed validation curriculum.
_Avoid_: Game (a matchup is the teams, not the played-out game), pairing (too generic)
_See_: `docs/plans/self-play.md` §6

**Meta priors**:
Tournament-data-derived probability distributions over what sets each species is likely to run, conditioned on teammates and revealed info. The abstraction layer that makes the opponent's combinatorially huge private-state space finite and tractable.
_Avoid_: Priors (overloaded with policy prior), metagame data (that's the raw input, not the processed distributions)
_See_: `docs/architecture/state-encoding.md` §7.4, §9; `docs/research/article_summary_5.md` "Counterfactual Reach at Play Time + Meta Priors"

**Archetype**:
A fuzzy cluster of similar sets for a species (e.g. "bulky TR attacker Torkoal"). Being retired in favor of data-driven top-K candidates from the conditional sampler. The problem with archetypes: subjective boundaries, loses within-cluster variance, doesn't give GT-CFR the concrete distinguishable candidates it needs.
_Avoid_: Using this term in new code — prefer "candidate" or "candidate set"
_See_: `docs/architecture/state-encoding.md` §9, §11.3
