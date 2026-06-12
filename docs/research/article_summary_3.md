Switching to `$$` for equations. Many of these questions point at the same conceptual confusion, so I'll address that first, then go through each question.

## What "Iteration" Means in CFR

> Your initial paragraph in the previous response was too dense and jargon-filled for me. What is MDP? What is a regret table -- is this a mapping from (info-set + action) to (regret)? When you say "is across the 10^6 CFR iterations of the algorithm", is this analogous to in MCTS when you reach an already-explored leaf, and you visit that leaf say 200 times (out of your 800 total simulations) and you ask "what if I had played a all 200 times at this leaf"? Does there exist a corollary for "always play a" in the MCTS framework? I'm still confused by this, as your explanation was a bit too jargon-filled for me.

**MDP**: Markov Decision Process. The standard reinforcement learning framework: states, actions, transition probabilities, rewards. **Value iteration** is an algorithm that solves MDPs by repeatedly updating value estimates `V(s)` via the Bellman equation until they converge. The point of mentioning it: value iteration runs purely on a model of the MDP without ever "playing" the MDP. Same flavor as CFR.

**Regret table**: yes, exactly what you said. A data structure mapping `(info_set, action) → cumulative_regret_scalar`. One regret table per player. CFR also maintains a **strategy table** mapping `info_set → distribution over actions`, and an **average strategy table** that accumulates strategies across iterations.

**Now the key clarification on what "iteration" means.** A CFR iteration is one full pass through the algorithm. In vanilla CFR, that means traversing the entire game tree once. In each pass:

1. The algorithm visits every info set in the tree.
2. At each info set, it computes counterfactual values for each action under the current strategy.
3. It updates the cumulative regret table at that info set.
4. After the pass, it derives the next iteration's strategy at every info set via regret matching.

So iteration 1 visits all info sets once. Iteration 2 visits all info sets again. After $10^6$ iterations, every info set has been visited $10^6$ times, and its cumulative regret reflects $10^6$ updates.

**There isn't a clean MCTS analogy for "always play `a`."** MCTS doesn't have iteration counts in this sense — its "simulations" each add one new node. Each MCTS sim does something *different* (explores a different leaf). CFR iterations all do *the same thing* (traverse the tree, update everything) — they differ only because the strategy `σ^t` evolves between iterations.

The closest analogy: think of CFR like training a neural network. Each iteration is one epoch. Every parameter (every regret entry) gets updated each epoch. The strategy converges over many epochs. MCTS, by contrast, is more like depth-first exploration that grows a tree — fundamentally different shape of computation.

So when I write "always played `a` at `I` across iterations," I mean: across the $10^6$ tree passes, the regret says "if you'd picked `a` at info set `I` in every one of these passes, instead of whatever your strategy actually said for each pass, you'd have done better/worse by this cumulative amount."

## Why Factor Out Player `i`'s Own Probabilities

> Why does factoring out player i's own probabilities along the path help evaluate the quality of player i's decisions at I? In your example where action a at I is amazing, why don't we just evaluate the goodness of the path based on that? The path led us here, even if it was a result of unlikely choices by player i, and now we get to make an amazing decision. Why does including the information about the opponent's probabilities of getting here make sense? Why is that relevant?

Genuine subtle question. Two reasons.

**Reason 1: Locality of decision quality.**

The quality of a decision at info set `I` should not depend on how often the player chooses to reach `I`. Consider: I have a strategy that almost never bluffs (passes with K with probability 0.99). The bet branch from `I1_K` is rarely visited under my strategy. But if I *do* bet K, the consequences are great regardless.

If I weighted by full reach probability (including my own), the value of "bet" at `I1_K` would be tiny because I rarely get there:

$$\text{full reach value of bet} = 0.01 \cdot 0.5 = 0.005$$

This is misleading. The action "bet" isn't worth 0.005 — it's worth 0.5 *given that I'm here*. The 0.01 just reflects how rarely I am here. Counterfactual reach factors out my contribution, giving:

$$v_1(I1_K, \text{bet}) = 0.5 \cdot (\text{whatever})$$

This decouples "how good is the action here" from "how often do I reach here."

**Reason 2: It's what makes the convergence proof work.**

This is the technical reason. The CFR theorem (Zinkevich et al. 2007) proves that if you minimize *counterfactual* regret at each info set independently using regret matching, the average strategy converges to Nash. The proof requires the counterfactual definition specifically. If you used full reach, the proof breaks — the regret bounds don't compose properly across info sets.

You can think of it as: counterfactual reach makes each info set's regret minimization a local subproblem that can be solved independently, while still composing globally to a Nash strategy. Including your own probabilities entangles info sets in a way that breaks the local independence.

## Can Player `i` Compute Reach Probabilities?

> Also, how can player i know about the reach probabilities, if they don't know the opponent's info set? Or is this only in the context of "I am the omniscient orchestrator of the simulation, so I know both player i's and the opponent's info sets, so I can compute the reach probabilities for both players"? Player i shouldn't be able to compute their own counterfactual reach probability, right?

Right intuition, important distinction.

**At training time** (when CFR is running): the algorithm is omniscient. It has the full game tree, knows everyone's current strategies, and computes reach probabilities by multiplying along paths. CFR is a centralized procedure run offline; both players are simulated by the same algorithm with full visibility.

**At play time** (after CFR has converged): the player only needs the average strategy table. They look up their info set, sample from the action distribution, play. No reach probability calculation is performed during play.

So CFR is structured like AlphaZero's training/inference split:

- **Training**: omniscient, expensive, offline, uses self-play with full information access.
- **Deployment**: each agent only sees its own info set, queries its strategy table, acts.

You're correct that a player at play time can't compute their opponent's reach probabilities. But they don't need to — all the inference about hidden information is baked into the precomputed strategy.

## Counterfactual Value, Histories, and Utilities

> The counterfactual value (and the implicit utility u_i) is not a summation over previous turns / nodes, right? It considers "at this info set, if I take action a, AND GIVEN a fully known history h (including opponent's info -- we sample over distributions here?), what is the value of taking action a?" So I guess it does sort of consider previous turns, implicit in the h (does h mean a node or a history?). I feel like I'm almost there at understanding this counterfactual value, but I'm still not fully clear understanding it.

> Is u_i like the NN evaluation in the MCTS discussion we had earlier? Given a state plus action, what is the probability of winning from here -- a synonym for utility? Then we consider the probability of winning GIVEN A FULLY KNOWN history and sum across ALL histories that fall under this info set, weighted by....the probability the opponent wanted to be at this history? I don't quite have the intuition for what the role of the counterfactual reach probability is here in this summation. It looks like a weighting of various values to sum to a total value, but I don't quite understand why we weight by this. I think you're trying to explain it by saying it decoupled from "how often do I even reach I" but this just looks like it's being weighted by opponent's desired to reach I, which I don't understand.

Let me restate the equation precisely with terminology cleared up:

$$v_i(I, a) = \sum_{h \in I} \pi_{-i}(h) \cdot u_i(h \cdot a, \sigma)$$

**Terminology**:
- **History `h`**: a complete sequence of actions from the root, equivalent to a "node" in the game tree. They're interchangeable in extensive-form games. So "history" and "node" mean the same thing.
- **Info set `I`**: a set of histories indistinguishable to the acting player.
- **`h · a`**: the history `h` extended by appending action `a`. A new node deeper in the tree.
- **`u_i(h · a, σ)`**: the **expected utility** for player `i` from being at history `h · a`, when play continues under everyone's strategy `σ`. This is computed by recursion through the subtree below `h · a`. It's not a single deterministic payoff — it's an expected value over all the random outcomes (chance, opponent strategies, your own future strategies) that could follow.

So `u_i(h · a, σ)` **is essentially a value function**: "from this game position, with everyone playing strategy `σ`, what's player `i`'s expected payoff?" In vanilla CFR, this is computed exactly by recursion. In Deep CFR / GT-CFR, it's approximated by a neural network at leaves of the search tree.

Yes, your intuition is right: this is directly analogous to MCTS's value head. The neural network in Player of Games (the CVPN) approximates `u` (and counterfactual values, and policies) at leaf info sets.

The summation is **not over previous turns**. It's over the parallel histories that are bundled into the same info set `I` — different "worlds" the player can't distinguish. In Pokemon, the histories within an info set differ in things like which specific item the opponent is holding, which specific Pokemon they have in the back, etc.

**Why weight by counterfactual reach in the sum?**

Each history within `I` has different probability of actually being the "real world." From the algorithm's perspective:

- History `h_a`: I have K, opponent has J. Counterfactual reach `π_{-1}(h_a)` reflects how likely this specific deal-and-actions chain was, ignoring my own contributions.
- History `h_b`: I have K, opponent has Q. Counterfactual reach `π_{-1}(h_b)` similarly.

If opponent has played in a way that makes `h_a` more likely than `h_b` (say, by passing on the first action — which is more consistent with weak hands like J than strong hands), then `π_{-1}(h_a) > π_{-1}(h_b)`. The expected value of action `a` should weight more toward what happens in `h_a` than `h_b`.

This is **automatic Bayesian inference**. The counterfactual reach probabilities, after factoring out my own actions, encode the relative likelihood of each history given everything I've observed. Worlds inconsistent with observed play get zero weight. Worlds consistent get weighted by how plausible they are under the opponent's current strategy and chance.

So the weighting captures: "given what I've observed, what's the relative likelihood of each hidden state, and what's the value of action `a` averaged over those likelihoods?"

## "Time" Disambiguation

> Time can mean two things in this discussion, and I think I got confused by what t corresponds to. A difference in time could mean the difference in turn 1, simulation 1 vs turn 1, simulation 800 -- a delta in sim number, like wall-clock time. Or a difference in time could mean the difference in turn 1 vs turn 2, where presumably turn 2 happens after turn 1 in a real game -- thus a difference in time. So as we simulate turn 1 many times over and build a regret table, the info set stays the same (since we're staying with turn 1) but the "time" increments as we go from sim 1 to sim 473 and so on. Is that correct?

Critical clarification.

There are **two distinct uses of `t`** in our discussions, and I should have separated them cleanly:

1. **CFR iteration `t`**: the algorithm's iteration counter. CFR runs for `T = 10^6` iterations, where each iteration is one full traversal of the game tree updating all info sets. This `t` is in the regret equations: `R^T(I, a) = sum_{t=1}^T [v^t - ...]`.

2. **Real game turn `t`**: the turn number in an actual played game (Pokemon turn 1, turn 2, ...). This is *not* what appears in the regret equations.

CFR doesn't "simulate turn 1 many times over" — that's not how CFR works. **In a single CFR iteration, the algorithm visits info sets at all game turns simultaneously** (turn 1 info sets, turn 2 info sets, turn 3 info sets, etc.) and updates them all in parallel.

After 10^6 CFR iterations, every info set in the game tree (at every turn) has accumulated 10^6 regret updates. The strategy at each info set converges to its Nash equilibrium component.

Your framing was: "as we simulate turn 1 many times over and build a regret table, the info set stays the same, the time increments." This is what MCTS does (multiple sims of the next-decision tree). CFR is different: CFR runs the same kind of update repeatedly across iterations, but each update covers the *whole tree*, not just one turn.

## What Is "Strategy" in CFR

> What is the strategy term? Is that analogous to the MCTS case where you balance exploration vs exploitation? Or is the strategy term a separate function indexed to each particular player? Or is it just the mapping from simulation index to (what action did you actually take)? And this strategy gets updated as we increment the sim, presumably? So this is the relation to the MCTS approach, where as you increment the sim count, you choose different actions at the same info set by balancing exploration and exploitation, and here you choose a different strategy as you increase sim count by minimizing regret over previous sim counts? How does this balance exploration and exploitation, or does the analog not really exist?

A **strategy** for player `i` is:

$$\sigma_i: \text{InfoSet} \to \text{Distribution over Actions}$$

So `σ_i(I, a)` is the probability player `i` plays action `a` at info set `I`. A strategy is a complete plan: "at every possible info set I might find myself in, here's my action distribution."

It's **not** indexed to any particular game or simulation — it's a stationary policy. It gets indexed by CFR iteration only because it changes between iterations as we improve it: `σ^1, σ^2, ..., σ^T`.

**Comparison with MCTS:**

| MCTS | CFR |
|---|---|
| UCB explicitly trades off exploitation (Q) and exploration (U term) within a single search | Regret matching has no explicit exploration constant |
| Each sim picks one action via UCB, walks down, expands one leaf | Each iteration visits all info sets, computes regret-weighted strategy |
| Output: visit count distribution at root for current decision | Output: full strategy mapping every info set to action distribution |
| Re-run from scratch for each new game state | Trained once offline, deployed as lookup table |

There IS an implicit exploration in regret matching: any action with positive cumulative regret gets non-zero probability in the next iteration. So actions that have done well get weighted higher, but actions that haven't been "tried enough" can still accumulate regret if their counterfactual value would have been higher than what was played.

Unlike UCB, regret matching doesn't have a tunable exploration parameter. The convergence guarantees come from the math of regret bounds.

## Why Kuhn Reach Is `1/6` Not `1/2`

> In the Kuhn poker example, why isn't the counterfactual reach 1/2 and 1/2, or 1/3 and 1/3? It seems to include the inormation from Player 1 in the calculation of the counterfactual reach. As an omniscient observer, we know that the probability of (K, J) is perfectly 1/6 and the same for (K, Q). But from Player 1's perspective, they already know they have K, so why isn't this taken into account? It seems like the counterfactual reach of 1/6 and 1/6 can be known by everyone -- is this supposed to be true? Can we compute the counterfactual reach as player 1, or do we need to be the omniscient observer?

Sharp question, and the answer reveals something nice.

**Reach probabilities are unconditional, computed from the algorithm's omniscient perspective.** The chance node deals 6 equally likely outcomes (3 cards, choose one for each player, ordered): KJ, KQ, JK, JQ, QK, QJ. Each has probability `1/6` — that's the chance contribution.

But here's the elegant part: **the algorithm uses unconditional reaches, but the relative weighting within an info set automatically gives the player's conditional view.**

At info set `I1_K` containing histories `h_a` (P1=K, P2=J) and `h_b` (P1=K, P2=Q):

- `π_{-1}(h_a) = 1/6`
- `π_{-1}(h_b) = 1/6`

When we compute `v_1(I1_K, a)`:

$$v_1(I1_K, a) = (1/6) \cdot u_1(h_a \cdot a) + (1/6) \cdot u_1(h_b \cdot a)$$

If we factored out the absolute scaling, the *relative* weights are `1:1`, which IS the player's belief: given P1=K, opponent has J or Q with equal probability `1/2 : 1/2`.

The absolute `1/6` rather than `1/2` reflects "globally, how often does this scenario occur," and it scales the regret update appropriately. When you sum regrets across info sets, the correct global weighting matters for the algorithm to converge to a globally optimal strategy.

In Kuhn this is "uniform chance" so the relative weights happen to match the player's conditional belief trivially. In poker more generally, opponent strategy probabilities affect the relative weights of histories within an info set — and that's exactly the Bayesian inference the algorithm performs automatically.

So you can think of it as: the algorithm uses unconditional reach probabilities for technical correctness, but the *information* the player effectively has is encoded in the *relative weights* of histories within each info set. From the player's perspective at play time, only the strategy matters; the reach probability machinery is all behind-the-scenes computation during training.

## Variable Nash Equilibrium

> How can a Nash equilibrium be variable, like in the Kuhn poker example with alpha? My only exposure to Nash equilibria is in the case of the prisoner's dilemma; despite it being worse for the system overall, both players' optimal choice is to inform on the other. This is a binary problem (inform or not), so it's easy to visualize as a 2x2 matrix and the equilibrium is a single square in that matrix. There's no variability here. What does it mean for a Nash equilibrium to be variable like this? How can it be an equilibrium if it's variable?

Great question. The intuition you have from prisoner's dilemma is misleading because that game has a *unique* Nash equilibrium due to strict dominance. Most games don't.

**General fact**: in two-player zero-sum games, the **value of the game** (expected payoff under optimal play) is unique, but the **set of Nash equilibrium strategies** is generally a convex set, not a single point. Multiple strategies achieve the same value.

Kuhn poker has expected value `-1/18` for player 1 (player 2 has a slight advantage). There's a one-parameter family of Nash strategies for player 1, parameterized by α ∈ [0, 1/3]. All of them achieve exactly `-1/18` against optimal play. Player 1 can pick any α in this range and be unexploitable.

It IS still an equilibrium because: at any α, neither player can deviate unilaterally to do better. The opponent's best response to my α ∈ [0, 1/3] gives me exactly `-1/18`, regardless of which α I picked. So I have no incentive to switch.

The "variability" is just slack in the equilibrium — it's a set of strategies that are all mutually best-responding. The Nash *value* is unique; the Nash *strategy set* often isn't.

For your Pokemon project: you might converge to different unexploitable strategies in different training runs. All of them achieve the same value against optimal opponents. They might differ in how they exploit *suboptimal* opponents, which is why the broader research field eventually wants tools to find specific equilibria within the equilibrium set (e.g., maximizing exploitation of human-style errors).

## Strategy Update for `I1_K_pb` Branch

> Also in the Kuhn poker example, how would the strategy get updated for if P1 passes with King and P2 bets? I understand how in iteration 2, they always bet 100% with king, but for the sake of teaching, would the subcase of P1 pass and P2 bet get its strategy updated, too? What does the computation look like there?

Let me work through this one. Info set `I1_K_pb`: P1 has K, history is pass-bet.

**Histories in `I1_K_pb`**:
- `h_a' = (deal: P1=K, P2=J, P1 passed, P2 bet)`
- `h_b' = (deal: P1=K, P2=Q, P1 passed, P2 bet)`

**Counterfactual reach** (factor out P1's contributions, keep chance + P2):

- `π_{-1}(h_a') = (1/6) · σ_2^1(I2_J_p, bet) = (1/6)(0.5) = 1/12`
- `π_{-1}(h_b') = (1/6) · σ_2^1(I2_Q_p, bet) = (1/6)(0.5) = 1/12`

Notice P1's pass at `I1_K` is *factored out* (would have been 0.5 otherwise). P2's bet at their respective info sets is included.

**Counterfactual value of "call"**:
- If `h_a'`: P1 calls, showdown, K beats J, payoff +2.
- If `h_b'`: P1 calls, showdown, K beats Q, payoff +2.

$$v_1(I1_{K_{pb}}, \text{call}) = (1/12)(2) + (1/12)(2) = 1/3$$

**Counterfactual value of "fold"**:
- Both histories: P1 folds, loses ante, payoff -1.

$$v_1(I1_{K_{pb}}, \text{fold}) = (1/12)(-1) + (1/12)(-1) = -1/6$$

**Counterfactual value of current strategy** (uniform 0.5/0.5):

$$v_1(I1_{K_{pb}}, \sigma^1) = 0.5 \cdot (1/3) + 0.5 \cdot (-1/6) = 1/6 - 1/12 = 1/12$$

**Instantaneous regret**:

$$r^1(I1_{K_{pb}}, \text{call}) = 1/3 - 1/12 = 1/4$$
$$r^1(I1_{K_{pb}}, \text{fold}) = -1/6 - 1/12 = -1/4$$

**Cumulative regret after iteration 1**: same as instantaneous (first iteration).

**Strategy for iteration 2**:

$$\sigma^2(I1_{K_{pb}}, \text{call}) = 1.0, \quad \sigma^2(I1_{K_{pb}}, \text{fold}) = 0$$

So in iteration 2, P1 calls with K when reached at the pass-bet branch. Confirms intuition: K is best, never fold.

**The key point**: this update happens in the *same iteration* as the `I1_K` update. CFR updates all 12 info sets in parallel each iteration. By the end of iteration 1, every info set has new cumulative regrets and a new iteration-2 strategy.

In iteration 2, the algorithm runs the whole tree again with the new strategies (which are now mostly extreme: bet K, fold J, etc.), and the counterfactual values shift. Regrets accumulate differently. Eventually, after many iterations, the strategies stabilize at Nash. The "always bet K" first-iteration jump gets corrected over time toward the slow-play probability `3α` from the Nash family.

## Does CFR Need to Play the Game?

> Does CFR iterate to find the optimal strategies in the Kuhn poker example without actually playing the game? There's no self-play there because it's such a simple game you can compute the full info sets and store them, right? For a more complex game, you have to move to a different framework that can handle not expanding the full tree out, correct?

**No.** Vanilla CFR is purely an offline algorithm operating on the game tree. No actual game play required.

For Kuhn (12 info sets, ~50 histories), you can fully enumerate the tree in code and run vanilla CFR for ~10^4 iterations. Converges to a Nash strategy in seconds. No "play" happens.

For larger games where you can't enumerate the tree:

- **MCCFR (Monte Carlo CFR)**: instead of full tree traversal per iteration, sample a path or subtree per iteration. Variance is higher per iteration but iterations are cheaper. Still no real game play — just sampled tree traversals.
- **Deep CFR**: replace explicit regret tables (one entry per info set) with a neural network that predicts regrets from info set features. The network generalizes across info sets. Training the network requires sampling info sets and computing target regrets — this still happens over the model of the game, not actual game plays.
- **GT-CFR / Player of Games**: grow the search tree on the fly during decision-time search, using a neural network at leaves. Now we're at search time — this is invoked when we need to play an actual decision. The CVPN itself is trained via self-play (similar to AlphaZero's training pipeline).

So the picture for a project like yours:

- **CFR-style search algorithm**: operates on the game model + a value/policy network. Doesn't "play" the game; it does math on the model.
- **Self-play training pipeline**: needed to train the value/policy network. This involves actual game play (or simulated game play through `@pkmn/sim`) to generate training data.

These are two separate concerns. The search algorithm is what you'd query at decision time during a real game; the training pipeline is what runs offline to make the search algorithm strong.

For Pokemon: you can't enumerate the game tree (too many states), so you need GT-CFR or similar with a CVPN. The CVPN is trained via self-play through your simulator. The search algorithm at play time runs GT-CFR from the current public state to produce a strategy at the root info set.

This is roughly the architecture you'd be aiming for after the perfect-information AlphaZero milestone is working.

Want me to walk through what GT-CFR does operationally on a single Pokemon turn now that the CFR foundation is in place? That would be a concrete bridge between "I understand CFR on Kuhn" and "I see how this applies to my project."