Several of your questions point at the same root confusion, which I'll address first because it unlocks everything else.

## CFR Iterations Are Not Game Iterations

> How does regret work in a game like Pokemon where you don't play the same game 100x over? Turn 2 is different than turn 1. But I guess that's what the simulation is, right? We sim turn one 100 times without knowing what the opponent will do, then reveal the truth of what they did and compute regret against what they actually did to know "how we would have done if we'd picked the best option 100x instead"? Is that understanding correct of how regret would apply to this game as opposed to RPS (where you DO play the identical game each time)? I think this is what you're saying right before the CFR section: "But this only works for normal-form games (single-shot decisions). Real games have sequences of decisions and information asymmetry. That's where CFR comes in."

The fundamental confusion: you're picturing CFR as "play the actual game over and over and learn from the results." That's not what CFR is.

**CFR is an iterative algorithm operating on a model of the game tree, much like value iteration in dynamic programming.** The "T" in regret equations is the algorithm's iteration counter, not the game's turn number, and not the number of real games played.

Compare to value iteration for MDPs: you don't play the MDP 1000 times — you have a model of states, actions, and transitions, and you iteratively update value estimates until convergence. CFR is analogous. You have a model of the game tree (or a sampled subset), and you iteratively update regret tables at each info set until the average strategy converges to Nash.

So when you ask "how can I 'always play action `a`' across iterations when each turn of Pokemon is different?" — the answer is: **`a` doesn't refer to the same action across game turns. It refers to the same action at the same info set, across CFR iterations.** Different turns of the game are different info sets, with separate regret tables. The "always play `a`" is "across the 10⁶ CFR iterations of the algorithm, what if at info set `I` I had picked action `a` every time?"

This dissolves several of your confusions at once.

## Extensive-Form Games (Formal Definition)

> Define what you mean by extensive-form games a bit more clearly.

An **extensive-form game** is a game tree with the following components:

- **Nodes**: positions in the game.
- **Edges**: actions taken from a node.
- **Player function** `P(h)`: identifies which player acts at non-terminal node `h` (or "chance," for stochastic events).
- **Action function** `A(h)`: the legal actions at node `h`.
- **Chance probabilities**: at chance nodes, the probability distribution over outcomes.
- **Terminal nodes**: leaf nodes with payoff vectors `u(h) = (u_1(h), u_2(h), ...)` for each player.
- **Information sets** `I`: a partition of each player's decision nodes into equivalence classes such that nodes in the same info set are indistinguishable to the acting player. All nodes `h ∈ I` must have the same action set `A(I)`.

**Perfect information** means every info set is a singleton (every node is distinguishable). Chess and Go are perfect-information extensive-form games.

**Imperfect information** means some info sets contain multiple nodes. Poker, Pokemon, and Hanabi are imperfect-information extensive-form games.

Your earlier RPS example was a **normal-form game** — a single simultaneous decision, no tree structure. Extensive-form games strictly generalize normal-form games.

## Reach Probabilities — Concrete Decomposition

> I don't understand the decomposition of the reach probability. Maybe an example would help. How do you compute / identify the contributions from each player to reach a history? In e.g. poker, does this take into account "if my opponent had a 2 and 10, he would have folded by now, so he must not have a 2 and 10"? Or does that information come into play elsewhere?

A **history** `h` is a sequence of actions from the root to a node. The **reach probability** `π(h)` is the probability of arriving at `h` given everyone's strategies and chance.

It decomposes as a product over the path:

$$ \pi(h) = \prod_{(h', a) \in \text{path}(h)} \text{Pr}(a \text{ taken at } h') $$

Each factor is one of:
- A **chance probability** (if `h'` is a chance node)
- A **player strategy probability** `σ_i(I(h'), a)` (if `h'` is player `i`'s decision node, where `I(h')` is the info set containing `h'`)

Decomposed by player:

$$ \pi(h) = \pi_c(h) \cdot \pi_1(h) \cdot \pi_2(h) $$

- `π_c(h)` = product of chance probabilities along the path
- `π_i(h)` = product of player `i`'s strategy probabilities along the path
- "Counterfactual" reach for player `i`: `π_{-i}(h) = π_c(h) · π_{j≠i}(h)` (everything except `i`'s own probabilities)

**Concrete poker example.** Kuhn poker: 3 cards (J, Q, K), each player gets one, one unused. Both ante 1 chip. P1 acts first: pass or bet. Etc.

Consider history `h = (chance dealt P1=K and P2=Q, P1 passed, P2 bet, P1 called)`. Suppose:

- P1's strategy at info set `I1_K` (has K, no actions yet): pass with 0.7, bet with 0.3
- P2's strategy at `I2_Q_p` (has Q, P1 passed): bet with 0.4, pass with 0.6
- P1's strategy at `I1_K_pb` (has K, history pass-bet): call with 0.9, fold with 0.1

Then:

- `π_c(h) = 1/6` (probability chance dealt this specific card distribution)
- `π_1(h) = 0.7 × 0.9 = 0.63` (P1 chose pass at `I1_K`, then call at `I1_K_pb`)
- `π_2(h) = 0.4` (P2 chose bet at `I2_Q_p`)
- `π(h) = (1/6)(0.63)(0.4) ≈ 0.042`
- `π_{-1}(h) = (1/6)(0.4) ≈ 0.067` (everything except P1's contributions)

**Crucially**: this decomposition is purely arithmetic. We don't reason about "would P2 have folded with hand X?" — that's already baked into P2's strategy probabilities. If P2's strategy says "fold X with probability 1," then any history where P2 didn't fold with X has zero `π_2` contribution and thus zero reach probability. Bayesian inference about hidden information falls out automatically from the strategy probabilities; you don't compute it separately.

This is part of why CFR is elegant for imperfect-info games — the reach probabilities encode all the inference you'd want to do about what's consistent with the opponent's actual play.

## "Trying to Reach" the Info Set

> What does it mean to try to reach this info set? Some more clarification about players "trying to reach" or "contributing to" reach an info set or history would be helpful.

The counterfactual reach probability `π_{-i}(h)` is "the probability of arriving at `h` if player `i` were trying to get there with certainty." We're factoring out player `i`'s own probabilities along the path because we want to evaluate the *quality* of player `i`'s decision *at* `I`, not the probability that they bothered to come here at all.

Why? Suppose at info set `I`, action `a` is amazing and gives expected value +100, but `I` is reached only when player `i` makes a long sequence of unlikely decisions earlier. We don't want to say "action `a` at `I` is worthless because we never reach `I`." We want to say "if we did reach `I`, here's how good `a` is." So we strip out `i`'s own probabilities along the path.

The counterfactual value of action `a` at info set `I`:

$$ v_i(I, a) = \sum_{h \in I} \pi_{-i}(h) \cdot u_i(h \cdot a, \sigma) $$

`u_i(h · a, σ)` = expected utility for `i` of being at `h`, taking `a`, and continuing under everyone's current strategy `σ`. The sum is over all histories in the info set, weighted by counterfactual reach.

The intuition: `v_i(I, a)` is "given the strategies, how much expected payoff does action `a` at info set `I` contribute to my total?" Counterfactual weighting decouples this from "how often do I even reach `I`."

## Info Sets Are Not Time-Indexed

> In the equation for counterfactual regret of action a, the info set is a function of time too, right? So I should be I_t (or superscript)? And how can you always have played action a for all times t up to T? In a pokemon game, you can't do the same thing on turn 2 as you do turn 1 because it's a different turn; it thus makes no sense to ask "what if I had always picked action a here"

You asked whether info sets should be `I_t` (time-indexed). They should not.

An info set is a fixed object in the extensive-form game — a specific decision point characterized by what the acting player has observed up to that point. Examples in Pokemon:

- `I_A` = "Turn 1 of battle, my team is X, my private knowledge is Y, I have observed nothing about opponent yet, I must choose move and target for both my active Pokemon"
- `I_B` = "Turn 2, history of revealed actions on turn 1 is Z, my team status is W, I must choose actions"

`I_A` and `I_B` are different info sets. They live at different positions in the game tree. They have separate regret tables.

The strategy `σ` is iteration-indexed (`σ^t` evolves over CFR iterations), but info sets are static features of the game. So:

$$ R^T(I, a) = \sum_{t=1}^T \left[ v_i^t(I, a) - v_i^t(I, \sigma^t) \right] $$

means: across CFR iterations 1 through T, sum up "how much better would action `a` have been at info set `I` than the strategy I used at `I` in iteration `t`?"

In iteration `t`, I used some strategy `σ^t(I, ·)` at info set `I`. That strategy gave some counterfactual value. Action `a` would have given a different counterfactual value. The difference is the instantaneous regret. We accumulate over iterations.

The strategy at `I` evolves: in iteration 1 it might have been uniform, in iteration 100 it might have been heavily biased toward bet, in iteration 1000 it might be slightly biased toward pass, etc. We're asking, in hindsight after `T` iterations, "how much of my accumulated payoff came from suboptimal choices at this info set, compared to having always picked `a` here?"

For Pokemon's turn-2 info sets vs turn-1 info sets: each has its own regret table, updated independently.

## How CFR Sampling Doesn't Leak History

> Because I don't understand these things before, I'm still unclear on the CFR algorithm itself and how it works. How can you sample the game tree and encounter info sets without unintentionally leaking the associated histories? I don't know what my turn 2 info set even CAN be without knowing the history. More explanation about this algorithm after answering my other questions

Subtle question. The answer is in how the strategy is structured.

A strategy `σ_i` is a function from **info sets** to action distributions. The strategy at info set `I` is a single distribution `σ_i(I, ·)` over actions, used at every history `h ∈ I`. Player `i` cannot tell which `h` they're at, so their strategy can't depend on `h` — only on `I`.

When CFR samples a path through the tree, that path passes through specific histories. The regret update at info set `I` along that path uses `π_{-i}(h)` for the sampled `h`. But the resulting regret update is unbiased: averaged over many CFR iterations and the random sampling, the expected update equals the true counterfactual regret summed over all `h ∈ I`.

The key point: the *strategy* doesn't look at `h`; only the *update calculations* do. The update is internal to the algorithm, not exposed to the player. Once CFR converges, you have a strategy that maps info set → action distribution, with no reference to specific histories. When the player actually plays, they only see their info set, so they can faithfully execute the strategy.

This is the whole reason CFR is the right framework for imperfect-info games. Determinized MCTS samples a specific history (a "world") and finds the optimal action *for that world*, which can violate the constraint "I must play the same way at every history in my info set." CFR enforces that constraint by construction.

## CFR vs Minimax

> Is tree traversal in CFR comparable to doing a full minimax optimization of a game in the perfect information world? Where we sample EVERY possible state and compute win percent and go from there -- in "true" CFR, do you sample the ENTIRE tree from info-sets and go from there?

Not the same, even in perfect-info games.

**Minimax** (with alpha-beta pruning) computes optimal play in a perfect-info game in one pass: at every node, recursively compute "best response to opponent's best response." It produces an optimal strategy directly, no iteration.

**Vanilla CFR** traverses the entire game tree once **per iteration**, but a single iteration does not produce optimal play. Each iteration computes:
1. Counterfactual values under the current strategy (forward pass).
2. Regret updates at each info set (during traversal).
3. New strategy via regret matching (next iteration's strategy).

You need many iterations (often 10⁶ or more) for the *average* strategy to converge to Nash.

**For perfect-information games**: vanilla CFR works but is much slower than minimax. You'd never use CFR for chess.

**For imperfect-information games**: minimax doesn't apply directly because there's no single "value" of a node — the value depends on what you can infer about hidden information, which is captured by info sets, not nodes. CFR fills the gap.

For your Pokemon project: CFR is the relevant tool because of the imperfect information. Within each iteration, you do traverse (sampled paths through) the game tree, but you need many iterations to converge.

To answer your question literally: yes, vanilla CFR's full traversal of the game tree per iteration is comparable in spirit to a brute-force enumeration. The differences are (a) it's iterative, not one-shot, and (b) it operates on info sets, not states. For huge games it's intractable; that's why MCCFR, Deep CFR, and GT-CFR exist.

## Kuhn Poker Worked Example

> I was having a hard time understanding the rest of your response at/after GT-CFR because I didn't have a solid grasp of the earlier parts. I think your final question of working through it on a tiny game would be helpful

Kuhn poker: 3 cards (J, Q, K). Each player antes 1 chip, draws one card, third unused. Pot starts at 2.

- P1 acts: **p**ass or **b**et (1 chip)
- If P1 bets, P2: pass (fold, lose ante) or bet (call, showdown for pot of 4)
- If P1 passes, P2: pass (showdown for pot of 2) or bet (1 chip)
 - If P2 bets, P1: pass (fold, lose ante) or bet (call, showdown for pot of 4)

Twelve info sets total. Six for P1 (J/Q/K × first decision/after pass-bet), six for P2 (J/Q/K × after-pass/after-bet).

Let's run **one iteration of CFR** focused on info set `I = I1_K` (P1 has K, first decision).

**Initial strategy (iteration 1)**: every info set plays uniform 0.5/0.5.

**Histories in `I1_K`**:
- `h_a` = chance dealt P1=K, P2=J (probability 1/6)
- `h_b` = chance dealt P1=K, P2=Q (probability 1/6)

`π_{-1}(h_a) = 1/6`, `π_{-1}(h_b) = 1/6` (just chance, no opponent actions on the path yet).

**Compute counterfactual value of action "bet" at `I1_K`**.

If `h_a` (P2 has J): P1 bets, P2 acts at `I2_J_b` with σ uniform.
- 0.5 fold → P1 wins 1
- 0.5 call → showdown, K > J, P1 wins 2
- Expected: `0.5(1) + 0.5(2) = 1.5`

If `h_b` (P2 has Q): P1 bets, P2 acts at `I2_Q_b` with σ uniform.
- 0.5 fold → P1 wins 1
- 0.5 call → showdown, K > Q, P1 wins 2
- Expected: `0.5(1) + 0.5(2) = 1.5`

$$ v_1(I1_K, \text{bet}) = (1/6)(1.5) + (1/6)(1.5) = 0.5 $$

**Compute counterfactual value of action "pass" at `I1_K`**.

If `h_a` (P2 has J): P1 passes, P2 acts at `I2_J_p` with σ uniform.
- 0.5 P2 passes → showdown, K > J, P1 wins 1
- 0.5 P2 bets → P1 acts at `I1_K_pb` with σ uniform.
    - 0.5 fold → P1 loses 1
    - 0.5 call → showdown, K > J, P1 wins 2
    - Expected: `0.5(-1) + 0.5(2) = 0.5`
- Expected: `0.5(1) + 0.5(0.5) = 0.75`

If `h_b` (P2 has Q): symmetric.
- 0.5 P2 passes → showdown, K > Q, P1 wins 1
- 0.5 P2 bets → P1's pass-bet response averages 0.5
- Expected: `0.5(1) + 0.5(0.5) = 0.75`

$$ v_1(I1_K, \text{pass}) = (1/6)(0.75) + (1/6)(0.75) = 0.25 $$

**Counterfactual value of the current strategy at `I1_K`** (uniform 0.5/0.5):

$$ v_1(I1_K, \sigma^1) = 0.5 \cdot v_1(\text{pass}) + 0.5 \cdot v_1(\text{bet}) = 0.5(0.25) + 0.5(0.5) = 0.375 $$

**Instantaneous regret** at iteration 1:

$$ r^1(I1_K, \text{bet}) = v_1(I1_K, \text{bet}) - v_1(I1_K, \sigma^1) = 0.5 - 0.375 = +0.125 $$

$$ r^1(I1_K, \text{pass}) = v_1(I1_K, \text{pass}) - v_1(I1_K, \sigma^1) = 0.25 - 0.375 = -0.125 $$

**Cumulative regret** after iteration 1:

$$ R^1(I1_K, bet) = +0.125 $$
$$ R^1(I1_K, pass) = -0.125 $$

**Regret matching** for iteration 2:

$$ \sigma^2(I1_K, \text{bet}) = \frac{\max(0.125, 0)}{0.125 + 0} = 1.0 $$

$$ \sigma^2(I1_K, \text{pass}) = 0 $$

So in iteration 2, P1 with K bets with probability 1. Makes intuitive sense — King is the best card, you should value bet it.

Within the same iteration 1, you'd compute analogous updates at all 11 other info sets simultaneously. After thousands of iterations, every info set's regret table stabilizes, and the **average strategy** (weighted average of `σ^1, σ^2, σ^3, ...` per info set) converges to Nash equilibrium.

Kuhn's known Nash equilibrium for player 1: there's a one-parameter family parameterized by `α ∈ [0, 1/3]`:
- With J: bet with probability `α` (occasional bluff)
- With Q: never bet on first action; with `Q` after pass-bet, call with probability `α + 1/3`
- With K: bet with probability `3α` (slow-play sometimes)

CFR finds this strategy automatically through iteration. No reasoning about "should I bluff?" was ever explicit — it emerges from regret minimization.

## Mapping Back to Pokemon

Now you can see what CFR would look like for your project:

- An info set is "everything I, the agent, have observed up to this decision point." Examples: my full team, opponent's revealed Pokemon and revealed moves, current HP/status of revealed Pokemon, current field state, my hidden info (back-row Pokemon, untouched items).
- The opponent has different info sets (their hidden info differs from mine).
- CFR maintains regret tables across all info sets in the (sampled portion of the) game tree.
- Counterfactual values at info sets account for the uncertainty over opponent's hidden state — automatically, through reach probabilities.
- The output is a strategy: for each info set, a distribution over my actions (the "joint action" of move-target choices for both my active Pokemon).

This is the principled framework you'd target for the imperfect-info handling in your project. The complexity is real — vastly more than vanilla MCTS — but the algorithmic structure is now hopefully clear.

Want me to walk through what GT-CFR would do *operationally* on a Pokemon turn now that the CFR foundation is in place? That's the natural next step, and it should make a lot more sense after this.