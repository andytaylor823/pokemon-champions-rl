The [TDS article](https://towardsdatascience.com/why-player-of-games-is-needed-a01505c4bad7/) is genuinely poorly written — it references equations that aren't visible in the text and skips the key definitions. Let me rebuild this from first principles. Regret first, then CFR, then GT-CFR.

## Regret: The Foundational Concept

Regret is a precisely defined mathematical quantity from **online learning theory**. The setup:

You face a sequence of decisions over `T` rounds. At each round `t`, you pick an action `a_t` from some set `A` and receive a payoff `u(a_t)`. After the game, you compare your actual total payoff to the best you *could have* gotten by playing a single fixed action the whole time:

$$ \text{Regret}(T) = \max_{a^* \in A} \sum_{t=1}^T u_t(a^*) - \sum_{t=1}^T u_t(a_t) $$

In words: "How much better would I have done if I'd known in hindsight which action was best, and just played that one every round?"

A learning algorithm is called **no-regret** if `Regret(T) / T → 0` as `T → ∞`. That is, average regret goes to zero — over time, you're playing essentially as well as the best fixed action in hindsight.

**Worked rock-paper-scissors example.** Suppose you play RPS for 10 rounds against an opponent. You play `R, R, P, S, R, P, P, S, R, R`. Their plays end up being `S, P, R, R, P, S, R, P, S, P`.

Your payoffs (win=+1, draw=0, lose=-1) are: `+1, -1, -1, -1, -1, -1, -1, -1, -1, -1` = -8 total.

Now compute regret: what if you'd played all R? You'd score `0, +1, -1, -1, +1, 0, -1, +1, 0, +1` = +1. Regret of "all R" vs your actual play = 1 - (-8) = **9**. You "regret" not having just spammed Rock — you'd have done 9 points better.

Compute it for all P (you'd get `+1, -1, +1, -1, -1, +1, -1, -1, +1, -1` = -2): regret = -2 - (-8) = **6**.

For all S: would get `-1, +1, -1, +1, -1, -1, +1, -1, -1, +1` = -2: regret = -2 - (-8) = **6**.

Your maximum regret is 9 (vs always-R). A no-regret algorithm would, over enough rounds, drive this to 0 — meaning your actual play would be approximately as good as the best fixed action.

## Regret Matching

This is the algorithm that takes us from "regret is a measurement" to "use regret to play."

Define **cumulative regret** for action `a` after round `T`:

$$ R^T(a) = \sum_{t=1}^T \left[ u_t(a) - u_t(a_t) \right] $$

This is the cumulative "I should have played `a` instead of what I did" feeling.

**Regret matching algorithm**: on round `T+1`, pick action `a` with probability proportional to its **positive** regret:

$$ \sigma^{T+1}(a) = \frac{R^T_+(a)}{\sum_b R^T_+(b)} $$

where $`R^T_+(a) = max(R^T(a), 0)`$. If all regrets are non-positive, play uniformly random.

**Theorem (Hart & Mas-Colell, 2000)**: regret matching is a no-regret algorithm.

Concrete RPS application: after 10 rounds your regrets are `R: 9, P: 6, S: 6`. On round 11, regret matching plays:

- R with probability `9 / (9+6+6) = 9/21 ≈ 0.43`
- P with probability `6/21 ≈ 0.29`
- S with probability `6/21 ≈ 0.29`

Notice this is mixed — not pure. Over many rounds against a fixed opponent, this converges to the best response to their distribution.

**The killer fact for game theory**: in a two-player zero-sum game, if **both** players use a no-regret algorithm and you average their strategies over all rounds, the average strategies converge to a **Nash equilibrium** of the game.

This is huge. It means: if I run regret matching against an opponent who's also running regret matching, our long-run average strategies are unexploitable. We don't need to compute the Nash equilibrium directly — self-play with no-regret learning *finds it*.

But this only works for **normal-form games** (single-shot decisions). Real games have sequences of decisions and information asymmetry. That's where CFR comes in.

## Counterfactual Regret Minimization (CFR)

CFR extends regret matching to **extensive-form games** — games with sequential decisions and imperfect information. Three new concepts you need:

**1. Information set (info set, denoted `I`)**

A set of game states (called "histories" `h`) that are indistinguishable to the acting player. In poker: when it's your turn after the opponent bets, all possible opponent hand cards put you in different histories, but you can't tell them apart. So those histories form a single info set. **You must play the same strategy at every history in an info set** (you have no way to differentiate them).

**2. Reach probability**

The probability of arriving at history `h`, given the strategies of all players plus chance. Decompose it:

$$ \pi(h) = \pi_i(h) \cdot \pi_{-i}(h) $$

- `π_i(h)`: contribution to reach probability from player `i`'s own actions
- `π_{-i}(h)`: contribution from opponent + chance

The "counterfactual" reach probability is `π_{-i}(h)` — the probability of getting here if player `i` *was trying to* reach it (i.e., we ignore player `i`'s own probabilities along the path).

**3. Counterfactual value of action `a` at info set `I`**

$$ v_i(I, a) = \sum_{h \in I} \pi_{-i}(h) \cdot u_i(h \cdot a) $$

Where `u_i(h · a)` is player `i`'s expected utility after history `h` followed by action `a`. This sums "if I were trying to reach this info set, weighted by how each history within it could plausibly arise from opponent + chance, what's my expected payoff if I pick `a` here?"

**Counterfactual regret of action `a` at info set `I` after iteration `T`**:

$$ R^T(I, a) = \sum_{t=1}^T \left[ v_i^t(I, a) - v_i^t(I, \sigma^t) \right] $$

Where `σ^t` is the strategy played at iteration `t`. In words: "How much better would I have done at this info set if I'd always played `a` here, vs. what I actually did?"

**The CFR algorithm**: maintain cumulative counterfactual regret tables for every info set in the game. Each iteration:

1. Traverse the entire game tree (or sample paths through it).
2. For each info set encountered, compute counterfactual values for each action.
3. Update regret tables.
4. Generate next iteration's strategy via regret matching at each info set.
5. Also accumulate the average strategy over all iterations.

**Theorem (Zinkevich et al., 2007)**: in two-player zero-sum games, the average strategy from CFR converges to a Nash equilibrium. Total regret grows as `O(sqrt(T))` so per-iteration regret → 0.

**Why this is the right tool for imperfect information.** Notice CFR operates at the **info set** level, not the determinized state level. It directly models "I don't know which world I'm in, but I have to commit to a strategy that handles all of them." Strategy fusion (the failure mode of determinized MCTS) doesn't happen because you're never pretending to know hidden information.

Compare to determinized MCTS: that algorithm samples worlds and finds the best action *per world*, then averages. CFR finds the best **single mixed strategy** that works across worlds, weighted by their counterfactual reach probabilities.

## CFR+ Improvements

Vanilla CFR works but converges slowly. CFR+ (Tammelin et al., 2014) is a battery of empirical improvements that's currently the standard:

**Regret-matching+**: clip cumulative regrets to be non-negative each iteration. If regret for action `a` would go negative, clamp to 0:

$$ R^{T+1}_+(I, a) = \max(R^T_+(I, a) + r^T(I, a), 0) $$

This prevents "ghost regret" where an action accumulated lots of negative regret early and is now permanently shut out.

**Linearly weighted strategy averaging**: when computing the average strategy, weight iteration `t` by `t` instead of `1`. Later iterations matter more because they're closer to converged. This makes the average strategy converge faster.

**Alternating updates**: instead of updating both players' strategies simultaneously each iteration, alternate. Player 1 updates this iteration, player 2 next, etc. Empirically this speeds up convergence.

These three changes give CFR+ ~1000x faster convergence in practice for poker. CFR+ was the algorithm that essentially solved heads-up limit Texas hold 'em (Bowling et al., 2015).

## CFR Doesn't Scale Naively

Vanilla CFR/CFR+ traverses the **entire game tree** on each iteration. Heads-up limit Texas hold 'em has ~10^14 info sets — solvable with abstraction tricks, ~70 CPU-years of compute. No-limit hold 'em has ~10^160 info sets — utterly infeasible to enumerate.

For larger games (no-limit poker, your Pokemon project), you need approximations:

- **MCCFR (Monte Carlo CFR)**: sample paths through the tree instead of full traversal. Various sampling schemes (outcome, external, public-chance). Trades higher variance per iteration for cheaper iterations.
- **Deep CFR / Neural CFR**: replace the regret table (which is O(num info sets)) with a neural network that predicts regrets from features of the info set. Generalizes across info sets the way AlphaZero generalizes across board states. Used by DeepStack and Libratus.
- **Subgame solving**: don't solve the whole game; when you reach a subgame, run CFR on just that part using neural nets to estimate values at the subgame boundary.

## GT-CFR (Growing-Tree CFR) — What PoG Actually Does

Now we can finally make sense of the article. GT-CFR is the CFR analogue of MCTS: instead of operating on a fully-enumerated game tree, you **grow the tree on the fly** during search, using a neural network at the leaves.

Two interleaved phases each iteration:

**Phase 1 — Policy update (CFR+ on the current tree)**

Treat the current tree as if it's the whole game. Run a CFR+ update step:

1. Traverse the current tree.
2. At each info set in the tree, compute counterfactual values for each action.
3. At leaf nodes (which haven't been expanded yet), query the **CVPN** (Counterfactual Value-and-Policy Network) to get estimated counterfactual values and policy. This is the analogue of AlphaZero's value head.
4. Update regret tables and policies via regret-matching+ throughout the tree.

**Phase 2 — Tree expansion (PUCT-guided growth)**

Like MCTS, walk down the tree using a PUCT-style selection rule (essentially the same formula as AlphaZero — the article shows it as `pUCT`), pick a leaf, expand it by adding its children. Each expansion grows the tree by a few nodes.

**Iterate**: alternate between policy updates on the current tree and tree expansion, for some number of iterations. Then return the **average strategy at the root** (CFR's output, not MCTS's visit counts).

**The CVPN**

Same role as AlphaZero's NN, but trained to predict different quantities:

- **Value head**: outputs **counterfactual values** for each player at this info set (not just a single scalar). Counterfactual values, not raw values, because that's what CFR needs at leaves.
- **Policy head**: outputs the **average strategy** the network expects from this info set.

Trained via self-play, similar to AlphaZero. The training targets are:
- Counterfactual values from completed search (analogous to AlphaZero's outcome `z`)
- Average strategies from completed search (analogous to AlphaZero's MCTS visit distribution)

So: same self-play / search-distillation paradigm as AlphaZero. The differences are (a) what the network predicts (counterfactual values, not just scalar value) and (b) how the search is conducted (CFR+ updates instead of UCB updates).

## Putting It Together: What the Article Was Trying to Say

The TDS article is structured around three claims:

1. **AlphaZero (MCTS-based)** works for perfect-information games but fails on imperfect-information ones because MCTS doesn't converge to Nash equilibrium with hidden information (this is exactly the strategy fusion problem you and I discussed).

2. **PoG uses GT-CFR + CVPN** instead, which gives:
 - CFR's correctness for imperfect-info games (operates on info sets, converges to Nash)
 - MCTS's scalability to large games (grow tree on the fly, NN at leaves)
 - Self-play training (same paradigm as AlphaZero)

3. The result is a single algorithm that handles both perfect-info games (chess, Go) **and** imperfect-info games (poker) with strong performance in both.

The article's unclear bits — Q values, regret-matching+, linearly weighted policy averaging, the CVPN equations — are all standard CFR+ machinery being adapted into the tree-growth setting. None of it is novel; all of it has been used in CFR+ and Deep CFR for years. The novelty of PoG is the *combination*: CFR+ as the policy-update rule + neural-network-guided tree growth + a CVPN that predicts counterfactual values.

## Why This Matters for Your Pokemon Project

The PoG paper (Schmid et al. 2021, [arXiv:2112.03178](https://arxiv.org/abs/2112.03178)) is the most directly applicable algorithmic framework to what you're trying to build. Pokemon doubles is:

- Sequential and imperfect-information (like poker)
- Large state and action space (needs NN-guided search like AlphaZero)
- Stochastic outcomes (chance nodes, like poker)
- Two-player zero-sum (CFR convergence guarantees apply)

This is exactly the design space PoG was built for. The mapping:
- "Public state" in Pokemon ≈ everything both players have observed (revealed Pokemon, used moves, items revealed via effects, HP, status, field conditions)
- "Private state" ≈ each player's hidden info (back-row Pokemon, unrevealed items/moves)
- CVPN takes the public state plus your private state and predicts counterfactual values + policy

The downside: GT-CFR is significantly harder to implement than MCTS. CFR+ machinery, info-set bookkeeping, counterfactual value computation, and the CVPN training pipeline are all more involved.

My earlier recommendation stands: build vanilla AlphaZero infrastructure first on a perfect-info simplification of Pokemon. Then you'll have all the engineering pieces (sim integration, encoding, NN, training loop) in place. Swapping the search algorithm from MCTS to GT-CFR is then a focused replacement of one component, not a from-scratch rebuild. The CVPN architecture can reuse the AlphaZero NN backbone almost wholesale.

For deeper reading: the PoG paper itself ([arxiv.org/abs/2112.03178](https://arxiv.org/abs/2112.03178)), the original CFR paper (Zinkevich et al. 2007), and the Deep CFR paper (Brown et al. 2019) are the foundational set. The PoG paper has a clearer presentation than the TDS summary you're reading.

Want me to walk through a concrete worked example of CFR on a tiny game (Kuhn poker is the standard toy problem — 3 cards, ~12 info sets, fully solvable by hand) so you can see exactly how the regret tables update? That's usually the moment CFR clicks for people.