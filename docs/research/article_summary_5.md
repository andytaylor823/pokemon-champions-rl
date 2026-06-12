Many of these questions cluster around the same conceptual gaps. Let me address each in order, with corrections where needed.

## Deep CFR — How the Network Learns Regret

> For Deep CFR, how does it work to have a NN learn the regret table? Since the regrets are updated iteratively, how do you know when the regret table is considered stable enough to use as training data for the NN? Do you just assume some number of iterations, like 800 steps from the MCTS case? Or is there a more technical way to do this?

Deep CFR (Brown et al. 2019) replaces the explicit regret table with a neural network. Mechanics:

- Maintain a **value/regret network** `V_θ(I) → regret_vector` per player. Input: info-set features. Output: predicted cumulative regret for each action.
- Maintain an **average strategy network** `Π_φ(I) → strategy_distribution` per player. Predicts the running average strategy.
- Run iterations like this:
 1. Sample a trajectory through the game tree using current strategies (derived from `V_θ` via regret matching).
 2. Along the trajectory, compute instantaneous counterfactual regrets at each info set encountered. Use `V_θ` itself or rollouts for the values at terminal/leaf nodes.
 3. Add `(info_set_features, regret_targets)` to a regret replay buffer.
 4. Periodically (every K iterations), retrain `V_θ` from scratch on the replay buffer.
 5. Same idea for `Π_φ`: store `(info_set_features, current_strategy)` tuples in an average strategy buffer, retrain periodically.

There's no "stable point" you wait for. Training is continuous, like AlphaZero. You measure progress via:

- **Exploitability**: how much a best-response opponent could exploit your current strategy. Computed by training a best responder against your fixed strategy and measuring the gap to the game value. As CFR converges, exploitability → 0.
- **Loss on held-out tuples**: standard ML metric.
- **Practical compute budget**: people pick a target number of iterations (millions for large games) and let it run.

For your Pokemon project, you'd run training as long as compute allows and measure exploitability against a held-out best-responder. The Deep CFR paper showed convergence to within bounded distance of vanilla CFR for poker games.

## Tree Node Types — Clarification

> I don't understand how the different node types work in a tree. The decision node for me and my opponent occur simultaneously, and I thought these were the same as CVPN nodes while the search was happening? Also what does PUCT mean? And the CVPN is the NN, which predicts regret tables from info sets, right?

You're conflating several different "node" concepts. Let me untangle.

**Game-theoretic node types** (intrinsic to the game model):

- **Decision node**: a specific player must act. Belongs to one player. Conceptually lives at one info set (multiple histories can share an info set).
- **Chance node**: nature samples from a fixed probability distribution.
- **Terminal node**: game ends, payoffs are determined.

**Search tree status** (a property of where we are in the search, separate from game-theoretic type):

- **Expanded node**: search has already visited and processed this node.
- **Leaf node**: a node in the search tree we haven't expanded past yet. Can be any of the three game-theoretic types above. At a leaf, GT-CFR queries the CVPN to estimate values.

A "leaf node" is not a special game type — it's just where the search tree currently ends. As GT-CFR runs, leaves get expanded and new leaves form deeper.

**Simultaneous moves**: in extensive-form games, "simultaneous" is encoded by sequential decision nodes where the second player's info set bundles together all first-player histories. So in the search tree, structurally it's:

```
Your decision node (info set I_you)
├── child for joint action A_1 → Opp decision node at info set I_opp
├── child for joint action A_2 → Opp decision node at info set I_opp
├── child for joint action A_3 → Opp decision node at info set I_opp
└── ...
```

All those opponent decision nodes have **the same info set `I_opp`**. The opponent's strategy at `I_opp` is one distribution `σ_opp(I_opp, ·)`, applied identically to every child. This enforces that opponent's choice doesn't depend on your choice — i.e., simultaneity.

**PUCT**: stands for "Predictor + UCT" (UCT = Upper Confidence bounds applied to Trees). It's the action selection formula used during tree expansion. AlphaZero formula:

$$\text{PUCT}(s, a) = Q(s, a) + c_{puct} \cdot P(s, a) \cdot \frac{\sqrt{\sum_b N(s, b)}}{1 + N(s, a)}$$

Where `P(s, a)` is the prior from the policy network, `N(s, a)` is visit count, `Q(s, a)` is the running average value. The product structure means: actions with high prior get more exploration, but exploration shrinks as visits grow.

**CVPN**: Counterfactual Value-and-Policy Network. The neural network in Player of Games / GT-CFR. Given an info set, outputs:

- **Counterfactual values**: estimated `v_i(I, ·)` for each player.
- **Policy priors**: estimated action distributions for each player.

It's the analogue of AlphaZero's NN (which had value and policy heads), but the value head produces counterfactual values (per-action) instead of a single scalar value.

It does **not** predict the regret table directly. It predicts counterfactual values, from which CFR+ machinery derives regrets inside the search. (Contrast with Deep CFR, where the network does directly predict regrets — different architectural choice.)

## Counterfactual Reach at Play Time + Meta Priors

> I also don't understand how we compute the counterfactual opponent reach probabilities during real play time. My opponent's hidden info is hidden, so how can I know what their reach probs are? Is this where we use my priors that I'm computing for the Pokemon metagame, e.g. 99% of Incineroar run Fake Out but only 1% run Brick Break? In Kuhn poker, the counterfactual reaches are easily computed, because while the opponent's hidden info (which card they have) is hidden, you know the full scope and probability of all the possible things they do. Does that translate into Pokemon as "I don't know what your full set and backline is, but I have metagame-weighted prior information that tells me what it likely is"? And we use these prior probabilities to estimate the opponent's sets and thus compute reach probabilities?

You've gotten this exactly right. Yes, this is precisely where your meta priors module pays off.

At play time, you run GT-CFR from your current public state. The search:

1. Builds a search tree rooted at your info set.
2. The tree's nodes encode the opponent's hidden information via info sets that bundle all hidden states consistent with public observations.
3. Counterfactual reach probabilities along paths use:
 - Chance probabilities (known)
 - Your own strategy at your info sets (controlled by you)
 - Opponent's strategy at their info sets — **estimated by the CVPN**

The CVPN was trained on self-play data and has learned to output reasonable strategies at opponent info sets. So opponent reach probabilities are computed using the CVPN's estimate of how the opponent plays.

**Where meta priors enter**:

1. **Sampling opponent hidden states for tree construction**: at the root, the search needs to instantiate the search tree. The opponent's info set bundles many possible hidden states (full back row, items, unrevealed moves). Meta priors give a probability distribution over these states. You can:
 - Sample a few representative hidden states (e.g., 10-20) weighted by priors, build the tree over those.
 - Or use the priors directly as the marginal distribution at the root chance node.

2. **CVPN input**: the CVPN takes a state representation that may include meta-prior information explicitly (e.g., "opponent's revealed Pokemon are X, Y; under our priors, their likely back row is Z with probability P(Z)").

3. **Initial regret table at unseen states**: when the search reaches a deep info set, the CVPN provides priors that incorporate meta information.

Your intuition translates directly: "I don't know your full set and backline, but I have metagame-weighted prior information that tells me what it likely is" — exactly right. The reach probabilities computed during search use these priors implicitly via the CVPN's outputs and the tree's chance-node structure.

In Kuhn, the prior on opponent's hand is trivially uniform (no choice involved, just dealing cards). In Pokemon, the prior is meta-derived. Mathematically the same role.

## What the Pokemon Tree Actually Looks Like

> Again, I don't understand what this tree looks like to "walk down". It's not like chess where it's a sequence of me then opponent, repeating. It's (me+opponent) Simultaneously, plus some chance nodes, that may possibly lead to a non-simultaneous opponent or me decision (fainting, pokemon pivoting out). Explain to me more what the tree looks like in this realistic case.

Let me draw out a single turn's worth of tree structure in detail.

Starting at the root (your decision at turn N):

```
[Your decision node @ I_you_N]
   │
   ├─ joint_action_1 (e.g., Garchomp:Earthquake, Rotom-Wash:Hydro Pump @ Charizard)
   │   │
   │   └─ [Opp decision node @ I_opp_N]   ← same I_opp_N for all branches
   │       │
   │       ├─ opp_joint_1 (e.g., Charizard:Heat Wave, Whimsicott:Moonblast @ Garchomp)
   │       │   │
   │       │   └─ [Chance node]
   │       │       │
   │       │       ├─ outcome_1 (all moves hit, average damage, no crits)
   │       │       │   │
   │       │       │   └─ resulting state → [Your decision node @ I_you_{N+1}]
   │       │       │       (or terminal if game ended)
   │       │       │
   │       │       ├─ outcome_2 (Hydro Pump misses)
   │       │       │   └─ ...
   │       │       │
   │       │       └─ outcome_3 (Heat Wave crits Garchomp, Garchomp faints)
   │       │           │
   │       │           └─ [Forced switch decision @ I_you_switch]  ← unilateral!
   │       │               (opp does nothing this sub-turn)
   │       │
   │       ├─ opp_joint_2 (Charizard:Protect, Whimsicott:Moonblast @ Rotom-Wash)
   │       │   └─ ...
   │       │
   │       └─ opp_joint_k
   │
   ├─ joint_action_2 (e.g., Garchomp:Protect, Rotom-Wash:Volt Switch out)
   │   └─ ...
   │
   └─ joint_action_m
```

Key structural points:

1. **The two `I_opp_N` nodes under different joint_actions of yours are the SAME info set**. The opponent's strategy at `I_opp_N` is one distribution, applied identically across all your action branches. This is how simultaneity is encoded.

2. **Chance nodes are where all randomness lives**. After both players commit, the chance node samples (or enumerates) outcomes: accuracy rolls, damage rolls, crits, secondary effects, multi-hit counts.

3. **Forced switches break the simultaneous pattern**. If only your Garchomp fainted (opponent's Pokemon are fine), you make a unilateral switch decision. The tree has a single-player decision node here, no opposing decision node. Then play resumes with the normal turn structure.

4. **Both-sides forced switch** (if both lose Pokemon simultaneously): again two simultaneous decision nodes with info sets that don't reveal each other's switch choice.

5. **Game-end conditions**: terminal node when one side has no remaining Pokemon. Payoff is +1 / -1.

So the tree alternates: `[your decision] → [opp decision] → [chance] → next state → ...` for normal turns, with possible single-player insertions for forced switches.

## PUCT Plus Regret-Based Strategy

> Explain this more to me: "select action to expand based on prior + PUCT bonus + Q from current regrets." What is the formula for this? I thought we didn't use Q here outside of MCTS, and what is PUCT bonus? Is this how exploration vs exploitation is balanced? Say more about this.

In GT-CFR's expansion phase, you walk down the tree to find a leaf to expand. The selection rule combines current CFR strategy (which evolved via regret matching) with PUCT-style exploration.

Schematically (this is roughly the Player of Games formula):

$$\text{score}(I, a) = \sigma^t(I, a) \cdot f(N) + c_{puct} \cdot P(I, a) \cdot \frac{\sqrt{\sum_b N(I, b)}}{1 + N(I, a)}$$

Three terms:

- **`σ^t(I, a)`** is the current strategy at `I` from CFR+ regret matching. This concentrates on actions CFR currently believes are good. It plays the role of "Q from current regrets" — actions with more accumulated positive regret have higher strategy weight.

- **`P(I, a)`** is the CVPN's policy prior. It biases exploration toward actions the network thinks are promising.

- **PUCT bonus** `sqrt(N_total) / (1 + N(a))` shrinks as visits grow, ensuring under-visited actions still get explored.

`f(N)` is a small visit-dependent factor (in some implementations it's just identity).

The role splits cleanly:

- **Strategy term**: exploitation. Use what CFR has learned so far about action quality.
- **PUCT term**: exploration. Make sure even less-favored actions get visited enough to gather data.

This is the same exploration/exploitation trade-off as MCTS, but with CFR's regret-based Q values instead of MCTS's value backups.

At chance nodes during expansion, you don't use PUCT — you sample from the chance distribution. At opponent info sets, you use the same formula but with opponent's strategy and priors.

## How GT-CFR Handles Step Thinkers

> Moreover, generally asking, how does this work vs a non-optimal opponent? In Pokemon it can be helpful to think in terms of "number of step thinkers". Let's assume a zero step thinker always stays in and attacks. A one-step thinker might see an opposing Aerodactyl vs their Charizard and decide to switch / Protect the Charizard to not die to a Rock Slide. A two-step thinker might expect a higher-level opponent to expect me to protect my Charizard and instead use Tailwind, so it's worthwhile to get off free damage while I can. And then it goes back and forth and back and forth whether you should attack / protect the Charizard. How would you expect this to play out in a GT-CFR type scenario, where the "optimal play" can evolve back and forth depending on what the opponent does (and what you CAN REASONABLY EXPECT the opponent to do)? Does this become a Nash equilibrium parametrized by parameter alpha or something?

Your Aerodactyl-vs-Charizard example is exactly the kind of cycle CFR resolves via mixed strategies.

The "step thinking" you described is a chain of best-responses:

- Step 0: Aerodactyl just attacks. Loses to Protect.
- Step 1: Aerodactyl expects Protect, so doesn't attack. Loses to Tailwind.
- Step 2: Aerodactyl expects Tailwind, attacks. Loses to Protect.
- ...

This cycle has no pure-strategy equilibrium. The Nash equilibrium is a **mixed strategy** for both players:

- Aerodactyl plays "Rock Slide" with probability `p`, something else with probability `1-p`.
- Charizard's side plays "Protect" with probability `q`, "Tailwind" with probability `1-q`.

The values of `p` and `q` depend on the payoffs in each cell of the (action × counter-action) matrix. CFR converges to these mixing probabilities via regret minimization — without anyone explicitly reasoning about "what step thinker is my opponent."

When you actually play, you sample from your Nash strategy. Sometimes you Rock Slide, sometimes you don't. The opponent can't reliably exploit either choice because your mixing probabilities are calibrated to make them indifferent.

**Against a non-optimal opponent**: the Nash strategy is *unexploitable* but not necessarily *maximally exploitative*. If your opponent is a pure zero-step thinker who always attacks, you'd do better with a deterministic counter-strategy than with the Nash mix. But you'd also be exploitable yourself if you committed to that counter.

Practical implication: the Nash agent will beat any opponent on expectation but may not crush weaker opponents as hard as a tailored exploit would. For your project's first milestones, Nash is the right target. Later you can add opponent modeling to exploit specific weaknesses — but that's beyond the GT-CFR framework.

The Nash equilibrium in this case isn't variable like the `α` family in Kuhn — Pokemon's action and outcome structure produces specific mixing probabilities, not a range. Variability in Nash strategies happens when a game has redundant actions that achieve the same expected value. Most Pokemon situations don't have this redundancy.

## GT-CFR on Kuhn Poker

> Can you walk through how Kuhn Poker might be played out in a few iterations assuming we were using a GT-CFR (not vanilla CFR) for this? I want to understand the GT part better, so going over this in detail on a simple game would help me understand the process better.

Walking through a few iterations to make the algorithm concrete. We're solving Kuhn for P1's perspective; P2 is also being optimized.

**Setup**: build the tree starting from the root, which represents "the deal hasn't happened yet" — actually, conceptually, the root is the chance node that deals cards. Let me restart from the chance node.

**Initialization (iteration 0)**:

Tree contains just the root chance node (the deal). Below it are six possible deal outcomes (KJ, KQ, JK, JQ, QK, QJ), all with chance probability `1/6`. We don't expand these yet — they're leaves.

For each leaf, query the CVPN. Suppose the CVPN (untrained, random) returns:

- Counterfactual values: random small numbers, say `~0` for both players.
- Policy priors at downstream info sets: roughly uniform.

**Iteration 1 — Expansion phase**:

PUCT-walk from the root. At the chance node, sample one outcome — say KJ (P1 gets K, P2 gets J). Visit `(deal=KJ)`. This leaf gets expanded:

- Add the structure underneath: P1's decision node at `I1_K`, then P2's decision node at `I2_{·}_·` (depending on P1's action), etc.
- Actually, you may expand `k` children — let's say we expand both P1 actions (bet, pass).
- Each new node is a sub-leaf; query CVPN for each.

After this, the tree has the root chance node, deal=KJ branch with `I1_K` and its two children expanded. Other deals (KQ, JK, JQ, QK, QJ) still unexpanded.

**Iteration 1 — Policy update phase**:

Run CFR+ on the current tree. Bottom-up:

- At `I1_K` (under deal=KJ): we have leaf children with CVPN value estimates. Compute counterfactual values for "bet" and "pass" at `I1_K` using these estimates. Update regret table at `I1_K`.
- Strategy at `I1_K` is updated via regret matching.

Other info sets (`I1_Q`, `I1_J`) aren't in the tree yet so they have no update. Their strategies remain whatever CVPN suggested initially.

**Iteration 2 — Expansion phase**:

Walk from root again. The chance node samples another deal — say KQ. Expand the KQ branch: add `I1_K` as decision node (but wait — this is the SAME info set as the one we already expanded under KJ, just with different counterfactual reach).

Important detail: in the tree, `I1_K` appears in multiple places (under KJ and under KQ), because the info set bundles histories from different deals where P1 has K. CFR's regret table at `I1_K` aggregates counterfactual values across all these tree locations.

So expanding KQ adds new tree nodes that contribute to the *same* regret table entry at `I1_K`.

**Iteration 2 — Policy update phase**:

CFR+ runs again. Now `I1_K`'s regret update sums contributions from both KJ and KQ branches.

**Many iterations later**:

Eventually all deals have been expanded, and all info sets have been visited many times. The regret table converges. The average strategy at the root converges to a Nash equilibrium for player 1 (and player 2).

For Kuhn, this terminates quickly because the tree is small. For Pokemon, the tree never fully expands — but enough of it gets explored that the strategy at the root info set stabilizes for our decision.

The key GT-CFR property visible in this walkthrough: the tree starts tiny and grows during search. The CVPN fills in values where we haven't expanded. CFR+ runs on whatever tree we currently have. Each iteration both grows the tree and refines the strategy.

## Your CFR Summary — Corrections

> Is this a fair summary of (GT-)CFR: "you repeatedly simulate a turn many times (across all histories for this info set, or sampling a possible history if space is too large) and at each one compute your regret for that action+history, then do a bunch of aggregation, and eventually you pick the action which minimizes your aggregated regret"? And specifically, "regret" is a function of history and action, where r(h*a) is the difference in utility (value) function between action a and the action you actually chose -- so regret is always zero for the action you *did* choose? Regret can only be computed AFTER you actually make a move, correct? How do we pick the first move then? I'm trying to re-understand at a high-level how this works, but I don't quite understand some of the details. Do you make a real move, THEN compute regret and update your weights? Or do you make a bunch of theoretical moves against a bunch of theoretical histories (that match this info-set) and compute the theoretical regret, and then make your real move to minimize the theoretical regret?

Your summary:

> "you repeatedly simulate a turn many times (across all histories for this info set, or sampling a possible history if space is too large) and at each one compute your regret for that action+history, then do a bunch of aggregation, and eventually you pick the action which minimizes your aggregated regret"

Several specific corrections:

1. **CFR doesn't "simulate a turn many times."** Each CFR iteration is one pass through the game tree (or search tree). The "many" is iterations of the algorithm, not turn replays. Each iteration computes regret updates at every info set in the tree simultaneously, not just one info set.

2. **Regret is per `(info_set, action)`, not per `(action, history)`.** Histories within an info set are aggregated by the counterfactual reach weighting. The regret table has one entry per info-set-action pair, not per history-action pair.

3. **The output is a mixed strategy, not a single action.** Regret matching produces a *distribution* over actions: actions with positive cumulative regret get probability proportional to their regret. You don't pick the single argmin-regret action; you sample from the distribution.

4. **The strategy you actually deploy is the *average* across all iterations**, not the latest iteration's strategy. The average converges to Nash; individual iterations don't.

> "regret is always zero for the action you *did* choose"

Not quite. The instantaneous regret is:

$$r^t(I, a) = v^t(I, a) - v^t(I, \sigma^t)$$

where `σ^t` is the (mixed) strategy at I. If σ is a single pure action `a^*`, then `v(I, σ) = v(I, a^*)`, and the regret of `a^*` is zero. But CFR's strategies are typically mixed, so regret is zero for the strategy as a whole, not for any single action.

Concretely, the action-weighted sum of regrets equals zero by definition:

$$\sum_a \sigma^t(I, a) \cdot r^t(I, a) = 0$$

But individual actions can have positive or negative regret.

> "Regret can only be computed AFTER you actually make a move"

No. CFR computes regret *during the algorithm*, on the model of the game, without playing any moves. It's purely computational. When the algorithm finishes, you have a strategy. *Then* you play a move by sampling from that strategy. The regret computation happens entirely before you act.

> "Do you make a real move, THEN compute regret and update your weights? Or do you make a bunch of theoretical moves against a bunch of theoretical histories... and compute the theoretical regret, and then make your real move to minimize the theoretical regret?"

The second one, with a refinement: the "theoretical moves" aren't really moves — they're values computed by traversing the game tree under hypothetical strategies. CFR runs the tree model with each player's current strategy, computes counterfactual values, derives regret updates, generates next iteration's strategies. After many iterations, the average strategy is your output. You then sample one real move from the strategy at your current info set.

## "PUCT for Expansion, Regret Matching for Strategy"

> I'm coming back to this after awhile away. Can you re-explain this part in your previous response: "PUCT for expansion, regret matching for strategy" -- how do we compute regret and use it to determine strategy again? Do we choose the action which minimizes cumulative regret, where regret is computed as the (non-negative) diff between the utility from action a and the selected action? Also explain this a bit more: "Ouptut at root: average strategy from CFR+" What is this average strategy? Is it a mapping between action a (and info set I) and the expected utility / probability? What's the difference between utility and probability here?

Decomposing this once more:

**Expansion** (growing the search tree): when the algorithm needs to pick which path to walk down for tree growth, it uses **PUCT-style selection**, which combines the current CFR strategy (from regret matching) with an exploration bonus and CVPN priors. This determines tree shape — which actions get more search effort.

**Strategy** (the output of CFR): the actual probability distribution at each info set is computed via **regret matching** from the cumulative regret table. This is the policy you'd actually play with.

These are two different "policies" in a sense:

- The PUCT selection rule is used internally during the search to decide where to expand.
- The regret-matching strategy is the search's output — the policy you sample from when actually playing.

The PUCT rule includes regret-matching values as one of its components (the strategy term `σ^t(I, a)`), so they're not independent — they share inputs. But they serve different purposes.

**Average strategy** at the output:

$$\bar{\sigma}(I, a) = \frac{\sum_{t=1}^T t \cdot \sigma^t(I, a)}{\sum_{t=1}^T t}$$

(Linear weighting, where later iterations matter more — this is CFR+ specific.) This is a probability distribution over actions, NOT a utility map. For info set `I`, the value `σ̄(I, a)` is "the probability you should play action `a` here, after T iterations of CFR+."

**Difference between utility and probability**:

- **Utility `u(I, a)`** or **counterfactual value `v(I, a)`**: scalar in some range (typically `[-1, +1]` or in the payoff scale). "How good is action `a` at info set `I` in expectation."
- **Strategy / probability `σ(I, a)`**: scalar in `[0, 1]`. "What fraction of the time you should play action `a` at info set `I`." Sums to 1 across actions.

CFR's output is the strategy (probabilities), not utilities. Utilities are intermediate computations used to derive regrets, which determine strategies.

When you actually play: look up your info set, get the probability distribution `σ̄(I, ·)`, sample one action from it.

Want me to walk through any of these in even more detail, or start digging into state encoding (which I think is the next major design question for your project)?