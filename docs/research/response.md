Excellent set of questions. Let me work through each carefully — and I'll concede where you've made me sharpen my thinking (your credit assignment argument is largely right and I was imprecise).

## The UCB Equation: All Terms Defined

> Explain all the terms you use in the equation in select for MCTS. Q is the average value of what, averaged over what? Average UCB over all other children under that root? What does UCB stand for?

UCB stands for **Upper Confidence Bound**. AlphaZero technically uses a variant called **PUCT** (Predictor + UCT, where UCT itself is "Upper Confidence Bounds applied to Trees"). The formula:

$$a^* = \arg\max_a \left[ Q(s,a) + c_{puct} \cdot P(s,a) \cdot \frac{\sqrt{\sum_b N(s,b)}}{1 + N(s,a)} \right]$$

Term by term, for a state `s` and action `a`:

- **`Q(s,a)`**: The **average value** of leaf evaluations from all simulations that have passed through this specific edge `(s,a)`. Concretely: each simulation reaches a leaf node, the NN evaluates that leaf and returns a value `v`. That value is propagated back up the path. For edge `(s,a)`, you maintain a running total `W(s,a) = Σv` across all simulations through this edge, and `Q(s,a) = W(s,a) / N(s,a)`. So Q is **not** averaged over siblings — it's the average value seen by simulations that chose this particular action from this particular state.

- **`N(s,a)`**: Visit count for edge `(s,a)`. How many simulations have selected action `a` from state `s`.

- **`Σ_b N(s,b)`**: Sum of visit counts over all sibling actions from state `s`. Effectively the parent's total visit count.

- **`P(s,a)`**: The **prior probability** for action `a` at state `s`, output by the policy head of the neural network when state `s` was first expanded.

- **`c_puct`**: A scalar hyperparameter (~1.0 to 5.0). Higher = more exploration.

The first term `Q(s,a)` is **exploitation** (pick what's been good). The second term is **exploration** with two flavors: it's higher for actions the network thinks are promising (`P(s,a)` large), and it shrinks as you visit them more (`N(s,a)` in denominator), so it eventually loses to Q.

## Policy Vector vs Value Scalar

> Explain more about what the NN returns. What is the difference between the policy vector and the win prob? What does the policy vector mean -- "At this child node, what is the next child I should call?"?

The neural network has a **shared backbone** (some layers that compute features from the state) which splits into **two heads**:

- **Policy head**: outputs a vector of length |action space|, then softmax. Each entry is a probability. Semantically: "given this state, which action would a strong player pick?" or "which actions are worth exploring?" It's not "which child to call" — MCTS uses these probabilities as priors `P(s,a)` in the UCB formula, biasing exploration toward promising actions.

- **Value head**: outputs a single scalar in `[-1, 1]` (via tanh). Semantically: "from this state, what's my expected game outcome?" +1 = certain win, -1 = certain loss, 0 = even.

These answer different questions:

| Question | Answer | Used for |
|---|---|---|
| "What should I do here?" | policy vector | priors in MCTS exploration |
| "How am I doing here?" | value scalar | leaf evaluation in MCTS backup |

## NN Called Once Per Unique State

> The neural network is only called once at each unique child, correct? So for chess, there's a lot of unique boardstates. In pokemon, HP is pseudo-continuous, so the neural network may need to be called a lot, right? Each new child is basically a unique new state. That seems bad, so I'm interested in thinking more about this a few steps later from this discussion.

Yes, exactly once per unique node in the tree — when a leaf is first expanded. After that, the policy priors and value are cached on the node.

Your concern about Pokemon's near-continuous state space is sharp. In chess, transpositions are common (different move sequences reach the same board position), so the same node might be hit many times across simulations. In Pokemon, between HP percentages, status condition timers, stat boost stages, weather/terrain countdowns, and partial information about both teams, **you essentially never see the same state twice**. Practical implication:

- **Within a single MCTS search** (one decision, ~800 sims): you'll still revisit nodes, because each simulation walks down the *same* tree from the root. The tree expands by one new leaf per simulation. So the NN gets called ~800 times per turn, not 800 × tree-depth times.
- **Across MCTS searches** (across turns and games): essentially every leaf is unique, so caching across searches is mostly useless.

The big consequence isn't inference cost — it's that the network must **generalize**. It has to interpolate from training states to unseen test states via learned features. This is exactly what neural networks are good at, but it places a high burden on your state encoding (semantically meaningful features matter more when no two states match exactly). This is why your encoding work later will be load-bearing.

## What Gets Backpropagated; NN is Static

> What exactly do you backpropagate? Just the counts? You don't update the NN, right? That's presumed to be static.

You backpropagate **the leaf value `v`** (not just counts — the value is what makes Q updateable). For every edge `(s,a)` on the path from root to leaf:

- `N(s,a) += 1`
- `W(s,a) += v`  (value gets sign-flipped at opponent nodes — propagating "my win" up to opponent's perspective inverts sign)
- `Q(s,a) = W(s,a) / N(s,a)`  (recompute, or maintain incrementally)

The NN is **frozen during MCTS**. You only update the network during training phases, between batches of self-play games. During search, the NN is a black-box oracle.

## Why Not Just Trust the NN's Policy Directly?

> Then at the end of all your MCTS for a move / turn, how do you decide which move to select? Why isn't it just the NN's call anyway, since it predicts win probability for each child?

Great question, and the answer is the heart of why AlphaZero works.

The NN's policy is a **single forward pass** — it's a fast pattern-matching judgment. It might output "action A: 0.6 probability of being best." But if you actually simulate playing A and look 5 plies ahead, you might find A leads to a position where every continuation is losing. The NN didn't have time to look ahead — it gave a snap judgment.

MCTS does the lookahead. It uses the NN's prior as a *guide* (so it doesn't waste sims on obviously bad actions), but then it explores actual consequences and lets the value backups override the prior when reality contradicts the network's intuition.

Concrete chess analogy: a strong player given 1 second per move plays worse than the same player given 10 minutes. Same intuition, but more search produces better moves. MCTS converts compute into play strength on top of whatever the NN gives you.

## What MCTS Looks Like After All Children Are Visited

> What do you do when you've been simulating a turn for awhile, and you have some number of counts such that all first-level children from the root have been seen at least once? Do you choose a child based on maximizing UCB, then repeat until you hit a new, unseen leaf, then backpropagate? 

Yes, exactly what you described. The procedure:

1. Start at root.
2. At current node, select child via `argmax(Q + U)` (PUCT formula above).
3. Move to that child. If it's an existing node, repeat from step 2.
4. If it's a new (unexpanded) leaf, **expand it**: call NN, get `(policy, value)`, attach policy as priors to its children edges.
5. **Backpropagate** the value `v` up the path you walked, updating N and W for every edge.
6. Done with one simulation.

Each simulation potentially walks deeper than the last because the tree grows by one new node per sim. After 800 sims, you have a tree with 800 nodes (well, +1 for root). The shape of that tree is heavily skewed: actions with high Q+U get explored repeatedly and develop deep subtrees; actions with low Q+U barely get visited.

## "Visit Count Distribution as Training Target"

> What does this statement mean? Explain it in more detail: "The visit count distribution at the root becomes the training target for the policy head — this is how the network learns. It tries to predict what MCTS would have concluded, so over time raw network policy gets closer to MCTS-improved policy, which means search gets more efficient, which means MCTS produces even better policies, and so on."

Suppose at state `s` after 800 sims, the root's visit counts look like:

- action A: 400 visits
- action B: 250 visits
- action C: 100 visits
- action D: 50 visits

Normalize: `π_MCTS = [0.500, 0.3125, 0.125, 0.0625]`. This is the **MCTS-improved policy** at state `s`.

When training, you take the (state, MCTS_policy, outcome) tuple `(s, π_MCTS, z)`:

- **Policy loss** = cross-entropy between `π_NN(s)` (current network's policy output) and `π_MCTS`. The network is trained to make its raw policy output closer to what MCTS concluded.
- **Value loss** = `(v_NN(s) - z)²` where `z` is the eventual game outcome (+1 or -1).

Why this is a virtuous cycle:

- The NN starts outputting roughly uniform priors. MCTS does work to figure out what's actually good. Visit counts reflect that work.
- We train the NN to match those visit counts. Now, next time the NN sees a similar state, its raw policy output is closer to what MCTS used to laboriously figure out.
- Now MCTS starts with better priors → it focuses sims on actually-good actions → produces even better visit count distributions → those become the new training target → NN gets better still.

Over thousands of training iterations, the NN absorbs the MCTS improvements. This is **policy distillation** from MCTS into the network.

## Self-Play From Scratch

> Training the value network, you say this: "Play a complete self-play game (both sides use MCTS + current network).". Starting from scratch, the current network is just random, correct? Or maybe we guide it a little, saying don't attack your partner or whatever, but it's basically random, right? So we do self-play with a random network (basically what I suggested), then store a bunch of training data from this random network. Then we build version 2 of the NN and repeat self-play with this new, less-random network to build more training data, and use that to build version 3 and so on. Is that correct? I'm still a bit lost on this statement: "Train the network on these tuples: policy head tries to match MCTS visit distributions, value head tries to predict the eventual game outcome."

Yes, your understanding is right with one important refinement.

Iteration 0: NN initialized randomly. Policy head outputs ≈uniform; value head outputs ≈random scalars. You play self-play games where both sides use MCTS + this random network.

Even with a random network, MCTS produces *slightly* better than random play. Why? Because once a simulation reaches a terminal state (game ends in the tree), the actual game-end value (+1/-1) gets backpropagated. So `Q` values reflect *real* outcomes, not just NN guesses, for nodes where simulations reach terminations. Even random play occasionally results in immediate wins/losses, and MCTS captures that. So self-play with a random network produces:

- Mostly noisy data
- But occasionally captures "if I attack a fainted slot it's wasted" or "if I switch into a super-effective move I lose the Pokemon" — gross signal

You train v1 of the NN on this data. v1 is slightly less random than v0. Self-play with v1 produces slightly better data. v2 is better still. Etc.

For chess, AlphaZero went from zero to superhuman in ~4 hours on 5000 TPUs. For your project, expect orders of magnitude more iterations to get to mediocre play, and the early iterations will be very noisy.

You're roughly right that "we do self-play with a random network (basically what I suggested)" is the start. The important difference from your original framing is that even at iteration 0, you're not picking actions randomly during play — you're picking them via MCTS, which uses the (random) NN as a guide but does still benefit from terminal value backups.

## What is the "Policy Head"?

> I think I need a lot more explanation about the terms "policy head" and how that applies here and in AlphaZero.

Architecturally:

```
state → [shared layers / backbone] → features
                                       ├→ [policy head layers] → softmax → policy vector
                                       └→ [value head layers] → tanh → value scalar
```

Two outputs, one shared body. Training updates all parameters jointly.

The policy head is just the part of the network whose output is interpreted as a distribution over actions. "Head" is jargon for "output branch with task-specific layers."

In AlphaZero for chess, the action space is ~4672 possible moves (8x8x73 representation), and the policy head outputs a 4672-dimensional vector. For your Pokemon project, the action space is ~100 joint actions per turn (or you might decompose it into per-Pokemon action heads, which is a design choice).

## Strategy Fusion — Concrete Example

> I also don't understand what you're saying about strategy fusion. I understand the example of different moves being better/worse in different situations, but I don't see how that's a problem.

Here's a worked example. You're at turn 1 against an opponent's Charizard. Their item is hidden. Your prior says:

- 50% Charizardite Y (special attacker, slow)
- 50% Choice Scarf (revealed Earthquake, very fast, locked into Earthquake)

Your relevant choices:
- Switch to your Water-type (resists Heat Wave but dies to Earthquake)
- Switch to your Flying-type (immune to Earthquake but melted by Heat Wave)
- Stay in with current Pokemon (mediocre vs both)

Determinized MCTS samples opponent items per simulation:

| Sample world | Best response | Win rate |
|---|---|---|
| Charizardite Y world | Switch Water | 0.85 |
| Choice Scarf world | Switch Flying | 0.85 |

Now: in the Y-world simulations, MCTS explores all your actions. "Switch Water" wins 85%, "Switch Flying" wins 30% (you eat a Heat Wave on the Flying mon), "Stay in" wins 50%.

In the Scarf-world simulations: "Switch Water" wins 20% (eats Earthquake), "Switch Flying" wins 85%, "Stay in" wins 50%.

Aggregated across both world types:

- Switch Water: 0.5 × 0.85 + 0.5 × 0.20 = **0.525**
- Switch Flying: 0.5 × 0.30 + 0.5 × 0.85 = **0.575**
- Stay in: 0.5 × 0.50 + 0.5 × 0.50 = **0.500**

MCTS recommends "Switch Flying" with 0.575 expected value. Looks fine, right?

**Here's the problem.** The issue isn't the *aggregation* — it's the *strategy* that MCTS implicitly assumed during search. In each Y-world sim, MCTS played "Switch Water" because that's optimal in that world. In each Scarf-world sim, MCTS played "Switch Flying." But you can only pick ONE action at decision time; you don't know the world.

The Q value for "Switch Water" in the search tree was computed assuming subsequent play conditioned on knowing the world. After turn 1, in Y-world sims, your turn 2+ play knows you're in Y-world. In Scarf-world sims, your turn 2+ play knows you're in Scarf-world. **But in reality, on turn 2 you also won't know the world** (until the opponent reveals more).

The Q values you computed are inflated relative to reality. They reflect "expected value if I get to play optimally per world" not "expected value of a single fixed strategy that must work across all worlds."

In games where the right answer is "play a mixed strategy" (sometimes switch Water, sometimes switch Flying, randomly, to be unexploitable), determinized MCTS won't even consider this — it always commits to whichever pure action looked best in expected value.

The "fusion" name comes from: information from different information-set worlds is being inappropriately fused, as if you had access to the hidden information when picking your strategy.

This is why the project notes mention CFR or Player-of-Games style approaches as alternatives — they reason at the information-set level rather than the determinized-state level.

## Credit Assignment — You're Mostly Right

> I don't agree with your point 4 about the branching factor. In your example, you make a bad turn 1 and that ends up losing you the game on turn 8. The credit belongs with the turn 1 misplay -- but isn't this handled by adequately training the NN? If you grossly misplay turn 1, then I would expect you to be in a state where the NN predicts your win probability for all leaf nodes to be low, so we can then correctly assign the critical failure to be the misplayed turn 1. If your win probabilties for the best child nodes hover around 60-70% say, then you make a bad turn and your win probabilities drop down to 20-30%, it's clear that THAT turn was what deserves the credit. Perhaps I'm missing something, but this seems relatively straightforward to me.

I'm going to walk back my point 4. Your counterargument is solid: in the limit of a well-trained value network, MCTS at turn 1 will correctly identify that "set Trick Room" leaves the team in a state where the value head predicts 0.7, while "don't set Trick Room" leaves it at 0.2. The credit assignment is implicitly handled by the value function.

The remaining nuance is about **training-time sample efficiency**, not about whether the algorithm correctly assigns credit in principle. Specifically:

- The value network learns from `(state, outcome)` pairs. Both "good turn 1" and "bad turn 1" trajectories produce some wins and some losses, because turn 2-8 also vary. The network has to learn to discriminate via signal in the state representation.
- In Pokemon, the difference between "set Trick Room" and "didn't set Trick Room" might just be one bit in the state encoding (`trick_room_active=True/False` plus a counter). Small state delta, big strategic delta. The network needs many examples to learn that this one feature is highly diagnostic.
- In chess, the state delta from a good vs bad move is usually larger (you actually moved a piece), so the discriminating feature is more easily learnable.

So the *learning problem* is harder, but you're right that the *algorithm* handles credit assignment correctly given enough training. I overstated this as a structural problem when it's really a sample-efficiency problem. For your goal of "better than random," this won't be a blocker.

## Why MCTS Improves on the Network

> Explain this statement to me more: "Given any network, MCTS produces a better policy than the network alone.". Why is that?

This is the **policy improvement theorem** for MCTS, and there's both an intuitive and a formal answer.

**Intuitive**: the NN gives a one-shot pattern-matched judgment. MCTS uses that judgment but also actually explores consequences. When the network's intuition is wrong about a particular action, MCTS catches it via the value backups from leaves.

Suppose the NN says action A has prior 0.6, action B has prior 0.3. MCTS starts exploring A more (high prior). After some sims through A, it finds A leads to states where the value head gives -0.4 (bad). Q(s, A) drops to -0.4. Now A's UCB score = -0.4 + (exploration term shrinking). Meanwhile B's UCB score = 0 (untouched, still has high U term). MCTS shifts to B. After exploring B, it finds B reaches positions with value +0.3. Q(s, B) = +0.3. Now MCTS settles on B.

The visit counts now reflect: B got more visits than A, despite A having a higher prior. MCTS "discovered" that the network's prior was wrong.

**Formal**: Under reasonable assumptions, the policy obtained by normalizing MCTS visit counts has expected value ≥ the policy obtained by sampling from the network's policy directly. The lookahead acts as a contraction toward the true optimal policy. This is a generalization of the policy iteration theorem from dynamic programming.

The clean implication: **as long as your value head is reasonably accurate, MCTS produces play that exceeds raw NN play**. This is what lets you use MCTS as a teacher signal during training. The student (NN policy head) is being trained to match a teacher (MCTS visit distribution) that is provably stronger than the student. So training drives the student up. As the student improves, the teacher (which uses the student as priors) also improves. The fixed point of this process is very strong play.

This decoupling — using a weak-but-fast policy network and a strong-but-slow MCTS teacher — is the key engineering insight of AlphaZero. The network alone would be limited by its parameter count and inference speed. The MCTS teacher converts compute into search depth, and that gets distilled back into the network.

---

If you want, the next pieces I'd tackle are: (1) handling simultaneous-move nodes (the structural challenge for Pokemon-vs-Chess), and (2) state encoding design — what features do you actually feed the network, and how do you handle hidden information in the input.