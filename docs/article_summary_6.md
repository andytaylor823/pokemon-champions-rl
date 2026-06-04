> Now that I understand this a bit better, I have some more questions:

> Since cumulative regret is iteratively computed, the corresponding strategy (which depends on the ratios of each cumulative regret to the total sum) will vary according to how many iterations you have run so far. Over time, it will converge such that each individual iteration's contribution is negligible, but when? This is what I was intending to ask in my first question previously. How do you know when you've done enough iterations and the strategy is stable enough to train a NN on? Or am I thinking about this wrong; you say that training is continuous -- does this mean that you never "reset" the cumulative regret, and that as you iterate for a bunch of steps, you continue to update your strategy and cumulative regrets? Even so, how do you know when to move on to the next step -- act on the info you've gained and move to turn 2 of the pokemon battle?



A lot of these questions share a single root confusion: there are **two completely separate loops**, each with its own "iteration" counter, its own notion of "convergence," and its own relationship to the network. Almost every question below gets clearer once we pin those down, so I'll start there and refer back to it.

## The Two Loops (read this first)

**Inner loop — the search at one decision point.** When it's your turn (say turn 1 of the battle), you build a search tree rooted at the current public state and run CFR iterations on it: $t = 1, 2, ..., T$. Each iteration traverses the current tree, updates an **ephemeral cumulative regret table**, and refines the strategy. After $T$ iterations you read off the average strategy at the root, **sample one action, play it, and throw the entire regret table away.**

**Outer loop — network training across self-play.** Separately, over days of wall-clock, you play thousands of full self-play games. Every game uses the inner loop at every decision. You collect training targets from those searches into a replay buffer and periodically update the network. This is the loop that has "generations."

When I said earlier "training is continuous," I meant the **outer** loop. When you asked "how do I know when the strategy is stable enough to act and move to turn 2," that's the **inner** loop. They're different and the answers are different:

- **Inner loop stopping:** you do **not** wait for full convergence. You spend a fixed budget (e.g., $N$ iterations or $M$ milliseconds), then act. CFR+ exploitability shrinks roughly like $O(1/T)$ in practice, so a fixed budget gets you "good enough." You then move to turn 2 by building a **brand-new tree** rooted at turn 2's public state. The regret table does **not** carry over — turn 2 is a different public state with a different tree.
- **Outer loop stopping:** you measure **exploitability** (train a best-responder against your frozen strategy and measure how much it beats you) or head-to-head win rate vs. previous checkpoints. You stop when it plateaus or compute runs out.

So: you never "reset cumulative regret" within a search (it monotonically accumulates over the $T$ iterations), but you **discard it entirely between decision points**. Keep this picture in mind for everything below.

## Deep CFR vs. GT-CFR Are Different Algorithms

You're (understandably) blending two designs. They answer your questions differently, so let me separate them:

- **Deep CFR (Brown et al. 2019):** Uses **two networks** — an *advantage/regret network* and an *average-strategy network*. The network **does** predict regrets. No tree-growth-with-leaf-evaluation at play time; it's a way to scale tabular CFR via function approximation and sampling.
- **GT-CFR / Player of Games (Schmid et al. 2021):** Uses **one network with two heads** — a **counterfactual-value head** and a **policy head**. The network does **not** predict regrets; it predicts *values* and *priors*. CFR machinery computes regrets *inside* the search from those values. This is the AlphaZero-shaped one (search at play time + leaf evaluation by the net).

Several of your questions assume the GT-CFR network predicts regrets — it doesn't. That's the Deep CFR design. I'll flag this each time it matters.

## Q: Why a network for both regret AND strategy? (the "two networks" question)

> Why do you need a network for both regret AND strategy? Since strategy is directly computed from cumulative regret, this seems like wasted effort.

This applies to **Deep CFR**, and the reason is subtle but important: **the thing you actually play (the Nash approximation) is the *average* strategy over all iterations, and that average is not recoverable from the regret table alone.**

Regret matching at iteration $t$ gives you $\sigma^t$ , the *current* iterate's strategy:

$$\sigma^{t+1}(I,a) = \frac{R^{t,+}(I,a)}{\sum_b R^{t,+}(I,b)}$$

But $\sigma^t$ by itself oscillates and does **not** converge to Nash. The object that converges is:

$$\bar{\sigma}(I,a) = \frac{\sum_{t=1}^T w_t\,\sigma^t(I,a)}{\sum_{t=1}^T w_t}$$

If you only stored the final regret table, you could reconstruct $\sigma^{T}$ (the last iterate) but **not** the running average $\bar\sigma$ — the average depends on the entire history $\sigma^1,\ldots,\sigma^T$, which the final regrets don't encode. So Deep CFR keeps a second network whose only job is to memorize $\bar\sigma$.

In **GT-CFR/PoG** this question dissolves: there aren't two networks. There's one net with a value head and a policy head. The policy head is **not** $\bar\sigma$ — it's a *prior* used to guide tree expansion (the $P$ in the PUCT-style rule). The actual $\bar\sigma$ you play is tracked explicitly in the ephemeral tables during the search, not stored in a network.

## Q: Why carry an ephemeral regret table if you discard it?

> Why do you carry around an ephemeral cumulative regret table in GT-CFR if you discard it after each search? Is this used to balance out the exploration vs exploitation? Is this not what's used to train the next generation of NN too? Why carry it around at all, if you end up discarding it?

Because the regret table **is the CFR computation**. Within one search it does three jobs:

1. **Produces the current strategy** each iteration via regret matching (you can't run CFR without it).
2. **Feeds the running average** $\bar\sigma$ (the search's output).
3. **Informs expansion** — the current strategy (derived from regrets) is one term in the selection rule that decides where to grow the tree.

You discard it afterward because it's specific to *this* public state's tree. The next decision is a different tree; the numbers don't transfer.

It is **not** itself the training target. The training targets are derived *from* the converged search: the **refined counterfactual values** at the root and the **average strategy** $\bar\sigma$. Those are extracted at the end and pushed to the replay buffer; the raw regret table is garbage-collected.

## Q: Does the network predict instantaneous or cumulative regret? And the "depends on $t$" problem.

> Is the network predicting the instantaneous regrets, or the cumulative regrets? If it predicts the instantaneous regrets, that depends on the current strategy, so how does that get passed as an input to the NN? I guess the cumulative regrets are a function of the current strategy too, so how does it predict those if the current strategy evolves in each iteration? What are the regret targets for a GT-CFR system? Is it the Instantaneous regret? Or the cumulative regret?

Separate by algorithm:

**Deep CFR:** the advantage network predicts something **proportional to cumulative regret**, not instantaneous. Mechanically: each traversal computes *sampled instantaneous* regrets and stores them in a memory tagged with iteration $t$. The network is retrained to fit the (iteration-weighted) **mean** of those samples, which is proportional to cumulative regret. Crucially, **the network does not take the strategy as input** — its input is just info-set features. The strategy-dependence is "baked in" through the *data*: the samples were generated under the evolving strategies, and the accumulation over iterations is exactly what cumulative regret is. So your worry ("instantaneous regret depends on the current strategy, how is that an input?") is resolved by the fact that the net predicts the *accumulated* quantity from accumulated data, not a single $t$'s value conditioned on a strategy.

**GT-CFR/PoG:** the network predicts **counterfactual values and policy priors — never regrets.** This directly answers your last question (#9):

> I know I already asked this earlier in this message, but the counterfactual values predicted by the NN -- you say that it's the "estimated v_i(I, *) for each player" but this is implicity a function of t, the iteration number. How can the NN predict this if it changes as a function of the iteration number? Does this statement you made hold the answer? "It does not predict the regret table directly. It predicts counterfactual values, from which CFR+ machinery derives regrets inside the search. (Contrast with Deep CFR, where the network does directly predict regrets — different architectural choice.)"

yes, the statement you quoted holds the answer. The per-iteration counterfactual values $v_i^t(I,\cdot)$ inside the search **do** depend on $t$ — but those are computed by *tree traversal*, not by the network. The network is queried only at the **leaf public states** of the search tree, where it returns "the value of this public state under good play" — a quantity that is **not** indexed by the inner-loop $t$, exactly like AlphaZero's value head returns "value of this position," not "value at MCTS simulation #347." During a given search, the leaf values are a fixed boundary condition; the $t$-dependence lives entirely in the internal-node regret/value tables that traversal updates.

So: **regret targets are a Deep CFR concept** (targets = accumulated instantaneous regret samples). **GT-CFR has no regret targets**; its targets are value targets and policy targets.

## Q: Why train the next net using $V_\theta$'s own leaf values? Isn't that "no new information"?

> You said this: "Along the trajectory, compute instantaneous counterfactual regrets at each info set encountered. Use V_θ itself or rollouts for the values at terminal/leaf nodes." Why would you use V_θ itself to train the next generation NN? That seems like you're adding no new information. You should compute the instantaneous counterfactual regrets using real data, not the NN's predictions, right?

This is the sharpest challenge in your message, and the answer is **bootstrapping**, the same mechanism behind TD-learning, value iteration, and AlphaZero.

The search does not just regurgitate $V_\theta$. It adds information from two sources:

1. **It reaches real terminals.** Wherever the search tree (after expansion) hits an actual game-ending node, it uses the **true payoff** $\pm 1$, not a network guess. Those exact values propagate back through the tree and correct the network's estimates near the frontier.
2. **The CFR/lookahead operator is an improvement operator.** Even using imperfect leaf values, running CFR over a tree of them produces a root estimate that is *closer to truth* than the raw leaf estimate, because it combines many leaf evaluations through correct game dynamics and equilibrium reasoning (the analogue of "MCTS produces a better policy than the raw net").

The target you train on is therefore "network estimate **plus** search correction," which is strictly more informed than the raw network. You're right that if the search *never* touched a terminal and the improvement operator did nothing, you'd just be reshuffling the net's own beliefs — that degenerate case would add no grounding. What saves it is (1) terminals and (2) a genuine improvement operator. Early in training, when the net is near-random, **almost all** the real signal comes from terminals, and it propagates slowly outward — exactly the AlphaZero cold-start story. I'll show this concretely in the Kuhn example below: you'll see a wrong network leaf value get **overwritten** by a real terminal value during expansion.

The reason you don't just always roll out to terminal (and skip the net) is cost: in Pokemon the tree to terminal is enormous and stochastic. The net lets you **truncate** the search at a manageable depth while still getting a usable value. Bootstrapping trades a little bias for an enormous reduction in compute.

## Q: The summation over histories is combinatorially huge in Pokemon. How is that tractable?

> Also, in all these regret calculations, there's a summation over all histories that belong to this info set, but this is a huge problem of scale for this pokemon example. There's combinatorially huge numbers of histories that belong to the same info set, making it impossible to sum over them. How is Instantaneous regret then computed for this seemingly intractibly large history-space? E.g. a huge breadth of possibilities for items and moves (discrete, not combinatorially huge) AND ev spreads (incredibly variable, adds a ton of combinatorial size)

Correct — you cannot literally compute $\sum_{h\in I}$ when $I$ bundles every (item × moveset × stat-spread × back-row) the opponent could have. Three techniques, used in combination:

1. **Monte Carlo sampling (MCCFR).** Don't sum over all histories — **sample** them according to reach probability and form an *unbiased* estimate of counterfactual regret. Outcome sampling traverses one history per iteration; external sampling samples opponent+chance but enumerates the traverser's actions. This is the primary scalability tool.

2. **Belief/range representation (DeepStack, PoG).** Represent "the histories in this info set" as a **probability vector (a range) over possible private states**, and compute the counterfactual value as a weighted sum / matrix product over that vector rather than a tree enumeration. The public state factorizes the game so you carry a distribution, not an explicit history set.

3. **Abstraction / bucketing.** Collapse strategically-similar private states into a small number of buckets and solve the abstract game. This is how poker handled $10^{160}$ states.

For Pokemon specifically, this is **exactly where your `meta_priors` and `clustering.py` work earns its keep.** The killer problem is the continuous stat-point (EV) space, which is genuinely intractable if treated literally. The fix is to **discretize via your clustering**: each species gets a small finite set of **archetype sets** ("bulky TR attacker," "fast offensive," "redirector support," etc.), each with a prior probability from tournament data. Then:

- Items/moves: enumerate the **top-$k$ most probable sets** per species from priors instead of the full combinatorial space.
- Stat spreads: snap to the cluster centroids rather than a continuum.
- The opponent's private-state space per slot becomes "one of $k$ archetypes," and the sum/sample over histories becomes a sum over a few hundred plausible team configurations, not $10^{\text{huge}}$.

So the meta-prior pipeline isn't just for determinization flavor — it's the **abstraction layer** that makes the counterfactual sums finite. The relevant cardinality is the opponent's *plausible* set space (since $\pi_{-i}$ ranges over opponent + chance), and priors bound that directly.

## Q: How does the training actually work — batched, iterative, or continuous?

> I think I need more clarification on how the training works for an algorithm like this. In my experience with NNs, you have training data, and you train the NN on that data to predict some target. Then, if you get new, better data that makes the old data seem bad or wrong (what I'm imagining here, with the NN being trained in generations or batches), you train a new NN from scratch (or maybe use transfer learning to save time) on the new data only. The old data is discarded because you've generated new, better training data. It sounds like you're suggesting some sort of continuous train + data generation, where as soon as a data point is generated, it's added to the training data? Does it at all work in batches? Explain this more to me, as carefully as you can. Is it batched and iterative, or truly continuous?

Your mental model ("get better data → retrain a fresh net on the new data only, discard the old") is **partially** right and maps cleanly onto one of the two standard patterns. Here are both:

**Pattern A — AlphaZero / PoG style (sliding-window replay).**
- Self-play workers continuously generate games and push training tuples into a **replay buffer**.
- A trainer continuously samples **minibatches** from the buffer and does gradient steps, publishing a new checkpoint every so often.
- The buffer is a **sliding window**: it keeps roughly the last $N$ games and **drops the oldest** (data from much weaker nets). It does **not** keep everything forever, and it does **not** discard everything each generation.
- So: "continuous" at the system level, "batched" (minibatch SGD) at the gradient level. Weights are **never** reset — it's one continuously-updated network.

**Pattern B — Deep CFR style (reservoir memory + periodic from-scratch retrain).**
- Instantaneous regret samples accumulate in a **reservoir buffer** (a uniform sample across *all* iterations so far — old data is kept, just subsampled).
- Each CFR iteration, the advantage net is **retrained from scratch** on the whole memory. The from-scratch reset is deliberate: it prevents the net from over-anchoring to early, bad-strategy data (chasing a moving target).

Your intuition ("train a fresh net") matches **Pattern B's weight reset**, but with one correction: **the data is accumulated, not discarded.** You keep (a subsample of) the historical data and reset only the *weights*. Pattern A doesn't reset weights at all and discards only very stale data via the sliding window.

For your project I'd use **Pattern A** (it's simpler to operate and is what AlphaZero/PoG use): one network, replay buffer with a sliding window, continuous minibatch training, periodic checkpoint publication that self-play workers reload.

## Q: The PUCT rule in GT-CFR — what are Q, P, N?

> So in the PUCT equation for GT-CFR, the Q value is the iterative, cumulative regret for this node (this is why we carry this ephemeral regret table around?). And then P is the predicted cumulative regret(?) from the NN, and the N counts are still the same as the MCTS case? I think once my earlier questions are clarified, I will understand this part (given that what I've said here is true?)

One correction and two confirmations:

- **$Q$-analogue = the current CFR strategy/value from the ephemeral regret table.** Yes — this is one reason you carry that table. The exploitation signal is "what does CFR currently think is good here," which comes from regret matching on cumulative regret. (Confirmed.)
- **$P$ = the CVPN *policy-head prior*, NOT predicted regret.** (Correction.) In GT-CFR/PoG the net never outputs regret; $P$ is the prior probability over actions, exactly like AlphaZero. Your statement "P is the predicted cumulative regret" is the Deep CFR design leaking in; in GT-CFR it's the policy prior.
- **$N$ = visit counts, same role as MCTS.** (Confirmed.) Provides the shrinking exploration bonus.

Conceptually the selection score at an info set is:

$$\text{score}(I,a) \;=\; \underbrace{\sigma^t(I,a)}_{\text{exploit (from regret table)}} \;+\; \underbrace{c\cdot P(I,a)\cdot\frac{\sqrt{\sum_b N(I,b)}}{1+N(I,a)}}_{\text{explore (prior + visit bonus)}}$$

Exact functional forms vary between implementations, but the **roles** are: regret table → exploitation, policy prior → where to look first, visit counts → don't ignore under-explored actions. So yes, this is how exploration/exploitation is balanced during tree growth, and yes the ephemeral regret table is load-bearing for the exploitation term.

## The Centerpiece: GT-CFR on Kuhn, with full arithmetic

> In your example of GT-CFR for Kuhn Poker, you actually skip over the important details that I'm looking for! I want to see how you actually expand I1_K in the first step (and do you similarly expand I2_J?) and compute the regrets. I want you to show me the math for this one step in how that leaf gets expanded and how the policy gets updated. Basically, go into a lot more detail in the sections "Iteration 1 -- Expandion phase" and "Iteration 1 -- Policy update phase"

Now the detailed walkthrough you actually wanted. I'll focus on info set $I_{1K}$ (Player 1 holds the King) and show (a) the shallow tree with **CVPN leaf values**, (b) one CFR update, (c) an **expansion** that grounds a leaf in real terminals, and (d) the next CFR update so you see the policy move.

**Kuhn recap.** Cards J < Q < K, one each, third unused, each antes 1. P1 acts: **bet** or **check**. Payoffs to P1 (net chips): bet→fold $=+1$; check-check showdown $=\pm1$; any showdown-after-call $=\pm2$; check-bet-fold $=-1$. For $I_{1K}$, the two histories are $(K,J)$ and $(K,Q)$, each with counterfactual reach $\pi_{-1}=1/6$ (just the chance deal; P1's own action is factored out).

We fix everyone's iteration-1 strategy to **uniform** (the standard CFR initialization), so P2 calls/folds/bets 50/50 wherever it acts.

### Step 1 — Shallow tree, leaves evaluated by the CVPN

Initially the tree is shallow: after P1's action we stop at P2's decision node and ask the **CVPN** for P1's value there. Suppose the (imperfect, partially-trained) CVPN returns these **per-history** leaf values $u_1(h\cdot a)$:

| leaf (P1 action, opp card) | CVPN value to P1 |
|---|---|
| bet, $(K,J)$ | $0.6$ |
| bet, $(K,Q)$ | $0.6$ |
| check, $(K,J)$ | $1.2$ |
| check, $(K,Q)$ | $1.2$ |

This network wrongly thinks **checking** the King is much better than betting it. Watch the search fix that.

Counterfactual values at $I_{1K}$:

$$v_1(I_{1K},\text{bet}) = \tfrac16(0.6)+\tfrac16(0.6) = 0.2$$
$$v_1(I_{1K},\text{check}) = \tfrac16(1.2)+\tfrac16(1.2) = 0.4$$

Strategy value under uniform $\sigma^1=(0.5,0.5)$:

$$v_1(I_{1K},\sigma^1) = 0.5(0.2)+0.5(0.4) = 0.3$$

Instantaneous regrets (iteration 1):

$$r^1(\text{bet}) = 0.2-0.3 = -0.1,\qquad r^1(\text{check}) = 0.4-0.3 = +0.1$$

CFR+ cumulative regrets (clip negatives to 0): $R^+(\text{bet})=0,\ R^+(\text{check})=0.1$.

Regret-matched strategy for the next iteration:

$$\sigma^2(I_{1K}) = (\text{bet}=0,\ \text{check}=1)$$

After one iteration the search has (wrongly) committed to checking the King — because it trusted the network's leaf values and nothing has contradicted them yet. **This is the "no new information" failure mode in action — and it's about to be corrected by expansion.**

### Step 2 — Expansion: grow the "bet" leaf into real terminals

GT-CFR's growth phase now picks a leaf to expand. Say it expands the **bet** node. In Kuhn, expanding "P1 bets with K" reveals **P2's decision node** (call/fold) — these are $I_{2J\,b}$ (P2 holds J, facing a bet) and $I_{2Q\,b}$ (P2 holds Q, facing a bet) — and their children are **terminal** (showdown or fold). So no CVPN is needed below them; we use **real payoffs**.

(To your sub-question "do you expand $I_{2J}$ similarly?" — **yes**: expanding the bet node *creates* P2's nodes as the new frontier. In Kuhn they bottom out in terminals immediately. In a deeper game they'd become new CVPN leaves, and they'd get their **own** regret tables because P2 is also minimizing regret.)

Compute the real value of betting K under P2's current uniform call/fold:

- vs J: fold (0.5)→$+1$, call (0.5)→showdown $K>J$→$+2$. Value $=0.5(1)+0.5(2)=1.5$.
- vs Q: fold (0.5)→$+1$, call (0.5)→showdown $K>Q$→$+2$. Value $=1.5$.

So the **grounded** counterfactual value replaces the CVPN's guess:

$$v_1(I_{1K},\text{bet}) = \tfrac16(1.5)+\tfrac16(1.5) = 0.5 \quad(\text{was } 0.2)$$

The CVPN's wrong $0.2$ is gone — overwritten by real terminal information. **This is precisely the "new information" your Q5 said couldn't come from a bootstrap: it comes from reaching terminals during expansion.** The "check" branch is still an un-expanded CVPN leaf, so it keeps $v_1(I_{1K},\text{check})=0.4$ for now.

### Step 3 — CFR iteration 2 on the grown tree

Now use the strategy from step 1, $\sigma^2=(\text{bet}=0,\text{check}=1)$, and the updated values $v(\text{bet})=0.5$, $v(\text{check})=0.4$:

$$v_1(I_{1K},\sigma^2) = 0(0.5)+1(0.4) = 0.4$$
$$r^2(\text{bet}) = 0.5-0.4 = +0.1,\qquad r^2(\text{check}) = 0.4-0.4 = 0$$

Accumulate (CFR+) onto step-1's cumulative $(\text{bet}=0,\ \text{check}=0.1)$:

$$R^+(\text{bet}) = \max(0+0.1,0)=0.1,\qquad R^+(\text{check}) = \max(0.1+0,0)=0.1$$

Regret-matched strategy for iteration 3:

$$\sigma^3(I_{1K}) = \left(\text{bet}=\tfrac{0.1}{0.2}=0.5,\ \text{check}=0.5\right)$$

The policy has swung from "check 100%" back toward betting, **driven entirely by the real terminal payoffs that expansion injected.** If the growth phase now also expands the "check" branch (grounding it to its true value of $0.25$ from the fully-solved game), subsequent iterations get $v(\text{bet})=0.5 > v(\text{check})=0.25$, regret keeps accumulating for "bet," and $\sigma(I_{1K})$ converges toward **bet-heavy** — the correct Kuhn answer for a King (modulo the small slow-play frequency $3\alpha$ from the Nash family once P2's strategy also evolves).

### What this example demonstrates

- **The "GT" mechanic:** the tree starts shallow with CVPN-estimated leaves and **grows**, replacing estimates with deeper (eventually terminal) computation.
- **The CFR update math:** per iteration, compute $v(I,a)$, subtract the strategy-weighted average to get instantaneous regret, accumulate with CFR+ clipping, regret-match for the next strategy.
- **Why bootstrapping with the net still adds information (Q5):** you literally watched a wrong CVPN value ($0.2$) get overwritten by a real terminal value ($0.5$), which then moved the policy. The net provides a starting estimate; expansion + terminals provide the correction; over training generations that correction is distilled back into the net.
- **P2 is doing the same thing simultaneously** at its own info sets ($I_{2J\,b}$, $I_{2Q\,b}$, etc.), with its own regret tables, which is why the final result is a two-sided equilibrium rather than a best response to fixed uniform play.

---

If you want to go further, the natural next step is to run this same Kuhn tree for **both players over ~10 iterations** until the regrets stabilize, so you can watch $\bar\sigma$ (the *average*, not the last iterate) converge to the known Nash family — that's the cleanest way to *see* why the average strategy, not the current one, is the deliverable. Or we can pivot to the **state/belief encoding** for Pokemon, since that abstraction layer (Q6) is now clearly on the critical path and ties directly into your existing clustering work.