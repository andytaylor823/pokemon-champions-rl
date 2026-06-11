# Theoretical Foundations: Search + Self-Play for Pokémon Champions

**Status:** Living reference / ground truth for the algorithmic theory behind this project.
**Audience:** The maintainer and any future coding agent working in this repo.
**Scope:** *Theory only* — the conceptual backbone from AlphaZero → CFR → GT-CFR (Player of Games), plus how it maps onto Pokémon VGC. Engineering, infrastructure, data sourcing, and game-format details live elsewhere (see "Related artifacts" below) and are referenced rather than restated here.

---

## 0. How to use this document

This is the distilled, organized version of an extended theory discussion. It is meant to be read top-to-bottom once, then used as a lookup reference. The two worked Kuhn-poker examples (Sections 8 and 11) are the fastest way to re-anchor intuition; the glossary (Section 15) is the fastest way to re-anchor vocabulary.

**Related artifacts (do not duplicate — read these for their domains):**

- `.cursor/rules/agent/overview.mdc` — system architecture, milestones, action-space size, imperfect-info inventory.
- `.cursor/rules/game-domain/overview.mdc` — Regulation M-A rules, stat system, Mega mechanics, doubles mechanics.
- `.cursor/rules/game-domain/legality.mdc` — how to validate teams (CLI helper) without burning tokens.
- `research/notes.md` — algorithm survey (PPO / AlphaZero / R-NaD), simulation-engine choices (`@pkmn/sim`, poke-env), data sources for priors. **The engineering counterpart to this doc.**
- `docs/state-encoding.md` — how the battle state is tokenized and consumed by the neural network. Covers entity tokens, field tokens, the Transformer architecture, belief-weighted candidate tokens, Phase 1 vs Phase 4 value-head designs, and alternatives considered. **The encoding/NN architecture counterpart to this doc.**
- `docs/search-nn-interface.md` — when the NN is called during search, caching, deal sampling, the composite-private-state problem, the three-tier computation split (backbone/policy/value), and chance-node bucketing. **The runtime interface counterpart to this doc.**
- `docs/article_summary*.md` — the long-form conversational Q&A derivations that this document distills. Go there for the verbose back-and-forth and additional examples.

**Notation conventions:** display math in `$$ … $$`, inline math in `$ … $`. Player $i$; opponent $-i$. Strategies $\sigma$; the search/CFR iteration counter is $t$ (do **not** confuse it with the in-game turn number).

---

## 1. What kind of game is Pokémon VGC?

The choice of algorithm is dictated entirely by the game's properties. Pokémon Champions doubles (Regulation M-A) is:

| Property | Pokémon VGC | Chess / Go |
|---|---|---|
| Players | 2, **zero-sum** | 2, zero-sum |
| Information | **Imperfect** (hidden items, sets, back row, spreads) | Perfect |
| Move timing | **Simultaneous** (both submit, then resolve) | Sequential |
| Dynamics | **Stochastic** (damage rolls, accuracy, crits, secondaries) | Deterministic |
| Structure | Extensive-form (a game tree) | Extensive-form |
| Horizon | Short (~6–12 turns) | Long |
| Joint actions/turn | ~100 (see project-overview rule) | ~30–250 |

Four of these properties (imperfect, simultaneous, stochastic, zero-sum) are exactly the ones that **break** a naive AlphaZero port and **motivate** the CFR family. The two that help us are the short horizon and the modest action space — search is genuinely feasible here.

**Target solution concept:** a **Nash equilibrium** — a strategy that is *unexploitable* (no opponent can profit by deviating). In two-player zero-sum games this is the principled "play perfectly" target. (Exploiting *weak* opponents harder than Nash is a later, separate goal; see R-NaD note in `research/notes.md`.)

---

## 2. The single most important mental model: two nested loops

Almost every confusion in this material dissolves once you separate these two loops. They have different "iteration" counters and different notions of "convergence."

**Inner loop — search at one decision point (ephemeral).**
When it is the agent's turn, it builds a search tree rooted at the current state and runs many search iterations $t = 1, 2, \dots, T$ on it. After the budget is spent, it reads off a move, plays it, and **throws the entire search tree and its tables away**. The next decision builds a fresh tree.

- *Stopping rule:* a **fixed budget** (N iterations or M milliseconds). You do **not** wait for full convergence; "good enough" is enough.
- *Moving to the next turn:* build a brand-new tree at the new state. Nothing carries over.

**Outer loop — training across self-play (persistent).**
Over long wall-clock time, the agent plays thousands of full self-play games. Every decision in every game uses the inner loop. Targets extracted from those searches train the neural network, which is then used by future inner-loop searches.

- *Stopping rule:* **exploitability** plateaus (train a best-responder against the frozen agent and measure the gap to the game value), or head-to-head win rate vs. prior checkpoints plateaus.

> Whenever a question contains the word "converge" or "iteration," first ask: *inner loop or outer loop?* The answer is usually the whole answer.

---

## 3. AlphaZero: the perfect-information foundation

AlphaZero is the scaffold. GT-CFR replaces *one component* of it (the search), so understanding it first is non-negotiable. **Recommended Phase 1–2 of the build (Section 14) is literally AlphaZero on a perfect-info simplification of Pokémon.**

### 3.1 The network

One network $f_\theta(s) = (\boldsymbol{p}, v)$ with a shared backbone and **two heads**:

- **Policy head** $\boldsymbol{p}$: a distribution over actions. "Which actions are worth considering here?" Used as priors to guide search. *Not* a value.
- **Value head** $v \in [-1, 1]$: a scalar. "How good is this position for the player to move?" Used to evaluate search leaves.

### 3.2 MCTS (the search): select → expand → evaluate → backup

Each simulation walks from the root down the existing tree, adds **one** new leaf, evaluates it with the network, and propagates the value back up. **There is no random rollout to the end of the game** — the value head replaces it. This is AlphaZero's key departure from classical MCTS.

**Selection** uses the PUCT rule:

$$
a^{\star} = \arg\max_a \left[\, Q(s,a) + c_{\text{puct}} \cdot P(s,a) \cdot \frac{\sqrt{\sum_b N(s,b)}}{1 + N(s,a)} \,\right]
$$

- $Q(s,a)$ — mean value of leaf evaluations from simulations that passed through edge $(s,a)$; $Q = W/N$ where $W$ accumulates backed-up values. **Exploitation.**
- $P(s,a)$ — policy-head prior for $a$. Biases exploration toward promising actions.
- $N(s,a)$ — visit count of the edge; $\sum_b N(s,b)$ is the parent visit count.
- $c_{\text{puct}}$ — exploration constant (~1–4). The whole second term is **exploration**; it shrinks as $N(s,a)$ grows.

**Expansion + evaluation:** at the first unvisited node, call $f_\theta$ once, cache its $v$ and attach $P$ to the node's edges. **Backup:** add $v$ to $W$ and increment $N$ along the path (flipping sign at opponent nodes). After $T$ simulations, the tree has ~$T$ nodes, deep along promising lines.

### 3.3 MCTS is a policy-improvement operator

The move actually played is sampled from the **root visit-count distribution** $\pi_{\text{MCTS}}(a) \propto N(\text{root}, a)^{1/\tau}$. This distribution is **provably better on average than the raw policy head**, because search does concrete lookahead and lets value backups override the network's snap judgment. This is why **search is run at play time, not just during training** — in chess the gap between "net + MCTS" and "net alone" is ~1500 Elo, and it does not close as the net improves (search improves whatever policy it is given).

### 3.4 Self-play training

For every position in a self-play game, store $(s, \pi_{\text{MCTS}}, z)$ where $z \in \{+1, -1\}$ is the eventual game outcome. Train:

$$
L(\theta) = \underbrace{\big(v_\theta(s) - z\big)^2}_{\text{value}} \;+\; \underbrace{-\,\pi_{\text{MCTS}} \cdot \log \boldsymbol{p}_\theta(s)}_{\text{policy (cross-entropy)}} \;+\; \lambda \lVert \theta \rVert^2
$$

The cycle: better net → better MCTS → better targets → better net. Starting from random weights, the only real signal early on comes from games that happen to reach terminal states; it propagates outward slowly.

---

## 4. Why naive MCTS breaks under imperfect information

You cannot run MCTS directly on Pokémon because you do not know the opponent's hidden state. The obvious patch is **determinization** (Information Set MCTS): sample a concrete opponent state from priors, run MCTS as if perfect-info, repeat, average. This fails via **strategy fusion**.

**Strategy fusion (worked example).** Opponent's Charizard item is hidden: 50% Charizardite Y (special, slow), 50% Choice Scarf (fast, locked). Your options: switch to Water (walls Y, dies to Scarf-EQ), switch to Flying (walls Scarf, melts to Y-Heat Wave), or stay.

| World | Best response | Win rate |
|---|---|---|
| Y-world | Switch Water | 0.85 |
| Scarf-world | Switch Flying | 0.85 |

Determinized search finds the *per-world optimum* and implicitly assumes it can play the right one in each world. But you **cannot** — you do not know the world when you choose. The backed-up $Q$ values are inflated relative to reality because the search "fused" information from worlds it should not have been able to distinguish. The fix is to reason at the **information-set** level (Section 7), never at the determinized-state level.

> See also `research/notes.md` → "Strategy Fusion" for the Focus-Sash framing and the two directions of corruption.

---

## 5. Game-theory primitives

### 5.1 Extensive-form game

A game tree with: **nodes** (positions), **edges** (actions), a **player function** $P(h)$ (who acts at node $h$, or "chance"), **chance probabilities** at chance nodes, **terminal payoffs** $u_i(h)$, and **information sets**.

### 5.2 History and information set

- **History $h$** = the full action sequence from the root to a node. ("History" and "node" are interchangeable.)
- **Information set $I$** = a set of histories that are *indistinguishable to the acting player*. The player must use **one strategy across the whole info set** (they cannot tell the histories apart). Perfect information ⇔ every info set is a singleton.

### 5.3 Reach probabilities (and the "counterfactual" qualifier)

The probability of reaching history $h$ factors along the path:

$$
\pi^\sigma(h) = \underbrace{\pi_c(h)}_{\text{chance}} \cdot \underbrace{\pi_i^\sigma(h)}_{\text{player }i} \cdot \underbrace{\pi_{-i}^\sigma(h)}_{\text{opponent(s)}}
$$

The **counterfactual reach probability** for player $i$ is everything *except* $i$'s own contribution:

$$
\pi_{-i}^\sigma(h) = \pi_c(h)\,\textstyle\prod_{j \ne i} \pi_j^\sigma(h)
$$

Read it as: "the probability of arriving at $h$ *if player $i$ had tried to get here with certainty*." We strip out $i$'s own probabilities so that the **quality of a decision at $I$ is decoupled from how often $i$ chooses to reach $I$**. This is a probability used as a *weight* — it is **not** a value (this distinction is the single most common trap; see Section 7.1).

These weights also perform **automatic Bayesian inference**: histories inconsistent with the opponent's observed play get near-zero $\pi_{-i}$, so they barely count. You never write down Bayes' rule explicitly; it falls out of the strategy probabilities.

### 5.4 Nash equilibria can be a *set*, not a point

In two-player zero-sum games the **value** of the game is unique, but the **set of equilibrium strategies is generally a convex set**. Kuhn poker's equilibrium for Player 1 is a one-parameter family indexed by $\alpha \in [0, \tfrac13]$; every member achieves the same value ($-\tfrac{1}{18}$ for P1). "Variable" does not contradict "equilibrium": at any $\alpha$, neither player can profitably deviate. (Different training runs may land on different equilibria; they are equally unexploitable but may exploit *weak* opponents differently.)

---

## 6. Regret and regret matching

### 6.1 Regret (online-learning definition)

Over $T$ rounds, picking $a_t$ each round, **regret** compares your payoff to the best *single fixed action* in hindsight:

$$
\text{Regret}(T) = \max_{a^{\star}} \sum_{t=1}^T u_t(a^{\star}) - \sum_{t=1}^T u_t(a_t)
$$

An algorithm is **no-regret** if $\text{Regret}(T)/T \to 0$. (See `docs/article_summary.md` for the worked 10-round rock-paper-scissors computation.)

### 6.2 Cumulative regret and regret matching

Cumulative regret of action $a$, and the **CFR+** non-negative-clipped variant:

$$
R^T(a) = \sum_{t=1}^T \big[u_t(a) - u_t(a_t)\big], \qquad
R^{T,+}(a) = \max\!\big(R^{T-1,+}(a) + r^T(a),\; 0\big)
$$

**Regret matching** sets the next strategy proportional to positive cumulative regret:

$$
\sigma^{T+1}(a) = \begin{cases}
\dfrac{R^{T,+}(a)}{\sum_b R^{T,+}(b)} & \text{if } \sum_b R^{T,+}(b) > 0 \\[2mm]
1/|A| & \text{otherwise (uniform fallback)}
\end{cases}
$$

It has **no tunable exploration constant** (unlike PUCT); the convergence guarantees come from the regret bound itself. Regret matching is no-regret (Hart & Mas-Colell, 2000).

### 6.3 The folk theorem that makes self-play work

In a two-player zero-sum game, if **both** players run a no-regret algorithm, their **time-averaged** strategies converge to a Nash equilibrium. We never solve for Nash directly — no-regret self-play *finds* it. Crucial subtlety: it is the **average** strategy $\bar\sigma$ that converges, **not** the current iterate $\sigma^t$ (which keeps oscillating).

---

## 7. CFR: counterfactual regret minimization

CFR lifts regret matching to extensive-form games with imperfect information by running an independent regret-matching process **at every information set**, glued together by counterfactual values.

### 7.1 Counterfactual value (and the trap)

The **counterfactual value** of action $a$ at info set $I$ under strategy profile $\sigma$:

$$
v_i(I, a) = \sum_{h \in I} \pi_{-i}^\sigma(h)\, \cdot\, u_i^\sigma(h \cdot a)
$$

where $u_i^\sigma(h\cdot a)$ is $i$'s expected utility from taking $a$ at $h$ and continuing under $\sigma$ (computed by recursion through the subtree; this is the object a value network later approximates). The value of the info set under the current strategy is the strategy-weighted average:

$$
v_i(I) = \sum_{a} \sigma(I,a)\, v_i(I,a)
$$

> **The trap:** counterfactual **value** $v_i(I,a)$ ≠ counterfactual **reach probability** $\pi_{-i}(h)$. The reach probability is one *weight inside* the value's summation. The value is the whole weighted sum. (Section 11 grounds this numerically: the $\tfrac16$'s are reach probabilities; the resulting $0.5$ is the value.)

**Why weight by $\pi_{-i}$ at all, if you already have utilities?** Because an info set bundles *many* histories (the opponent's different hidden states), each with its own utility. To collapse them into one number you must average, and the correct averaging weight is "how likely is each hidden world, given everyone's strategies" — which is exactly $\pi_{-i}$. Without it you would implicitly assume all hidden worlds equally likely.

### 7.2 Counterfactual regret and the algorithm

Instantaneous and cumulative counterfactual regret:

$$
r_i^t(I,a) = v_i^t(I,a) - v_i^t(I), \qquad R_i^T(I,a) = \sum_{t=1}^T r_i^t(I,a)
$$

**One CFR iteration** = one pass over the (whole) game tree that, **for every info set simultaneously**, computes counterfactual values, updates the cumulative regret table, derives the next strategy by regret matching, and accumulates the average strategy. After $T$ iterations every info set has $T$ updates and $\bar\sigma$ approximates Nash.

- **Convergence (Zinkevich et al., 2007):** in 2p zero-sum games, $\bar\sigma$ → Nash; total regret is $O(\sqrt{T})$.
- **It does not "play" the game.** CFR is an offline computation over the model, like value iteration over an MDP. The "$t$" is the algorithm's iteration counter, never the in-game turn.
- **No strategy fusion:** because it operates on info sets, it finds the single best *mixed* strategy that must work across all indistinguishable worlds — it never pretends to know hidden info.

### 7.3 CFR+ (the practical standard)

Three changes give ~1000× faster convergence (and solved heads-up limit hold'em, Bowling et al. 2015):

1. **Regret-matching+** — clip cumulative regret to $\ge 0$ each step (Section 6.2), so an action that fell out of favor early can recover quickly.
2. **Linearly weighted averaging** — weight iteration $t$'s contribution to $\bar\sigma$ by $t$.
3. **Alternating updates** — update one player per iteration, not both.

---

## 8. Worked example A — vanilla CFR on Kuhn poker

**Kuhn poker.** Deck $\{J, Q, K\}$ with $J<Q<K$; each player antes 1 and is dealt one card (third unused). P1 acts: **bet** or **check**. Payoffs to P1 (net chips): bet→fold $=+1$; check-check→showdown $=\pm1$; any showdown after a call $=\pm2$; check-bet-fold $=-1$. There are 12 info sets (6 per player).

Focus on $I_{1K}$ (P1 holds the King). Its two histories are $(K,J)$ and $(K,Q)$; before P1 acts the only thing on the path is the deal, so the **counterfactual reach** of each is the chance probability $\tfrac16$. Initialize all strategies **uniform** (so P2 calls/folds/bets 50/50).

**Counterfactual value of "bet"** (P2 then folds/calls 50/50; K wins every showdown):

$$
u_1(\text{bet}, K\,\text{vs}\,J) = u_1(\text{bet}, K\,\text{vs}\,Q) = 0.5(+1) + 0.5(+2) = 1.5
$$
$$
v_1(I_{1K},\text{bet}) = \tfrac16(1.5) + \tfrac16(1.5) = 0.5
$$

**Counterfactual value of "check"** (P2 checks → showdown +1; or P2 bets → P1 in a later info set, which under uniform play is worth $0.5$):

$$
u_1(\text{check}, \cdot) = 0.5(+1) + 0.5(0.5) = 0.75 \;\Rightarrow\; v_1(I_{1K},\text{check}) = \tfrac16(0.75)\cdot 2 = 0.25
$$

**Regrets and update** (current strategy uniform, so $v_1(I_{1K}) = 0.5(0.5)+0.5(0.25) = 0.375$):

$$
r(\text{bet}) = 0.5 - 0.375 = +0.125, \qquad r(\text{check}) = 0.25 - 0.375 = -0.125
$$
$$
R^+ = (\text{bet}: 0.125,\ \text{check}: 0) \;\Rightarrow\; \sigma^2(I_{1K}) = (\text{bet}: 1.0,\ \text{check}: 0)
$$

Iteration 1 already says "bet the King," which is correct. Across many iterations (with P2 also updating), $\bar\sigma$ converges to the Kuhn equilibrium family ($\alpha$ controls bluff/slow-play frequencies). **Remember: the deliverable is $\bar\sigma$, not the latest $\sigma^t$.**

---

## 9. Scaling CFR to large games

Vanilla CFR/CFR+ traverse the entire tree per iteration — fine for Kuhn (24 regret entries), impossible for Pokémon. Three tools, used together:

1. **MCCFR (Monte Carlo CFR):** sample paths/subtrees per iteration instead of full traversal (outcome / external / public-chance sampling). Unbiased; cheaper iterations, higher variance.
2. **Abstraction / bucketing:** collapse strategically-similar private states into a manageable set. **This is where Pokémon's continuous stat-point (EV) space and combinatorial item/move space get tamed** — via the meta-prior clustering (`src/meta_priors/clustering.py`): each species reduces to a few **archetype sets** (top-$k$ by prior), so the opponent's per-slot private space becomes "one of $k$ archetypes," and the history sum becomes finite. The meta-priors pipeline is the project's abstraction layer, not just flavor.
3. **Function approximation:** replace explicit tables with neural networks that generalize across info sets (next two sections).

### 9.1 Deep CFR (distinct from GT-CFR — know the difference)

Deep CFR (Brown et al., 2019) scales tabular CFR with **two networks**:

- An **advantage/regret network** that predicts a quantity **proportional to cumulative regret**. Its training targets ("regret targets") are **sampled instantaneous** regrets stored across iterations; trained on the iteration-weighted mean ≈ cumulative regret. It is typically **retrained from scratch each iteration** (on an accumulated reservoir buffer) to avoid chasing a moving target. Its input is just info-set features — the strategy dependence is baked into the *data*, not passed as input.
- An **average-strategy network** that memorizes $\bar\sigma$. *Why a second network?* Because $\bar\sigma$ (the actual Nash approximation) is the historical average over all iterations and is **not** recoverable from the final regret table — only the last iterate is. So you need a separate object to remember the average.

Deep CFR is a way to *solve* a game offline with sampling + approximation. It is **not** the search-at-play-time design we want; that is GT-CFR.

---

## 10. GT-CFR and Player of Games (the destination)

GT-CFR is "CFR shaped like AlphaZero": grow a search tree on the fly, evaluate its leaves with a network, but run **CFR+** (not UCB backups) to produce a **mixed** strategy that is sound under imperfect information. This is the algorithm the project is ultimately targeting (Player of Games, Schmid et al., 2021).

### 10.1 The CVPN — one network, two heads

| Head | Output | Synonyms | Role |
|---|---|---|---|
| **Policy head** | distribution over actions per info set | "policy," "prior," "policy prior," $P(I,a)$ | **input/guide** to the search (PUCT term + warm start) |
| **Value head** | a **vector of counterfactual values**, one per info state | "CFV," $v_i(I)$ | **leaf evaluation** inside the search |

Two distinctions that prevent the most common errors:

- **Policy-head prior $P$ ≠ average strategy $\bar\sigma$.** $P$ is the cheap *input* guess; $\bar\sigma$ is the refined *output* of the search (and the policy training target). Over generations $P \to \bar\sigma$ (distillation), exactly like AlphaZero's policy head chases the MCTS visit distribution.
- **The value head is a *vector*, not a scalar** (the big architectural contrast with AlphaZero). At one public state there are many possible private states; each is its own info set with its own value. The net must evaluate the **whole belief distribution at once** and return a value per info set, because CFR needs all of them. A scalar cannot express "great if they hold the Scarf set, terrible if they hold the Specs set"; a vector can. This is *why* PoG handles hidden information.

The CVPN does **not** predict regrets (that is the Deep CFR design). It predicts **values** and **priors**; CFR machinery derives regrets inside the search. Because the net only evaluates *leaves* (returning "value under good play," not "value at iteration $t$"), its outputs are **not** indexed by the inner-loop $t$ — the $t$-dependence lives only in the ephemeral tables of internal nodes.

### 10.2 The search: two interleaved phases

Per inner-loop iteration, on the **current** (growing) tree:

- **Regret-update phase (CFR+):** traverse the current tree; at each info set compute counterfactual values for each action; at **leaves**, substitute the CVPN's CFV estimate for the unknown continuation value; update cumulative regrets (regret-matching+) and accumulate $\bar\sigma$.
- **Expansion phase (PUCT-guided growth):** walk down and add new nodes at a chosen leaf, querying the CVPN for the new frontier's values and priors.

Selection during expansion combines exploitation (the current CFR strategy from the **ephemeral** regret table) with exploration (policy prior + visit bonus):

$$
\text{score}(I,a) = \underbrace{\sigma^t(I,a)}_{\text{from regret table (exploit)}} + \; c \cdot \underbrace{P(I,a)}_{\text{policy head}} \cdot \frac{\sqrt{\sum_b N(I,b)}}{1 + N(I,a)}
$$

Note the roles vs. MCTS: the "$Q$-analogue" is the regret-derived current strategy, "$P$" is the policy-head prior (**not** predicted regret), "$N$" is visit counts. The **output** at the root is the **average strategy** $\bar\sigma$ (you sample your real move from it) — not visit counts.

### 10.3 Where CFVs plug into the equations

The CVPN's CFVs are the **leaf boundary condition** for the CFR backup. At a leaf $\ell$ the search uses $\hat v_i(\ell, I)$ in place of the recursive $u_i^\sigma$; those propagate up to form internal-node values $v_i^t(I,a)$, which feed straight into $r_i^t(I,a) = v_i^t(I,a) - v_i^t(I)$. Net supplies frontier numbers → CFR turns them into regrets → regret matching turns those into the strategy → average is the output.

### 10.4 The ephemeral regret table

Carried **per search** because it *is* the CFR computation (produces each iteration's strategy, accumulates $\bar\sigma$, and feeds the expansion's exploit term). **Discarded after acting** because the next decision is a different tree. It is *not* the training target; the targets are derived *from* it (the refined CFVs and $\bar\sigma$).

---

## 11. Worked example B — GT-CFR on Kuhn (tree growth + leaf overwrite)

This shows the "GT" mechanic and why bootstrapping with the net still injects real information. Focus again on $I_{1K}$; reach weights $\tfrac16$ each; start uniform.

**Step 1 — shallow tree, leaves scored by an imperfect CVPN.** Suppose the (under-trained) net returns per-history leaf values that *wrongly* favor checking the King:

| Leaf (action, opp card) | CVPN value to P1 |
|---|---|
| bet, $(K,J)$ / bet, $(K,Q)$ | $0.6$ / $0.6$ |
| check, $(K,J)$ / check, $(K,Q)$ | $1.2$ / $1.2$ |

$$
v_1(I_{1K},\text{bet}) = \tfrac16(0.6)\cdot2 = 0.2, \quad v_1(I_{1K},\text{check}) = \tfrac16(1.2)\cdot2 = 0.4
$$
$$
v_1(I_{1K}) = 0.5(0.2)+0.5(0.4) = 0.3 \;\Rightarrow\; r(\text{bet}) = -0.1,\; r(\text{check}) = +0.1
$$
$$
R^+ = (\text{bet}: 0,\ \text{check}: 0.1) \;\Rightarrow\; \sigma^2(I_{1K}) = (\text{bet}: 0,\ \text{check}: 1)
$$

The search has (wrongly) committed to checking, trusting the net.

**Step 2 — expansion grounds the "bet" branch in real terminals.** Expanding the bet node reveals P2's call/fold nodes, whose children are **terminal** (real payoffs, no net needed). Under P2's uniform play, betting the King is worth $1.5$ per history, so the CFV is **overwritten**:

$$
v_1(I_{1K},\text{bet}) : 0.2 \;\longrightarrow\; \tfrac16(1.5)\cdot 2 = 0.5
$$

This is the "new information": it comes from **reaching terminals during expansion**, not from the net's own beliefs.

**Step 3 — next CFR iteration swings the policy back.** With $\sigma^2 = (\text{bet}:0,\text{check}:1)$, so $v_1(I_{1K}) = 0.4$:

$$
r(\text{bet}) = 0.5 - 0.4 = +0.1,\quad r(\text{check}) = 0.4 - 0.4 = 0
$$
$$
R^+ = (\text{bet}: 0.1,\ \text{check}: 0.1) \;\Rightarrow\; \sigma^3(I_{1K}) = (0.5,\ 0.5)
$$

Expanding the check branch too (true value $0.25$) pushes further toward betting. **Lesson:** the net provides a fast prior; expansion + terminals + the CFR improvement operator correct it; over outer-loop generations that correction is distilled back into the net.

---

## 12. The outer training loop in detail

**Data generation.** Play full self-play games; every decision runs a GT-CFR search. Each search emits a training tuple at the searched state:

$$
\big(\;\beta,\;\; v^{\text{search}}(\beta),\;\; \bar\sigma(\beta)\;\big)
$$

- $\beta$ — the encoded **public belief state** (public state + belief ranges over each player's private states).
- $v^{\text{search}}(\beta)$ — the **search-refined CFVs** (value target). Better than the raw net because search = net + terminals + CFR improvement; grounding traces to real $\pm1$ payoffs.
- $\bar\sigma(\beta)$ — the search's **average strategy** (policy target).

This is the GT-CFR analogue of AlphaZero's $(s, z, \pi_{\text{MCTS}})$.

**Replay buffer.** A fixed-capacity **sliding window** (FIFO): newest tuples enter, stalest (from weak early checkpoints) are evicted. The data is *not* wholesale discarded each generation — only stale data ages out.

**Training.** A trainer samples **minibatches** and steps:

$$
L(\theta) = \big\lVert \hat v_\theta(\beta) - v^{\text{search}}(\beta) \big\rVert^2 \;+\; \text{CE}\big(\hat\pi_\theta(\beta),\, \bar\sigma(\beta)\big) \;+\; \lambda \lVert\theta\rVert^2
$$

It periodically publishes a checkpoint; self-play workers reload it. **System level: continuous** (generation and training run concurrently). **Gradient level: batched** (minibatch SGD; weights are *not* reset — contrast Deep CFR's from-scratch advantage-net retrains).

---

## 13. Mapping the theory onto Pokémon VGC

**State decomposition.**
- **Public state** — everything both players have observed: revealed Pokémon, revealed moves, items revealed by effect, HP, status, stat stages, field conditions (weather/terrain/Tailwind/Trick Room) **with their turn counters**, Mega availability.
- **Private state** — your hidden info: back-row identities, unrevealed moves/items, stat-point spreads.
- **Info set** — public state + *your* private state (you act on it). The **opponent's** info set bundles every private state consistent with what they have observed of you.
- **Belief range** — a probability distribution over the opponent's possible private states, supplied by **meta priors** and reduced to top-$k$ **archetype sets** per species (Section 9, `src/meta_priors/`).

**Per-turn tree structure** (how simultaneity, chance, and forced switches appear):

```
[your decision @ I_you]
  ├─ joint action A1 ─┐
  ├─ joint action A2 ─┤→ [opp decision @ I_opp]   ← SAME I_opp under every A_k
  └─ joint action Ak ─┘     (encodes simultaneity: opp can't see your choice)
                              ├─ opp joint B1 ─┐
                              └─ opp joint Bm ─┤→ [chance node]
                                                 ├─ outcome (rolls/accuracy/crit/secondary/multi-hit)
                                                 │     └─ next state → [your decision @ I_you']  (or terminal)
                                                 └─ outcome where a mon faints
                                                       └─ [forced switch] ← unilateral; no opposing decision
```

- **Simultaneity** is encoded by routing all of your action branches into the *same* opponent info set $I_{\text{opp}}$, so the opponent's one strategy cannot depend on your choice.
- **Chance nodes** carry all randomness (damage roll, accuracy, crit, secondary effect, multi-hit). Sample one outcome per traversal (Monte Carlo) or enumerate a few discrete branches for high-variance, high-impact events.
- **Forced switches** (after a faint) are **unilateral** single-player decision nodes that break the simultaneous pattern until normal turn structure resumes.
- **Terminal** = one side has no Pokémon left → payoff $\pm1$.
- **Action space** ≈ 100 joint actions/turn; represent as a fixed-size vector with **illegal-action masking** (choice-lock, disable, taunt, no-PP, etc. flip mask bits). See project-overview rule.

---

## 14. Recommended build order

Derived repeatedly across the discussion; reuse it as the default roadmap. (Cross-reference the milestones in `.cursor/rules/agent/overview.mdc`.)

> **Priority (rebalanced toward shipping).** The objective is an agent that *works in real games*, reached through visible, celebratable increments — not theoretical completeness. Treat Phases 1–3 (the MCTS / determinization line) as a deliberately **brief pit-stop**, held only long enough to clear two exit criteria: (1) **infrastructure is proven** — we can simulate Champions games and theoretical turns, encode state, mask actions, and run the self-play loop end to end; and (2) **feasibility is shown** — a from-random agent becomes a semi-coherent battler that **reliably beats earlier versions of itself** (even while "cheating" with perfect information). The moment both hold, stop polishing the baseline and commit to **GT-CFR / Player of Games (Phase 4) as the north star and the real product.** The phases below are still the right *order*; this note rebalances how long to *dwell* in each.

1. **Perfect-info AlphaZero.** Reveal both teams' full sets. MCTS + two-head net + self-play. Validates the hard engineering: `@pkmn/sim` integration & state cloning, state encoding, NN, training loop, action masking. *Hardest engineering phase.*
2. **Add stochastic resolution.** Introduce chance nodes (damage/accuracy/crit/secondary/multi-hit). Still perfect-info, now stochastic.
3. **Imperfect-info baseline (determinized IS-MCTS).** Hide opponent items/back-row/unrevealed moves; sample worlds from meta priors. Accept that **strategy fusion** is happening; measure exploitability so you know the ceiling.
4. **Replace search with GT-CFR + CVPN.** Swap UCB backups for CFR+; change the value head from scalar to a **vector of CFVs**; reuse the backbone, encoding, sim, and training scaffolding. *Hardest algorithmic phase;* should be measurably less exploitable than Phase 3.
5. **Refine / harden.** Better priors and encoding, larger capacity, belief tracking; later, push toward robust mixed strategies / Nash hardening (R-NaD; see `research/notes.md`).

> Risk note: starting at full Player-of-Games complexity strands you in theory with nothing running; staying on determinized MCTS forever silently plateaus at "decent but exploitable." The phased order de-risks both.

---

## 15. Glossary and quick-reference equations

**The three nested levels of game state (do not conflate).** A common slip is to treat `info set` as "the public information." It is *not* — an info set already includes the acting player's own private state. There are **three** levels, with strict containment $\text{public state } \beta \;\supset\; \text{info set } I \;\supset\; \text{history } h$:

- **History $h$** — the full ground truth: the complete path including *both* players' private states and all chance outcomes. Perfect information; a single node in the full game tree. Most granular.
- **Information set $I$** — *one* player's view: the **public state plus that player's own private state** (everything *that* player knows). Bundles every history that differs only in the *other* player's hidden info (and chance). The acting player cannot tell those histories apart, so must play one strategy across the whole $I$.
- **Public state / public belief state $\beta$** — common knowledge only: what *both* players have observed. Bundles all histories regardless of *either* player's private state; decomposes into many info sets per player (one per possible private state of that player). This is the network input ($\beta$ in §12).

**Where each appears in GT-CFR (the usual confusion):** the CVPN value head returns one counterfactual value **per info set** ($v_i(I)$, §10.1) — *not* per history. In poker terms it emits a value for each of your 1326 *hands* (your info sets), not one per joint hand-vs-hand deal. Histories are the *summands inside* a single info-set value, weighted by the counterfactual reach (the belief): $v_i(I,a)=\sum_{h\in I}\pi_{-i}^\sigma(h)\,u_i^\sigma(h\cdot a)$ (§7.1). So a regret update at a public state consumes per-**info-set** CFVs (one per enumerated private state); per-**history** continuation values are what those CFVs are built from.

**Symbols.** $i$ player / $-i$ opponent; $h$ history (= node); $I$ info set; $A(I)$ legal actions; $\sigma$ strategy, $\sigma^t$ iterate $t$, $\bar\sigma$ average strategy; $\pi^\sigma(h)$ reach prob, $\pi_{-i}^\sigma(h)$ counterfactual reach; $u_i$ utility/payoff; $v_i(I,a)$ counterfactual value; $r_i^t / R_i^T$ instantaneous / cumulative regret; $P(I,a)$ policy-head prior; $N$ visit count; $\beta$ public belief state.

**Synonym map (stop the vocabulary churn).**
- **policy head output** = prior = policy prior = $P(I,a)$ → guides/seeds search; trained toward $\bar\sigma$.
- **average strategy** = $\bar\sigma$ → the search's *output*; what you sample your move from; the policy training target. Computed in the ephemeral tables, *not* a live network output.
- **value head output** = counterfactual value = CFV = $v_i(I)$ → leaf evaluation feeding the regret backup; a *value*, not a probability; a **vector** (one per info set) in GT-CFR.
- **counterfactual reach probability** = $\pi_{-i}(h)$ → a *weight* inside the CFV sum; from chance + opponent strategy; never a network output.

**Core equations.**

$$
\text{PUCT (AlphaZero): } \; Q(s,a) + c\,P(s,a)\frac{\sqrt{\sum_b N(s,b)}}{1+N(s,a)}
$$
$$
\text{counterfactual value: } \; v_i(I,a) = \sum_{h\in I}\pi_{-i}^\sigma(h)\,u_i^\sigma(h\cdot a), \qquad v_i(I)=\sum_a \sigma(I,a)v_i(I,a)
$$
$$
\text{regret: } \; r_i^t(I,a)=v_i^t(I,a)-v_i^t(I), \qquad R^{T,+}(I,a)=\max\!\big(R^{T-1,+}(I,a)+r^T(I,a),0\big)
$$
$$
\text{regret matching: } \; \sigma^{T+1}(I,a)=\frac{R^{T,+}(I,a)}{\sum_b R^{T,+}(I,b)}
$$
$$
\text{GT-CFR loss: } \; L(\theta)=\lVert \hat v_\theta(\beta)-v^{\text{search}}(\beta)\rVert^2 + \text{CE}(\hat\pi_\theta(\beta),\bar\sigma(\beta)) + \lambda\lVert\theta\rVert^2
$$

---

## 16. References

**Papers**
- AlphaZero — Silver et al., 2018. https://arxiv.org/abs/1712.01815
- Regret matching — Hart & Mas-Colell, 2000 (Econometrica).
- CFR — Zinkevich et al., 2007. https://poker.cs.ualberta.ca/publications/NIPS07-cfr.pdf
- MCCFR — Lanctot et al., 2009.
- CFR+ / heads-up limit hold'em — Tammelin 2014; Bowling et al., 2015 (Science).
- DeepStack — Moravčík et al., 2017. https://arxiv.org/abs/1701.01724
- Deep CFR — Brown et al., 2019. https://arxiv.org/abs/1811.00164
- Player of Games (GT-CFR) — Schmid et al., 2021. https://arxiv.org/abs/2112.03178
- R-NaD / DeepNash (Stratego) — Perolat et al., 2022. https://arxiv.org/abs/2206.15378

**Internal artifacts**
- `research/notes.md` — algorithm survey, sim-engine choices, data sources (engineering counterpart).
- `.cursor/rules/agent/overview.mdc`, `.cursor/rules/game-domain/overview.mdc`, `.cursor/rules/game-domain/legality.mdc`.
- `docs/article_summary*.md` — verbose conversational derivations this document distills.

---

*If you are a future agent picking up implementation: this document is theory only. For what to build next and with which tools, start from `research/notes.md` and the build order in Section 14. Consider the `create-rule` skill to add a short auto-loaded rule pointing here, and `relentless-executor` once a phase has a concrete success criterion.*
