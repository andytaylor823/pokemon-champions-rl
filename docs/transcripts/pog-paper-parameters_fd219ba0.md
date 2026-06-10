# Player of Games Paper Parameters

**Source:** `agent-transcripts/fd219ba0-c42a-48e6-8fc2-d58f7f206e6d/fd219ba0-c42a-48e6-8fc2-d58f7f206e6d.jsonl`

---

## User (Turn 1)

I was re-reading the PoG paper, and I didn't understand their concept of s and c -- the number of steps per....something. In AlphaZero, every iteration of the MCTS, it expanded a new node, right? Are they saying in this paper that they don't always explore a new node?

I'm imagining it like this. We're at turn T (in Pokemon), and we're in the middle of doing the GT-CFR simulation for this turn, say at iteration K. We walk forward in theoretical time down the tree, as much as it's previously explored, using PUCT to select our action at each step. Then, once we encounter a node we haven't seen before, we expand it, query the CVPN for that next leaf's CFVs and then walk back up to update the regrets. We then advance to iteration K+1 and repeat this until we run out of compute budget and make our real choice based on this info.

First, is my understanding generally correct? Second, where in all this does that paper's c and s values come in? Refer to the parameters in @toy_examples/kuhn_poker/self_play.py if it's present there, too

---

## Assistant (Turn 1)

*[Tools: Read(self_play.py), Read(gt-cfr-theory.md)]*

Let me read the relevant files to give you a precise answer.

The user is asking about the Player of Games (PoG) paper's concepts of `s` and `c` parameters, and how they relate to the GT-CFR search process. They want to understand:

1. Whether their mental model of the search is correct
2. What `s` and `c` mean in the PoG paper
3. How these map to parameters in the Kuhn poker implementation

Let me read the relevant files to answer this properly.

*[Tools: Read(gt_cfr_search.py)]*

Great question -- your mental model has the right spirit but conflates two fundamentally different iteration structures. Let me untangle it.

## Your model is MCTS, not GT-CFR

What you described -- "walk down one path using PUCT, expand the first unseen node, evaluate the leaf, back up" -- is precisely how AlphaZero MCTS works. Each MCTS simulation = one path = one new node. After $T$ simulations you have ~$T$ nodes, deep along promising lines. That's correct for Section 3.2 of the theory doc.

**GT-CFR is structurally different.** A CFR+ "iteration" is not a single-path walk. It is a **full traversal of the entire currently-expanded tree**, updating regrets at *every* info set in one pass. Look at the core loop:

```335:343:toy_examples/kuhn_poker/gt_cfr_search.py
    for t in range(1, n_iterations + 1):
        # Traverse all consistent deals, updating both players' regrets
        for p1_card, p2_card in consistent_deals:
            _cfr_traverse(root, net, PLAYER_1, p1_card, p2_card, 1.0, 1.0, t)
            _cfr_traverse(root, net, PLAYER_2, p1_card, p2_card, 1.0, 1.0, t)

        # Incremental expansion (only in non-full-expand mode)
        if not full_expand and t % expansion_interval == 0 and t < n_iterations:
            _puct_expand(root, net, c_puct)
```

Each iteration $t$ walks **all** 6 deals times 2 traversing players = 12 full-tree passes. Every expanded node gets its regrets updated. Then, separately and less frequently, one new node gets added.

## What `s` and `c` are

In the PoG paper, these two parameters control the **expansion phase** specifically:

- **`s`** = the number of CFR+ iterations to run **between** tree expansions. In the Kuhn code, this is `expansion_interval` (default 10). It means: do 10 full-tree CFR+ iterations (letting regrets settle), then grow the tree by one node, then do 10 more, grow again, etc.

- **`c`** = the PUCT exploration constant used during expansion to decide **which** leaf to expand next. In the code, this is `c_puct` (default 2.0).

You can see both in `TrainingConfig`:

```108:119:toy_examples/kuhn_poker/self_play.py
    c_puct: float = Field(
        default=2.0,
        description="Exploration constant for PUCT-guided tree expansion. Controls the "
        "trade-off between exploiting the current strategy (low c) and exploring "
        "actions the NN policy recommends (high c). Only matters when full_expand=False."
    )
    expansion_interval: int = Field(
        default=10,
        description="Expand one new tree node every this many CFR+ iterations. Only "
        "active when using incremental expansion (full_expand=False). Lower = faster "
        "tree growth but fewer CFR iterations between expansions."
    )
```

And they flow through to the search call:

```315:321:toy_examples/kuhn_poker/self_play.py
            sigma_bar, cfvs, _ = gt_cfr_search(
                root_state=state,
                net=self.net,
                n_iterations=self.config.search_iterations,
                c_puct=self.config.c_puct,
                expansion_interval=self.config.expansion_interval,
            )
```

## Why the decoupling matters

The reason GT-CFR can't expand a new node every iteration (like MCTS does) is that **CFR needs repetition on a fixed tree to converge**. Regret matching works by accumulating regrets over many iterations and averaging -- if you changed the tree topology every iteration, the regrets would never stabilize and $\bar\sigma$ would not converge toward Nash.

So the rhythm is:

1. Start with a small tree (just the root expanded, children are leaves scored by the CVPN)
2. Run `s` full-tree CFR+ iterations -- regrets accumulate, strategy improves, $\bar\sigma$ starts converging on this tree
3. Use PUCT to pick the most promising unexpanded leaf and expand it (one CVPN call for the new frontier)
4. Run `s` more CFR+ iterations -- now the tree is slightly deeper, and some leaf values that were CVPN estimates are now grounded in deeper structure (or real terminals)
5. Repeat until the budget `n_iterations` is exhausted
6. Read off $\bar\sigma$ and CFVs from the root, play your move, throw the whole tree away

The interplay between `s` and total budget determines how many nodes get added. With `n_iterations=100` and `expansion_interval=10`, you get ~9 expansion events (at $t = 10, 20, \ldots, 90$). For Kuhn this is moot since `full_expand=True` expands the whole tree upfront (it only has ~10 nodes). But for Pokemon, where the tree is enormous, incremental expansion is the only viable mode -- and `s` becomes a real tuning knob: too low and regrets don't settle between expansions; too high and you waste budget refining a shallow tree when going deeper would help more.

Now I have a thorough understanding. Let me answer the user's questions.

The user is asking about:
1. Whether their mental model of GT-CFR search is correct (the AlphaZero-like "walk down, expand one node, back up" model)
2. What `s` and `c` are in the PoG paper
3. How these map to the Kuhn poker implementation

The key insight is that the user's mental model is slightly wrong - they're conflating the AlphaZero MCTS model with GT-CFR. In AlphaZero MCTS, each iteration = one simulation = walk down, expand one leaf, evaluate, back up. 

But in GT-CFR, iterations work completely differently. Each CFR iteration is a full traversal of the entire expanded tree, computing counterfactual values at every info set and updating regrets everywhere. That's nothing like MCTS where each simulation follows a single path. The expansion happens on its own schedule, and the PoG paper's `s` parameter controls how many CFR iterations run between expansions - this maps directly to `expansion_interval` in the Kuhn implementation. I see the distinction now—the user is conflating GT-CFR with MCTS. In GT-CFR, each iteration traverses the entire tree to update regrets and strategies, while expansion happens separately every `s` iterations using PUCT to select which leaf to grow. This is fundamentally different from AlphaZero's single-path-per-iteration model. The `expansion_interval` parameter controls how frequently new nodes are added to the tree. So the key insight is that their mental model is fundamentally wrong—in GT-CFR, each iteration isn't a single path expansion like in AlphaZero MCTS, but rather a complete traversal of the entire current tree updating regrets everywhere. The expansion happens separately at interval `s`, meaning you run `s` CFR+ iterations to let regrets stabilize after adding a new node, then expand again. The parameter `c` controls exploration during that expansion phase.

---

## User (Turn 2)

I don't understand. Why do you run multiple CFR+ iterations between tree expansions? So you're running CFR+ (which is a full-tree process) on the same tree for s times in a row? This seems wasteful. What new information is gained by running CFR+ on the same tree a bunch of times in a row? Why is this valuable?

---

## Assistant (Turn 2)

This is the right question to ask -- it gets at the core difference between perfect and imperfect information games.

## Why one pass isn't enough: the strategies are *coupled*

In a perfect-information game, you can solve a fixed tree in **one backward pass** (minimax / backward induction). Walk from terminals to root, pick the best action at each node, done. One pass, exact answer.

In an imperfect-information game, **you cannot.** The reason is that the optimal strategy at one info set depends on the strategy at *other* info sets, which depends on the first one, which depends on... it's circular.

Concrete Kuhn example. Should P1 bet with a Queen at the root? That depends on:
- How often P2 calls a bet (P2's strategy at info sets `J:bet`, `Q:bet`, `K:bet`)
- But P2's calling frequency depends on what P1's bet *means* -- if P1 only bets Kings, P2 should fold everything except K. If P1 also bluffs with Jacks, P2 should call more.
- But P1's bluffing frequency with J depends on how often P2 calls...

This is a **fixed-point problem**. Nash equilibrium is a simultaneous solution to all these interdependent info sets. No single backward pass can find it because changing the strategy at one info set changes the counterfactual values at others.

## What changes between iteration $t$ and $t+1$ on the same tree

The tree topology doesn't change. The **strategies** do. Each iteration:

1. Compute counterfactual values at every info set **under the current strategy profile** $\sigma^t$
2. Compute instantaneous regrets: "how much better would action $a$ have been than what I actually played?"
3. Update cumulative regrets → derive $\sigma^{t+1}$ via regret matching
4. Accumulate into $\bar\sigma$ (the running average)

The new strategy $\sigma^{t+1}$ is different from $\sigma^t$ because the regrets changed. And when you traverse the tree under $\sigma^{t+1}$, the counterfactual values will be different again, because the *opponent's* strategy also updated.

Here's a sketch of what happens across iterations on the same tree:

| Iter | P1's J strategy | P2's Q:bet strategy | What happened |
|------|----------------|---------------------|---------------|
| 1 | bet 50%, check 50% (uniform) | call 50%, fold 50% (uniform) | Initial |
| 2 | bet 80% (bluffing looks great -- P2 folds half the time!) | call 70% (P1 bets a lot, better call more) | Both overreact |
| 3 | bet 40% (P2 is calling too much, stop bluffing!) | call 55% (P1 pulled back, maybe fold more) | Correction |
| 4 | bet 55% (P2 folding more, bluffing is good again) | call 60% | Oscillation dampens |
| ... | oscillates toward ~33% | oscillates toward ~67% | |
| $\bar\sigma$ | converges to Nash | converges to Nash | Average is stable |

The iterates $\sigma^t$ keep oscillating -- that's inherent to the dynamics. But the **time-weighted average** $\bar\sigma$ converges to equilibrium. More iterations on the same tree = tighter convergence of that average. This is the content of the regret-bound theorem: total regret grows as $O(\sqrt{T})$, so average regret $\to 0$, so $\bar\sigma \to$ Nash.

## The analogy

Think of it like two people negotiating by alternately adjusting their positions. One round of "I'll adjust, then you adjust" doesn't reach agreement -- they keep reacting to each other. But the *average* of all their proposals converges to a fair split. That's CFR. The "new information" at each iteration isn't from the tree -- it's from the opponent's updated strategy.

The user is asking a really good question. They don't understand why you'd run CFR+ multiple times on the same tree. Let me think about this carefully.

The key insight is: **CFR+ doesn't converge in one iteration.** Each iteration produces a *different* strategy (via regret matching on the updated cumulative regrets), and it's the **average** of all these strategies that converges to Nash. One iteration gives you a crude strategy; 100 iterations on the same tree gives you a much better average strategy.

Think of it like this: on iteration 1, you start with uniform strategies everywhere. You compute counterfactual values, get regrets, and update. Now the strategy at iteration 2 is different -- maybe it overreacts to what it learned. Iteration 3 corrects the overcorrection. 

The strategies oscillate around the equilibrium, but their average converges to Nash. This is fundamentally different from value iteration in an MDP, where a single backward pass gives you the exact answer if the tree is fully known. In imperfect information games, there's no single correct strategy to compute in one pass because both players are updating simultaneously, each player's optimal strategy depends on the other's, and Nash equilibrium is a fixed-point problem where neither player can improve given the other's strategy. Each CFR+ iteration is one step toward that fixed point, with strategies continuously adjusting.

The back-and-forth is concrete: Player 1 sees betting is profitable and shifts toward betting more, Player 2 observes this and calls more often, Player 1 then sees increased calling and bluffs less, and so on. This oscillation, when averaged across iterations, converges to the equilibrium mixed strategy. In perfect information games like chess, you could do one backward induction pass and get the minimax value, but imperfect information prevents this because a player doesn't know which node they're at within an information set—they must play the same strategy across all nodes in that set. This constraint means you can't solve it in one pass; you need iterative adjustment.

With a concrete Kuhn poker example, whether Player 1 should bet or check with a Queen depends on what Player 2 does after a bet, which in turn depends on what Player 2 holds and Player 1's overall betting strategy. If Player 1 only bets with a King, Player 2 should always fold a Queen and always call a King; if Player 1 also bluffs with a Jack, Player 2 should call more often. This circular dependency—where Player 1's strategy at Queen depends on Player 2's strategy, which depends on Player 1's strategy at Jack and King—is why multiple iterations are necessary to resolve it.

---

## User (Turn 3)

I just downloaded the Player of Games paper and added it to this repo. I want you to find in the paper where it explains what it calls the variable c (I think that's the same as our "s" value) and also where it says that GT-CFR runs multiple iterations of CFR+ inside it, rather than GT-CFR being an advancement completely beyond CFR+, eschewing it

---

## Assistant (Turn 3)

*[Tools: Read(PlayerOfGames.pdf)]*

*[Tools: Read(PlayerOfGames.pdf)]*

*[Tools: Grep]*

I found the key definition. Let me read more of the Materials and Methods section for the formal details.

Let me find the key section. The user wants to see:
1. Where the paper defines `s` and `c`
2. Where it says GT-CFR runs multiple CFR+ iterations inside it

I found a critical passage already. Let me re-read it:

> "When reporting the results we use the notation SOG(s, c) for SOG running GT-CFR with s total expansion simulations, and c expansion simulations per regret update phase, so the total number of GT-CFR iterations is then s/⌈c⌉."

So:
- `s` = total number of expansion simulations
- `c` = number of expansion simulations per regret update phase

- Total GT-CFR iterations = s/⌈c⌉

This means each GT-CFR iteration consists of one regret update phase followed by one expansion phase, where the expansion phase runs `c` simulations. So if we have `s` total simulations and `c` simulations per expansion, we get s/c iterations total.

With SOG(8000, 10) for example: 8000 total simulations, 10 simulations per GT-CFR iteration, giving us 800 iterations. Each iteration does a full-tree CFR+ regret update followed by expanding 10 nodes.

Let me look back at the paper's formal definition to make sure I'm understanding this correctly.

*[Tools: Read(PlayerOfGames.pdf)]*

Good call, let me find the exact passages. The PDF is large so let me search the key sections I already partially read.

The two things you want are both in the "Search via Growing-Tree CFR" section on pages 14-15. Here are the exact passages:

**Where `s` and `c` are defined** (lines 613-625):

> "GT-CFR starts with an initial tree, L0, containing β and all of its child public states. Then each iteration, t, of GT-CFR consists of two phases:
> 1. The regret update phase runs **several public tree CFR updates** on the current tree Lt.
> 2. The expansion phase expands Lt by adding new public states via simulation-based expansion trajectories, producing a new larger tree Lt+1."

> "When reporting the results we use the notation **SOG(s, c)** for SOG running GT-CFR with **s total expansion simulations**, and **c expansion simulations per regret update phase**, so the total number of GT-CFR iterations is then s/⌈c⌉. **The c can be fractional, so e.g. 0.1 indicates a new node every 10 regret update phases.**"

And then the line that directly answers your question about multiple CFR+ iterations (lines 626-631):

> "The regret update phase runs **⌈1/c⌉ updates (iterations) of public tree CFR** on Lt using simultaneous updates, **regret-matching+**, and **linearly-weighted policy averaging**."

That last sentence is key: "regret-matching+" and "linearly-weighted policy averaging" *are* CFR+. The paper is explicitly saying GT-CFR runs CFR+ as its internal engine.

## Correcting what I told you earlier

I need to correct my earlier mapping. I said the paper's `s` was our `expansion_interval` and `c` was our `c_puct`. That was wrong. Here's the actual mapping:

| Paper (SOG notation) | Meaning | Our Kuhn code |
|---|---|---|
| `c` | expansion simulations per regret update (the **ratio**) | ≈ `1 / expansion_interval` |
| `s` | total expansion simulations across the whole search | ≈ `n_iterations / expansion_interval` |
| `c_puct` (separate variable, used in expansion) | PUCT exploration constant | `c_puct` |

So the paper's `c` is **not** our `c_puct` -- they're completely different variables that unfortunately both use the letter "c." The paper's `c` controls how many CFR updates happen between expansions:

- **c = 0.1** → ⌈1/0.1⌉ = **10 CFR+ updates per expansion** (expand rarely, let regrets converge -- the imperfect-info regime). This corresponds to our `expansion_interval=10`.
- **c = 1** → 1 CFR+ update per expansion (balanced).
- **c = 10** → 10 expansions per CFR+ update (expand aggressively -- the perfect-info regime, where regret convergence isn't needed).

The paper's poker experiments use SOG(10, 0.01), meaning c=0.01 → **100 CFR+ updates between each expansion**. That's how much settling time imperfect-info games need. And the passage on page 12 (line 533-536) confirms this tuning matters:

> "Intuitively, we would expect c = 1 (corresponding to one regret update per expansion simulation) to be best choice. Due to these computational constraints, we chose by hand a small number of values for c > 1. Interestingly, we did notice that c = 1 is not always the best choice in practice."

So to directly answer: yes, the paper is explicit that GT-CFR is CFR+ **wrapped in** a growing-tree loop, not a replacement for it. Each GT-CFR iteration runs one or more full-tree CFR+ passes, then optionally grows the tree.

Now I have the crucial passage. Let me pull out the key quotes. The most important ones:

From pages 14-15 (lines 613-646):

"GT-CFR starts with an initial tree, L0, containing β and all of its child public states. Then each iteration, t, of GT-CFR consists of two phases:
1. The regret update phase runs several public tree CFR updates on the current tree Lt.
2. The expansion phase expands Lt by adding new public states via simulation-based expansion trajectories, producing a new larger tree Lt+1."

"When reporting the results we use the notation SOG(s, c) for SOG running GT-CFR with s total expansion simulations, and c expansion simulations per regret update phase, so the total number of GT-CFR iterations is then s/⌈c⌉. The c can be fractional, so e.g. 

0.1 indicates a new node every 10 regret update phases."

"The regret update phase runs 1/c updates (iterations) of public tree CFR on Lt using simultaneous updates, regret-matching+, and linearly-weighted policy averaging."

I need to reconsider my earlier interpretation of the notation. Looking at this more carefully, s represents the total expansion simulations while c is the expansion simulations per regret update phase, making the total GT-CFR iterations s/⌈c⌉. The regret update phase itself runs 1/c iterations of public tree CFR.

But wait, the dimensional analysis isn't working out. If c equals 10 expansions per regret update phase, then 1/c would be 0.1 CFR updates, which seems backwards. Let me reconsider what c actually represents—if it's the ratio of expansions to regret updates, then c=10 means 10 expansions happen during each regret update phase, while c=0.1 would mean one expansion every 10 regret updates. Yet the formula "1/c updates" still doesn't align properly with this interpretation. I'm working through some concrete examples to understand the relationship between c and the number of CFR updates. When c is fractional like 0.1, it means fewer expansions per regret update cycle, so you'd need more CFR iterations overall. But when c is larger like 10, you get more expansions per cycle with fewer CFR updates each time. The notation "1/c updates" is confusing me though—if c=10, that would suggest 0.1 updates, which doesn't make sense as a discrete count. So with c=10, that means 1/10 of a CFR update per regret phase, which doesn't make sense unless they're using ceiling notation or I'm misinterpreting the fraction notation in the PDF. Let me look more carefully at what they actually mean by this phrasing. Ah, so the formula is s/⌈c⌉ where s is total expansions and c is expansions per regret update. For SOG(8000, 10), that's 8000/10 = 800 GT-CFR iterations. But for SOG(10, 0.01) in poker, I get 10/1 = 10 iterations with only 10 total expansions, meaning each iteration has roughly one expansion—that seems like it should give me fractional expansions per iteration, which doesn't quite add up. I'm working through the different values of c to verify the formula. With c=0.1, I get 10 CFR updates per iteration but only 1 expansion, and with c=0.01, that jumps to 100 CFR updates per iteration. Now I'm checking if the total iterations formula s/⌈c⌉ holds up across the test cases. CFR updates = 100 CFR updates, 100 expansions
- SOG(10, 0.01): 10 iterations × 100 CFR updates = 1000 CFR updates, 10 expansions

This pattern reveals the fundamental tradeoff: imperfect information games like poker need frequent regret updates between expansions to converge, while perfect information games like chess or Go benefit from rapid expansion since regret convergence isn't the bottleneck. The paper suggests c=1 as the intuitive sweet spot, though fractional values like 0.1 allow even finer control over the update frequency.

So c < 1 means multiple regret updates per expansion (the imperfect-info regime where CFR convergence matters), while c > 1 means multiple expansions per regret update (the perfect-info regime where tree expansion dominates).

Now mapping this to our Kuhn code: `expansion_interval` corresponds to roughly ⌈1/c⌉ when c ≤ 1, so an expansion_interval of 10 is like c=0.1. I need to correct my earlier claim that `s` maps to `expansion_interval` — that was wrong. Instead, the paper's `c` parameter relates to our `expansion_interval` through that inverse relationship.

The total number of expansion events in our code is approximately `n_iterations / expansion_interval`. With 100 iterations and an interval of 10, we'd get around 9 expansion events plus 100 CFR updates, which mirrors the paper's setup with c=0.1 and s=9.

Looking back at what the user actually asked: they wanted to know where the paper defines `c` and `s`, and where it describes GT-CFR running multiple CFR+ iterations. The key passage is on page 15 (lines 613-625), where it explains that each GT-CFR iteration has two phases, starting with the regret update phase running several public tree CFR updates.

The paper explicitly defines `c` as the expansion simulations per regret update phase—it can be fractional, so 0.1 means a new node every 10 regret update phases. The regret update phase runs 1/c iterations of public tree CFR using regret-matching+ with linearly-weighted policy averaging, which is exactly CFR+. This directly answers both questions: the definitions are right there in lines 620-629, and it clearly states that GT-CFR uses CFR+ as its inner mechanism. The paper's `c` represents expansion simulations per regret update phase, not the PUCT constant. So the mapping is: `s` is total expansion simulations across the entire search, `c` is the ratio of expansions to regret updates, and our code's `expansion_interval` corresponds to ⌈1/c⌉ — the number of CFR+ iterations between expansions.

---

## User (Turn 4)

/create-rule some minor indexing of key sections for this Player Of Games paper, since it's 60 some pages long. More than just the table of contents. Include what you found here in your search. Add to the end of the rule that as more relevant sections are found, the rule should be updated with the locations of the new informations

---

## Assistant (Turn 4)

*[Tools: Read(PlayerOfGames.pdf), Read(PlayerOfGames.pdf), Read(PlayerOfGames.pdf), Read(PlayerOfGames.pdf), Read(PlayerOfGames.pdf), Read(PlayerOfGames.pdf)]*

Let me scan more of the paper to build a thorough index before creating the rule.

The user wants me to create a Cursor rule that indexes key sections of the Player of Games paper, since it's ~63 pages long. They want more than just a table of contents - they want specific locations of important content, including what I found during our conversation. Let me first scan more of the paper to build a good index, then create the rule.

Let me read more sections to get a comprehensive index.

*[Tools: Read(PlayerOfGames.pdf)]*

---

## User (Turn 5)

Don't use line numbers to mark where the info lives -- use page numbers and (sub) section markers, and even rough spots on the page if you can to identify where the text is. I guess you can also include the line numbers too for future coding agents' ease, but I want to clarify that you should tell me where sections are in the text based on stuff BESIDES the line numbers

---

## Assistant (Turn 5)

*[Tools: Write(player-of-games-paper-index.mdc)]*

Good call. I have enough context from scanning the paper. Let me create the rule now.

The user wants me to create a rule that indexes the Player of Games paper using page numbers, section names, and rough positions on the page (top/middle/bottom) rather than just line numbers. I should include line numbers too for coding agents, but the primary reference should be human-readable page/section markers.

Let me map out the page numbers from the "-- X of 63 --" markers I saw in the PDF text:

- Page 1 (line 1): Title, Abstract
- Page 2 (line 76): Introduction
- Page 3 (line 119): Background and Terminology (bottom of page 3 → continues)

- Page 4 (line 175): Tree Search and Machine Learning
- Page 5 (line 218): Game-Theoretic Reasoning and CFR (bottom of page 5 → continues)
- Page 6 (line 271): Imperfect Information Search, Decomposition, and Re-Solving
- Page 7 (line 313): Related Work (bottom of page 7 → continues)
- Page 8 (line 358): Related Work continued
- Page 9 (line 401): Descriptions of Challenge Domains
- Page 10 (line 441): Results → Algorithm Summary
- Page 11 (line 481): Results continued with exploitability metrics
- Page 12 (line 526): Performance across chess and Go domains
- Page 13 (line 569): Discussion section covering poker and Scotland Yard outcomes
- Page 14 (line 608): Methods section detailing the CVPN approach and growing-tree CFR search
- Page 15 (line 654): Search mechanics including expansion phases and parameter tuning
- Page 16 (line 700): Modified continual re-solving strategy
- Page 17 (line 751): Theoretical guarantees and main theorems
- Page 18 (line 791): Data generation through self-play and training procedures
- Page 19 (line 830): Training continuation with recursive query handling and consistency checks
- Page 20 (line 870): References section
- Pages 20-27: Full reference list
- Page 27 (line 1124): Supplementary materials section begins
- Pages 28-35: Figures and tables including belief state visualizations, CFR depth analysis, exploitability curves, Elo ratings, scalability plots, poker and Scotland Yard results, CVPN architecture diagram, and GT algorithm illustration
- Page 36 (line 1368): Figure 8 showing the SOG training process
- Page 37 (line 1382): Supplementary text transitions into network architecture details
- Pages 38-41: Algorithm pseudocode for GT-CFR and sound self-play approaches with implementation notes
- Page 42 (line 1556): Poker betting abstraction and evaluation methodology
- Page 43 (line 1592): RL and search experiments with IS-MCTS results
- Page 44 (line 1657): Mathematical proofs begin, starting with value functions for subgames
- Page 45 (line 1700): Growing trees proof
- Pages 46-52: Extended proof section covering lemmas and theorems
- Page 53 (line 2325): Continual re-solving proof
- Pages 54-57: Supplementary figures including FOSG example, public tree visualization, Scotland Yard map, and RL exploitability curves
- Pages 58-59: Tables documenting neural network architectures and hyperparameters across different games
- Pages 60-63: Additional supplementary tables with full Elo rankings, IS-MCTS parameters, and RL settings

Now I'm identifying the key concepts to reference for the rule: the SOG(s,c) notation, how GT-CFR applies CFR+ internally, the CVPN definition, the pseudocode sections, and the critical figures and tables that define the approach. I'll create a file-specific rule anchored to PlayerOfGames.pdf.

Created the rule at `.cursor/rules/player-of-games-paper-index.mdc`. It includes:

- **Full section-by-section index** with page numbers, rough page positions (top/middle/bottom), and line numbers for coding agents
- **Key figures and tables** with page locations and what each shows
- **Specific Findings** section with the things we dug up in this conversation -- the SOG(s,c) notation, the CFR+ internals, c=0.01 for poker, the vector-valued CVPN, etc.
- **Extension note** at the bottom inviting future updates as more content is located

The rule is scoped to `PlayerOfGames.pdf` via the glob, so it'll activate when the paper is referenced.

---
