Great progression. Let me address each in order.

## Why Counterfactual Reach Even When You Have Utilities

> You say that counterfactual regret "This decouples "how good is the action here" from "how often do I reach here."" -- but doesn't the utility function just do this regardless? The utility function (in your example, value = 0.5) shows how great it is. So to answer the question "how good is the action here", why not just use the utility value? Why do you have to weight it by the probability of getting here, if you're only trying to answer "how good is the action here"?

Sharp question, and the answer reveals a key subtlety.

The utility `u_i(h · a, σ)` gives the expected payoff **for a specific history `h`**. But an info set `I` contains **multiple histories** that the player can't distinguish. Different histories within `I` typically have different utilities for the same action `a` — because the opponent's hidden information differs, the action's consequences differ.

To get a single value `v_i(I, a)` from many `u_i(h · a, σ)` values, you have to **aggregate across the histories in `I`**. The natural aggregation is a weighted average. The weighting is counterfactual reach.

**Concrete contrast**. Consider a hypothetical poker info set: "P1 has K, P1 passed, P2 bet" (`I1_K_pb`). Two histories:

- `h_1`: P2 has J. Took action "bet at `I2_J_p`."
- `h_2`: P2 has Q. Took action "bet at `I2_Q_p`."

Suppose opponent's current strategy is: at `I2_J_p` bet with probability 0.05 (rare bluff), at `I2_Q_p` bet with probability 0.6 (frequent value bet).

Then `π_{-1}(h_1) = (1/6) · 0.05`. Pretty small.
And `π_{-1}(h_2) = (1/6) · 0.6`. Much larger.

If P1 calls:
- `u_1(h_1 · call) = +2` (K beats J)
- `u_1(h_2 · call) = +2` (K beats Q)

Same in this case. But if P1 had Q instead of K, the utilities would differ across histories (Q beats J but loses to opponent's K-bluff if they had K — though Kuhn doesn't allow that exact case; imagine a richer game).

The point: **the value of the action at the info set depends on which history is more likely**. Counterfactual reach probabilities tell us this. Without weighting, we'd be implicitly assuming all histories equally likely, which is wrong whenever opponent's strategy differs across their own info sets.

So utilities give you the value *given a specific world*. Counterfactual reach gives you the *probability of being in each world, given everyone's strategies*. Combining them gives the value of the action at the info set, which is what regret matching needs.

The decoupling-from-`π_i(h)` part is separate — that's why we use *counterfactual* reach instead of full reach. Weighting is needed regardless; the choice of which probabilities to factor out is what makes it counterfactual.

## Does AlphaZero Need MCTS at Play Time?

> When AlphaZero is done training and plays against an opponent, does it need to do MCTS live? Or is the repeated self-play cycle sufficient to train the NN up so that when you play a real game, you can just use the NN to make moves? After all, the NN is trained to mimic the previous batch of MCTS moves, so over 5000 batches or so, there's minimal difference to playing against NN vs NN+MCTS, right?

Yes, absolutely. The MCTS is not just a training crutch — it adds significant strength at play time. Your intuition that "NN converges to mimic MCTS" is partially right but misses an important point.

The ELO numbers from the AlphaZero paper:

- Network alone (single forward pass, take argmax): around 2000 ELO in chess. Strong club player level.
- Network + MCTS (800 sims/move): around 3500+ ELO. Superhuman.

That's roughly a 1500 ELO gap. Far from "minimal difference."

Why doesn't the gap close as the network improves?

**Reason 1: MCTS is a policy improvement operator on top of *whatever* policy you give it.** If the network's policy is `π`, MCTS produces a refined policy `π_MCTS` that's strictly better on average. This holds for any policy — even an already-good one. Better network → better MCTS, but the gap between the two stays roughly constant in relative terms.

**Reason 2: Networks are finite-capacity function approximators with inherent bias.** The network compresses the entire game's value function into a fixed number of parameters. There's information loss. MCTS recovers some of this loss by doing concrete search instead of pattern-matching.

**Reason 3: The training objective only matches the MCTS distribution from training-time states.** At test time, you might encounter states the network underrepresented or got wrong. MCTS does fresh search and corrects.

**Reason 4: Search depth is fundamentally important in tactically rich games.** A position with a forced 6-ply combination is hard to evaluate without searching it. The network might gist the position but miss the tactic. MCTS finds it.

For your project, you should plan to run search at play time, not just deploy the network alone. The compute cost is real but the strength benefit is enormous.

The same logic applies to GT-CFR for Pokemon: the CVPN trained via self-play won't be enough to play strongly on its own. You'd run GT-CFR live during each turn of the actual game, using the CVPN at search-tree leaves.

## CFR With Self-Play and Without Full Game Tree

> I also think I understand how/why CFR works while computing reach probabilities from an omniscient perspective. It's trained offline and it sees every info set and the whole game tree, so it has no need to consider a player's real perspective. How does this change when we can't see the full game tree and it's not trained offline, but rather through self-play? Again, I think I have a good grasp on vanilla CFR for now, but I can see some clear places where it doesn't translate 1:1 to a more complex game

You correctly identified the key shift. Vanilla CFR is omniscient over the entire game tree at training time. For large games, you can't do that. Two adaptations:

**Sampling (MCCFR family)**: Instead of traversing the full tree per iteration, traverse a *sampled path* per iteration. The information sets along the sampled path get full updates; others get nothing this iteration. Over many iterations and many samples, every info set gets updated in expectation. Convergence guarantees survive (with worse rates).

**Function approximation (Deep CFR family)**: Replace the explicit regret table (one entry per info set) with a neural network. The network takes info-set features as input and outputs predicted regrets. New info sets get reasonable regret estimates via generalization. Training the network requires generating regret targets — done via sampled tree traversals using current strategies.

**Combined with self-play**: in Player of Games / GT-CFR, the system runs in two intertwined modes:

1. **Inference-time search**: when an actual decision is needed (e.g., it's your turn in a real game), GT-CFR runs from the current public state. It builds a search tree, expands it on the fly, queries the CVPN at leaves, and outputs an average strategy. The agent samples an action and plays.

2. **Self-play training**: agents using mode 1 play full games against each other. Game trajectories are stored. The CVPN is trained on data from these games to predict what GT-CFR's longer searches would conclude.

The self-play data feeds the CVPN, the CVPN powers the search, the search produces good play, the good play generates better data. Same virtuous cycle as AlphaZero, with CFR replacing MCTS as the search algorithm.

The omniscience question: at *each search invocation*, the algorithm is omniscient over its small search tree (it knows the structure of the tree it built and can compute counterfactual values within it). But it doesn't have the full game tree. The CVPN approximates everything outside the search tree.

## GT-CFR Operationally on a Pokemon Turn

> I think I understand vanilla, offline CFR now. Help me to take the next step in understanding GT-CFR -- how it is similar and different to CFR, and examples of how it would play out in Kuhn poker or in my pokemon example. Give a thorough description here.

Concrete setup. Imagine turn 3 of a doubles battle. The state:

- **Your team (full info)**: Garchomp + Rotom-Wash active, Dragonite + Tyranitar in back. You know all moves, items, stat point spreads.
- **Your private info**: items on your back-row Pokemon, full movesets of unrevealed Pokemon, etc.
- **Public info about opponent**: revealed Charizard (used Will-O-Wisp turn 1, used Heat Wave turn 2) and Whimsicott (used Tailwind turn 1, used Moonblast turn 2). Charizard's item not revealed (could be Charizardite Y, Charizardite X, or other). Their back row not revealed.
- **Field state**: Tailwind on opponent's side, 3 turns remaining. No weather.
- **HP/status**: Charizard 60% HP no status, Whimsicott 80% HP, Garchomp 100%, Rotom-Wash 70% HP.

This is your **public state**, which combined with your private state defines your **info set**.

### Search Tree Construction

GT-CFR builds a tree rooted at this info set. Initially, it's just the root.

The tree has nodes of several types:

1. **Decision nodes for you**: where you choose your joint action (move and target for both Garchomp and Rotom-Wash, or switch).
2. **Decision nodes for opponent**: where they choose their joint action. Critically, these are *info-set nodes*, not history-specific nodes — they represent the opponent's decision under their own uncertainty.
3. **Chance nodes**: where stochastic resolution happens (damage rolls, accuracy, secondary effects).
4. **Leaf nodes**: where the CVPN is queried for counterfactual values and policy.

### Initial CVPN Query at the Root

The root is expanded immediately: query the CVPN with the public belief state.

- **Public belief state input**: a tensor encoding everything publicly observable (revealed Pokemon, revealed moves, HP, status, field, etc.) plus your private info (since you can act based on it).
- **CVPN outputs**:
 - Counterfactual values for both players at this info set (estimates of how each player is doing in the search).
 - Policy distribution: for both your action space and the opponent's action space, prior probabilities over the joint actions.

The CVPN's policy outputs become the priors used by PUCT during tree expansion. The CVPN's counterfactual value estimates seed the regret minimization.

### Iteration Loop

GT-CFR runs for a budget — say, 1000 iterations or 30 seconds wall-clock. Each iteration has two phases.

**Phase A — Tree Expansion (PUCT walk)**

1. Start at root. Walk down via PUCT-like selection:
 - At your decision nodes: select action to expand based on `prior + PUCT bonus + Q from current regrets`.
 - At opponent decision nodes: same logic, using opponent's policy / regrets.
 - At chance nodes: sample one outcome from the chance distribution.
2. Stop when you reach an unexpanded node (a "leaf" of the current tree).
3. Expand the leaf: add `k` children based on top-k actions from priors. Or expand with all actions, depending on implementation.
4. Query the CVPN at each new node to get its counterfactual value estimate and policy priors.
5. Done with expansion phase.

The expansion grows the tree. After 1000 iterations, the tree has thousands of nodes, with deep expansion along promising branches and shallow expansion along unpromising ones.

**Phase B — CFR+ Update on Current Tree**

1. Traverse the current tree (post-expansion).
2. **Bottom-up**: at each node, compute counterfactual values for each action under the current strategy. At leaves, use the CVPN's stored value estimate. At internal nodes, recursively combine child values weighted by current strategy and chance probabilities.
3. **At each info-set node**: compute instantaneous regret for each action. Update cumulative regret with regret-matching+ (clip to non-negative).
4. Derive next iteration's strategy at each info-set node via regret matching.
5. Update average strategy table (the cumulative strategy across iterations, weighted by iteration number for CFR+'s linear weighting).

The strategy at every info-set node in the tree gets updated each iteration. The strategy at the root is what we care about ultimately.

### After All Iterations: Extract the Strategy

After the budget is spent (say 1000 iterations), look at the root info set's **average strategy**. This is a probability distribution over your joint actions.

For example, the output might be:

| Joint action (your Garchomp / your Rotom-Wash) | Probability |
|---|---|
| Earthquake (allies safe via Levitate) / Hydro Pump on Charizard | 0.42 |
| Dragon Claw on Charizard / Volt Switch on Charizard | 0.21 |
| Switch Garchomp to Dragonite / Hydro Pump on Charizard | 0.18 |
| Protect / Hydro Pump on Charizard | 0.10 |
| ... | ... |

This is a **mixed strategy**, not a single deterministic move. This is correct for imperfect-info games — you sometimes need to randomize to be unexploitable.

You sample a joint action from this distribution and play it in the actual game.

### Crucial Detail: Simultaneous Moves and Info Sets

In Pokemon, both players choose actions simultaneously. The search tree has to model this without giving either player access to the other's choice.

Approach: at the top of each turn in the search tree, your decision node is followed by the opponent's decision node, but the opponent's decision is at an *info set* that explicitly does **not** see your action. Concretely:

- Root → your decision node → 100 child nodes (one per joint action you could take).
- Each of your child nodes → opponent's decision node → these all map to the *same opponent info set* `I_opp`.
- Opponent's strategy at `I_opp` is a single distribution over their joint actions, used identically across all your action choices.
- After both players' actions are committed, a chance node resolves any randomness (damage rolls, accuracy, etc.) → resulting state.

This is the standard way to model simultaneous moves in extensive-form games. The CFR+ updates correctly handle this because regret matching at `I_opp` produces a single strategy that doesn't depend on which of your actions came before.

Operationally, in code, you typically represent the joint outcome as a chance node sampling over `(your_action, opponent_action, stochastic_resolution)` triples.

### Crucial Detail: Hidden Information in the Tree

The opponent's info set `I_opp` doesn't know your hidden info either. The tree structure handles this by constructing `I_opp` as the bundle of histories consistent with opponent's public observations.

For example, at turn 3, the opponent has observed your Garchomp using Earthquake and Rotom-Wash using Hydro Pump (in turns 1-2). They have not observed your back-row Pokemon's identity. So their info set on turn 3 contains all histories consistent with "Garchomp + Rotom-Wash active, knows Earthquake + Hydro Pump, plus *any plausible back row*."

In the search tree, this info set bundles together all those histories. The opponent's strategy is one distribution that must work across the bundle. Strategy fusion is avoided because we never construct nodes that pretend the opponent knows your back row.

For tractability, GT-CFR doesn't enumerate every possible back-row pair (too combinatorial). Instead, it samples representative back-row possibilities weighted by your meta priors, or uses the CVPN to estimate over the distribution implicitly. This is where your meta_priors module connects.

### Crucial Detail: Stochastic Resolution

After your action and opponent's action are committed, a chance node resolves:

- Move accuracy rolls (e.g., Hydro Pump 80% accuracy)
- Damage rolls (15% variance, 16 possible values)
- Crits (4.17% base)
- Secondary effects (Hydro Pump 10% chance to lower opponent's accuracy? — actually that's not a thing for Hydro Pump but Scald 30% burn etc.)

Approach: sample one resolution per traversal. Different traversals sample different resolutions, building up a Monte Carlo estimate of the expected counterfactual value. Convergence is slower for high-variance chance nodes (more iterations needed before averages stabilize).

For very high-variance scenarios (like Charizard's coin-flip Mega), you might want to enumerate the few discrete branches and weight them, rather than sample.

### Putting It All Together: A Single Iteration's Walk

Concrete iteration `t = 47`. Tree currently has ~50 nodes. Walk:

1. **Root** (your decision): PUCT picks "Earthquake / Hydro Pump on Charizard" because it has high prior (CVPN said this looks promising) and decent Q (regrets accumulated from 46 prior iterations).
2. **Your child node** → opponent decision node `I_opp`: PUCT picks "Heat Wave / Moonblast" for opponent (high prior because it's the obvious Tailwind-aggressive line).
3. **Joint actions committed → chance node**: sample resolutions. Suppose: Hydro Pump hits, damage roll 87/100 (above average), no crit; Earthquake hits both Charizard and Whimsicott, both above-average rolls; Heat Wave hits both your Pokemon, average rolls; Moonblast hits Garchomp, average roll.
4. **Resulting state**: Charizard now at 5% HP, Whimsicott at 35%, your Garchomp at 50%, Rotom-Wash at 30%. Tailwind ticks down to 2 turns.
5. **This is a new state never reached before**: it's an **unexpanded leaf**. Stop here.
6. **Expand the leaf**: add child nodes for the next turn's options (your decisions + opponent decisions, simultaneous structure).
7. **Query CVPN**: get counterfactual value estimates and policy priors for the leaf.
8. **End of expansion phase.**

Now Phase B (CFR+ update) runs over the entire tree (not just the path):

9. Bottom-up traverse the tree. At every node, recompute counterfactual values under the new current strategies.
10. At every info-set node, update cumulative regret based on the difference between counterfactual values per action and the strategy-weighted average. Apply CFR+ clipping (no negative regrets).
11. Update strategy via regret matching+: actions with high cumulative regret get higher probability next iteration.
12. Update average strategy: accumulate iteration `t = 47`'s strategy weighted by `t = 47` (linear weighting).

This whole iteration takes one CVPN forward pass (at the new leaf) plus the tree-update arithmetic. After 1000 iterations, you've done 1000 CVPN calls and 1000 tree updates. The root's average strategy is the output.

### Comparison: GT-CFR vs MCTS at Play Time

| | MCTS (AlphaZero) | GT-CFR (Player of Games) |
|---|---|---|
| Tree nodes | states (perfect info) | info sets (imperfect info) |
| Action selection in tree | UCB/PUCT | PUCT for expansion, regret matching for strategy |
| Backup | NN value at leaf, average up the tree | CFR+ counterfactual value computation |
| Output at root | visit count distribution | average strategy from CFR+ |
| Handles hidden info | no (strategy fusion) | yes |
| Network role | value + policy at leaf | counterfactual value + policy at leaf |

GT-CFR is structurally more complex than MCTS but operates on the same self-play training paradigm. Most of your AlphaZero-style infrastructure (state encoding, NN backbone, training loop, simulator integration) carries over. The replacements are: search tree node type (info sets instead of states), search algorithm (CFR+ instead of UCB updates), network output (counterfactual values instead of single value).

### What This Means for Your Project Plan

Concrete recommendation, refining what I said before:

1. **Phase 1 (Perfect-info AlphaZero)**: Build everything as if both teams' full info were public. State = full Pokemon Showdown battle state with both teams revealed. Use vanilla AlphaZero MCTS+NN. Get end-to-end self-play training running. This validates your simulator integration, encoding, NN architecture, and training pipeline. Estimated milestones 4-5 of your roadmap.

2. **Phase 2 (Add stochastic resolution)**: Introduce chance nodes in the search tree for damage rolls, accuracy, secondary effects. Still perfect info but stochastic. This validates your chance-node handling.

3. **Phase 3 (Imperfect info via determinized MCTS as baseline)**: Hide the opponent's items, back row, unrevealed moves. Use determinized MCTS (sample opponent worlds via meta priors) as a first imperfect-info baseline. Acknowledge strategy fusion is happening; measure how exploitable the resulting agent is.

4. **Phase 4 (Replace with GT-CFR)**: Swap the search algorithm from determinized MCTS to GT-CFR. Replace the network output (single value) with counterfactual values. This is the largest single step but reuses most infrastructure. The result should be measurably less exploitable than Phase 3.

5. **Phase 5 (Refine)**: Better priors, larger networks, more compute, better state encoding, better belief tracking, etc.

Phase 1 is the hardest engineering phase because you're building everything from scratch. Phase 4 is the hardest algorithmic phase because GT-CFR is genuinely complex. Phases 2, 3, 5 are incremental refinements.

If you'd like, the next concrete step we could discuss is the **state encoding for Pokemon** — what features the network actually sees as input. That's the next major design question and it's load-bearing for everything else.