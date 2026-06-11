# Search–NN Interface: How GT-CFR Uses the Network at Runtime

**Status:** Living reference / ground truth for the runtime interaction between the GT-CFR search and the CVPN neural network.
**Audience:** The maintainer and any future coding agent working in this repo.
**Scope:** When the NN is called during search, what it sees, what it returns, how results are cached and consumed, and how this scales from Kuhn poker to Pokémon VGC. Bridges the gap between the algorithmic theory (`docs/gt-cfr-theory.md`) and the encoding/architecture design (`docs/state-encoding.md`). Covers the composite-private-state problem, the three-tier computation split, MCCFR deal sampling, and chance-node bucketing — all from the perspective of "what computation happens at runtime."

---

## 0. How to use this document

This is the distilled, organized version of a design discussion about how the search loop calls the NN, what scales and what doesn't, and how Pokémon's game structure creates challenges that Kuhn poker (and even poker in general) does not face. It is meant to be read top-to-bottom once, then used as a lookup reference. Section 3 (the three node types) is the fastest way to re-anchor "when does the NN get called?"; Section 7 (three-tier split) is the fastest way to re-anchor the proposed Phase 4 architecture.

**Related artifacts (do not duplicate — read these for their domains):**

- `docs/gt-cfr-theory.md` — the algorithmic theory (CFR, GT-CFR, two-loop model, training targets). **Read it first**; this document assumes familiarity with info sets, CFVs, regret matching, and the ephemeral regret table.
- `docs/state-encoding.md` — token anatomy, Transformer architecture, belief-weighted candidates, Phase 1 vs Phase 4 value-head designs. **Read it second**; this document assumes familiarity with CLS tokens, candidate tokens, and the School A vs School B distinction.
- `.cursor/rules/agent/overview.mdc` — milestones, action space, system architecture.
- `toy_examples/kuhn_poker/gt_cfr_search.py` — the working Kuhn implementation of GT-CFR search. Referenced throughout as the concrete example.
- `toy_examples/kuhn_poker/network.py` — the Kuhn CVPN. Illustrates the "public-state-in, per-card-vector-out" pattern (which differs from the Pokémon design; see Section 4).

---

## 1. The search–NN boundary

The neural network appears at exactly **two points** in the GT-CFR search:

1. **Node expansion** — when the search grows the tree by one node, it calls the NN to get a policy prior (for PUCT guidance) and value estimates (for leaf evaluation). These are **cached** on the node.
2. **Leaf evaluation during CFR+ traversal** — when a traversal reaches an unexpanded frontier node, it reads the **cached** NN values. No new forward pass.

Everything else in the search — regret updates, strategy derivation, value backup through internal nodes — is pure arithmetic over cached values and the ephemeral regret table. The NN is never called during a CFR+ iteration on already-expanded nodes.

This means:

$$\text{NN forward passes per search} = \text{number of expanded tree nodes}$$

Not "number of info sets," not "number of CFR iterations," not "number of sampled deals." The CFR+ iterations are the cheap part; the expansions are where NN cost lives.

---

## 2. The two interleaved phases (recap from gt-cfr-theory.md §10.2)

Per inner-loop iteration on the growing tree:

- **CFR+ regret-update phase:** traverse the current tree; at each info set compute counterfactual values for each action; at **frontier leaves** substitute the cached CVPN estimates; update cumulative regrets (regret-matching+) and accumulate $\bar\sigma$.
- **PUCT expansion phase** (every `expansion_interval` iterations): walk down the tree via PUCT scores and expand one frontier node, calling the CVPN once for the new node's values and priors.

The tree grows incrementally: frontier leaves become internal nodes as they are expanded, and the NN's estimates are progressively replaced by real computation grounded in deeper nodes and terminals. This is the correction mechanism from `gt-cfr-theory.md` §11 — the NN provides a fast prior, expansion + terminals + CFR improvement correct it.

---

## 3. The three node types during a CFR+ traversal

Every node the traversal visits falls into one of three categories, each with a different value source:

| Node type | Value source | NN call? | Kuhn code path (`gt_cfr_search.py`) |
|---|---|---|---|
| **Terminal** | Real game payoff | No | `is_terminal(history)` → `terminal_utility(...)` |
| **Frontier leaf** (unexpanded, non-terminal) | Cached CVPN value estimate | No (cached from expansion) | `not node.expanded` → `node.nn_values[card]` |
| **Internal** (expanded, non-terminal) | Recurse into children, weight by current strategy, back up | No | `for action in actions: child_val = _cfr_traverse(...)` |

At internal nodes, the counterfactual value flows **up** from children. The current strategy (derived from the regret table) determines the weighting. Regrets are computed from the difference between per-action values and the strategy-weighted node value. No network is involved — only the ephemeral tables.

As the tree grows, the boundary between "frontier leaf" and "internal" shifts deeper. More of the value computation is grounded in real game mechanics (terminals) rather than NN estimates. This is how the search self-corrects even when the NN is wrong.

---

## 4. What the NN sees: Kuhn vs Pokémon

The Kuhn toy and the Pokémon system handle private information differently in the network input. Understanding this difference prevents a common confusion.

### 4.1 Kuhn: public-state input, per-card vector output

The Kuhn CVPN takes **only public information** as input (action history + acting player + uniform belief) and outputs a **vector** — one CFV and one policy per possible private card:

- Input: `[history one-hots (12) | acting player (2) | belief (3)]` = 17 dims. No card identity.
- Output: `policy_logits [3, 2]` (3 cards × 2 actions) and `values [3]` (one per card).

The network simultaneously answers "if I hold J / Q / K, what should I do and what's my value?" The private card is an **output dimension**, not an input.

This works because Kuhn's private-state space is tiny (3 cards). The network has 3+6 = 9 output dimensions to cover all info sets at one public state.

### 4.2 Pokémon: private-state input, per-opponent-candidate vector output

For Pokémon, the acting player's private state (their full team) is too complex to enumerate as output dimensions. Instead, it is part of the **input**:

| Input component | What it is | Kuhn equivalent |
|---|---|---|
| CLS + field tokens | Global/public state | Action history + acting player |
| Your 6 Pokémon tokens (belief weight = 1.0) | Your private state (known to you) | *Not in input* (card is an output dim) |
| Opponent candidate tokens (belief weight < 1.0) | Belief over opponent's hidden state | Uniform belief vector |

The value head outputs one CFV per **opponent candidate**, not per "your possible private state." Your private state is fixed and known — the uncertainty is about the opponent.

### 4.3 Consequence for the Kuhn toy

The Kuhn design is a valid shortcut for the toy but does **not** transfer to Pokémon. If the Kuhn toy were redesigned to mirror the Pokémon architecture more closely, the acting player's card would be in the input and the value head would output CFVs over the **opponent's** possible cards (2 remaining). One forward pass per (public state, your card) pair, not one per public state covering all cards.

---

## 5. Deal sampling: why CFR+ traversals need concrete worlds

A common confusion: "If the NN takes belief-weighted candidates and outputs per-candidate CFVs, why do we need to sample specific opponent configurations ('deals') at all? Can't we just do CFR+ directly with the beliefs?"

**No.** The per-candidate NN output makes **leaf evaluation** efficient (one forward pass covers all candidates), but the **CFR+ traversal through the tree interior** must follow one concrete opponent configuration at a time. Here is why.

### 5.1 Game mechanics are concrete

At an internal node, after both players choose actions, a chance node resolves the outcome. "Did Heat Wave KO the opponent?" depends on which specific opponent is there — Scarf Basculegion has different defensive stats than Sash Basculegion. The traversal must commit to one concrete target to resolve mechanics and continue down the tree.

### 5.2 Child values are world-specific

The counterfactual value of action $a$ at info set $I$ is:

$$v_i(I, a) = \sum_{h \in I} \pi_{-i}^\sigma(h) \cdot u_i^\sigma(h \cdot a)$$

Each $h \in I$ is a different history (different opponent private state). The continuation value $u_i^\sigma(h \cdot a)$ differs across histories because the game plays out differently against different opponents. Computing this sum requires evaluating each term — each "deal" — individually.

### 5.3 This is NOT strategy fusion

The per-deal traversals compute per-deal **values**, but the **strategy** is shared across all deals at each info set. Multiple deals contribute (weighted by counterfactual reach) to the **same** cumulative regret table. Regret matching produces one strategy per info set that balances all worlds. The strategy never conditions on hidden information.

Strategy fusion happens when the search makes **different action choices** in different worlds (exploiting information it shouldn't have). GT-CFR's regret table is indexed by info set, not by deal, so this cannot happen.

### 5.4 MCCFR: sampling instead of enumerating

Full CFR enumerates all consistent deals per iteration (the Kuhn toy does this — 6 permutations). For Pokémon, the deal space is combinatorially large (all combinations of opponent candidate sets across slots). MCCFR replaces the full sum with a **sample**:

- Draw a joint opponent configuration from the belief distribution.
- Run one (or a few) CFR traversals on that concrete world.
- Accumulate regrets into the shared per-info-set table.
- Repeat with different samples across iterations.

The regret estimate is unbiased (in expectation it equals the full sum), just noisier. Sampling is biased toward high-probability deals (proportional to belief weight), which reduces variance because high-weight worlds dominate the sum anyway. Importance weights correct for non-uniform sampling if needed.

### 5.5 What "sampling a deal" means for Pokémon

A "deal" is one specific assignment of the opponent's hidden information:

- Which candidate set each opponent Pokémon is running (Scarf Basculegion vs Sash Basculegion).
- Which 4 of their 6 Pokémon they brought (if not all revealed yet).

Your own private state is known to you and fixed in the input — it is not part of the deal. The deal is sampled from the **joint** belief distribution (respecting item clause, team-building correlations, and all observations so far).

---

## 6. The composite-private-state problem

Player of Games (Schmid et al., 2021) was applied to games where each player's private state is one atomic object (e.g., hole cards in poker). Pokémon has **composite** private states: the opponent's team is 4–6 slots, each with hidden information, and those slots are **correlated** (item clause, team-building patterns). This creates a challenge that goes beyond what PoG addressed.

### 6.1 Cross-slot correlations in beliefs

If Basculegion and Garchomp are both on the opponent's team and both commonly carry Choice Scarf, item clause dictates that at most one can. The belief model must produce **joint** configurations, not independent per-slot samples:

$$P(\text{full team config} \mid \text{observations}) \neq \prod_{\text{slot } s} P(\text{slot } s \text{ config} \mid \text{observations})$$

The conditional sampler (see `state-encoding.md` §7.4, §9) handles this: it draws from the joint distribution over real tournament teams, which inherently respects item clause and captures team-building correlations. This can be factored as a chain of conditionals: $P(\text{slot 1}) \times P(\text{slot 2} \mid \text{slot 1}) \times \ldots$ if full joint enumeration is too expensive.

### 6.2 Cross-slot correlations in value

Even if the belief model correctly samples joint configs, the NN's value head must evaluate them. If the value head outputs one scalar per **candidate token** (per slot), it cannot directly express "the joint value of Scarf Basculegion + Lum Garchomp as a combination." Per-slot values miss interactions between opponent slots.

This is the hardest open question from `state-encoding.md` §12.3. Three approaches:

**Option A — Per-slot independence (approximate).** Treat each candidate's CFV as independent. Rely on the Transformer's attention to partially capture cross-slot interactions within each candidate's embedding (each Basculegion candidate attends to all Garchomp candidates during the forward pass). The approximation error is corrected over training as the NN adjusts per-slot values to account for typical cross-slot pairings.

**Option B — Joint-team candidates (correct, expensive).** Enumerate a support of $M$ full opponent team configurations. Each is a composite token; the value head outputs one CFV per joint config. Preserves all correlations but loses the compositional per-slot attention structure and scales as $O(M)$ rather than $O(\sum_s K_s)$.

**Option C — Split backbone/head architecture (proposed).** Use per-slot candidate tokens in the Transformer (preserving compositional attention), but at the value head, cross-reference the specific deal being evaluated. See Section 7.

---

## 7. The three-tier computation split (proposed Phase 4 architecture)

This is the proposed resolution of the composite-private-state problem. The computation splits into three tiers with different costs and call frequencies.

### Tier 1: Transformer backbone (expensive, run ONCE per node expansion)

Process all tokens through attention: CLS, field, your 6 Pokémon, and **all** opponent candidate tokens across all slots. If each of 4 opponent slots has $K = 5$ candidates, that is $8 + 6 + 20 = 34$ tokens. The $O(n^2)$ attention computation produces a contextualized embedding for every token.

After this pass, each candidate token's embedding has "seen" every other candidate through attention. The Scarf Basculegion embedding knows about the Lum Garchomp candidate (and vice versa). All cross-slot reasoning happens here, in the attention layers.

### Tier 2: Policy head (cheap, run ONCE per expansion)

Read from CLS, output one action distribution. The policy does not vary per deal — the agent plays one strategy across all possible opponent worlds (this is the whole point of info-set-level reasoning). CLS has attended to all candidates and formed a belief-weighted summary, which is the correct input for a single robust policy.

### Tier 3: Value head (cheap, run ONCE PER SAMPLED DEAL)

For each MCCFR-sampled joint config, **select** the relevant candidate embeddings (one per opponent slot) from the cached Transformer output, concatenate them, and pass through a small MLP:

```
Transformer output (cached): all_embeddings [n_tokens, d_model]

For deal d = (slot1=candA, slot2=candC, slot3=candE, slot4=candG):
  selected = concat(
      all_embeddings[idx_candA],    # active slot 1's selected candidate
      all_embeddings[idx_candC],    # active slot 2's selected candidate
      all_embeddings[idx_candE],    # back-row slot 3's selected candidate
      all_embeddings[idx_candG],    # back-row slot 4's selected candidate
  )   # shape: [4 * d_model]

  cfv = value_head_mlp(selected)   # [4 * d_model] → scalar
```

The value head MLP is tiny compared to the Transformer. With $d_{\text{model}} = 128$ and 4 opponent slots, the input is 512 dims. A 512 → 256 → 1 MLP is negligible per call. Multiple deals can be **batched**: stack $D$ selected-embedding vectors into a $[D, 4 \cdot d_{\text{model}}]$ tensor and run one batched forward pass.

### 7.1 Why all 4 opponent slots, not just the 2 active

The back-row matters for strategic evaluation. Knowing the opponent has Mega Gengar in the back changes the value of committing your Fake Out user now. The value head must see the full team configuration to assess the position correctly.

### 7.2 Cost analysis

| Operation | Count per search | Cost each | Total |
|---|---|---|---|
| Transformer forward pass | ~20 (one per expansion) | ~1–3 ms | 20–60 ms |
| Policy head (from CLS) | ~20 | ~0.01 ms | negligible |
| Value head (per deal, batched) | ~20 nodes × ~15 deals | ~0.01 ms | negligible |

The Transformer dominates. The per-deal value head calls do not multiply the expensive part. This is the core advantage of the split: the expensive backbone runs once per node, and the cheap head runs per deal.

### 7.3 Training targets

During training, each search emits tuples of:

$$\big(\;\text{all token embeddings (from backbone)},\;\; \text{deal } d,\;\; v^{\text{search}}(d)\;\big)$$

The value head learns:

$$\hat{v}(\text{concat of selected candidate embeddings for deal } d) \approx v^{\text{search}}(d)$$

Over training, the Transformer learns to produce embeddings from which the value head can accurately assess joint configurations.

---

## 8. Chance nodes and caching limits

In Kuhn poker, the game tree after the deal is **deterministic**: bet → call always leads to the same showdown. Once a node is expanded and its NN values cached, every CFR traversal reuses the cache.

In Pokémon, chance nodes (damage rolls, accuracy, crits, secondaries, multi-hit) produce **different output states** from the same action sequence. The same "Heat Wave into Garchomp" plays out differently depending on the roll. This limits the value of caching: each chance outcome leads to a distinct child state that needs its own evaluation.

### 8.1 Mitigations

**Chance bucketing / discretization.** Do not enumerate the full damage distribution. Bucket outcomes into a small number of representative results:

| Bucket | Example | Probability (approx.) |
|---|---|---|
| Low roll, no crit | 15th-percentile damage | ~40% |
| High roll, no crit | 85th-percentile damage | ~40% |
| Crit | Representative crit damage | ~6% |
| Miss | Zero damage | ~5–15% (move-dependent) |

HP differences within a non-crit damage range rarely change the strategic decision (68% HP vs 65% HP almost never flips what you should do). Aggressive bucketing is sound. DeepStack handled the poker flop (48 possible cards) by bucketing into ~3–5 strategically-distinct groups.

**Selective expansion.** PUCT expansion follows the most promising path. Most chance branches remain unexpanded frontier leaves evaluated by the NN. Only the high-impact outcomes (KO vs survive, status landed vs missed) get expanded.

**Shallow search.** Pokémon games are 6–12 turns. The search looks 1–3 turns ahead; the NN fills in the rest. The tree is wide but not deep, bounding the total number of nodes.

### 8.2 Estimated NN call budget

With chance bucketing (~4 outcomes per chance node), ~20 PUCT expansions per search, and 1–3 turns of depth:

- ~20–50 NN forward passes per search (Transformer, Tier 1).
- At ~1–5 ms per pass (small Transformer on GPU): 20–250 ms per decision.
- Within a 60-second turn timer, this is feasible with margin for the CFR+ iterations (which are the cheap arithmetic part).

---

## 9. Summary: what scales, what doesn't, and how to fix it

| Aspect | Kuhn | Pokémon (naive) | Pokémon (with mitigations) |
|---|---|---|---|
| **NN calls per search** | ~7 (full tree) | Explosive (wide + deep + stochastic) | ~20–50 (PUCT expansion + chance bucketing) |
| **CFR+ traversals per iteration** | 6 deals × 2 players = 12 | $K^S$ joint configs × 2 = huge | ~15 MCCFR samples × 2 = 30 |
| **Per-traversal NN cost** | 0 (cached) | 0 (cached, same as Kuhn) | 0 (cached) |
| **Value head per-deal cost** | N/A (per-card output dims) | Per-slot approximation (Option A) | Per-deal MLP from cached embeddings (Option C) |
| **Cross-slot correlations in beliefs** | N/A (one card each) | Must respect item clause, team-building | Joint sampling from conditional model |
| **Cross-slot correlations in values** | N/A | Missed by per-slot CFVs | Captured by attention + joint value head (Option C) |
| **Chance branching** | None (deterministic after deal) | Near-continuous damage/accuracy/crit | Bucketed into ~4 representative outcomes |

---

## 10. Open questions specific to this interface

1. **Value-head combination function.** In the three-tier split (Section 7), the value head concatenates selected candidate embeddings. Alternatives: pool (mean/max), cross-attend, or use a small Transformer over just the 4 selected embeddings. The best approach is an empirical question.
2. **MCCFR sample count.** How many deals per CFR iteration? Too few = high variance in regret estimates. Too many = wasted compute. DeepStack used ~1000 rollouts for poker; Pokémon's shorter horizon may allow fewer. Needs tuning.
3. **Chance bucket granularity.** How many buckets per chance node? More = more accurate but more tree nodes. Domain knowledge helps: KO thresholds and status effects are the strategically-critical boundaries.
4. **Interaction with the training loop.** The three-tier split changes the training target from "one CFV per candidate slot" to "one CFV per joint config." The replay buffer must store enough information to reconstruct which candidate embeddings to select for each deal. This may require storing the full Transformer output per training sample, which is more expensive than storing per-slot CFVs.

---

*If you are a future agent picking up implementation: this document covers only the runtime search–NN interface. For the algorithmic theory, read `docs/gt-cfr-theory.md`. For token design and architecture, read `docs/state-encoding.md`. For milestones and tooling, see `research/notes.md` and `.cursor/rules/agent/overview.mdc`.*
