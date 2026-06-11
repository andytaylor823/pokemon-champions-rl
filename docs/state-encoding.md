# State Encoding & Neural Network Architecture

**Status:** Living reference / ground truth for how the battle state is represented and consumed by the neural network.
**Audience:** The maintainer and any future coding agent working in this repo.
**Scope:** How the game state is tokenized, what information each token carries, what architecture consumes it, and how the design evolves from Phase 1 (perfect info, scalar value) to Phase 4 (GT-CFR, belief-weighted candidates, per-candidate CFVs). Alternatives considered and rejected are documented with rationale to prevent re-derivation.

---

## 0. How to use this document

This is the distilled, organized version of an extended design discussion about state encoding and NN architecture. It is meant to be read top-to-bottom once, then used as a lookup reference. Section 4 (token anatomy) is the fastest way to re-anchor "what does a Pokemon token contain?" Section 11 (alternatives) is the place to check before proposing a different design.

**Related artifacts (do not duplicate — read these for their domains):**

- `docs/gt-cfr-theory.md` — the algorithmic theory (CFR, GT-CFR, value heads, training loop). **Read it first**; this document assumes familiarity with info sets, CFVs, and the two-loop model.
- `docs/search-nn-interface.md` — when the NN is called during search, caching, deal sampling, the composite-private-state problem (cross-slot correlations in beliefs and values), the three-tier computation split, and chance-node bucketing. **The runtime interface counterpart to this doc.**
- `.cursor/rules/agent/overview.mdc` — milestones, action space, imperfect-info inventory.
- `.cursor/rules/game-domain/overview.mdc` — Regulation M-A rules, stat system, legal pool sizes.
- `.cursor/rules/game-domain/stat-points.mdc` — Champions stat-point formula (replaces EVs; 66 total, 32 max per stat).
- `src/meta_priors/clustering.py` — current archetype clustering (to be replaced by a conditional sampler; see Section 9).
- `data/legal/` — legal lists: ~195 species, ~120 items, ~551 unique moves across all learnsets, median 61 moves per species.

---

## 1. What the encoding is for and who consumes it

The encoding is the **translation of everything strategically relevant into tensors**, because the neural network is the only thing that reads it (the battle simulator has its own internal state and receives a separate, slimmed-down action format).

The encoding appears at exactly two points in the system:

1. **Inside search, at every leaf evaluation** (the hot path — thousands of encode → NN calls per move).
2. **During training**, on stored encodings from the replay buffer.

```
Battle state (what the player may legally know)
    │
    ▼
ENCODER (the thing this document specifies)
    │
    ▼
Tensors: entity tokens + field token + action mask (+ belief, Phase 4)
    │
    ▼
Neural net: policy head + value head
    │
    ▼
Search: MCTS (pit-stop) / GT-CFR (north star)
    │       │
    │       └── runs at every leaf ──► back to ENCODER (thousands of times per move)
    ▼
Move played + training tuple (encoded state, search policy, search value)
    │
    ▼
Replay buffer (stores the encoded states)
    │
    ▼
Training: SAME encoder feeds the SAME NN (updated weights cycle back up)
```

**What must be present:** everything that could change the right decision, and nothing the player could not legally know. A feature omitted is a permanent blind spot; a feature included lets the net *potentially* learn to use it. The action mask (which actions are legal) is also part of the encoding.

---

## 2. The core principle: encode features, never identities

The network never sees the same battle state twice. Like AlphaZero in chess (~$10^{40}$ positions, trained on a vanishing fraction), it must **generalize** from features, not memorize states. This means:

- Encode a Pokémon by **what it is** (species, moves, item, stats, status, position), not by an opaque id ("set #37").
- A never-before-seen set — Sash Basculegion when training only included Scarf — is a new *point in feature space*, evaluable by composing "what Basculegion does" + "what Focus Sash does." Features generalize; ids cannot.
- The simulator computes all mechanical effects correctly regardless of training coverage. Only the net's *heuristic prior* can be off for a novel set, and search + future training correct it.

This is load-bearing for robustness against metagame drift: the value net evaluates from features and can interpolate; only the *belief model* (which assigns prior probabilities to candidates) can become stale, and that is a data-refresh problem, not an architecture problem.

---

## 3. The entity-centric representation

Pokémon battles are **entity-centric**: 6+6 Pokémon (each an entity with attributes) plus a small set of global/field facts. This is not a spatial grid (no translation invariance, so CNNs are a poor fit) and not a flat unstructured blob (so giant MLPs bury important signals). The natural representation is **a set of entity tokens + a field token**, consumed by an architecture that processes sets (a Transformer or a DeepSets-style shared encoder).

The fundamental encoding unit is a **token**: a fixed-length vector representing one entity (a Pokémon) or one global object (the field). All tokens share the same dimensionality $d_{\text{model}}$ after an initial projection, so they can be processed uniformly.

**Phase 1 (perfect info)** — 14 tokens per game state:

| Token type | Count | What it represents |
|---|---|---|
| CLS | 1 | Learnable summary; read by heads after attention |
| Field | 1 | Weather, terrain, Trick Room, gravity, turn counters |
| Your Pokémon | 6 | Concrete, fully known (active, back-row, or fainted) |
| Opponent Pokémon | 6 | Concrete, fully known (perfect info = everything revealed) |

**Phase 4 (GT-CFR, imperfect info)** — variable token count per game state:

| Token type | Count | What it represents |
|---|---|---|
| CLS | 1 | Same as Phase 1 |
| Field | 1 | Same as Phase 1 |
| Your Pokémon | 6 | Concrete, fully known (always) |
| Opponent candidates | variable | Multiple tokens **per opponent slot**, each a concrete candidate set with a belief weight (see Section 7) |

The backbone (projections + Transformer layers) is identical across phases; only the **heads** change (scalar value → per-candidate CFVs; see Section 6).

---

## 4. Token anatomy: what goes inside a single Pokémon token

Each Pokémon token is a **flat, fixed-length vector** encoding exactly what this (hypothetical or real) Pokémon is. Nothing probabilistic lives inside a token — every feature is a concrete value. Probability lives *between* tokens (as belief weights on candidate tokens; see Section 7).

### 4.1 Feature groups (all are concrete per-token)

| Group | Features | Notes |
|---|---|---|
| **Identity** | Species, ability | Categorical; one value each |
| **Moves** | Up to 4 move identifiers | Categorical; order within the 4 is arbitrary |
| **Item** | Held item | Categorical; one value. "No item" / "item consumed" are distinct |
| **Stats** | 6 final level-50 stat values (HP, Atk, Def, SpA, SpD, Spe) | Continuous; derived from base + stat points + nature via the Champions formula. The net sees *results*, not the allocation that produced them |
| **Nature** | Nature identifier | Categorical; largely redundant once final stats are present, but cheap to include |
| **Battle state** | Current HP %, status condition, stat stages (−6..+6 per stat), turns on field, Protect counter, volatiles (Encore/Taunt/Disable/etc. with their turn counters), Substitute HP, is-active, is-back-row, is-fainted, is-not-brought, has-Mega'd, item-consumed | Mixed continuous/categorical |
| **Derived features** | Effective Speed (after Tailwind/TR/paralysis/Scarf/stat stages), Choice-lock status + locked move (if inferred) | Continuous / categorical; hand-crafted to surface decision-critical info the net would otherwise need many examples to re-derive from raw inputs |
| **Side affiliation** | "My team" vs "opponent team" flag | Binary; replaces positional encoding (no slot-order meaning) |
| **Belief weight** | $p_i$ = probability this candidate is the real set | Float in $[0, 1]$; always $1.0$ for your own Pokémon and for all Pokémon in Phase 1. Only <1 for opponent candidate tokens in Phase 4 |

### 4.2 Tokenization of categorical features — open design question

Two approaches, both viable, decision deferred to implementation:

**Multi-hot / one-hot encoding.** Species is a one-hot over ~195 species; moves are a "four-hot" over ~551 moves (length 551, exactly 4 entries are 1); item is a one-hot over ~120 items; etc. Simple, interpretable, no information loss. Results in a wide but sparse per-token vector (roughly ~900 dims for identity+moves+item+ability before battle-state features). AlphaZero used this approach (119 binary planes for chess), and it works — but AlphaZero had effectively unlimited compute and a CNN whose weight-sharing exploited spatial structure.

**Learned embeddings.** Each categorical feature (species, each move, item, ability) gets an integer id mapped through a learned embedding table (e.g. `species_id → 32-dim`, `move_id → 16-dim`). The net learns that Fake Out and Quick Attack are both priority, that Charizardite Y and Charizardite X are related, etc. Dramatically smaller token width; better generalization across related features. Standard in NLP and entity-modeling. Less immediately human-readable, but the id↔name mapping is always available for inspection.

**Hybrid** is also possible: embed high-cardinality features (moves, species) and one-hot low-cardinality ones (nature, status). There is no principled way to choose without experimentation. **The token's overall shape and meaning are the same regardless of which tokenization is used** — only the width of the raw vector (before the `nn.Linear` projection into $d_{\text{model}}$) changes. This is a tuning decision, not an architectural one.

### 4.3 What is NOT inside a token

- **Probabilistic distributions** — a token never says "78% chance of Wave Crash." It says "this Pokémon has Wave Crash" or "this Pokémon does not have Wave Crash." Probability lives in the belief weight *on* the token.
- **Other Pokémon's info** — cross-entity reasoning (type matchups, speed comparisons, teammate synergy) is the Transformer's job, not the encoder's. Each token is self-contained.
- **Information the player could not legally know** — unrevealed opponent info is never in a "known" token; it is represented via the *set of candidate tokens* with belief weights.

---

## 5. Field, side, and global tokens

Not everything is per-Pokémon. Field/global state becomes its own token (or tokens), projected to the same $d_{\text{model}}$ width so it can participate in attention alongside Pokémon tokens.

### 5.1 Field token features

| Feature | Encoding | Notes |
|---|---|---|
| Weather (Sun/Rain/Sand/Snow/none) | Categorical + turns remaining (small one-hot, e.g. 0–8) | Duration is decision-critical (stalling out weather); encode as one-hot, not a raw float |
| Terrain (Electric/Grassy/Misty/Psychic/none) | Same as weather | |
| Trick Room active + turns remaining | Same pattern | Especially critical — determines speed order for the whole field |
| Gravity | Boolean | |
| Turn number | Normalized float or small one-hot | |
| Phase/decision type | Categorical: normal turn / forced switch / team preview | Determines action space structure |
| Whose decision | Binary: is it my turn to act? | For perspective canonicalization |

### 5.2 Per-side features

Some effects are per-side rather than global (Tailwind, Reflect, Light Screen, Aurora Veil, hazards, Mega-used flag). Two options:

- **Include in the field token** with a "side A" / "side B" prefix. Keeps the token count at 1 but makes the field token wider.
- **Two side tokens** (one per player), each containing that side's Tailwind/screens/hazards/Mega status. Adds 2 tokens but keeps each one cleaner.

Either works; minor design choice.

### 5.3 Why the field is a peer token, not a prefix

One alternative considered was prepending field info to every Pokémon token (so each token is "Pokémon features ⊕ field features"). This duplicates the field 12+ times, inflates every token, and makes the field update problem harder (change one field value → re-encode all tokens). Making the field its own token that every Pokémon **attends to** is cleaner: the field info reaches every Pokémon through one hop of attention, without duplication.

---

## 6. The CLS token and output heads

### 6.1 What CLS is

CLS ("classification") is one extra, **learnable** vector prepended to the token sequence. It carries no input data. Its job: soak up a summary of all real tokens via attention, providing a single vector for output heads. It is an alternative to mean-pooling (averaging all token outputs). CLS can learn *content-weighted* pooling — attending more to active threats, less to fainted Pokémon — whereas mean-pooling weights all tokens equally.

**Why random initialization.** CLS is an `nn.Parameter`, initialized randomly like any weight matrix. Training shapes it into a useful "summarize the battle" vector. The initial values are irrelevant; only the learned result matters.

**Why position 0.** A Transformer has no processing order along the token axis — all positions are recomputed simultaneously at every layer, and every position attends to every other. We read CLS from position 0 purely because that is where we inserted it (`torch.cat([cls, field, entities])`). If inserted last, we would read the last position. The index is bookkeeping, not a claim about information flow.

### 6.2 How CLS is used (it IS used — twice)

```
tokens = cat([cls, field, entities])     # CLS enters at position 0
h = transformer(tokens)                  # inside: CLS attends to all real tokens,
                                         #   accumulating a holistic summary layer by layer
pooled = h[:, 0]                         # read the EVOLVED CLS back out
policy = policy_head(pooled)             # action distribution
value  = value_head(pooled)              # scalar value (Phase 1)
```

The CLS vector at `h[:, 0]` is **not** the random init — it has been overwritten layer by layer with information from the entire battle.

### 6.3 Phase 1 vs Phase 4 output heads

| Phase | Policy head reads | Value head reads | Value output shape |
|---|---|---|---|
| **Phase 1** (perfect info) | CLS → action distribution | CLS → scalar value | `[batch, 1]` |
| **Phase 4** (GT-CFR) | CLS → action distribution | Each **candidate token** → scalar CFV | `[batch, n_candidates]` |

**Why the split in Phase 4.** Policy and value answer fundamentally different questions:

- **Policy** = "what should I do?" One answer (a distribution over actions), regardless of how many opponent candidates exist. The CLS summary — which has attended to *all* candidates and formed a blended view — is the right input. One policy, read from one token.
- **Value** = "how good is this position *conditional on each possible opponent private state*?" As many answers as there are candidates. Each candidate token, after attention, contains a contextualized representation of "what the battle looks like *if this candidate is real*" (because it attended to your team, the field, and the other candidates). The value head reads each one and emits a separate scalar CFV.

The **backbone** (projections + Transformer layers) is identical across phases. Only the heads change. This is by design: upgrade the heads, not the encoder.

---

## 7. Opponent uncertainty: concrete candidates with belief weights

This section is Phase 4 only. In Phase 1 (perfect info), every Pokémon is known; skip to Section 8.

### 7.1 Candidates, not distributions

The opponent's hidden state (unrevealed moves, item, ability, stat-point spread) is represented as a **set of concrete candidates per slot**, each with a belief weight. A candidate is a fully specified set — no internal probabilities.

Example for an opponent's Basculegion:

| Candidate token | Item | Moves | Spread | Belief weight |
|---|---|---|---|---|
| A | Choice Scarf | Wave Crash, Aqua Jet, Flip Turn, Protect | max Atk/Spe | 0.50 |
| B | Focus Sash | Liquidation, Aqua Jet, Flip Turn, Protect | max Atk/Spe | 0.35 |
| C | Mystic Water | Wave Crash, Liquidation, Flip Turn, Protect | bulkier | 0.15 |

Each row becomes one token (length $F$, same as any Pokémon token). The belief weight is one feature inside the token. Three tokens for one slot, differing in item/move/stat features and belief weight.

### 7.2 Joint information is preserved automatically

The earlier-considered **marginal encoding** (one column per move = P(knows it)) would capture "78% chance of Wave Crash, 60% chance of Liquidation" independently but lose the joint: "if Scarf then *always* Wave Crash, if Sash then *always* Liquidation." With concrete candidates, the joint is trivially preserved because each candidate *is* a joint — Scarf and Wave Crash appear in the same row; Sash and Liquidation appear in the same row. No marginals anywhere.

### 7.3 What "support" means

The **support** of a probability distribution is the set of outcomes with non-zero probability. "Live support" = the support right now, after conditioning on what has been observed. If the opponent's Basculegion has revealed Wave Crash, candidates without Wave Crash drop out — the support shrinks.

The live support determines **how many candidate tokens** are fed per slot and therefore the token count of the Transformer input (which varies per game state; see Section 8.3).

### 7.4 Where candidates come from (the conditional belief model)

Candidates are drawn from the **meta-priors conditional sampler**: a model that answers $P(\text{set} \mid \text{species, revealed info, teammates on the rest of the team})$, estimated from real tournament team data. This captures cross-slot correlations (Incineroar runs Protect much more when paired with Mega Gengar) because it conditions on the observed team composition, not just the species in isolation.

As observations accumulate mid-battle (moves used, items revealed, ability triggered), the sampler is re-queried with those constraints, producing a fresh top-$K$ support. A candidate absent from the pre-battle top-$K$ can enter the mid-battle support once observations make it plausible. See Section 9 for the meta-priors rewrite direction.

### 7.5 Generalization to unseen candidates

A never-before-seen set (not in the training data at all) cannot appear in the sampled support, so the belief model is blind to it. However:

- The **value net generalizes** from features: it knows what Focus Sash does and what Shadow Ball does, so it can evaluate their combination on a new species even without that exact combination in training. Only the prior probability is off, not the evaluation.
- **In-game observation** self-corrects: the moment Sash triggers or Shadow Ball is used, the conditional sampler updates and candidates matching the evidence rise to the top.
- This is **equally true of human players** — a wholly novel set surprises everyone for a turn, then adaptation begins.

The worst case is a slightly miscalibrated prior on a novel set for the opening turn, self-correcting on reveal. Not "blind to innovation."

---

## 8. The Transformer architecture

### 8.1 Why a Transformer

| Requirement | MLP | Transformer |
|---|---|---|
| Entity-centric structure (12+ Pokémon as "same kind of thing") | Flattens to one vector; no weight-sharing across slots | Shared encoder; weight-sharing is free |
| Variable token count (different numbers of candidates per game) | Fixed input dim; would need separate architectures or padding tricks | Native: attention matrix resizes to match tokens |
| Permutation equivariance (slot order is arbitrary) | Not inherent; would need heavy data augmentation | Built in: permute input rows → outputs permute identically |
| Cross-entity reasoning (Incineroar's value depends on what it faces) | Must learn to find entity $j$'s features buried in the flat vector | Attention explicitly routes info between entities |
| Field info not "buried" among thousands of entity features | One float among thousands; data-hungry to learn its importance | Field is a peer token; every entity attends to it directly |

The Transformer is not the only viable architecture (see Section 11.1 for alternatives), but it best matches the structure of the problem.

### 8.2 How attention works

For every token vector $x_i$, project it three ways with learned weight matrices (each is a single `nn.Linear` / dense layer):

- **Query** $q_i = x_i W_Q$ — "what is this token looking for?"
- **Key** $k_i = x_i W_K$ — "what does this token advertise about itself?"
- **Value** $v_i = x_i W_V$ — "what content does this token contribute if attended to?"

(Note: "value" here is the attention-mechanism term, unrelated to the RL value head.)

Relevance of token $i$ to token $j$ is the dot product $q_i \cdot k_j$. Scale by $\sqrt{d_k}$ to prevent large-dimensional dot products from saturating the softmax, then softmax over $j$ to get weights summing to 1:

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V
$$

Each token's output is the weighted sum of all tokens' values — a soft, differentiable dictionary lookup.

**Critical property: the learned weight matrices $W_Q$, $W_K$, $W_V$ do NOT depend on the token count.** Their shape is $[d_{\text{model}}, d_k]$, fixed at architecture time. The token count $n$ flows through as a batch dimension:

$$
Q = x \, W_Q \quad \Rightarrow \quad [n, d_{\text{model}}] \times [d_{\text{model}}, d_k] = [n, d_k]
$$

The only place $n$ appears squared is in the attention-score matrix $Q K^\top$, which is a **computation** (dot products between all pairs), not a learned parameter. 14 tokens → $14 \times 14$ scores; 30 tokens → $30 \times 30$ scores. Same $W_Q$, $W_K$, $W_V$ produced both.

### 8.3 Variable token count across game states

The Transformer natively handles variable-length inputs. Different games produce different token counts (different numbers of opponent candidates), and the same trained network processes all of them. The weights are token-count-agnostic; only intermediate computations (score matrices) resize.

**Batching caveat:** within a training/inference batch, tensors must be rectangular. The standard solution is **padding + masking**: pad shorter sequences with dummy tokens to the longest sequence in the batch, and pass a `src_key_padding_mask` so attention ignores padding tokens. This is identical to how NLP Transformers handle variable-length sentences.

### 8.4 Multi-head attention

Run the Q/K/V projection $h$ times in parallel with smaller, independent $W_Q^{(j)}, W_K^{(j)}, W_V^{(j)}$ (each projecting to $d_k / h$ dims), concatenate outputs, project back to $d_{\text{model}}$. Each "head" can specialize: one learns speed relationships, one learns type matchups, one learns "who threatens whom."

### 8.5 The Transformer block: attention + per-token MLP

A single Transformer block is:

$$
z = x + \text{Attention}(\text{LayerNorm}(x))
$$
$$
\text{out} = z + \text{MLP}(\text{LayerNorm}(z))
$$

The MLP (called the "feed-forward network") is a standard 2-layer dense network applied **independently to each token** using **shared weights**:

```
ffn = Linear(d_model, 4*d_model) → ReLU → Linear(4*d_model, d_model)
out = ffn(x)   # x: [n_tokens, d_model] → out: [n_tokens, d_model]
                # out[:, i] depends ONLY on x[:, i] — never on x[:, j≠i]
```

Not 12 separate MLPs (that would break weight-sharing), and not one MLP over the whole `[12, d_model]` (that would mix across entities, which is attention's job). It is **one MLP, applied to each token independently but sharing weights**.

The division of labor: **attention = "let the Pokémon talk to each other."** **FFN = "let each Pokémon think about what it just heard."** Residual connections ($x + \ldots$) and LayerNorm keep deep stacks trainable.

Stack $N$ blocks (typically 3–6 for this scale) and the representation deepens: early layers learn simple relationships, later layers compose them.

### 8.6 Permutation equivariance and slot ordering

Attention has no built-in notion of order: permute the input rows and the outputs permute identically. For an *unordered* set of 6 Pokémon, this is correct — slot order is arbitrary and should not matter. (LLMs add positional encodings because word order matters in language. **Deliberately omit positional encodings on the team axis**; instead, mark "active vs back," "my side vs theirs," and "left slot vs right slot" as **features inside each token**, not as positions.)

---

## 9. Meta-priors rewrite direction

The current `src/meta_priors/clustering.py` groups observed tournament sets into fuzzy **archetypes** via distance-based clustering. This is being replaced because:

1. **Archetype boundaries are subjective and fragile.** "Is Sash vs Scarf Basculegion one archetype or two?" depends on human judgment that is hard to make consistently and keep current.
2. **Archetypes lose within-cluster variance.** Two distinct sets collapsed into the same archetype are indistinguishable to the encoding.
3. **GT-CFR needs concrete, distinguishable candidates** (Section 7), not fuzzy buckets.

**Target design: a conditional sampler over real sets.** Given (species, observed teammates, revealed moves/items/ability), return the top-$K$ most probable fully-specified sets from the tournament data, with probabilities. This:

- Keeps the conditional-query capability from the existing Streamlit dashboard (the part that works well: "if I know its item is X, how does the distribution update?").
- Extends conditioning to **teammates** (Incineroar + Mega Gengar → P(Protect) jumps), which the current module does not do.
- Replaces hard cluster boundaries with **empirical frequency ranking**: the top-$K$ are literally the $K$ most common matching sets in the data. No judgment call about what "defines" an archetype.
- Auto-refreshes from new tournament scrapes, making metagame staleness a data-pipeline problem (automatable) rather than a modeling problem.

**Smoothing/backoff** is needed for sparse observations (conditioning on 3+ constraints may yield zero or few exact matches). Options: similarity-weighted (kernel) counting over sets, factored marginal fallback, or a small learned conditional model. This is the main open design problem in the rewrite.

**Decoupled from Phase 1.** Phase 1 reveals both teams, so meta-priors are not on the critical path. The rewrite is Phase 4 work.

---

## 10. Phasing: how the encoding evolves

| Aspect | Phase 1 (pit-stop) | Phase 4 (GT-CFR north star) |
|---|---|---|
| **Opponent info** | Fully revealed (perfect info) | Hidden; represented as candidate tokens |
| **Token count** | Fixed: 14 (CLS + field + 6 yours + 6 theirs) | Variable: 8 + your 6 + $\sum_i K_i$ opponent candidates |
| **Belief weights** | All 1.0 (everything known) | $< 1.0$ for opponent candidates |
| **Value head input** | CLS token | Each candidate token |
| **Value head output** | Scalar: $v \in [-1, 1]$ | Vector: one CFV per candidate |
| **Policy head** | CLS → action distribution | CLS → action distribution (unchanged) |
| **Meta-priors** | Not used | Conditional sampler supplies candidate support |
| **Backbone** | Shared Transformer | Same shared Transformer |

Phase 1 is the **degenerate case** of Phase 4: belief = delta (one "candidate" per slot with weight 1.0), value head = scalar (one CFV, which *is* the scalar). The same schema throughout; Phase 4 stops pinning the belief to a delta and swaps the value head.

---

## 11. Alternatives considered

### 11.1 Architecture: flat MLP vs shared encoder vs Transformer

| Architecture | Pros | Cons | Verdict |
|---|---|---|---|
| **Flat MLP** on one big concatenated vector | Simplest to implement; no attention complexity | Forces fixed token count; no weight-sharing across slots; field info buried among thousands of features; no permutation equivariance (slot order matters); "buried needle" risk for small-but-critical features | Viable for Phase 1 as a quick first pass, but would need replacement |
| **Shared per-entity encoder + pooling** (DeepSets-style) | Good weight-sharing; simpler than attention; permutation-invariant after pooling | No cross-entity reasoning before pooling (each entity encoded in isolation); field must be concatenated after pooling or prepended to each entity | A reasonable middle ground if Transformer training proves too finicky |
| **Transformer** (attention over entity tokens + field token) | Best structural match: weight-sharing, permutation equivariance, cross-entity reasoning, variable token count, field as peer token | Slightly more complex to implement and train (LayerNorm placement, learning-rate warmup, mask bugs) | **Selected.** The structural advantages compound, especially for Phase 4 |

A legitimate path: ship a flat-MLP encoder for Phase 1 (confirm the self-play loop works), then swap in the Transformer behind the same interface. The encoding schema does not change; only the consumer architecture does.

### 11.2 Belief representation: marginal probabilities vs concrete candidates

**Marginal encoding** (one column per move = $P(\text{knows it})$): each move/item/ability gets a probability independently. "78% chance of Wave Crash, 60% chance of Liquidation." Simple, but:

- Loses **joint information**: cannot express "if Scarf then Wave Crash, if Sash then Liquidation." Two very different sets can produce identical marginals.
- Provides **no clean enumeration** for the value head to index against — what does "CFV for marginal vector" even mean?
- Not compatible with the GT-CFR vector value head, which needs one CFV per distinguishable private state.

**Concrete candidates with belief weights** (Section 7): each candidate is a fully-specified joint; the belief is a weight *on* the candidate. Preserves joints; provides the enumeration the value head needs; compatible with GT-CFR.

**Verdict:** marginals are a viable quick-and-dirty encoding for Phase 1 (where there is no hidden info and the question is moot) or for a non-GT-CFR baseline. For GT-CFR, concrete candidates are required.

### 11.3 Opponent private-state space: archetypes vs data-driven candidates

**Hand-defined archetypes** (current `src/meta_priors/clustering.py`): collapse real sets into fuzzy buckets. Problems: subjective boundaries, hard to maintain, loses within-cluster variance. See Section 9.

**Data-driven top-$K$ candidates**: literally the $K$ most common *real* sets matching the observations, drawn from tournament data. No judgment calls; auto-refreshable. **Selected** (pending the smoothing/backoff design).

### 11.4 Belief placement: School A vs School B

These terms were used in the design discussion to distinguish two approaches to handling hidden information in the network:

**School B (sample-one-world, no belief in input).** The network only ever sees one concrete opponent state (sampled from the belief), as if perfect info. Scalar value head. The distribution lives entirely in the search/sampling layer: sample many worlds, call the net once per world, average. Problems: (a) many forward passes per search node (one per sampled world); (b) at depth-limited search leaves, a single concrete-world scalar cannot represent "great vs Scarf range, terrible vs Sash range" — this is the core limitation that motivated DeepStack's range-conditioned value function; (c) when paired with determinized MCTS (not CFR), causes strategy fusion.

**School A (belief in input, vector value head).** The network sees all candidates simultaneously (as tokens with belief weights) and returns one CFV per candidate in a single forward pass. Range-aware: the net can express "this position is great if they're Scarf, terrible if they're Sash" in one evaluation. One forward pass gives all the CFVs the CFR regret update needs. Problems: (a) heavier network per call; (b) requires a finite candidate support (the enumeration/support design problem).

**Verdict:** School A is correct for GT-CFR. The heavier-per-call cost is dominated by the many-fewer-calls advantage. School B is the natural Phase 1 design (everything known = one "candidate" per slot), and the transition is: stop feeding deltas, start feeding real candidate sets.

### 11.5 Field delivery: peer token vs broadcast vs prefix

Three ways to give every Pokémon access to field information:

- **Peer token** (selected): field is its own token; Pokémon attend to it through attention. Clean, no duplication, field updates don't require re-encoding all Pokémon tokens.
- **Broadcast**: concatenate field features onto every Pokémon token before the Transformer. Duplicates the field 12+ times; field updates require re-building all tokens. Wastes space.
- **Post-pool concatenation**: encode entities without field, pool them, concatenate field to the pooled vector. Works for the output heads but prevents per-entity field-awareness *during* attention (the entity tokens can't attend to the field because it isn't there yet).

### 11.6 Stat-point / spread encoding

**Raw stat-point allocation** (6 × 32 one-hot): 192 sparse dims that lose the only thing the net cares about (what stats result). Wastes capacity, ignores the sum-to-66 constraint.

**Derived final stats** (selected): 6 continuous values (the level-50 stats after base + points + nature). The net cares about "does this outspeed Flutter Mane," not the allocation that produced it. Nature rides along as a separate categorical feature (largely redundant once final stats are present, but cheap to include). For opponents, the spread is part of the concrete candidate — the candidate's stats are fed directly.

---

## 12. Open design questions

1. **Tokenization method** (Section 4.2) — one-hot/multi-hot vs learned embeddings vs hybrid for categorical features. Determines per-token width. Requires experimentation.
2. **Top-$K$ per slot** — how many candidates per opponent Pokémon slot? Smaller $K$ = less compute, more abstraction error. Larger $K$ = better coverage, heavier attention. Start with $K = 5$–$10$, tune empirically.
3. **Per-slot independence vs joint-team candidates** — per-slot candidates are computationally trivial but cannot express "the joint value of Scarf-Incin + Sash-Gengar together." Joint-team candidates preserve cross-slot correlations but explode combinatorially ($K^6$ in the worst case). Options: (a) per-slot independence, relying on attention to learn cross-slot correlations; (b) a sampled support of $M$ full-team configs drawn from the conditional joint model. This is the hardest open question for Phase 4.
4. **Smoothing/backoff for the conditional sampler** (Section 9) — how to handle sparse observations (conditioning on 3+ constraints yields zero exact matches).
5. **$d_{\text{model}}$, number of heads, number of layers** — hyperparameters to tune. Starting point: $d_{\text{model}} = 128$, 4 heads, 3 layers (small enough to iterate fast, large enough to learn).

---

## 13. Glossary

- **Token** — a fixed-length vector representing one entity (Pokémon, field, CLS). The fundamental input unit of the Transformer.
- **CLS token** — a learnable summary token with no input data; accumulates a holistic battle summary via attention. Read by the policy head (and value head in Phase 1).
- **Candidate** — one fully-specified concrete set for an opponent Pokémon slot (species + moves + item + ability + stats). Nothing probabilistic inside it.
- **Belief weight** — the probability assigned to a candidate being the true opponent set. A float in $[0, 1]$; one feature inside the candidate token. Sums to 1 across all candidates for a given slot.
- **Support** — the set of candidates with non-zero belief weight. "Live support" = the support after conditioning on observations (it shrinks as info reveals).
- **$d_{\text{model}}$** — the common vector width all tokens are projected into before entering the Transformer. Fixed at architecture time (e.g. 128).
- **Attention** — the mechanism that lets tokens exchange information: each token forms a query, matches it against all tokens' keys, and retrieves a weighted blend of their values. The only operation that moves information between entities.
- **Per-token MLP / FFN** — the feed-forward network inside a Transformer block, applied independently to each token with shared weights. Attention mixes across entities; the FFN digests what each entity just heard.
- **Phase 1** — perfect-info pit-stop: all Pokémon known, scalar value head, fixed 14 tokens. Validates infrastructure and proves self-play feasibility.
- **Phase 4** — GT-CFR north star: imperfect info, belief-weighted candidate tokens, per-candidate CFV vector. The real product.
- **School A** — belief in the network input, vector CFV output. Selected for GT-CFR.
- **School B** — one concrete world per forward pass, scalar output. Used implicitly in Phase 1 (where the one world *is* the known truth) but not the target architecture.

---

*If you are a future agent picking up implementation: for the algorithmic theory underlying CFR, info sets, and the training loop, read `docs/gt-cfr-theory.md` first. This document covers only the encoding and NN architecture. For milestones and tooling, see `research/notes.md` and `.cursor/rules/agent/overview.mdc`.*
