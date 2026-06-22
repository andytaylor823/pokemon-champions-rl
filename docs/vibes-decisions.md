# Vibes-Based Decisions Tracker

**Status:** Living document. Updated as decisions are made or revisited.
**Purpose:** Track every design decision that is not strongly rooted in evidence — choices made on intuition, convention, reasonable defaults, or "feels." Each is tagged with an estimated impact level so we know where to invest empirical validation first.

**Impact levels:**
- **High** — could fundamentally change agent strength, training dynamics, or whether the approach works at all. Wrong choices here may require a significant rearchitecture.
- **Med** — likely affects convergence speed, sample efficiency, or quality ceiling, but a bad choice is recoverable via tuning or a contained swap.
- **Low** — unlikely to matter much; a different choice would produce similar results. Mostly aesthetic or conventional.

---

## 1. Neural Network Architecture

### 1.1 Transformer as the backbone architecture
- **Choice:** Transformer (attention over entity + field tokens) rather than flat MLP, DeepSets, or GNN.
- **Evidence level:** Structural argument only — the problem is entity-centric, variable-length, permutation-equivariant, and requires cross-entity reasoning, all of which Transformers handle natively. No empirical comparison on this domain.
- **Alternatives considered:** Flat MLP (simple but loses structure), shared-encoder + pooling / DeepSets (reasonable middle ground, no cross-entity reasoning before pooling). See `state-encoding.md` §11.1.
- **REVISIT trigger:** If Transformer training proves too finicky (LayerNorm issues, LR warmup sensitivity, mask bugs) or too slow for the search hot path, a shared-encoder fallback is viable.
- **Impact: High** — the backbone determines the representational capacity of the entire system. A fundamentally wrong architecture family would cap agent strength regardless of training.

### 1.2 Transformer dimensions: `d_model = 128`
- **Choice:** 128-dimensional model width.
- **Evidence level:** Pure convention / "small enough to iterate fast, large enough to learn" (`state-encoding.md` §12). No ablation.
- **Alternatives:** 64 (faster, possibly too small for 12+ entity interactions), 256 (more capacity, slower iteration), 512+ (likely overkill for this scale).
- **REVISIT trigger:** Once the self-play loop is running, sweep `d_model` ∈ {64, 128, 256} and measure Elo vs compute.
- **Impact: Med** — too small starves the model; too large wastes compute and may overfit with limited self-play data. But the right ballpark (64–256) is likely fine.

### 1.3 Number of Transformer layers: `n_layers = 3`
- **Choice:** 3 attention blocks.
- **Evidence level:** "3–6 typical at this scale" (`state-encoding.md` §8.5). No ablation.
- **Alternatives:** 2 (faster, possibly insufficient depth for multi-hop reasoning like "does my mon outspeed theirs after Tailwind?"), 4–6 (more compositional depth, diminishing returns likely for 14-token sequences).
- **REVISIT trigger:** Same sweep as `d_model`.
- **Impact: Med** — affects depth of reasoning. Too few layers means the net can't compose relationships (speed after Tailwind after Intimidate); too many is wasted compute. The token count (~14–34) is small enough that depth matters less than in NLP.

### 1.4 Number of attention heads: `n_heads = 4`
- **Choice:** 4 attention heads.
- **Evidence level:** Convention (4 heads × 32 dims each = 128 `d_model`). No ablation.
- **Alternatives:** 2 (coarser but each head has more capacity), 8 (more specialization per head, thinner per-head dim of 16).
- **REVISIT trigger:** Sweep alongside other backbone hyperparameters.
- **Impact: Low** — head count is among the least sensitive Transformer hyperparameters in practice. 2–8 heads in this range are all likely fine.

### 1.5 FFN expansion factor: `ffn_mult = 4`
- **Choice:** FFN hidden dim = 4 × `d_model` = 512.
- **Evidence level:** The standard Transformer convention from Vaswani et al. Universally used; never questioned.
- **Impact: Low** — standard and unlikely to be wrong.

### 1.6 Activation function: GELU
- **Choice:** GELU in the Transformer FFN layers (set in the `nn.TransformerEncoderLayer` config).
- **Evidence level:** Modern Transformer convention (used in BERT, GPT, etc.). ReLU is the older default.
- **Impact: Low** — GELU vs ReLU vs SiLU rarely makes a material difference.

### 1.7 Pre-norm (norm-first) Transformer
- **Choice:** `norm_first=True` in `TransformerEncoderLayer` — LayerNorm before attention/FFN, with a final LayerNorm after the last block.
- **Evidence level:** Pre-norm is the modern standard (more stable training, especially without warmup). Post-norm is the original Vaswani et al. design.
- **Impact: Low** — pre-norm is almost universally preferred now. Unlikely to revisit.

### 1.8 Dropout rate: `dropout = 0.1`
- **Choice:** 0.1 dropout throughout the Transformer.
- **Evidence level:** Conventional starting point.
- **Alternatives:** 0.0 (no regularization — may overfit), 0.05, 0.2.
- **REVISIT trigger:** If overfitting is observed (training loss << validation loss) or underfitting (both high), adjust.
- **Impact: Low** — tunable, and self-play generates essentially unlimited data, so overfitting may not be the primary concern.

### 1.9 No positional encoding on the token axis
- **Choice:** Deliberately omit positional encodings. Slot identity (active/back, side, left/right) is encoded as per-token *features*, not positions.
- **Evidence level:** Principled — slot order is arbitrary and the Transformer should be permutation-equivariant over the team. Marking slot identity as features rather than positions is the correct approach for unordered sets.
- **Impact: Low** — this is the right call for this domain. Revisiting would only make sense if slot ordering somehow carried strategic signal (it doesn't).

---

## 2. Embedding Dimensions

### 2.1 Species embedding: `d_species = 32`
- **Choice:** 32-dimensional learned embedding for ~196 species.
- **Evidence level:** "~16–32" from `encoder.md` Decision 1. No ablation. Intuition: species is the highest-information categorical (determines base stats, typing, movepool), so it gets the largest embedding.
- **REVISIT trigger:** If species-related patterns are slow to learn, increase. If 32 dims dominate the token width unnecessarily, decrease.
- **Impact: Low–Med** — species is important but the final stats, typing, and movepool are also encoded through other channels (stats as floats, moves via their own embeddings). The embedding captures "everything else about being this species."

### 2.2 Move embedding: `d_move = 32`
- **Choice:** 32-dimensional learned embedding per move, shared across all 4 move slots.
- **Evidence level:** Same source as species. Moves are the second-highest cardinality (~552 unique moves) and each move's identity carries dense information (type, power, priority, effects).
- **REVISIT trigger:** Same as species.
- **Impact: Low–Med** — the move embedding must capture enough to distinguish Heat Wave from Flamethrower, which matters for damage calcs and coverage. But much of this is learnable from the move's effects in games.

### 2.3 Item embedding: `d_item = 16`
- **Choice:** 16-dimensional learned embedding for ~120 items.
- **Evidence level:** "Low-to-mid cardinality" reasoning — fewer items than species/moves, so smaller embedding.
- **Alternatives:** 8 (possibly too small — items like Choice Scarf vs Life Orb have very different strategic implications), 32 (probably more than needed).
- **REVISIT trigger:** If item-related strategic errors persist (e.g. not respecting Choice lock, misvaluing berries), the item embedding may be too small.
- **Impact: Low** — items matter a lot strategically, but with only ~120 of them, even a small embedding should capture the key distinctions.

### 2.4 Ability embedding: `d_ability = 16`
- **Choice:** 16-dimensional learned embedding for ~140 abilities.
- **Evidence level:** Same as item — moderate cardinality.
- **Impact: Low** — same reasoning as items. Abilities are important but the cardinality is low enough that 16 dims should suffice.

---

## 3. Move Encoding Strategy

### 3.1 Attention-pool to reduce 4 move embeddings → 1 summary vector
- **Choice:** A single learned query attends to 4 move embedding vectors (K/V projections), producing one `d_move`-dim summary per entity. This is done *before* the main Transformer backbone.
- **Evidence level:** Architectural intuition — it shares move semantics across slots (no per-slot weight duplication) and can learn synergies (Protect + Fake Out). But it **loses individual move identity** after pooling — the backbone sees "move summary" not "move 1 is Heat Wave, move 2 is Protect." Explicitly flagged as REVISIT in `cvpn.md` D3 and `encoder.md` Decision 2.
- **Alternatives:** (a) Concatenate all 4 move embeddings (simple, 4× wider token, no info loss). (b) Per-move tokens — each move becomes its own token in the Transformer sequence (4× more tokens per entity = 48+ move tokens, much heavier attention but full cross-move × cross-entity reasoning). This is "Option C" in the docs.
- **REVISIT trigger:** If the model struggles with move-order-dependent strategies or cannot distinguish between "has Protect as move 1" vs "has Protect as move 3" (though move order shouldn't matter), or more broadly if move-level detail proves lost.
- **Impact: Med–High** — this is a real information bottleneck. Collapsing 4 × 32 = 128 dims of move information into 32 dims forces the network to learn a very compressed representation of the moveset. If critical move distinctions are lost (e.g. "this mon has both Fake Out and Protect" vs "this mon has Fake Out and Follow Me"), it could cap the agent's tactical ceiling. The REVISIT flag exists precisely because this is uncertain.

---

## 4. Token Sequence & Global State

### 4.1 Five global prefix tokens (CLS, field, side_p1, side_p2, meta)
- **Choice:** Five dedicated peer tokens before the entity tokens, each with its own input projection.
- **Evidence level:** Principled design — separating symmetric (field), asymmetric (per-side), and meta (turn/phase) state avoids semantic muddle. The CLS pattern is borrowed from BERT with the same purpose (learned summary for output heads).
- **Alternatives:** Fewer tokens (merge sides into field, merge meta into field) or more (separate weather/terrain tokens). The current split is a reasonable middle ground.
- **Impact: Low** — the exact factoring of global state across tokens is unlikely to matter much. The Transformer will attend to all of them anyway.

### 4.2 CLS token for output heads (vs mean pooling)
- **Choice:** A learnable CLS token read by both heads, rather than mean-pooling all token outputs.
- **Evidence level:** CLS can learn content-weighted pooling (attend more to active threats, less to fainted mons), which mean-pooling cannot. Standard in BERT-family models. No ablation for this domain.
- **REVISIT trigger:** If CLS proves to be a bottleneck (all strategic reasoning funneled through one vector), consider reading from entity tokens directly.
- **Impact: Low** — both approaches are viable and well-studied. CLS is a reasonable default.

### 4.3 Side conditions as separate tokens (not merged with field)
- **Choice:** Two side tokens (one per player) for Tailwind/screens/hazards/mega-used, separate from the field token.
- **Evidence level:** Semantic cleanliness — "field = symmetric, sides = asymmetric." Minor design choice acknowledged in `state-encoding.md` §5.2.
- **Impact: Low** — doesn't change what information is available, just how it's organized.

---

## 5. Encoding Decisions

### 5.1 Hybrid tokenization: embeddings for high-cardinality, one-hot for low-cardinality
- **Choice:** Integer IDs + learned embeddings for species/ability/item/moves; one-hot encoding for status (7 values) and nature (25 values).
- **Evidence level:** Standard practice — embeddings generalize better for high-cardinality features; one-hot is fine when the cardinality is small enough.
- **Impact: Low** — this is a well-understood tradeoff. The only question is where the cutoff should be (nature at 25 values is borderline — could be embedded too), but it's unlikely to matter.

### 5.2 Stats as derived final values (not raw stat-point allocations)
- **Choice:** Encode the 6 final level-50 stats as continuous floats, not the raw stat-point allocation.
- **Evidence level:** Principled — the net cares about "does this outspeed Flutter Mane," not "32 points in Speed." The allocation is many-to-one with the result.
- **Impact: Low** — this is almost certainly correct. The stat formula is deterministic; encoding inputs rather than outputs wastes capacity.

### 5.3 Nature as one-hot (25 dims) despite being largely redundant with stats
- **Choice:** Include nature as a one-hot even though the final stats already reflect it.
- **Evidence level:** "Largely redundant once final stats are present, but cheap to include" (`state-encoding.md` §4.1). The argument is that nature might carry signal beyond the stat modification (e.g. the net could learn "Jolly nature = likely physical attacker" as a prior).
- **REVISIT trigger:** If feature importance analysis shows nature has near-zero influence, drop it to save 25 dims per token.
- **Impact: Low** — 25 dims is a tiny cost, and dropping it would save almost nothing.

### 5.4 Normalization constants: MAX_STAT = 200, MAX_TURNS = 20
- **Choice:** Stats normalized by 200, turns/durations normalized by 20.
- **Evidence level:** Rough round numbers. At level 50, most stats fall in 50–200 (Blissey HP can hit ~350 but that's an extreme outlier in this format). 20 turns is longer than almost any VGC game.
- **REVISIT trigger:** If inputs are frequently near 0 or saturated near 1, adjust the constants.
- **Impact: Low** — normalization constants rarely matter much as long as features are in a reasonable range.

### 5.5 No derived features initially (e.g. effective Speed)
- **Choice:** Encoder forwards raw facts only; the net must learn relationships like effective speed (after Tailwind/TR/paralysis/stat stages) from raw inputs.
- **Evidence level:** "Let the net learn it" philosophy, with a REVISIT flag for effective Speed specifically (`encoder.md` Decision 6).
- **Alternatives:** Pre-compute effective Speed (and potentially other derived features like type effectiveness matchups) so the net doesn't have to learn multiplicative interactions from scratch.
- **REVISIT trigger:** If the agent struggles with speed-related decisions (attacking into a faster threat, misjudging Trick Room), add effective Speed as a derived feature (computed by Showdown).
- **Impact: Med** — speed ordering is one of the most critical tactical factors in VGC. Making the net re-derive `base_speed × nature × stages × tailwind × trick_room × paralysis` from raw components is asking it to learn a complex multiplicative interaction. This is one of the more likely candidates for "we should just pre-compute this."

---

## 6. Action Space Design

### 6.1 Flat joint action space (`A = 1089`) vs factored per-slot heads
- **Choice:** A single flat softmax over all 1089 joint actions (360 team preview + 729 move phase), with phase-split internal heads assembled into one output.
- **Evidence level:** Resolved via design grilling (`cvpn.md` D4). The flat joint approach avoids marginalization issues and handles team preview's permutation structure naturally.
- **Alternatives:** Factored per-slot heads (each slot outputs its own distribution, and joint actions are the product). Simpler per-head but requires marginalizing the joint search target σ̄ and doesn't fit team preview.
- **REVISIT trigger:** If the 729-dim move phase head struggles to learn with sparse rewards (most of the 729 entries are masked anyway), revisit factored heads.
- **Impact: Med** — a flat joint space over ~729 move-phase actions is large but manageable. The masking reduces the effective action space to ~100 per turn. If the policy head can't distinguish between joint actions that differ only in one slot's choice, factored heads would help.

### 6.2 Team preview as P(6,4) = 360 orderings (before canonicalization)
- **Choice:** All 360 permutations of choosing and ordering 4 from 6, even though lead order and bench order are strategically equivalent.
- **Evidence level:** Known to be suboptimal — canonicalization to C(6,2)·C(4,2) = 90 is explicitly planned but deferred (`cvpn.md` D7, `repo-architecture.md` §6.8).
- **REVISIT trigger:** Already planned. Execute when action-space cleanup is prioritized.
- **Impact: Low** — the 4× redundancy in team preview wastes some policy-head capacity but is masked correctly and the net will likely learn equivalent orderings produce similar outcomes. Canonicalization is a clean improvement but not urgent.

---

## 7. Value Head Design

### 7.1 Scalar value from CLS (Phase 1)
- **Choice:** `Linear(d_model, 1)` → `tanh` → scalar in `[-1, 1]`, read from the CLS token.
- **Evidence level:** Standard AlphaZero design. Correct for perfect-information Phase 1.
- **Impact: Low** — this is the canonical approach for Phase 1. The Phase 4 vector CFV swap is already designed.

### 7.2 `tanh` activation for value (bounding to `[-1, 1]`)
- **Choice:** `tanh` to bound the value output to the game payoff range `[-1, 1]`.
- **Evidence level:** Standard — AlphaZero uses this. The game outcome is ±1, so bounding the value to that range is natural.
- **Alternatives:** No activation (unbounded — training signal may push values outside `[-1, 1]`, but MSE loss would pull them back). `sigmoid` scaled to `[-1, 1]`.
- **Impact: Low** — `tanh` is the standard choice for bounded value outputs.

---

## 8. Training & Self-Play (not yet built — decisions from plans)

### 8.1 Replay buffer: FIFO sliding window
- **Choice:** Fixed-capacity FIFO — newest tuples in, oldest out.
- **Evidence level:** AlphaZero lineage convention. Simplest scheme that avoids stale data.
- **Alternatives:** Prioritized experience replay (weight by surprise/error), reservoir sampling, or full history with importance weighting.
- **REVISIT trigger:** If early training plateaus because the buffer contains mostly low-quality early-game data, consider prioritized replay.
- **Impact: Med** — the replay buffer strategy affects sample efficiency and training stability. FIFO is reasonable but may not be optimal. Prioritized replay could accelerate learning, especially early on.

### 8.2 Training loss weights (value MSE vs policy CE vs regularization)
- **Choice:** Equal weighting of value MSE and policy cross-entropy losses, plus L2 regularization. Exact λ not yet set.
- **Evidence level:** AlphaZero uses equal weighting. The relative importance of value vs policy accuracy is domain-dependent and not yet measured.
- **REVISIT trigger:** If one head trains much faster/slower than the other, adjust relative weights.
- **Impact: Med** — loss weighting directly affects whether the net prioritizes getting values right or policies right. In domains where the value landscape is smooth, value can afford less weight; in tactical domains where one-move blunders are fatal, policy weight matters more.

### 8.3 Continuous training (no weight reset between generations)
- **Choice:** The CVPN weights are never reset — continuous SGD from a single initialization, contrast Deep CFR's from-scratch retrains.
- **Evidence level:** Standard AlphaZero/PoG approach. Deep CFR retrains from scratch because it needs the full history's cumulative regret; GT-CFR's CVPN predicts values/policies, not regrets, so continuous learning is appropriate.
- **Impact: Low** — this follows directly from the GT-CFR algorithm choice and is well-justified.

---

## 9. Search & Algorithm Choices (not yet built — decisions from plans)

### 9.1 GT-CFR directly for Phase 1 (perfect-information regime) — *supersedes the earlier MCTS plan*
- **Choice:** Skip the MCTS pit-stop. Build **GT-CFR directly** in Phase 1, staying in the perfect-information *regime* (both teams revealed, belief = delta, scalar value head). Decision and full design recorded in `docs/plans/search.md`.
- **Evidence level:** Principled — simultaneous moves make even *perfect-information* Pokémon a genuine imperfect-information game (each turn is a matrix game whose solution is a mixed average strategy, which minimax MCTS does not produce); and building MCTS first means writing and discarding a second search, since the poker toys and the CVPN are already GT-CFR-shaped. The earlier "MCTS pit-stop" framing in `agent/overview.mdc` milestones and `CONTEXT.md` "Phase 1" is **superseded for the search algorithm** (the perfect-information *regime* is unchanged).
- **REVISIT trigger:** If GT-CFR in the perfect-info regime proves too slow or finicky to get a feasibility loop running, a determinized/MCTS fallback for Phase 1 is still viable.
- **Impact: Med** — well-grounded in theory, but commits the project to GT-CFR earlier than the phased plan assumed. The Phase-4 imperfect-information swap (belief, MCCFR, vector value head) remains the single most important architectural step, but is well-grounded, not vibes.

### 9.2 ~20–50 NN forward passes per search (expansion budget)
- **Choice:** PUCT expansion budget of ~20–50 nodes per search (Phase-1 default `expansion_budget` ≈ 24, `docs/plans/search.md` §9).
- **Evidence level:** Back-of-envelope from `search-nn-interface.md` §8.2 — driven by the 60-second turn timer and ~1–5 ms per Transformer forward pass.
- **Cost-model caveat (added with the Phase-1 Search design):** that estimate assumed *sequential* single-child expansion. With the simultaneous turn-grid (`search.md` §3), expanding one turn-node evaluates ~`k²` joint cells (k ≈ 6 → ~36 leaf states), batched into ~one forward. So per-*search* CVPN work scales with `k²` and exceeds the original figure; wall-clock stays ~one forward per expansion via batching, but self-play throughput will feel it.
- **REVISIT trigger:** Measure actual latency once the Transformer is running; co-tune `expansion_budget` and `k` (9.7). If faster, expand more; if slower, expand less.
- **Impact: Med** — more expansions = deeper/wider tree = better strategy, but with diminishing returns. The budget is a time-constrained optimization problem.

### 9.3 Chance node bucketing (~3–5 buckets) — *Phase-4 / deferred target*
- **Choice:** Discretize damage/accuracy/crit outcomes into ~3–5 representative buckets per chance node carrying their real probabilities. **This is the deferred target, not what Phase 1 does** — Phase 1 instead uses sample-one (reseed) + uniform-weighted Monte-Carlo outcomes capped at K=5 (see 9.8 and `docs/plans/search.md` §5). Bucketing needs `SimClient`'s *structured* chance outcome, which is not yet built (`docs/plans/sim-client.md`). The ChanceNode abstraction is designed so the swap is localized: only child-generation + the weight vector change.
- **Evidence level:** Inspired by DeepStack's flop bucketing (~3–5 groups). Domain reasoning: within a non-crit damage range, the strategic decision rarely changes (68% HP vs 65% HP). The critical boundaries are KO thresholds and status effects.
- **REVISIT trigger:** If the agent makes systematic errors around damage roll boundaries (e.g. not playing around low rolls that KO), increase bucket granularity for KO-threshold ranges. Also the natural successor to the Phase-1 uniform-sampling stopgap (9.8).
- **Impact: Med** — too few buckets loses important distinctions (survive vs faint); too many creates too wide a tree. The choice is domain-sensitive and needs empirical validation.

### 9.4 MCCFR sample count per CFR iteration (Phase 4)
- **Choice:** Not yet set. DeepStack used ~1000 rollouts for poker; the docs suggest Pokémon's shorter horizon may allow fewer.
- **Evidence level:** No evidence yet. Purely a "we'll tune it" placeholder.
- **REVISIT trigger:** When Phase 4 search is running, measure regret convergence vs sample count.
- **Impact: Med** — too few samples = high-variance regret estimates = noisy strategy; too many = wasted compute.

### 9.5 Top-K candidates per opponent slot
- **Choice:** Not yet set. Starting suggestion: K = 5–10 (`state-encoding.md` §12).
- **Evidence level:** Intuition — more than 10 candidates per slot produces diminishing returns in belief coverage while increasing compute quadratically in attention. Fewer than 5 may miss important set variants.
- **REVISIT trigger:** Measure belief coverage (what fraction of real opponent sets fall in the top-K?) and search quality vs K.
- **Impact: Med** — K directly affects how well the agent models opponent uncertainty. Too low and it's blind to plausible threats; too high and search becomes slow.

### 9.6 Simultaneous turn-node representation (Phase-1 Search)
- **Choice:** Model each turn as one **TurnNode** holding *both* players' regret tables, with children keyed by joint cell `(a1, a2)` (`docs/plans/search.md` §2). Chosen over the sequential shared-info-set representation used by the poker toys.
- **Evidence level:** Principled — both representations reach the same stage Nash, but the single-node form makes no-information-leak *structural* (P2's marginal is drawn from the same node, so it cannot condition on a1), removing the shared-info-set-keying footgun that the sequential form carries.
- **REVISIT trigger:** If the joint-cell bookkeeping proves awkward when forced switches / team preview are layered in, reconsider the sequential form.
- **Impact: Med** — structural choice; affects node-type code and the per-node cost shape, not the equilibrium reached.

### 9.7 Top-k action abstraction per turn-node (k ≈ 6)
- **Choice:** At each turn-node, give children only to each player's **top-k legal joint actions by CVPN policy-prior mass** (default k = 6) and solve the `k×k` grid (`docs/plans/search.md` §3). The full legal grid (~100×100 joint cells) is computationally infeasible per node; this is the "top-k from priors" branch of `article_summary_4.md:129`.
- **Evidence level:** Intuition + tractability necessity. k = 6 is a guess balancing grid cost (`k²` leaf evaluations per expansion) against coverage.
- **REVISIT trigger:** If the agent systematically misses strong actions the network underrates, raise k or switch to a grow-on-demand width / probability-mass cutoff. Measure regret vs k.
- **Impact: High** — the CVPN policy prior becomes load-bearing: an action outside the top-k is never considered, so a weak prior directly caps search quality. This is the most consequential Phase-1 search knob.

### 9.8 Chance fan-out cap K = 5, uniform weights (Phase-1 stopgap)
- **Choice:** Each chance node accumulates up to **K = 5** sampled outcome worlds (reseeded `step`s), uniformly weighted, grown only during expansion and frozen during CFR+ updates (`docs/plans/search.md` §5). The successor is real-probability bucketing (9.3).
- **Evidence level:** Intuition — uniform sampling from the true game PRNG is an unbiased Monte-Carlo estimate of the chance node's expectation; K = 5 balances variance against tree size. K = 1 is a one-sample (biased) estimate.
- **REVISIT trigger:** If damage-roll/accuracy variance visibly destabilizes the strategy, raise K or move to bucketing (9.3) once SimClient emits structured chance outcomes.
- **Impact: Med** — too small K = noisy/biased CFVs at chance nodes; too large = wider tree. Recoverable by tuning or the bucketing swap.

### 9.9 Full multi-turn PUCT tree growth in Phase 1
- **Choice:** Phase-1 search grows a multi-turn tree via PUCT expansion (not a shallow one-turn matrix solve) (`docs/plans/search.md` §4).
- **Evidence level:** Intuition — multi-turn lookahead should produce stronger play and better training targets than a single-turn solve, at the cost of a larger first implementation.
- **REVISIT trigger:** If multi-turn growth is too costly to get a feasibility loop running, fall back to shallower trees and lean harder on the value head.
- **Impact: Med** — affects search strength and cost; the depth is tunable via `expansion_budget`.

### 9.10 Inner-loop budgets (CFR+ cadence, c_puct, k, K)
- **Choice:** `cfr_iters_per_expansion` ≈ 10, `c_puct` = 2.0, `expansion_budget` ≈ 24, `k` = 6, `K` = 5 (`docs/plans/search.md` §9, §11). The CFR+ cadence mirrors the toys' `expansion_interval = 10`.
- **Evidence level:** Convention / starting guesses carried over from the poker toys and the architecture docs; no Pokémon ablation.
- **REVISIT trigger:** Sweep once the loop runs and latency is measured; co-tune with 9.2 and 9.7.
- **Impact: Low–Med** — standard CFR/PUCT knobs; the right ballpark is likely fine, and all are tunable behind `SearchConfig`.

---

## 10. Belief Model / Meta-Priors (Phase 4 — not yet built)

### 10.1 Conditional sampler over real tournament sets (replacing clustering)
- **Choice:** Replace the legacy fuzzy-archetype clustering with a top-K conditional sampler drawing from real tournament data, conditioned on species + revealed info + teammates.
- **Evidence level:** Principled — real tournament sets are ground truth, and conditional sampling avoids the subjective boundary problem of clustering. But the **smoothing/backoff for sparse conditioning** is the main open problem and has no solution yet.
- **Impact: High** — the belief model determines how well the agent reasons about opponent uncertainty, which is the core challenge of the project. A bad belief model means the agent is playing against the wrong opponents in its head.

### 10.2 Per-slot candidate independence vs joint-team candidates
- **Choice:** Not yet resolved. Described as "the hardest open question for Phase 4" (`state-encoding.md` §12.3).
- **Evidence level:** None — pure architectural tradeoff. Per-slot is tractable but misses cross-slot correlations (item clause, team synergies); joint-team is correct but combinatorially explosive.
- **Alternatives:** The three-tier split (`search-nn-interface.md` §7) is a proposed middle ground — per-slot tokens in the backbone for compositional attention, with a joint value head that evaluates specific deals.
- **Impact: High** — cross-slot correlations are strategically critical in VGC (team synergies, item clause). Getting this wrong means the belief model assigns probability to impossible or unrealistic team configurations.

### 10.3 Value-head combination function for joint deals
- **Choice:** Concatenate selected candidate embeddings and pass through a small MLP (`search-nn-interface.md` §7).
- **Evidence level:** The simplest approach. Alternatives (pool, cross-attend, small Transformer over 4 selected embeddings) are explicitly listed as open (`search-nn-interface.md` §10.1).
- **Impact: Med** — affects how well the value head captures interactions between opponent slots. Concatenation is simple but may not capture complex interactions well.

---

## 11. Infrastructure & System Design

### 11.1 Python-centric production system with TS engine as a service
- **Choice:** All ML/search/training in Python; Showdown engine in TypeScript reached via SimClient.
- **Evidence level:** Pragmatic — PyTorch is Python, the Showdown engine is TS. The language boundary is minimized to one seam.
- **Impact: Low** — this is driven by the ecosystem, not by vibes. The only real alternative (rewrite the engine in Python) would be a massive effort for no strategic benefit.

### 11.2 Handle-based SimClient (opaque handles, JSON-lines IPC)
- **Choice:** Heavy Battle objects stay in Node; Python holds opaque handles and exchanges compact JSON messages.
- **Evidence level:** Principled — avoids serializing multi-KB battle objects across the boundary on every call. Validated by the clone/reseed/latency spike (~1.6 ms per fork+step, ~72 KiB/clone).
- **Impact: Low** — the design is validated and working. The transport (JSON-lines vs MessagePack vs socket) is a tuning knob.

### 11.3 Self-play parallelism: not yet chosen (Ray vs multiprocessing)
- **Choice:** Deferred. Options are Ray (distributed, more complex) or plain multiprocessing (simpler, single-machine).
- **Evidence level:** None — will depend on scale needs.
- **Impact: Med** — affects throughput of the training pipeline. Ray enables multi-machine scaling but adds complexity. Multiprocessing is simpler but limits to one machine.

---

## Summary by Impact

| Impact | Count | Key items |
|--------|-------|-----------|
| **High** | 4 | Transformer as backbone (1.1), top-k action abstraction (9.7), conditional belief sampler design (10.1), per-slot vs joint candidates (10.2) |
| **Med** | 16 | d_model (1.2), n_layers (1.3), move attention pool (3.1), no derived features (5.5), flat joint action space (6.1), replay buffer strategy (8.1), loss weights (8.2), GT-CFR-direct for Phase 1 (9.1), expansion budget (9.2), chance bucketing (9.3), MCCFR samples (9.4), top-K candidates (9.5), simultaneous turn-node (9.6), chance fan-out K=5 (9.8), full multi-turn growth (9.9), inner-loop budgets (9.10) |
| **Med (value head)** | 1 | Value-head combination function (10.3) |
| **Low** | 14 | n_heads (1.4), ffn_mult (1.5), GELU (1.6), pre-norm (1.7), dropout (1.8), no positional encoding (1.9), all embedding dims (2.1–2.4), hybrid tokenization (5.1), derived stats (5.2), nature one-hot (5.3), normalization constants (5.4), team preview redundancy (6.2), scalar value + tanh (7.1–7.2), continuous training (8.3), system design (11.1–11.2) |
