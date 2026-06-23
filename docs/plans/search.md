---
last_synced: 3c3811f
watches:
  - src/
  - src/search.py
  - docs/architecture/
  - docs/plans/
---

# Search — Phase-1 Design (GT-CFR, perfect-information regime)

> **Module ref:** `docs/architecture/repo-architecture.md` §3.5 (Search — the inner-loop planner; the project's primary seam)
> **Design refs:** `docs/architecture/gt-cfr-theory.md` (the algorithm), `docs/architecture/search-nn-interface.md` (how the CVPN is called at runtime)
> **Discussion refs:** `docs/research/article_summary_4.md` (the GT-CFR iteration loop on Pokémon, simultaneous-move tree), `docs/research/article_summary_5.md` (node types, forced switches), `toy_examples/{kuhn,leduc}_poker/gt_cfr_search.py` (working reference implementations)

## Status

**Design locked (grilling session). Implementation not started.** This document is the
complete Phase-1 specification: the design decisions, the tree structure, the inner-loop
algorithm, the interfaces consumed, and the seam exposed to the outer loop. Tuning knobs that
were chosen on intuition are recorded in `docs/vibes-decisions.md` §9 and cross-referenced
inline as *(vibes N.N)*.

## Context — what changed from the original plan, and why

The milestones (`agent/overview.mdc`, `gt-cfr-theory.md` §14) and `CONTEXT.md`'s "Phase 1
(pit-stop)" originally framed the first search as **MCTS** — perfect-info AlphaZero — as a
brief stop before swapping to GT-CFR for the Phase-4 north star. **This design supersedes that
for the search algorithm:** Phase 1 builds **GT-CFR directly**, while staying in the Phase-1
*information regime* (perfect information). Two reasons:

1. **Simultaneous moves make perfect-info Pokémon a genuine imperfect-information game.** Even
   with both teams fully revealed, the two players submit their joint actions without seeing
   each other's choice. In extensive-form terms the second mover's info set bundles every one
   of the first mover's joint actions (`article_summary_4.md` §"Simultaneous Moves and Info
   Sets"). A turn is therefore a matrix game whose solution is a **mixed** average strategy, not
   a single best move — exactly the regime CFR+ is built for and minimax MCTS is not. This is
   *step thinking* ("they'll Protect, so I attack; but they expect that…"), which has no
   pure-strategy solution.
2. **The reference code and the CVPN are already GT-CFR-shaped.** The poker toys implement
   GT-CFR, and `src/cvpn.py` already exposes a policy head over the joint action space plus a
   scalar value head designed to become the Phase-4 CFV vector. Building MCTS first would mean
   writing — and throwing away — a second search.

So Phase 1 is the **degenerate case of the Phase-4 north star** (`repo-architecture.md` §0.3):
same algorithm (GT-CFR), same network, with the imperfect-information machinery pinned to
trivial values. The upgrade to Phase 4 is un-pinning those values, not a rewrite.

### What Phase 1 pins to trivial values

| Phase-4 concept | Phase-1 degenerate value | Consequence for Search |
|---|---|---|
| **Belief range** over opponent private state | A delta — one **world**, the true revealed teams | No belief support to carry; the encoder sets every candidate's **belief weight** to 1.0 |
| **MCCFR** sampling of worlds | Not needed — there is exactly one world | CFR+ traverses the single true world; no per-world importance weighting |
| **Meta priors** supplying candidates | Unused | Search never calls the belief model |
| **Value head** = vector of CFVs (one per world) | A length-1 vector = the **scalar** value head | One scalar per leaf; zero-sum (`+v` for p1, `−v` for p2) |
| **Chance bucketing** (~3–5 strategic buckets) | Sample-one outcomes, uniform weights (see §5) | Chance handled by reseeding the simulator; bucketing deferred |

The **only** source of stochasticity that survives into Phase 1 is the game's own RNG —
the **chance nodes** (damage rolls, accuracy, crits, secondaries, multi-hit, speed ties). All
the difficulty of Phase-1 Search is concentrated there and in the simultaneous-move grid.

## 1. The two loops, and where this module sits

This module *is* the **inner loop**: at one decision point it builds a search tree, runs a
fixed budget of CFR+ updates interleaved with tree growth, reads off an average strategy,
samples nothing itself, and returns. The caller (SelfPlay, the **outer loop**) samples the
real joint action from the returned average strategy, plays it through `SimClient`, and
discards the tree. The tree and the regret tables are **ephemeral** — rebuilt fresh for every
decision.

```
search(view, sim, net, session, config) ──► SearchResult
        │                                       │  per-side average strategy σ̄,
        │  builds an ephemeral tree,            │  zero-sum scalar value,
        │  ~budget expansions × ~10 CFR+        │  full-[A] policy targets
        │  updates each, then throws it away    └──► outer loop samples + plays
        ▼
   SimClient (transitions) · Encoder (history → tokens) · CVPN (leaf eval + policy prior)
```

## 2. The search tree

The tree is over **public states**. Because Phase 1 is perfect-information, each node also
corresponds to a single concrete **history** (one world, no hidden branching), which lets us
hold a live `SimClient` handle on it. Node type is decided by reading `StateView.phase` and
`StateView.to_move` after each transition.

```
            ┌─ TurnNode (both players to move) ───────────────────────────────┐
            │  info-set tables for p1 AND p2 (regret, strategy-sum, visits)    │
            │  cached policy prior P(I,·) per side  ·  one SimClient handle     │
            │                                                                  │
            │  grid of joint cells, top-k(p1) × top-k(p2):                     │
            │     (a1,a2) ─► ChanceNode ─► outcome worlds ─► next TurnNode      │
            └──────────────────────────────────────────────────────────────────┘
   a1 \ a2 │  a2=m1      a2=m2      a2=switch ...
   ────────┼──────────────────────────────────────
   a1=m1   │ ChanceNode  ChanceNode  ChanceNode        each ChanceNode:
   a1=m2   │ ChanceNode  ChanceNode  ChanceNode          ≤K outcome children (worlds),
   a1=sw   │ ChanceNode  ChanceNode  ChanceNode          uniform weights, value = Σ wᵢ·v(childᵢ)
```

### Node types

- **TurnNode** — `to_move = [p1, p2]` (the `move` phase, and `teamPreview`). Holds **two**
  info sets (one per player), each with its own **regret table** (cumulative regret per
  action), strategy-sum (for the **average strategy**), and visit counts; plus the cached
  **policy prior** per side and the live `SimClient` handle. Children form the joint-action
  grid (§3). Simultaneity is structural: the two players' regret tables are updated from the
  same set of cells, so neither player's average strategy can condition on the other's
  current-turn choice — no information leak, by construction.
- **UnilateralNode** — `to_move = [one side]`. The canonical case is a **forced switch** after
  a faint (`article_summary_5.md` §"Forced switches break the simultaneous pattern"). Only the
  acting player has a decision; modeled as a degenerate TurnNode where the non-acting side has
  a single no-op action, so the grid is `1×k`. (A double-faint where both must switch is a
  normal simultaneous TurnNode over the switch action space.)
- **ChanceNode** — one per joint cell `(a1, a2)`. Carries up to **K** outcome children, each a
  concrete resulting world (a `SimClient` handle from one reseeded `step`), with uniform
  weights. Its CFV is the weighted average over its children (§5). Frozen during CFR+ updates;
  grown only during expansion.
- **Frontier leaf** — any unexpanded non-terminal node. Its CFV is the cached CVPN scalar
  written when its parent was expanded. The CVPN is **never** called during a CFR+ update.
- **Terminal leaf** — `StateView.terminal` is true. CFV is the real payoff `StateView.utility`
  (±1 for p1). No network involved — this is the ground truth that **bootstrapping** ultimately
  traces back to.

`teamPreview` is handled by the same TurnNode machinery over the team-preview region of the
action space; the raw 360×360 grid is bounded by top-k exactly as the move phase is.

## 3. Action abstraction — the top-k grid

A simultaneous turn is irreducibly a **grid**: to know how good your joint action `a1` is, you
need its payoff against every joint action `a2` the opponent might play, because regret
matching marginalizes over the opponent's mixed strategy:

$$v_{\text{p1}}(a_1) \;=\; \sum_{a_2} \sigma_{\text{p2}}(a_2)\, \cdot\, \text{CFV}(a_1, a_2)$$

You cannot skip a column: dropping `a2` from the grid silently assumes the opponent never plays
it, which is exploitable. The only sound way to shrink the grid is to shrink the **action set**.

**Why this can't be left entirely to incremental expansion (the Pokémon-vs-poker wrinkle).** In
the poker toys, moves are sequential — after P1 bets there is a *real* "facing-a-bet" history
the network can score, so each action maps to one child with one CFV, and PUCT can ignore the
rest until it cares. In Pokémon the moves are simultaneous, so there is no real "after my move,
before theirs" history to hand the network; the only histories the CVPN can encode are **turn
boundaries** (after both joint actions resolve through chance). So evaluating a joint action
requires pairing it against the opponent's candidate replies — a `k×k` block, not a single
child.

**Decision:** at TurnNode creation, keep each player's **top-k legal joint actions by CVPN
policy-prior mass** and solve the `k×k` grid. Default **k = 6** *(vibes 9.7)*. This is the
"add k children based on top-k actions from priors" option from `article_summary_4.md:129`; the
listed alternative ("expand with all actions") is the ~100×100 grid, which is computationally
infeasible per node. Consequence: the **policy prior is load-bearing** — a strong joint action
the network underrates never enters the grid, a real and tunable source of bias.

## 4. The inner loop — two interleaved phases

Per `search()` call, on a fixed budget, GT-CFR alternates two phases (`gt-cfr-theory.md` §10.2):

**Phase A — CFR+ update (the regret-update phase).** Run `cfr_iters_per_expansion` traversals
over the **current, frozen** tree. At each TurnNode, compute the CFV of every grid cell
(recurse into expanded ChanceNodes; read the cached scalar at frontier leaves; read the real
payoff at terminals), marginalize to per-action CFVs for each player, update both players'
cumulative regret with **regret matching+** (instantaneous regret added, then clipped to ≥ 0),
derive the next iterate's strategy, and accumulate the **average strategy** with CFR+ linear
(iteration-weighted) weighting. **No CVPN calls and no chance rolls happen in this phase** — it
is pure arithmetic over cached CFVs and the ephemeral regret tables.

**Phase B — Expansion (tree growth).** Walk down from the root by the PUCT score and expand one
frontier node (`gt-cfr-theory.md` §10.2):

$$\text{score}(I,a) \;=\; \underbrace{\sigma^{t}(I,a)}_{\text{current strategy from the regret table}} \;+\; c_{\text{puct}} \cdot \underbrace{P(I,a)}_{\text{policy prior}} \cdot \frac{\sqrt{\sum_b N(I,b)}}{1 + N(I,a)}$$

`c_puct = 2.0` *(vibes 9.10)*. The Q-analogue is the regret-derived current strategy, **not** a
predicted value; `P` is the **policy prior** (the CVPN's one-shot guess, distinct from the
average strategy `σ̄` the search outputs); `N` are visit counts. Expanding a node is the **only**
place the CVPN is called (`search-nn-interface.md` §1): one expansion = one CVPN forward.

### The frozen-tree invariant (why chance does not re-roll every iteration)

A natural-but-wrong design re-rolls chance on every CFR+ traversal. That would manufacture a
new, never-seen world each iteration, forcing a CVPN call per iteration and blowing up the "one
CVPN call per expansion" cost model. **Phase 1 does not do this.** The tree — including each
ChanceNode's set of materialized outcome worlds — is **frozen** during the CFR+ phase. New
chance outcomes (and all CVPN calls) happen **only** during expansion, where the cost is
budgeted.

## 5. Chance handling

Each joint cell `(a1, a2)` owns a **ChanceNode**. Its lifecycle:

1. **Created** when its parent TurnNode is expanded: reseed and `SimClient.step` the cell's
   joint action once → one resulting **world** (a child handle), CVPN-evaluated as a frontier
   leaf. Children = `[world₁]`, weights `[1.0]`.
2. **Frozen during CFR+:** `CFV(ChanceNode) = Σᵢ wᵢ · CFV(childᵢ)`. With one child, it is that
   child's CFV. No rolls, no CVPN.
3. **Grown during expansion:** when a PUCT descent passes through a ChanceNode that has fewer
   than **K** children, reseed and `step` again → a new world, CVPN-evaluate it, append, and
   re-normalize to uniform weights. Beyond K, descents pass through to grow the decision nodes
   below.

**Cap K = 5, uniform weights** *(vibes 9.8)*. Because `SimClient` reseeds from the true game
PRNG, the uniform average over sampled worlds is an unbiased Monte-Carlo estimate of the chance
node's expectation; its variance shrinks as outcomes accrue. The K=1 case is a one-sample
estimate — biased toward the rolls that happened to be drawn, acceptable for a feasibility
phase.

**Deferred (Phase-4 door):** replace "uniformly sample worlds" with **chance bucketing** —
~3–5 strategically-distinct buckets (KO/survive, hit/miss, crit) carrying their real
probabilities (`search-nn-interface.md` §8). That needs `SimClient`'s *structured* chance
outcome, which is not yet built (`docs/plans/sim-client.md`, "Structured chance outcome").
The ChanceNode abstraction is designed so this is a localized swap: only the child-generation
method and the weight vector change — the averaging in step 2 and the whole tree above are
untouched.

## 6. CVPN interface contract (consumed by Search)

The CVPN is called **once per expansion**, batched. Expanding a TurnNode requires:

- **Two policy priors.** `encode(view, "p1")` and `encode(view, "p2")` (the Encoder builds each
  side's action mask from `view.legal[side]`), then the CVPN policy head gives each side's
  prior over the joint action space `[A]`; the top-k per side defines the grid.
- **The cell-leaf CFVs.** For each of the `k×k` cells, `SimClient.step` produces a resulting
  world; encode each and read the scalar value head.
- **One batched forward.** All of the above (`2 + k²` token bundles) are collated
  (`obs_bundle.collate_obs_bundles`) into a single `CVPN.forward`, so an expansion is ~one
  network call regardless of `k`. The value is **zero-sum**: one scalar per world serves p1 as
  `+v` and p2 as `−v`, so only one perspective's value is needed.

The CVPN's `value` is read from the **CLS token** (Phase-1 scalar). The Phase-4 swap reads one
CFV per candidate **token** instead — same forward, different head wiring (`repo-architecture.md`
§3.3), no change to this module's call pattern.

## 7. The seam to the outer loop — `SearchResult`

`search()` solves one turn matrix and returns an average strategy for **both** players at once
(both regret tables are solved together), plus the zero-sum scalar value. This is the GT-CFR
analogue of AlphaZero's `(state, MCTS policy, outcome)` and is the **stable boundary** between
inner and outer loops (`repo-architecture.md` §3.7).

```python
@dataclass(frozen=True)          # frozen dataclass per agent/overview.mdc (not Pydantic; this is a hot-path internal)
class SearchResult:
    strategy: dict[Side, dict[int, float]]   # σ̄ per side over canonical action indices (top-k support)
    value: float                              # search-refined CFV for p1 (negate for p2)
    policy_target: dict[Side, np.ndarray]     # σ̄ as a full [A] vector per side (mass on top-k, zeros elsewhere)
```

The outer loop samples one **joint action** per side from `strategy`, plays both through
`SimClient.step`, and emits **two training tuples per decision** — one per perspective, so the
CVPN trains on both sides of the same solve:

$$\big(\text{encode}(\cdot,\text{p1}),\; +v,\; \bar\sigma_{\text{p1}}\big) \qquad \big(\text{encode}(\cdot,\text{p2}),\; -v,\; \bar\sigma_{\text{p2}}\big)$$

The policy target is the **average strategy** written into the full `[A]` vector (mass on the
top-k support, zeros elsewhere), matching the CVPN policy head's width. The value target is the
search-refined CFV — better than the raw network because search folds in real terminals and the
CFR improvement operator (**bootstrapping**).

## 8. Adaptive compute

- **Single-legal-move skip (build now).** If a side's legal mask has exactly one legal joint
  action, fix it without searching; if both sides do, return immediately with mass-1.0 average
  strategy (value from a single CVPN evaluation if a training target is needed). Composes with
  the planned action-space canonicalization (`repo-architecture.md` §6): the double-faint
  "bring both back" case collapses to one canonical action and is auto-skipped.
- **All other early-stopping is DEFERRED** — including an average-strategy-convergence early
  stop (abort once σ̄ stops changing between CFR+ passes, KL < ε; the GT-CFR analogue of Lc0's
  smart pruning) and any dynamic per-turn time budget. Defer until search cost is a measured
  bottleneck. An exact σ̄-convergence stop carries no strategic risk; heuristic pruning does, and
  under-searched nodes would bias the training targets baked into the network.
- **TODO before building the 60 s-turn time manager:** how AlphaGo allocated thinking time
  across moves in the Lee-Sedol match (win-rate-vs-budget), and Lc0's curve-based time
  management. Refs: AlphaZero arXiv:1712.01815 (fixed 800-sim self-play budget, no adaptive
  stop), KataGo arXiv:1902.10565, lczero.org/blog/2018/09/time-management.

## 9. Cost model and budget

| Quantity | Phase-1 value | Note |
|---|---|---|
| CVPN forwards per expansion | ~1 (batched: `2 + k²` token bundles) | the expensive operation; everything else is arithmetic |
| Expansions per search | `expansion_budget` ≈ 24 *(vibes 9.2)* | the tree-size / strength knob |
| CFR+ updates per expansion | `cfr_iters_per_expansion` ≈ 10 *(vibes 9.10)* | no CVPN calls |
| Grid width per player | `k` ≈ 6 *(vibes 9.7)* | `k²` token bundles per expanded TurnNode |
| Chance children per cell | ≤ `K` = 5 *(vibes 9.8)* | uniform-weighted sampled worlds |

**Throughput flag.** A simultaneous turn costs ~k² leaf evaluations per expanded node — well
above the ~20–50-CVPN-passes-per-search figure in `search-nn-interface.md` §8.2, which assumed
sequential single-child expansion. Batching keeps wall-clock at ~one forward per expansion, but
self-play **throughput** (many actors, many decisions) will feel the `k²` factor. Revisit
`k` / `expansion_budget` once measured (tracked in vibes 9.2).

## 10. Dependencies and reused interfaces

Search is a pure **consumer**; it reimplements nothing. (Battle rules are never reimplemented
in Python — `SimClient` is the sole source of truth.)

- `src/sim_client.py` — `open_search(from_handle)` opens a per-search **session**; `step(handle,
  {Side: choice_string}, seed)` clones the parent, reseeds, and resolves (the **mandatory seed**
  is what makes each chance roll a fresh sampled world); `view(handle)`; `close_search(session)`
  frees the whole tree's handles in one shot at the end. Handles are immutable snapshots, so the
  parent stays steppable with other cells/seeds.
- `src/action_space.py` — `legal_mask(request, phase) -> [A] bool`; `index_to_choice_string(idx)
  -> str` (canonical action index → Showdown choice string for `step`); `choice_string_to_index`;
  constants `A`, `TEAM_PREVIEW_*`, `MOVE_PHASE_*`.
- `src/encoder.py` — `encode(view, perspective) -> ObsBundle` (perspective ∈ {"p1","p2"}; builds
  the side's action mask).
- `src/cvpn.py` — `forward(obs) -> (policy_logits, value)`; batches via
  `obs_bundle.collate_obs_bundles`.
- Reference for the CFR+ math (regret matching+, σ̄ averaging, chance enumeration, PUCT
  expansion): `toy_examples/leduc_poker/gt_cfr_search.py`.

## 11. Module shape

New file `src/search.py` (split to a package only if it grows unwieldy). Frozen dataclasses
for `SearchConfig`, the node types, and `SearchResult` (per `agent/overview.mdc`: dataclasses
over Pydantic for hot-path internals; Pydantic is reserved for wire boundaries, which
`SimClient` already owns). Tests in `tests/unit/test_search.py` and
`tests/integration/test_search.py`, wired into `.github/workflows/test.yml`.

```python
@dataclass(frozen=True)
class SearchConfig:
    k_actions: int = 6                 # top-k joint actions per player (vibes 9.7)
    max_chance_children: int = 5       # chance fan-out cap K (vibes 9.8)
    expansion_budget: int = 24         # PUCT expansions per search (vibes 9.2)
    cfr_iters_per_expansion: int = 10  # CFR+ updates between expansions (vibes 9.10)
    c_puct: float = 2.0                # PUCT exploration constant (vibes 9.10)
```

## 12. Open questions and Phase-4 doors

All deferred deliberately; none blocks Phase 1.

1. **Chance bucketing** — replace uniform-sampled worlds with real-probability strategic buckets;
   needs `SimClient` structured chance outcome (`sim-client.md`; vibes 9.3).
2. **Belief range + MCCFR** — Phase 4 un-pins the delta belief: candidate tokens with belief
   weights < 1, worlds sampled from the conditional sampler, per-world CFV vector. Search's tree
   gains a per-world traversal; `BeliefModel` activates (`repo-architecture.md` §3.4).
3. **Vector value head** — the scalar becomes one CFV per candidate; one forward yields every CFV
   the regret update needs (the three-tier split, `search-nn-interface.md` §7).
4. **Action abstraction quality / top-k local maxima** — fixed top-k selects only the actions
   the CVPN's current policy prior already favors, which converges fast but is liable to get
   stuck in local maxima (the network never sees actions it never searches). Mitigations to
   evaluate: epsilon-greedy slot injection (reserve 1–2 top-k slots for uniformly-sampled legal
   actions), progressive widening (start narrow, widen with visit count), periodic full-width
   audit searches, or adapting `k` to how sharp/flat the prior is (vibes 9.7).
5. **Budgets** — `expansion_budget`, `cfr_iters_per_expansion`, `c_puct`, `k`, `K` are all
   empirical; the values here are starting guesses, not commitments (vibes 9.2, 9.8, 9.10).

## 13. Exit criteria for Phase 1 (what "done" means)

From `gt-cfr-theory.md` §14 and `agent/overview.mdc`: (1) **infrastructure proven end-to-end** —
a full self-play game where every decision is one `search()` call runs to a terminal, reseeding
each step, encoding, masking, and emitting training tuples; and (2) **feasibility shown** — a
from-random agent self-improves into a semi-coherent battler that **reliably beats earlier
versions of itself** (even while "cheating" with perfect information). When both hold, un-pin the
imperfect-information machinery (§12) and pivot to the Phase-4 north star.
