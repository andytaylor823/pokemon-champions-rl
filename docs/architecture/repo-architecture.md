# Repository Architecture — Modules, Interfaces, and Seams

**Status:** *Tentative.* A conceptual map of how the production codebase should decompose, written before most of it exists. Meant to be argued with, sharpened, and revised — not obeyed.
**Audience:** The maintainer and any future coding agent building the agent.
**Scope:** The **Pokémon production system only**. This is a *module map* (responsibilities, interfaces, and where one implementation can be swapped for another), **not** a directory layout and **not** an implementation spec.

**Companion docs (read for their domains; not duplicated here):**
- `docs/architecture/gt-cfr-theory.md` — the algorithm (CFR, GT-CFR, the two-loop model, training targets). The ground truth; trusted deeply.
- `docs/architecture/state-encoding.md` — token anatomy, Transformer, belief-weighted candidates, value-head designs.
- `docs/architecture/search-nn-interface.md` — when the NN is called in search, caching, deal sampling, the three-tier split, chance bucketing.
- `.cursor/rules/agent/overview.mdc` — milestones, action space, imperfect-info inventory.

> **A note on trust.** `gt-cfr-theory.md` was produced with Opus 4.8 and is trusted deeply. `state-encoding.md` and `search-nn-interface.md` were produced with Opus 4.6 — trusted *directionally*, but their specific numbers (token counts, `d_model`, K, ms estimates, sample counts) are tentative defaults to tune, not commitments. Numbers in this doc inherit that caveat and are marked accordingly.

---

## 0. Three decisions this map assumes

These were settled before writing; everything below follows from them.

1. **Production is Python-centric.** Search, encoding, the neural net, self-play, and training all live in Python. The Pokémon Showdown battle engine (`@pkmn/sim`, TypeScript) is the *one* external service, reached through a single Python seam (`SimClient`). **Battle rules are never reimplemented in Python** — Showdown is the sole source of truth for game mechanics.
2. **The poker toys are an independent learning artifact.** `toy_examples/{kuhn,leduc}_poker/` are reference implementations of GT-CFR on solved games. They are **not** coupled to the production system, share no code with it, and are out of scope for this map. The production engine is designed fresh; the toys inform it only as worked examples (e.g. the `SearchNode` / regret-table shapes, the CFR+ math, the value-head-as-vector idea).
3. **Phase 1 is the degenerate case of Phase 4.** Every module is designed so the perfect-information pit-stop (Phase 1: MCTS, scalar value, belief = a delta) is the *same schema* as the GT-CFR north star (Phase 4: belief-weighted candidates, vector CFV value head), with the Phase-4 pieces pinned to trivial values. We never build the encoder, net, or search twice — we swap heads and un-pin beliefs. (See `gt-cfr-theory.md` §14, `state-encoding.md` §10.)

---

## 1. Vocabulary (the lens this doc uses)

Borrowed from the *improve-codebase-architecture* skill, applied to this project:

- **Module** — anything with an interface and an implementation (a function, class, package, or a whole service like the sim worker).
- **Interface** — everything a caller must know: types, tensor shapes, invariants, error modes, ordering, units. *The interface is the test surface.*
- **Implementation** — the code behind the interface; ideally swappable without callers noticing.
- **Depth** — leverage at the interface. A **deep** module hides a lot behind a small interface (good); a **shallow** one's interface is nearly as complex as its implementation (a smell).
- **Seam** — where an interface lives: a place behavior can be changed without editing callers in place. **One adapter = a hypothetical seam; two adapters = a real seam.** The project has several *real* seams because it ships two implementations of the same role over its lifetime (MCTS→GT-CFR, MLP→Transformer, scalar→vector head, clustering→conditional sampler).
- **Locality** — what the maintainer gets from depth: change, bugs, and knowledge concentrated in one place. The whole Python↔TS boundary having exactly one home (`SimClient`) is a locality win.
- **Deletion test** — imagine deleting the module. If complexity vanishes, it was a pass-through. If the same complexity reappears smeared across N callers, the module was earning its keep.

---

## 2. The map at a glance

Data flow, not a directory tree. Read top-to-bottom for one decision (inner loop), and as a cycle for training (outer loop).

```
                         ┌─────────────────────────── OUTER LOOP (persistent) ───────────────────────────┐
                         │                                                                                │
   Showdown engine       │   SimClient ──► Encoder ──► CVPN (net) ──► Search ──► Self-play ──► Replay ──► Trainer
   (@pkmn/sim, TS)       │   (Python      (battle      (policy +      (GT-CFR/   (full games   buffer    (SGD;
        ▲                │    seam over    obs →        value heads)   MCTS;      via Search;   (FIFO     publishes
        │ handles +      │    the engine)  tensors)         ▲          inner      emits          window)  checkpoints)
        │ choice strings │        ▲            ▲            │          loop)      training                   │
        └────────────────┘        │            │      BeliefModel        │        tuples)                    │
                                   │            │      (meta-priors:      │                                  │
                                   └────────────┴───── candidates +  ◄────┘   checkpoints ───────────────────┘
                                       (β: public belief state)    deal sampling)   reloaded by Self-play
```

- **Inner loop (ephemeral):** at one decision, `Search` builds a tree rooted at the current state, using `SimClient` for transitions, `BeliefModel` for opponent worlds, `Encoder`+`CVPN` for leaf evaluation. It outputs a strategy, samples a move, and throws the tree away.
- **Outer loop (persistent):** `Self-play` runs full games (each decision = one inner loop), emits training tuples into `Replay`, `Trainer` learns and publishes checkpoints, workers reload them. Generation and training run concurrently.

The two roles that the whole architecture is organized to keep swappable: **the search algorithm** (MCTS ↔ GT-CFR) and **the value head** (scalar ↔ CFV vector). Almost everything else is shared scaffolding designed to survive that swap.

---

## 3. Modules

Each entry: **Responsibility · Interface · Implementations (Phase 1 → Phase 4) · Seam · Depth.**

### 3.1 `SimClient` — the battle-engine seam

**Responsibility.** Provide game mechanics — legal actions, state transitions (including chance resolution), terminal detection and payoff, and *cloning/forking* — without the rest of the system knowing Showdown, TypeScript, or the process boundary exists. This is the single home of the Python↔TS boundary.

**Interface (Python-side, handle-based — the load-bearing design choice).** Heavy `Battle` objects stay inside a per-worker Node subprocess; Python holds opaque handles and exchanges only compact, **engine-native** messages (structured snapshots + Showdown choice strings — no tensors, no protocol-log text). *Settled by interview; see the SimClient plan for the full rationale.*
```
new_battle(team_a, team_b, seed) -> (handle, StateView)   # teams as structured sets; Node packs+validates
open_search(from_handle=None) -> session                  # snapshot the live battle into a search-owned root
step(h, choices: dict[Side,str], seed) -> StepResult      # clone parent, reseed to `seed`, resolve
view(h) -> StateView
release(h); close_search(session); close()
   StateView   = { phase: teamPreview|move|forceSwitch|terminal,
                   to_move: [sides that must act],
                   legal:  { side -> raw Showdown request (move slots, targets, switches, canMega) },
                   snapshot: <omniscient structured state: sides→pokemon, field, side conditions>,
                   terminal: bool, utility: {p1: ±1}|None }
   StepResult  = { child: handle, view: StateView,
                   outcome: <what randomness fired this step, so Search can bucket> }
```
Invariants callers rely on: **handles are immutable snapshots** — `step` clones internally and returns a *new* child handle, leaving the parent steppable with other choices/seeds. **The seed is mandatory** — cloning copies the PRNG state, so stepping a clone without reseeding repeats the same outcome; the search supplies a fresh seed per chance sample (§3.5). The state tells you *which side(s)* must act (simultaneous moves, unilateral forced switches, team preview all handled uniformly). Every clone in a search belongs to its `session` and is freed in one shot by `close_search` (it cannot leak); `release` exists for capping peak memory mid-search. v1 snapshots are **omniscient** — correct for self-play / perfect-info Phase 1; redaction/belief is a later encoder-layer concern.

**Implementations.** Built on the **synchronous** engine API (`new Battle` + `setPlayer` + `makeChoices`) plus `State.serializeBattle`/`deserializeBattle` for cloning (the PRNG seed is part of the serialized state) — *not* the async `BattleStream`. The installed engine is the full `pokemon-showdown` package (the docs' `@pkmn/sim` reference is stale; same engine). Transport is a **persistent Node subprocess per worker, JSON-lines over stdio**. `sim/src/battle-runner.ts` (full-game `Strategy` runner) is retained only for replay export; its `packTeam`/`validateStatPoints` (66/32 stat points) are reused by the worker. **Status (built & tested):** the clone/reseed/latency spike, `sim/src/sim-worker.ts` (stdio worker), and the Python `SimClient` all exist and are covered by automated suites — `sim/tests/` (Vitest: worker dispatch units, clone-determinism/reseed integration) and `tests/integration/test_sim_client.py` (pytest: full Champions doubles game driven to terminal, re-seeding every step, then scratchpad cleanup verified), both wired into CI (`.github/workflows/test.yml`). **Remaining:** (a) *snapshot completeness* — the v1 snapshot drops volatile/status turn-counters, the Protect counter, and Substitute HP, all of which the Encoder's token anatomy wants; (b) a *structured* chance `outcome` — `step` currently returns the raw protocol-log delta, not the structured "what randomness fired" the interface above promises (so chance-bucketing in Search would have to parse log text today).

**Seam.** The `SimClient` protocol. Realistically a **hypothetical seam** (one adapter: the `pokemon-showdown` worker; alternatives like poke-engine are rejected for lacking doubles/VGC correctness). It earns its place anyway via *locality*: every serialization, IPC, handle-lifecycle, and snapshot-introspection concern is trapped here. Deletion test: delete it and the boundary smears into Search, Encoder, and Self-play. The transport (stdio JSON) is a nested swap point — socket/MessagePack can replace it behind the same Python interface if profiling demands.

**Depth.** Deep — tiny interface, enormous hidden complexity (the entire battle engine + cross-language boundary + fork pool).

> **Cost characteristics (measured by the Milestone-0 spike, `sim/src/clone-spike.ts`).** Dex load is a *one-time* per-process cost, amortized by keeping the worker alive. A fork (serialize → JSON round-trip → deserialize) **plus one step ≈ 1.6 ms**, **~72 KiB/clone** (~30 KiB serialized), and clone determinism + reseed variance are both verified. So a search tree of dozens of concrete states is ~tens of ms of fork compute and a few MiB — comfortable within a 60 s real-play turn. The IPC round-trip is on top (small messages only, by the handle design). *Self-play throughput* is where the per-fork cost multiplies — mitigate with one worker per parallel actor and, if needed, batching across the boundary.

### 3.2 `Encoder` — battle observation → tensors

**Responsibility.** Translate everything strategically relevant *and legally knowable* into the NN's input bundle. A feature omitted is a permanent blind spot; hidden ground truth must never leak in (encode the info set / public belief state, never the history). The NN is its only consumer. (Full design: `state-encoding.md`.)

**Interface.**
```
encode(observation: PublicObservation,
       perspective: Player,
       belief: Belief | None) -> ObsBundle
```
`ObsBundle` is a *named bundle of tensors of different shapes* (dict/dataclass/TensorDict), not one rectangular array:
```
entities   : float[12, F]     # 6 yours then 6 theirs; to-move player first
ids        : { species:int[12], ability:int[12], item:int[12], moves:int[12,4] }
field      : float[Ff]        # weather/terrain/TR/gravity + one-hot turn counters
sides      : float[2, Fs]     # tailwind / screens / hazards / mega-used, per side
scalars    : float[Fg]        # turn #, phase, whose decision
belief     : {...} | None     # Phase 4 only: per-opponent-candidate weights
action_mask: bool[A]          # legal-action mask over the canonical index (§3.6)
```
Invariants: keep the entity axis (`[12, F]`, do not flatten across Pokémon); the field is a *peer token*, not broadcast onto every entity; **encode features, never opaque set ids** (so novel sets are points in feature space, not untrained slots). Probability lives *between* tokens (belief weights), never inside one.

**Implementations.** *Phase 1:* perfect info — 14 fixed tokens (CLS + field + 6 + 6), belief = delta (weight 1.0). *Phase 4:* variable token count — opponent slots become *multiple candidate tokens* with belief weights < 1.0 supplied by `BeliefModel`. The schema is identical; Phase 4 just stops pinning belief to a delta.

**Seam.** The `ObsBundle` schema is a **real seam** — it sits unchanged between (a) two producers (perfect-info state vs belief state) and (b) two consumers (flat-MLP vs Transformer backbone). A secondary, deferred seam is the **tokenization of categoricals** (one-hot/multi-hot vs learned embeddings vs hybrid): it changes only `F` and the projection width, never the schema, so it is a tuning knob behind the seam, not an architecture change.

**Depth.** Deep — collapses messy battle facts into one stable contract that both the net and the trainer code against.

### 3.3 `CVPN` — the neural network (counterfactual value-and-policy net)

**Responsibility.** Map an `ObsBundle` to a **policy prior** (search guidance) and **value(s)** (leaf evaluation). One shared backbone, two heads.

**Interface.**
```
forward(obs: ObsBundle) -> (policy_logits: float[A], value: Value)
```
where `Value` is the head seam:
- *Phase 1:* scalar in `[-1, 1]` (read from CLS).
- *Phase 4:* a **vector of counterfactual values, one per opponent candidate** (read per candidate token). One forward pass yields every CFV the regret update needs.

To let Search exploit caching, the net also exposes its **three tiers separately** (an implementation detail surfaced deliberately — see `search-nn-interface.md` §7):
```
backbone(obs)            -> embeddings[n_tokens, d_model]   # EXPENSIVE; once per node expansion
policy_from(embeddings)  -> policy_logits[A]                # cheap; once per expansion
value_from(embeddings, deal) -> cfv: float                  # cheap; once per sampled deal (batchable)
```

**Implementations.** *Backbone:* a flat MLP first (ship the loop fast), then a **Transformer over entity+field tokens** behind the same `backbone()` interface (`d_model≈128`, ~4 heads, ~3 layers — *all tentative*). *Heads:* scalar → per-candidate vector. The **backbone is byte-for-byte shared across phases; only heads change.**

**Seams.** Two **real seams**: backbone (MLP ↔ Transformer) and value head (scalar ↔ CFV vector). Both have two planned adapters, so both are real, not hypothetical.

**Depth.** Deep on the backbone; the heads are intentionally thin so they can be swapped cheaply.

### 3.4 `BeliefModel` — meta-priors / opponent uncertainty

**Responsibility.** Answer "what hidden sets could the opponent have, and how likely is each?" — supplying both the **candidate support** the encoder turns into tokens and the **concrete joint deals** the search traverses. Maintains belief as observations accumulate. (Design: `state-encoding.md` §7,§9; `search-nn-interface.md` §5,§6.)

**Interface.**
```
candidates(species, observed, teammates) -> [(SetSpec, prob)]   # top-K real sets; feeds Encoder tokens
sample_deal(public_state, observations) -> JointOpponentConfig   # one concrete world; feeds Search/MCCFR
update(belief, revealed) -> belief                               # Bayesian narrowing on reveal
```
Invariants: deals are drawn from the **joint** distribution (respects item clause and team-building correlations — never independent per-slot draws); the agent's own state is known and is *not* part of a deal; "live support" shrinks as info is revealed and may *grow* a candidate back in once evidence makes it plausible.

**Implementations.** *Legacy / reference:* the clustering-into-archetypes pipeline (`src/meta_priors/clustering.py`, `auto_cluster`, `ArchetypeSummary`) + the Streamlit explorer (`src/meta_priors/app.py`). *Target (Phase 4):* a **conditional sampler over real tournament sets** — top-K by empirical frequency, conditioned on species + revealed info + teammates, auto-refreshed from scrapes. Smoothing/backoff for sparse conditioning (kernel counting / factored fallback / small learned model) is the **main open problem** here.

**Seam.** The "supply candidates + sample joint deals" interface is a **real seam** (clustering today, conditional sampler planned = two adapters). Phase 1 degenerates it to a delta (one candidate per slot, weight 1.0), so Phase 1 needs no belief model at all — it's off the critical path until Phase 4.

**Depth.** Deep — hides scraping, legality correction, conditioning, and correlation structure behind three small calls.

**Supporting sub-system — the data pipeline** (`src/meta_priors/legality.py`, `scrape_learnsets.py`, `download_sprites.py`, `data/`, `scripts/check_team_legality.py`). Responsibility: turn Limitless tournament JSON + scraped learnsets into clean, legal, queryable set data. Its own seam is the on-disk `data/` layout + the legality checker. Locality note from `meta-priors/data-pipeline.mdc`: Limitless does **not** validate legality, so cached tournament JSON is **manually corrected** after download — *never re-download a file that already exists* or you lose the fixes. (Flag to verify: `legality.py` references a `data/legal/moves.txt` that isn't on disk; confirm it imports cleanly.)

### 3.5 `Search` — the inner-loop planner (the project's primary seam)

**Responsibility.** At one decision point, produce a near-optimal **strategy** over legal joint actions by building and evaluating an ephemeral tree. This is the component the entire architecture is shaped to keep swappable.

**Interface.**
```
search(root_obs, sim: SimClient, net: CVPN, belief: BeliefModel, budget) -> SearchResult
   SearchResult = { strategy: dict[InfoSet, dict[Action, float]],   # σ̄ at the root; sample your move from it
                    value: Value,                                   # search-refined CFV(s) — the value training target
                    policy_target: ... }                           # σ̄ — the policy training target
```
The caller samples one joint action from `strategy`, plays it via `SimClient`, and **discards the tree**. The `SearchResult` *is* the training tuple `(β, v_search, σ̄)` (§3.7).

**Implementations.** *Phase 1–3:* MCTS/PUCT with scalar value backups; Phase 3 adds determinization (sample a world from `BeliefModel`, search as if perfect-info) — knowingly accepting strategy fusion as the measured ceiling. *Phase 4:* **GT-CFR** — interleave CFR+ regret-update traversals over the grown tree (cached CVPN values at frontier leaves, real payoffs at terminals) with PUCT-guided expansion; output the average strategy σ̄. Internals (ephemeral `SearchNode` tree, per-info-set regret/strategy-sum tables, MCCFR deal sampling, the three node types: terminal / frontier-leaf / internal) are **implementation, not interface** — they stay local to this module.

**Sub-seams worth naming** (all internal to Search): **chance handling** (sample-one vs enumerate-buckets, ~3–5 buckets per chance node — *tentative*), **expansion policy** (which frontier node to grow), and the **NN-call boundary** (NN is hit only on expansion; CFR+ iterations over expanded nodes are pure arithmetic). NN forward passes per search ≈ number of expanded nodes (~20–50 — *tentative*), not number of iterations or deals.

**Seam.** `search(...)` is **the** real seam of the project: "swap the search, reuse everything else" (`gt-cfr-theory.md` §14, `mcts-vs-alphazero` transcript). Two adapters (MCTS, GT-CFR) make it real, not hypothetical.

**Depth.** The deepest module — vast hidden complexity behind a single call, and the largest source of leverage.

> **Phase-1 search decision (recorded in `docs/plans/search.md`):** Phase 1 builds **GT-CFR directly in the perfect-information regime** — the MCTS pit-stop named in the milestones is skipped, because simultaneous moves make even perfect-info Pokémon an imperfect-information (matrix) game, and the toys + CVPN are already GT-CFR-shaped. See `vibes-decisions.md` §9.1, §9.6–§9.10.

> **Forced-node collapse (flagged from the SelfPlay design).** Forced decisions (every acting side has one legal action) carry no strategic choice; the SelfPlay loop skips them entirely (no CVPN/CFR/tuple) and Search should **collapse** them in-tree rather than instantiate degenerate decision nodes. See `docs/plans/search.md` §8/§12, `docs/plans/self-play.md` §2.2/§11, `vibes-decisions.md` §8.7.

### 3.6 The action space (a shared contract, not a module)

A **stable canonical index `0..A`** over all joint actions, with legality expressed as a mask (choice-lock, disable, taunt, no-PP, forced-switch all flip mask bits). Three modules code against it: `Encoder` (emits `action_mask`), `CVPN` (policy head width `A`), and `SimClient` (translates an index ↔ a Showdown choice string like `"move heatwave 1, move protect"`). The index↔choice-string translation lives inside `SimClient`. Size ≈ ~100 joint actions/turn (*tentative*; `agent/overview.mdc` §Action Space). Open sub-decision: a single flat joint head vs **per-Pokémon factored heads**.

### 3.7 `SelfPlay` — outer-loop data generation

**Responsibility.** Play full self-play games, each decision driven by `Search`, and emit training tuples.

**Interface.**
```
run(checkpoint, config) -> stream[TrainingTuple]
   TrainingTuple = (β: encoded public-belief state,
                    v_search: Value,            # search-refined value target
                    σ̄: policy target)
```
This tuple is the GT-CFR analogue of AlphaZero's `(s, z, π_MCTS)` and is a **stable seam** between inner and outer loops.

**Design (fleshed — `docs/plans/self-play.md`).** The Phase-1 SelfPlay design is complete (not yet implemented). Key decisions from that grilling:
- **Generator seam.** `run(net, matchup_source, config) -> Iterator[TrainingTuple]`, flushing one game's worth of tuples at each terminal — SelfPlay knows nothing about `ReplayBuffer`.
- **Value target = search-refined value (bootstrapping); each tuple *also* tagged with the final result z** (a logged side-signal in Phase 1, requiring per-game buffering). Tuple stores the encoded `ObsBundle` (β) + sparse σ̄. *(vibes 8.4, 8.6)*
- **Forced decisions are fully skipped** — no `search()`, no CVPN, no CFR, no tuple — in the loop now, and (flagged to `search.md` §8/§12) collapsed inside the search tree. Emission rule: one full tuple per side that had a genuine choice. *(vibes 8.7)*
- **Team source = a pluggable `MatchupSource`**, defaulting to a deliberate **validation curriculum** of fixed matchups of increasing complexity (Stage 0 = all-Fire vs all-Grass / 2 moves … Stage 3 = balanced teams); per-stage from-scratch (primary) + warm-start (side experiment). *(vibes 8.8)*

**Implementation.** Single-process, sequential games first (in-process net). Later: worker processes, each owning a `SimClient` (its own Node worker) and reading the latest `CVPN` checkpoint (directly, or via a batched inference server). Parallelism mechanism (Ray vs plain multiprocessing) is an implementation choice, not part of the interface *(vibes 8.9, 11.3)*.

**Depth.** Medium — mostly orchestration; its value is concentrating the "play a game, harvest targets" protocol in one place.

### 3.8 `ReplayBuffer`, `Trainer`, `Evaluation`

- **`ReplayBuffer`** — fixed-capacity FIFO sliding window. **Design fleshed: `docs/plans/replay-buffer.md`.** Interface: `add(iterable[TrainingTuple])`, `sample(n) -> list[TrainingTuple]`, `__len__`, plus `save`/`load`. Implementation is a `deque(maxlen=capacity)` of tuples (per-tuple FIFO eviction; capacity in tuples); sampling is uniform, without-replacement, seeded; `sample` raises on over-ask (the warmup threshold lives in the Trainer, not here). Deliberately **thin** — it returns raw tuples; the **Trainer** owns collation (`collate_obs_bundles`), sparse→dense σ̄, and device placement. `save`/`load` (torch.save) persist the deque + RNG state behind a **fail-loud schema-fingerprint guard** (a stale-schema reload raises rather than silently corrupting training). Plain constructor args (`capacity`, `seed`) — the "typed config via Pydantic" note below scopes to the **Trainer**, not the buffer. A fresh buffer is built per curriculum stage (no cross-stage bleed). Stalest tuples (from weak early checkpoints) age out; data is not wholesale discarded per generation. **Seam-type note:** `TrainingTuple`/`SparsePolicy`/`TupleMeta` are extracted from `self_play.py` into a shared `src/training_types.py` that `self_play`, `replay_buffer`, and `trainer` all import.
- **`Trainer`** — minibatch SGD: `L = ‖v̂−v_search‖² + CE(π̂, σ̄) + λ‖θ‖²`. Interface: `train_step(batch)`, plus **checkpoint publish**. The checkpoint store is the seam decoupling training from generation (they run concurrently; weights are never reset — contrast Deep CFR). Typed config via Pydantic (the toys' `TrainingConfig` pattern is a good template, not a shared dependency).
- **`Evaluation`** — **Built** (`src/evaluation.py`, `scripts/evaluate.py`; full design: `docs/plans/evaluation.md`). Read-only measurement module — owns no SimClient lifecycle, reimplements no game rules or search logic. Three entry points: `curriculum_report(net, matchup, sim, *, config) -> CurriculumReport` (one net drives both sides of a lopsided matchup; tracks favored-team win-rate and win speed across checkpoints — the Phase-1 headline metric); `head_to_head(agent_a, agent_b, matchup, sim, *, config) -> HeadToHeadReport` (two distinct checkpoints; each runs its own independent search on its own side, no side-swap); `fitness(checkpoint_path, opponents, matchup, sim, *, config) -> dict[label, HeadToHeadReport]` (satisfies the `fitness(checkpoint) -> metric` seam shape). Three `Agent` implementations: `SearchAgent` (greedy over σ̄, 1-slot handle cache for one search-per-turn in curriculum mode), `PolicyAgent` (greedy over raw policy logits, no tree), `RandomAgent` (uniform over legal mask, the floor baseline). Full Elo ladder and approximate/local best-response deferred.

---

## 4. Cross-cutting seams

- **The Python↔TS boundary** — exists in exactly one place (`SimClient`). Nothing else knows TypeScript exists. This is the single biggest locality decision in the repo.
- **`β`, the public belief state** — the shared currency of `Encoder` output, `Search` leaf input, `BeliefModel`, and the training tuple. Get this representation right once; four modules depend on it.
- **The action index/mask** — shared contract across `Encoder`, `CVPN`, `SimClient` (§3.6).
- **The checkpoint store** — decouples `Trainer` from `SelfPlay`; lets generation and training run continuously and independently.
- **The phase pin** — a single config switch (`perfect_info` / belief = delta / scalar-head vs vector-head) flips the whole system between Phase 1 and Phase 4, because every module was built with Phase 1 as the degenerate case.

---

## 5. Phase 1 → Phase 4 as a sequence of seam-swaps

Nothing below is a rewrite; each is swapping one adapter behind a stable interface.

| Module | Phase 1 (perfect-info pit-stop) | Phase 4 (GT-CFR north star) | Interface change? |
|---|---|---|---|
| `SimClient` | clone/step/legal/terminal | *unchanged* | none |
| `Encoder` | 14 fixed tokens, belief = delta | + opponent candidate tokens, belief < 1 | none (same `ObsBundle`) |
| `CVPN` backbone | flat MLP, then Transformer | *same Transformer* | none |
| `CVPN` value head | scalar | per-candidate CFV vector | **head swap** |
| `BeliefModel` | unused (delta) | conditional sampler: candidates + joint deals | activated |
| `Search` | MCTS / determinized IS-MCTS | GT-CFR (CFR+ + vector leaves) | **algorithm swap** (same `search()`) |
| `SelfPlay`/`Replay`/`Trainer` | scalar value target | vector value target | tuple value-shape only |
| `Evaluation` | win-rate vs random / prior selves | + approximate exploitability | metric swap |

Exit criteria for leaving Phase 1 (from `agent/overview.mdc` §Milestones and `gt-cfr-theory.md` §14): (1) infrastructure proven end-to-end (simulate games + theoretical turns, encode, mask, run the loop) and (2) a from-random agent reliably beats earlier versions of itself. Then pivot hard to Phase 4.

---

## 6. Open decisions (seams not yet resolved)

These are genuine forks the map deliberately leaves open; they belong in design discussion, not silent defaults.

1. **Per-slot vs joint-team value head** (`state-encoding.md` §12.3 — "the hardest open question for Phase 4"). Per-slot independence (lean on attention) vs the proposed three-tier split where `value_from(embeddings, deal)` concatenates the sampled deal's per-slot embeddings. Affects `CVPN` internals and the training tuple, not the public `search()` interface.
2. **Belief sampler smoothing/backoff** (`state-encoding.md` §9) — how to condition on 3+ constraints when exact matches run out.
3. **Action head: flat joint vs per-Pokémon factored** (§3.6). *→ Resolved (`cvpn.md` D4): phase-split heads (team-preview + move-phase) internally, assembled into a unified flat `[A]` masked output externally.*
4. **Tokenization of categoricals** (one-hot vs embeddings vs hybrid) — tuning behind the `ObsBundle` seam. *→ Resolved (`encoder.md` Decision 1 / `cvpn.md` D8): hybrid — IDs for high-cardinality (species/ability/item/moves), one-hot for low-cardinality; CVPN owns the embedding tables.*
5. **Search/MCCFR budgets, chance-bucket granularity, K candidates per slot** — all empirical; the Opus-4.6 docs give starting guesses, not commitments.
6. **Self-play parallelism** (Ray vs multiprocessing) and **inference serving** (in-worker net vs batched server).
7. **The runtime-boundary benchmark** (§3.1) — validate the handle-based Node worker's per-step latency before committing to Python-orchestrated search at self-play scale.
8. **Action-space canonicalization** (resolved, deferred; decided in the CVPN grilling). Collapse symmetric orderings to one canonical action — team-preview lead-pair and bench-pair order, and the double-faint forced-switch order, are all immaterial (A+B ≡ B+A). Team preview `360 → C(6,2)·C(4,2) = 90`; `A → ~819`. Lives in `action_space` (canonical index + deterministic canonical→Showdown-string expansion). The `CVPN` is built against `A = 1089` first and absorbs this for free (head widths derive from `action_space` constants). See `cvpn.md`.
9. **Adaptive search compute** (deferred; decided in the CVPN grilling). Only a single-legal-move skip is planned initially (`search.md`); intelligent early-stopping (e.g. σ̄-convergence) and dynamic per-turn time allocation are deferred, and KataGo-style Playout Cap Randomization for self-play throughput is deferred (`self-play.md`). TODO: study AlphaGo Lee-Sedol-match & Lc0 dynamic time allocation before building the 60 s-turn time manager.

---

## 7. Observed drift / cleanup flags

Not architecture, but worth recording so future reviews don't trip on it:

- **`pyproject.toml` carries abandoned deps.** `stable-baselines3`, `ray[rllib]`, `gymnasium`, `tensorboard` reflect an earlier PPO/RLlib plan that the rebalanced strategy dropped (the `mcts-vs-alphazero` transcript explicitly calls PPO "a detour"). PPO is **not** part of this architecture. Prune when convenient (keep `ray` only if chosen for §6.6).
- **`sim/` now exposes the search seam (done — this note was stale).** An earlier revision said the `clone`/`step`/handle interface was "the first thing to build"; it is now built and tested. `sim/src/sim-worker.ts` implements the handle protocol and `src/sim_client.py` wraps it, covered by passing Vitest + pytest suites (`sim/tests/`, `tests/integration/test_sim_client.py`). `battle-runner.ts` (full-game `Strategy` runner) is retained only for replay export. The remaining SimClient work is snapshot completeness + structured chance outcome, not the seam itself (see §3.1).
- **`research/notes.md` predates the rebalance** — it frames PPO as "Phase 1 workhorse" and AlphaZero as "Phase 2." Treat `gt-cfr-theory.md` §14 + `agent/overview.mdc` as the current phasing; `research/notes.md` is still the right home for sim-engine and data-source rationale.

---

*This is a living, tentative map. The next step is to grill any one module's interface in depth (most likely `SimClient` or `Search`), record load-bearing rejections as they surface, and let the directory layout fall out of the modules once their interfaces firm up.*
