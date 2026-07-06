# Volatile Encoding Audit

**Status:** Reference — catalogues every volatile the Showdown engine can emit and documents encoding decisions.
**Last updated:** July 2026 (initial audit)

---

## 1. Intentional gap: mid-turn decision volatiles

The encoder captures state at the **start-of-turn decision point** — the moment the agent
must commit to joint actions for both active Pokemon simultaneously. This means volatiles
that only exist *during resolution* (within a single turn) are intentionally excluded.

### The mid-turn information gap

There is a small class of situations where mid-turn information could theoretically affect
a decision node:

> Player A has Pokemon X + Y in play. Opponent has Pokemon C + D.
> C uses Helping Hand and acts first. Then X uses U-Turn (a pivot move).
> X now has a **mid-turn decision node** (choosing which bench Pokemon to switch in),
> and technically the player knows that a super-charged Helping Hand attack is incoming —
> this information could affect the switch-in choice.

This pattern applies whenever:
1. A pivot move (U-Turn, Volt Switch, Flip Turn, Parting Shot) creates a mid-turn
   forceSwitch decision node
2. Some other action already resolved this turn has set a single-turn volatile on a
   Pokemon still in play (e.g. Helping Hand boosting an ally, Follow Me redirecting)

**Why we intentionally omit this:** Including all mid-turn state that could influence these
pivot-move decisions would require encoding the *full turn resolution history* up to the
current sub-action. This massively expands scope:
- We'd need to track resolution order, which actions already fired, and their side effects
- The encoder would need a variable-length "action history within this turn" representation
- The neural network architecture would need to handle this variable context
- In practice, these situations are rare (pivot moves are uncommon in VGC doubles, and the
  intersection of "pivot move user + prior action set a relevant single-turn volatile" is
  rarer still)

The expected EV loss from not conditioning on mid-turn info is negligible compared to the
architectural complexity cost. This gap can be revisited in Phase 4+ if agent strength
reaches a ceiling that this edge case could explain.

### Volatiles affected by this gap

These Tier 3 single-turn volatiles exist only during resolution and are excluded from
encoding because they cannot influence the main start-of-turn decision:

| Volatile | Reason excluded |
|----------|----------------|
| `helpinghand` | Applied to ally this turn; expires before next decision |
| `followme` / `ragepowder` / `spotlight` | Redirection; single-turn |
| `protect` / `banefulbunker` / `kingsshield` / `silktrap` / `obstruct` / `burningbulwark` | Protection; the `stall` counter already covers consecutive-use penalty |
| `endure` | Single-turn survival; mid-resolution only |
| `focuspunch` / `shelltrap` / `beakblast` | Mid-resolution trigger commitments |
| `roost` | Single-turn type change during resolution |
| `powder` | Single-turn Fire nullification |
| `electrify` | Single-turn type change to Electric |
| `flinch` | Applied during resolution, never present at decision time |

---

## 2. Full volatile catalogue

### How the 121 volatile names were derived

The engine adds volatiles via three distinct mechanisms. No single extraction pattern
captures them all — combining all three and deduplicating yields the complete set.

**Mechanism 1 — Move `volatileStatus` field (64 unique):**
Declarative: a move's data object names the volatile it inflicts (e.g. Taunt has
`volatileStatus: 'taunt'`). Extract with:

```bash
cd sim/node_modules/pokemon-showdown
grep -o "volatileStatus: '[a-z]*'" data/moves.ts \
  | sed "s/volatileStatus: '//;s/'$//" | sort -u
```

**Mechanism 2 — Imperative `addVolatile()` calls (43 unique from data/, 3 from sim/):**
Abilities, items, move effect handlers, and sim core code programmatically add volatiles
at runtime. These are NOT declared on the move's data — they only appear as runtime calls.
Examples: `choicelock` (added by `items.ts` for Choice items), `unburden` (added by ability
handler).

```bash
grep -roh "addVolatile('[a-z]*'" data/*.ts sim/*.ts \
  | sed "s/addVolatile('//;s/'$//" | sort -u
```

**Mechanism 3 — Condition definitions in `conditions.ts` (34 entries):**
Top-level named condition handlers define lifecycle hooks (`onStart`, `onEnd`,
`onResidual`, etc.). Some overlap with statuses/weathers (already encoded elsewhere), but
this captures `mustrecharge`, `choicelock`, `lockedmove`, `twoturnmove` which have their
logic defined here.

```bash
grep -E "^\t[a-z]+:" data/conditions.ts \
  | sed 's/:.*//' | tr -d '\t' | sort -u
```

**Why no single pattern suffices:**
- Pattern 1 misses volatiles added imperatively (e.g. `choicelock` is never a
  `volatileStatus` — it's added by `items.ts`)
- Pattern 2 misses volatiles only declared as move `volatileStatus` and never explicitly
  `addVolatile()`'d (e.g. `roost`, `endure`, `magnetrise`)
- Pattern 3 defines handlers that may overlap with statuses/weather, but also captures
  engine-internal volatiles like `mustrecharge` and `lockedmove`

**Combined extraction:**
```bash
cd sim/node_modules/pokemon-showdown
{
  grep -o "volatileStatus: '[a-z]*'" data/moves.ts | sed "s/volatileStatus: '//;s/'$//";
  grep -roh "addVolatile('[a-z]*'" data/*.ts sim/*.ts | sed "s/addVolatile('//;s/'$//";
  grep -E "^\t[a-z]+:" data/conditions.ts | sed 's/:.*//' | tr -d '\t';
} | sort -u
# → 121 unique volatile names
```

### Filtering criteria

From the raw 121 names:
- **Already encoded (18):** `brn`, `par`, `slp`, `frz`, `tox`, `psn` (status one-hot),
  `substitute`, `stall`, `yawn`, `flashfire` (`_volatile_counters`), `raindance`,
  `sunnyday`, `sandstorm`, `hail`, `snowscape`, `deltastream`, `desolateland`,
  `primordialsea` (field encoder weather)
- **Not applicable to Champions VGC (15):** `dynamax`, `maxguard`, `gmaxchistrike`
  (Dynamax banned), `arceus`, `silvally`, `zacian`, `zamazenta` (not in legal dex),
  `healreplacement`, `trapper`, `rolloutstorage`, `futuremove`, `gem`, `leppaberry`,
  `micleberry`, `metronome` (internal engine markers)
- **Remaining to triage (88):** classified into Tier 1 (action-constraining), Tier 2
  (mechanic-altering / multi-turn), and Tier 3 (single-turn / mid-resolution)

### Legend

| Status | Meaning |
|--------|---------|
| **ENCODED** | Explicitly encoded in `_volatile_counters` or another sub-encoder |
| **ENCODED-ELSEWHERE** | Captured by a different sub-encoder (e.g. status one-hot, field encoder) |
| **IMPLEMENTED** | Added in this batch (new binary flag in `_volatile_counters`) |
| **OMITTED** | Deliberately excluded with documented reason |
| **N/A** | Not applicable to Pokemon Champions format or is an internal engine marker |
| **CRASH-SAFE** | In `TestVolatilesDoNotCrashEncoder` — encoder ignores but doesn't crash |

### Full list (alphabetical)

| Volatile | Status | Notes |
|----------|--------|-------|
| `allyswitch` | CRASH-SAFE | Internal marker for Ally Switch resolution |
| `aquaring` | OMITTED | Not common enough in Champions meta |
| `arceus` | N/A | Species not in Champions dex |
| `attract` | OMITTED | Not common enough in Champions meta |
| `banefulbunker` | CRASH-SAFE | Single-turn protection (mid-turn only; `stall` covers penalty) |
| `beakblast` | CRASH-SAFE | Mid-resolution trigger |
| `bide` | CRASH-SAFE | Internal counter for Bide |
| `brn` | ENCODED-ELSEWHERE | Status one-hot (`_status_onehot`) |
| `burningbulwark` | CRASH-SAFE | Single-turn protection |
| `charge` | IMPLEMENTED | Next Electric move is 2x — binary flag |
| `chillyreception` | CRASH-SAFE | Not in Champions dex (Snow Warning switch) |
| `choicelock` | OMITTED | Enforced in legal actions mask; future belief module handles opponent inference |
| `commanded` | OMITTED | Too specific (Tatsugiri/Dondozo interaction) |
| `commanding` | OMITTED | Too specific (Tatsugiri/Dondozo interaction) |
| `confusion` | IMPLEMENTED | May hit self for 2–5 turns — binary flag |
| `counter` | CRASH-SAFE | Internal marker for Counter resolution |
| `curse` | OMITTED | Not common enough in Champions meta (Ghost-type version) |
| `defensecurl` | CRASH-SAFE | Doubles Rollout/Ice Ball damage; neither common |
| `deltastream` | ENCODED-ELSEWHERE | Field encoder weather |
| `desolateland` | ENCODED-ELSEWHERE | Field encoder weather |
| `destinybond` | CRASH-SAFE | Single-turn / niche |
| `disable` | IMPLEMENTED | One move blocked for 4 turns — binary flag |
| `dragoncheer` | CRASH-SAFE | Single-turn crit boost to ally (mid-turn) |
| `dynamax` | N/A | Dynamax banned in Champions |
| `electrify` | CRASH-SAFE | Single-turn type change |
| `embargo` | OMITTED | Not common enough in Champions meta |
| `encore` | IMPLEMENTED | Forced to repeat last move for 3 turns — binary flag |
| `endure` | OMITTED | Not common enough; single-turn (mid-resolution) |
| `flashfire` | ENCODED | Fire moves boosted 1.5x — binary flag (added prior session) |
| `flinch` | CRASH-SAFE | Applied during resolution, never at decision time |
| `fling` | CRASH-SAFE | Internal marker for Fling |
| `focusenergy` | IMPLEMENTED | +2 crit stages — binary flag |
| `focuspunch` | OMITTED | Not common enough in Champions meta |
| `followme` | CRASH-SAFE | Single-turn redirection |
| `foresight` | CRASH-SAFE | Not common / niche |
| `frz` | ENCODED-ELSEWHERE | Status one-hot (`_status_onehot`) |
| `furycutter` | CRASH-SAFE | Internal damage multiplier counter |
| `futuremove` | N/A | Internal engine marker for Future Sight/Doom Desire |
| `gastroacid` | OMITTED | Not common enough in Champions meta |
| `gem` | N/A | Internal engine marker for Gem item consumption |
| `glaiverush` | CRASH-SAFE | Niche; doubles damage taken next turn |
| `gmaxchistrike` | N/A | G-Max banned in Champions |
| `grudge` | CRASH-SAFE | Niche single-turn effect |
| `hail` | ENCODED-ELSEWHERE | Field encoder weather (aliased to snow) |
| `healblock` | IMPLEMENTED | Can't heal for 5 turns — binary flag |
| `healreplacement` | N/A | Internal engine marker |
| `helpinghand` | CRASH-SAFE | Single-turn; expires before next decision |
| `iceball` | CRASH-SAFE | Internal damage multiplier counter |
| `imprison` | IMPLEMENTED | Opponent can't use shared moves — binary flag |
| `ingrain` | OMITTED | Not common enough in Champions meta |
| `kingsshield` | CRASH-SAFE | Single-turn protection |
| `laserfocus` | CRASH-SAFE | Single-turn guaranteed crit |
| `leechseed` | IMPLEMENTED | Drains HP each turn — binary flag |
| `leppaberry` | N/A | Internal engine marker |
| `lockedmove` | IMPLEMENTED | Forced to repeat move (Outrage/Thrash) — binary flag |
| `lockon` | CRASH-SAFE | Niche; next move guaranteed hit |
| `magiccoat` | CRASH-SAFE | Single-turn status reflection |
| `magnetrise` | IMPLEMENTED | Ground immunity for 5 turns — binary flag |
| `maxguard` | N/A | Dynamax banned |
| `mefirst` | CRASH-SAFE | Internal marker for Me First |
| `metronome` | N/A | Internal engine marker for Metronome item |
| `micleberry` | N/A | Internal engine marker |
| `minimize` | CRASH-SAFE | Niche; doubles Stomp/Body Slam/etc damage |
| `miracleeye` | CRASH-SAFE | Niche identification move |
| `mirrorcoat` | CRASH-SAFE | Internal marker for Mirror Coat resolution |
| `mustrecharge` | IMPLEMENTED | Cannot act next turn (Hyper Beam aftermath) — binary flag |
| `nightmare` | CRASH-SAFE | Only works on sleeping targets; rare in Champions |
| `noretreat` | IMPLEMENTED | Cannot switch out + boosts received — binary flag |
| `obstruct` | CRASH-SAFE | Single-turn protection |
| `octolock` | CRASH-SAFE | Niche; not in common Champions usage |
| `par` | ENCODED-ELSEWHERE | Status one-hot (`_status_onehot`) |
| `partiallytrapped` | IMPLEMENTED | Cannot switch + residual damage — binary flag |
| `perishsong` | IMPLEMENTED | Faints in 3 turns unless switched — counter (duration/3) |
| `powder` | CRASH-SAFE | Single-turn Fire nullification |
| `powershift` | CRASH-SAFE | Niche stat swap |
| `powertrick` | CRASH-SAFE | Niche stat swap |
| `primordialsea` | ENCODED-ELSEWHERE | Field encoder weather |
| `protect` | CRASH-SAFE | Single-turn protection (`stall` counter covers penalty) |
| `protosynthesis` | IMPLEMENTED | Highest stat boosted (Paradox ability) — binary flag |
| `psn` | ENCODED-ELSEWHERE | Status one-hot (`_status_onehot`) |
| `pursuit` | CRASH-SAFE | Internal marker for Pursuit |
| `quarkdrive` | IMPLEMENTED | Highest stat boosted (Paradox ability) — binary flag |
| `rage` | CRASH-SAFE | Internal Rage counter |
| `ragepowder` | CRASH-SAFE | Single-turn redirection |
| `raindance` | ENCODED-ELSEWHERE | Field encoder weather |
| `rollout` | CRASH-SAFE | Internal damage multiplier counter |
| `rolloutstorage` | N/A | Internal engine marker |
| `roost` | CRASH-SAFE | Single-turn type change |
| `saltcure` | IMPLEMENTED | Residual damage, extra vs Water/Steel — binary flag |
| `sandstorm` | ENCODED-ELSEWHERE | Field encoder weather |
| `shelltrap` | CRASH-SAFE | Mid-resolution trigger |
| `silktrap` | CRASH-SAFE | Single-turn protection |
| `silvally` | N/A | Species not in Champions dex |
| `slp` | ENCODED-ELSEWHERE | Status one-hot (`_status_onehot`) |
| `smackdown` | IMPLEMENTED | Loses Ground immunity (Flying/Levitate) — binary flag |
| `snatch` | CRASH-SAFE | Single-turn effect steal |
| `snowscape` | ENCODED-ELSEWHERE | Field encoder weather |
| `sparklingaria` | CRASH-SAFE | Internal marker for Sparkling Aria |
| `spikyshield` | CRASH-SAFE | Single-turn protection |
| `spotlight` | CRASH-SAFE | Single-turn redirection |
| `stall` | ENCODED | Consecutive Protect penalty counter (`_volatile_counters`) |
| `stockpile` | CRASH-SAFE | Niche; Stockpile/Spit Up/Swallow |
| `substitute` | ENCODED | Sub HP fraction (`_volatile_counters`) |
| `sunnyday` | ENCODED-ELSEWHERE | Field encoder weather |
| `syrupbomb` | OMITTED | Not common enough in Champions meta |
| `tarshot` | OMITTED | Not common enough in Champions meta |
| `taunt` | IMPLEMENTED | Cannot use status moves for 3 turns — binary flag |
| `telekinesis` | OMITTED | Not common enough in Champions meta |
| `throatchop` | IMPLEMENTED | Can't use sound moves for 2 turns — binary flag |
| `torment` | IMPLEMENTED | Cannot use same move twice in a row — binary flag |
| `tox` | ENCODED-ELSEWHERE | Status one-hot (`_status_onehot`) |
| `trapped` | IMPLEMENTED | Cannot switch out — binary flag |
| `trapper` | N/A | Internal engine marker for trapping source |
| `truant` | CRASH-SAFE | Truant ability loaf turn (not in Champions dex) |
| `twoturnmove` | IMPLEMENTED | Committed to charge turn (Solar Beam, Dig, Fly) — binary flag |
| `unburden` | IMPLEMENTED | Speed doubled after item consumed — binary flag |
| `uproar` | CRASH-SAFE | Niche; prevents sleep for all on field |
| `yawn` | ENCODED | Drowsy countdown (`_volatile_counters`) |
| `zacian` | N/A | Species not in Champions dex |
| `zamazenta` | N/A | Species not in Champions dex |
| `zenmode` | CRASH-SAFE | Niche ability form change |

---

## 3. Summary of encoding decisions

### Newly implemented (this batch): 24 binary flags + 1 counter

| # | Volatile | Type | Rationale |
|---|----------|------|-----------|
| 1 | `trapped` | Binary | Constrains switching — directly affects legal actions |
| 2 | `partiallytrapped` | Binary | Constrains switching + residual damage |
| 3 | `lockedmove` | Binary | Forces move repetition, confusion after |
| 4 | `mustrecharge` | Binary | Cannot act next turn — zero legal moves |
| 5 | `twoturnmove` | Binary | Committed to charge move — no choice next turn |
| 6 | `encore` | Binary | Forced to repeat last move |
| 7 | `taunt` | Binary | Blocks status moves for 3 turns |
| 8 | `disable` | Binary | Blocks one specific move |
| 9 | `torment` | Binary | Cannot repeat same move consecutively |
| 10 | `focusenergy` | Binary | +2 crit stages — alters damage expectations |
| 11 | `charge` | Binary | Next Electric move 2x — alters move choice EV |
| 12 | `throatchop` | Binary | Sound moves blocked 2 turns |
| 13 | `confusion` | Binary | 33% chance of hitting self each turn |
| 14 | `leechseed` | Binary | HP drain + heal each turn |
| 15 | `perishsong` | Counter (÷3) | Turns until faint — critically affects switch timing |
| 16 | `magnetrise` | Binary | Ground immunity for 5 turns |
| 17 | `healblock` | Binary | Cannot heal for 5 turns |
| 18 | `smackdown` | Binary | Loses Ground immunity |
| 19 | `imprison` | Binary | Opponent can't use shared moves |
| 20 | `saltcure` | Binary | Residual damage each turn |
| 21 | `unburden` | Binary | Speed doubled — major speed tier shift |
| 22 | `protosynthesis` | Binary | Stat boost active (Paradox ability) |
| 23 | `quarkdrive` | Binary | Stat boost active (Paradox ability) |
| 24 | `noretreat` | Binary | Cannot switch out + received boosts |

### Previously encoded (5 features): unchanged

| Feature | Type |
|---------|------|
| `substitute` | HP fraction (÷200) |
| `stall` | Counter (÷6) |
| `activeTurns` | Normalized (÷20) |
| `yawn` | Binary |
| `flashfire` | Binary |

### Total `_volatile_counters` width: 5 (prior) + 24 (new) = **29**

### ENTITY_FEATURE_DIM: 67 (prior) + 24 = **91**

(1 hp + 6 stats + 7 boosts + 7 status + 25 nature + 8 moves + 29 volatile + 8 flags = 91)

---

## 4. Omission rationale

| Volatile | Why omitted |
|----------|-------------|
| `choicelock` | Already enforced via legal actions mask (choice-locked mon only sees the locked move as legal). Future belief module will infer opponent Choice items from observable behavior. |
| `focuspunch` | Focus Punch is too uncommon in Champions VGC doubles to justify a feature dimension. |
| `commanding` / `commanded` | Too-specific Tatsugiri/Dondozo interaction; niche team archetype. |
| `endure` | Single-turn survival move, not common in VGC doubles. |
| `gastroacid` | Gastro Acid (suppress ability) too uncommon in Champions meta. |
| `attract` | Attract's 50% failure is unreliable and rarely seen competitively. |
| `curse` | Ghost Curse (25% HP/turn) uncommon in Champions doubles. |
| `ingrain` | Ingrain (root, can't switch, heal/turn) too uncommon. |
| `aquaring` | Aqua Ring (heal 1/16) too uncommon. |
| `telekinesis` | Telekinesis (3 turns) too uncommon. |
| `embargo` | Embargo (can't use items) too uncommon. |
| `syrupbomb` | Syrup Bomb (speed drop/turn) too uncommon in Champions. |
| `tarshot` | Tar Shot (2x Fire weakness) too uncommon in Champions. |
| All Tier 3 | Single-turn / mid-resolution only; see §1 gap documentation. |
