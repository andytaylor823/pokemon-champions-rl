---
name: check-encoding
description: >-
  Check whether a specific Pokemon game mechanic (volatile, side condition, field
  effect, status, action constraint, etc.) is present in the state encoding pipeline.
  If missing, add it to all downstream locations. Use when the user asks "is X
  encoded?", "add X to the encoding", "check encoding coverage for X", or
  "does the encoder capture X?".
---

# Check / Add Encoding Feature

## Purpose

Verify whether a given game mechanic is captured in the state encoding pipeline, end
to end. If it IS present, confirm coverage and report. If it is NOT present, add it at
every downstream location so the encoding stays consistent.

## Pipeline locations (in dependency order)

The encoding pipeline has 7 checkpoints. Work through them top-to-bottom:

| # | File | Search anchor | What lives here |
|---|------|--------------|-----------------|
| 1 | `sim/src/sim-worker.ts` | `// [ENCODING-CHECKPOINT-1] snapshotPokemon` or `snapshotSide` or `snapshotBattle` | Raw engine data extraction — if the mechanic's data isn't captured from the engine, nothing downstream can see it. |
| 2 | `sim/src/types.ts` | The TypeScript interface definitions | Wire-format type (must match Python state_types 1:1). |
| 3 | `src/state_types.py` | `# [ENCODING-CHECKPOINT-3] Pydantic wire models` | Pydantic models that validate the JSON from the worker. |
| 4 | `src/encoder.py` | `# [ENCODING-CHECKPOINT-4] Sub-encoders` | The actual tensor encoding logic — sub-encoder functions that produce float/int tensors. |
| 5 | `src/encoder.py` (bottom) | `ENTITY_FEATURE_DIM`, `FIELD_FEATURE_DIM`, `SIDE_FEATURE_DIM`, `SCALAR_FEATURE_DIM` | Sentinel-derived dimension constants. These auto-update when you change a sub-encoder's width. |
| 6 | `src/cvpn.py` | `CVPNConfig` dataclass | The neural net reads encoder dims. If dims change, the net's input projections must match. CVPNConfig already reads live constants, so it auto-adapts — but verify. |
| 7 | `src/replay_buffer.py` | `_current_schema()` | Schema fingerprint. If feature width changed, bump `ENCODER_SCHEMA_VERSION` in encoder.py (the fingerprint reads it). Only bump on same-width semantic changes the dims can't catch. |

## Verification locations (tests)

| File | What to check / update |
|------|----------------------|
| `tests/unit/test_encoder.py` | Add or verify a test for the new/existing feature in the appropriate exhaustive test class. |
| `tests/integration/test_encoder_pipeline.py` | If the feature can be triggered by real engine moves, add a test that drives the engine and asserts the feature encodes correctly. |
| `sim/tests/sim-worker.test.ts` | If checkpoint 1 was changed, add a TS unit test for the snapshot extraction. |

## Workflow

### Step 1: Identify which sub-encoder owns this mechanic

Classify the mechanic:

- **Entity-level volatile/counter** (substitute, stall, yawn, confusion, etc.) → `_volatile_counters` or `_encode_pokemon_features`
- **Status condition** (burn, paralysis, sleep, etc.) → `_status_onehot`
- **Side condition** (tailwind, reflect, aurora veil, hazards, etc.) → `_encode_side` / `_side_screen` / `_side_hazards`
- **Field condition** (weather, terrain, trick room, gravity, etc.) → `_encode_field` / sub-functions
- **Scalar/global** (turn number, phase, who's acting) → `_encode_scalars`
- **Per-pokemon categorical** (species, ability, item, moves) → vocab IDs in `_encode_move_ids` / the `encode()` body
- **Per-pokemon stat/flag** (HP, stats, boosts, position, fainted) → the relevant sub-encoder

### Step 2: Search for the mechanic

Use Grep to search `src/encoder.py` for the mechanic name (lowercased, no spaces — matching Showdown ID style). For example:
- Aurora Veil → `auroraveil`
- Trick Room → `trickroom`
- Stealth Rock → `stealthrock`

Also search `sim/src/sim-worker.ts` and `src/state_types.py` to confirm data flows through.

### Step 3a: If FOUND — verify test coverage

1. Search `tests/unit/test_encoder.py` for the mechanic name.
2. If tested → report "X is encoded and tested" with the location.
3. If NOT tested → add a test following the patterns in the existing exhaustive test classes.

### Step 3b: If NOT FOUND — add it everywhere

Work through checkpoints 1–7 in order:

1. **sim-worker.ts** — Does `snapshotPokemon` / `snapshotSide` / `snapshotBattle` extract this data from the engine? If not, add extraction. Look at Showdown's `battle.field`, `side.sideConditions`, `pokemon.volatiles`, etc.

2. **types.ts** — Does the TS interface include a field for this data? Usually side conditions and volatiles are already covered by the generic `dict`/`Record` patterns, so this may already be fine.

3. **state_types.py** — Same as above — the generic `dict[str, SideConditionSnapshot]` or `dict[str, VolatileDetail]` patterns usually cover new conditions without schema changes.

4. **encoder.py** — This is the main work. Add the feature to the appropriate sub-encoder:
   - For a new **side condition**: add a `_side_screen("newcondition", conds)` call in `_encode_side`, or a dedicated helper if it's not a simple active+duration pattern.
   - For a new **volatile**: add extraction logic in `_volatile_counters` or create a new sub-encoder and append it to `_encode_pokemon_features`.
   - For a new **field condition**: add to `_encode_field`.
   - For a new **scalar**: add to `_encode_scalars`.

5. **Dimension constants** — They auto-derive from sentinel calls at the bottom of encoder.py. Just verify they update (re-import or re-run the module).

6. **CVPN** — `CVPNConfig` reads `ENTITY_FEATURE_DIM` etc. at class-definition time. If dims changed, the CVPN's linear layers will auto-resize on next instantiation. No manual change needed unless you added a new top-level ObsBundle key.

7. **Replay buffer** — If the feature changed an existing dim width, the fingerprint auto-detects it. If it's a same-width semantic change (reordering features within F), bump `ENCODER_SCHEMA_VERSION`.

8. **Tests** — Add AND update:
   - A unit test in `tests/unit/test_encoder.py` in the appropriate `TestXxxExhaustive` class.
   - Optionally an integration test if engine interaction is needed to trigger the mechanic.
   - **Update existing shape/dim assertions** that hardcode the old width:
     - The sub-encoder's own `test_shape` (e.g. `vec.shape == (4,)` → `(5,)`)
     - The composite `ENTITY_FEATURE_DIM == N` assertion in `TestEncodePhaseVariations`
     - Search for the old dim value to catch any others (e.g. `grep "== 66"`)
   - **Update the sub-encoder's docstring** width annotation (e.g. `[4]` → `[5]`).
   - If the mechanic was previously listed in `TestVolatilesDoNotCrashEncoder` (the "encoder ignores this" safety net), **remove it** — it's now explicitly encoded, not ignored.

### Step 4: Run tests

```bash
# Python unit tests (fast)
pytest tests/unit/test_encoder.py -v

# Integration tests (needs sim worker)
pytest tests/integration/test_encoder_pipeline.py -v

# TypeScript tests (if sim-worker.ts changed)
cd sim && npm test
```

### Step 5: Report

Summarize:
- Whether the mechanic was already present or newly added.
- Which files were modified.
- The new dimension constants (if changed).
- Whether `ENCODER_SCHEMA_VERSION` was bumped.
- Test results.

## Example: Checking Aurora Veil

```
1. Grep encoder.py for "auroraveil"
   → Found in _encode_side: `_side_screen("auroraveil", conds)`
2. Grep test_encoder.py for "auroraveil"
   → Found in TestSideScreensExhaustive parametrize list
3. Conclusion: Aurora Veil IS encoded and tested. No changes needed.
```

## Example: Adding Toxic Spikes (hypothetical)

```
1. Grep encoder.py for "toxicspikes" → NOT in any sub-encoder (only in a "does not crash" test)
2. Checkpoint 1: sim-worker.ts snapshotSide already captures all sideConditions generically ✓
3. Checkpoint 2-3: types.ts and state_types.py use generic dict patterns ✓
4. Checkpoint 4: Add to _encode_side:
   - Add `_side_hazard_toxic(conds)` helper returning [layers/2] (toxic spikes has 1-2 layers)
   - Append to _encode_side's torch.cat list
5. Checkpoint 5: SIDE_FEATURE_DIM auto-updates (sentinel call)
6. Checkpoint 6: CVPNConfig.side_feature_dim reads the new constant ✓
7. Checkpoint 7: Dim changed → fingerprint auto-detects, no manual bump needed
8. Tests: Add TestToxicSpikesExhaustive in test_encoder.py
9. Run pytest → pass
```

## Important notes

- **Never hard-code dimension numbers.** Always let the sentinel calls at the bottom of encoder.py derive them.
- **The sim worker captures data generically** for volatiles, side conditions, and pseudo-weather — so checkpoint 1 rarely needs changes for battle conditions that already exist in the engine. It's usually just the Python encoder that needs updating.
- **If you add width to a sub-encoder**, all downstream consumers auto-adapt (CVPNConfig reads live constants, replay buffer fingerprint reads live constants). But existing saved replay buffers and checkpoints become incompatible — this is by design.
- **Existing tests will break on width changes.** The test suite hardcodes expected shapes and dim values (e.g. `vec.shape == (4,)`, `ENTITY_FEATURE_DIM == 66`). After modifying a sub-encoder, grep the test file for the old numeric value and update all assertions. Also update the sub-encoder's docstring width comment (e.g. `[4]` → `[5]`).
- **"Does not crash" vs "explicitly encoded"** — `TestVolatilesDoNotCrashEncoder` covers volatiles the encoder *ignores*. If you promote a volatile from "ignored" to "explicitly encoded", remove it from that parametrize list.
