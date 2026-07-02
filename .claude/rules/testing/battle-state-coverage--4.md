---
alwaysApply: false
paths: src/obs_bundle.py
---

# Battle-State Test Completeness — Standing Rule

## When this rule fires

Whenever you are:
- Working on `src/encoder.py`, `src/action_space.py`, `src/state_types.py`, or `src/obs_bundle.py`
- Discussing a new battle mechanic, condition, volatile, ability effect, or item interaction
- Designing a module that produces or consumes battle state (e.g. belief model, search, meta priors)
- Reviewing code that touches the encoder/action-space boundary

## What to check

Ask: **"Is this battle state tested in the encoder/action-space test suite?"**

Specifically:
- Is this **volatile** tested in `TestVolatilesDoNotCrashEncoder`? (e.g. encore, taunt, perish song, confusion, yawn, flinch, disable, attract, etc.)
- Is this **side condition** tested in `TestSideScreensExhaustive` / `TestSideHazardsExhaustive` / `TestSideConditionCombinations`?
- Is this **field condition** tested in `TestWeatherExhaustive` / `TestTerrainExhaustive` / `TestPseudoWeatherCombinations`?
- Is this **status** tested in `TestStatusOnehotExhaustive`?
- Is this **action-space constraint** tested in `TestTrappedMon` / `TestBenchStates` / `TestZeroPpExclusion`?

## What to do if it's not tested

Add a test immediately. Do not defer. The test files are:
- `tests/unit/test_encoder.py` — sub-encoder and full encode() tests
- `tests/unit/test_action_space.py` — legal mask and choice string tests
- `tests/unit/test_state_types.py` — Pydantic model contract tests
- `tests/integration/test_encoder_pipeline.py` — real engine round-trip tests

## Examples of mechanics to watch for

These are NOT exhaustive — new ones should be added as the format evolves:

- **Volatiles**: substitute, stall, encore, taunt, disable, confusion, perish song, yawn, flinch, protect, leech seed, torment, attract, imprison, heal block, embargo
- **Side conditions**: tailwind, reflect, light screen, aurora veil, stealth rock, spikes, toxic spikes, sticky web
- **Field**: rain, sun, sand, snow/hail, electric/grassy/misty/psychic terrain, trick room, gravity, magic room, wonder room
- **Action constraints**: trapped (Shadow Tag, Arena Trap), Choice-locked (Choice Scarf/Band/Specs), Encore (forces same move), Disable (blocks one move), Taunt (blocks status moves), Assault Vest (prevents status), Gorilla Tactics (locks move), encore + choice interactions
- **Phase transitions**: Baton Pass creating switch phase, U-turn/Volt Switch, Parting Shot, emergency forceSwitch after a faint
- **Item states**: consumed (Berry eaten), knocked off (Knock Off), switched (Trick/Switcheroo), mega stone consumed by Mega Evolution
