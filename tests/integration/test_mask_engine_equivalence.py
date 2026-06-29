"""Integration test: legal_mask must agree with the Showdown engine over real play.

The engine (`battle.choose`) is the sole source of truth for legality. This test
drives random legal play through a real sim-worker and, at every decision, asserts
the two invariants whose violation actually breaks self-play:

  1. NON-EMPTY: an acting side always has >= 1 legal action. An empty mask yields an
     empty strategy and aborts the game — exactly the double-faint forceSwitch bug
     this suite was written for (one bench mon left, the lone switch collision-erased).
  2. SOUND: every choice the mask marks legal is accepted by the engine — the
     "sample a joint and hope it's legal" failure mode.

Why not also assert completeness (every engine-legal choice is masked)? Showdown
accepts many equivalent *string* encodings the canonical mask intentionally does not
emit — single-slot shorthands like "move 1" that auto-pass a fainted slot, and
"default". Comparing accepted strings would false-alarm on those. Exact completeness
is covered deterministically by the unit tests (see TestForceSwitchDoubleFaint and
the legal_mask suites in tests/unit/test_action_space.py).

Non-empty is free (no engine calls), so this runs many games to maximise the chance
of reaching rare states (double faints, forced switches). Soundness is checked on a
sample of the mask per decision, and exhaustively at the small forceSwitch masks.

Marked slow. Tunable via MASK_EQUIV_GAMES / MASK_EQUIV_STEP_CAP / MASK_EQUIV_SAMPLE.
"""
from __future__ import annotations

import os
import random

import numpy as np
import pytest

from action_space import action_to_choice_contextual, legal_mask
from sim_client import SimClient, SimError

GAMES = int(os.environ.get("MASK_EQUIV_GAMES", "20"))
STEP_CAP = int(os.environ.get("MASK_EQUIV_STEP_CAP", "200"))
SAMPLE = int(os.environ.get("MASK_EQUIV_SAMPLE", "8"))  # mask choices soundness-checked per side per move decision
PROBE_SEED = [1, 2, 3, 4]


def _mask_choices(req: dict, phase: str) -> list[str]:
    """Decoded choice strings the mask marks legal (deduped, stable order)."""
    seen: dict[str, None] = {}
    for i in np.flatnonzero(legal_mask(req, phase)):
        seen.setdefault(action_to_choice_contextual(int(i), req), None)
    return list(seen)


def _engine_accepts(sim: SimClient, handle: int, side: str, choice: str, acting: list[str]) -> bool:
    """True iff the engine accepts `choice` for `side`.

    Other acting sides are held at "default" (always legal) and per-side validation is
    independent, so any step failure is attributable to `side`'s candidate. Rejections
    surface in two shapes — the worker's "illegal choice for <side>" and Showdown's own
    "[Invalid choice] ..." — so attribute by step success, not by error text.
    """
    choices = {side: choice}
    for other in acting:
        if other != side:
            choices[other] = "default"
    try:
        result = sim.step(handle, choices, PROBE_SEED)
        sim.release(result.child)
        return True
    except SimError:
        return False


def _random_legal_joint(view, rng: random.Random) -> dict[str, str]:
    """Pick a uniformly-random mask-legal action per acting side (to advance play)."""
    choices: dict[str, str] = {}
    for side in view.to_move:
        legal_idx = np.flatnonzero(legal_mask(view.legal.get(side), view.phase))
        idx = int(rng.choice(legal_idx))
        choices[side] = action_to_choice_contextual(idx, view.legal.get(side))
    return choices


@pytest.mark.slow
def test_mask_never_empty_and_sound_over_random_play(sim_client: SimClient, team_a: list, team_b: list):
    """Across many random games, every acting side always has a legal mask, and every
    mask-legal choice (sampled; exhaustive at forceSwitch) is accepted by the engine."""
    rng = random.Random(20260629)
    decisions = 0
    forceswitch_decisions = 0
    soundness_checks = 0

    for game in range(GAMES):
        handle, view = sim_client.new_battle(team_a, team_b, seed=[game + 1, 7, 13, 29])
        try:
            steps = 0
            while not view.terminal and steps < STEP_CAP:
                if not view.to_move or view.phase == "none":
                    result = sim_client.step(handle, {}, PROBE_SEED)
                    sim_client.release(handle)
                    handle, view = result.child, result.view
                    continue

                acting = list(view.to_move)
                phase = view.phase
                for side in acting:
                    req = view.legal.get(side)
                    choices = _mask_choices(req, phase)

                    # (1) Non-empty — the abort trigger. An acting side must have an action.
                    assert choices, (
                        f"empty legal mask for acting side {side} at {phase} "
                        f"(game {game}, step {steps}) — this aborts the game"
                    )
                    decisions += 1
                    if phase == "forceSwitch":
                        forceswitch_decisions += 1

                    # (2) Soundness — exhaustive at forceSwitch (tiny), sampled elsewhere.
                    if phase == "forceSwitch" or len(choices) <= SAMPLE:
                        to_check = choices
                    else:
                        to_check = rng.sample(choices, SAMPLE)
                    for choice in to_check:
                        assert _engine_accepts(sim_client, handle, side, choice, acting), (
                            f"mask-legal choice rejected by engine: side={side} "
                            f"phase={phase} choice={choice!r} (game {game}, step {steps})"
                        )
                        soundness_checks += 1

                result = sim_client.step(handle, _random_legal_joint(view, rng), PROBE_SEED)
                sim_client.release(handle)
                handle, view = result.child, result.view
                steps += 1
        finally:
            sim_client.release(handle)

    assert decisions > 0, "no decisions exercised — test drove no games"
    assert soundness_checks > 0, "no soundness checks ran"
    print(
        f"\nchecked {decisions} side-decisions ({forceswitch_decisions} forceSwitch), "
        f"{soundness_checks} soundness probes across {GAMES} games"
    )
