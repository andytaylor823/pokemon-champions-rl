/**
 * Shared in-process harness for sim-worker unit tests.
 *
 * Every helper drives the worker through `dispatch` (the same entry the stdio
 * loop uses), so tests exercise the real code path. The standard test teams and
 * the canonical "advance past team preview" / rollout sequences live here once
 * instead of being re-spelled in every test.
 */
import { dispatch } from "../../src/sim-worker";
import { TEAM_A, TEAM_B } from "../fixtures/teams";

export type Seed = [number, number, number, number];

/** Seed used by the overwhelming majority of fixed-outcome tests. */
export const STD_SEED: Seed = [1, 2, 3, 4];

/** A fresh battle from the standard test teams; returns `{ handle, view }`. */
export function freshBattle(seed: Seed = STD_SEED) {
  return dispatch({ cmd: "new_battle", team_a: TEAM_A, team_b: TEAM_B, seed });
}

/** Advance a handle past team preview into the move phase (both sides "team 1234"). */
export function advancePastTeamPreview(handle: number, seed: Seed = [10, 20, 30, 40]) {
  return dispatch({
    cmd: "step", handle,
    choices: { p1: "team 1234", p2: "team 1234" },
    seed,
  });
}

/** Live handle / session counts, read through the public `stats` command. */
export const handleCount = () => dispatch({ cmd: "stats" }).handles;
export const sessionCount = () => dispatch({ cmd: "stats" }).sessions;

/** Step a handle using "default" auto-choice for all acting sides. */
export function autoStep(handle: number, seedOff: number) {
  const v = dispatch({ cmd: "view", handle }).view;
  const choices: Record<string, string> = {};
  for (const sid of v.to_move) choices[sid] = "default";
  return dispatch({
    cmd: "step", handle, choices,
    seed: [seedOff, seedOff + 1, seedOff + 2, seedOff + 3],
  });
}

/** Auto-play a battle to terminal state via default choices. */
export function runToTerminal(startHandle: number, maxSteps = 200) {
  let h = startHandle;
  for (let i = 0; i < maxSteps; i++) {
    const v = dispatch({ cmd: "view", handle: h }).view;
    if (v.terminal) return { handle: h, view: v };
    h = autoStep(h, i * 4 + 1).child;
  }
  throw new Error("Battle did not terminate within maxSteps");
}

/** Auto-play until a target phase is reached (returns null if terminal first). */
export function advanceUntilPhase(startHandle: number, targetPhase: string, maxSteps = 200) {
  let h = startHandle;
  for (let i = 0; i < maxSteps; i++) {
    const v = dispatch({ cmd: "view", handle: h }).view;
    if (v.phase === targetPhase) return { handle: h, view: v };
    if (v.terminal) return null;
    h = autoStep(h, i * 4 + 1).child;
  }
  return null;
}

/** Roll a handle to terminal, collecting per-step outcomes + every view. */
export function rollout(startHandle: number, maxSteps = 250) {
  let h = startHandle;
  const outcomes: string[][] = [];
  const views: any[] = [];
  for (let i = 0; i < maxSteps; i++) {
    const v = dispatch({ cmd: "view", handle: h }).view;
    views.push(v);
    if (v.terminal) return { handle: h, view: v, outcomes, views };
    const step = autoStep(h, i * 4 + 1);
    outcomes.push(step.outcome);
    h = step.child;
  }
  throw new Error("rollout did not terminate");
}
