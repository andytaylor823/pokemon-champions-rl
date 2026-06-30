/**
 * sim-worker search-correctness contracts, exercised through the worker
 * boundary (`dispatch`), not just at the raw State level: clone == parent
 * fidelity, step reproducibility, and the real cloneBattle/step code path.
 */
import { describe, it, expect, beforeEach } from "vitest";
import { dispatch, resetState } from "../src/sim-worker";
import { freshBattle, advancePastTeamPreview, rollout } from "./helpers/worker-harness";

beforeEach(() => resetState());

// A1 — Search-correctness contracts, exercised through the worker boundary
// (dispatch), not just at the raw State level. Covers clone == parent
// fidelity, step reproducibility, and the real cloneBattle/step code path.
describe("search contracts through the worker", () => {
  /** Stable JSON for deep comparison of snapshots/views. */
  const json = (x: unknown) => JSON.stringify(x);

  it("a clone reproduces the parent snapshot exactly (team preview)", () => {
    const battle = freshBattle();
    const source = dispatch({ cmd: "view", handle: battle.handle }).view;

    const search = dispatch({ cmd: "open_search", from: battle.handle });
    const clone = search.root_view;

    expect(json(clone.snapshot)).toBe(json(source.snapshot));
    expect(clone.phase).toBe(source.phase);
    expect(clone.to_move).toEqual(source.to_move);
    expect(clone.terminal).toBe(source.terminal);
    expect(clone.utility).toEqual(source.utility);

    dispatch({ cmd: "close_search", session: search.session });
  });

  it("a clone reproduces the parent snapshot exactly (mid-battle)", () => {
    const battle = freshBattle();
    const step1 = advancePastTeamPreview(battle.handle);
    expect(step1.view.phase).toBe("move");

    // Clone the mid-battle handle and compare the clone's snapshot.
    const source = dispatch({ cmd: "view", handle: step1.child }).view;
    const search = dispatch({ cmd: "open_search", from: step1.child });
    const clone = search.root_view;

    expect(json(clone.snapshot)).toBe(json(source.snapshot));
    expect(clone.phase).toBe(source.phase);
    expect(clone.to_move).toEqual(source.to_move);

    dispatch({ cmd: "close_search", session: search.session });
  });

  it("step is reproducible: same parent + same seed + same choices => identical result", () => {
    const battle = freshBattle();
    const step1 = advancePastTeamPreview(battle.handle);
    expect(step1.view.phase).toBe("move");

    const choices = { p1: "move 1, move 1", p2: "move 1 1, move 1 1" };
    for (const seed of [
      [100, 200, 300, 400],
      [7, 7, 7, 7],
      [999, 1, 2, 3],
    ] as [number, number, number, number][]) {
      const a = dispatch({ cmd: "step", handle: step1.child, choices, seed });
      const b = dispatch({ cmd: "step", handle: step1.child, choices, seed });
      // Identical reseed => identical RNG => identical events and state.
      expect(a.outcome).toEqual(b.outcome);
      expect(json(a.view.snapshot)).toBe(json(b.view.snapshot));
      expect(a.view.utility).toEqual(b.view.utility);
    }
  });

  it("two independent rollouts from one root with identical seeds agree", () => {
    const battle = freshBattle();
    const search = dispatch({ cmd: "open_search", from: battle.handle });

    // Step the SAME root twice (step is immutable) with the same per-step seeds.
    const r1 = rollout(search.root);
    const r2 = rollout(search.root);

    expect(r1.outcomes).toEqual(r2.outcomes);
    expect(json(r1.view.snapshot)).toBe(json(r2.view.snapshot));
    expect(r1.view.utility).toEqual(r2.view.utility);
    expect(r1.view.terminal).toBe(true);

    dispatch({ cmd: "close_search", session: search.session });
  });

  it("a full rollout via the real cloneBattle/step path completes without error", () => {
    // Each step clones the parent (the worker's cloneBattle, with its
    // sentLogPos fix) and applies choices. Historically a missing sentLogPos
    // fix produced a false "Infinite loop" error during such deep sequential
    // stepping. This exercises the worker's actual code path end-to-end; the
    // >1000-line stress variant is covered at the State level in
    // integration/battle.test.ts.
    const battle = freshBattle();
    const search = dispatch({ cmd: "open_search", from: battle.handle });

    const r = rollout(search.root);

    expect(r.view.terminal).toBe(true);
    // A complete battle accumulates a substantial log across the clone chain.
    const totalLines = r.outcomes.reduce((n, o) => n + o.length, 0);
    expect(totalLines).toBeGreaterThan(100);
    // Every intermediate clone produced a well-shaped, finite snapshot.
    for (const v of r.views) {
      expect(v.snapshot.sides).toHaveLength(2);
      for (const side of v.snapshot.sides) {
        for (const mon of side.pokemon) {
          expect(Number.isFinite(mon.hp)).toBe(true);
          expect(Number.isFinite(mon.maxhp)).toBe(true);
        }
      }
    }

    dispatch({ cmd: "close_search", session: search.session });
  });
});
