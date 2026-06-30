/**
 * sim-worker dispatch tests — handle/session lifecycle, stepping, error paths,
 * terminal state, and adversarial inputs. Everything is driven through
 * `dispatch` via the shared harness in tests/helpers/worker-harness.ts.
 */
import { describe, it, expect, beforeEach } from "vitest";
import { dispatch, resetState } from "../src/sim-worker";
import { TEAM_A, TEAM_B } from "./fixtures/teams";
import {
  freshBattle,
  advancePastTeamPreview,
  handleCount,
  sessionCount,
  autoStep,
  runToTerminal,
  advanceUntilPhase,
} from "./helpers/worker-harness";

beforeEach(() => resetState());

describe("config", () => {
  it("returns the format id", () => {
    const result = dispatch({ cmd: "config", format_id: "gen9championsvgc2026regma" });
    expect(result.format_id).toBe("gen9championsvgc2026regma");
  });

  it("updates the format id when provided", () => {
    dispatch({ cmd: "config", format_id: "gen9ou" });
    const result = dispatch({ cmd: "config" });
    expect(result.format_id).toBe("gen9ou");
  });
});

describe("unknown command", () => {
  it("throws for an unrecognized cmd", () => {
    expect(() => dispatch({ cmd: "bogus" })).toThrow(/unknown cmd/);
  });
});

describe("handle lifecycle", () => {
  it("new_battle creates a handle and returns a view", () => {
    const result = freshBattle();
    expect(typeof result.handle).toBe("number");
    expect(result.view).toBeDefined();
    expect(result.view.phase).toBe("teamPreview");
    expect(result.view.terminal).toBe(false);
  });

  it("assigns incrementing handle ids", () => {
    const r1 = freshBattle();
    const r2 = freshBattle([5, 6, 7, 8]);
    expect(r2.handle).toBe(r1.handle + 1);
  });

  it("view returns the state of an existing handle", () => {
    const battle = freshBattle();
    const result = dispatch({ cmd: "view", handle: battle.handle });
    expect(result.view.phase).toBe("teamPreview");
    expect(result.view.snapshot.sides).toHaveLength(2);
  });

  it("release removes a handle from the registry", () => {
    const battle = freshBattle();
    expect(handleCount()).toBe(1);
    dispatch({ cmd: "release", handle: battle.handle });
    expect(handleCount()).toBe(0);
  });

  it("throws when viewing a released handle", () => {
    const battle = freshBattle();
    dispatch({ cmd: "release", handle: battle.handle });
    expect(() => dispatch({ cmd: "view", handle: battle.handle })).toThrow(/unknown handle/);
  });

  it("stats reports handle and session counts", () => {
    freshBattle();
    const stats = dispatch({ cmd: "stats" });
    expect(stats.handles).toBe(1);
    expect(stats.sessions).toBe(0);
  });
});

describe("search sessions", () => {
  it("open_search clones a battle into a session", () => {
    const battle = freshBattle();
    const search = dispatch({ cmd: "open_search", from: battle.handle });
    expect(typeof search.session).toBe("number");
    expect(typeof search.root).toBe("number");
    expect(search.root).not.toBe(battle.handle);
    expect(search.root_view.phase).toBe("teamPreview");
  });

  it("close_search frees all handles in the session", () => {
    const battle = freshBattle();
    const search = dispatch({ cmd: "open_search", from: battle.handle });

    // Step to create another child handle in the session
    advancePastTeamPreview(search.root);

    // 3 handles: live battle + root clone + step child
    expect(handleCount()).toBe(3);
    const closeResult = dispatch({ cmd: "close_search", session: search.session });
    expect(closeResult.freed).toBe(2); // root + child

    // Only the live battle remains
    expect(handleCount()).toBe(1);
    expect(() => dispatch({ cmd: "view", handle: battle.handle })).not.toThrow();
  });

  it("open_search without from creates an empty session", () => {
    const search = dispatch({ cmd: "open_search" });
    expect(search.session).toBeDefined();
    expect(search.root).toBeNull();
    expect(search.root_view).toBeNull();
    expect(sessionCount()).toBe(1);
  });
});

describe("step", () => {
  it("creates a child handle without mutating the parent", () => {
    const battle = freshBattle();
    const search = dispatch({ cmd: "open_search", from: battle.handle });

    const step = advancePastTeamPreview(search.root, [50, 60, 70, 80]);

    expect(typeof step.child).toBe("number");
    expect(step.child).not.toBe(search.root);

    // Parent still in teamPreview
    const parentView = dispatch({ cmd: "view", handle: search.root });
    expect(parentView.view.phase).toBe("teamPreview");

    // Child advanced past teamPreview
    expect(step.view.phase).not.toBe("teamPreview");
  });

  it("returns outcome log lines from the step", () => {
    const battle = freshBattle();
    const search = dispatch({ cmd: "open_search", from: battle.handle });
    const step = advancePastTeamPreview(search.root, [50, 60, 70, 80]);
    expect(Array.isArray(step.outcome)).toBe(true);
    expect(step.outcome.length).toBeGreaterThan(0);
  });

  it("places child handle in the same session as parent", () => {
    const battle = freshBattle();
    const search = dispatch({ cmd: "open_search", from: battle.handle });
    advancePastTeamPreview(search.root, [50, 60, 70, 80]);

    // close_search should free both root and child
    const closeResult = dispatch({ cmd: "close_search", session: search.session });
    expect(closeResult.freed).toBe(2);
  });

  it("throws for a missing choice on an acting side", () => {
    const battle = freshBattle();
    const search = dispatch({ cmd: "open_search", from: battle.handle });
    // Both sides must act in team preview; only provide p1
    expect(() => dispatch({
      cmd: "step", handle: search.root,
      choices: { p1: "team 1234" },
      seed: [50, 60, 70, 80],
    })).toThrow(/missing choice for acting side p2/);
  });
});

describe("adversarial inputs", () => {
  it("throws for missing cmd field", () => {
    expect(() => dispatch({})).toThrow(/unknown cmd/);
  });

  it("throws for view with missing handle", () => {
    expect(() => dispatch({ cmd: "view" })).toThrow(/unknown handle/);
  });

  it("throws for step with non-existent handle", () => {
    expect(() => dispatch({ cmd: "step", handle: 999 })).toThrow(/unknown handle 999/);
  });

  it("throws for step with empty choices when both sides must act", () => {
    const battle = freshBattle();
    expect(() => dispatch({
      cmd: "step", handle: battle.handle, choices: {}, seed: [10, 20, 30, 40],
    })).toThrow(/missing choice for acting side/);
  });

  it("close_search with non-existent session returns freed: 0", () => {
    const result = dispatch({ cmd: "close_search", session: 999 });
    expect(result.freed).toBe(0);
    expect(result.closed).toBe(999);
  });

  it("throws for choice given to non-existent side", () => {
    const battle = freshBattle();
    expect(() => dispatch({
      cmd: "step", handle: battle.handle,
      choices: { p1: "team 1234", p2: "team 1234", p3: "team 1234" } as any,
      seed: [10, 20, 30, 40],
    })).toThrow(/choice given for non-acting side p3/);
  });
});

describe("additional config and command edge cases", () => {
  it("config returns default format when no format_id provided", () => {
    const result = dispatch({ cmd: "config" });
    expect(result.format_id).toBe("gen9championsvgc2026regma");
  });

  it("close command returns shutdown: true", () => {
    const result = dispatch({ cmd: "close" });
    expect(result.shutdown).toBe(true);
  });
});

describe("error paths", () => {
  it("throws for illegal choice string (strictChoices)", () => {
    const battle = freshBattle();
    const step1 = advancePastTeamPreview(battle.handle);
    expect(step1.view.phase).toBe("move");

    // With strictChoices, the engine throws directly for invalid moves
    expect(() => dispatch({
      cmd: "step", handle: step1.child,
      choices: { p1: "move 99, move 99", p2: "move 1 1, move 1 1" },
      seed: [50, 60, 70, 80],
    })).toThrow(/Invalid choice/);
  });
});

describe("handle and session edge cases", () => {
  it("step on live battle creates sessionless child", () => {
    const battle = freshBattle();
    advancePastTeamPreview(battle.handle);

    // Both handles are sessionless — no search session was opened
    expect(handleCount()).toBe(2);
    expect(sessionCount()).toBe(0);

    // Opening and closing a search session doesn't affect live handles
    const session = dispatch({ cmd: "open_search" });
    dispatch({ cmd: "close_search", session: session.session });
    expect(handleCount()).toBe(2);
  });

  it("release on unknown handle does not throw", () => {
    const result = dispatch({ cmd: "release", handle: 999 });
    expect(result.released).toBe(999);
  });

  it("multi-depth search tree: all handles freed on close", () => {
    const battle = freshBattle();
    const session = dispatch({ cmd: "open_search", from: battle.handle });

    // root -> child1 -> child2 -> child3 (3 levels of stepping)
    const child1 = advancePastTeamPreview(session.root);
    const child2 = autoStep(child1.child, 50);
    const child3 = autoStep(child2.child, 100);

    // 5 handles total: live battle + root + child1 + child2 + child3
    expect(handleCount()).toBe(5);

    const closed = dispatch({ cmd: "close_search", session: session.session });
    expect(closed.freed).toBe(4);
    expect(handleCount()).toBe(1);
  });

  it("concurrent sessions are isolated", () => {
    const battle = freshBattle();
    const s1 = dispatch({ cmd: "open_search", from: battle.handle });
    const s2 = dispatch({ cmd: "open_search", from: battle.handle });

    // Step in both sessions
    advancePastTeamPreview(s1.root);
    advancePastTeamPreview(s2.root, [50, 60, 70, 80]);

    // 5 handles: live + s1(root + child) + s2(root + child)
    expect(handleCount()).toBe(5);

    // Closing session 1 should only free its handles
    dispatch({ cmd: "close_search", session: s1.session });
    expect(handleCount()).toBe(3);

    // Session 2's handles still work
    expect(() => dispatch({ cmd: "view", handle: s2.root })).not.toThrow();

    dispatch({ cmd: "close_search", session: s2.session });
    expect(handleCount()).toBe(1);
  });

  it("step without seed inherits parent PRNG state", () => {
    const battle = freshBattle();
    // Step without providing a seed — child inherits parent's PRNG
    const step = dispatch({
      cmd: "step", handle: battle.handle,
      choices: { p1: "team 1234", p2: "team 1234" },
    });
    expect(step.child).toBeDefined();
    expect(step.view.phase).not.toBe("teamPreview");
  });

  it("new_battle without seed creates a valid battle", () => {
    const result = dispatch({ cmd: "new_battle", team_a: TEAM_A, team_b: TEAM_B });
    expect(typeof result.handle).toBe("number");
    expect(result.view.phase).toBe("teamPreview");
  });
});

describe("terminal state and utility", () => {
  it("terminal view has correct phase, utility, and empty to_move", () => {
    const battle = freshBattle();
    const session = dispatch({ cmd: "open_search", from: battle.handle });
    const result = runToTerminal(session.root);

    expect(result.view.phase).toBe("terminal");
    expect(result.view.terminal).toBe(true);
    expect(result.view.to_move).toHaveLength(0);

    // Utility is +1/-1 for winner/loser (zero-sum)
    const u = result.view.utility;
    expect(u).not.toBeNull();
    expect(Math.abs(u.p1)).toBe(1);
    expect(Math.abs(u.p2)).toBe(1);
    expect(u.p1).toBe(-u.p2);

    dispatch({ cmd: "close_search", session: session.session });
  });
});

describe("forceSwitch phase", () => {
  it("appears when a Pokemon faints during battle", () => {
    const battle = freshBattle();
    const session = dispatch({ cmd: "open_search", from: battle.handle });
    const fs = advanceUntilPhase(session.root, "forceSwitch");

    expect(fs).not.toBeNull();
    expect(fs!.view.phase).toBe("forceSwitch");
    expect(fs!.view.to_move.length).toBeGreaterThanOrEqual(1);
    expect(fs!.view.to_move.length).toBeLessThanOrEqual(2);

    dispatch({ cmd: "close_search", session: session.session });
  });
});

describe("tie game utility", () => {
  it("utility values are valid and zero-sum across multiple seeds", () => {
    for (const seedBase of [1, 10, 50, 100, 200]) {
      resetState();
      const seed: [number, number, number, number] = [seedBase, seedBase + 1, seedBase + 2, seedBase + 3];
      const battle = freshBattle(seed);
      const session = dispatch({ cmd: "open_search", from: battle.handle });
      const result = runToTerminal(session.root);
      const u = result.view.utility;

      expect(u).not.toBeNull();
      expect([1, -1, 0]).toContain(u.p1);
      expect([1, -1, 0]).toContain(u.p2);
      expect(u.p1 + u.p2).toBe(0);

      dispatch({ cmd: "close_search", session: session.session });
    }
  });
});

describe("reseed variance through dispatch", () => {
  it("same parent with different seeds produces different children", () => {
    const battle = freshBattle();
    const step1 = advancePastTeamPreview(battle.handle);
    expect(step1.view.phase).toBe("move");

    // Step the SAME parent handle with two different seeds
    const childA = dispatch({
      cmd: "step", handle: step1.child,
      choices: { p1: "move 1, move 1", p2: "move 1 1, move 1 1" },
      seed: [100, 200, 300, 400],
    });
    const childB = dispatch({
      cmd: "step", handle: step1.child,
      choices: { p1: "move 1, move 1", p2: "move 1 1, move 1 1" },
      seed: [999, 888, 777, 666],
    });

    // Different seeds → different RNG → different damage rolls / outcomes
    expect(childA.outcome).not.toEqual(childB.outcome);
  });
});

describe("step on terminal handle", () => {
  it("throws when attempting to step a terminal battle", () => {
    const battle = freshBattle();
    const session = dispatch({ cmd: "open_search", from: battle.handle });
    const result = runToTerminal(session.root);
    expect(result.view.terminal).toBe(true);

    // No acting sides → applyChoices should throw for non-acting side
    expect(() => dispatch({
      cmd: "step", handle: result.handle,
      choices: { p1: "move 1", p2: "move 1" },
      seed: [1, 2, 3, 4],
    })).toThrow();

    dispatch({ cmd: "close_search", session: session.session });
  });
});

describe("session nesting", () => {
  it("open_search from search-owned handle produces independent session", () => {
    const battle = freshBattle();
    const sessionA = dispatch({ cmd: "open_search", from: battle.handle });

    // Open a second session cloning from the first session's root
    const sessionB = dispatch({ cmd: "open_search", from: sessionA.root });
    expect(sessionB.session).not.toBe(sessionA.session);
    expect(sessionB.root).not.toBe(sessionA.root);
    expect(sessionB.root_view).toBeDefined();

    // 3 handles: live battle + sessionA root + sessionB root
    expect(handleCount()).toBe(3);

    // Closing session A should NOT affect session B
    dispatch({ cmd: "close_search", session: sessionA.session });
    expect(() => dispatch({ cmd: "view", handle: sessionB.root })).not.toThrow();
    expect(handleCount()).toBe(2);

    dispatch({ cmd: "close_search", session: sessionB.session });
    expect(handleCount()).toBe(1);
  });
});

describe("additional robustness edge cases", () => {
  it("step with no choices key throws for acting sides", () => {
    const battle = freshBattle();
    // msg.choices ?? {} → empty choices → missing choice for acting side
    expect(() => dispatch({
      cmd: "step", handle: battle.handle, seed: [10, 20, 30, 40],
    })).toThrow(/missing choice for acting side/);
  });

  it("double release of same handle does not throw", () => {
    const battle = freshBattle();
    dispatch({ cmd: "release", handle: battle.handle });
    const result = dispatch({ cmd: "release", handle: battle.handle });
    expect(result.released).toBe(battle.handle);
  });

  it("independent battles are isolated", () => {
    const b1 = freshBattle();
    const b2 = freshBattle([5, 6, 7, 8]);

    // Advance battle 1 past team preview
    const step = advancePastTeamPreview(b1.handle);
    expect(step.view.phase).toBe("move");

    // Battle 2 should still be in team preview, completely unaffected
    const v2 = dispatch({ cmd: "view", handle: b2.handle }).view;
    expect(v2.phase).toBe("teamPreview");
    expect(v2.snapshot.turn).toBe(0);
  });
});
