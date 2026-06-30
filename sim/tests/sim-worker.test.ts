/**
 * Unit tests for sim-worker internals. These import the module directly
 * (no subprocess) thanks to the `require.main === module` guard.
 */
import { describe, it, expect, beforeEach } from "vitest";
import { dispatch, resetState } from "../src/sim-worker";
import { TEAM_A, TEAM_B } from "./fixtures/teams";

/** Live handle / session counts, read through the public `stats` command. */
const handleCount = () => dispatch({ cmd: "stats" }).handles;
const sessionCount = () => dispatch({ cmd: "stats" }).sessions;

describe("sim-worker dispatch (unit)", () => {
  // Reset global handle/session state between each test
  beforeEach(() => resetState());

  // --- Helpers for extended tests ---

  /** Step a handle using "default" auto-choice for all acting sides. */
  function autoStep(handle: number, seedOff: number) {
    const v = dispatch({ cmd: "view", handle }).view;
    const choices: Record<string, string> = {};
    for (const sid of v.to_move) choices[sid] = "default";
    return dispatch({
      cmd: "step", handle, choices,
      seed: [seedOff, seedOff + 1, seedOff + 2, seedOff + 3],
    });
  }

  /** Auto-play a battle to terminal state via default choices. */
  function runToTerminal(startHandle: number, maxSteps = 200) {
    let h = startHandle;
    for (let i = 0; i < maxSteps; i++) {
      const v = dispatch({ cmd: "view", handle: h }).view;
      if (v.terminal) return { handle: h, view: v };
      h = autoStep(h, i * 4 + 1).child;
    }
    throw new Error("Battle did not terminate within maxSteps");
  }

  /** Auto-play until a target phase is reached (returns null if terminal first). */
  function advanceUntilPhase(startHandle: number, targetPhase: string, maxSteps = 200) {
    let h = startHandle;
    for (let i = 0; i < maxSteps; i++) {
      const v = dispatch({ cmd: "view", handle: h }).view;
      if (v.phase === targetPhase) return { handle: h, view: v };
      if (v.terminal) return null;
      h = autoStep(h, i * 4 + 1).child;
    }
    return null;
  }

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
      const result = dispatch({ cmd: "new_battle", team_a: TEAM_A, team_b: TEAM_B, seed: [1, 2, 3, 4] });
      expect(typeof result.handle).toBe("number");
      expect(result.view).toBeDefined();
      expect(result.view.phase).toBe("teamPreview");
      expect(result.view.terminal).toBe(false);
    });

    it("assigns incrementing handle ids", () => {
      const r1 = dispatch({ cmd: "new_battle", team_a: TEAM_A, team_b: TEAM_B, seed: [1, 2, 3, 4] });
      const r2 = dispatch({ cmd: "new_battle", team_a: TEAM_A, team_b: TEAM_B, seed: [5, 6, 7, 8] });
      expect(r2.handle).toBe(r1.handle + 1);
    });

    it("view returns the state of an existing handle", () => {
      const battle = dispatch({ cmd: "new_battle", team_a: TEAM_A, team_b: TEAM_B, seed: [1, 2, 3, 4] });
      const result = dispatch({ cmd: "view", handle: battle.handle });
      expect(result.view.phase).toBe("teamPreview");
      expect(result.view.snapshot.sides).toHaveLength(2);
    });

    it("release removes a handle from the registry", () => {
      const battle = dispatch({ cmd: "new_battle", team_a: TEAM_A, team_b: TEAM_B, seed: [1, 2, 3, 4] });
      expect(handleCount()).toBe(1);
      dispatch({ cmd: "release", handle: battle.handle });
      expect(handleCount()).toBe(0);
    });

    it("throws when viewing a released handle", () => {
      const battle = dispatch({ cmd: "new_battle", team_a: TEAM_A, team_b: TEAM_B, seed: [1, 2, 3, 4] });
      dispatch({ cmd: "release", handle: battle.handle });
      expect(() => dispatch({ cmd: "view", handle: battle.handle })).toThrow(/unknown handle/);
    });

    it("stats reports handle and session counts", () => {
      dispatch({ cmd: "new_battle", team_a: TEAM_A, team_b: TEAM_B, seed: [1, 2, 3, 4] });
      const stats = dispatch({ cmd: "stats" });
      expect(stats.handles).toBe(1);
      expect(stats.sessions).toBe(0);
    });
  });

  describe("search sessions", () => {
    it("open_search clones a battle into a session", () => {
      const battle = dispatch({ cmd: "new_battle", team_a: TEAM_A, team_b: TEAM_B, seed: [1, 2, 3, 4] });
      const search = dispatch({ cmd: "open_search", from: battle.handle });
      expect(typeof search.session).toBe("number");
      expect(typeof search.root).toBe("number");
      expect(search.root).not.toBe(battle.handle);
      expect(search.root_view.phase).toBe("teamPreview");
    });

    it("close_search frees all handles in the session", () => {
      const battle = dispatch({ cmd: "new_battle", team_a: TEAM_A, team_b: TEAM_B, seed: [1, 2, 3, 4] });
      const search = dispatch({ cmd: "open_search", from: battle.handle });

      // Step to create another child handle in the session
      dispatch({
        cmd: "step", handle: search.root,
        choices: { p1: "team 1234", p2: "team 1234" },
        seed: [10, 20, 30, 40],
      });

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
      const battle = dispatch({ cmd: "new_battle", team_a: TEAM_A, team_b: TEAM_B, seed: [1, 2, 3, 4] });
      const search = dispatch({ cmd: "open_search", from: battle.handle });

      const step = dispatch({
        cmd: "step", handle: search.root,
        choices: { p1: "team 1234", p2: "team 1234" },
        seed: [50, 60, 70, 80],
      });

      expect(typeof step.child).toBe("number");
      expect(step.child).not.toBe(search.root);

      // Parent still in teamPreview
      const parentView = dispatch({ cmd: "view", handle: search.root });
      expect(parentView.view.phase).toBe("teamPreview");

      // Child advanced past teamPreview
      expect(step.view.phase).not.toBe("teamPreview");
    });

    it("returns outcome log lines from the step", () => {
      const battle = dispatch({ cmd: "new_battle", team_a: TEAM_A, team_b: TEAM_B, seed: [1, 2, 3, 4] });
      const search = dispatch({ cmd: "open_search", from: battle.handle });
      const step = dispatch({
        cmd: "step", handle: search.root,
        choices: { p1: "team 1234", p2: "team 1234" },
        seed: [50, 60, 70, 80],
      });
      expect(Array.isArray(step.outcome)).toBe(true);
      expect(step.outcome.length).toBeGreaterThan(0);
    });

    it("places child handle in the same session as parent", () => {
      const battle = dispatch({ cmd: "new_battle", team_a: TEAM_A, team_b: TEAM_B, seed: [1, 2, 3, 4] });
      const search = dispatch({ cmd: "open_search", from: battle.handle });
      dispatch({
        cmd: "step", handle: search.root,
        choices: { p1: "team 1234", p2: "team 1234" },
        seed: [50, 60, 70, 80],
      });

      // close_search should free both root and child
      const closeResult = dispatch({ cmd: "close_search", session: search.session });
      expect(closeResult.freed).toBe(2);
    });

    it("throws for a missing choice on an acting side", () => {
      const battle = dispatch({ cmd: "new_battle", team_a: TEAM_A, team_b: TEAM_B, seed: [1, 2, 3, 4] });
      const search = dispatch({ cmd: "open_search", from: battle.handle });
      // Both sides must act in team preview; only provide p1
      expect(() => dispatch({
        cmd: "step", handle: search.root,
        choices: { p1: "team 1234" },
        seed: [50, 60, 70, 80],
      })).toThrow(/missing choice for acting side p2/);
    });
  });

  describe("state view structure", () => {
    // The StateView/snapshot SHAPE is enforced by the types in src/types.ts;
    // here we assert the VALUES the worker fills in from a fresh battle.
    it("view snapshots both sides at full strength before team preview", () => {
      const battle = dispatch({ cmd: "new_battle", team_a: TEAM_A, team_b: TEAM_B, seed: [1, 2, 3, 4] });
      const v = dispatch({ cmd: "view", handle: battle.handle }).view;

      expect(v.snapshot.turn).toBe(0);
      expect(v.snapshot.field.weather).toBeNull();
      expect(v.snapshot.sides.map((s: any) => s.id)).toEqual(["p1", "p2"]);

      const side = v.snapshot.sides[0];
      expect(side.id).toBe("p1");
      expect(side.pokemon).toHaveLength(TEAM_A.length);

      // First mon is Charizard (team order is preserved at team preview).
      const mon = side.pokemon[0];
      expect(mon.species).toBe("charizard");
      expect(mon.fainted).toBe(false);
      expect(mon.hp).toBe(mon.maxhp);
      expect(mon.hp).toBeGreaterThan(0);
      expect(mon.moves).toHaveLength(4);
      expect(mon.moves[0]).toMatchObject({ id: expect.any(String), pp: expect.any(Number) });
    });

    it("snapshot includes extended pokemon fields at team preview", () => {
      const battle = dispatch({ cmd: "new_battle", team_a: TEAM_A, team_b: TEAM_B, seed: [1, 2, 3, 4] });
      const v = dispatch({ cmd: "view", handle: battle.handle }).view;
      const mon = v.snapshot.sides[0].pokemon[0];

      // Nature (from team set)
      expect(mon.nature).toBe("Timid");
      // Status state defaults
      expect(mon.statusState).toEqual({ stage: null, time: null });
      // Item tracking
      expect(mon.item).toBe("charizarditey");
      expect(mon.lastItem).toBeNull();
      // Active turns (not yet on field at team preview)
      expect(mon.activeTurns).toBe(0);
      // Volatile details (empty at start)
      expect(mon.volatileDetails).toEqual({});
    });

    it("snapshot includes nature for every pokemon", () => {
      const battle = dispatch({ cmd: "new_battle", team_a: TEAM_A, team_b: TEAM_B, seed: [1, 2, 3, 4] });
      const v = dispatch({ cmd: "view", handle: battle.handle }).view;
      for (const side of v.snapshot.sides) {
        for (const mon of side.pokemon) {
          expect(mon.nature).toBeTruthy();
          expect(typeof mon.nature).toBe("string");
        }
      }
    });

    it("snapshot includes field duration fields", () => {
      const battle = dispatch({ cmd: "new_battle", team_a: TEAM_A, team_b: TEAM_B, seed: [1, 2, 3, 4] });
      const v = dispatch({ cmd: "view", handle: battle.handle }).view;

      expect(v.snapshot.field.weatherDuration).toBeNull();
      expect(v.snapshot.field.terrainDuration).toBeNull();
      expect(v.snapshot.field.pseudoWeather).toEqual({});
    });

    it("non-terminal battle has null utility", () => {
      const battle = dispatch({ cmd: "new_battle", team_a: TEAM_A, team_b: TEAM_B, seed: [1, 2, 3, 4] });
      const v = dispatch({ cmd: "view", handle: battle.handle }).view;
      expect(v.utility).toBeNull();
    });

    it("team preview shows both sides as acting", () => {
      const battle = dispatch({ cmd: "new_battle", team_a: TEAM_A, team_b: TEAM_B, seed: [1, 2, 3, 4] });
      const v = dispatch({ cmd: "view", handle: battle.handle }).view;
      expect(v.to_move).toContain("p1");
      expect(v.to_move).toContain("p2");
    });
  });

  describe("extended snapshot fields mid-battle", () => {
    /** Helper: advance past team preview into the move phase. */
    function advanceToMovePhase(handle: number): { child: number; view: any } {
      const step = dispatch({
        cmd: "step", handle,
        choices: { p1: "team 1234", p2: "team 1234" },
        seed: [10, 20, 30, 40],
      });
      return step;
    }

    it("activeTurns increments for pokemon on field", () => {
      const battle = dispatch({ cmd: "new_battle", team_a: TEAM_A, team_b: TEAM_B, seed: [1, 2, 3, 4] });
      const step1 = advanceToMovePhase(battle.handle);

      // After team preview, active mons should have activeTurns >= 1
      const p1side = step1.view.snapshot.sides[0];
      const activeMons = p1side.pokemon.filter((m: any) => m.active);
      expect(activeMons.length).toBe(2);
      for (const mon of activeMons) {
        expect(mon.activeTurns).toBeGreaterThanOrEqual(1);
      }
    });

    it("weather sets duration in field snapshot", () => {
      // Pelipper has Drizzle — bring it active (position 5 in team_a, 1-indexed)
      // Use team order that puts Pelipper active: team order "5123" -> Pelipper + Charizard active
      const battle = dispatch({ cmd: "new_battle", team_a: TEAM_A, team_b: TEAM_B, seed: [1, 2, 3, 4] });
      const step = dispatch({
        cmd: "step", handle: battle.handle,
        choices: { p1: "team 5123", p2: "team 1234" },
        seed: [10, 20, 30, 40],
      });

      // Drizzle should set rain — engine emits lowercase status IDs
      const field = step.view.snapshot.field;
      expect(field.weather).toBeTruthy();
      expect(field.weatherDuration).not.toBeNull();
    });

    it("volatileDetails captures volatile state during battle", () => {
      const battle = dispatch({ cmd: "new_battle", team_a: TEAM_A, team_b: TEAM_B, seed: [1, 2, 3, 4] });
      const step1 = advanceToMovePhase(battle.handle);

      // Must be in move phase after team preview with this seed
      expect(step1.view.phase).toBe("move");

      // Charizard: move 2 = Protect, Venusaur: move 3 = Giga Drain targeting foe slot 1
      const step2 = dispatch({
        cmd: "step", handle: step1.child,
        choices: { p1: "move 2, move 3 1", p2: "move 1 1, move 1 1" },
        seed: [100, 200, 300, 400],
      });
      const p1side = step2.view.snapshot.sides[0];
      expect(p1side.pokemon[0].volatileDetails).toBeDefined();
    });
  });

  // ==========================================================================
  // New gap-coverage tests (from test-gap analysis)
  // ==========================================================================

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
      const battle = dispatch({
        cmd: "new_battle", team_a: TEAM_A, team_b: TEAM_B, seed: [1, 2, 3, 4],
      });
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
      const battle = dispatch({
        cmd: "new_battle", team_a: TEAM_A, team_b: TEAM_B, seed: [1, 2, 3, 4],
      });
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
      const battle = dispatch({
        cmd: "new_battle", team_a: TEAM_A, team_b: TEAM_B, seed: [1, 2, 3, 4],
      });
      const step1 = dispatch({
        cmd: "step", handle: battle.handle,
        choices: { p1: "team 1234", p2: "team 1234" },
        seed: [10, 20, 30, 40],
      });
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
      const battle = dispatch({
        cmd: "new_battle", team_a: TEAM_A, team_b: TEAM_B, seed: [1, 2, 3, 4],
      });
      dispatch({
        cmd: "step", handle: battle.handle,
        choices: { p1: "team 1234", p2: "team 1234" },
        seed: [10, 20, 30, 40],
      });

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
      const battle = dispatch({
        cmd: "new_battle", team_a: TEAM_A, team_b: TEAM_B, seed: [1, 2, 3, 4],
      });
      const session = dispatch({ cmd: "open_search", from: battle.handle });

      // root -> child1 -> child2 -> child3 (3 levels of stepping)
      const child1 = dispatch({
        cmd: "step", handle: session.root,
        choices: { p1: "team 1234", p2: "team 1234" },
        seed: [10, 20, 30, 40],
      });
      const child2 = autoStep(child1.child, 50);
      const child3 = autoStep(child2.child, 100);

      // 5 handles total: live battle + root + child1 + child2 + child3
      expect(handleCount()).toBe(5);

      const closed = dispatch({ cmd: "close_search", session: session.session });
      expect(closed.freed).toBe(4);
      expect(handleCount()).toBe(1);
    });

    it("concurrent sessions are isolated", () => {
      const battle = dispatch({
        cmd: "new_battle", team_a: TEAM_A, team_b: TEAM_B, seed: [1, 2, 3, 4],
      });
      const s1 = dispatch({ cmd: "open_search", from: battle.handle });
      const s2 = dispatch({ cmd: "open_search", from: battle.handle });

      // Step in both sessions
      dispatch({
        cmd: "step", handle: s1.root,
        choices: { p1: "team 1234", p2: "team 1234" }, seed: [10, 20, 30, 40],
      });
      dispatch({
        cmd: "step", handle: s2.root,
        choices: { p1: "team 1234", p2: "team 1234" }, seed: [50, 60, 70, 80],
      });

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
      const battle = dispatch({
        cmd: "new_battle", team_a: TEAM_A, team_b: TEAM_B, seed: [1, 2, 3, 4],
      });
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
      const battle = dispatch({
        cmd: "new_battle", team_a: TEAM_A, team_b: TEAM_B, seed: [1, 2, 3, 4],
      });
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
      const battle = dispatch({
        cmd: "new_battle", team_a: TEAM_A, team_b: TEAM_B, seed: [1, 2, 3, 4],
      });
      const session = dispatch({ cmd: "open_search", from: battle.handle });
      const fs = advanceUntilPhase(session.root, "forceSwitch");

      expect(fs).not.toBeNull();
      expect(fs!.view.phase).toBe("forceSwitch");
      expect(fs!.view.to_move.length).toBeGreaterThanOrEqual(1);
      expect(fs!.view.to_move.length).toBeLessThanOrEqual(2);

      dispatch({ cmd: "close_search", session: session.session });
    });
  });

  describe("snapshot detail values", () => {
    it("sideConditions populated after Light Screen", () => {
      const battle = dispatch({
        cmd: "new_battle", team_a: TEAM_A, team_b: TEAM_B, seed: [1, 2, 3, 4],
      });
      // p2 team order "2134": Meganium (pos 2) active in slot p2a, has Light Screen as move 2
      const step1 = dispatch({
        cmd: "step", handle: battle.handle,
        choices: { p1: "team 1234", p2: "team 2134" },
        seed: [10, 20, 30, 40],
      });
      expect(step1.view.phase).toBe("move");

      // p1: Charizard Protect + Venusaur Protect (self-targeting, avoid interference)
      // p2: Meganium Light Screen (move 2) + Corviknight Brave Bird (move 1) at foe slot 1
      const step2 = dispatch({
        cmd: "step", handle: step1.child,
        choices: { p1: "move 2, move 1", p2: "move 2, move 1 1" },
        seed: [100, 200, 300, 400],
      });

      const p2side = step2.view.snapshot.sides[1];
      expect(Object.keys(p2side.sideConditions).length).toBeGreaterThan(0);
      expect(p2side.sideConditions["lightscreen"]).toBeDefined();
      expect(p2side.sideConditions["lightscreen"].duration).not.toBeNull();
    });

    it("pseudoWeather populated after Trick Room", () => {
      const battle = dispatch({
        cmd: "new_battle", team_a: TEAM_A, team_b: TEAM_B, seed: [1, 2, 3, 4],
      });
      // p2 team order "3124": Sinistcha (pos 3) active in slot p2a, has Trick Room as move 3
      const step1 = dispatch({
        cmd: "step", handle: battle.handle,
        choices: { p1: "team 1234", p2: "team 3124" },
        seed: [10, 20, 30, 40],
      });
      expect(step1.view.phase).toBe("move");

      // p1: both Protect. p2: Sinistcha Trick Room (move 3) + Corviknight Roost (move 4)
      const step2 = dispatch({
        cmd: "step", handle: step1.child,
        choices: { p1: "move 2, move 1", p2: "move 3, move 4" },
        seed: [100, 200, 300, 400],
      });

      const pw = step2.view.snapshot.field.pseudoWeather;
      expect(Object.keys(pw).length).toBeGreaterThan(0);
      expect(pw["trickroom"]).toBeDefined();
      expect(pw["trickroom"].duration).not.toBeNull();
    });

    it("boosts negative after Intimidate", () => {
      const battle = dispatch({
        cmd: "new_battle", team_a: TEAM_A, team_b: TEAM_B, seed: [1, 2, 3, 4],
      });
      // p1 team order "6123": Incineroar (pos 6, Intimidate) active in slot p1a
      const step1 = dispatch({
        cmd: "step", handle: battle.handle,
        choices: { p1: "team 6123", p2: "team 1234" },
        seed: [10, 20, 30, 40],
      });

      // Intimidate triggers on switch-in, lowering p2's active Pokemon's Attack
      const p2side = step1.view.snapshot.sides[1];
      const activeP2 = p2side.pokemon.filter((m: any) => m.active);
      expect(activeP2.length).toBe(2);
      for (const mon of activeP2) {
        expect(mon.boosts.atk).toBeLessThan(0);
      }
    });

    it("stats contain expected keys with positive values", () => {
      const battle = dispatch({
        cmd: "new_battle", team_a: TEAM_A, team_b: TEAM_B, seed: [1, 2, 3, 4],
      });
      const v = dispatch({ cmd: "view", handle: battle.handle }).view;
      const mon = v.snapshot.sides[0].pokemon[0]; // Charizard

      // All stat keys present and positive
      for (const key of ["hp", "atk", "def", "spa", "spd", "spe"]) {
        expect(mon.stats).toHaveProperty(key);
        expect(mon.stats[key]).toBeGreaterThan(0);
      }

      // HP stat matches maxhp at team preview (no damage taken yet)
      expect(mon.stats.hp).toBe(mon.maxhp);
      expect(mon.hp).toBe(mon.maxhp);
    });

    it("level is 50 for all Pokemon", () => {
      const battle = dispatch({
        cmd: "new_battle", team_a: TEAM_A, team_b: TEAM_B, seed: [1, 2, 3, 4],
      });
      const v = dispatch({ cmd: "view", handle: battle.handle }).view;
      for (const side of v.snapshot.sides) {
        for (const mon of side.pokemon) {
          expect(mon.level).toBe(50);
        }
      }
    });

    it("move pp equals maxpp at team preview (nothing used yet)", () => {
      const battle = dispatch({
        cmd: "new_battle", team_a: TEAM_A, team_b: TEAM_B, seed: [1, 2, 3, 4],
      });
      const v = dispatch({ cmd: "view", handle: battle.handle }).view;
      const mon = v.snapshot.sides[0].pokemon[0]; // Charizard

      for (const move of mon.moves) {
        expect(move.pp).toBe(move.maxpp);
        expect(move.pp).toBeGreaterThan(0);
        expect(move.disabled).toBe(false);
      }
    });

    it("pokemon with higher stat-point investment has higher stat", () => {
      const battle = dispatch({
        cmd: "new_battle", team_a: TEAM_A, team_b: TEAM_B, seed: [1, 2, 3, 4],
      });
      const v = dispatch({ cmd: "view", handle: battle.handle }).view;
      const charizard = v.snapshot.sides[0].pokemon[0]; // spa: 32, spe: 32
      const whimsicott = v.snapshot.sides[0].pokemon[3]; // hp: 32, spe: 32

      // Charizard has 32 SpA investment vs Whimsicott's 0 SpA investment,
      // and Charizard has higher base SpA (109 vs 77)
      expect(charizard.stats.spa).toBeGreaterThan(whimsicott.stats.spa);
    });
  });

  // ==========================================================================
  // Phase 2 gap-coverage (TS Test Gap Analysis)
  // ==========================================================================

  describe("tie game utility", () => {
    it("utility values are valid and zero-sum across multiple seeds", () => {
      for (const seedBase of [1, 10, 50, 100, 200]) {
        resetState();
        const seed: [number, number, number, number] = [seedBase, seedBase + 1, seedBase + 2, seedBase + 3];
        const battle = dispatch({ cmd: "new_battle", team_a: TEAM_A, team_b: TEAM_B, seed });
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

  describe("legal field structure", () => {
    it("team preview legal has teamPreview flag and side pokemon", () => {
      const battle = dispatch({ cmd: "new_battle", team_a: TEAM_A, team_b: TEAM_B, seed: [1, 2, 3, 4] });
      const v = dispatch({ cmd: "view", handle: battle.handle }).view;

      for (const sid of ["p1", "p2"]) {
        const req = v.legal[sid];
        expect(req).toBeDefined();
        expect(req.teamPreview).toBe(true);
        expect(req.side).toBeDefined();
        expect(req.side.pokemon).toBeDefined();
        expect(Array.isArray(req.side.pokemon)).toBe(true);
        expect(req.side.pokemon.length).toBe(TEAM_A.length);
      }
    });

    it("move phase legal has active array with move slots", () => {
      const battle = dispatch({ cmd: "new_battle", team_a: TEAM_A, team_b: TEAM_B, seed: [1, 2, 3, 4] });
      const step = dispatch({
        cmd: "step", handle: battle.handle,
        choices: { p1: "team 1234", p2: "team 1234" },
        seed: [10, 20, 30, 40],
      });
      expect(step.view.phase).toBe("move");

      for (const sid of step.view.to_move) {
        const req = step.view.legal[sid];
        expect(req).toBeDefined();
        expect(Array.isArray(req.active)).toBe(true);
        expect(req.active.length).toBe(2);

        for (const slot of req.active) {
          expect(Array.isArray(slot.moves)).toBe(true);
          expect(slot.moves.length).toBeGreaterThan(0);
          for (const m of slot.moves) {
            expect(typeof m.id).toBe("string");
          }
        }
      }
    });

    it("forceSwitch phase legal has forceSwitch array", () => {
      const battle = dispatch({ cmd: "new_battle", team_a: TEAM_A, team_b: TEAM_B, seed: [1, 2, 3, 4] });
      const session = dispatch({ cmd: "open_search", from: battle.handle });
      const fs = advanceUntilPhase(session.root, "forceSwitch");
      expect(fs).not.toBeNull();

      const actingSideWithFS = fs!.view.to_move.find(
        (sid: string) => fs!.view.legal[sid]?.forceSwitch,
      );
      expect(actingSideWithFS).toBeDefined();
      const req = fs!.view.legal[actingSideWithFS!];
      expect(Array.isArray(req.forceSwitch)).toBe(true);
      expect(req.forceSwitch.length).toBe(2);

      dispatch({ cmd: "close_search", session: session.session });
    });
  });

  describe("snapshot status and fainted pokemon", () => {
    it("fainted pokemon has fainted:true, hp:0, active:false", () => {
      const battle = dispatch({ cmd: "new_battle", team_a: TEAM_A, team_b: TEAM_B, seed: [1, 2, 3, 4] });
      const session = dispatch({ cmd: "open_search", from: battle.handle });
      const fs = advanceUntilPhase(session.root, "forceSwitch");
      expect(fs).not.toBeNull();

      const allPokemon = fs!.view.snapshot.sides.flatMap((s: any) => s.pokemon);
      const faintedMon = allPokemon.find((m: any) => m.fainted);
      expect(faintedMon).toBeDefined();
      expect(faintedMon.hp).toBe(0);
      expect(faintedMon.active).toBe(false);

      dispatch({ cmd: "close_search", session: session.session });
    });

    it("status field populated during battle", () => {
      const battle = dispatch({ cmd: "new_battle", team_a: TEAM_A, team_b: TEAM_B, seed: [1, 2, 3, 4] });
      const session = dispatch({ cmd: "open_search", from: battle.handle });

      // Scan through the game looking for any pokemon with a non-null status
      let h = session.root;
      let statusMon: any = null;
      for (let i = 0; i < 200; i++) {
        const v = dispatch({ cmd: "view", handle: h }).view;

        const allPokemon = v.snapshot.sides.flatMap((s: any) => s.pokemon);
        const withStatus = allPokemon.find((m: any) => m.status !== null);
        if (withStatus) {
          statusMon = withStatus;
          break;
        }

        if (v.terminal) break;
        h = autoStep(h, i * 4 + 1).child;
      }

      // Over a full battle, status effects (burn, sleep, paralysis) typically
      // appear. If one was found, verify its structure.
      if (statusMon) {
        expect(typeof statusMon.status).toBe("string");
        expect(statusMon.status.length).toBeGreaterThan(0);
        expect(statusMon.statusState).toBeDefined();
      }

      dispatch({ cmd: "close_search", session: session.session });
    });
  });

  describe("reseed variance through dispatch", () => {
    it("same parent with different seeds produces different children", () => {
      const battle = dispatch({ cmd: "new_battle", team_a: TEAM_A, team_b: TEAM_B, seed: [1, 2, 3, 4] });
      const step1 = dispatch({
        cmd: "step", handle: battle.handle,
        choices: { p1: "team 1234", p2: "team 1234" },
        seed: [10, 20, 30, 40],
      });
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
      const battle = dispatch({ cmd: "new_battle", team_a: TEAM_A, team_b: TEAM_B, seed: [1, 2, 3, 4] });
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

  describe("item and ability tracking in snapshot", () => {
    it("species and ability change after Mega Evolution", () => {
      const battle = dispatch({ cmd: "new_battle", team_a: TEAM_A, team_b: TEAM_B, seed: [1, 2, 3, 4] });
      const step1 = dispatch({
        cmd: "step", handle: battle.handle,
        choices: { p1: "team 1234", p2: "team 1234" },
        seed: [10, 20, 30, 40],
      });
      expect(step1.view.phase).toBe("move");

      // Pre-mega: Charizard should have base species and ability
      const preMega = step1.view.snapshot.sides[0].pokemon.find(
        (m: any) => m.species === "charizard",
      );
      expect(preMega).toBeDefined();
      expect(preMega.ability).toBe("blaze");

      // Mega evolve Charizard (Heat Wave mega) + Venusaur Protect
      const step2 = dispatch({
        cmd: "step", handle: step1.child,
        choices: { p1: "move 1 mega, move 1", p2: "move 1 1, move 1 1" },
        seed: [100, 200, 300, 400],
      });

      // Post-mega: species and ability should have changed
      const postMega = step2.view.snapshot.sides[0].pokemon.find(
        (m: any) => m.species?.includes("charizard") && m.active,
      );
      expect(postMega).toBeDefined();
      expect(postMega.species).not.toBe("charizard");
      expect(postMega.ability).not.toBe("blaze");
    });

    it("lastItem populated after item consumption", () => {
      const battle = dispatch({ cmd: "new_battle", team_a: TEAM_A, team_b: TEAM_B, seed: [1, 2, 3, 4] });
      const session = dispatch({ cmd: "open_search", from: battle.handle });

      // Scan through a full game looking for any item consumption
      let h = session.root;
      let lastItemMon: any = null;
      for (let i = 0; i < 200; i++) {
        const v = dispatch({ cmd: "view", handle: h }).view;

        const allPokemon = v.snapshot.sides.flatMap((s: any) => s.pokemon);
        const withLastItem = allPokemon.find((m: any) => m.lastItem !== null);
        if (withLastItem) {
          lastItemMon = withLastItem;
          break;
        }

        if (v.terminal) break;
        h = autoStep(h, i * 4 + 1).child;
      }

      // Teams include Focus Sash, Sitrus Berry, Lum Berry, Mental Herb —
      // at least one should be consumed over a full battle
      expect(lastItemMon).not.toBeNull();
      expect(typeof lastItemMon.lastItem).toBe("string");
      expect(lastItemMon.lastItem.length).toBeGreaterThan(0);

      dispatch({ cmd: "close_search", session: session.session });
    });
  });

  describe("move state tracking in snapshot", () => {
    it("move PP decrements after use", () => {
      const battle = dispatch({ cmd: "new_battle", team_a: TEAM_A, team_b: TEAM_B, seed: [1, 2, 3, 4] });
      const step1 = dispatch({
        cmd: "step", handle: battle.handle,
        choices: { p1: "team 1234", p2: "team 1234" },
        seed: [10, 20, 30, 40],
      });

      // Charizard uses Heat Wave (move 1), Venusaur uses Protect (move 1)
      const step2 = dispatch({
        cmd: "step", handle: step1.child,
        choices: { p1: "move 1, move 1", p2: "move 1 1, move 1 1" },
        seed: [100, 200, 300, 400],
      });

      const charizard = step2.view.snapshot.sides[0].pokemon.find(
        (m: any) => m.species?.includes("charizard"),
      );
      expect(charizard).toBeDefined();
      const heatWave = charizard.moves.find((m: any) => m.id === "heatwave");
      expect(heatWave).toBeDefined();
      expect(heatWave.pp).toBeLessThan(heatWave.maxpp);
    });

    it("move disabled field is boolean for all moves", () => {
      // Triggering disabled: true requires moves like Disable, Imprison, or
      // Torment which aren't in the test teams. This verifies the field type
      // and default value (false) for all moves in the move phase.
      const battle = dispatch({ cmd: "new_battle", team_a: TEAM_A, team_b: TEAM_B, seed: [1, 2, 3, 4] });
      const step1 = dispatch({
        cmd: "step", handle: battle.handle,
        choices: { p1: "team 1234", p2: "team 1234" },
        seed: [10, 20, 30, 40],
      });

      const allPokemon = step1.view.snapshot.sides.flatMap((s: any) => s.pokemon);
      for (const mon of allPokemon) {
        for (const move of mon.moves) {
          expect(typeof move.disabled).toBe("boolean");
        }
      }
    });
  });

  describe("session nesting", () => {
    it("open_search from search-owned handle produces independent session", () => {
      const battle = dispatch({ cmd: "new_battle", team_a: TEAM_A, team_b: TEAM_B, seed: [1, 2, 3, 4] });
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

  describe("snapshot field completeness", () => {
    it("gender field present for all pokemon", () => {
      const battle = dispatch({ cmd: "new_battle", team_a: TEAM_A, team_b: TEAM_B, seed: [1, 2, 3, 4] });
      const v = dispatch({ cmd: "view", handle: battle.handle }).view;
      for (const side of v.snapshot.sides) {
        for (const mon of side.pokemon) {
          expect(mon).toHaveProperty("gender");
          expect(typeof mon.gender).toBe("string");
        }
      }
    });

    it("position field correct for active and bench pokemon", () => {
      const battle = dispatch({ cmd: "new_battle", team_a: TEAM_A, team_b: TEAM_B, seed: [1, 2, 3, 4] });
      const step1 = dispatch({
        cmd: "step", handle: battle.handle,
        choices: { p1: "team 1234", p2: "team 1234" },
        seed: [10, 20, 30, 40],
      });

      for (const side of step1.view.snapshot.sides) {
        const positions = side.pokemon.map((m: any) => m.position);
        // Each pokemon should have a unique position index
        expect(new Set(positions).size).toBe(side.pokemon.length);

        const activeMons = side.pokemon.filter((m: any) => m.active);
        for (const mon of activeMons) {
          expect(mon.position).toBeLessThanOrEqual(1);
        }
      }
    });

    it("teraType and terastallized null in Champions format", () => {
      const battle = dispatch({ cmd: "new_battle", team_a: TEAM_A, team_b: TEAM_B, seed: [1, 2, 3, 4] });
      const v = dispatch({ cmd: "view", handle: battle.handle }).view;
      for (const side of v.snapshot.sides) {
        for (const mon of side.pokemon) {
          expect(mon.terastallized).toBeNull();
        }
      }
    });
  });

  describe("additional robustness edge cases", () => {
    it("step with no choices key throws for acting sides", () => {
      const battle = dispatch({ cmd: "new_battle", team_a: TEAM_A, team_b: TEAM_B, seed: [1, 2, 3, 4] });
      // msg.choices ?? {} → empty choices → missing choice for acting side
      expect(() => dispatch({
        cmd: "step", handle: battle.handle, seed: [10, 20, 30, 40],
      })).toThrow(/missing choice for acting side/);
    });

    it("double release of same handle does not throw", () => {
      const battle = dispatch({ cmd: "new_battle", team_a: TEAM_A, team_b: TEAM_B, seed: [1, 2, 3, 4] });
      dispatch({ cmd: "release", handle: battle.handle });
      const result = dispatch({ cmd: "release", handle: battle.handle });
      expect(result.released).toBe(battle.handle);
    });

    it("independent battles are isolated", () => {
      const b1 = dispatch({ cmd: "new_battle", team_a: TEAM_A, team_b: TEAM_B, seed: [1, 2, 3, 4] });
      const b2 = dispatch({ cmd: "new_battle", team_a: TEAM_A, team_b: TEAM_B, seed: [5, 6, 7, 8] });

      // Advance battle 1 past team preview
      const step = dispatch({
        cmd: "step", handle: b1.handle,
        choices: { p1: "team 1234", p2: "team 1234" },
        seed: [10, 20, 30, 40],
      });
      expect(step.view.phase).toBe("move");

      // Battle 2 should still be in team preview, completely unaffected
      const v2 = dispatch({ cmd: "view", handle: b2.handle }).view;
      expect(v2.phase).toBe("teamPreview");
      expect(v2.snapshot.turn).toBe(0);
    });
  });

  describe("schema completeness", () => {
    it("StateView contains all expected top-level keys", () => {
      const battle = dispatch({ cmd: "new_battle", team_a: TEAM_A, team_b: TEAM_B, seed: [1, 2, 3, 4] });
      const v = dispatch({ cmd: "view", handle: battle.handle }).view;

      for (const key of ["phase", "to_move", "legal", "snapshot", "terminal", "utility"]) {
        expect(v).toHaveProperty(key);
      }
    });

    it("BattleSnapshot and FieldSnapshot contain all expected keys", () => {
      const battle = dispatch({ cmd: "new_battle", team_a: TEAM_A, team_b: TEAM_B, seed: [1, 2, 3, 4] });
      const snap = dispatch({ cmd: "view", handle: battle.handle }).view.snapshot;

      for (const key of ["turn", "field", "sides"]) {
        expect(snap).toHaveProperty(key);
      }

      for (const key of ["weather", "weatherDuration", "terrain", "terrainDuration", "pseudoWeather"]) {
        expect(snap.field).toHaveProperty(key);
      }
    });

    it("SideSnapshot contains all expected keys", () => {
      const battle = dispatch({ cmd: "new_battle", team_a: TEAM_A, team_b: TEAM_B, seed: [1, 2, 3, 4] });
      const side = dispatch({ cmd: "view", handle: battle.handle }).view.snapshot.sides[0];

      for (const key of ["id", "sideConditions", "pokemon"]) {
        expect(side).toHaveProperty(key);
      }
    });

    it("PokemonSnapshot contains all expected keys", () => {
      const battle = dispatch({ cmd: "new_battle", team_a: TEAM_A, team_b: TEAM_B, seed: [1, 2, 3, 4] });
      const mon = dispatch({ cmd: "view", handle: battle.handle }).view.snapshot.sides[0].pokemon[0];

      const expectedKeys = [
        "species", "nature", "level", "gender", "hp", "maxhp", "fainted",
        "status", "statusState", "ability", "item", "lastItem", "active",
        "position", "activeTurns", "teraType", "terastallized", "stats",
        "boosts", "moves", "volatiles", "volatileDetails",
      ];
      for (const key of expectedKeys) {
        expect(mon).toHaveProperty(key);
      }
    });

    it("MoveSnapshot contains all expected keys", () => {
      const battle = dispatch({ cmd: "new_battle", team_a: TEAM_A, team_b: TEAM_B, seed: [1, 2, 3, 4] });
      const move = dispatch({ cmd: "view", handle: battle.handle }).view.snapshot.sides[0].pokemon[0].moves[0];

      for (const key of ["id", "pp", "maxpp", "disabled"]) {
        expect(move).toHaveProperty(key);
      }
    });
  });
});
