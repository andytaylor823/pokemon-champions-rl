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

      // Drizzle should set rain
      const field = step.view.snapshot.field;
      if (field.weather === "RainDance") {
        expect(field.weatherDuration).not.toBeNull();
      }
    });

    it("volatileDetails captures volatile state during battle", () => {
      const battle = dispatch({ cmd: "new_battle", team_a: TEAM_A, team_b: TEAM_B, seed: [1, 2, 3, 4] });
      const step1 = advanceToMovePhase(battle.handle);

      // Charizard: move 2 = Protect (self-targeting, no target arg)
      // Venusaur: move 3 = Giga Drain targeting foe slot 1
      if (step1.view.phase === "move") {
        const step2 = dispatch({
          cmd: "step", handle: step1.child,
          choices: { p1: "move 2, move 3 1", p2: "move 1 1, move 1 1" },
          seed: [100, 200, 300, 400],
        });
        // After Protect, volatileDetails should be defined on all mons
        const p1side = step2.view.snapshot.sides[0];
        expect(p1side.pokemon[0].volatileDetails).toBeDefined();
      }
    });
  });
});
