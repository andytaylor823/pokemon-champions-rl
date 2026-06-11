/**
 * Unit tests for sim-worker internals. These import the module directly
 * (no subprocess) thanks to the `require.main === module` guard.
 */
import { describe, it, expect, beforeEach } from "vitest";
import {
  dispatch,
  resetState,
  battles,
  sessions,
} from "../src/sim-worker";

// Reusable Champions-format teams (structured PokemonSet objects)
const TEAM_A = [
  { species: "Charizard", item: "Charizardite Y", ability: "Blaze",
    moves: ["Heat Wave", "Protect", "Air Slash", "Solar Beam"],
    nature: "Timid", statPoints: { hp: 2, atk: 0, def: 0, spa: 32, spd: 0, spe: 32 } },
  { species: "Venusaur", item: "Lum Berry", ability: "Chlorophyll",
    moves: ["Protect", "Sleep Powder", "Giga Drain", "Sludge Bomb"],
    nature: "Modest", statPoints: { hp: 2, atk: 0, def: 0, spa: 32, spd: 0, spe: 32 } },
  { species: "Garchomp", item: "Choice Scarf", ability: "Rough Skin",
    moves: ["Earthquake", "Dragon Claw", "Rock Slide", "Protect"],
    nature: "Jolly", statPoints: { hp: 2, atk: 32, def: 0, spa: 0, spd: 0, spe: 32 } },
  { species: "Whimsicott", item: "Mental Herb", ability: "Prankster",
    moves: ["Tailwind", "Helping Hand", "Encore", "Protect"],
    nature: "Timid", statPoints: { hp: 32, atk: 0, def: 2, spa: 0, spd: 0, spe: 32 } },
  { species: "Pelipper", item: "Wacan Berry", ability: "Drizzle",
    moves: ["Hydro Pump", "Hurricane", "Tailwind", "Protect"],
    nature: "Bold", statPoints: { hp: 32, atk: 0, def: 32, spa: 0, spd: 2, spe: 0 } },
  { species: "Incineroar", item: "Sitrus Berry", ability: "Intimidate",
    moves: ["Flare Blitz", "Darkest Lariat", "Fake Out", "Parting Shot"],
    nature: "Adamant", statPoints: { hp: 32, atk: 32, def: 0, spa: 0, spd: 2, spe: 0 } },
];

const TEAM_B = [
  { species: "Corviknight", item: "Leftovers", ability: "Pressure",
    moves: ["Brave Bird", "Tailwind", "Iron Defense", "Roost"],
    nature: "Careful", statPoints: { hp: 32, atk: 0, def: 2, spa: 0, spd: 32, spe: 0 } },
  { species: "Meganium", item: "Meganiumite", ability: "Overgrow",
    moves: ["Body Press", "Light Screen", "Reflect", "Synthesis"],
    nature: "Bold", statPoints: { hp: 32, atk: 0, def: 32, spa: 0, spd: 2, spe: 0 } },
  { species: "Sinistcha", item: "Focus Sash", ability: "Hospitality",
    moves: ["Matcha Gotcha", "Rage Powder", "Trick Room", "Life Dew"],
    nature: "Bold", statPoints: { hp: 32, atk: 0, def: 32, spa: 0, spd: 2, spe: 0 } },
  { species: "Kingambit", item: "Chople Berry", ability: "Defiant",
    moves: ["Kowtow Cleave", "Swords Dance", "Iron Defense", "Sucker Punch"],
    nature: "Adamant", statPoints: { hp: 32, atk: 32, def: 2, spa: 0, spd: 0, spe: 0 } },
  { species: "Meowstic", item: "Kasib Berry", ability: "Prankster",
    moves: ["Psychic", "Light Screen", "Reflect", "Helping Hand"],
    nature: "Timid", statPoints: { hp: 32, atk: 0, def: 0, spa: 2, spd: 0, spe: 32 } },
  { species: "Talonflame", item: "Sharp Beak", ability: "Gale Wings",
    moves: ["Brave Bird", "Roost", "Feather Dance", "Bulk Up"],
    nature: "Jolly", statPoints: { hp: 32, atk: 0, def: 2, spa: 0, spd: 0, spe: 32 } },
];

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
      expect(battles.size).toBe(1);
      dispatch({ cmd: "release", handle: battle.handle });
      expect(battles.size).toBe(0);
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
      expect(battles.size).toBe(3);
      const closeResult = dispatch({ cmd: "close_search", session: search.session });
      expect(closeResult.freed).toBe(2); // root + child

      // Only the live battle remains
      expect(battles.size).toBe(1);
      expect(() => dispatch({ cmd: "view", handle: battle.handle })).not.toThrow();
    });

    it("open_search without from creates an empty session", () => {
      const search = dispatch({ cmd: "open_search" });
      expect(search.session).toBeDefined();
      expect(search.root).toBeNull();
      expect(search.root_view).toBeNull();
      expect(sessions.size).toBe(1);
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
    it("view contains all expected fields", () => {
      const battle = dispatch({ cmd: "new_battle", team_a: TEAM_A, team_b: TEAM_B, seed: [1, 2, 3, 4] });
      const v = dispatch({ cmd: "view", handle: battle.handle }).view;

      // Top-level StateView fields
      expect(v).toHaveProperty("phase");
      expect(v).toHaveProperty("to_move");
      expect(v).toHaveProperty("legal");
      expect(v).toHaveProperty("snapshot");
      expect(v).toHaveProperty("terminal");
      expect(v).toHaveProperty("utility");

      // Snapshot structure
      expect(v.snapshot).toHaveProperty("turn");
      expect(v.snapshot).toHaveProperty("field");
      expect(v.snapshot).toHaveProperty("sides");
      expect(v.snapshot.sides).toHaveLength(2);

      // Field structure
      expect(v.snapshot.field).toHaveProperty("weather");
      expect(v.snapshot.field).toHaveProperty("terrain");
      expect(v.snapshot.field).toHaveProperty("pseudoWeather");

      // Side structure
      const side = v.snapshot.sides[0];
      expect(side).toHaveProperty("id");
      expect(side).toHaveProperty("sideConditions");
      expect(side).toHaveProperty("pokemon");
      expect(side.pokemon.length).toBeGreaterThan(0);

      // Pokemon structure
      const mon = side.pokemon[0];
      expect(mon).toHaveProperty("species");
      expect(mon).toHaveProperty("hp");
      expect(mon).toHaveProperty("maxhp");
      expect(mon).toHaveProperty("fainted");
      expect(mon).toHaveProperty("status");
      expect(mon).toHaveProperty("ability");
      expect(mon).toHaveProperty("item");
      expect(mon).toHaveProperty("stats");
      expect(mon).toHaveProperty("boosts");
      expect(mon).toHaveProperty("moves");
      expect(mon).toHaveProperty("volatiles");
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
});
