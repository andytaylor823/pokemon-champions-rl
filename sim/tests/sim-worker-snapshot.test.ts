/**
 * sim-worker snapshot/view tests — the VALUES the worker fills into StateView
 * and BattleSnapshot from live battles (field, status, items, moves, boosts),
 * plus schema-completeness key checks. Driven through the shared harness.
 */
import { describe, it, expect, beforeEach } from "vitest";
import { dispatch, resetState } from "../src/sim-worker";
import { TEAM_A } from "./fixtures/teams";
import {
  freshBattle,
  advancePastTeamPreview,
  autoStep,
  advanceUntilPhase,
} from "./helpers/worker-harness";

beforeEach(() => resetState());

describe("state view structure", () => {
  // The StateView/snapshot SHAPE is enforced by the types in src/types.ts;
  // here we assert the VALUES the worker fills in from a fresh battle.
  it("view snapshots both sides at full strength before team preview", () => {
    const battle = freshBattle();
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
    const battle = freshBattle();
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
    const battle = freshBattle();
    const v = dispatch({ cmd: "view", handle: battle.handle }).view;
    for (const side of v.snapshot.sides) {
      for (const mon of side.pokemon) {
        expect(mon.nature).toBeTruthy();
        expect(typeof mon.nature).toBe("string");
      }
    }
  });

  it("snapshot includes field duration fields", () => {
    const battle = freshBattle();
    const v = dispatch({ cmd: "view", handle: battle.handle }).view;

    expect(v.snapshot.field.weatherDuration).toBeNull();
    expect(v.snapshot.field.terrainDuration).toBeNull();
    expect(v.snapshot.field.pseudoWeather).toEqual({});
  });

  it("non-terminal battle has null utility", () => {
    const battle = freshBattle();
    const v = dispatch({ cmd: "view", handle: battle.handle }).view;
    expect(v.utility).toBeNull();
  });

  it("team preview shows both sides as acting", () => {
    const battle = freshBattle();
    const v = dispatch({ cmd: "view", handle: battle.handle }).view;
    expect(v.to_move).toContain("p1");
    expect(v.to_move).toContain("p2");
  });
});

describe("extended snapshot fields mid-battle", () => {
  it("activeTurns increments for pokemon on field", () => {
    const battle = freshBattle();
    const step1 = advancePastTeamPreview(battle.handle);

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
    const battle = freshBattle();
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
    const battle = freshBattle();
    const step1 = advancePastTeamPreview(battle.handle);

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

describe("snapshot detail values", () => {
  it("sideConditions populated after Light Screen", () => {
    const battle = freshBattle();
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
    const battle = freshBattle();
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
    const battle = freshBattle();
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
    const battle = freshBattle();
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
    const battle = freshBattle();
    const v = dispatch({ cmd: "view", handle: battle.handle }).view;
    for (const side of v.snapshot.sides) {
      for (const mon of side.pokemon) {
        expect(mon.level).toBe(50);
      }
    }
  });

  it("move pp equals maxpp at team preview (nothing used yet)", () => {
    const battle = freshBattle();
    const v = dispatch({ cmd: "view", handle: battle.handle }).view;
    const mon = v.snapshot.sides[0].pokemon[0]; // Charizard

    for (const move of mon.moves) {
      expect(move.pp).toBe(move.maxpp);
      expect(move.pp).toBeGreaterThan(0);
      expect(move.disabled).toBe(false);
    }
  });

  it("pokemon with higher stat-point investment has higher stat", () => {
    const battle = freshBattle();
    const v = dispatch({ cmd: "view", handle: battle.handle }).view;
    const charizard = v.snapshot.sides[0].pokemon[0]; // spa: 32, spe: 32
    const whimsicott = v.snapshot.sides[0].pokemon[3]; // hp: 32, spe: 32

    // Charizard has 32 SpA investment vs Whimsicott's 0 SpA investment,
    // and Charizard has higher base SpA (109 vs 77)
    expect(charizard.stats.spa).toBeGreaterThan(whimsicott.stats.spa);
  });
});

describe("legal field structure", () => {
  it("team preview legal has teamPreview flag and side pokemon", () => {
    const battle = freshBattle();
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
    const battle = freshBattle();
    const step = advancePastTeamPreview(battle.handle);
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
    const battle = freshBattle();
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
    const battle = freshBattle();
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

  it("status field populated after Sleep Powder lands", () => {
    const battle = freshBattle();
    // Put Venusaur (pos 2, has Sleep Powder as move 2) active in p1a slot
    const step1 = dispatch({
      cmd: "step", handle: battle.handle,
      choices: { p1: "team 2134", p2: "team 1234" },
      seed: [10, 20, 30, 40],
    });
    expect(step1.view.phase).toBe("move");

    // Try Sleep Powder across several turns with different seeds until it
    // lands — 75% accuracy means it almost always hits on the first try.
    const session = dispatch({ cmd: "open_search", from: step1.child });
    let h = session.root;
    let statusMon: any = null;
    for (let attempt = 0; attempt < 5; attempt++) {
      const seed: [number, number, number, number] = [
        attempt * 100 + 1, attempt * 100 + 2, attempt * 100 + 3, attempt * 100 + 4,
      ];
      // Venusaur Sleep Powder (move 2) targeting opponent slot 1,
      // Charizard Protect (move 2) to avoid interference
      const step = dispatch({
        cmd: "step", handle: h,
        choices: { p1: "move 2 1, move 2", p2: "move 4, move 4" },
        seed,
      });

      const p2side = step.view.snapshot.sides[1];
      const withStatus = p2side.pokemon.find((m: any) => m.status !== null);
      if (withStatus) {
        statusMon = withStatus;
        break;
      }
      if (step.view.terminal) break;
      h = step.child;
    }

    expect(statusMon).not.toBeNull();
    expect(typeof statusMon.status).toBe("string");
    expect(statusMon.status.length).toBeGreaterThan(0);
    expect(statusMon.statusState).toBeDefined();

    dispatch({ cmd: "close_search", session: session.session });
  });
});

describe("item and ability tracking in snapshot", () => {
  it("species and ability change after Mega Evolution", () => {
    const battle = freshBattle();
    const step1 = advancePastTeamPreview(battle.handle);
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
    const battle = freshBattle();
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
    const battle = freshBattle();
    const step1 = advancePastTeamPreview(battle.handle);

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
    const battle = freshBattle();
    const step1 = advancePastTeamPreview(battle.handle);

    const allPokemon = step1.view.snapshot.sides.flatMap((s: any) => s.pokemon);
    for (const mon of allPokemon) {
      for (const move of mon.moves) {
        expect(typeof move.disabled).toBe("boolean");
      }
    }
  });
});

describe("snapshot field completeness", () => {
  it("gender field present for all pokemon", () => {
    const battle = freshBattle();
    const v = dispatch({ cmd: "view", handle: battle.handle }).view;
    for (const side of v.snapshot.sides) {
      for (const mon of side.pokemon) {
        expect(mon).toHaveProperty("gender");
        expect(typeof mon.gender).toBe("string");
      }
    }
  });

  it("position field correct for active and bench pokemon", () => {
    const battle = freshBattle();
    const step1 = advancePastTeamPreview(battle.handle);

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

  it("terastallized null and teraType populated in Champions format", () => {
    const battle = freshBattle();
    const v = dispatch({ cmd: "view", handle: battle.handle }).view;
    for (const side of v.snapshot.sides) {
      for (const mon of side.pokemon) {
        // teraType is populated with the mon's primary type even though
        // Terastallization is unavailable in Champions Reg M-A.
        expect(typeof mon.teraType).toBe("string");
        expect(mon.terastallized).toBeNull();
      }
    }
  });
});

describe("schema completeness", () => {
  it("StateView contains all expected top-level keys", () => {
    const battle = freshBattle();
    const v = dispatch({ cmd: "view", handle: battle.handle }).view;

    for (const key of ["phase", "to_move", "legal", "snapshot", "terminal", "utility"]) {
      expect(v).toHaveProperty(key);
    }
  });

  it("BattleSnapshot and FieldSnapshot contain all expected keys", () => {
    const battle = freshBattle();
    const snap = dispatch({ cmd: "view", handle: battle.handle }).view.snapshot;

    for (const key of ["turn", "field", "sides"]) {
      expect(snap).toHaveProperty(key);
    }

    for (const key of ["weather", "weatherDuration", "terrain", "terrainDuration", "pseudoWeather"]) {
      expect(snap.field).toHaveProperty(key);
    }
  });

  it("SideSnapshot contains all expected keys", () => {
    const battle = freshBattle();
    const side = dispatch({ cmd: "view", handle: battle.handle }).view.snapshot.sides[0];

    for (const key of ["id", "sideConditions", "pokemon"]) {
      expect(side).toHaveProperty(key);
    }
  });

  it("PokemonSnapshot contains all expected keys", () => {
    const battle = freshBattle();
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
    const battle = freshBattle();
    const move = dispatch({ cmd: "view", handle: battle.handle }).view.snapshot.sides[0].pokemon[0].moves[0];

    for (const key of ["id", "pp", "maxpp", "disabled"]) {
      expect(move).toHaveProperty(key);
    }
  });
});
