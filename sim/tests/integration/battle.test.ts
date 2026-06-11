/**
 * Integration tests for the Showdown battle engine: clone determinism, reseed
 * variance, and full-battle completion. Extracted from clone-spike.ts.
 *
 * These use the SYNC engine API directly (not the worker subprocess), so they
 * run in-process and are fast (~100s of ms), but exercise the real battle
 * engine with real teams.
 */
import { describe, it, expect } from "vitest";
import { packTeam } from "../../src/battle-runner";

// eslint-disable-next-line @typescript-eslint/no-var-requires
const { Battle } = require("pokemon-showdown");
const { State } = require("pokemon-showdown/dist/sim/state");

const FORMAT_ID = "gen9championsvgc2026regma";
const ROOT_SEED: [number, number, number, number] = [1, 2, 3, 4];
const ALT_SEED: [number, number, number, number] = [9, 8, 7, 6];
const CLONE_AT_TURN = 3;
const MAX_STEPS = 200;

const TEAM_A = packTeam([
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
]);

const TEAM_B = packTeam([
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
]);

function makeBattle(seed: [number, number, number, number]): any {
  const battle = new Battle({ formatid: FORMAT_ID, seed });
  battle.setPlayer("p1", { name: "P1", team: TEAM_A });
  battle.setPlayer("p2", { name: "P2", team: TEAM_B });
  return battle;
}

/** Auto-choose until the given turn (if set) or the battle ends. */
function advance(battle: any, untilTurn: number | null): void {
  let steps = 0;
  while (!battle.ended && steps < MAX_STEPS) {
    if (untilTurn !== null && battle.turn >= untilTurn) return;
    battle.makeChoices();
    steps++;
  }
}

/** Serialize -> JSON round-trip -> deserialize, mimicking the real IPC wire. */
function cloneViaJson(battle: any): any {
  const json = JSON.stringify(State.serializeBattle(battle));
  return State.deserializeBattle(JSON.parse(json));
}

describe("clone determinism (from clone-spike.ts)", () => {
  it("a restored clone continues identically to the original", () => {
    const root = makeBattle(ROOT_SEED);
    advance(root, CLONE_AT_TURN);

    // Snapshot mid-battle
    const snapshotJson = JSON.stringify(State.serializeBattle(root));

    // Continue the original to the end
    advance(root, null);
    const originalLog = root.log.join("\n");

    // Restore from snapshot and continue — should produce the same log
    const restored = State.deserializeBattle(JSON.parse(snapshotJson));
    advance(restored, null);
    const restoredLog = restored.log.join("\n");

    expect(restoredLog).toBe(originalLog);
  });

  it("two independent restores from the same snapshot agree", () => {
    const root = makeBattle(ROOT_SEED);
    advance(root, CLONE_AT_TURN);
    const snapshotJson = JSON.stringify(State.serializeBattle(root));

    const r1 = State.deserializeBattle(JSON.parse(snapshotJson));
    advance(r1, null);
    const r2 = State.deserializeBattle(JSON.parse(snapshotJson));
    advance(r2, null);

    expect(r1.log.join("\n")).toBe(r2.log.join("\n"));
  });
});

describe("reseed variance (from clone-spike.ts)", () => {
  it("a reseeded clone diverges from the original", () => {
    const root = makeBattle(ROOT_SEED);
    advance(root, CLONE_AT_TURN);
    const snapshotJson = JSON.stringify(State.serializeBattle(root));

    // Continue original
    advance(root, null);
    const originalLog = root.log.join("\n");

    // Restore and reseed with a different seed
    const reseeded = State.deserializeBattle(JSON.parse(snapshotJson));
    reseeded.resetRNG(ALT_SEED);
    advance(reseeded, null);
    const reseededLog = reseeded.log.join("\n");

    expect(reseededLog).not.toBe(originalLog);
  });
});

describe("full battle completion", () => {
  it("a battle with auto-choices runs to terminal", () => {
    const battle = makeBattle(ROOT_SEED);
    advance(battle, null);
    expect(battle.ended).toBe(true);
    // battle.winner is the player name string (e.g. "P1", "P2") or "" for tie
    expect(typeof battle.winner).toBe("string");
  });

  it("serialized snapshot is reasonably sized", () => {
    const battle = makeBattle(ROOT_SEED);
    advance(battle, CLONE_AT_TURN);
    const snapshotJson = JSON.stringify(State.serializeBattle(battle));
    const sizeKiB = snapshotJson.length / 1024;
    // The spike measured ~30 KiB; allow generous headroom
    expect(sizeKiB).toBeLessThan(200);
    expect(sizeKiB).toBeGreaterThan(1);
  });
});
