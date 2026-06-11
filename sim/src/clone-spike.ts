/**
 * Milestone 0 de-risk spike for SimClient (see docs/repo-architecture.md §3.1
 * and the SimClient plan). Validates the load-bearing assumptions of the
 * Python-orchestrated / TS-sim design BEFORE any worker code is written:
 *
 *   1. CLONE DETERMINISM  — State.serializeBattle -> JSON round-trip ->
 *      deserializeBattle, then continue, reproduces the ORIGINAL battle's
 *      continuation exactly (same seed => identical protocol log).
 *   2. RESEED VARIANCE    — a restored clone with resetRNG(newSeed) diverges
 *      (different damage rolls / outcomes). This is how the search will sample
 *      a chance node's variance.
 *   3. LATENCY / MEMORY   — rough ms/op for a clone+step round-trip and peak
 *      memory holding N live clones. The go/no-go for clone-heavy search.
 *
 * Run:  npx tsx src/clone-spike.ts   (from the sim/ dir)
 *
 * Uses the SYNCHRONOUS Battle API (new Battle + setPlayer + makeChoices) and
 * State serialize/deserialize — NOT the async BattleStream used by
 * battle-runner.ts. Drives turns with no-arg makeChoices() (auto-choose) so the
 * spike needs no request parsing; the real worker will supply explicit choices.
 */

import { packTeam } from "./battle-runner";

// eslint-disable-next-line @typescript-eslint/no-var-requires
const { Battle } = require("pokemon-showdown");
// State is not re-exported from the package root; import the module directly.
const { State } = require("pokemon-showdown/dist/sim/state");

const FORMAT_ID = "gen9championsvgc2026regma";
const ROOT_SEED: [number, number, number, number] = [1, 2, 3, 4];
const ALT_SEED: [number, number, number, number] = [9, 8, 7, 6];
const CLONE_AT_TURN = 3; // serialize a few turns in, mid-battle
const MAX_TURNS = 200; // safety guard against a non-terminating loop

// Two known-legal Champions Reg M-A teams (copied from test-battle.ts).
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

function makeRootBattle(seed: [number, number, number, number]): any {
  const battle = new Battle({ formatid: FORMAT_ID, seed });
  battle.setPlayer("p1", { name: "P1", team: TEAM_A });
  battle.setPlayer("p2", { name: "P2", team: TEAM_B });
  return battle;
}

/** Advance with auto-choose until `untilTurn` (if given) or the battle ends. */
function advance(battle: any, untilTurn: number | null): void {
  let steps = 0;
  while (!battle.ended && steps < MAX_TURNS) {
    if (untilTurn !== null && battle.turn >= untilTurn) return;
    battle.makeChoices(); // no args => auto-choose for every side that must act
    steps++;
  }
}

/** Serialize -> JSON string -> parse -> deserialize, mimicking the real wire. */
function cloneViaJson(battle: any): any {
  const json = JSON.stringify(State.serializeBattle(battle));
  return State.deserializeBattle(JSON.parse(json));
}

function summary(battle: any) {
  return { ended: battle.ended, winner: battle.winner, turn: battle.turn, logLen: battle.log.length };
}

function main(): void {
  console.log("=== SimClient Milestone 0 spike ===");
  console.log(`format: ${FORMAT_ID}  root seed: [${ROOT_SEED}]  clone @ turn ${CLONE_AT_TURN}\n`);

  // Build a mid-battle root and snapshot it.
  const root = makeRootBattle(ROOT_SEED);
  advance(root, CLONE_AT_TURN);
  console.log(`[root @ clone point] ${JSON.stringify(summary(root))}`);
  const snapshotJson = JSON.stringify(State.serializeBattle(root));
  console.log(`[snapshot] serialized size: ${(snapshotJson.length / 1024).toFixed(1)} KiB\n`);

  // Continue the ORIGINAL to the end — this is the reference continuation.
  advance(root, null);
  const log0 = root.log.join("\n");
  console.log(`[A] original continued:  ${JSON.stringify(summary(root))}`);

  // (1) DETERMINISM: two independent restores continue identically, and match
  //     the original's continuation exactly.
  const r1 = State.deserializeBattle(JSON.parse(snapshotJson));
  advance(r1, null);
  const log1 = r1.log.join("\n");
  const r2 = State.deserializeBattle(JSON.parse(snapshotJson));
  advance(r2, null);
  const log2 = r2.log.join("\n");
  const restoreMatchesOriginal = log1 === log0;
  const restoresAgree = log1 === log2;
  console.log(`[B] restore #1 continued: ${JSON.stringify(summary(r1))}`);
  console.log(`[C] restore #2 continued: ${JSON.stringify(summary(r2))}`);
  console.log(`    determinism: restore==original? ${restoreMatchesOriginal}  restore1==restore2? ${restoresAgree}\n`);

  // (2) RESEED VARIANCE: restore, reseed, continue — should diverge from log0.
  const r3 = State.deserializeBattle(JSON.parse(snapshotJson));
  r3.resetRNG(ALT_SEED);
  advance(r3, null);
  const log3 = r3.log.join("\n");
  const diverged = log3 !== log0;
  console.log(`[D] reseeded restore:     ${JSON.stringify(summary(r3))}`);
  console.log(`    reseed diverges from original? ${diverged}\n`);

  // (3) LATENCY: clone+one-step round-trips from the snapshot.
  const N = 500;
  const t0 = process.hrtime.bigint();
  for (let i = 0; i < N; i++) {
    const b = State.deserializeBattle(JSON.parse(snapshotJson));
    if (!b.ended) b.makeChoices();
  }
  const t1 = process.hrtime.bigint();
  const msPerOp = Number(t1 - t0) / 1e6 / N;
  console.log(`[E] clone+step latency: ${msPerOp.toFixed(3)} ms/op over ${N} ops`);

  // (3b) MEMORY: hold N live clones at once (peak footprint of a search tree).
  const held: any[] = [];
  const memBefore = process.memoryUsage().heapUsed;
  for (let i = 0; i < N; i++) held.push(State.deserializeBattle(JSON.parse(snapshotJson)));
  const memAfter = process.memoryUsage().heapUsed;
  console.log(`[F] ${N} live clones held: +${((memAfter - memBefore) / 1024 / 1024).toFixed(1)} MiB heap ` +
    `(~${((memAfter - memBefore) / 1024 / N).toFixed(1)} KiB/clone)`);
  console.log(`    (held ${held.length} to defeat dead-code elimination)\n`);

  const pass = restoreMatchesOriginal && restoresAgree && diverged;
  console.log(pass ? "SPIKE PASSED: clone is deterministic and reseed varies." :
    "SPIKE FAILED — see flags above.");
  if (!pass) process.exit(1);
}

main();
