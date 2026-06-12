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
import { TEAM_A as TEAM_A_SETS, TEAM_B as TEAM_B_SETS } from "../fixtures/teams";

// eslint-disable-next-line @typescript-eslint/no-var-requires
const { Battle } = require("pokemon-showdown");
const { State } = require("pokemon-showdown/dist/sim/state");

const FORMAT_ID = "gen9championsvgc2026regma";
const ROOT_SEED: [number, number, number, number] = [1, 2, 3, 4];
const ALT_SEED: [number, number, number, number] = [9, 8, 7, 6];
const CLONE_AT_TURN = 3;
const MAX_STEPS = 200;

const TEAM_A = packTeam(TEAM_A_SETS);
const TEAM_B = packTeam(TEAM_B_SETS);

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

/**
 * Join a battle's log for comparison, dropping Showdown's `|t:|` timer lines.
 * Those carry the wall-clock seconds at which each turn was processed, so two
 * continuations that straddle a one-second boundary differ on timestamps alone.
 * We compare game events, not clocks.
 */
function gameLog(battle: any): string {
  return battle.log.filter((line: string) => !line.startsWith("|t:|")).join("\n");
}

describe("clone determinism (from clone-spike.ts)", () => {
  it("a restored clone continues identically to the original", () => {
    const root = makeBattle(ROOT_SEED);
    advance(root, CLONE_AT_TURN);

    // Snapshot mid-battle
    const snapshotJson = JSON.stringify(State.serializeBattle(root));

    // Continue the original to the end
    advance(root, null);
    const originalLog = gameLog(root);

    // Restore from snapshot and continue — should produce the same log
    const restored = State.deserializeBattle(JSON.parse(snapshotJson));
    advance(restored, null);
    const restoredLog = gameLog(restored);

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

    expect(gameLog(r1)).toBe(gameLog(r2));
  });
});

describe("reseed variance (from clone-spike.ts)", () => {
  it("a reseeded clone diverges from the original", () => {
    const root = makeBattle(ROOT_SEED);
    advance(root, CLONE_AT_TURN);
    const snapshotJson = JSON.stringify(State.serializeBattle(root));

    // Continue original
    advance(root, null);
    const originalLog = gameLog(root);

    // Restore and reseed with a different seed
    const reseeded = State.deserializeBattle(JSON.parse(snapshotJson));
    reseeded.resetRNG(ALT_SEED);
    advance(reseeded, null);
    const reseededLog = gameLog(reseeded);

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
