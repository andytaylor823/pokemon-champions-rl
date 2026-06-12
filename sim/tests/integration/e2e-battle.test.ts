/**
 * End-to-end battle test: runs a specific team matchup with scripted strategies
 * and asserts the expected outcome. Extracted from test-battle.ts.
 *
 * Uses the async BattleRunner (BattleStream path) to exercise the full game
 * lifecycle including replay generation.
 */
import { describe, it, expect, afterAll } from "vitest";
import * as fs from "node:fs";
import * as path from "node:path";
import * as os from "node:os";
import { BattleRunner, packTeam } from "../../src/battle-runner";
import { saveReplay } from "../../src/replay-export";
import type { Strategy } from "../../src/types";
import { TEAM_A, E2E_TEAM_B } from "../fixtures/teams";

// --- Teams (from test-battle.ts) -------------------------------------------

const teamA = packTeam(TEAM_A);
const teamB = packTeam(E2E_TEAM_B);

// --- Strategies (from test-battle.ts) --------------------------------------

function pickSwitches(request: any): string {
  const pokemon = request.side.pokemon;
  const claimed = new Set<number>();
  return request.forceSwitch
    .map((mustSwitch: boolean) => {
      if (!mustSwitch) return "pass";
      for (let j = 1; j <= pokemon.length; j++) {
        const p = pokemon[j - 1];
        if (!p.active && !p.condition.endsWith(" fnt") && !claimed.has(j)) {
          claimed.add(j);
          return `switch ${j}`;
        }
      }
      return "pass";
    })
    .join(", ");
}

/** Team A: Charizard spams Heat Wave (Mega turn 1), Venusaur Protects. */
const teamAStrategy: Strategy = (request: any) => {
  if (request.teamPreview) return "team 1234";
  if (request.forceSwitch) return pickSwitches(request);

  const choices: string[] = [];
  for (let i = 0; i < request.active.length; i++) {
    const pokemon = request.side.pokemon[i];
    const active = request.active[i];

    if (pokemon.condition.endsWith(" fnt") || pokemon.commanding) {
      choices.push("pass");
      continue;
    }

    const isCharizard = pokemon.ident.includes("Charizard");
    const isVenusaur = pokemon.ident.includes("Venusaur");

    if (isCharizard) {
      const heatWaveIdx = active.moves.findIndex((m: any) => m.id === "heatwave");
      const moveNum = heatWaveIdx >= 0 ? heatWaveIdx + 1 : 1;
      const mega = active.canMegaEvo ? " mega" : "";
      choices.push(`move ${moveNum}${mega}`);
    } else if (isVenusaur) {
      const protectIdx = active.moves.findIndex((m: any) => m.id === "protect");
      const moveNum = protectIdx >= 0 ? protectIdx + 1 : 1;
      choices.push(`move ${moveNum}`);
    } else {
      choices.push("move 1");
    }
  }
  return choices.join(", ");
};

/** Team B: use first non-Protect move each turn (all status/support). */
const teamBStrategy: Strategy = (request: any) => {
  if (request.teamPreview) return "team 1234";
  if (request.forceSwitch) return pickSwitches(request);

  const choices: string[] = [];
  for (let i = 0; i < request.active.length; i++) {
    const pokemon = request.side.pokemon[i];
    const active = request.active[i];

    if (pokemon.condition.endsWith(" fnt") || pokemon.commanding) {
      choices.push("pass");
      continue;
    }

    const nonProtect = active.moves.findIndex(
      (m: any) => m.id !== "protect" && !m.disabled,
    );
    const moveNum = nonProtect >= 0 ? nonProtect + 1 : 1;
    const mega = active.canMegaEvo ? " mega" : "";
    choices.push(`move ${moveNum}${mega}`);
  }
  return choices.join(", ");
};

// --- Tests -----------------------------------------------------------------

// Track temp files for cleanup
const tempFiles: string[] = [];
afterAll(() => {
  for (const f of tempFiles) {
    try { fs.unlinkSync(f); } catch { /* already gone */ }
  }
});

describe("E2E: Charizard-Y vs support team (from test-battle.ts)", () => {
  it("Team A (Charizard Heat Wave spam) wins with all 4 Pokemon alive", async () => {
    const runner = new BattleRunner({
      formatId: "gen9championsvgc2026regma",
      teamA,
      teamB,
      seed: [1, 2, 3, 4],
    });

    const result = await runner.run(teamAStrategy, teamBStrategy);

    expect(result.winner).toBe("p1");
    expect(result.p1Remaining).toBe(4);
    expect(result.turns).toBeGreaterThan(0);
    expect(result.log.length).toBeGreaterThan(0);
  });

  it("generates a valid HTML replay file", async () => {
    const runner = new BattleRunner({
      formatId: "gen9championsvgc2026regma",
      teamA,
      teamB,
      seed: [1, 2, 3, 4],
    });

    const result = await runner.run(teamAStrategy, teamBStrategy);

    const replayPath = path.join(os.tmpdir(), `vitest-replay-${Date.now()}.html`);
    tempFiles.push(replayPath);

    const savedPath = saveReplay(result, {
      formatId: "gen9championsvgc2026regma",
      p1: "Player 1",
      p2: "Player 2",
    }, replayPath);

    expect(fs.existsSync(savedPath)).toBe(true);
    const html = fs.readFileSync(savedPath, "utf-8");
    expect(html).toContain("<!DOCTYPE html>");
    expect(html).toContain("replay-embed.js");
    expect(html).toContain("|win|");
  });
});
