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

// --- Teams (from test-battle.ts) -------------------------------------------

const teamA = packTeam([
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

const teamB = packTeam([
  { species: "Corviknight", item: "Leftovers", ability: "Pressure",
    moves: ["Protect", "Tailwind", "Iron Defense", "Roost"],
    nature: "Careful", statPoints: { hp: 32, atk: 0, def: 2, spa: 0, spd: 32, spe: 0 } },
  { species: "Meganium", item: "Meganiumite", ability: "Overgrow",
    moves: ["Protect", "Light Screen", "Reflect", "Synthesis"],
    nature: "Bold", statPoints: { hp: 32, atk: 0, def: 32, spa: 0, spd: 2, spe: 0 } },
  { species: "Sinistcha", item: "Focus Sash", ability: "Hospitality",
    moves: ["Protect", "Rage Powder", "Trick Room", "Life Dew"],
    nature: "Bold", statPoints: { hp: 32, atk: 0, def: 32, spa: 0, spd: 2, spe: 0 } },
  { species: "Kingambit", item: "Chople Berry", ability: "Defiant",
    moves: ["Protect", "Swords Dance", "Iron Defense", "Thunder Wave"],
    nature: "Careful", statPoints: { hp: 32, atk: 0, def: 2, spa: 0, spd: 32, spe: 0 } },
  { species: "Meowstic", item: "Kasib Berry", ability: "Prankster",
    moves: ["Protect", "Light Screen", "Reflect", "Helping Hand"],
    nature: "Timid", statPoints: { hp: 32, atk: 0, def: 0, spa: 2, spd: 0, spe: 32 } },
  { species: "Talonflame", item: "Sharp Beak", ability: "Gale Wings",
    moves: ["Protect", "Roost", "Feather Dance", "Bulk Up"],
    nature: "Jolly", statPoints: { hp: 32, atk: 0, def: 2, spa: 0, spd: 0, spe: 32 } },
]);

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
