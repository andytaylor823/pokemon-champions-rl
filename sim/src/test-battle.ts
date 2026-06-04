import { BattleRunner, packTeam } from "./battle-runner";
import { saveReplay } from "./replay-export";
import type { Strategy } from "./types";

const teamA = packTeam([
  {
    species: "Charizard",
    item: "Charizardite Y",
    ability: "Blaze",
    moves: ["Heat Wave", "Protect", "Air Slash", "Solar Beam"],
    nature: "Timid",
    statPoints: { hp: 2, atk: 0, def: 0, spa: 32, spd: 0, spe: 32 },
  },
  {
    species: "Venusaur",
    item: "Lum Berry",
    ability: "Chlorophyll",
    moves: ["Protect", "Sleep Powder", "Giga Drain", "Sludge Bomb"],
    nature: "Modest",
    statPoints: { hp: 2, atk: 0, def: 0, spa: 32, spd: 0, spe: 32 },
  },
  {
    species: "Garchomp",
    item: "Choice Scarf",
    ability: "Rough Skin",
    moves: ["Earthquake", "Dragon Claw", "Rock Slide", "Protect"],
    nature: "Jolly",
    statPoints: { hp: 2, atk: 32, def: 0, spa: 0, spd: 0, spe: 32 },
  },
  {
    species: "Whimsicott",
    item: "Mental Herb",
    ability: "Prankster",
    moves: ["Tailwind", "Helping Hand", "Encore", "Protect"],
    nature: "Timid",
    statPoints: { hp: 32, atk: 0, def: 2, spa: 0, spd: 0, spe: 32 },
  },
  {
    species: "Pelipper",
    item: "Wacan Berry",
    ability: "Drizzle",
    moves: ["Hydro Pump", "Hurricane", "Tailwind", "Protect"],
    nature: "Bold",
    statPoints: { hp: 32, atk: 0, def: 32, spa: 0, spd: 2, spe: 0 },
  },
  {
    species: "Incineroar",
    item: "Sitrus Berry",
    ability: "Intimidate",
    moves: ["Flare Blitz", "Darkest Lariat", "Fake Out", "Parting Shot"],
    nature: "Adamant",
    statPoints: { hp: 32, atk: 32, def: 0, spa: 0, spd: 2, spe: 0 },
  },
]);

const teamB = packTeam([
  {
    species: "Corviknight",
    item: "Leftovers",
    ability: "Pressure",
    moves: ["Protect", "Tailwind", "Iron Defense", "Roost"],
    nature: "Careful",
    statPoints: { hp: 32, atk: 0, def: 2, spa: 0, spd: 32, spe: 0 },
  },
  {
    species: "Meganium",
    item: "Meganiumite",
    ability: "Overgrow",
    moves: ["Protect", "Light Screen", "Reflect", "Synthesis"],
    nature: "Bold",
    statPoints: { hp: 32, atk: 0, def: 32, spa: 0, spd: 2, spe: 0 },
  },
  {
    species: "Sinistcha",
    item: "Focus Sash",
    ability: "Hospitality",
    moves: ["Protect", "Rage Powder", "Trick Room", "Life Dew"],
    nature: "Bold",
    statPoints: { hp: 32, atk: 0, def: 32, spa: 0, spd: 2, spe: 0 },
  },
  {
    species: "Kingambit",
    item: "Chople Berry",
    ability: "Defiant",
    moves: ["Protect", "Swords Dance", "Iron Defense", "Thunder Wave"],
    nature: "Careful",
    statPoints: { hp: 32, atk: 0, def: 2, spa: 0, spd: 32, spe: 0 },
  },
  {
    species: "Meowstic",
    item: "Kasib Berry",
    ability: "Prankster",
    moves: ["Protect", "Light Screen", "Reflect", "Helping Hand"],
    nature: "Timid",
    statPoints: { hp: 32, atk: 0, def: 0, spa: 2, spd: 0, spe: 32 },
  },
  {
    species: "Talonflame",
    item: "Sharp Beak",
    ability: "Gale Wings",
    moves: ["Protect", "Roost", "Feather Dance", "Bulk Up"],
    nature: "Jolly",
    statPoints: { hp: 32, atk: 0, def: 2, spa: 0, spd: 0, spe: 32 },
  },
]);

/**
 * Shared forceSwitch handler that avoids picking the same back-Pokemon
 * for two slots in a double-faint scenario.
 */
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

/**
 * Team A: Charizard spams Heat Wave (Mega on turn 1), Venusaur Protects.
 * Back Pokemon use their first attacking move if forced in.
 */
const teamAStrategy: Strategy = (request: any) => {
  if (request.teamPreview) return "team 1234";

  if (request.forceSwitch) {
    return pickSwitches(request);
  }

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
      const heatWaveIdx = active.moves.findIndex(
        (m: any) => m.id === "heatwave"
      );
      const moveNum = heatWaveIdx >= 0 ? heatWaveIdx + 1 : 1;
      const mega = active.canMegaEvo ? " mega" : "";
      choices.push(`move ${moveNum}${mega}`);
    } else if (isVenusaur) {
      const protectIdx = active.moves.findIndex(
        (m: any) => m.id === "protect"
      );
      const moveNum = protectIdx >= 0 ? protectIdx + 1 : 1;
      choices.push(`move ${moveNum}`);
    } else {
      choices.push("move 1");
    }
  }

  return choices.join(", ");
};

/**
 * Team B: use first non-Protect move each turn (all non-attacking).
 */
const teamBStrategy: Strategy = (request: any) => {
  if (request.teamPreview) return "team 1234";

  if (request.forceSwitch) {
    return pickSwitches(request);
  }

  const choices: string[] = [];
  for (let i = 0; i < request.active.length; i++) {
    const pokemon = request.side.pokemon[i];
    const active = request.active[i];

    if (pokemon.condition.endsWith(" fnt") || pokemon.commanding) {
      choices.push("pass");
      continue;
    }

    const nonProtect = active.moves.findIndex(
      (m: any) => m.id !== "protect" && !m.disabled
    );
    const moveNum = nonProtect >= 0 ? nonProtect + 1 : 1;

    const mega = active.canMegaEvo ? " mega" : "";
    choices.push(`move ${moveNum}${mega}`);
  }

  return choices.join(", ");
};

async function main() {
  const runner = new BattleRunner({
    formatId: "gen9championsvgc2026regma",
    teamA,
    teamB,
    seed: [1, 2, 3, 4],
  });

  console.log("=== Champions Reg M-A Test Battle ===");
  console.log("Team A: M-Charizard Y / Venusaur / Garchomp / Whimsicott / Pelipper / Incineroar");
  console.log("Team B: Corviknight / M-Meganium / Sinistcha / Kingambit / Meowstic / Talonflame");
  console.log("Strategy: Charizard Heat Wave spam + Venusaur Protect | Team B uses status moves");
  console.log("");

  const result = await runner.run(teamAStrategy, teamBStrategy);

  console.log("");
  console.log("=== Battle Log ===");
  const interestingPrefixes = [
    "|turn|", "|switch|", "|move|", "|-mega|", "|-weather|",
    "|-supereffective|", "|-damage|", "|faint|", "|-heal|",
    "|win|", "|tie|", "|-ability|", "|-enditem|", "|-status|",
    "|detailschange|", "|-singleturn|", "|-activate|",
  ];
  for (const line of result.log) {
    if (interestingPrefixes.some((p) => line.startsWith(p))) {
      console.log(line);
    }
  }

  console.log("");
  console.log("=== Result ===");
  console.log(`Winner: ${result.winner}`);
  console.log(`Turns: ${result.turns}`);
  console.log(`Team A remaining: ${result.p1Remaining}`);
  console.log(`Team B remaining: ${result.p2Remaining}`);

  const replayPath = saveReplay(result, {
    formatId: "gen9championsvgc2026regma",
    p1: "Player 1",
    p2: "Player 2",
  });
  console.log(`Replay saved: ${replayPath}`);

  const success = result.winner === "p1" && result.p1Remaining === 4;
  console.log("");
  if (success) {
    console.log("TEST PASSED: Team A wins with all 4 Pokemon alive");
  } else {
    console.log("TEST FAILED:");
    if (result.winner !== "p1") console.log(`  Expected winner p1, got ${result.winner}`);
    if (result.p1Remaining !== 4) console.log(`  Expected 4 remaining, got ${result.p1Remaining}`);
    process.exit(1);
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
