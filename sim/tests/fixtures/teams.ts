/**
 * Shared Champions Reg M-A teams for the sim tests.
 *
 * TEAM_A / TEAM_B are the standard pair used by the worker and clone tests.
 * E2E_TEAM_B is the support-only variant the e2e test pairs with TEAM_A to get
 * its deterministic "Charizard sweeps with all 4 alive" outcome — keep its
 * movesets/stats byte-identical, the e2e assertions depend on them.
 */
import type { PokemonSet } from "../../src/battle-runner";

export const TEAM_A: PokemonSet[] = [
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

export const TEAM_B: PokemonSet[] = [
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

/** Support-only Team B for the e2e test (pairs with TEAM_A). */
export const E2E_TEAM_B: PokemonSet[] = [
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
];
