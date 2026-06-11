import type { BattleResult, Strategy } from "./types";

const {
  BattleStream,
  getPlayerStreams,
} = require("pokemon-showdown/dist/sim/battle-stream");
const { BattlePlayer } = require("pokemon-showdown/dist/sim/battle-stream");
const { Teams } = require("pokemon-showdown");

class StrategyPlayer extends BattlePlayer {
  private strategy: Strategy;
  private megaUsed = false;

  constructor(
    playerStream: any,
    strategy: Strategy,
    debug = false
  ) {
    super(playerStream, debug);
    this.strategy = strategy;
  }

  receiveError(error: Error) {
    if (error.message.startsWith("[Unavailable choice]")) return;
    throw error;
  }

  receiveRequest(request: any) {
    if (request.wait) return;

    const choice = this.strategy(request);
    this.choose(choice);
  }
}

export interface BattleRunnerOptions {
  formatId: string;
  teamA: string;
  teamB: string;
  seed?: [number, number, number, number];
}

export class BattleRunner {
  private options: BattleRunnerOptions;

  constructor(options: BattleRunnerOptions) {
    this.options = options;
  }

  async run(p1Strategy: Strategy, p2Strategy: Strategy): Promise<BattleResult> {
    const stream = new BattleStream();
    const streams = getPlayerStreams(stream);

    const p1 = new StrategyPlayer(streams.p1, p1Strategy);
    const p2 = new StrategyPlayer(streams.p2, p2Strategy);

    const log: string[] = [];
    let winner: BattleResult["winner"] = null;
    let turns = 0;

    const logPromise = (async () => {
      for await (const chunk of streams.omniscient) {
        for (const line of (chunk as string).split("\n")) {
          log.push(line);

          if (line.startsWith("|win|")) {
            const winnerName = line.slice(5);
            winner = winnerName === "Player 1" ? "p1" : "p2";
          } else if (line.startsWith("|tie")) {
            winner = "tie";
          } else if (line.startsWith("|turn|")) {
            turns = parseInt(line.slice(6), 10);
          }
        }
      }
    })();

    void p1.start();
    void p2.start();

    const spec = {
      formatid: this.options.formatId,
      seed: this.options.seed,
    };
    const p1spec = {
      name: "Player 1",
      team: this.options.teamA,
    };
    const p2spec = {
      name: "Player 2",
      team: this.options.teamB,
    };

    void streams.omniscient.write(
      `>start ${JSON.stringify(spec)}\n` +
      `>player p1 ${JSON.stringify(p1spec)}\n` +
      `>player p2 ${JSON.stringify(p2spec)}`
    );

    await logPromise;

    const p1Remaining = countRemaining(log, "p1");
    const p2Remaining = countRemaining(log, "p2");

    return { winner, turns, p1Remaining, p2Remaining, log };
  }
}

function countRemaining(log: string[], player: string): number {
  let teamSize = 0;
  const fainted = new Set<string>();

  for (const line of log) {
    if (line.startsWith(`|teamsize|${player}|`)) {
      teamSize = parseInt(line.split("|")[3], 10);
    }
    if (line.startsWith("|faint|")) {
      const ident = line.slice(7).trim();
      if (ident.startsWith(`${player}a:`) || ident.startsWith(`${player}b:`)) {
        fainted.add(ident);
      }
    }
  }

  if (teamSize === 0) teamSize = 4;
  return teamSize - fainted.size;
}

type StatSpread = { hp: number; atk: number; def: number; spa: number; spd: number; spe: number };

export interface PokemonSet {
  species: string;
  item: string;
  ability: string;
  moves: string[];
  nature: string;
  statPoints: StatSpread;
  level?: number;
  gender?: string;
}

const MAX_SP_PER_STAT = 32;
const MAX_SP_TOTAL = 66;

function validateStatPoints(species: string, sp: StatSpread): void {
  const stats = Object.entries(sp) as [string, number][];
  for (const [stat, val] of stats) {
    if (val < 0 || val > MAX_SP_PER_STAT) {
      throw new Error(
        `${species}: ${stat} has ${val} stat points (must be 0–${MAX_SP_PER_STAT})`
      );
    }
  }
  const total = stats.reduce((sum, [, v]) => sum + v, 0);
  if (total > MAX_SP_TOTAL) {
    throw new Error(
      `${species}: total stat points ${total} exceeds limit of ${MAX_SP_TOTAL}`
    );
  }
}

export function packTeam(pokemon: PokemonSet[]): string {
  for (const p of pokemon) {
    validateStatPoints(p.species, p.statPoints);
  }
  return Teams.pack(
    pokemon.map((p) => ({
      name: "",
      species: p.species,
      item: p.item,
      ability: p.ability,
      moves: p.moves,
      nature: p.nature,
      evs: p.statPoints,
      ivs: { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 },
      level: p.level ?? 50,
      gender: p.gender ?? "",
    }))
  );
}
