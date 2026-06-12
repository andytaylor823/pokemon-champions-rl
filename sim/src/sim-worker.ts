/**
 * SimClient Node worker (Milestone 1).
 *
 * A long-lived process that owns the Pokémon Showdown battle engine and speaks
 * line-delimited JSON over stdin/stdout (one request per line, one response per
 * line). Python holds opaque integer HANDLES; the heavy Battle objects live
 * here. See docs/architecture/repo-architecture.md §3.1 and the SimClient plan.
 *
 * Design decisions realised here:
 *   - SYNC engine API (new Battle + setPlayer + choose), not BattleStream.
 *   - Clone = State.serialize -> deserialize (carries the PRNG seed).
 *   - step() is immutable: clones the parent, RESEEDS to the given seed, then
 *     applies choices for exactly the side(s) the engine is asking. Search
 *     re-seeds per chance sample; the worker owns no chance strategy.
 *   - Engine-native boundary: snapshots are raw engine fields; choices are
 *     Showdown choice strings. No tensors, no action numbering here.
 *   - Scratchpad lifetime: handles belong to a session; close_search frees all.
 *   - strictChoices: illegal choices error loudly (trust the mask).
 *
 * Protocol: every request is {id, cmd, ...args}; every response echoes {id} and
 * is either {id, ok:true, ...result} or {id, ok:false, error}. Diagnostics go
 * to stderr so stdout stays clean JSON.
 *
 * Run standalone for a smoke test:  npx tsx src/sim-worker.ts < cmds.jsonl
 */

import * as readline from "node:readline";
import { packTeam, PokemonSet } from "./battle-runner";
import type {
  BattleSnapshot,
  PokemonSnapshot,
  Side,
  SideSnapshot,
  StateView,
} from "./types";

// eslint-disable-next-line @typescript-eslint/no-var-requires
const { Battle } = require("pokemon-showdown");
const { State } = require("pokemon-showdown/dist/sim/state");

type Seed = string | [number, number, number, number];

// --- format config (hoisted above registry so resetState can reference it) --
const FORMAT_ID_DEFAULT = "gen9championsvgc2026regma";
let FORMAT_ID = FORMAT_ID_DEFAULT;

// --- handle / session registry ---------------------------------------------
let nextHandle = 1;
let nextSession = 1;
const battles = new Map<number, any>(); // handle -> Battle
const handleSession = new Map<number, number | null>(); // handle -> session (null = global/live)
const sessions = new Map<number, Set<number>>(); // session -> handles

/** Clear all state — used by tests to isolate each test case. */
function resetState(): void {
  nextHandle = 1;
  nextSession = 1;
  battles.clear();
  handleSession.clear();
  sessions.clear();
  FORMAT_ID = FORMAT_ID_DEFAULT;
}

function alloc(battle: any, session: number | null): number {
  const h = nextHandle++;
  battles.set(h, battle);
  handleSession.set(h, session);
  if (session !== null) sessions.get(session)!.add(h);
  return h;
}

function freeHandle(h: number): void {
  const s = handleSession.get(h);
  if (s != null) sessions.get(s)?.delete(h);
  battles.delete(h);
  handleSession.delete(h);
}

function getBattle(h: number): any {
  const b = battles.get(h);
  if (!b) throw new Error(`unknown handle ${h}`);
  return b;
}

// --- engine helpers ---------------------------------------------------------
function newBattle(teamA: PokemonSet[], teamB: PokemonSet[], seed?: Seed): any {
  const battle = new Battle({ formatid: FORMAT_ID, seed, strictChoices: true });
  battle.setPlayer("p1", { name: "p1", team: packTeam(teamA) });
  battle.setPlayer("p2", { name: "p2", team: packTeam(teamB) });
  return battle;
}

function cloneBattle(battle: any): any {
  return State.deserializeBattle(State.serializeBattle(battle));
}

function actingSides(battle: any): Side[] {
  return battle.sides
    .filter((s: any) => s.activeRequest && !s.activeRequest.wait)
    .map((s: any) => s.id as Side);
}

/** Apply choices for exactly the side(s) being asked; fail loud otherwise. */
function applyChoices(battle: any, choices: Partial<Record<Side, string>>): void {
  const acting = actingSides(battle);
  for (const sid of Object.keys(choices) as Side[]) {
    if (!acting.includes(sid)) throw new Error(`choice given for non-acting side ${sid}`);
  }
  for (const sid of acting) {
    const c = choices[sid];
    if (c == null) throw new Error(`missing choice for acting side ${sid}`);
    if (!battle.choose(sid, c)) throw new Error(`illegal choice for ${sid}: "${c}"`);
  }
}

function phaseOf(battle: any): string {
  if (battle.ended) return "terminal";
  switch (battle.requestState) {
    case "teampreview": return "teamPreview";
    case "switch": return "forceSwitch";
    case "move": return "move";
    default: return battle.requestState || "none";
  }
}

function utilityOf(battle: any): Record<Side, number> | null {
  if (!battle.ended) return null;
  if (!battle.winner) return { p1: 0, p2: 0 }; // tie
  const p1won = battle.winner === "p1";
  return { p1: p1won ? 1 : -1, p2: p1won ? -1 : 1 };
}

/** A clean, JSON-safe omniscient snapshot built from live objects (v1). */
function snapshotPokemon(p: any): PokemonSnapshot {
  return {
    species: p.species?.id ?? null,
    level: p.level,
    gender: p.gender,
    hp: p.hp,
    maxhp: p.maxhp,
    fainted: p.fainted,
    status: p.status || null,
    ability: p.ability || null,
    item: p.item || null,
    active: p.isActive,
    position: p.position,
    teraType: p.teraType ?? null,
    terastallized: p.terastallized ?? null,
    stats: { hp: p.maxhp, ...(p.storedStats ?? {}) },
    boosts: { ...(p.boosts ?? {}) },
    moves: (p.moveSlots ?? []).map((m: any) => ({
      id: m.id, pp: m.pp, maxpp: m.maxpp, disabled: !!m.disabled,
    })),
    volatiles: Object.keys(p.volatiles ?? {}),
  };
}

function snapshotSide(side: any): SideSnapshot {
  const conds: Record<string, number | null> = {};
  for (const id of Object.keys(side.sideConditions ?? {})) {
    conds[id] = side.sideConditions[id]?.duration ?? null;
  }
  return {
    id: side.id,
    sideConditions: conds,
    pokemon: side.pokemon.map(snapshotPokemon),
  };
}

function snapshotBattle(battle: any): BattleSnapshot {
  const field = battle.field;
  return {
    turn: battle.turn,
    field: {
      weather: field.weather || null,
      terrain: field.terrain || null,
      pseudoWeather: Object.keys(field.pseudoWeather ?? {}),
    },
    sides: battle.sides.map(snapshotSide),
  };
}

function view(battle: any): StateView {
  const legal: Record<string, any> = {};
  for (const s of battle.sides) if (s.activeRequest) legal[s.id] = s.activeRequest;
  return {
    phase: phaseOf(battle),
    to_move: actingSides(battle),
    legal,
    snapshot: snapshotBattle(battle),
    terminal: battle.ended,
    utility: utilityOf(battle),
  };
}

// --- command dispatch -------------------------------------------------------
function dispatch(msg: any): any {
  switch (msg.cmd) {
    case "config": {
      if (msg.format_id) FORMAT_ID = msg.format_id;
      return { format_id: FORMAT_ID };
    }
    case "new_battle": {
      const battle = newBattle(msg.team_a, msg.team_b, msg.seed);
      const handle = alloc(battle, null);
      return { handle, view: view(battle) };
    }
    case "open_search": {
      const session = nextSession++;
      sessions.set(session, new Set());
      let root: number | null = null;
      if (msg.from != null) root = alloc(cloneBattle(getBattle(msg.from)), session);
      return { session, root, root_view: root != null ? view(battles.get(root)) : null };
    }
    case "step": {
      const parent = getBattle(msg.handle);
      const child = cloneBattle(parent);
      if (msg.seed != null) child.resetRNG(msg.seed);
      const logStart = child.log.length;
      applyChoices(child, msg.choices ?? {});
      const outcome = child.log.slice(logStart);
      // Child inherits the parent's session. Stepping a search clone keeps the
      // child inside that session (freed en masse by close_search). Stepping the
      // live battle (session = null) is the intended way to advance the real
      // game: the child is a new live-line handle, NOT swept by any search —
      // the caller advances by reassigning to it and `release`s the old handle.
      const session = handleSession.get(msg.handle) ?? null;
      const childHandle = alloc(child, session);
      return { child: childHandle, view: view(child), outcome };
    }
    case "view":
      return { view: view(getBattle(msg.handle)) };
    case "release":
      freeHandle(msg.handle);
      return { released: msg.handle };
    case "close_search": {
      const set = sessions.get(msg.session);
      const freed = set ? set.size : 0;
      if (set) {
        // Iterate a copy: freeHandle deletes from `set` as it cleans each handle.
        for (const h of [...set]) freeHandle(h);
        sessions.delete(msg.session);
      }
      return { closed: msg.session, freed };
    }
    case "stats":
      return { handles: battles.size, sessions: sessions.size };
    case "close":
      process.stderr.write("[sim-worker] closing\n");
      return { shutdown: true };
    default:
      throw new Error(`unknown cmd: ${msg.cmd}`);
  }
}

// --- stdio loop -------------------------------------------------------------
function main(): void {
  const rl = readline.createInterface({ input: process.stdin });
  rl.on("line", (line: string) => {
    const trimmed = line.trim();
    if (!trimmed) return;
    let msg: any;
    try {
      msg = JSON.parse(trimmed);
    } catch (e) {
      process.stdout.write(JSON.stringify({ id: null, ok: false, error: `bad json: ${String(e)}` }) + "\n");
      return;
    }
    try {
      const result = dispatch(msg);
      process.stdout.write(JSON.stringify({ id: msg.id ?? null, ok: true, ...result }) + "\n");
      if (result.shutdown) process.exit(0);
    } catch (e: any) {
      process.stdout.write(JSON.stringify({ id: msg.id ?? null, ok: false, error: e?.message ?? String(e) }) + "\n");
    }
  });
  rl.on("close", () => process.exit(0));
  process.stderr.write("[sim-worker] ready\n");
}

// Only start the stdio loop when executed directly, not when imported for tests.
// This is the TypeScript equivalent of Python's `if __name__ == "__main__"`.
if (require.main === module) {
  main();
}

// Public test surface: drive everything through `dispatch` (the same entry the
// stdio loop uses), and `resetState` to isolate cases. Internals — the handle
// registries and engine helpers — stay private; inspect state via the `stats`
// and `view` commands.
export { dispatch, resetState };
