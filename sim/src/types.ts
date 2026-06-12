export interface BattleResult {
  winner: "p1" | "p2" | "tie" | null;
  turns: number;
  /** Per-player summary: how many Pokemon survived */
  p1Remaining: number;
  p2Remaining: number;
  log: string[];
}

/**
 * Given a parsed request object from Showdown, return a choice string.
 *
 * Request shapes vary by phase:
 *   - teamPreview: pick team order, e.g. "team 1234"
 *   - active + side.pokemon: normal turn, e.g. "move heatwave mega, move protect"
 *   - forceSwitch: pick a replacement, e.g. "switch 3"
 */
export type Strategy = (request: any) => string;

// --- sim-worker wire contract ----------------------------------------------
// The shapes the worker emits over JSON and SimClient consumes as dicts. These
// are the canonical definition of the snapshot/view boundary; keep the worker's
// builders (snapshotPokemon/Side/Battle, view) returning these.

export type Side = "p1" | "p2";

export interface MoveSnapshot {
  id: string;
  pp: number;
  maxpp: number;
  disabled: boolean;
}

export interface PokemonSnapshot {
  species: string | null;
  level: number;
  gender: string;
  hp: number;
  maxhp: number;
  fainted: boolean;
  status: string | null;
  ability: string | null;
  item: string | null;
  active: boolean;
  position: number;
  teraType: string | null;
  terastallized: string | null;
  stats: Record<string, number>;
  boosts: Record<string, number>;
  moves: MoveSnapshot[];
  volatiles: string[];
}

export interface SideSnapshot {
  id: string;
  sideConditions: Record<string, number | null>;
  pokemon: PokemonSnapshot[];
}

export interface FieldSnapshot {
  weather: string | null;
  terrain: string | null;
  pseudoWeather: string[];
}

export interface BattleSnapshot {
  turn: number;
  field: FieldSnapshot;
  sides: SideSnapshot[];
}

/** The omniscient state the worker returns from new_battle / step / view. */
export interface StateView {
  phase: string;
  to_move: Side[];
  /** Raw Showdown request object per acting side (engine-native legal moves). */
  legal: Record<string, any>;
  snapshot: BattleSnapshot;
  terminal: boolean;
  utility: Record<Side, number> | null;
}
