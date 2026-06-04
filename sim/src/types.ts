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
