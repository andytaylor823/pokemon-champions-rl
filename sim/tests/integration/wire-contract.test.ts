/**
 * Cross-language wire-contract guard (TS side).
 *
 * The worker's StateView is "the canonical definition of the snapshot/view
 * boundary" (sim/src/types.ts). Python's SimClient validates every worker
 * response against Pydantic models in src/state_types.py — a documented 1:1
 * mirror of types.ts — using the shared fixture tests/fixtures/real_snapshot.json.
 *
 * That check is one-directional: nothing asserts the TS worker still EMITS that
 * shape, so the fixture can silently go stale. These tests close the loop:
 *
 *   1. The live worker output AND the committed fixture both conform to one
 *      schema (exact key sets — catches added / removed / renamed fields).
 *   2. The live worker output and the fixture share the same fixed-schema key
 *      topology (direct parity, belt-and-suspenders).
 *   3. Every emitted view is wire-safe: survives JSON round-trip and contains
 *      no NaN / Infinity (which JSON.stringify turns into `null`, breaking the
 *      Python int/float fields).
 *
 * If a test here fails, the TS shape drifted: regenerate real_snapshot.json and
 * update state_types.py deliberately.
 */
import { describe, it, expect } from "vitest";
import * as fs from "node:fs";
import * as path from "node:path";
import { dispatch, resetState } from "../../src/sim-worker";
import { TEAM_A, TEAM_B } from "../fixtures/teams";

// Repo-root fixture shared with the Python suite.
//   sim/tests/integration -> ../../.. -> repo root
const FIXTURE_PATH = path.resolve(
  __dirname,
  "../../../tests/fixtures/real_snapshot.json",
);
const FIXTURE = JSON.parse(fs.readFileSync(FIXTURE_PATH, "utf-8"));

// --- Expected key sets, mirroring sim/src/types.ts --------------------------
const STATE_VIEW_KEYS = ["phase", "to_move", "legal", "snapshot", "terminal", "utility"];
const BATTLE_SNAPSHOT_KEYS = ["turn", "field", "sides"];
const FIELD_KEYS = ["weather", "weatherDuration", "terrain", "terrainDuration", "pseudoWeather"];
const SIDE_KEYS = ["id", "sideConditions", "pokemon"];
const POKEMON_KEYS = [
  "species", "nature", "level", "gender", "hp", "maxhp", "fainted",
  "status", "statusState", "ability", "item", "lastItem", "active",
  "position", "activeTurns", "teraType", "terastallized", "stats",
  "boosts", "moves", "volatiles", "volatileDetails",
];
const MOVE_KEYS = ["id", "pp", "maxpp", "disabled"];
const STATUS_STATE_KEYS = ["stage", "time"];
const SIDE_CONDITION_KEYS = ["duration", "layers"];
const PSEUDO_WEATHER_KEYS = ["duration"];
const VOLATILE_DETAIL_KEYS = ["duration", "time", "hp", "counter"]; // each entry's keys ⊆ this

const sortedKeys = (o: object) => Object.keys(o).sort();

function expectExactKeys(obj: any, keys: string[], label: string) {
  expect(obj, `${label} should be a plain object`).toBeTypeOf("object");
  expect(obj, `${label} should not be null`).not.toBeNull();
  expect(sortedKeys(obj), `${label} keys`).toEqual([...keys].sort());
}

/** Assert a value is a Record<string, number> (open map: keys vary). */
function expectNumberMap(obj: any, label: string) {
  expect(obj, label).toBeTypeOf("object");
  for (const [k, v] of Object.entries(obj)) {
    expect(typeof v, `${label}.${k}`).toBe("number");
  }
}

function assertMoveShape(m: any, label: string) {
  expectExactKeys(m, MOVE_KEYS, label);
  expect(typeof m.id).toBe("string");
  expect(typeof m.pp).toBe("number");
  expect(typeof m.maxpp).toBe("number");
  expect(typeof m.disabled).toBe("boolean");
}

function assertPokemonShape(p: any, label: string) {
  expectExactKeys(p, POKEMON_KEYS, label);
  expectExactKeys(p.statusState, STATUS_STATE_KEYS, `${label}.statusState`);
  expectNumberMap(p.stats, `${label}.stats`);
  expectNumberMap(p.boosts, `${label}.boosts`);

  expect(Array.isArray(p.moves), `${label}.moves`).toBe(true);
  p.moves.forEach((m: any, i: number) => assertMoveShape(m, `${label}.moves[${i}]`));

  expect(Array.isArray(p.volatiles), `${label}.volatiles`).toBe(true);
  for (const v of p.volatiles) expect(typeof v).toBe("string");

  // volatileDetails: Record<string, {duration?,time?,hp?,counter?}>
  expect(p.volatileDetails, `${label}.volatileDetails`).toBeTypeOf("object");
  for (const [id, detail] of Object.entries<any>(p.volatileDetails)) {
    for (const k of Object.keys(detail)) {
      expect(VOLATILE_DETAIL_KEYS, `${label}.volatileDetails.${id} key "${k}"`).toContain(k);
      expect(typeof detail[k]).toBe("number");
    }
  }
}

function assertSideShape(s: any, label: string) {
  expectExactKeys(s, SIDE_KEYS, label);
  // sideConditions: Record<string, {duration, layers}>
  for (const [id, sc] of Object.entries<any>(s.sideConditions)) {
    expectExactKeys(sc, SIDE_CONDITION_KEYS, `${label}.sideConditions.${id}`);
  }
  expect(Array.isArray(s.pokemon), `${label}.pokemon`).toBe(true);
  s.pokemon.forEach((p: any, i: number) => assertPokemonShape(p, `${label}.pokemon[${i}]`));
}

function assertFieldShape(f: any, label: string) {
  expectExactKeys(f, FIELD_KEYS, label);
  // pseudoWeather: Record<string, {duration}>
  for (const [id, pw] of Object.entries<any>(f.pseudoWeather)) {
    expectExactKeys(pw, PSEUDO_WEATHER_KEYS, `${label}.pseudoWeather.${id}`);
  }
}

function assertStateViewShape(v: any, label: string) {
  expectExactKeys(v, STATE_VIEW_KEYS, label);
  expect(typeof v.phase, `${label}.phase`).toBe("string");
  expect(Array.isArray(v.to_move), `${label}.to_move`).toBe(true);
  expect(typeof v.terminal, `${label}.terminal`).toBe("boolean");

  // legal: Record<side, any> — untyped on both sides; just keyed by p1/p2.
  expect(v.legal, `${label}.legal`).toBeTypeOf("object");
  for (const sid of Object.keys(v.legal)) expect(["p1", "p2"]).toContain(sid);

  // utility: null | {p1:number, p2:number}
  if (v.utility !== null) {
    expectExactKeys(v.utility, ["p1", "p2"], `${label}.utility`);
    expectNumberMap(v.utility, `${label}.utility`);
  }

  expectExactKeys(v.snapshot, BATTLE_SNAPSHOT_KEYS, `${label}.snapshot`);
  expect(typeof v.snapshot.turn).toBe("number");
  assertFieldShape(v.snapshot.field, `${label}.snapshot.field`);
  expect(Array.isArray(v.snapshot.sides)).toBe(true);
  v.snapshot.sides.forEach((s: any, i: number) =>
    assertSideShape(s, `${label}.snapshot.sides[${i}]`),
  );
}

/** Generate live team_preview and move_phase views via the worker. */
function liveViews() {
  resetState();
  const battle = dispatch({ cmd: "new_battle", team_a: TEAM_A, team_b: TEAM_B, seed: [1, 2, 3, 4] });
  const team_preview = dispatch({ cmd: "view", handle: battle.handle }).view;
  const step = dispatch({
    cmd: "step", handle: battle.handle,
    choices: { p1: "team 1234", p2: "team 1234" },
    seed: [10, 20, 30, 40],
  });
  expect(step.view.phase).toBe("move");
  return { team_preview, move_phase: step.view };
}

describe("wire contract: StateView schema conformance", () => {
  const live = liveViews();

  it("the fixture has the expected scenarios", () => {
    expect(Object.keys(FIXTURE).sort()).toEqual(["move_phase", "team_preview"]);
  });

  for (const scenario of ["team_preview", "move_phase"] as const) {
    it(`live worker ${scenario} view conforms to the StateView schema`, () => {
      assertStateViewShape((live as any)[scenario], `live.${scenario}`);
    });

    it(`committed fixture ${scenario} conforms to the same schema (not stale)`, () => {
      assertStateViewShape(FIXTURE[scenario], `fixture.${scenario}`);
    });
  }
});

describe("wire contract: live output matches fixture topology", () => {
  const live = liveViews();

  /** Compare the fixed-schema key sets at each level between two views. */
  function expectSameTopology(a: any, b: any, label: string) {
    expect(sortedKeys(a), `${label} top-level`).toEqual(sortedKeys(b));
    expect(sortedKeys(a.snapshot), `${label} snapshot`).toEqual(sortedKeys(b.snapshot));
    expect(sortedKeys(a.snapshot.field), `${label} field`).toEqual(sortedKeys(b.snapshot.field));
    expect(sortedKeys(a.snapshot.sides[0]), `${label} side`).toEqual(sortedKeys(b.snapshot.sides[0]));
    expect(sortedKeys(a.snapshot.sides[0].pokemon[0]), `${label} pokemon`)
      .toEqual(sortedKeys(b.snapshot.sides[0].pokemon[0]));
    expect(sortedKeys(a.snapshot.sides[0].pokemon[0].moves[0]), `${label} move`)
      .toEqual(sortedKeys(b.snapshot.sides[0].pokemon[0].moves[0]));
    expect(sortedKeys(a.snapshot.sides[0].pokemon[0].statusState), `${label} statusState`)
      .toEqual(sortedKeys(b.snapshot.sides[0].pokemon[0].statusState));
  }

  it("team_preview: live worker output matches fixture key topology", () => {
    expectSameTopology(live.team_preview, FIXTURE.team_preview, "team_preview");
  });

  it("move_phase: live worker output matches fixture key topology", () => {
    expectSameTopology(live.move_phase, FIXTURE.move_phase, "move_phase");
  });
});

describe("wire contract: JSON wire-safety", () => {
  const live = liveViews();

  /** Recursively collect every numeric leaf for finiteness checks. */
  function collectNumbers(x: any, acc: number[] = []): number[] {
    if (typeof x === "number") acc.push(x);
    else if (Array.isArray(x)) for (const e of x) collectNumbers(e, acc);
    else if (x && typeof x === "object") for (const v of Object.values(x)) collectNumbers(v, acc);
    return acc;
  }

  for (const scenario of ["team_preview", "move_phase"] as const) {
    it(`${scenario} view survives a JSON round-trip unchanged`, () => {
      const v = (live as any)[scenario];
      // The worker emits via JSON.stringify over stdout; Python does json.loads.
      expect(JSON.parse(JSON.stringify(v))).toEqual(v);
    });

    it(`${scenario} view contains no NaN / Infinity (JSON would corrupt them to null)`, () => {
      const nums = collectNumbers((live as any)[scenario]);
      expect(nums.length).toBeGreaterThan(0);
      for (const n of nums) expect(Number.isFinite(n)).toBe(true);
    });
  }
});
