import { describe, it, expect } from "vitest";
import { packTeam, countRemaining, type PokemonSet } from "../src/battle-runner";
import { TEAM_A } from "./fixtures/teams";

// A minimal valid Champions-format Pokemon for reuse across tests
function validMon(overrides: Partial<PokemonSet> = {}): PokemonSet {
  return {
    species: "Charizard",
    item: "Charizardite Y",
    ability: "Blaze",
    moves: ["Heat Wave", "Protect", "Air Slash", "Solar Beam"],
    nature: "Timid",
    statPoints: { hp: 2, atk: 0, def: 0, spa: 32, spd: 0, spe: 32 },
    ...overrides,
  };
}

describe("packTeam", () => {
  it("produces a non-empty packed string for a valid team", () => {
    const packed = packTeam([validMon()]);
    expect(typeof packed).toBe("string");
    expect(packed.length).toBeGreaterThan(0);
  });

  it("packs multiple Pokemon separated by Showdown delimiter", () => {
    const packed = packTeam([
      validMon({ species: "Charizard", item: "Charizardite Y" }),
      validMon({ species: "Venusaur", item: "Lum Berry", ability: "Chlorophyll" }),
    ]);
    // Showdown packed format uses ']' between team members
    expect(packed).toContain("]");
    const parts = packed.split("]");
    expect(parts.length).toBeGreaterThanOrEqual(2);
  });

  it("includes species, ability, and moves in packed string", () => {
    const packed = packTeam([validMon()]);
    expect(packed).toContain("Charizard");
    // Packed format strips spaces from identifiers
    expect(packed).toContain("CharizarditeY");
    expect(packed).toContain("Blaze");
    expect(packed).toContain("HeatWave");
  });

  it("defaults level to 50 (omitted in packed format as it is the standard)", () => {
    const packedDefault = packTeam([validMon()]);
    const packedExplicit = packTeam([validMon({ level: 50 })]);
    // Both should produce the same output since 50 is the default
    expect(packedDefault).toBe(packedExplicit);
  });

  it("includes non-default level in packed string", () => {
    const packed = packTeam([validMon({ level: 100 })]);
    // Non-default levels appear in the packed format
    expect(packed).not.toBe(packTeam([validMon()]));
  });
});

describe("validateStatPoints (via packTeam)", () => {
  it("accepts a valid spread summing to exactly 66", () => {
    // Total: 2 + 0 + 0 + 32 + 0 + 32 = 66
    expect(() => packTeam([validMon()])).not.toThrow();
  });

  it("accepts a valid spread summing to less than 66", () => {
    const mon = validMon({
      statPoints: { hp: 10, atk: 10, def: 10, spa: 10, spd: 10, spe: 10 },
    });
    expect(() => packTeam([mon])).not.toThrow();
  });

  it("accepts all-zero stat points", () => {
    const mon = validMon({
      statPoints: { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 },
    });
    expect(() => packTeam([mon])).not.toThrow();
  });

  it("rejects total stat points exceeding 66", () => {
    const mon = validMon({
      statPoints: { hp: 32, atk: 32, def: 32, spa: 0, spd: 0, spe: 0 },
    });
    expect(() => packTeam([mon])).toThrow(/total stat points 96 exceeds limit of 66/);
  });

  it("rejects a single stat exceeding 32", () => {
    const mon = validMon({
      statPoints: { hp: 0, atk: 0, def: 0, spa: 33, spd: 0, spe: 0 },
    });
    expect(() => packTeam([mon])).toThrow(/spa has 33 stat points.*must be 0–32/);
  });

  it("rejects negative stat points", () => {
    const mon = validMon({
      statPoints: { hp: 0, atk: -1, def: 0, spa: 0, spd: 0, spe: 0 },
    });
    expect(() => packTeam([mon])).toThrow(/atk has -1 stat points.*must be 0–32/);
  });

  it("validates each Pokemon in a multi-mon team", () => {
    const team = [
      validMon({ species: "Charizard" }),
      validMon({
        species: "Venusaur",
        item: "Lum Berry",
        statPoints: { hp: 0, atk: 0, def: 0, spa: 50, spd: 0, spe: 0 },
      }),
    ];
    expect(() => packTeam(team)).toThrow(/Venusaur.*spa has 50/);
  });

  it("accepts the max single stat of 32 within budget", () => {
    const mon = validMon({
      statPoints: { hp: 32, atk: 0, def: 0, spa: 32, spd: 2, spe: 0 },
    });
    expect(() => packTeam([mon])).not.toThrow();
  });
});

describe("packTeam edge cases", () => {
  it("packs a single-Pokemon team without team delimiter", () => {
    const packed = packTeam([validMon()]);
    expect(packed.length).toBeGreaterThan(0);
    // Single mon should not contain the ']' team delimiter
    expect(packed).not.toContain("]");
  });

  it("packs an empty team without error", () => {
    const packed = packTeam([]);
    expect(typeof packed).toBe("string");
  });

  it("accepts maximally-spread stat distribution (11 per stat, total 66)", () => {
    const mon = validMon({
      statPoints: { hp: 11, atk: 11, def: 11, spa: 11, spd: 11, spe: 11 },
    });
    expect(() => packTeam([mon])).not.toThrow();
  });

  it("handles non-integer stat points via floor behavior", () => {
    // Float stat points — validateStatPoints checks val < 0 || val > 32,
    // so 10.5 passes the range check. Documents current behavior.
    const mon = validMon({
      statPoints: { hp: 10.5, atk: 10, def: 10, spa: 10, spd: 10, spe: 10 } as any,
    });
    // Should not throw since 10.5 < 32 and total ~60.5 < 66
    expect(() => packTeam([mon])).not.toThrow();
  });
});

describe("packTeam additional coverage", () => {
  it("packs a full 6-pokemon team with correct delimiter count", () => {
    const packed = packTeam(TEAM_A);
    const parts = packed.split("]");
    // 6 mons produce 5 ']' delimiters → 6 parts after split
    expect(parts.length).toBe(6);
  });

  it("includes gender override in packed string", () => {
    const withGender = validMon({ gender: "F" });
    const withoutGender = validMon();
    const packedF = packTeam([withGender]);
    const packedDefault = packTeam([withoutGender]);
    expect(packedF).not.toBe(packedDefault);
  });

  it("rejects total stat points of exactly 67 (one over budget)", () => {
    const mon = validMon({
      statPoints: { hp: 32, atk: 32, def: 3, spa: 0, spd: 0, spe: 0 },
    });
    expect(() => packTeam([mon])).toThrow(/total stat points 67 exceeds limit of 66/);
  });
});

describe("countRemaining", () => {
  it("counts remaining pokemon from teamsize and faint lines", () => {
    const log = [
      "|teamsize|p1|4",
      "|teamsize|p2|4",
      "|faint|p1a: Charizard",
      "|faint|p2a: Corviknight",
      "|faint|p2b: Meganium",
    ];
    expect(countRemaining(log, "p1")).toBe(3);
    expect(countRemaining(log, "p2")).toBe(2);
  });

  it("deduplicates faint lines for the same pokemon identifier", () => {
    const log = [
      "|teamsize|p1|4",
      "|faint|p1a: Charizard",
      "|faint|p1a: Charizard",
    ];
    // Same ident string → Set deduplicates → still 4 - 1 = 3
    expect(countRemaining(log, "p1")).toBe(3);
  });

  it("defaults teamSize to 4 when no teamsize line present", () => {
    const log = [
      "|faint|p1a: Charizard",
    ];
    expect(countRemaining(log, "p1")).toBe(3);
  });

  it("returns full team size when no faints", () => {
    const log = [
      "|teamsize|p1|4",
    ];
    expect(countRemaining(log, "p1")).toBe(4);
  });

  it("counts faints from both active slots (a and b)", () => {
    const log = [
      "|teamsize|p1|4",
      "|faint|p1a: Charizard",
      "|faint|p1b: Venusaur",
    ];
    expect(countRemaining(log, "p1")).toBe(2);
  });

  it("returns 0 when all pokemon fainted", () => {
    const log = [
      "|teamsize|p1|4",
      "|faint|p1a: Charizard",
      "|faint|p1b: Venusaur",
      "|faint|p1a: Garchomp",
      "|faint|p1b: Whimsicott",
    ];
    expect(countRemaining(log, "p1")).toBe(0);
  });

  it("only counts faints for the specified player", () => {
    const log = [
      "|teamsize|p1|4",
      "|teamsize|p2|4",
      "|faint|p1a: Charizard",
      "|faint|p2a: Corviknight",
      "|faint|p2b: Meganium",
    ];
    expect(countRemaining(log, "p1")).toBe(3);
    expect(countRemaining(log, "p2")).toBe(2);
  });

  it("handles empty log", () => {
    expect(countRemaining([], "p1")).toBe(4);
  });
});
