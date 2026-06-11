import { describe, it, expect } from "vitest";
import { packTeam, type PokemonSet } from "../src/battle-runner";

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
