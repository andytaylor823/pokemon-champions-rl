/**
 * Pure decision helpers (`utilityOf` / `phaseOf`), tested at every branch with
 * stub battle objects. Their tie / fallthrough branches are unreachable through
 * a real battle deterministically, so they are asserted directly here.
 */
import { describe, it, expect } from "vitest";
import { phaseOf, utilityOf } from "../src/sim-worker";

describe("utilityOf", () => {
  it("returns null for a non-terminal battle", () => {
    expect(utilityOf({ ended: false } as any)).toBeNull();
  });

  it("returns +1/-1 (zero-sum) when p1 wins", () => {
    expect(utilityOf({ ended: true, winner: "p1" } as any)).toEqual({ p1: 1, p2: -1 });
  });

  it("returns -1/+1 (zero-sum) when p2 wins", () => {
    expect(utilityOf({ ended: true, winner: "p2" } as any)).toEqual({ p1: -1, p2: 1 });
  });

  it("returns 0/0 for a tie (empty-string winner)", () => {
    expect(utilityOf({ ended: true, winner: "" } as any)).toEqual({ p1: 0, p2: 0 });
  });

  it("returns 0/0 for a tie (null winner)", () => {
    expect(utilityOf({ ended: true, winner: null } as any)).toEqual({ p1: 0, p2: 0 });
  });

  it("returns 0/0 for a tie (undefined winner)", () => {
    expect(utilityOf({ ended: true } as any)).toEqual({ p1: 0, p2: 0 });
  });

  it("is always zero-sum across every terminal outcome", () => {
    for (const winner of ["p1", "p2", "", null]) {
      const u = utilityOf({ ended: true, winner } as any)!;
      expect(u.p1 + u.p2).toBe(0);
    }
  });
});

describe("phaseOf", () => {
  it("returns 'terminal' for an ended battle regardless of requestState", () => {
    expect(phaseOf({ ended: true, requestState: "move" } as any)).toBe("terminal");
  });

  it("maps engine requestState 'teampreview' to 'teamPreview'", () => {
    expect(phaseOf({ ended: false, requestState: "teampreview" } as any)).toBe("teamPreview");
  });

  it("maps engine requestState 'switch' to 'forceSwitch'", () => {
    expect(phaseOf({ ended: false, requestState: "switch" } as any)).toBe("forceSwitch");
  });

  it("maps engine requestState 'move' to 'move'", () => {
    expect(phaseOf({ ended: false, requestState: "move" } as any)).toBe("move");
  });

  it("returns 'none' when requestState is empty string (fallthrough)", () => {
    expect(phaseOf({ ended: false, requestState: "" } as any)).toBe("none");
  });

  it("returns 'none' when requestState is undefined (fallthrough)", () => {
    expect(phaseOf({ ended: false } as any)).toBe("none");
  });

  it("passes through an unrecognised requestState verbatim (fallthrough)", () => {
    expect(phaseOf({ ended: false, requestState: "weird" } as any)).toBe("weird");
  });
});
