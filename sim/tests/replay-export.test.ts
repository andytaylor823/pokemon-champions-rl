/**
 * Unit tests for replay-export: generateReplayHtml and saveReplay.
 * Tests HTML generation, escaping, and file I/O independently of the battle engine.
 */
import { describe, it, expect, afterAll } from "vitest";
import * as fs from "node:fs";
import * as path from "node:path";
import * as os from "node:os";
import { generateReplayHtml, saveReplay } from "../src/replay-export";
import type { BattleResult } from "../src/types";

/** Minimal BattleResult for unit testing (no engine needed). */
function minimalResult(overrides: Partial<BattleResult> = {}): BattleResult {
  return {
    winner: "p1",
    turns: 5,
    p1Remaining: 3,
    p2Remaining: 0,
    log: ["|turn|1", "|move|p1a: Charizard|Heat Wave", "|win|Player 1"],
    ...overrides,
  };
}

describe("generateReplayHtml", () => {
  it("returns valid HTML with required structural elements", () => {
    const html = generateReplayHtml(minimalResult(), {
      formatId: "gen9championsvgc2026regma",
      p1: "Alice",
      p2: "Bob",
    });
    expect(html).toContain("<!DOCTYPE html>");
    expect(html).toContain("replay-embed.js");
    expect(html).toContain("Alice");
    expect(html).toContain("Bob");
    expect(html).toContain("Turns: 5");
    expect(html).toContain("|win|Player 1");
  });

  it("renders winner as 'none' when winner is null", () => {
    const html = generateReplayHtml(minimalResult({ winner: null }), {
      formatId: "test", p1: "A", p2: "B",
    });
    expect(html).toContain("Winner: none");
  });

  it("renders tie winner correctly", () => {
    const html = generateReplayHtml(minimalResult({ winner: "tie" }), {
      formatId: "test", p1: "A", p2: "B",
    });
    expect(html).toContain("Winner: tie");
  });

  it("escapes angle brackets in player names (prevents XSS)", () => {
    const html = generateReplayHtml(minimalResult(), {
      formatId: "test",
      p1: '<script>alert("xss")</script>',
      p2: "Normal",
    });
    expect(html).not.toContain("<script>alert");
    expect(html).toContain("&lt;script&gt;");
  });

  it("escapes ampersands in player names", () => {
    const html = generateReplayHtml(minimalResult(), {
      formatId: "test", p1: "Bob & Alice", p2: "C",
    });
    expect(html).toContain("Bob &amp; Alice");
  });

  it("escapes double quotes in metadata", () => {
    const html = generateReplayHtml(minimalResult(), {
      formatId: "test",
      p1: 'Player "One"',
      p2: "B",
    });
    expect(html).toContain("Player &quot;One&quot;");
  });

  it("embeds the full battle log in the script tag", () => {
    const result = minimalResult({
      log: ["|turn|1", "|move|p1a: Foo|Bar", "|turn|2"],
    });
    const html = generateReplayHtml(result, { formatId: "t", p1: "A", p2: "B" });
    expect(html).toContain("|turn|1\n|move|p1a: Foo|Bar\n|turn|2");
  });

  it("handles empty log without crashing", () => {
    const html = generateReplayHtml(minimalResult({ log: [] }), {
      formatId: "t", p1: "A", p2: "B",
    });
    expect(html).toContain("<!DOCTYPE html>");
    expect(html).toContain('class="battle-log-data"');
  });

  it("includes format id in the title", () => {
    const html = generateReplayHtml(minimalResult(), {
      formatId: "gen9championsvgc2026regma", p1: "X", p2: "Y",
    });
    expect(html).toContain("gen9championsvgc2026regma");
  });

  it("embeds log content containing </script> without escaping (documents limitation)", () => {
    const result = minimalResult({
      log: ["|turn|1", "</script><script>alert(1)</script>", "|win|Player 1"],
    });
    const html = generateReplayHtml(result, { formatId: "t", p1: "A", p2: "B" });
    // The log is embedded raw in <script type="text/plain">. A literal </script>
    // in the log terminates the tag prematurely. This documents current behavior.
    expect(html).toContain("</script><script>alert(1)</script>");
  });

  it("single quotes in player names pass through unescaped", () => {
    const html = generateReplayHtml(minimalResult(), {
      formatId: "test", p1: "O'Brien", p2: "B",
    });
    expect(html).toContain("O'Brien");
    expect(html).not.toContain("&#39;");
    expect(html).not.toContain("&apos;");
  });
});

describe("saveReplay", () => {
  const tempFiles: string[] = [];
  const tempDirs: string[] = [];

  afterAll(() => {
    for (const f of tempFiles) {
      try { fs.unlinkSync(f); } catch { /* already cleaned */ }
    }
    for (const d of tempDirs) {
      try { fs.rmSync(d, { recursive: true, force: true }); } catch { /* already cleaned */ }
    }
  });

  it("writes HTML file to the specified path", () => {
    const dest = path.join(os.tmpdir(), `vitest-replay-unit-${Date.now()}.html`);
    tempFiles.push(dest);

    const saved = saveReplay(minimalResult(), {
      formatId: "test", p1: "A", p2: "B",
    }, dest);

    expect(fs.existsSync(saved)).toBe(true);
    const content = fs.readFileSync(saved, "utf-8");
    expect(content).toContain("<!DOCTYPE html>");
  });

  it("returns an absolute path", () => {
    const dest = path.join(os.tmpdir(), `vitest-replay-abs-${Date.now()}.html`);
    tempFiles.push(dest);

    const saved = saveReplay(minimalResult(), {
      formatId: "test", p1: "A", p2: "B",
    }, dest);

    expect(path.isAbsolute(saved)).toBe(true);
  });

  it("creates parent directories if they do not exist", () => {
    const tempBase = path.join(os.tmpdir(), `vitest-nested-${Date.now()}`);
    tempDirs.push(tempBase);
    const dest = path.join(tempBase, "sub", "replay.html");

    const saved = saveReplay(minimalResult(), {
      formatId: "test", p1: "A", p2: "B",
    }, dest);

    expect(fs.existsSync(saved)).toBe(true);
  });

  it("uses default path under replays/ when no filePath provided", () => {
    const saved = saveReplay(minimalResult(), {
      formatId: "test", p1: "DefaultA", p2: "DefaultB",
    });
    tempFiles.push(saved);

    expect(path.isAbsolute(saved)).toBe(true);
    expect(saved).toContain("replays");
    expect(saved).toContain("DefaultA-vs-DefaultB-");
    expect(saved).toMatch(/\.html$/);
    expect(fs.existsSync(saved)).toBe(true);
  });
});
