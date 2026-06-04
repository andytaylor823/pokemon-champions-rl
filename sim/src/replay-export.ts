import * as fs from "fs";
import * as path from "path";
import type { BattleResult } from "./types";

export interface ReplayMetadata {
  formatId: string;
  p1: string;
  p2: string;
}

/**
 * Generates a self-contained HTML replay file that can be opened in any
 * browser. Uses Showdown's replay-embed.js from their CDN to render the
 * battle animation — no local server required.
 */
export function generateReplayHtml(
  result: BattleResult,
  meta: ReplayMetadata
): string {
  const logText = result.log.join("\n");
  const title = `${meta.formatId}: ${meta.p1} vs. ${meta.p2}`;

  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>${escapeHtml(title)}</title>
<style>
  body { margin: 0; font-family: Verdana, Geneva, sans-serif; }
  .wrapper { max-width: 1180px; margin: 0 auto; }
  h1 { text-align: center; font-size: 1.1em; padding: 8px; }
  .battle-meta { text-align: center; color: #555; font-size: 0.85em; margin-bottom: 8px; }
</style>
</head>
<body>
<div class="wrapper replay-wrapper">
  <h1>${escapeHtml(title)}</h1>
  <div class="battle-meta">Turns: ${result.turns} &middot; Winner: ${result.winner ?? "none"}</div>
  <div class="battle"></div>
  <div class="battle-log"></div>
  <div class="replay-controls"></div>
  <div class="replay-controls-2"></div>
</div>

<script type="text/plain" class="battle-log-data">${logText}</script>
<script src="https://play.pokemonshowdown.com/js/replay-embed.js"></script>
</body>
</html>`;
}

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

/**
 * Write a replay HTML file to disk. Returns the absolute path written.
 *
 * If no `filePath` is given, writes to `sim/replays/<timestamp>.html`.
 */
export function saveReplay(
  result: BattleResult,
  meta: ReplayMetadata,
  filePath?: string
): string {
  const html = generateReplayHtml(result, meta);
  const dest =
    filePath ??
    path.join(
      __dirname,
      "..",
      "replays",
      `${meta.p1}-vs-${meta.p2}-${Date.now()}.html`
    );

  fs.mkdirSync(path.dirname(dest), { recursive: true });
  fs.writeFileSync(dest, html, "utf-8");
  return path.resolve(dest);
}
