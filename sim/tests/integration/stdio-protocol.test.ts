/**
 * Subprocess-level tests for the sim-worker stdio JSON-RPC protocol.
 *
 * These spawn sim-worker.ts as a real child process (like Python's SimClient does)
 * and verify the wire protocol: id echoing, error wrapping, bad JSON recovery,
 * empty line handling, and clean shutdown.
 */
import { describe, it, expect, beforeAll, afterAll } from "vitest";
import { spawn, type ChildProcess } from "node:child_process";
import * as path from "node:path";
import * as readline from "node:readline";

const WORKER_SCRIPT = path.resolve(__dirname, "../../src/sim-worker.ts");
const SIM_CWD = path.resolve(__dirname, "../..");

/** Spawn the sim-worker as a subprocess and return send/receive helpers. */
function spawnWorker() {
  const proc = spawn("npx", ["tsx", WORKER_SCRIPT], {
    stdio: ["pipe", "pipe", "pipe"],
    cwd: SIM_CWD,
  });

  const rl = readline.createInterface({ input: proc.stdout! });
  const lineBuffer: string[] = [];
  const waiters: Array<(line: string) => void> = [];

  rl.on("line", (line: string) => {
    if (waiters.length > 0) {
      waiters.shift()!(line);
    } else {
      lineBuffer.push(line);
    }
  });

  return {
    proc,

    /** Send a JSON message to the worker's stdin. */
    send(msg: object) {
      proc.stdin!.write(JSON.stringify(msg) + "\n");
    },

    /** Send raw text (not necessarily valid JSON) to the worker's stdin. */
    sendRaw(text: string) {
      proc.stdin!.write(text + "\n");
    },

    /** Read the next line from stdout, with a timeout to prevent hangs. */
    readLine(timeoutMs = 10_000): Promise<string> {
      if (lineBuffer.length > 0) return Promise.resolve(lineBuffer.shift()!);
      return new Promise((resolve, reject) => {
        const timer = setTimeout(
          () => reject(new Error(`readLine timed out after ${timeoutMs}ms`)),
          timeoutMs,
        );
        waiters.push((line) => {
          clearTimeout(timer);
          resolve(line);
        });
      });
    },

    /** Wait for a specific string to appear on stderr (e.g. "ready"). */
    waitForStderr(substring: string, timeoutMs = 10_000): Promise<void> {
      return new Promise((resolve, reject) => {
        const timer = setTimeout(
          () => reject(new Error(`stderr wait timed out for "${substring}"`)),
          timeoutMs,
        );
        const handler = (chunk: Buffer) => {
          if (chunk.toString().includes(substring)) {
            clearTimeout(timer);
            proc.stderr!.off("data", handler);
            resolve();
          }
        };
        proc.stderr!.on("data", handler);
      });
    },

    kill() {
      try { proc.kill(); } catch { /* already exited */ }
    },
  };
}

describe("stdio protocol (subprocess)", () => {
  let worker: ReturnType<typeof spawnWorker>;

  beforeAll(async () => {
    worker = spawnWorker();
    await worker.waitForStderr("ready");
  });

  afterAll(() => {
    worker.kill();
  });

  it("echoes the request id in the response", async () => {
    worker.send({ id: 42, cmd: "stats" });
    const resp = JSON.parse(await worker.readLine());
    expect(resp.id).toBe(42);
    expect(resp.ok).toBe(true);
  });

  it("uses null id when request omits id field", async () => {
    worker.send({ cmd: "stats" });
    const resp = JSON.parse(await worker.readLine());
    expect(resp.id).toBeNull();
    expect(resp.ok).toBe(true);
  });

  it("returns ok:false with error for bad JSON input", async () => {
    worker.sendRaw("this is not json");
    const resp = JSON.parse(await worker.readLine());
    expect(resp.ok).toBe(false);
    expect(resp.id).toBeNull();
    expect(resp.error).toContain("bad json");
  });

  it("recovers after bad JSON and processes the next valid request", async () => {
    // Send garbage, consume the error response
    worker.sendRaw("{invalid json}");
    const errResp = JSON.parse(await worker.readLine());
    expect(errResp.ok).toBe(false);

    // Next valid request should succeed
    worker.send({ id: 100, cmd: "stats" });
    const okResp = JSON.parse(await worker.readLine());
    expect(okResp.id).toBe(100);
    expect(okResp.ok).toBe(true);
  });

  it("skips empty lines without producing a response", async () => {
    // Send empty line followed by a valid request
    worker.sendRaw("");
    worker.send({ id: 200, cmd: "stats" });
    const resp = JSON.parse(await worker.readLine());
    // The response should be for the stats command, not an error for the empty line
    expect(resp.id).toBe(200);
    expect(resp.ok).toBe(true);
  });

  it("wraps dispatch errors as ok:false with the error message", async () => {
    worker.send({ id: 7, cmd: "view", handle: 999 });
    const resp = JSON.parse(await worker.readLine());
    expect(resp.id).toBe(7);
    expect(resp.ok).toBe(false);
    expect(resp.error).toContain("unknown handle 999");
  });
});

describe("stdio protocol: shutdown", () => {
  it("close command exits the process with code 0", async () => {
    const w = spawnWorker();
    await w.waitForStderr("ready");

    w.send({ id: 1, cmd: "close" });
    const resp = JSON.parse(await w.readLine());
    expect(resp.ok).toBe(true);
    expect(resp.shutdown).toBe(true);

    // Wait for the process to exit and verify exit code
    const exitCode = await new Promise<number | null>((resolve) => {
      w.proc.on("exit", (code) => resolve(code));
    });
    expect(exitCode).toBe(0);
  });
});
