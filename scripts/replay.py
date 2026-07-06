"""Generate a Showdown-compatible HTML replay from a checkpoint playing itself.

Loads one or two checkpoints, plays a single self-play game on a curriculum
stage, captures the full protocol log, and writes an HTML file you can open
in any browser to watch the animated battle replay.

Usage:

    # Self-play: same net on both sides
    python scripts/replay.py --checkpoint checkpoints/stage0_experiment1/gen_0100.pt \
        --stage 0 --out replays/gen100_selfplay.html

    # Two checkpoints: p1 vs p2
    python scripts/replay.py --checkpoint checkpoints/stage0_experiment1/gen_0100.pt \
        --checkpoint-p2 checkpoints/stage0_experiment1/gen_0010.pt \
        --stage 0 --out replays/gen100_vs_gen010.html

    # Policy-only (no search, faster):
    python scripts/replay.py --checkpoint checkpoints/stage0_experiment1/gen_0100.pt \
        --stage 0 --policy-only --out replays/gen100_fast.html
"""

from __future__ import annotations

import argparse
import html
import logging
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import action_space  # noqa: E402
import curriculum  # noqa: E402
from checkpoint import load_checkpoint  # noqa: E402
from encoder import encode  # noqa: E402
from evaluation import PolicyAgent, RandomAgent, SearchAgent  # noqa: E402
from search import SearchConfig  # noqa: E402
from sim_client import SimClient, SimError  # noqa: E402

_STAGES = {0: curriculum.STAGE_0, 1: curriculum.STAGE_1}

logger = logging.getLogger(__name__)


def _rng_seed(rng: random.Random) -> list[int]:
    return [rng.randint(0, 0xFFFF) for _ in range(4)]


def play_game_with_log(
    agent_p1,
    agent_p2,
    team_a: list[dict],
    team_b: list[dict],
    sim: SimClient,
    *,
    seed: int,
    max_decisions: int = 300,
) -> list[str] | None:
    """Play one game and return the full protocol log, or None on abort."""
    game_rng = random.Random(seed)
    battle_seed = _rng_seed(game_rng)

    handle, view = sim.new_battle(team_a, team_b, seed=battle_seed)
    decision_idx = 0

    try:
        while not view.terminal and decision_idx < max_decisions:
            # Empty phase: advance engine
            if not view.to_move or view.phase == "none":
                step_seed = _rng_seed(game_rng)
                res = sim.step(handle, {}, step_seed)
                sim.release(handle)
                handle, view = res.child, res.view
                continue

            # Forced decision: auto-step
            forced = action_space.forced_actions(view.legal, view.to_move, view.phase)
            if forced is not None:
                choices = {
                    s: action_space.action_to_choice_contextual(idx, view.legal.get(s))
                    for s, idx in forced.items()
                }
                step_seed = _rng_seed(game_rng)
                res = sim.step(handle, choices, step_seed)
                sim.release(handle)
                handle, view = res.child, res.view
                continue

            # Genuine decision: call agents
            choices: dict[str, str] = {}
            agents = {"p1": agent_p1, "p2": agent_p2}
            for side in view.to_move:
                idx = agents[side].act(view, side, sim, handle)
                choices[side] = action_space.action_to_choice_contextual(idx, view.legal.get(side))

            step_seed = _rng_seed(game_rng)
            res = sim.step(handle, choices, step_seed)
            sim.release(handle)
            handle, view = res.child, res.view
            decision_idx += 1

        # Retrieve the full accumulated log from the final battle state
        full_log = sim.get_log(handle)
        sim.release(handle)
        return full_log

    except (SimError, Exception) as exc:
        logger.error("Game aborted: %s", exc)
        try:
            sim.release(handle)
        except Exception:
            pass
        return None


def _strip_split_sections(log_lines: list[str]) -> list[str]:
    """Convert raw omniscient battle.log to replay-compatible format.

    The engine's internal log uses |split|<side> markers followed by two lines:
    the first with exact HP (owner's view), the second with percentage HP
    (opponent's view). Replay-embed.js doesn't handle these — it expects a
    flat log. We keep the exact-HP line (omniscient perspective) and discard
    the split marker and the percentage duplicate.
    """
    filtered: list[str] = []
    i = 0
    while i < len(log_lines):
        line = log_lines[i]
        if line.startswith("|split|"):
            # Next two lines are secret (exact HP) and public (percentage HP)
            if i + 2 < len(log_lines):
                filtered.append(log_lines[i + 1])  # keep the exact-HP version
                i += 3  # skip |split|, secret, public
            else:
                i += 1
        else:
            filtered.append(line)
            i += 1
    return filtered


def generate_replay_html(log_lines: list[str], p1_name: str, p2_name: str, format_id: str) -> str:
    """Generate a self-contained HTML replay using Showdown's replay-embed.js."""
    cleaned = _strip_split_sections(log_lines)
    log_text = "\n".join(cleaned).replace("</script", "<\\/script")
    title = f"{format_id}: {p1_name} vs. {p2_name}"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>{html.escape(title)}</title>
<style>
  body {{ margin: 0; font-family: Verdana, Geneva, sans-serif; }}
  .wrapper {{ max-width: 1180px; margin: 0 auto; }}
  h1 {{ text-align: center; font-size: 1.1em; padding: 8px; }}
  .battle-meta {{ text-align: center; color: #555; font-size: 0.85em; margin-bottom: 8px; }}
</style>
</head>
<body>
<div class="wrapper replay-wrapper">
  <h1>{html.escape(title)}</h1>
  <div class="battle"></div>
  <div class="battle-log"></div>
  <div class="replay-controls"></div>
  <div class="replay-controls-2"></div>
</div>

<script type="text/plain" class="battle-log-data">{log_text}</script>
<script src="https://play.pokemonshowdown.com/js/replay-embed.js"></script>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a Showdown replay HTML from a checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to gen_NNNN.pt (plays p1, or both sides if no --checkpoint-p2).")
    parser.add_argument("--checkpoint-p2", default=None, help="Optional second checkpoint for p2.")
    parser.add_argument("--stage", type=int, default=0, choices=sorted(_STAGES))
    parser.add_argument("--policy-only", action="store_true", help="Use PolicyAgent (no search) — faster.")
    parser.add_argument("--vs-random", action="store_true", help="P2 uses RandomAgent instead of a checkpoint.")
    parser.add_argument("--seed", type=int, default=None, help="Game seed (random if not set).")
    parser.add_argument("--max-decisions", type=int, default=300)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out", type=str, default=None, help="Output HTML path (default: replays/<auto>.html).")
    # Search config
    parser.add_argument("--expansion-budget", type=int, default=8)
    parser.add_argument("--cfr-iters", type=int, default=5)
    parser.add_argument("--k-actions", type=int, default=6)
    parser.add_argument("--max-chance-children", type=int, default=3)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    search_cfg = SearchConfig(
        k_actions=args.k_actions,
        max_chance_children=args.max_chance_children,
        expansion_budget=args.expansion_budget,
        cfr_iters_per_expansion=args.cfr_iters,
        c_puct=2.0,
    )

    # Load p1 checkpoint
    loaded_p1 = load_checkpoint(args.checkpoint, map_location=args.device)
    net_p1 = loaded_p1.net
    net_p1.eval()

    if args.policy_only:
        agent_p1 = PolicyAgent(net_p1)
    else:
        agent_p1 = SearchAgent(net_p1, search_cfg)

    # Load p2 agent
    p2_name = "self"
    if args.vs_random:
        agent_p2 = RandomAgent(random.Random(args.seed or 123))
        p2_name = "random"
    elif args.checkpoint_p2:
        loaded_p2 = load_checkpoint(args.checkpoint_p2, map_location=args.device)
        net_p2 = loaded_p2.net
        net_p2.eval()
        agent_p2 = PolicyAgent(net_p2) if args.policy_only else SearchAgent(net_p2, search_cfg)
        p2_name = Path(args.checkpoint_p2).stem
    else:
        # Self-play: same net on both sides
        agent_p2 = PolicyAgent(net_p1) if args.policy_only else SearchAgent(net_p1, search_cfg)

    stage = _STAGES[args.stage]
    game_seed = args.seed if args.seed is not None else random.randint(0, 2**31)
    team_a, team_b = stage.sample(random.Random(game_seed))

    sim = SimClient(inherit_stderr=True)
    try:
        log_lines = play_game_with_log(
            agent_p1, agent_p2,
            team_a, team_b,
            sim,
            seed=game_seed,
            max_decisions=args.max_decisions,
        )
    finally:
        sim.close()

    if log_lines is None:
        print("ERROR: Game aborted — no replay generated.", file=sys.stderr)
        sys.exit(1)

    # Determine output path
    p1_name = Path(args.checkpoint).stem
    if args.out:
        out_path = Path(args.out)
    else:
        out_path = Path("replays") / f"{p1_name}_vs_{p2_name}_seed{game_seed}.html"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    replay_html = generate_replay_html(
        log_lines,
        p1_name=f"Fire ({p1_name})",
        p2_name=f"Grass ({p2_name})",
        format_id="gen9championsvgc2026regma",
    )
    out_path.write_text(replay_html, encoding="utf-8")
    print(f"Replay written to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
