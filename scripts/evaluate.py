"""Evaluate checkpoints from a training run — Phase-1 progress signal.

Scans a checkpoint directory for ``gen_NNNN.pt`` files, runs
``curriculum_report`` on each, prints a per-generation table, and optionally
writes CSV/JSON.

Headline metrics:
  - **favored_win_rate** ↑  — Fire's win % vs Grass with one net on both sides.
  - **mean_favored_turns** ↓ — how quickly the favored team closes out.

Both curves trending in the right direction is the Phase-1 legibility signal
(``agent/overview.mdc`` §Milestones).

Optional extras:
  - ``--vs-random``    : head_to_head net vs RandomAgent floor.
  - ``--vs-baseline``  : head_to_head net vs a specific reference checkpoint.

Usage (from the repo root, with the venv active):

    python scripts/evaluate.py \\
        --checkpoint-dir checkpoints/smoke \\
        --stage 0 \\
        --n-games 30 \\
        --out eval.csv

    # Fast policy-only pass:
    python scripts/evaluate.py --checkpoint-dir checkpoints/smoke --policy-only

    # Head-to-head vs random:
    python scripts/evaluate.py --checkpoint-dir checkpoints/smoke --vs-random
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import re
import sys
from pathlib import Path

# src-layout: make project modules importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import curriculum
from evaluation import (
    EvalConfig,
    HeadToHeadReport,
    PolicyAgent,
    RandomAgent,
    SearchAgent,
    curriculum_report,
    head_to_head,
)
from search import SearchConfig
from checkpoint import load_checkpoint
from sim_client import SimClient

_STAGES = {0: curriculum.STAGE_0, 1: curriculum.STAGE_1}

# Natural sort: gen_0001.pt < gen_0002.pt < ... < gen_0010.pt
_GEN_PAT = re.compile(r"gen_(\d+)\.pt$")


def _natural_sort_key(p: Path) -> int:
    m = _GEN_PAT.search(p.name)
    return int(m.group(1)) if m else -1


def _scan_checkpoints(directory: str) -> list[Path]:
    """Return all gen_NNNN.pt files in ``directory``, natural-sorted."""
    d = Path(directory)
    paths = sorted(d.glob("gen_*.pt"), key=_natural_sort_key)
    if not paths:
        raise FileNotFoundError(f"No gen_*.pt checkpoints found in {d}")
    return paths


def _generation_from_path(p: Path) -> int:
    m = _GEN_PAT.search(p.name)
    return int(m.group(1)) if m else -1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Phase-1 checkpoints (curriculum win-rate + win speed).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint-dir", required=True, help="Directory of gen_NNNN.pt checkpoints.")
    parser.add_argument("--stage", type=int, default=0, choices=sorted(_STAGES), help="Curriculum stage.")
    parser.add_argument("--n-games", type=int, default=50, help="Games per checkpoint.")
    parser.add_argument("--max-decisions", type=int, default=300, help="Max genuine decisions before aborting a game.")
    parser.add_argument("--seed", type=int, default=0, help="Master seed for eval RNG.")
    parser.add_argument("--device", type=str, default="cpu", help="PyTorch device for checkpoint loading.")
    parser.add_argument("--policy-only", action="store_true", help="Use PolicyAgent (no search) — much faster.")
    # Eval search budget overrides
    parser.add_argument("--expansion-budget", type=int, default=8, help="PUCT expansion budget per search.")
    parser.add_argument("--cfr-iters", type=int, default=5, help="CFR+ iterations per expansion.")
    parser.add_argument("--k-actions", type=int, default=6, help="Top-k actions per player.")
    parser.add_argument("--max-chance-children", type=int, default=3, help="Max chance outcomes per cell.")
    # Output
    parser.add_argument("--out", type=str, default=None, help="Output file path (.csv or .json).")
    # Optional floor / baseline comparisons
    parser.add_argument("--vs-random", action="store_true", help="Also run head-to-head vs RandomAgent.")
    parser.add_argument("--vs-baseline", type=str, default=None, metavar="PATH", help="Also run head-to-head vs this checkpoint.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)

    search_cfg = SearchConfig(
        k_actions=args.k_actions,
        max_chance_children=args.max_chance_children,
        expansion_budget=args.expansion_budget,
        cfr_iters_per_expansion=args.cfr_iters,
        c_puct=2.0,
    )
    eval_cfg = EvalConfig(
        n_games=args.n_games,
        max_decisions=args.max_decisions,
        master_seed=args.seed,
        use_search=not args.policy_only,
        favored_side="p1",
        eval_search_config=search_cfg,
        device=args.device,
    )

    stage = _STAGES[args.stage]
    checkpoints = _scan_checkpoints(args.checkpoint_dir)
    logger.info("Found %d checkpoint(s) in %s", len(checkpoints), args.checkpoint_dir)

    rows: list[dict] = []

    sim = SimClient()
    try:
        for ckpt_path in checkpoints:
            gen = _generation_from_path(ckpt_path)
            logger.info("Evaluating gen %04d (%s) …", gen, ckpt_path.name)

            loaded = load_checkpoint(ckpt_path, map_location=args.device)
            net = loaded.net
            net.eval()

            report = curriculum_report(net, stage, sim, config=eval_cfg)

            row: dict = {
                "generation": gen,
                "n_games": report.n_games,
                "favored_win_rate": f"{report.favored_win_rate:.4f}",
                "draw_rate": f"{report.draw_rate:.4f}",
                "aborted": report.aborted,
                "mean_favored_turns": f"{report.mean_favored_turns:.2f}" if report.mean_favored_turns is not None else "N/A",
                "median_favored_turns": f"{report.median_favored_turns:.2f}" if report.median_favored_turns is not None else "N/A",
                "overall_mean_turns": f"{report.overall_mean_turns:.2f}" if report.overall_mean_turns is not None else "N/A",
            }

            # Optional: head-to-head vs random
            if args.vs_random:
                rng_agent = RandomAgent(random.Random(args.seed))
                agent_a = SearchAgent(net, search_cfg) if not args.policy_only else PolicyAgent(net)
                h2h: HeadToHeadReport = head_to_head(agent_a, rng_agent, stage, sim, config=eval_cfg)
                row["vs_random_win_rate"] = f"{h2h.a_win_rate:.4f}"
                row["vs_random_aborted"] = h2h.aborted

            # Optional: head-to-head vs baseline checkpoint
            if args.vs_baseline:
                baseline_path = Path(args.vs_baseline)
                baseline_loaded = load_checkpoint(baseline_path, map_location=args.device)
                baseline_net = baseline_loaded.net
                baseline_net.eval()
                baseline_agent = SearchAgent(baseline_net, search_cfg)
                net_agent = SearchAgent(net, search_cfg) if not args.policy_only else PolicyAgent(net)
                h2h_base: HeadToHeadReport = head_to_head(net_agent, baseline_agent, stage, sim, config=eval_cfg)
                row["vs_baseline_win_rate"] = f"{h2h_base.a_win_rate:.4f}"
                row["vs_baseline_aborted"] = h2h_base.aborted

            rows.append(row)

            # Print progress row
            print(
                f"gen {gen:04d}: "
                f"win={report.favored_win_rate:.3f} "
                f"draw={report.draw_rate:.3f} "
                f"abort={report.aborted} "
                f"mean_turns={row['mean_favored_turns']} "
                f"median_turns={row['median_favored_turns']}"
            )

    finally:
        sim.close()

    # Write output
    if args.out:
        out_path = Path(args.out)
        if out_path.suffix.lower() == ".json":
            out_path.write_text(json.dumps(rows, indent=2))
            logger.info("Wrote JSON to %s", out_path)
        else:
            # Default to CSV
            if rows:
                with out_path.open("w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    writer.writeheader()
                    writer.writerows(rows)
            logger.info("Wrote CSV to %s", out_path)


if __name__ == "__main__":
    main()
