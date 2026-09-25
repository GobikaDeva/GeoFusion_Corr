#!/usr/bin/env python
"""Driver for the required ablation table (docs/evaluation_and_gates.md): A0-A6, N1, N2.

For each variant config in configs/ablations/, this:
    1. trains across DEFAULT_SEEDS (training/seed_utils.py),
    2. evaluates each seed (evaluation/eval_dtu.py),
    3. aggregates mean/std across seeds,
    4. checks the "Repeatability" gate (direction of change consistent across seeds)
       against A0's aggregated stats.

Usage:
    python -m ablations.run_ablation --config configs/ablations/A3_geocorr_fixed.yaml \\
        --baseline_stats runs/A0_baseline/aggregated.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

from training.seed_utils import DEFAULT_SEEDS, aggregate_over_seeds


def run_training(config_path: str, seeds: list) -> None:
    cmd = [sys.executable, "-m", "training.train", "--config", config_path, "--seed", *map(str, seeds)]
    subprocess.run(cmd, check=True)


def run_eval(config_path: str, run_root: str, seeds: list, baseline_stats_path: str | None) -> list:
    stage_name = os.path.splitext(os.path.basename(config_path))[0]
    results = []
    for seed in seeds:
        ckpt = os.path.join(run_root, stage_name, f"seed{seed}", "ckpt_final.pt")
        out = os.path.join(run_root, stage_name, f"seed{seed}", "eval.json")
        cmd = [sys.executable, "-m", "evaluation.eval_dtu", "--config", config_path, "--ckpt", ckpt, "--out", out]
        if baseline_stats_path:
            cmd += ["--baseline_stats", baseline_stats_path]
        subprocess.run(cmd, check=True)
        with open(out) as f:
            results.append(json.load(f))
    return results


def check_repeatability(per_seed_overall: list, baseline_overall_mean: float) -> bool:
    """Repeatability gate: the sign of (variant - baseline) must be consistent across seeds."""
    diffs = [v - baseline_overall_mean for v in per_seed_overall]
    signs = {1 if d > 0 else (-1 if d < 0 else 0) for d in diffs}
    return len(signs - {0}) <= 1  # all non-zero diffs share one sign


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--run_root", default="runs")
    parser.add_argument("--baseline_stats", default=None, help="Aggregated JSON from A0")
    args = parser.parse_args()

    run_training(args.config, args.seeds)
    results = run_eval(args.config, args.run_root, args.seeds, args.baseline_stats)

    aggregated = {
        "overall": aggregate_over_seeds([r["overall"] for r in results]),
        "accuracy": aggregate_over_seeds([r["accuracy"] for r in results]),
        "completeness": aggregate_over_seeds([r["completeness"] for r in results]),
        "runtime_sec": aggregate_over_seeds([r["runtime_sec"] for r in results]),
    }

    if args.baseline_stats:
        with open(args.baseline_stats) as f:
            baseline = json.load(f)
        aggregated["gates"] = {
            "repeatability": check_repeatability(
                [r["overall"] for r in results], baseline["overall"]["mean"]
            ),
        }

    stage_name = os.path.splitext(os.path.basename(args.config))[0]
    out_path = os.path.join(args.run_root, stage_name, "aggregated.json")
    with open(out_path, "w") as f:
        json.dump(aggregated, f, indent=2)
    print(json.dumps(aggregated, indent=2))


if __name__ == "__main__":
    main()
