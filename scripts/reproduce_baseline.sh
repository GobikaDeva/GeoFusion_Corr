#!/usr/bin/env bash
# Stage 0: reproduce the GeoMVS baseline across >=3 seeds and aggregate mean/std.
# This MUST pass before any GeoCorr staged integration begins (docs/baseline_recovery_plan.md).
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="configs/ablations/A0_baseline.yaml"
SEEDS=(0 1 2)

python -m training.train --config "$CONFIG" --seed "${SEEDS[@]}"

for seed in "${SEEDS[@]}"; do
  python -m evaluation.eval_dtu \
    --config "$CONFIG" \
    --ckpt "runs/A0_baseline/seed${seed}/ckpt_final.pt" \
    --out "runs/A0_baseline/seed${seed}/eval.json"
done

python - <<'PY'
import json
from training.seed_utils import aggregate_over_seeds

results = []
for seed in (0, 1, 2):
    with open(f"runs/A0_baseline/seed{seed}/eval.json") as f:
        results.append(json.load(f))

aggregated = {
    "overall": aggregate_over_seeds([r["overall"] for r in results]),
    "accuracy": aggregate_over_seeds([r["accuracy"] for r in results]),
    "completeness": aggregate_over_seeds([r["completeness"] for r in results]),
    "runtime_sec": aggregate_over_seeds([r["runtime_sec"] for r in results]),
}
with open("runs/A0_baseline/aggregated.json", "w") as f:
    json.dump(aggregated, f, indent=2)
print(json.dumps(aggregated, indent=2))
PY

echo "Stage 0 baseline reproduction complete. Aggregated stats: runs/A0_baseline/aggregated.json"
echo "Use this file as --baseline_stats for every later stage's evaluation and ablation run."
