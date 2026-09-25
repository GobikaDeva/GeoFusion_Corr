#!/usr/bin/env bash
# Re-evaluates the Stage 0 seeds on the full-res DTU test set (configs/ablations/A0_baseline.yaml
# eval.test_data: 1600x1152, 22 scans). Keeps the earlier 640x512 eval.json/aggregated.json.
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG="configs/ablations/A0_baseline.yaml"
for seed in 0 1 2; do
  python -m evaluation.eval_dtu --config "$CONFIG" \
    --ckpt "runs/A0_baseline/seed${seed}/ckpt_final.pt" \
    --out "runs/A0_baseline/seed${seed}/eval_1600x1152.json"
done
python - <<'PY'
import json
from training.seed_utils import aggregate_over_seeds
results = [json.load(open(f"runs/A0_baseline/seed{s}/eval_1600x1152.json")) for s in (0, 1, 2)]
aggregated = {k: aggregate_over_seeds([r[k] for r in results])
              for k in ("overall", "accuracy", "completeness", "runtime_sec")}
aggregated["test_img_wh"] = results[0]["test_img_wh"]
aggregated["n_scans"] = [r["n_scans"] for r in results]
json.dump(aggregated, open("runs/A0_baseline/aggregated_1600x1152.json", "w"), indent=2)
print(json.dumps(aggregated, indent=2))
PY
