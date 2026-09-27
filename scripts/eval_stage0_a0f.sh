#!/usr/bin/env bash
# Stage 0 (A0f) final numbers: full-res (1600x1152) 22-scan DTU eval over 3 seeds, same
# protocol as scripts/reeval_baseline_fullres.sh. Seed N's chain is A0_baseline seedN ->
# A0_finetune_clmvs seedN -> A0f seedN; stages whose ckpt_final.pt already exists are
# skipped (seed2's chain was trained 2026-09-25..27).
set -euo pipefail
cd "$(dirname "$0")/.."
FT="configs/ablations/A0_finetune_clmvs.yaml"
A0F="configs/ablations/A0f_fixes_only_short.yaml"
OUT="runs/A0f_fixes_only_short"

eval_seed() {
  [ -f "$OUT/seed$1/eval_1600x1152.json" ] && return
  python -m evaluation.eval_dtu --config "$A0F" --ckpt "$OUT/seed$1/ckpt_final.pt" \
    --out "$OUT/seed$1/eval_1600x1152.json"
}

eval_seed 2
for seed in 0 1; do
  [ -f "runs/A0_finetune_clmvs/seed${seed}/ckpt_final.pt" ] || python -m training.train --config "$FT" --seed "$seed"
  [ -f "$OUT/seed${seed}/ckpt_final.pt" ] || python -m training.train --config "$A0F" --seed "$seed"
  eval_seed "$seed"
done

python - <<'PY'
import json
from training.seed_utils import aggregate_over_seeds
out = "runs/A0f_fixes_only_short"
results = [json.load(open(f"{out}/seed{s}/eval_1600x1152.json")) for s in (0, 1, 2)]
aggregated = {k: aggregate_over_seeds([r[k] for r in results])
              for k in ("overall", "accuracy", "completeness", "runtime_sec")}
aggregated["test_img_wh"] = results[0]["test_img_wh"]
aggregated["n_scans"] = [r["n_scans"] for r in results]
json.dump(aggregated, open(f"{out}/aggregated_1600x1152.json", "w"), indent=2)
print(json.dumps(aggregated, indent=2))
PY
