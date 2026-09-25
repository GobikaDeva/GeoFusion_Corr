#!/usr/bin/env bash
# Runs the staged integration plan (docs/baseline_recovery_plan.md) sequentially:
# Stage 1 -> 2 -> 3 -> 4 -> 5. Each stage is evaluated against Stage 0's baseline
# stats before the script continues -- this does NOT auto-advance past a failed gate.
set -euo pipefail
cd "$(dirname "$0")/.."

BASELINE_STATS="runs/A0_baseline/aggregated.json"
if [[ ! -f "$BASELINE_STATS" ]]; then
  echo "Missing $BASELINE_STATS -- run scripts/reproduce_baseline.sh first." >&2
  exit 1
fi

STAGES=(
  configs/stage1_ggf_residual.yaml
  configs/stage2_geocorr_lite_fixed.yaml
  configs/stage3_learned_gate.yaml
  configs/stage4_view_weighting.yaml
  configs/stage5_optional_sampling.yaml
)

for cfg in "${STAGES[@]}"; do
  echo "=== Training $cfg ==="
  python -m training.train --config "$cfg" --seed 0 1 2

  stage_name=$(python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['stage_name'])" "$cfg")

  echo "=== Evaluating $stage_name against baseline gates ==="
  for seed in 0 1 2; do
    python -m evaluation.eval_dtu \
      --config "$cfg" \
      --ckpt "runs/${stage_name}/seed${seed}/ckpt_final.pt" \
      --out "runs/${stage_name}/seed${seed}/eval.json" \
      --baseline_stats "$BASELINE_STATS"
  done

  echo "=== $stage_name done. Inspect runs/${stage_name}/*/eval.json 'gates' before advancing. ==="
done
