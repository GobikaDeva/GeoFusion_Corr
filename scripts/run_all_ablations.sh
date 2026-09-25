#!/usr/bin/env bash
# Runs the full required ablation table: A0-A6, N1, N2 (docs/evaluation_and_gates.md).
# A0 is run first and its aggregated stats feed every later variant's gate checks.
set -euo pipefail
cd "$(dirname "$0")/.."

bash scripts/reproduce_baseline.sh   # A0

BASELINE_STATS="runs/A0_baseline/aggregated.json"

TRAINED_ABLATIONS=(
  configs/ablations/A1_ggf.yaml
  configs/ablations/A2_normal_warp.yaml
  configs/ablations/A3_geocorr_fixed.yaml
  configs/ablations/A4_learned_gate.yaml
  configs/ablations/A5_view_weighting.yaml
  configs/ablations/A6_deformable.yaml
)

for cfg in "${TRAINED_ABLATIONS[@]}"; do
  echo "=== Running ablation $cfg ==="
  python -m ablations.run_ablation --config "$cfg" --baseline_stats "$BASELINE_STATS"
done

echo "=== N1/N2 controls (evaluation-time perturbations of the A4 checkpoint) ==="
echo "See ablations/controls.py -- apply shuffle_geometry_cues / perturb_priors when"
echo "wiring evaluation/eval_dtu.py's inference call for configs/ablations/N1_*.yaml"
echo "and N2_*.yaml, then re-run evaluation/eval_dtu.py against the A4 checkpoints."
