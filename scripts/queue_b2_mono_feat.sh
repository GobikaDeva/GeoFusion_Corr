#!/bin/bash
# B2_mono_feat seed 2 (2026-10-09): frozen Depth-Anything-V2-Small reference features on B2,
# no prior input, no L_geo. Phase-3 recipe from B2_nogeo_finetune/seed2, so the matched
# control is runs/B2_nogeo/seed2. Prerequisites (checked below):
#   - patches/B2_mono_feat.patch applied (git apply patches/B2_mono_feat.patch)
#   - the B2 seeds + nogeo queue finished (it produces B2_nogeo_finetune/seed2)
# Launch: setsid nohup bash scripts/queue_b2_mono_feat.sh > runs/queue_b2_mono_feat.log 2>&1 < /dev/null &
cd /home/gobika/Research_Gobika/geofusionnet_geocorr || exit 1
echo $$ > runs/queue_b2_mono_feat.pid
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # B2 training peak ~20.7 GB at batch 2
export HF_HUB_OFFLINE=1  # Depth-Anything-V2-Small from the local HF cache only
log() { echo "$(date) $*"; }
abort() { log "$*"; log QUEUE_ABORTED; exit 1; }

grep -q "class MonoFeatureInjection" models/geofusionnet.py || abort "patches/B2_mono_feat.patch is not applied"
PREV=$(cat runs/queue_b2_seeds_nogeo.pid 2>/dev/null)
while [ -n "$PREV" ] && kill -0 "$PREV" 2>/dev/null; do sleep 300; done  # wait only; never signals it
grep -q QUEUE_DONE runs/queue_b2_seeds_nogeo.log || abort "B2 seeds + nogeo queue did not finish (no QUEUE_DONE)"

wait_gpu_free() {
  local free=0
  while [ $free -lt 2 ]; do
    if [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then
      free=$((free + 1))
    else
      [ $free -gt 0 ] || log "GPU busy: $(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | tr '\n' ' ')"
      free=0
    fi
    [ $free -lt 2 ] && sleep 60
  done
}
step() {
  local out=$1; shift
  wait_gpu_free
  log "START $*  (log $out)"
  "$@" > "$out" 2>&1
  local rc=$?
  log "EXIT $rc  $*"
  if [ $rc -ne 0 ]; then log "QUEUE_ABORTED"; exit $rc; fi
}

name=B2_mono_feat seed=2
C=configs/ablations/$name.yaml R=runs/$name/seed$seed
[ -e runs/B2_nogeo_finetune/seed$seed/ckpt_final.pt ] || abort "missing init runs/B2_nogeo_finetune/seed$seed/ckpt_final.pt"
[ -e $R/ckpt_final.pt ] && abort "$R/ckpt_final.pt already exists; refusing to overwrite"
mkdir -p $R
log "JOB $name seed$seed"
step $R/test_identity.log python -m pytest -q tests/test_mono_feat_identity.py
step $R/train.log python -m training.train --config $C --seed $seed
for c in 500 1000 final; do
  step $R/ckpt_${c}_costvol_diag.log python scripts/costvol_diag.py --config $C --ckpt $R/ckpt_$c.pt \
    --scans scan1 scan48 --out $R/ckpt_${c}_costvol_diag.json
done
step $R/probe_final.log python scripts/costvol_profile_probe.py --config $C --ckpt $R/ckpt_final.pt \
  --scans scan48 scan1 --out_dir $R/probe_final
step $R/eval_1600x1152.log python -m evaluation.eval_dtu --config $C --ckpt $R/ckpt_final.pt \
  --out $R/eval_1600x1152.json
log "JOB_DONE $name seed$seed"
log QUEUE_DONE
