#!/bin/bash
# 2026-10-08 queue, strictly one job at a time:
#   a) B2_capacity seed 0   b) B2_nogeo seed 2 (geo_weight 0.0)   c) B2_capacity seed 1
# Each job is the B2 seed-2 pipeline (runs/B2_capacity/seed2/run_b2.sh + queue_after_b2.sh):
# base 20k -> finetune 6k -> final 1500, costvol_diag on ckpt 500/1000/final, the cost-volume
# probe, and the official 22-scan DTU eval at 1600x1152 (evaluation.eval_dtu, as for A0f).
# Every step waits for the GPU to have no compute processes; any nonzero exit stops the queue.
# Does not touch runs/A0f_fixes_only_short. Launch:
#   setsid nohup bash scripts/queue_b2_seeds_nogeo.sh > runs/queue_b2_seeds_nogeo.log 2>&1 < /dev/null &
cd /home/gobika/Research_Gobika/geofusionnet_geocorr || exit 1
echo $$ > runs/queue_b2_seeds_nogeo.pid
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # B2 training peak ~20.7 GB at batch 2

log() { echo "$(date) $*"; }

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

# step <logfile> <cmd...>: wait for a free GPU, run, stop the whole queue on failure.
step() {
  local out=$1; shift
  wait_gpu_free
  log "START $*  (log $out)"
  "$@" > "$out" 2>&1
  local rc=$?
  log "EXIT $rc  $*"
  if [ $rc -ne 0 ]; then log "QUEUE_ABORTED"; exit $rc; fi
}

# job <name> <seed>: configs/ablations/<name>{_base,_finetune,}.yaml
job() {
  local name=$1 seed=$2
  local C=configs/ablations/$name.yaml R=runs/$name/seed$seed
  for d in runs/${name}_base/seed$seed runs/${name}_finetune/seed$seed $R; do
    if [ -e "$d/ckpt_final.pt" ]; then log "$d/ckpt_final.pt already exists; refusing to overwrite"; log QUEUE_ABORTED; exit 1; fi
  done
  mkdir -p $R
  log "JOB $name seed$seed"
  step $R/train_base.log     python -m training.train --config configs/ablations/${name}_base.yaml --seed $seed
  step $R/train_finetune.log python -m training.train --config configs/ablations/${name}_finetune.yaml --seed $seed
  step $R/train.log          python -m training.train --config $C --seed $seed
  for c in 500 1000 final; do
    step $R/ckpt_${c}_costvol_diag.log python scripts/costvol_diag.py --config $C --ckpt $R/ckpt_$c.pt \
      --scans scan1 scan48 --out $R/ckpt_${c}_costvol_diag.json
  done
  step $R/probe_final.log python scripts/costvol_profile_probe.py --config $C --ckpt $R/ckpt_final.pt \
    --scans scan48 scan1 --out_dir $R/probe_final
  step $R/eval_1600x1152.log python -m evaluation.eval_dtu --config $C --ckpt $R/ckpt_final.pt \
    --out $R/eval_1600x1152.json
  log "JOB_DONE $name seed$seed"
}

job B2_capacity 0
job B2_nogeo 2
job B2_capacity 1
log QUEUE_DONE
