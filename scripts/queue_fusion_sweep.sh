#!/bin/bash
# Fusion-threshold sweep, A0f seed2 vs C1 (B2_capacity) seed2, all 22 test scans (2026-10-11).
# Inference (dump) needs the GPU, so it waits for the B2_mono_feat queue to finish; the sweep
# and report are CPU only. Never signals the other queues.
# Launch: setsid nohup bash scripts/queue_fusion_sweep.sh > runs/queue_fusion_sweep.log 2>&1 < /dev/null &
cd /home/gobika/Research_Gobika/geofusionnet_geocorr || exit 1
echo $$ > runs/queue_fusion_sweep.pid
log() { echo "$(date) $*"; }
abort() { log "$*"; log QUEUE_ABORTED; exit 1; }
OUT=runs/fusion_sweep
SCANS="scan1 scan4 scan9 scan10 scan11 scan12 scan13 scan15 scan23 scan24 scan29 scan32 scan33 scan34 scan48 scan49 scan62 scan75 scan77 scan110 scan114 scan118"

for q in queue_b2_seeds_nogeo queue_b2_mono_feat; do
  PREV=$(cat runs/$q.pid 2>/dev/null)
  while [ -n "$PREV" ] && kill -0 "$PREV" 2>/dev/null; do sleep 300; done  # wait only
done
grep -q QUEUE_DONE runs/queue_b2_mono_feat.log || log "note: B2_mono_feat queue ended without QUEUE_DONE; continuing (this job is independent of it)"

wait_gpu_free() {
  local free=0
  while [ $free -lt 2 ]; do
    if [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then free=$((free + 1)); else free=0; fi
    [ $free -lt 2 ] && sleep 60
  done
}
step() {
  local out=$1; shift
  log "START $*  (log $out)"
  "$@" > "$out" 2>&1
  local rc=$?
  log "EXIT $rc  $*"
  [ $rc -eq 0 ] || abort "step failed"
}
mkdir -p $OUT
wait_gpu_free
step $OUT/dump.log python scripts/fusion_sweep.py dump --out_dir $OUT --device cuda --scans $SCANS
step $OUT/sweep.log env CUDA_VISIBLE_DEVICES="" python scripts/fusion_sweep.py sweep --out_dir $OUT --device cpu \
  --workers 6 --threads 8 --scans $SCANS
step $OUT/report.log env CUDA_VISIBLE_DEVICES="" python scripts/fusion_sweep.py report --out_dir $OUT --scans $SCANS
log QUEUE_DONE
