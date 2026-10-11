#!/bin/bash
# Reduced fusion sweep (user-approved 2026-10-11): A0f s2 vs C1 s2, all 22 scans, geo distance
# {0.1,0.2,0.4,0.8} mm at the official 4 views only. Reuses runs/fusion_sweep/maps (symlink) and
# the 6 finished full-grid JSONs (subset copies). CPU only, 6 workers. Never signals other jobs.
# Launch: setsid nohup bash scripts/queue_fusion_sweep_dist_n4.sh > runs/queue_fusion_sweep_dist_n4.log 2>&1 < /dev/null &
cd /home/gobika/Research_Gobika/geofusionnet_geocorr || exit 1
echo $$ > runs/queue_fusion_sweep_dist_n4.pid
log() { echo "$(date -u) $*"; }
OUT=runs/fusion_sweep_dist_n4
SCANS="scan1 scan4 scan9 scan10 scan11 scan12 scan13 scan15 scan23 scan24 scan29 scan32 scan33 scan34 scan48 scan49 scan62 scan75 scan77 scan110 scan114 scan118"
SETS="geo_d0.1_n4 geo_d0.2_n4 geo_d0.4_n4 geo_d0.8_n4"
step() { local lf=$1; shift; log "START $*  (log $lf)"; "$@" >> "$lf" 2>&1; local rc=$?; log "EXIT $rc  $*"; [ $rc -eq 0 ] || { log QUEUE_ABORTED; exit 1; }; }
log "provenance $(git rev-parse HEAD) $(git diff --quiet scripts/fusion_sweep.py || echo '+uncommitted --settings filter')"
step $OUT/sweep.log env CUDA_VISIBLE_DEVICES="" python scripts/fusion_sweep.py sweep --out_dir $OUT --device cpu \
     --workers 6 --threads 8 --scans $SCANS --settings $SETS
step $OUT/report.log env CUDA_VISIBLE_DEVICES="" python scripts/fusion_sweep.py report --out_dir $OUT --scans $SCANS --settings $SETS
log QUEUE_DONE
