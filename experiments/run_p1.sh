#!/bin/bash
# Phase 16 P1 [DECISION D-27]: base-class calibration of the background on trained checkpoints.
# No training. Per checkpoint: a bank from 1,000 seeded training episodes of its own fold, then scoring.
#
# PROTOCOL, fixed before the run.
#   select  S1 checkpoints only, S1 *valid* draw (1,500 episodes) [DECISION D-15] [DECISION D-22]:
#           margin delta {0, 0.05, 0.1, 0.2}; freezes the delta with the largest mean gain.
#   test    the frozen delta (and delta = 0 as reference) once, fixed100 *test* draw: S1 checkpoints on S1,
#           S0 checkpoints on S0. Paired bootstrap over episodes, 2,000 resamples.
#   decide  the rules of D-27:
#     P1.1 go          gain >= +0.5 with 95% CI above 0 on all four checkpoints: train the calibration (R3)
#     P1.2 stop        no S1 checkpoint gains with CI above 0 (training-free form)
#     P1.3 in between  otherwise
#     P1.4             AUC(false vs true foreground by base margin) >= 0.70 on both VIP-Seg checkpoints:
#                      a trained calibration is justified; < 0.60 on both: drop D-27
#     P1.5             collapse watch: a positive gain with a test class falling by > 3 points
# Every checkpoint passes eval.py's protocol guard (own fold only); no scored class may be in the bank.
set -u
cd "$(dirname "$0")/.." || exit 1
D=datasets/S3DIS/blocks_bs1_s1
OUT=results/phase16_p1
mkdir -p "$OUT"
LOG="$OUT/p1.log"
PY=.venv/bin/python

VIP_S1=${VIP_S1:-vipseg_S1_N2_K1.pt}  # log_s3dis_VIPSeg/log_S1_N2_K1_0.760875/checkpoint.pt, pinned commit, blob 2eda6e2a
VIP_S0=${VIP_S0:-vipseg_S0_N2_K1.pt}
OURS_S1=${OURS_S1:-log_s1/s3dis_S1_N2_K1_point_T0/last.pt}                # our baseline, num_stages=0
OURS_S0=${OURS_S0:-log_phase14/s3dis_S0_N2_K1_point_T0/last.pt}

for f in "$VIP_S1" "$VIP_S0" "$OURS_S1" "$OURS_S0"; do
  [ -f "$f" ] || { echo "missing checkpoint: $f (set VIP_S1/VIP_S0/OURS_S1/OURS_S0)"; exit 1; }
done

{
  echo "=== $(date -Is) commit $(git rev-parse --short HEAD)"
  $PY experiments/p1_bpc_probe.py select --data_path "$D" \
      --checkpoint "vipseg_S1:vipseg:1:$VIP_S1" --checkpoint "ours_S1:ours:1:$OURS_S1" || exit 1
  $PY experiments/p1_bpc_probe.py test --data_path "$D" \
      --checkpoint "vipseg_S1:vipseg:1:$VIP_S1" --checkpoint "ours_S1:ours:1:$OURS_S1" \
      --checkpoint "vipseg_S0:vipseg:0:$VIP_S0" --checkpoint "ours_S0:ours:0:$OURS_S0" || exit 1
  $PY experiments/p1_bpc_probe.py decide
  echo "=== P1 FINISHED $(date -Is)"
} >> "$LOG" 2>&1
tail -30 "$LOG"
