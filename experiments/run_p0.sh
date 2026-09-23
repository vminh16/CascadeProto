#!/bin/bash
# Phase 16 P0 [DECISION D-26]: query-side, entropy-weighted EM refinement on trained checkpoints.
# No training. About 1 GPU-hour: one encoder pass per episode, every arm of the grid on the same pass.
#
# PROTOCOL, fixed before the run.
#   select  S1 checkpoints only, S1 *valid* draw (1,500 episodes) [DECISION D-15] [DECISION D-22]:
#           grid weight {entropy, none, ssp} x kappa {0.5, 1, 2, 4, 8, 16} x T {1, 2, 3}, plus oracle arms;
#           freezes the non-oracle setting with the largest mean gain over the S1 checkpoints.
#   test    the frozen setting once, fixed100 *test* draw: S1 checkpoints on S1, S0 checkpoints on S0
#           (S0 is looked at here for the first time). Paired bootstrap over episodes, 2,000 resamples.
#   decide  the rules of D-26:
#     P0.1 go          frozen - model >= +1.5 on both S1 checkpoints and >= +1.0 on both S0 checkpoints
#     P0.2 stop        < +0.5 on both S1 checkpoints: drop the direction (next: base-class calibration)
#     P0.3 in between  otherwise: train R2 with learned kappa, expect the low end
#     P0.4             the S0 gain's 95% CI, per checkpoint
#     P0.5             "entropy weighting helps" only if frozen - (same, weight=none) has CI > 0 on both S0
#     P0.6             oracle (query labels) - model < 10 on VIP-Seg: the features bound the model
#     collapse watch   a positive gain with a class falling by > 3 points (TIM §3.4)
# Every checkpoint passes eval.py's protocol guard (own fold only).
set -u
cd "$(dirname "$0")/.." || exit 1
D=datasets/S3DIS/blocks_bs1_s1
OUT=results/phase16_p0
mkdir -p "$OUT"
LOG="$OUT/p0.log"
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
  $PY experiments/p0_em_probe.py select --data_path "$D" \
      --checkpoint "vipseg_S1:vipseg:1:$VIP_S1" --checkpoint "ours_S1:ours:1:$OURS_S1" || exit 1
  $PY experiments/p0_em_probe.py test --data_path "$D" \
      --checkpoint "vipseg_S1:vipseg:1:$VIP_S1" --checkpoint "ours_S1:ours:1:$OURS_S1" \
      --checkpoint "vipseg_S0:vipseg:0:$VIP_S0" --checkpoint "ours_S0:ours:0:$OURS_S0" || exit 1
  $PY experiments/p0_em_probe.py decide
  echo "=== P0 FINISHED $(date -Is)"
} >> "$LOG" 2>&1
tail -30 "$LOG"
