#!/bin/bash
# Phase 16 P2 [DECISION D-28]: D-26's EM with D-27's base margin filtering the foreground M-step.
# No training. P1's banks (results/phase16_p1/bank_<name>_1000.pt) are reused.
#
# PROTOCOL, fixed before the run.
#   select  S1 checkpoints, S1 *valid* draw: weight {ssp, entropy} x kappa {0.5..8} x T {1, 2} x filter
#           r {0, 0.1, 0.2, 0.3}; freezes the best arm on the VIP-Seg S1 checkpoint (route B, D-25).
#   test    the frozen arm and the same arm with r = 0, fixed100, all four checkpoints, paired bootstrap.
#   decide  the rules of D-28:
#     P2.0 mechanism   the false share of the first foreground M-step falls to <= 0.75x on both VIP-Seg
#                      checkpoints, else stop whatever the mIoU
#     P2.1 go          P2.0 and gain >= +0.5 with CI above 0 on VIP-Seg S1 and S0: train R2
#     P2.2 stop        VIP-Seg S1 has no CI above 0, or P2.0 fails
#     P2.3 in between  otherwise
#     P2.4             the filter is claimable only if frozen - unfiltered has CI above 0 on VIP-Seg S0
#     P2.5             collapse watch
set -u
cd "$(dirname "$0")/.." || exit 1
D=datasets/S3DIS/blocks_bs1_s1
OUT=results/phase16_p2
mkdir -p "$OUT"
LOG="$OUT/p2.log"
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
  $PY experiments/p2_fused_probe.py select --data_path "$D" \
      --checkpoint "vipseg_S1:vipseg:1:$VIP_S1" --checkpoint "ours_S1:ours:1:$OURS_S1" || exit 1
  $PY experiments/p2_fused_probe.py test --data_path "$D" \
      --checkpoint "vipseg_S1:vipseg:1:$VIP_S1" --checkpoint "ours_S1:ours:1:$OURS_S1" \
      --checkpoint "vipseg_S0:vipseg:0:$VIP_S0" --checkpoint "ours_S0:ours:0:$OURS_S0" || exit 1
  $PY experiments/p2_fused_probe.py decide
  echo "=== P2 FINISHED $(date -Is)"
} >> "$LOG" 2>&1
tail -30 "$LOG"
