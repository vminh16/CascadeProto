#!/bin/bash
# Phase 16 P3 [DECISION D-31]: where E1's prototype error sits (Part A) and a training-free text prior
# (Part B). No training. E1 = route B's base on VIP-Seg's update count (D-30).
#
# PROTOCOL, fixed before the run.
#   select  S1 valid draw: text source {ridge, retrieval tau 30/100} x prompt {template, bare, ensemble,
#           descriptions} x weight {1, entropy} x kappa {0.25..32}; freezes the best arm.
#   test    fixed100: E1, the support-prototype rule, the equal-norm oracle, the frozen arm, its weight
#           sibling and an oracle-weight upper bound; Part A's decomposition; paired bootstrap.
#   decide  P3.0 mechanism (text acc >= 0.60 and alignment CI > 0), P3.1 go, P3.2 stop (oracle weight
#           < +0.5), P3.3 in between, P3.4 entropy weight, P3.5 collapse; Part A interpretation bands.
set -u
cd "$(dirname "$0")/.." || exit 1
D=datasets/S3DIS/blocks_bs1_s1
OUT=results/phase16_p3
mkdir -p "$OUT"
LOG="$OUT/p3.log"
PY=.venv/bin/python
E1=${E1:-log_r2/s3dis_S1_N2_K1_point_T4_vip_b1/last.pt}
[ -f "$E1" ] || { echo "missing checkpoint: $E1"; exit 1; }
{
  echo "=== $(date -Is) commit $(git rev-parse --short HEAD)"
  $PY experiments/p3_probe.py select --data_path "$D" --checkpoint "e1:ours:1:$E1" || exit 1
  $PY experiments/p3_probe.py test --data_path "$D" --checkpoint "e1:ours:1:$E1" || exit 1
  $PY experiments/p3_probe.py decide --name e1
  echo "=== P3 FINISHED $(date -Is)"
} >> "$LOG" 2>&1
tail -30 "$LOG"
