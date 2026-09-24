#!/bin/bash
# Phase 16 P4 [DECISION D-32]: is E1's background prototype contaminated by the episode's own classes,
# causally? No training. Episodes carry the other way's labels on the support points (support=False read).
#
# PROTOCOL, fixed before the run.
#   select  draw A (seed 1, 100 episodes per S1 test pair): purification fraction q in {0.05..0.3} frozen.
#   test    draw B (seed 2): E1, clean_bg (other way's points removed with labels: the intervention), the
#           frozen label-free purification, background-only / foreground-only / full oracle rows.
#   decide  P4.1 causal (clean_bg >= +1.0, CI > 0, floor and wall recall up; < +0.5 stops the line),
#           P4.2 label-free fix (>= +0.5, CI > 0), P4.3 neck (clean_bg - purify >= +1.0 -> learned purification).
# Unattended: an on-VM watcher shuts the VM down when this script ends.
set -u
cd "$(dirname "$0")/.." || exit 1
D=datasets/S3DIS/blocks_bs1_s1
OUT=results/phase16_p4
mkdir -p "$OUT"
LOG="$OUT/p4.log"
PY=.venv/bin/python
E1=${E1:-log_r2/s3dis_S1_N2_K1_point_T4_vip_b1/last.pt}
[ -f "$E1" ] || { echo "missing checkpoint: $E1"; exit 1; }
{
  echo "=== $(date -Is) commit $(git rev-parse --short HEAD)"
  $PY experiments/p4_background_probe.py select --data_path "$D" --checkpoint "e1:ours:1:$E1" || exit 1
  $PY experiments/p4_background_probe.py test --data_path "$D" --checkpoint "e1:ours:1:$E1" || exit 1
  $PY experiments/p4_background_probe.py decide --name e1
  echo "=== P4 FINISHED $(date -Is)"
} >> "$LOG" 2>&1
tail -30 "$LOG"
