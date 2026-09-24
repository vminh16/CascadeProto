#!/bin/bash
# Phase 16 P5 [DECISION D-35]: E1's oracle gap split by sampling condition and class presence. No training.
#
# ARMS (S1 only; E1 last.pt, VIP-Seg released for A and C):
#   A  fixed100: recall own vs other, FP in own / other / absent blocks, counterfactuals cf-a / cf-b,
#      the current oracle and a presence-fair oracle, the support rule without head
#   B  intervention (seed 3): the other class of a query block re-sampled in the same scan (V1) and uniformly (V2)
#   C  batch statistics at test (support and query forwards separate, running statistics frozen), fixed100 + random600:0
#   D  leak-free draw (seed 4): support and query sampled uniformly
#   E  training-free dual-condition prototypes against a foreground-count-matched control
# RULES, fixed before the run (p5_condition_probe.py decide):
#   P5.1 cf-a - E1 >= +1.0; then P5.1a phi >= 0.5 with V1 - V0 CI > 0 -> M1 + M2, P5.1b phi < 0.2 -> M1 only
#   P5.2 cf-b - E1 >= +1.0 -> presence estimation (M3)
#   P5.3 batch statistics >= +1.0, CI > 0 on fixed100 and > 0 on random600:0 -> test-time rule (transductive)
#   P5.4 dual - control >= +0.5, CI > 0 -> M2 is trained
#   P5.5 reported: presence-fair oracle, head gain with and without the leak, FP composition
#
# Usage on the VM:  bash experiments/run_p5.sh smoke          (5 episodes per arm, files *_smoke, ~5 min)
#                   AUTOSTOP=1 nohup bash experiments/run_p5.sh full > /tmp/p5.log 2>&1 &
#   AUTOSTOP=1 shuts the VM down 15 minutes after the script ends, success or failure.
set -u
cd "$(dirname "$0")/.." || exit 1
MODE=${1:-}
case "$MODE" in
  smoke) EXTRA=(--max_episodes 5 --tag _smoke) ;;
  full) EXTRA=() ;;
  *) echo "usage: $0 smoke|full"; exit 1 ;;
esac
OUT=results/phase16_p5
mkdir -p "$OUT"
if [ "${AUTOSTOP:-0}" = 1 ]; then
  trap 'echo "P5 script ended $(date -Is); shutdown in 15 min" >> results/phase16_p5/autostop.log; nohup bash -c "sleep 900; sudo shutdown -h now" > /dev/null 2>&1 &' EXIT
fi
D=datasets/S3DIS/blocks_bs1_s1
PY=.venv/bin/python
E1=${E1:-log_r2/s3dis_S1_N2_K1_point_T4_vip_b1/last.pt}
VIP=${VIP:-vipseg_S1_N2_K1.pt}
for f in "$E1" "$VIP"; do
  [ -f "$f" ] || { echo "missing checkpoint: $f"; exit 1; }
done
LOG="$OUT/p5_${MODE}.log"
{
  echo "=== $(date -Is) commit $(git rev-parse --short HEAD) P5 $MODE"
  $PY experiments/p5_condition_probe.py fixed --data_path "$D" --checkpoint "e1:ours:1:$E1" \
      --checkpoint "vipseg:vipseg:1:$VIP" "${EXTRA[@]}" || exit 1
  $PY experiments/p5_condition_probe.py intervene --data_path "$D" --checkpoint "e1:ours:1:$E1" "${EXTRA[@]}" || exit 1
  $PY experiments/p5_condition_probe.py leakfree --data_path "$D" --checkpoint "e1:ours:1:$E1" "${EXTRA[@]}" || exit 1
  if [ "$MODE" = smoke ]; then
    $PY experiments/p5_condition_probe.py decide --tag _smoke
  else
    $PY experiments/p5_condition_probe.py decide
  fi
  echo "=== P5 $MODE FINISHED $(date -Is)"
} >> "$LOG" 2>&1
tail -25 "$LOG"
