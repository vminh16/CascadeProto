#!/bin/bash
# Phase 16 P6 [DECISION D-38]: where the clean base's prototype gap lies, and whether a training-free query adaptation
# recovers it. Inference only, CR last.pt (D-37's base), S1.
#
# ARMS (one geometry: unit rows, sums of unit features; absent rows keep their support direction):
#   U           support directions (the base of every rule); model and raw support rule reported beside it
#   oracle_x    query's own direction in the background row / foreground rows / every present row (bounds)
#   ssp_*       entropy-gated self-support, grid rho x alpha x T x rows = 54 arms, frozen on S1 valid
#   km3, km5    spherical k-means background, frozen on S1 valid
# TEST fixed100, random600 seeds 0-2, leak-free draw (seed 4); paired bootstrap over episodes.
# RULES, fixed before the run (p6_prototype_probe.py decide):
#   P6.1 location: g_fg >= 2/3 g_all -> foreground, g_bg >= 2/3 g_all -> background, otherwise joint
#   P6.2 / P6.3 go: frozen arm - U >= +1.0 on fixed100 with CI > 0 and > 0 on all random600 -> in D-39
#   P6.4 stop: both < +0.5 -> D-39 is a trained module;   P6.5 reported: leak-free, per class, gap closed
#
# Usage on the VM:  bash experiments/run_p6.sh smoke          (5 episodes per draw, files *_smoke)
#                   AUTOSTOP=1 nohup bash experiments/run_p6.sh full > /tmp/p6.log 2>&1 &
#   AUTOSTOP=1 shuts the VM down 15 minutes after the script ends, success or failure.
set -u
cd "$(dirname "$0")/.." || exit 1
MODE=${1:-}
case "$MODE" in
  smoke) EXTRA=(--max_episodes 5 --tag _smoke) ;;
  full) EXTRA=() ;;
  *) echo "usage: $0 smoke|full"; exit 1 ;;
esac
OUT=results/phase16_p6
mkdir -p "$OUT"
if [ "${AUTOSTOP:-0}" = 1 ]; then
  trap 'echo "P6 script ended $(date -Is); shutdown in 15 min" >> results/phase16_p6/autostop.log; nohup bash -c "sleep 900; sudo shutdown -h now" > /dev/null 2>&1 &' EXIT
fi
D=datasets/S3DIS/blocks_bs1_s1
PY=.venv/bin/python
CR=${CR:-log_d37/s3dis_S1_N2_K1_point_T4_vip_clean_b1_qrandom/last.pt}
[ -f "$CR" ] || { echo "missing checkpoint: $CR"; exit 1; }
LOG="$OUT/p6_${MODE}.log"
{
  echo "=== $(date -Is) commit $(git rev-parse --short HEAD) P6 $MODE"
  if [ "$MODE" = smoke ]; then
    $PY -m pytest tests/test_prototype_probe.py -q -p no:cacheprovider || exit 1
  fi
  $PY experiments/p6_prototype_probe.py select --data_path "$D" --checkpoint "cr:ours:1:$CR" "${EXTRA[@]}" || exit 1
  $PY experiments/p6_prototype_probe.py test --data_path "$D" --checkpoint "cr:ours:1:$CR" "${EXTRA[@]}" || exit 1
  if [ "$MODE" = smoke ]; then
    $PY experiments/p6_prototype_probe.py decide --tag _smoke
  else
    $PY experiments/p6_prototype_probe.py decide
  fi
  echo "=== P6 $MODE FINISHED $(date -Is)"
} >> "$LOG" 2>&1
tail -25 "$LOG"
