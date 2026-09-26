#!/bin/bash
# Phase 16 P9 [DECISION D-44]: where density still enters M1 (D-43), and which module the one remaining run can use.
# Inference only, S1.
#
# step 0  geometry: M1's ball grouping replayed on P8's arm-B events (CPU; run locally, results/phase16_p9/geometry.json
#         is committed and read here)
# A       trace, M1 last.pt: per-slice cosine of V0 / V1 (V0 / V2) features at shared raw points; the ball-count
#         intervention psi = (R_ref - R_cap) / (R_ref - R_low), caps at stage 1, 2, 3 and all three, both directions
# B       modules, CR last.pt, valid: D-42's P9.3-P9.8 (metric, nuisance, heads, components, collapse, neck)
# RULES, fixed before the run (p9_placement_probe.py decide):
#   D44.0 mean m of V0's c-centres >= 15.5 at every stage -> no cap arms
#   D44.1 psi (all stages) >= 0.5 in a direction -> the ball count carries density (M1b candidate); <= 0.2 in both ->
#         encoder branch closed; otherwise undecided (not trained on this budget)
#   D44.2 P9.3-P9.8 admissibility;  D44.3 the run is chosen in a new decision
#
# Usage on the VM:  bash experiments/run_p9.sh smoke          (GPU tests, 3 episodes per stage, files *_smoke)
#                   AUTOSTOP=1 nohup bash experiments/run_p9.sh full > /tmp/p9.log 2>&1 &
#   AUTOSTOP=1 shuts the VM down 15 minutes after the script ends, success or failure.
set -u
cd "$(dirname "$0")/.." || exit 1
MODE=${1:-}
case "$MODE" in
  smoke) EXTRA=(--max_episodes 3 --tag _smoke) ; TRAIN=(--train_episodes 4) ;;
  full) EXTRA=() ; TRAIN=() ;;
  *) echo "usage: $0 smoke|full"; exit 1 ;;
esac
OUT=results/phase16_p9
mkdir -p "$OUT"
if [ "${AUTOSTOP:-0}" = 1 ]; then
  trap 'echo "P9 script ended $(date -Is); shutdown in 15 min" >> results/phase16_p9/autostop.log; nohup bash -c "sleep 900; sudo shutdown -h now" > /dev/null 2>&1 &' EXIT
fi
D=datasets/S3DIS/blocks_bs1_s1
PY=.venv/bin/python
CR=${CR:-log_d37/s3dis_S1_N2_K1_point_T4_vip_clean_b1_qrandom/last.pt}
M1=${M1:-log_d43/s3dis_S1_N2_K1_point_T4_vip_clean_b1_qrandom_dens/last.pt}
for f in "$CR" "$M1"; do [ -f "$f" ] || { echo "missing checkpoint: $f"; exit 1; }; done
LOG="$OUT/p9_${MODE}.log"
{
  echo "=== $(date -Is) commit $(git rev-parse --short HEAD) P9 $MODE"
  if [ "$MODE" = smoke ]; then
    $PY -m pytest tests/test_placement_probe.py tests/test_density_encoder.py -q -s -p no:cacheprovider || exit 1
    $PY experiments/p9_placement_probe.py geometry --data_path "$D" "${EXTRA[@]}" || exit 1
  else
    [ -f "$OUT/geometry.json" ] || { echo "missing $OUT/geometry.json (step 0, run locally)"; exit 1; }
  fi
  $PY experiments/p9_placement_probe.py trace --data_path "$D" --checkpoint "m1:ours:1:$M1" "${EXTRA[@]}" || exit 1
  echo "=== trace done $(date -Is)"
  $PY experiments/p9_placement_probe.py modules --data_path "$D" --checkpoint "cr:ours:1:$CR" "${EXTRA[@]}" \
      "${TRAIN[@]}" || exit 1
  if [ "$MODE" = smoke ]; then
    $PY experiments/p9_placement_probe.py decide --tag _smoke
  else
    $PY experiments/p9_placement_probe.py decide
  fi
  echo "=== P9 $MODE FINISHED $(date -Is)"
} >> "$LOG" 2>&1
tail -30 "$LOG"
