#!/bin/bash
# Phase 16 P8 [DECISION D-41]: the clean base's oracle gap split by sampling condition, and D-35's condition
# intervention (arm B) re-run on it. Inference only, CR last.pt (D-37's base), S1; rules on U.
#
# A  oracle_own / oracle_other / oracle_fg on U's rows (valid for the rules, fixed100 reported)
# B  arm B of D-35, seed 3, 100 episodes per pair: V0 protocol block, V1 same scan sampled for the other class c,
#    V2 sampled uniformly; phi = (R_V1 - R_V0) / (R_own - R_V0) under U (U + both reported)
# C  cos(s_c, o_c) per block and class, by condition, with the block's recall
# RULES, fixed before the run (p8_condition_probe.py decide):
#   P8.1 g_other >= +1.0 on valid;  P8.3 g_own >= +1.0 on valid
#   P8.2 phi >= 0.5 with CI(R_V1 - R_V0) > 0 -> density causal; phi < 0.2 -> not density; otherwise partial
#   P8.4 condition-balanced training admissible iff P8.1 and P8.2 causal/partial; instance-alignment training iff P8.3;
#        both -> the larger bound first; neither -> revisit the backbone
#
# Usage on the VM:  bash experiments/run_p8.sh smoke          (tests, 5 episodes per stage, files *_smoke)
#                   AUTOSTOP=1 nohup bash experiments/run_p8.sh full > /tmp/p8.log 2>&1 &
#                   AUTOSTOP=1 nohup bash experiments/run_p8.sh intervene > /tmp/p8.log 2>&1 &   (arm B only)
#   AUTOSTOP=1 shuts the VM down 15 minutes after the script ends, success or failure.
set -u
cd "$(dirname "$0")/.." || exit 1
MODE=${1:-}
case "$MODE" in
  smoke) EXTRA=(--max_episodes 5 --tag _smoke) ;;
  full|intervene) EXTRA=() ;;
  *) echo "usage: $0 smoke|full|intervene"; exit 1 ;;
esac
OUT=results/phase16_p8
mkdir -p "$OUT"
if [ "${AUTOSTOP:-0}" = 1 ]; then
  trap 'echo "P8 script ended $(date -Is); shutdown in 15 min" >> results/phase16_p8/autostop.log; nohup bash -c "sleep 900; sudo shutdown -h now" > /dev/null 2>&1 &' EXIT
fi
D=datasets/S3DIS/blocks_bs1_s1
PY=.venv/bin/python
CR=${CR:-log_d37/s3dis_S1_N2_K1_point_T4_vip_clean_b1_qrandom/last.pt}
[ -f "$CR" ] || { echo "missing checkpoint: $CR"; exit 1; }
LOG="$OUT/p8_${MODE}.log"
{
  echo "=== $(date -Is) commit $(git rev-parse --short HEAD) P8 $MODE"
  if [ "$MODE" = smoke ]; then
    $PY -m pytest tests/test_condition_split.py tests/test_propagation_probe.py tests/test_prototype_probe.py -q \
        -p no:cacheprovider || exit 1
  fi
  if [ "$MODE" != intervene ]; then  # intervene: re-run arm B only, the split files already exist
    $PY experiments/p8_condition_probe.py split --data_path "$D" --checkpoint "cr:ours:1:$CR" "${EXTRA[@]}" || exit 1
    echo "=== split done $(date -Is)"
  fi
  $PY experiments/p8_condition_probe.py intervene --data_path "$D" --checkpoint "cr:ours:1:$CR" "${EXTRA[@]}" || exit 1
  if [ "$MODE" = smoke ]; then
    $PY experiments/p8_condition_probe.py decide --tag _smoke
  else
    $PY experiments/p8_condition_probe.py decide
  fi
  echo "=== P8 $MODE FINISHED $(date -Is)"
} >> "$LOG" 2>&1
tail -25 "$LOG"
