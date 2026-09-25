#!/bin/bash
# Phase 16 P7 [DECISION D-40]: label propagation on the query's own point graph, seeded by D-39's U + both, with its
# preconditions measured (P7a). Inference only, CR last.pt (D-37's base), S1.
#
# SEEDS Y0 = one-hot of U + both (D39.1's rule on U's rows)
# ARMS  Z = (1 - beta)(I - beta S)^-1 Y0 per query block, S = D^-1/2 (A + A^T) D^-1/2 on a kNN graph:
#       graph xyz (a = 1) / xyzf (xyz neighbours, a = [cos]_+^3) / feat (cosine neighbours, a = [cos]_+^3)
#       x k {8, 16} x beta {0.5, 0.8, 0.9, 0.99} = 24 arms, frozen on S1 valid (largest gain over Y0)
# P7a   homophily, local seed recall / false rate, one-hop vote, same-class reachability per class and condition,
#       on every draw
# TEST  fixed100, random600 seeds 0-2, leak-free draw (seed 4), only if the gate passes; paired bootstrap.
# RULES, fixed before the run (p7_propagation_probe.py decide; "holds" = gain >= threshold, CI > 0, > 0 on all random600):
#   P7.1 gate: frozen - Y0 >= +0.5 on valid, else stop without scoring propagation on a test draw
#   P7.2 adopt: lp - Y0 holds at +0.5;  P7.3 trained form admissible: holds at +1.0;  P7.4 stop: < +0.5
#   P7.5 leak-free gain <= 0 while adopted -> protocol-dependent;  P7.6 mechanism: fg fixed > broken, p_fixed > p_broken
#   P7.7 reported: P7a, recall / precision by condition, lp_u - U (composition)
#
# Usage on the VM:  bash experiments/run_p7.sh smoke          (tests, 5 episodes per draw, files *_smoke)
#                   AUTOSTOP=1 nohup bash experiments/run_p7.sh full > /tmp/p7.log 2>&1 &
#   AUTOSTOP=1 shuts the VM down 15 minutes after the script ends, success or failure.
set -u
cd "$(dirname "$0")/.." || exit 1
MODE=${1:-}
case "$MODE" in
  smoke) EXTRA=(--max_episodes 5 --tag _smoke) ;;
  full) EXTRA=() ;;
  *) echo "usage: $0 smoke|full"; exit 1 ;;
esac
OUT=results/phase16_p7
mkdir -p "$OUT"
if [ "${AUTOSTOP:-0}" = 1 ]; then
  trap 'echo "P7 script ended $(date -Is); shutdown in 15 min" >> results/phase16_p7/autostop.log; nohup bash -c "sleep 900; sudo shutdown -h now" > /dev/null 2>&1 &' EXIT
fi
D=datasets/S3DIS/blocks_bs1_s1
PY=.venv/bin/python
CR=${CR:-log_d37/s3dis_S1_N2_K1_point_T4_vip_clean_b1_qrandom/last.pt}
[ -f "$CR" ] || { echo "missing checkpoint: $CR"; exit 1; }
LOG="$OUT/p7_${MODE}.log"
{
  echo "=== $(date -Is) commit $(git rev-parse --short HEAD) P7 $MODE"
  if [ "$MODE" = smoke ]; then
    $PY -m pytest tests/test_propagation_probe.py tests/test_prototype_probe.py tests/test_self_support.py -q \
        -p no:cacheprovider || exit 1
  fi
  $PY experiments/p7_propagation_probe.py select --data_path "$D" --checkpoint "cr:ours:1:$CR" "${EXTRA[@]}" || exit 1
  echo "=== select done $(date -Is)"
  $PY experiments/p7_propagation_probe.py test --data_path "$D" --checkpoint "cr:ours:1:$CR" "${EXTRA[@]}" || exit 1
  if [ "$MODE" = smoke ]; then
    $PY experiments/p7_propagation_probe.py decide --tag _smoke
  else
    $PY experiments/p7_propagation_probe.py decide
  fi
  echo "=== P7 $MODE FINISHED $(date -Is)"
} >> "$LOG" 2>&1
tail -25 "$LOG"
