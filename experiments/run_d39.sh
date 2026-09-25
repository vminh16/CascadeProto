#!/bin/bash
# Phase 16 D-39 [DECISION D-39]: trained self-support prototypes on the clean base, and the combined background rule.
#
# ARMS (S1, E1's schedule: batch 1, 24,000 updates, LR halved every 7,200, 13 validations, seed 0, random query order):
#   A0 = prototype_rule unit, no head (trained prototype matching)
#   A1 = A0 + two trained self-support steps (alpha_bg, alpha_fg learned) + support_aux 1
#   CR = D-37's clean base (existing checkpoint, reference)
# TEST every checkpoint (last, best) with model / base / ssp_bg / km3 / both / oracle on fixed100, random600 x 3 and
#   the leak-free draw (d39_eval.py test).
# RULES, fixed before the run (d39_eval.py decide; "holds" = gain >= threshold, CI > 0, > 0 on all random600):
#   D39.1 cr both - better single holds at +0.5 -> combined background rule, else the better single
#   D39.2 a1 - a0 holds at +1.0 -> trained self-support; < +0.5 -> not
#   D39.3 candidate (A1 if D39.2 else A0, with D39.1's rule) - cr model holds at +1.0 -> new base
#   D39.4 reported: learned alphas, oracle gaps, leak-free, per class
#
# Usage on the VM:  bash experiments/run_d39.sh smoke      (tests, one step per arm, 3-episode evaluation)
#                   AUTOSTOP=1 nohup bash experiments/run_d39.sh full > /tmp/d39.log 2>&1 &
#   AUTOSTOP=1 shuts the VM down 15 minutes after the script ends, success or failure.
set -u
cd "$(dirname "$0")/.." || exit 1
MODE=${1:-}
case "$MODE" in
  smoke|full) ;;
  *) echo "usage: $0 smoke|full"; exit 1 ;;
esac
OUT=results/phase16_d39
mkdir -p "$OUT"
if [ "${AUTOSTOP:-0}" = 1 ]; then
  trap 'echo "D39 script ended $(date -Is); shutdown in 15 min" >> results/phase16_d39/autostop.log; nohup bash -c "sleep 900; sudo shutdown -h now" > /dev/null 2>&1 &' EXIT
fi
D=datasets/S3DIS/blocks_bs1_s1
PY=.venv/bin/python
CR=log_d37/s3dis_S1_N2_K1_point_T4_vip_clean_b1_qrandom
[ -f "$CR/last.pt" ] || { echo "missing: $CR/last.pt"; exit 1; }
COMMON=(--dataset s3dis --data_path "$D" --cvfold 1 --n_way 2 --k_shot 1 --use_lma false --num_stages 0
        --prototype_rule unit --batch_size 1 --lr_step_epochs 15 --valid_every 4 --seed 0 --query_order random)
A1X=(--self_support_steps 2 --support_aux 1)
LOG="$OUT/d39_${MODE}.log"
{
  echo "=== $(date -Is) commit $(git rev-parse --short HEAD) D39 $MODE"
  if [ "$MODE" = smoke ]; then
    $PY -m pytest tests/test_self_support.py tests/test_prototype_probe.py -q -p no:cacheprovider || exit 1
    $PY train.py "${COMMON[@]}" --save_dir log_d39_smoke --dry_run true || exit 1
    $PY train.py "${COMMON[@]}" "${A1X[@]}" --save_dir log_d39_smoke --dry_run true || exit 1
    $PY experiments/d39_eval.py test --data_path "$D" --checkpoint "cr:ours:1:$CR/last.pt" \
        --max_episodes 3 --tag _smoke || exit 1
    echo "=== D39 smoke FINISHED $(date -Is)"
  else
    SAVE=log_d39
    A0=$SAVE/s3dis_S1_N2_K1_point_T0_b1_qrandom_unit
    A1=$SAVE/s3dis_S1_N2_K1_point_T0_b1_qrandom_unit_ssp2_aux1
    for X in a0 a1; do
      if [ $X = a0 ]; then EXTRA=(); else EXTRA=("${A1X[@]}"); fi
      $PY train.py "${COMMON[@]}" "${EXTRA[@]}" --save_dir "$SAVE" --resume true
      rc=$?
      echo "=== TRAIN $X exit=$rc $(date -Is)"
      [ $rc = 0 ] || exit $rc
    done
    cp "$A0/log_train.txt" "$OUT/train_a0_S1.log"; cp "$A1/log_train.txt" "$OUT/train_a1_S1.log"
    $PY experiments/d39_eval.py test --data_path "$D" --checkpoint "cr:ours:1:$CR/last.pt" \
        --checkpoint "a0:ours:1:$A0/last.pt" --checkpoint "a1:ours:1:$A1/last.pt" \
        --checkpoint "a0_best:ours:1:$A0/best.pt" --checkpoint "a1_best:ours:1:$A1/best.pt" || exit 1
    $PY experiments/d39_eval.py decide
    echo "=== D39 full FINISHED $(date -Is)"
  fi
} >> "$LOG" 2>&1
tail -30 "$LOG"
