#!/bin/bash
# Phase 16 C4 + D-37 [DECISION D-36] [DECISION D-37]: where the query-position shortcut lives and what the head
# is worth without it.
#
# C4  (inference) E1 and VIP-Seg's released head re-run with the cross-term native / scrambled / clean, fixed100
#     stored and swapped. C4.0a scrambled == native, C4.0b native == C3; C4.1 clean order-free on both, or the
#     queue stops (exit 3).
# D37 arms on E1's schedule (batch 1, 24,000 updates, LR halved every 7,200, 13 validations, seed 0, S1):
#       VF = E1 (existing) | VR = scrambled head, random query order | CR = clean head, random query order
#       CF = CR by D-37's lemma (tests D37-T5, D37-T6). Episodes identical to E1's; only the query order differs.
# TEST every arm, last and best: fixed100 + random600 seeds 0-2 with oracle columns (r2_distill_eval.py test),
#      fixed100 swapped (c3_query_order.py), leak-free draw seed 4 (d37_eval.py leakfree).
# RULES, fixed before the run (d37_eval.py decide): E1 re-scored within 0.05 of E1's own evaluation, else nothing is
#   read; D37.2 CR swap gap <= 0.1, relabel <= 0.01; D37.1 VR swap gap <= 1.0, relabel <= 0.05 (origin) / > 0.5
#   stop; D37.3 CR - VR >= +1.0 with CI > 0 and > 0 on all random600 -> clean; <= -1.0 mirrored -> scrambled;
#   otherwise clean; D37.4 reported (VF - VR, leak-free levels, the base's oracle gap).
#
# Usage on the VM:  bash experiments/run_d37.sh smoke      (GPU tests, one training step per arm, 5-episode C4, ~15 min)
#                   AUTOSTOP=1 nohup bash experiments/run_d37.sh full > /tmp/d37.log 2>&1 &
#   AUTOSTOP=1 shuts the VM down 15 minutes after the script ends, success or failure.
set -u
cd "$(dirname "$0")/.." || exit 1
MODE=${1:-}
case "$MODE" in
  smoke|full) ;;
  *) echo "usage: $0 smoke|full"; exit 1 ;;
esac
OUT=results/phase16_d37
mkdir -p "$OUT"
if [ "${AUTOSTOP:-0}" = 1 ]; then
  trap 'echo "D37 script ended $(date -Is); shutdown in 15 min" >> results/phase16_d37/autostop.log; nohup bash -c "sleep 900; sudo shutdown -h now" > /dev/null 2>&1 &' EXIT
fi
D=datasets/S3DIS/blocks_bs1_s1
PY=.venv/bin/python
E1RUN=log_r2/s3dis_S1_N2_K1_point_T4_vip_b1
VIP=${VIP:-vipseg_S1_N2_K1.pt}
for f in "$E1RUN/last.pt" "$E1RUN/best.pt" "$VIP" results/phase16_p5/c3_query_order.json; do
  [ -f "$f" ] || { echo "missing: $f"; exit 1; }
done
COMMON=(--dataset s3dis --data_path "$D" --cvfold 1 --n_way 2 --k_shot 1)
HEAD=(--use_lma false --num_stages 4 --l2norm_point_proto true)  # r1_vip4 [D-25]
SCHEDULE=(--batch_size 1 --lr_step_epochs 15 --valid_every 4 --seed 0)  # E1 [D-30]
SAVE=log_d37
VR=$SAVE/s3dis_S1_N2_K1_point_T4_vip_b1_qrandom
CR=$SAVE/s3dis_S1_N2_K1_point_T4_vip_clean_b1_qrandom
C4CK=(--checkpoint "e1:ours:1:$E1RUN/last.pt" --checkpoint "vipseg:vipseg:1:$VIP")
LOG="$OUT/d37_${MODE}.log"
{
  echo "=== $(date -Is) commit $(git rev-parse --short HEAD) D37 $MODE"
  if [ "$MODE" = smoke ]; then
    $PY -m pytest tests/test_d37_trace.py tests/test_vip_stage.py -q -p no:cacheprovider || exit 1  # CPU and GPU checks
    for ST in vip vip_clean; do
      $PY train.py "${COMMON[@]}" "${HEAD[@]}" "${SCHEDULE[@]}" --stage_type $ST --query_order random \
          --save_dir log_d37_smoke --dry_run true || exit 1
    done
    $PY experiments/c4_crossterm_ablation.py run --data_path "$D" "${C4CK[@]}" --max_episodes 5 || exit 1
    $PY experiments/d37_eval.py leakfree --data_path "$D" --checkpoint "e1:ours:1:$E1RUN/last.pt" --max_episodes 3 || exit 1
    echo "=== D37 smoke FINISHED $(date -Is)"
  else
    $PY experiments/c4_crossterm_ablation.py run --data_path "$D" "${C4CK[@]}"
    rc=$?
    echo "=== C4 exit=$rc $(date -Is)"
    [ $rc = 0 ] || exit $rc  # C4.0 failed (1) or C4.1 found a second carrier (3): D-37 is not trained
    for ST in vip vip_clean; do
      $PY train.py "${COMMON[@]}" "${HEAD[@]}" "${SCHEDULE[@]}" --stage_type $ST --query_order random \
          --save_dir "$SAVE" --resume true
      rc=$?
      echo "=== TRAIN $ST exit=$rc $(date -Is)"
      [ $rc = 0 ] || exit $rc
    done
    CK=(--checkpoint "e1:ours:1:$E1RUN/last.pt" --checkpoint "e1_best:ours:1:$E1RUN/best.pt"
        --checkpoint "vr:ours:1:$VR/last.pt" --checkpoint "vr_best:ours:1:$VR/best.pt"
        --checkpoint "cr:ours:1:$CR/last.pt" --checkpoint "cr_best:ours:1:$CR/best.pt")
    $PY experiments/r2_distill_eval.py test --data_path "$D" --cvfold 1 --out_dir "$OUT" "${CK[@]}" || exit 1
    $PY experiments/c3_query_order.py --data_path "$D" "${CK[@]}" --out "$OUT/c3_swap.json" || exit 1
    $PY experiments/d37_eval.py leakfree --data_path "$D" "${CK[@]}" || exit 1
    $PY experiments/d37_eval.py decide
    echo "=== D37 full FINISHED $(date -Is)"
  fi
} >> "$LOG" 2>&1
tail -40 "$LOG"
