#!/bin/bash
# Phase 16 D-49 and D-48 arm A [DECISION D-49] [DECISION D-48 amendment 6]: three trainings on one GPU, then tests.
#
#   train   CB  = CR + --condition_balance 0.5 (seed 0)                                    [D-49]
#           A0, A1 = the correlation head, stage_type corr, 2 layers (seeds 0 and 1)      [D-48 amendment 6]
#           CR's schedule for all three (batch 1, 24,000 updates, LR halved every 7,200, 13 validations)
#   test    d49_eval.py mu / test on CR, CB, A0, A1 (last.pt): model, U, U + both, model + LP, U + both + LP,
#           LP re-selected on each checkpoint's valid; fixed100, random600 x 3, leak-free, valid_raw
#   rules   d49_eval.py decide49 (D49.1-D49.5) and decideA (D48.2')
#
# Usage:  bash experiments/run_d49.sh smoke
#         bash experiments/run_d49.sh full      (rerun after an interruption: finished steps are skipped)
set -u
cd "$(dirname "$0")/.." || exit 1
MODE=${1:-}
case "$MODE" in
  smoke|full) ;;
  *) echo "usage: $0 smoke|full"; exit 1 ;;
esac
OUT=results/phase16_d49
mkdir -p "$OUT"
D=datasets/S3DIS/blocks_bs1_s1
PY=.venv/bin/python
CR=log_d37/s3dis_S1_N2_K1_point_T4_vip_clean_b1_qrandom/last.pt
[ -f "$CR" ] || { echo "missing: $CR"; exit 1; }
BASE=(--dataset s3dis --data_path "$D" --cvfold 1 --n_way 2 --k_shot 1 --use_lma false --batch_size 1
      --lr_step_epochs 15 --valid_every 4 --query_order random)
CB_ARGS=("${BASE[@]}" --num_stages 4 --l2norm_point_proto true --stage_type vip_clean --condition_balance 0.5 --seed 0)
A_ARGS=("${BASE[@]}" --num_stages 2 --use_adrm false --stage_type corr)
CB_RUN=s3dis_S1_N2_K1_point_T4_vip_clean_b1_qrandom_cb0.5
A_RUN=s3dis_S1_N2_K1_point_T2_noadrm_corr_b1_qrandom
LOG="$OUT/d49_${MODE}.log"
{
  echo "=== $(date -Is) commit $(git rev-parse --short HEAD) D49 $MODE"
  if [ "$MODE" = smoke ]; then
    $PY -m pytest tests/test_condition_balance.py tests/test_p11.py -q -p no:cacheprovider || exit 1
    $PY train.py "${CB_ARGS[@]}" --save_dir log_d49_smoke --dry_run true || exit 1
    $PY train.py "${A_ARGS[@]}" --seed 0 --save_dir log_d49_smoke --dry_run true || exit 1
    rm -rf log_d49_smoke/$CB_RUN log_d49_smoke/$A_RUN
    $PY train.py "${CB_ARGS[@]}" --save_dir log_d49_smoke --epochs 2 --episodes_per_epoch 2 --valid_every 1 \
        > "$OUT/train_cb_smoke.log" 2>&1 || exit 1
    $PY train.py "${A_ARGS[@]}" --seed 0 --save_dir log_d49_smoke --epochs 2 --episodes_per_epoch 2 --valid_every 1 \
        > "$OUT/train_a0_smoke.log" 2>&1 || exit 1
    grep -h "epoch" "$OUT"/train_*_smoke.log | tail -4
    S=(--checkpoint "cr:ours:1:$CR" --checkpoint "cb:ours:1:log_d49_smoke/$CB_RUN/last.pt"
       --checkpoint "a0:ours:1:log_d49_smoke/$A_RUN/last.pt")
    $PY experiments/d49_eval.py mu --data_path "$D" "${S[@]}" --max_episodes 3 --tag _smoke || exit 1
    $PY experiments/d49_eval.py test --data_path "$D" "${S[@]}" --max_episodes 3 --tag _smoke || exit 1
    $PY experiments/d49_eval.py decide49 --cr cr --arm cb --tag _smoke || exit 1
    $PY experiments/d49_eval.py decideA --cr cr --arm a0 --arm a0 --tag _smoke || exit 1
    echo "=== D49 smoke FINISHED $(date -Is)"
  else
    TRAIN=()
    for spec in "cb:CB" "a0:A:0" "a1:A:1"; do
      name=${spec%%:*}; kind=$(echo "$spec" | cut -d: -f2); seed=$(echo "$spec" | cut -d: -f3)
      if [ "$kind" = CB ]; then args=("${CB_ARGS[@]}"); dir=log_d49/$CB_RUN
      else args=("${A_ARGS[@]}" --seed "$seed"); dir=log_d49/$A_RUN$([ "$seed" = 0 ] || echo "_seed$seed"); fi
      if [ -f "$dir/last.pt" ] && grep -q "done: best valid" "$dir/log_train.txt" 2>/dev/null; then
        echo "=== $name already trained"; continue
      fi
      $PY train.py "${args[@]}" --save_dir log_d49 --resume true >> "$OUT/train_$name.log" 2>&1 &
      TRAIN+=("$name:$!")
    done
    for t in "${TRAIN[@]}"; do wait "${t#*:}"; echo "=== TRAIN ${t%%:*} exit=$? $(date -Is)"; done
    CKS=(--checkpoint "cr:ours:1:$CR")
    for spec in "cb:log_d49/$CB_RUN" "a0:log_d49/$A_RUN" "a1:log_d49/${A_RUN}_seed1"; do
      name=${spec%%:*}; dir=${spec#*:}
      [ -f "$dir/last.pt" ] || { echo "=== $name has no last.pt"; continue; }
      cp "$dir/log_train.txt" "$OUT/train_${name}_S1.log"
      CKS+=(--checkpoint "$name:ours:1:$dir/last.pt")
    done
    $PY experiments/d49_eval.py mu --data_path "$D" "${CKS[@]}" || exit 1
    PIDS=()
    for i in 1 3 5 7; do  # one process per checkpoint (pairs of the flag and its value)
      ck=("${CKS[@]:$((i - 1)):2}")
      [ ${#ck[@]} -eq 2 ] || continue
      n=$(echo "${ck[1]}" | cut -d: -f1)
      $PY experiments/d49_eval.py test --data_path "$D" "${ck[@]}" > "$OUT/test_$n.log" 2>&1 &
      PIDS+=($!)
    done
    for p in "${PIDS[@]}"; do wait "$p" || echo "=== a test process failed"; done
    grep -h "^\[test\]" "$OUT"/test_*.log | cut -c1-300
    $PY experiments/d49_eval.py decide49 --cr cr --arm cb
    $PY experiments/d49_eval.py decideA --cr cr --arm a0 --arm a1
    echo "=== D49 full FINISHED $(date -Is)"
  fi
} >> "$LOG" 2>&1
tail -40 "$LOG"
