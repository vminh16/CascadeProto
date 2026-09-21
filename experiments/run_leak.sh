#!/bin/bash
# D-21 diagnostic: baseline and full model trained with the test classes visible, 20 epochs each
# (past the divergence point), scored on the usual fixed100 test episodes. CHANGELOG 15u.
set -u
cd "$(dirname "$0")/.." || exit 1
D=datasets/S3DIS/blocks_bs1_s1
SAVE=log_leak
OUT=results/leak
mkdir -p "$OUT"
LOG="$OUT/leak.log"
COMMON=(--dataset s3dis --data_path "$D" --cvfold 0 --n_way 2 --k_shot 1)

row () {  # name, run_dir, train flags...
  local name="$1" rd="$SAVE/$2"; shift 2
  echo "=== $(date -Is) row $name"
  .venv/bin/python train.py "${COMMON[@]}" --seed 0 --epochs 20 --train_classes all \
    --save_dir "$SAVE" --resume true "$@"
  .venv/bin/python eval.py "${COMMON[@]}" --checkpoint "$rd/best.pt" --eval_protocol fixed100 \
    --extra_metrics true --result_json "$OUT/${name}.json"
  echo "=== ROW $name DONE $(date -Is)"
  cat "$OUT/${name}.json"; echo
}

{
  echo "=== $(date -Is) commit $(git rev-parse --short HEAD)"
  row baseline_leak s3dis_S0_N2_K1_point_T0_leak --use_lma false --num_stages 0
  row full_leak     s3dis_S0_N2_K1_text_T4_leak
  echo "=== ALL RUNS FINISHED $(date -Is)"
} >> "$LOG" 2>&1
