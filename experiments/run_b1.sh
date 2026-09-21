#!/bin/bash
# D-12 probe (CHANGELOG 15v): the baseline on VIP-Seg's own schedule - batch 1, 24,000 steps,
# learning rate halved every 15 epochs = 7,200 steps (VIP-Seg: 7,000) - scored on fixed100.
# Waits for any running train.py / eval.py to finish first, so it can be queued behind another run.
set -u
cd "$(dirname "$0")/.." || exit 1
while pgrep -f "train.py|eval.py" > /dev/null; do sleep 60; done
D=datasets/S3DIS/blocks_bs1_s1
SAVE=log_b1
OUT=results/b1
mkdir -p "$OUT"
LOG="$OUT/b1.log"
COMMON=(--dataset s3dis --data_path "$D" --cvfold 0 --n_way 2 --k_shot 1)
RD="$SAVE/s3dis_S0_N2_K1_point_T0_b1"
{
  echo "=== $(date -Is) commit $(git rev-parse --short HEAD) row baseline_b1"
  .venv/bin/python train.py "${COMMON[@]}" --seed 0 --use_lma false --num_stages 0 \
    --batch_size 1 --lr_step_epochs 15 --save_dir "$SAVE" --resume true
  for CK in best last; do
    .venv/bin/python eval.py "${COMMON[@]}" --checkpoint "$RD/$CK.pt" --eval_protocol fixed100 \
      --extra_metrics true --result_json "$OUT/baseline_b1_${CK}.json"
  done
  echo "=== ROW baseline_b1 DONE $(date -Is)"
  cat "$OUT/baseline_b1_best.json"; echo
  echo "=== ALL RUNS FINISHED $(date -Is)"
} >> "$LOG" 2>&1
