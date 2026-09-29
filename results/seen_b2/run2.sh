#!/bin/bash
# Run 1 of 2: same baseline checkpoint scored on its own fold (S0 test classes, unseen). Waits for run 2 (fold 1, seen) already running.
cd /workspace/CascadeProto || exit 1
while kill -0 58190 2>/dev/null; do sleep 10; done
echo "=== $(date -Is) A_baseline_unseen_last"
nice -n 10 .venv/bin/python eval.py --dataset s3dis --data_path datasets/S3DIS/blocks_bs1_s1 --cvfold 0 --n_way 2 --k_shot 1 \
  --checkpoint log_phase14/s3dis_S0_N2_K1_point_T0/last.pt --eval_protocol fixed100 --extra_metrics true \
  --save_dir log_eval_seen_b2 --result_json results/seen_b2/A_baseline_unseen_last.json
echo "=== exit=$? $(date -Is) ALL DONE"
