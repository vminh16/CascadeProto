#!/bin/bash
# Table 4 part B (seen-class diagnostic, D-22): S0 checkpoints scored on fold-1 fixed100. Two controls first.
set -u
cd /workspace/CascadeProto || exit 1
D=datasets/S3DIS/blocks_bs1_s1
OUT=results/seen_b2
score () {
  local name="$1" ckpt="$2"
  [ -f "$ckpt" ] || { echo "=== SKIP $name: $ckpt missing"; return; }
  echo "=== $(date -Is) $name $ckpt"
  nice -n 10 .venv/bin/python eval.py --dataset s3dis --data_path "$D" --cvfold 1 --n_way 2 --k_shot 1 \
    --checkpoint "$ckpt" --eval_protocol fixed100 --extra_metrics true \
    --save_dir log_eval_seen_b2 --result_json "$OUT/$name.json" --allow_seen_classes true
  echo "=== $name exit=$?"
}
echo "=== $(date -Is) commit $(git rev-parse --short HEAD)"
score ctrl_baseline_last   log_phase14/s3dis_S0_N2_K1_point_T0/last.pt
score ctrl_full_last       log_phase14/s3dis_S0_N2_K1_text_T4/last.pt
score lma_best             log_phase15_t4rows/s3dis_S0_N2_K1_text_T0/best.pt
score lma_last             log_phase15_t4rows/s3dis_S0_N2_K1_text_T0/last.pt
score gate_last            log_phase15_t4rows/s3dis_S0_N2_K1_text_T1/last.pt
score cascade_best         log_phase15_t4rows/s3dis_S0_N2_K1_text_T4_noadrm/best.pt
score cascade_last         log_phase15_t4rows/s3dis_S0_N2_K1_text_T4_noadrm/last.pt
echo "=== ALL DONE $(date -Is)"
