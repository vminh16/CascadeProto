#!/bin/bash
# Re-score every full-schedule checkpoint of phases 14-15 and VIP-Seg's released checkpoint with the
# alternative mIoU definitions of pipeline/metrics_alt.py (diagnostic, CHANGELOG 15p).
#
#   bash experiments/rescore_alt_metrics.sh            # on the VM, from the repository root
#
# Hypothesis under test: the paper's own rows were scored with a different mIoU than the VIP-Seg row
# it reports. If so, one of the alternative numbers of our checkpoints lands near the paper's level.
# The VM powers itself off 20 minutes after the last evaluation, leaving time to copy the results.
set -u
cd "$(dirname "$0")/.." || exit 1
D=datasets/S3DIS/blocks_bs1_s1
OUT=results/rescore
mkdir -p "$OUT"
LOG="$OUT/rescore.log"

score () {  # name, checkpoint, extra eval flags...
  local name="$1" ckpt="$2"; shift 2
  if [ ! -f "$ckpt" ]; then echo "=== SKIP $name: $ckpt not found"; return; fi
  echo "=== $(date -Is) $name"
  .venv/bin/python eval.py --dataset s3dis --data_path "$D" --cvfold 0 --n_way 2 --k_shot 1 \
    --checkpoint "$ckpt" --eval_protocol fixed100 --extra_metrics true \
    --result_json "$OUT/$name.json" "$@"
  echo "=== $name exit=$?"
}

{
  echo "=== $(date -Is) commit $(git rev-parse --short HEAD)"
  score vipseg_released   vipseg_S0_N2_K1.pt                                    --model vipseg
  score baseline          log_phase14/s3dis_S0_N2_K1_point_T0/best.pt
  score baseline_l2       log_phase15_bl2/s3dis_S0_N2_K1_point_T0/best.pt
  score lma               log_phase15_t4rows/s3dis_S0_N2_K1_text_T0/best.pt
  score gate_T1           log_phase15_t4rows/s3dis_S0_N2_K1_text_T1/best.pt
  score cascade_T4        log_phase15_t4rows/s3dis_S0_N2_K1_text_T4_noadrm/best.pt
  score full              log_phase14/s3dis_S0_N2_K1_text_T4/best.pt
  echo "=== ALL RUNS FINISHED $(date -Is)"
} >> "$LOG" 2>&1

sudo shutdown -h +20 "rescore finished; powering off in 20 minutes" >> "$LOG" 2>&1 || true
