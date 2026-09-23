#!/bin/bash
# Phase 16 R2 [DECISION D-29]: oracle-direction distillation of the effective prototype, trained.
#
# ARMS. r0  = VIP-Seg's four alternating modules in our loop (stage_type=vip, T = 4, L2 prototypes, no
#             LMA, ADRM), the r1_vip4 configuration of R1: route B's base [DECISION D-25].
#       d29 = r0 + distill_beta 1.
# SCHEDULE. The full schedule of D-12 (50 x 480 episodes, batch 4, StepLR), seed 0, one run per arm
# (maintainer, 2026-09-23). last.pt is the headline [DECISION D-22].
# TEST. S1 first [DECISION D-22]: fixed100 + random600 seeds 0, 1, 2, r0 / d29 / VIP-Seg released on the
# same episodes of each draw, paired bootstrap per draw.
# RULES, fixed before the run (experiments/r2_distill_eval.py decide):
#   R2.0 reference   r0 - VIP-Seg released on fixed100, reported; below -2 our loop trains the head worse
#   R2.1 go          d29 - r0 >= +1.0 on fixed100 with CI above 0, > 0 on all three random600 draws and a
#                    higher cos(M_eff, O): train both arms on S0, same test
#   R2.2 stop        fixed100 gain < +0.5, or mean random600 gain < +0.5
#   R2.3 in between  otherwise: a second training seed per arm is needed
#   R2.4 mechanism   a gain without a higher cosine is treated as R2.3
#   R2.5             collapse watch (a test class below r0 by more than 3 IoU points)
#
# Usage on the VM:  bash experiments/run_r2.sh train r0|d29      (FOLD=1 by default)
#                   bash experiments/run_r2.sh eval
set -u
cd "$(dirname "$0")/.." || exit 1
D=datasets/S3DIS/blocks_bs1_s1
FOLD=${FOLD:-1}
SAVE=log_r2
OUT=results/phase16_r2
mkdir -p "$OUT"
PY=.venv/bin/python
COMMON=(--dataset s3dis --data_path "$D" --cvfold "$FOLD" --n_way 2 --k_shot 1)
HEAD=(--use_lma false --num_stages 4 --stage_type vip --l2norm_point_proto true)  # r1_vip4 [D-25]
RUN="$SAVE/s3dis_S${FOLD}_N2_K1_point_T4_vip"
VIP=${VIP:-vipseg_S${FOLD}_N2_K1.pt}  # VIP-Seg's released checkpoint of this fold, pinned commit

case "${1:-}" in
  train)
    case "${2:-}" in
      r0) EXTRA=() ;;
      d29) EXTRA=(--distill_beta 1) ;;
      *) echo "usage: $0 train r0|d29"; exit 1 ;;
    esac
    LOG="$OUT/train_${2}_S${FOLD}.log"
    {
      echo "=== $(date -Is) commit $(git rev-parse --short HEAD) arm $2 fold S$FOLD"
      $PY train.py "${COMMON[@]}" "${HEAD[@]}" "${EXTRA[@]}" --seed 0 --save_dir "$SAVE" --resume true
      echo "=== TRAIN $2 exit=$? $(date -Is)"
    } >> "$LOG" 2>&1
    ;;
  eval)
    for f in "$RUN/last.pt" "${RUN}_distill1/last.pt" "$VIP"; do
      [ -f "$f" ] || { echo "missing checkpoint: $f"; exit 1; }
    done
    LOG="$OUT/eval_S${FOLD}.log"
    {
      echo "=== $(date -Is) commit $(git rev-parse --short HEAD) eval fold S$FOLD"
      $PY experiments/r2_distill_eval.py test --data_path "$D" --cvfold "$FOLD" \
          --checkpoint "r0:ours:$FOLD:$RUN/last.pt" --checkpoint "d29:ours:$FOLD:${RUN}_distill1/last.pt" \
          --checkpoint "vipseg:vipseg:$FOLD:$VIP" || exit 1
      $PY experiments/r2_distill_eval.py decide --cvfold "$FOLD"
      echo "=== R2 EVAL FINISHED $(date -Is)"
    } >> "$LOG" 2>&1
    tail -20 "$LOG"
    ;;
  *) echo "usage: $0 train r0|d29 | eval"; exit 1 ;;
esac
