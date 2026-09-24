#!/bin/bash
# Phase 16 R2 [DECISION D-29]: oracle-direction distillation of the pairwise logit decisions, trained.
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
#                    higher logit-pair cosine to the oracle rule: train both arms on S0, same test
#   R2.2 stop        fixed100 gain < +0.5, or mean random600 gain < +0.5
#   R2.3 in between  otherwise: a second training seed per arm is needed
#   R2.4 mechanism   a gain without a higher cosine is treated as R2.3
#   R2.5             collapse watch (a test class below r0 by more than 3 IoU points)
#
# Usage on the VM:  bash experiments/run_r2.sh train r0|d29|e1   (FOLD=1 by default)
#                   bash experiments/run_r2.sh eval | eval_e1
#
# E1 [DECISION D-30]: r0 on VIP-Seg's update count (batch 1, 24,000 updates, LR halved every 7,200) with
# 13 validations, best.pt per D-22's amended rule 1. Rules E1.1-E1.3 (r2_distill_eval.py decide_e1):
#   E1.1 adopt the schedule if E1 last >= 73.0 on fixed100; E1.2 not the schedule if <= 71.0;
#   E1.3 in between: adopt only if E1 - r0 has CI above 0 and is > 0 on all three random600 draws.
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
      e1) EXTRA=(--batch_size 1 --lr_step_epochs 15 --valid_every 4) ;;  # [D-30]
      *) echo "usage: $0 train r0|d29|e1"; exit 1 ;;
    esac
    [ "$2" = e1 ] && { OUT=results/phase16_e1; mkdir -p "$OUT"; }  # [D-30]
    LOG="$OUT/train_${2}_S${FOLD}.log"
    {
      echo "=== $(date -Is) commit $(git rev-parse --short HEAD) arm $2 fold S$FOLD"
      $PY train.py "${COMMON[@]}" "${HEAD[@]}" "${EXTRA[@]}" --seed 0 --save_dir "$SAVE" --resume true
      rc=$?
      echo "=== TRAIN $2 exit=$rc $(date -Is)"
      exit $rc  # a failed run stops a chained queue (train r0 && train d29 && eval)
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
  eval_e1)
    E1RUN="${RUN}_b1"
    for f in "$E1RUN/last.pt" "$E1RUN/best.pt" "$RUN/last.pt" "$RUN/best.pt" "${RUN}_distill1/best.pt" "$VIP"; do
      [ -f "$f" ] || { echo "missing checkpoint: $f"; exit 1; }
    done
    E1OUT=results/phase16_e1
    mkdir -p "$E1OUT"
    LOG="$E1OUT/eval_S${FOLD}.log"
    {
      echo "=== $(date -Is) commit $(git rev-parse --short HEAD) eval_e1 fold S$FOLD"
      $PY experiments/r2_distill_eval.py test --data_path "$D" --cvfold "$FOLD" --out_dir "$E1OUT" \
          --checkpoint "e1:ours:$FOLD:$E1RUN/last.pt" --checkpoint "e1_best:ours:$FOLD:$E1RUN/best.pt" \
          --checkpoint "r0:ours:$FOLD:$RUN/last.pt" --checkpoint "r0_best:ours:$FOLD:$RUN/best.pt" \
          --checkpoint "d29_best:ours:$FOLD:${RUN}_distill1/best.pt" \
          --checkpoint "vipseg:vipseg:$FOLD:$VIP" || exit 1
      $PY experiments/r2_distill_eval.py decide_e1 --cvfold "$FOLD" --out_dir "$E1OUT"
      echo "=== E1 EVAL FINISHED $(date -Is)"
    } >> "$LOG" 2>&1
    tail -20 "$LOG"
    ;;
  *) echo "usage: $0 train r0|d29|e1 | eval | eval_e1"; exit 1 ;;
esac
