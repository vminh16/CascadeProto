#!/bin/bash
# Seen-class probe (no training): score the S0-trained checkpoints on the fold-1 test episodes.
# S0's training classes are exactly fold 1's test classes, so this measures what our models score on
# classes they were trained on for the full 50 epochs. D-21 exposes each test class ~5x less (all 12
# classes, 20 epochs), so this is the stronger form of the leakage hypothesis: it is the number the
# paper would print as "S0" if its training and evaluation folds were swapped.
# Decision rule, fixed before the run: baseline >= 0.75 supports the swapped-fold reading; <= 0.65
# rejects it. The fold-1 fixed100 cache is built on first use (seed 0).
# Scoring seen classes is refused by eval.py unless --allow_seen_classes true marks the run as a
# diagnostic [DECISION D-22]; VIP-Seg checkpoints record no fold, hence --checkpoint_cvfold.
set -u
cd "$(dirname "$0")/.." || exit 1
D=datasets/S3DIS/blocks_bs1_s1
OUT=results/seen
mkdir -p "$OUT"
LOG="$OUT/seen.log"

score () {  # name, checkpoint, extra eval flags...
  local name="$1" ckpt="$2"; shift 2
  if [ ! -f "$ckpt" ]; then echo "=== SKIP $name: $ckpt not found"; return; fi
  echo "=== $(date -Is) $name"
  .venv/bin/python eval.py --dataset s3dis --data_path "$D" --cvfold 1 --n_way 2 --k_shot 1 \
    --checkpoint "$ckpt" --eval_protocol fixed100 --extra_metrics true \
    --save_dir log_eval_seen --result_json "$OUT/$name.json" --allow_seen_classes true "$@"
  echo "=== $name exit=$?"
  cat "$OUT/$name.json" 2>/dev/null; echo
}

{
  echo "=== $(date -Is) commit $(git rev-parse --short HEAD)"
  score baseline_S0_on_S1        log_phase14/s3dis_S0_N2_K1_point_T0/best.pt
  score full_S0_on_S1            log_phase14/s3dis_S0_N2_K1_text_T4/best.pt
  score vipseg_S0_on_S1          vipseg_S0_N2_K1.pt                                    --model vipseg --checkpoint_cvfold 0
  echo "=== ALL RUNS FINISHED $(date -Is)"
} >> "$LOG" 2>&1
