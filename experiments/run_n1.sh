#!/bin/bash
# Phase 16 N1 [DECISION D-33]: a zero-initialised support -> query attention neck, warm-started from E1.
#
# ARMS, both from E1 last.pt, seed 0 (identical episodes), batch 1, constant LR 1.25e-4 (E1's last stage),
# 15 epochs x 480 = 7,200 updates, 3 validations:  ctl (no neck)  and  neck (--neck sq_attn).
# TEST: fixed100 + random600 seeds 0, 1, 2 with ctl, neck (last and best) and E1 on identical episodes.
# RULES (r2_distill_eval.py decide_n1): N1.1 go neck - ctl >= +1.0 on fixed100 last, CI > 0, > 0 on all three
# random600 draws; N1.2 stop < +0.5 or mean random600 < +0.5; N1.3 otherwise; N1.4 oracle gap reported.
# Unattended: an on-VM watcher shuts the VM down after this script ends.
set -u
cd "$(dirname "$0")/.." || exit 1
D=datasets/S3DIS/blocks_bs1_s1
OUT=results/phase16_n1
SAVE=log_n1
mkdir -p "$OUT"
PY=.venv/bin/python
E1=${E1:-log_r2/s3dis_S1_N2_K1_point_T4_vip_b1/last.pt}
VIP=${VIP:-vipseg_S1_N2_K1.pt}
[ -f "$E1" ] || { echo "missing checkpoint: $E1"; exit 1; }
COMMON=(--dataset s3dis --data_path "$D" --cvfold 1 --n_way 2 --k_shot 1 --use_lma false --num_stages 4
        --stage_type vip --l2norm_point_proto true --batch_size 1 --lr 1.25e-4 --lr_step_epochs 1000
        --epochs 15 --episodes_per_epoch 480 --valid_every 5 --init_checkpoint "$E1" --seed 0 --save_dir "$SAVE"
        --resume true)
RUN="$SAVE/s3dis_S1_N2_K1_point_T4_vip_b1"
for arm in ctl neck; do
  EXTRA=(); [ "$arm" = neck ] && EXTRA=(--neck sq_attn)
  (  # a subshell: its exit ends this arm only, and a failure stops the queue below
    echo "=== $(date -Is) commit $(git rev-parse --short HEAD) arm $arm"
    $PY train.py "${COMMON[@]}" "${EXTRA[@]}"
    rc=$?
    echo "=== TRAIN $arm exit=$rc $(date -Is)"
    exit $rc
  ) >> "$OUT/train_${arm}.log" 2>&1 || exit 1
done
{
  echo "=== $(date -Is) eval"
  $PY experiments/r2_distill_eval.py test --data_path "$D" --cvfold 1 --out_dir "$OUT" \
      --checkpoint "ctl:ours:1:${RUN}_ft/last.pt" --checkpoint "neck:ours:1:${RUN}_sq_attn_ft/last.pt" \
      --checkpoint "ctl_best:ours:1:${RUN}_ft/best.pt" --checkpoint "neck_best:ours:1:${RUN}_sq_attn_ft/best.pt" \
      --checkpoint "e1:ours:1:$E1" --checkpoint "vipseg:vipseg:1:$VIP" || exit 1
  $PY experiments/r2_distill_eval.py decide_n1 --cvfold 1 --out_dir "$OUT"
  echo "=== N1 FINISHED $(date -Is)"
} >> "$OUT/eval.log" 2>&1
tail -20 "$OUT/eval.log"
