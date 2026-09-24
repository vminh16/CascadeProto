#!/bin/bash
# Phase 16 N2 [DECISION D-34]: the support -> query attention neck of D-33 trained from scratch on E1's
# schedule (batch 1, 24,000 updates, LR 1e-3 halved every 7,200, 13 validations), seed 0, alpha = 0.1 at
# initialisation, against E1's EXISTING checkpoints (E1 is not retrained).
# TEST: fixed100 + random600 seeds 0, 1, 2; E1 last/best as ctl/ctl_best, N2 last/best as neck/neck_best, and
# VIP-Seg released, on identical episodes.
# RULES (r2_distill_eval.py decide_n1): N1.1 go N2 - E1 >= +1.0 on fixed100 last, CI > 0, > 0 on all three
# random600 draws (then a second seed before S0); N1.2 stop < +0.5 or mean random600 < +0.5; N1.3 otherwise.
# Two independent runs differ by ~1 point from training noise alone; the paired CI covers the episodes only.
#
# Usage on the VM:  AUTOSTOP=1 nohup bash experiments/run_n2.sh > /tmp/n2.log 2>&1 &
#   AUTOSTOP=1 shuts the VM down 15 minutes after the script ends, success or failure, leaving time for a
#   local waiter to copy results/phase16_n2 first. About 2.3 GPU-hours of training plus ~35 min of evaluation.
set -u
cd "$(dirname "$0")/.." || exit 1
if [ "${AUTOSTOP:-0}" = 1 ]; then
  trap 'mkdir -p results/phase16_n2; echo "N2 script ended $(date -Is); shutdown in 15 min" >> results/phase16_n2/autostop.log; nohup bash -c "sleep 900; sudo shutdown -h now" > /dev/null 2>&1 &' EXIT
fi
D=datasets/S3DIS/blocks_bs1_s1
OUT=results/phase16_n2
SAVE=log_n2
mkdir -p "$OUT"
PY=.venv/bin/python
E1RUN=${E1RUN:-log_r2/s3dis_S1_N2_K1_point_T4_vip_b1}
VIP=${VIP:-vipseg_S1_N2_K1.pt}
for f in "$E1RUN/last.pt" "$E1RUN/best.pt" "$VIP"; do
  [ -f "$f" ] || { echo "missing checkpoint: $f"; exit 1; }
done
RUN="$SAVE/s3dis_S1_N2_K1_point_T4_vip_b1_sq_attn_a0.1"
(
  echo "=== $(date -Is) commit $(git rev-parse --short HEAD) N2"
  $PY train.py --dataset s3dis --data_path "$D" --cvfold 1 --n_way 2 --k_shot 1 --use_lma false --num_stages 4 \
      --stage_type vip --l2norm_point_proto true --batch_size 1 --lr_step_epochs 15 --valid_every 4 \
      --neck sq_attn --neck_alpha_init 0.1 --seed 0 --save_dir "$SAVE" --resume true
  rc=$?
  echo "=== TRAIN N2 exit=$rc $(date -Is)"
  exit $rc
) >> "$OUT/train_n2.log" 2>&1 || exit 1
{
  echo "=== $(date -Is) eval"
  $PY experiments/r2_distill_eval.py test --data_path "$D" --cvfold 1 --out_dir "$OUT" \
      --checkpoint "ctl:ours:1:$E1RUN/last.pt" --checkpoint "ctl_best:ours:1:$E1RUN/best.pt" \
      --checkpoint "neck:ours:1:$RUN/last.pt" --checkpoint "neck_best:ours:1:$RUN/best.pt" \
      --checkpoint "vipseg:vipseg:1:$VIP" || exit 1
  $PY experiments/r2_distill_eval.py decide_n1 --cvfold 1 --out_dir "$OUT"
  echo "=== N2 FINISHED $(date -Is)"
} >> "$OUT/eval.log" 2>&1
tail -20 "$OUT/eval.log"
