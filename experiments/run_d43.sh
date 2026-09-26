#!/bin/bash
# Phase 16 D-43 [DECISION D-43]: M1, a density-invariant encoder (metric ball neighbourhoods, offsets in metres,
# per-block standardisation, metric coordinates), the first block of D-42's stack, against the clean base CR.
#
# ARM   M1 = CR's configuration (vip_clean, 4 stages, no LMA, L2 prototypes, random query order, E1's schedule:
#       batch 1, 24,000 updates, LR halved every 7,200, 13 validations, seed 0) + --encoder density
# TEST  CR and M1 with their own features: model / U / U + both / U + both + LP on fixed100, random600 x 3, leak-free;
#       P8's condition oracles (valid) and arm B (seed 3) on M1
# RULES, fixed before the run (d43_eval.py decide; stack U + both + LP):
#   D43.1 mechanism: M1's other-condition deficit and own-class uniform drop both <= half of CR's (0.588, 0.374)
#   D43.2 leak-free: m1 - cr >= +1.0 with CI > 0
#   D43.3 standard: holds at +1.0 -> M1 base on both protocols; D43.1 + D43.2 with a fixed100 loss -> base for the
#         leak-free protocol, protocol to the maintainer;  D43.4 stop if D43.1 fails
#
# Usage on the VM:  bash experiments/run_d43.sh smoke      (GPU tests, a dry run, a 2-episode training, 3-episode eval)
#                   AUTOSTOP=1 nohup bash experiments/run_d43.sh full > /tmp/d43.log 2>&1 &
#   AUTOSTOP=1 shuts the VM down 15 minutes after the script ends, success or failure.
set -u
cd "$(dirname "$0")/.." || exit 1
MODE=${1:-}
case "$MODE" in
  smoke|full) ;;
  *) echo "usage: $0 smoke|full"; exit 1 ;;
esac
OUT=results/phase16_d43
mkdir -p "$OUT"
if [ "${AUTOSTOP:-0}" = 1 ]; then
  trap 'echo "D43 script ended $(date -Is); shutdown in 15 min" >> results/phase16_d43/autostop.log; nohup bash -c "sleep 900; sudo shutdown -h now" > /dev/null 2>&1 &' EXIT
fi
D=datasets/S3DIS/blocks_bs1_s1
PY=.venv/bin/python
CR=log_d37/s3dis_S1_N2_K1_point_T4_vip_clean_b1_qrandom
[ -f "$CR/last.pt" ] || { echo "missing: $CR/last.pt"; exit 1; }
COMMON=(--dataset s3dis --data_path "$D" --cvfold 1 --n_way 2 --k_shot 1 --use_lma false --num_stages 4
        --l2norm_point_proto true --stage_type vip_clean --batch_size 1 --lr_step_epochs 15 --valid_every 4
        --seed 0 --query_order random --encoder density)
LOG="$OUT/d43_${MODE}.log"
{
  echo "=== $(date -Is) commit $(git rev-parse --short HEAD) D43 $MODE"
  if [ "$MODE" = smoke ]; then
    $PY -m pytest tests/test_density_encoder.py -q -s -p no:cacheprovider || exit 1
    $PY -m pytest tests/test_condition_split.py tests/test_propagation_probe.py -q -p no:cacheprovider || exit 1
    $PY train.py "${COMMON[@]}" --save_dir log_d43_smoke --dry_run true || exit 1
    $PY train.py "${COMMON[@]}" --save_dir log_d43_smoke --epochs 1 --episodes_per_epoch 2 --valid_every 1 || exit 1
    M1S=log_d43_smoke/s3dis_S1_N2_K1_point_T4_vip_clean_b1_qrandom_dens
    $PY experiments/d43_eval.py test --data_path "$D" --checkpoint "cr:ours:1:$CR/last.pt" \
        --checkpoint "m1:ours:1:$M1S/last.pt" --max_episodes 3 --tag _smoke || exit 1
    $PY experiments/d43_eval.py condition --data_path "$D" --checkpoint "m1:ours:1:$M1S/last.pt" \
        --max_episodes 3 --tag _smoke || exit 1
    $PY experiments/d43_eval.py decide --tag _smoke
    echo "=== D43 smoke FINISHED $(date -Is)"
  else
    M1=log_d43/s3dis_S1_N2_K1_point_T4_vip_clean_b1_qrandom_dens
    $PY train.py "${COMMON[@]}" --save_dir log_d43 --resume true
    rc=$?
    echo "=== TRAIN m1 exit=$rc $(date -Is)"
    [ $rc = 0 ] || exit $rc
    cp "$M1/log_train.txt" "$OUT/train_m1_S1.log"
    $PY experiments/d43_eval.py test --data_path "$D" --checkpoint "cr:ours:1:$CR/last.pt" \
        --checkpoint "m1:ours:1:$M1/last.pt" --checkpoint "m1_best:ours:1:$M1/best.pt" || exit 1
    $PY experiments/d43_eval.py condition --data_path "$D" --checkpoint "m1:ours:1:$M1/last.pt" || exit 1
    $PY experiments/d43_eval.py decide
    echo "=== D43 full FINISHED $(date -Is)"
  fi
} >> "$LOG" 2>&1
tail -30 "$LOG"
