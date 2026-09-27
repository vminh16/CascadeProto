#!/bin/bash
# Phase 16 D-45 [DECISION D-45]: M2, VICReg's variance and covariance terms on the query point features against the
# collapse P9 measured (participation ratio 5.56). CR's configuration and schedule, one seed per arm, both arms on one
# GPU, with a monitor (early stop) and a collapse census of the kept checkpoints next to the training.
#
# ARMS  M2-A: --vicreg_var 1 --vicreg_cov 0.04 (VICReg's 25 : 25 : 1 divided by 25)   M2-B: 4 and 0.16
# MONITOR every validation (4 epochs): participation ratio and U on 300 valid episodes; ratio < 8 at epoch 24 -> stop
# TEST  CR and the arms that finish: model / U / U + both / U + both + LP on fixed100, random600 x 3, leak-free;
#       P9 part B on each finished arm
# RULES (d45_monitor.py decide): D45.1 ratio >= 12; D45.2 U - CR's U holds at +1.0 -> base; D45.3 widened but U does
#       not gain -> part B decides; D45.4 both arms fail D45.1 -> stop
#
# Usage:  bash experiments/run_d45.sh smoke      (tests, a 4-epoch toy training with the monitor, 3-episode readers)
#         nohup bash experiments/run_d45.sh full > /workspace/d45_full_console.log 2>&1 &
set -u
cd "$(dirname "$0")/.." || exit 1
MODE=${1:-}
case "$MODE" in
  smoke|full) ;;
  *) echo "usage: $0 smoke|full"; exit 1 ;;
esac
OUT=results/phase16_d45
mkdir -p "$OUT"
D=datasets/S3DIS/blocks_bs1_s1
PY=.venv/bin/python
CR=log_d37/s3dis_S1_N2_K1_point_T4_vip_clean_b1_qrandom/last.pt
[ -f "$CR" ] || { echo "missing: $CR"; exit 1; }
COMMON=(--dataset s3dis --data_path "$D" --cvfold 1 --n_way 2 --k_shot 1 --use_lma false --num_stages 4
        --l2norm_point_proto true --stage_type vip_clean --batch_size 1 --lr_step_epochs 15 --valid_every 4
        --seed 0 --query_order random)
ARM_A=(--vicreg_var 1 --vicreg_cov 0.04)
ARM_B=(--vicreg_var 4 --vicreg_cov 0.16)
RUN=s3dis_S1_N2_K1_point_T4_vip_clean_b1_qrandom
CENSUS=(--checkpoint "cr:ours:1:$CR"
        --checkpoint "vr:ours:1:log_d37/s3dis_S1_N2_K1_point_T4_vip_b1_qrandom/last.pt"
        --checkpoint "e1:ours:1:log_r2/s3dis_S1_N2_K1_point_T4_vip_b1/last.pt"
        --checkpoint "r0:ours:1:log_r2/s3dis_S1_N2_K1_point_T4_vip/last.pt"
        --checkpoint "d29:ours:1:log_r2/s3dis_S1_N2_K1_point_T4_vip_distill1/last.pt"
        --checkpoint "n1:ours:1:log_n1/s3dis_S1_N2_K1_point_T4_vip_b1_ft/last.pt"
        --checkpoint "n1_attn:ours:1:log_n1/s3dis_S1_N2_K1_point_T4_vip_b1_sq_attn_ft/last.pt"
        --checkpoint "n2:ours:1:log_n2/s3dis_S1_N2_K1_point_T4_vip_b1_sq_attn_a0.1/last.pt"
        --checkpoint "a0:ours:1:log_d39/s3dis_S1_N2_K1_point_T0_b1_qrandom_unit/last.pt"
        --checkpoint "a1:ours:1:log_d39/s3dis_S1_N2_K1_point_T0_b1_qrandom_unit_ssp2_aux1/last.pt"
        --checkpoint "m1:ours:1:log_d43/s3dis_S1_N2_K1_point_T4_vip_clean_b1_qrandom_dens/last.pt"
        --checkpoint "vipseg:vipseg:1:vipseg_S1_N2_K1.pt")
LOG="$OUT/d45_${MODE}.log"
{
  echo "=== $(date -Is) commit $(git rev-parse --short HEAD) D45 $MODE"
  if [ "$MODE" = smoke ]; then
    $PY -m pytest tests/test_vicreg.py tests/test_adrm_loss.py tests/test_placement_probe.py -q -p no:cacheprovider \
        || exit 1
    $PY train.py "${COMMON[@]}" "${ARM_A[@]}" --save_dir log_d45_smoke --dry_run true || exit 1
    rm -rf log_d45_smoke/${RUN}_vic1_0.04
    $PY train.py "${COMMON[@]}" "${ARM_A[@]}" --save_dir log_d45_smoke --epochs 4 --episodes_per_epoch 2 \
        > "$OUT/train_m2a_smoke.log" 2>&1 &
    PA=$!
    $PY experiments/d45_monitor.py watch --data_path "$D" --run "m2a:log_d45_smoke/${RUN}_vic1_0.04:$PA" \
        --episodes 3 --poll 5 --out_dir "$OUT/smoke" || exit 1
    wait $PA || exit 1
    grep -h "epoch" "$OUT/smoke/monitor.jsonl"
    $PY experiments/d45_monitor.py census --data_path "$D" "${CENSUS[@]}" --episodes 3 --tag _smoke || exit 1
    M2A="log_d45_smoke/${RUN}_vic1_0.04/last.pt"
    $PY experiments/d43_eval.py test --data_path "$D" --checkpoint "cr:ours:1:$CR" --checkpoint "m2a:ours:1:$M2A" \
        --max_episodes 3 --tag _smoke --out_dir "$OUT" || exit 1
    $PY experiments/p9_placement_probe.py modules --data_path "$D" --checkpoint "m2a:ours:1:$M2A" --max_episodes 3 \
        --train_episodes 4 --tag _m2a_smoke --out_dir "$OUT" || exit 1
    $PY experiments/d45_monitor.py decide --tag _smoke
    echo "=== D45 smoke FINISHED $(date -Is)"
  else
    $PY train.py "${COMMON[@]}" "${ARM_A[@]}" --save_dir log_d45 --resume true > "$OUT/train_m2a.log" 2>&1 &
    PA=$!
    $PY train.py "${COMMON[@]}" "${ARM_B[@]}" --save_dir log_d45 --resume true > "$OUT/train_m2b.log" 2>&1 &
    PB=$!
    $PY experiments/d45_monitor.py watch --data_path "$D" --run "m2a:log_d45/${RUN}_vic1_0.04:$PA" \
        --run "m2b:log_d45/${RUN}_vic4_0.16:$PB" > "$OUT/monitor.log" 2>&1 &
    PW=$!
    $PY experiments/d45_monitor.py census --data_path "$D" "${CENSUS[@]}" > "$OUT/census.log" 2>&1
    grep "^\[census\]" "$OUT/census.log"
    wait $PA; echo "=== TRAIN m2a exit=$? $(date -Is)"
    wait $PB; echo "=== TRAIN m2b exit=$? $(date -Is)"
    wait $PW; grep "^\[watch\]" "$OUT/monitor.log"
    CKS=(--checkpoint "cr:ours:1:$CR")
    PIDS=()
    for arm in m2a:vic1_0.04 m2b:vic4_0.16; do
      name=${arm%%:*}; dir=log_d45/${RUN}_${arm#*:}
      if [ -f "$dir/last.pt" ]; then
        cp "$dir/log_train.txt" "$OUT/train_${name}_S1.log"
        CKS+=(--checkpoint "$name:ours:1:$dir/last.pt" --checkpoint "${name}_best:ours:1:$dir/best.pt")
        $PY experiments/p9_placement_probe.py modules --data_path "$D" --checkpoint "$name:ours:1:$dir/last.pt" \
            --tag "_$name" --out_dir "$OUT" > "$OUT/modules_$name.log" 2>&1 &
        PIDS+=($!)
      else
        echo "=== $name has no last.pt (stopped early)"
      fi
    done
    $PY experiments/d43_eval.py test --data_path "$D" "${CKS[@]}" --out_dir "$OUT" || exit 1
    for p in "${PIDS[@]}"; do wait "$p" || echo "=== a part B run failed"; done
    grep -h "^\[modules\]" "$OUT"/modules_m2*.log | cut -c1-200
    $PY experiments/d45_monitor.py decide
    echo "=== D45 full FINISHED $(date -Is)"
  fi
} >> "$LOG" 2>&1
tail -40 "$LOG"
