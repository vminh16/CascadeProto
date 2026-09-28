#!/bin/bash
# Phase 16 D-46 [DECISION D-46]: P10 (no training) and M5, prototype alignment training, gated by P10.1.
#
# PART A (parallel)  P10.1 the base-class gap of CR on 1,000 training episodes (p10_align_probe.py base / gate)
#                    P10.2 P3's text probe on CR (p3_probe.py select / test / decide, D-31's rules)
#                    P10.3 / P10.4 (amendment 1) Σ_w, Σ_η from base classes -> the base-class metric on valid,
#                          fixed100, random600 -> the K-curve on 5-shot episodes (p10_align_probe.py stats /
#                          metric / kcurve / amend)
# PART B (only if P10.1 holds, launched as soon as the gate reads, next to P10.2)
#                    M5-A --align_weight 0.25, M5-B --align_weight 1 (τ 0.1), CR's configuration, one seed each;
#                    then d43_eval test (CR, arms last and best), P9 part B and P10.1's gap on each arm
# RULES              p10_align_probe.py gate (P10.1) and decide (D46.1-D46.4); p3_probe.py decide (P10.2)
#
# Usage:  bash experiments/run_d46.sh smoke
#         bash experiments/run_d46.sh full      (rerun after an interruption: finished steps are skipped)
set -u
cd "$(dirname "$0")/.." || exit 1
MODE=${1:-}
case "$MODE" in
  smoke|full) ;;
  *) echo "usage: $0 smoke|full"; exit 1 ;;
esac
OUT=results/phase16_d46
mkdir -p "$OUT"
D=datasets/S3DIS/blocks_bs1_s1
PY=.venv/bin/python
CR=log_d37/s3dis_S1_N2_K1_point_T4_vip_clean_b1_qrandom/last.pt
[ -f "$CR" ] || { echo "missing: $CR"; exit 1; }
COMMON=(--dataset s3dis --data_path "$D" --cvfold 1 --n_way 2 --k_shot 1 --use_lma false --num_stages 4
        --l2norm_point_proto true --stage_type vip_clean --batch_size 1 --lr_step_epochs 15 --valid_every 4
        --seed 0 --query_order random)
RUN=s3dis_S1_N2_K1_point_T4_vip_clean_b1_qrandom
ARMS=("m5a:0.25" "m5b:1")
LOG="$OUT/d46_${MODE}.log"
{
  echo "=== $(date -Is) commit $(git rev-parse --short HEAD) D46 $MODE"
  if [ "$MODE" = smoke ]; then
    $PY -m pytest tests/test_proto_align.py tests/test_vicreg.py tests/test_adrm_loss.py tests/test_distill.py -q \
        -p no:cacheprovider || exit 1
    $PY train.py "${COMMON[@]}" --align_weight 1 --save_dir log_d46_smoke --dry_run true || exit 1
    rm -rf "log_d46_smoke/${RUN}_align1"
    $PY train.py "${COMMON[@]}" --align_weight 1 --save_dir log_d46_smoke --epochs 4 --episodes_per_epoch 2 \
        > "$OUT/train_m5b_smoke.log" 2>&1 || exit 1
    grep -h "epoch" "$OUT/train_m5b_smoke.log" | tail -3
    M5="log_d46_smoke/${RUN}_align1/last.pt"
    $PY experiments/p10_align_probe.py base --data_path "$D" --checkpoint "cr:ours:1:$CR" --episodes 4 \
        --max_episodes 4 --tag _smoke || exit 1
    $PY experiments/p10_align_probe.py gate --name cr --tag _smoke; echo "gate exit $?"
    $PY experiments/p10_align_probe.py base --data_path "$D" --checkpoint "m5b:ours:1:$M5" --episodes 4 \
        --max_episodes 4 --tag _smoke || exit 1
    $PY experiments/p3_probe.py select --data_path "$D" --checkpoint "crsmoke:ours:1:$CR" --bank_episodes 4 \
        --max_episodes 3 || exit 1
    $PY experiments/p3_probe.py test --data_path "$D" --checkpoint "crsmoke:ours:1:$CR" --bank_episodes 4 \
        --max_episodes 3 || exit 1
    $PY experiments/p3_probe.py decide --name crsmoke || exit 1
    $PY experiments/d43_eval.py test --data_path "$D" --checkpoint "cr:ours:1:$CR" --checkpoint "m5b:ours:1:$M5" \
        --max_episodes 3 --tag _smoke --out_dir "$OUT" || exit 1
    $PY experiments/p9_placement_probe.py modules --data_path "$D" --checkpoint "m5b:ours:1:$M5" --max_episodes 3 \
        --train_episodes 4 --tag _m5b_smoke --out_dir "$OUT" || exit 1
    $PY experiments/p10_align_probe.py decide --tag _smoke
    $PY experiments/p10_align_probe.py stats --data_path "$D" --checkpoint "cr:ours:1:$CR" --episodes 4         --max_episodes 4 --tag _smoke || exit 1
    $PY experiments/p10_align_probe.py metric --data_path "$D" --checkpoint "cr:ours:1:$CR" --max_episodes 3         --tag _smoke || exit 1
    $PY experiments/p10_align_probe.py kcurve --data_path "$D" --checkpoint "cr:ours:1:$CR" --max_episodes 3         --tag _smoke || exit 1
    $PY experiments/p10_align_probe.py amend --tag _smoke
    rm -f results/phase16_p3/select_crsmoke* results/phase16_p3/test_crsmoke* results/phase16_p1/bank_crsmoke_4.pt
    echo "=== D46 smoke FINISHED $(date -Is)"
  else
    # P10.2 text (D-31's P3 on CR), in the background for the whole run
    if [ -f results/phase16_p3/test_cr.json ]; then
      echo "=== P3 on CR already done"; PT=
    else
      { $PY experiments/p3_probe.py select --data_path "$D" --checkpoint "cr:ours:1:$CR" \
          && $PY experiments/p3_probe.py test --data_path "$D" --checkpoint "cr:ours:1:$CR"; } \
          > "$OUT/p3_cr.log" 2>&1 &
      PT=$!
    fi
    # P10.3 / P10.4 (amendment 1), in the background: statistics -> metric -> K-curve
    if [ -f "$OUT/p10_kcurve.json" ]; then
      echo "=== P10.3 / P10.4 already done"; PM=
    else
      {
        { [ -f "$OUT/p10_stats_cr.pt" ] \
            || $PY experiments/p10_align_probe.py stats --data_path "$D" --checkpoint "cr:ours:1:$CR"; } \
          && { [ -f "$OUT/p10_metric_random600_seed2.json" ] \
            || $PY experiments/p10_align_probe.py metric --data_path "$D" --checkpoint "cr:ours:1:$CR"; } \
          && $PY experiments/p10_align_probe.py kcurve --data_path "$D" --checkpoint "cr:ours:1:$CR"
      } >> "$OUT/p10_amend.log" 2>&1 &
      PM=$!
    fi
    # P10.1 base-class gap on CR
    [ -f "$OUT/p10_base_cr.json" ] || $PY experiments/p10_align_probe.py base --data_path "$D" \
        --checkpoint "cr:ours:1:$CR" > "$OUT/p10_cr.log" 2>&1
    grep -h "^\[p10\]" "$OUT/p10_cr.log"
    $PY experiments/p10_align_probe.py gate --name cr
    GATE=$?
    if [ "$GATE" -eq 0 ]; then
      TRAIN=()
      for arm in "${ARMS[@]}"; do
        name=${arm%%:*}; w=${arm#*:}
        $PY train.py "${COMMON[@]}" --align_weight "$w" --save_dir log_d46 --resume true \
            >> "$OUT/train_$name.log" 2>&1 &
        TRAIN+=("$name:$!")
      done
      for t in "${TRAIN[@]}"; do wait "${t#*:}"; echo "=== TRAIN ${t%%:*} exit=$? $(date -Is)"; done
      CKS=(--checkpoint "cr:ours:1:$CR")
      PIDS=()
      for arm in "${ARMS[@]}"; do
        name=${arm%%:*}; dir=log_d46/${RUN}_align${arm#*:}
        [ -f "$dir/last.pt" ] || { echo "=== $name has no last.pt"; continue; }
        cp "$dir/log_train.txt" "$OUT/train_${name}_S1.log"
        CKS+=(--checkpoint "$name:ours:1:$dir/last.pt" --checkpoint "${name}_best:ours:1:$dir/best.pt")
        { $PY experiments/p9_placement_probe.py modules --data_path "$D" --checkpoint "$name:ours:1:$dir/last.pt" \
            --tag "_$name" --out_dir "$OUT" \
          && $PY experiments/p10_align_probe.py base --data_path "$D" --checkpoint "$name:ours:1:$dir/last.pt"; } \
            > "$OUT/modules_$name.log" 2>&1 &
        PIDS+=($!)
      done
      $PY experiments/d43_eval.py test --data_path "$D" "${CKS[@]}" --out_dir "$OUT" \
          || echo "=== TEST FAILED (see d46_full.log); part B is kept"
      for p in "${PIDS[@]}"; do wait "$p" || echo "=== a part B run failed"; done
      grep -h "^\[modules\]\|^\[p10\]" "$OUT"/modules_m5*.log | cut -c1-200
      $PY experiments/p10_align_probe.py decide
    else
      echo "=== P10.1 fails: part B does not run"
    fi
    if [ -n "${PT:-}" ]; then wait "$PT" || echo "=== P3 on CR failed (see p3_cr.log)"; fi
    if [ -n "${PM:-}" ]; then wait "$PM" || echo "=== P10.3 / P10.4 failed (see p10_amend.log)"; fi
    grep -h "^\[stats\]\|^\[metric\]\|^\[kcurve\]" "$OUT/p10_amend.log" 2>/dev/null | cut -c1-300
    $PY experiments/p10_align_probe.py amend
    grep -h "^\[select\] model\|^\[test\]" "$OUT/p3_cr.log" 2>/dev/null | cut -c1-200
    $PY experiments/p3_probe.py decide --name cr
    echo "=== D46 full FINISHED $(date -Is)"
  fi
} >> "$LOG" 2>&1
tail -40 "$LOG"
