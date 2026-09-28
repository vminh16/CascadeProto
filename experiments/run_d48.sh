#!/bin/bash
# Phase 16 D-48 [DECISION D-48, amendments 1-3]: P11 on frozen CR features (no training).
#
#   fit        base-class moments, the base learner of [2] (P11.1b) and P11.4's descriptor probes, on 1,050 seeded
#              training episodes with raw labels
#   select     psi, kappa, (eps, rho) on valid by coordinate ascent in the full combination (LP = P7's arm)
#   factorial  the 2^4 combinations of {excl, text, ot, lp} on U + both, P11.3 / P11.4 rules, own / other split,
#              valid_raw mechanism readings; three processes in parallel on the one GPU
#   decide     P11.1b, P11.2, P11.4, P11.5 gates and CR's attribution (Shapley, D48.1')
# Arm A (the correlation neck) is trained in a later step, only if P11.4 holds.
#
# Usage:  bash experiments/run_d48.sh smoke
#         bash experiments/run_d48.sh full      (rerun after an interruption: finished steps are skipped)
set -u
cd "$(dirname "$0")/.." || exit 1
MODE=${1:-}
case "$MODE" in
  smoke|full) ;;
  *) echo "usage: $0 smoke|full"; exit 1 ;;
esac
OUT=results/phase16_d48
mkdir -p "$OUT"
D=datasets/S3DIS/blocks_bs1_s1
PY=.venv/bin/python
CR=log_d37/s3dis_S1_N2_K1_point_T4_vip_clean_b1_qrandom/last.pt
[ -f "$CR" ] || { echo "missing: $CR"; exit 1; }
P=experiments/p11_precheck.py
LOG="$OUT/d48_${MODE}.log"
{
  echo "=== $(date -Is) commit $(git rev-parse --short HEAD) D48 P11 $MODE"
  if [ "$MODE" = smoke ]; then
    $PY -m pytest tests/test_p11.py tests/test_proto_align.py -q -p no:cacheprovider || exit 1
    S=(--data_path "$D" --checkpoint "crsmoke:ours:1:$CR" --bank_episodes 40 --tag _smoke)
    $PY $P fit "${S[@]}" --max_episodes 20 || exit 1
    $PY $P select "${S[@]}" --max_episodes 4 || exit 1
    $PY $P factorial "${S[@]}" --max_episodes 3 || exit 1
    $PY $P decide --name crsmoke --tag _smoke || exit 1
    rm -f results/phase16_p1/bank_crsmoke_40.pt
    echo "=== D48 P11 smoke FINISHED $(date -Is)"
  else
    C=(--data_path "$D" --checkpoint "cr:ours:1:$CR")
    [ -f "$OUT/p11_fit_cr.pt" ] || $PY $P fit "${C[@]}" || exit 1
    grep -h "^\[fit\]" "$OUT"/*.log 2>/dev/null | tail -1
    [ -f "$OUT/p11_select_cr.json" ] || $PY $P select "${C[@]}" || exit 1
    PIDS=()
    for group in "valid fixed100" "random600:0 random600:1 random600:2" "leakfree valid_raw"; do
      args=()
      for d in $group; do
        [ -f "$OUT/p11_${d/:/_seed}_cr.json" ] || args+=(--draw "$d")
      done
      [ ${#args[@]} -eq 0 ] && continue
      $PY $P factorial "${C[@]}" "${args[@]}" > "$OUT/factorial_${group%%[: ]*}.log" 2>&1 &
      PIDS+=($!)
    done
    for p in "${PIDS[@]}"; do wait "$p" || echo "=== a factorial process failed"; done
    grep -h "^\[factorial\]" "$OUT"/factorial_*.log
    $PY $P decide --name cr
    echo "=== D48 P11 full FINISHED $(date -Is)"
  fi
} >> "$LOG" 2>&1
tail -40 "$LOG"
