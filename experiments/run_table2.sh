#!/bin/bash
# Table 2 of the paper on S3DIS: one training per (way, shot, fold, modality) cell, then best/last evaluation on fixed100.
#
#   text   (P3 + one cell of P2)  full_S1_N2K1, and N2K5, N3K1, N3K5 on S0 and S1                          7 runs
#   image  (P5)                   N2K1, N2K5, N3K1, N3K5 on S0 and S1, modality image  [DECISION D-47]     8 runs
#   audio  (P5)                   the same with modality audio                                            8 runs
#
# Full model, the paper's schedule (50 epochs x 480 episodes, batch 4, D-12). Every cell is its own training: VIP-Seg's
# released logs, the Table 2 row it is compared with, are one run per setting; a checkpoint of another setting is not a cell.
# N2K1 S0 of the text row already exists (Table 4's full model, log_phase14). Results land next to each checkpoint as
# eval_<best|last>_fixed100.json; experiments/summarize.py prints Table 2 (text, image, audio) from them.
#
# Every step is idempotent: training resumes from resume.pt, a finished training or evaluation is skipped, so rerun the
# same command after an interruption.
#
# Usage:  bash experiments/run_table2.sh smoke     CPU tests, then dry runs on real data (text N3K5, image, audio)
#         bash experiments/run_table2.sh list      print the pending steps
#         bash experiments/run_table2.sh full      train and evaluate everything pending
#         SUBSET=text|image|audio bash experiments/run_table2.sh full     one modality only
# Refuses to start while another train.py or eval.py is running on the GPU, unless SHARE_GPU=1 (do not kill other runs).
set -u
cd "$(dirname "$0")/.." || exit 1
MODE=${1:-}
case "$MODE" in
  smoke|list|full) ;;
  *) echo "usage: $0 smoke|list|full"; exit 1 ;;
esac
PY=.venv/bin/python
D=datasets/S3DIS/blocks_bs1_s1
SAVE=log_phase14
OUT=results/table2
mkdir -p "$OUT"
SUBSET=${SUBSET:-all}

names () {  # run names of Table 2 for one modality: text = P3 plus the S1 cell of P2's full model; image/audio = P5
  $PY - "$1" <<'PYEOF'
import sys
from experiments import phase14
m = sys.argv[1]
runs = phase14.all_runs()
if m == "text":
    picked = [r.name for r in runs if r.priority == "P3" or r.name == "full_S1_N2K1"]
else:
    picked = [r.name for r in runs if r.priority == "P5" and r.name.startswith(f"full_{m}_")]
print(" ".join(picked))
PYEOF
}

plan () {  # extra phase14 flags (--list prints, none runs)
  case "$SUBSET" in
    text|image|audio) subsets=("$SUBSET") ;;
    all) subsets=(text image audio) ;;
    *) echo "SUBSET must be text, image, audio or all"; exit 1 ;;
  esac
  for m in "${subsets[@]}"; do
    $PY experiments/phase14.py --data_path "$D" --save_dir "$SAVE" --only $(names "$m") "$@" || return 1
  done
}

busy () {
  pgrep -af "train.py|eval.py|d49_eval.py|d4[0-9]_eval.py" | grep -v "pgrep" || true
}

{
  echo "=== $(date -Is) commit $(git rev-parse --short HEAD) table2 $MODE subset=$SUBSET"
  case "$MODE" in
    list) plan --list ;;
    smoke)
      $PY -m pytest tests/test_phase14.py tests/test_summarize.py tests/test_modalities.py -q -p no:cacheprovider || exit 1
      # the D-47 assets must be in place: verified images, espeak-ng, Whisper, CLIP
      $PY experiments/d47_build.py --device cpu || exit 1
      for spec in "text 3 5" "image 2 1" "audio 2 1"; do
        set -- $spec
        $PY train.py --dataset s3dis --data_path "$D" --cvfold 0 --n_way "$2" --k_shot "$3" --modality "$1" \
          --save_dir log_table2_smoke --dry_run true || exit 1
      done
      echo "smoke passed: queue tests, D-47 rows, dry-run training of text N3K5, image N2K1, audio N2K1" ;;
    full)
      if [ -n "$(busy)" ] && [ "${SHARE_GPU:-0}" != "1" ]; then
        echo "another train/eval is running; not starting (SHARE_GPU=1 to share the GPU):"; busy; exit 1
      fi
      plan
      $PY experiments/summarize.py --save_dir "$SAVE" --protocol fixed100 > "$OUT/SUMMARY_fixed100.md"
      echo "=== $(date -Is) ALL RUNS FINISHED (table in $OUT/SUMMARY_fixed100.md)" ;;
  esac
} 2>&1 | tee -a "$OUT/table2_${MODE}.log"
