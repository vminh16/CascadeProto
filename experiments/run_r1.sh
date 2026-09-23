#!/bin/bash
# Phase 16 R1: which difference to VIP-Seg's head costs the 15 points?
#
# One variable at a time, T = 1, no LMA, 9,600 episodes = 2,400 steps at batch 4 (past the divergence
# point of CHANGELOG 15k), three seeds each, about 32 min per run.
#
#   (a) r1_eppm         the printed stage, one A per class slot from whole support blocks [D-01]
#   (b) r1_pooled       (a) with Eq.13's single S' [D-23]                      <- the leading hypothesis
#   (c) r1_eppms        the stripped stage with the shared S' and L2 prototypes [D-24]
#   (c') r1_eppms_slots (c) with the class slots of D-01, to separate stripping from pooling
#   (d) r1_vippem       one VIP-Seg PEM in our cascade [D-25]                  <- the reference head
#   r1_vip4             VIP-Seg's four alternating modules in our cascade      <- the 72-level reference
#   r1_baseline_l2      no stage at all, L2-normalised prototypes              <- the level to beat
#
# PROTOCOL. Trained and validated on **S1** (--cvfold 1): design choices must not be screened on the
# S0 test classes, which stay held out until the design is frozen [DECISION D-22]. Validation uses the
# `valid` draw of the fold's test classes, as every run in this repository does [DECISION D-15].
#
# DECISION RULES, fixed before the run (mean over seeds, sd from the three seeds):
#   R1.1  (b) - (a) >= +3 with t > 3           -> the per-slot correlation of D-01 is a main cause;
#                                                 revise D-01 and make `pooled` the default.
#   R1.2  |(c) - (d)| <= 2                     -> EPPM-S reaches the reference head; keep route A.
#   R1.3  (c) - (a) >= +3 while (b) ~ (a)      -> the cause is in the additions of Eq.19-21; ablate
#                                                 them one at a time before anything else.
#   R1.4  (d) - (a) >= +3 and (c) - (a) < +3   -> take route B: build on VIP-Seg's head [D-25].
#   R1.5  every variant within 2 points of r1_baseline_l2 -> no stage design in this family helps at
#                                                 this budget; stop R1 and re-open the oracle probe.
#   Differences below 2 points are noise at this budget (CHANGELOG 15e), hence three seeds.
#
# Usage on the VM:  bash experiments/run_r1.sh [additional variants...]
set -u
cd "$(dirname "$0")/.." || exit 1
D=datasets/S3DIS/blocks_bs1_s1
OUT=results/phase16_r1
mkdir -p "$OUT"
LOG="$OUT/r1.log"
PY=.venv/bin/python
SEEDS=${SEEDS:-"0 1 2"}
EPISODES=${EPISODES:-9600}
FOLD=${FOLD:-1}
VARIANTS=${*:-"r1_baseline_l2 r1_eppm r1_pooled r1_eppms r1_vippem"}

{
  echo "=== $(date -Is) commit $(git rev-parse --short HEAD)"
  echo "=== fold S$FOLD | episodes $EPISODES | seeds $SEEDS | variants $VARIANTS"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
  for v in $VARIANTS; do
    echo "=== $(date -Is) $v"
    $PY experiments/diag_short.py --data_path "$D" --variants "$v" --cvfold "$FOLD" \
      --episodes "$EPISODES" --seeds $SEEDS
    echo "=== $v exit=$?"
  done
  echo "=== ALL RUNS FINISHED $(date -Is)"
} 2>&1 | tee -a "$LOG"

echo "--- summary (variant: seeds -> mean)"
$PY experiments/summarize_r1.py "$LOG"
