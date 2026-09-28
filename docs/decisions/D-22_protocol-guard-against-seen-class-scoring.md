### D-22 — Protocol guard against seen-class scoring · `PROPOSED`

* **Problem.** Report §3.6 showed that scoring a checkpoint on classes it was trained on lifts the
  baseline by 22–25 points, and that this reading reproduces the paper's level. Nothing in `eval.py`
  prevented it: `--cvfold` was read from the command line, independently of the fold the checkpoint
  was trained on, and S0's training classes are exactly fold 1's test classes
  [VIPSEG dataloaders/s3dis.py:20-31]. Phase 16 compares new designs whose gains are expected to be a
  few points, so one silent fold swap would dominate any result.
* **Decision.** `eval.py` scores a checkpoint only on the test classes of the fold it was trained on.
  The training fold is read from the checkpoint (`args.cvfold`, which `train.py` stores in `best.pt`
  and `last.pt`) or, for checkpoints that record none (VIP-Seg's release saves only the model,
  iteration and IoU [VIPSEG runs/training.py:97-100]), stated with `--checkpoint_cvfold`; a stated
  fold that contradicts the recorded one raises. A different fold, or a checkpoint trained with
  `--train_classes all` [DECISION D-21], raises before any episode is built.
* **Diagnostic escape.** `--allow_seen_classes true` permits such a run; the log and the result JSON
  then carry `protocol_check = "SEEN-CLASS DIAGNOSTIC, not a few-shot result: …"`. Clean runs carry
  `"clean"`. `experiments/run_seen.sh` is the only script that sets the flag.
* **Reporting rules for phase 16** (recorded here so that they bind every later run):
  1. Headline numbers are `last.pt`. The inherited validation episodes are drawn from the test classes
     [VIPSEG runs/training.py:52-66]; `best.pt` is selected on them and is reported only next to
     `last.pt` [DECISION D-15].
  2. Design choices are screened on fold S1 (the `valid` episode draw) and the S0 test episodes are
     not looked at until the design is frozen, so S0 is a held-out fold for every phase-16 decision.
     S1 results are reported with that caveat.
  3. New hyper-parameters are never tuned on the classes being scored.
* **Amendment of rule 1 (maintainer, 2026-09-24).** VIP-Seg's published 76.09 / 72.20 are the best of 12
  validations on episodes of the test classes, re-tested once: its released S1 log reads 72.84 at the
  last update and 75.63 at the selected one, its S0 log 68.97 last and 72.94 selected, mean of the 12
  validations 69.29 [VIPSEG log_s3dis_VIPSeg/log_S{1,0}_N2_K1_*/log_vipseg.txt]
  (`docs/research/2026-09-24_r2_distill_analysis.md` §3). Rule 1 compared our `last.pt` with numbers
  selected that way. From now on every result reports **both**:
  * `best.pt` under VIP-Seg's own disclosed selection rule, the protocol of the published baselines:
    about 12 validations over training (`--valid_every` chosen so), each on the 1,500 `valid` episodes of
    the fold's test classes [DECISION D-15], the best one kept. This is the number compared with the
    published table, labelled "best-of-validation, VIP-Seg's protocol".
  * `last.pt`, the number free of selection on test classes, labelled that way.
  A claim of beating a baseline states which of the two it rests on; a claim on `best.pt` alone is
  marked as resting on selection. Rules 2 and 3 are unchanged.
* **Affects.** `eval.py`, `experiments/run_seen.sh`, `experiments/rescore_alt_metrics.sh`, 04 §6.1,
  05 §3.12 (PROT-1…6), README §5.
