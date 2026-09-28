### D-20 — Initialising from VIP-Seg's trained encoder · `PROPOSED`, diagnostic only

* **Problem.** Table 4's baseline, "a plain VIP-Seg backbone with masked average pooling and
  single-step prototype matching" [PAPER §4.3], scores 82.72 on S0, above VIP-Seg's own full model
  at 72.20 [PAPER Tab.6]. Trained from scratch, ours scores 49.08. No mIoU definition closes the gap
  (15q). One reading of "the shared encoder is VIP-Seg" [PAPER §4.1] that could: the encoder was taken
  from VIP-Seg's trained checkpoint, not trained from scratch.
* **Conflict.** That reading contradicts the paper's own "pre-training-free framework … without any
  pre-trained weights" [PAPER §1] and guardrail #1 of this repository. It is therefore a diagnostic,
  never a result configuration.
* **What the switch does.** `train.py --init_from_vipseg <checkpoint>` copies VIP-Seg's trained
  encoder, feature head and fixed projections into `model.features` (strict), before any resume, and
  tags the run directory `_vipinit`. `experiments/vipseg_init_probe.py` scores the baseline on those
  features with no training at all. VIP-Seg's S0 checkpoint saw only the S0 training classes, so the
  S0 test classes are not leaked.
* **Ablation flag.** `--init_from_vipseg` (default: none). Any number obtained with it is outside the
  paper and outside this repository's guardrails and must be reported as such.
