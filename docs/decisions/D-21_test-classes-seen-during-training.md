### D-21 — Test classes seen during training · `PROPOSED`, diagnostic only

* **Problem.** The absolute gap to the paper is almost uniform across Table 4 (33.6, 34.3, 28.6,
  30.9, 31.4 points), the signature of a data or protocol difference rather than a model difference.
  §4.1 says "using Areas 1, 2, 3, 4, 6 for training and Area 5 for testing under two category splits
  S0 and S1" [PAPER §4.1]. If the split was by area and the category split did not take effect, the
  S0 test classes were seen during training and "few-shot" becomes segmentation of learnt classes.
* **What the switch does.** `train.py --train_classes all` lets training episodes sample the fold's
  test classes as well (all 12 classes, clutter excluded as in the loader). The inherited loader is
  unchanged; only the instance's `classes` array, the one thing its sampler reads, is widened
  [VIPSEG dataloaders/loader.py:163]. Run directories are tagged `_leak`. Test episodes are the usual
  fixed100 cache, which draws from every area, so the diagnostic leaks classes *and* possibly blocks:
  it is an upper bound of the area-split scenario.
* **Prediction.** If this is what produced the paper's numbers, the baseline lands at 75-85 and the
  increments between rows stay small.
* **Ablation flag.** `train_classes = {split (default), all}`. `all` is a protocol violation, never a
  result configuration.
* **Result (2026-09-21).** 20 epochs each, fixed100: baseline 0.6354 (+14.46 over the split run),
  full 0.6924 (+12.09); full − baseline +5.70 against the paper's +5.81. Leakage explains part of the
  level but leaves the leaked full model below the paper's baseline (82.72), so it is not the whole
  explanation (report §3.5, CHANGELOG 15x).
* **Correction and full form (2026-09-22).** The 20-epoch run is a partial leak, not an upper bound: each
  test class fills 1,600 way slots against 8,000 for a normally trained class. The full form needs no
  training: an S0-trained checkpoint scored with `--cvfold 1` (and an S1-trained one with `--cvfold 0`) is
  scored on its own training classes. Baseline `last.pt`: 77.32 (S0 model on fold 1) and 71.58 (S1 model
  on fold 0) against the paper's 82.72 / 79.83; unseen 49.08 / 51.91. Seen-class scoring matches the
  paper to 5–8 points and reproduces its S0 > S1 order (report §3.6, CHANGELOG 15y). Diagnostic only.
