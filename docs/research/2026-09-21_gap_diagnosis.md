# Diagnosis of the absolute gap: which conclusions hold, which are wrong, which are weak

Status: 2026-09-22 (closed). A second pass over `2026-09-21_reproduction_report.md`, CHANGELOG 14e–15v, the
code on the baseline's path and the external sources in `2026-09-21_external_sources_on_gap.md`.
No GPU run was made for this note; every new claim is either read from the repository or marked
as a prediction to be tested.

## 1. Verdict

* **The central conclusion holds, and now has independent support.** Our baseline (49.08) is where
  the published literature puts a plain prototype baseline in this protocol: ProtoNet 48.39, Seg-NN
  49.45, and three ablations by CascadeProto's own first author — TaylorSeg without APP 49.42,
  DyPolySeg without PCM 49.17 / 52.21 (external note §1, §4). No published method reaches 80 on S3DIS
  S0 2-way 1-shot. The paper's 82.72 baseline sits about 30 points above **its own author's** earlier
  measurement of the same construct.
* **One factual error in the report** (§2 below): the paper column of rows 2–4 is not Table 4.
* **Several inferences are weaker than written** (§3). None of them changes the verdict.
* **The cause of the paper's level is not established.** §4 ranks the remaining hypotheses and gives a
  zero-training probe that can falsify the leading one.

## 2. Error: the paper column of Table 4 was reconstructed, not transcribed

Report §2, §3.1 and CHANGELOG 15n/15q give the paper's S0 as 83.93 / 85.35 / 87.41 for + LMA /
+ Entropy Gate / + Cascade. Table 4 prints **83.98 / 85.34 / 87.89** (audit §11 and
`experiments/summarize.py:22-23` both carry the printed values). The report's values are
82.72 plus the cumulative **Avg** increments of §4.3 (+1.21, +1.42, +2.06): S0 and Avg are mixed.
The increment table has the same defect: it compares our S0 increments with the paper's Avg
increments, and those sum to +5.25, not the +5.81 listed as the S0 total.

Corrected, S0 against S0:

| step | ours | paper S0 | paper Avg (as quoted before) |
| :--- | ---: | ---: | ---: |
| Baseline → + LMA | +0.57 | +1.26 | +1.21 |
| + LMA → + Entropy Gate (T = 1) | +7.07 | +1.36 | +1.42 |
| + Entropy Gate → + Cascade (T = 4) | −0.17 | **+2.55** | +2.06 |
| + Cascade → + ADRM | +0.60 | +0.64 | +0.56 |
| Baseline → full | +8.07 | +5.81 | +5.25 |

The verdicts survive — ADRM still matches (+0.60 against +0.64), and the unreproduced cascade claim
becomes larger (+2.55).

## 3. Inferences weaker than written

1. **"The pipeline is sound" rests on 600 steps.** VIP-Seg through our loop (0.6948) ran in the diag
   harness at 2,400 episodes = 600 steps at batch 4. The loop has never trained VIP-Seg over the
   full schedule, so a schedule-level defect (D-12: 6,000 steps, five halvings) would not show up. The
   literature now makes this unlikely to be large (DyPolySeg's backbone-only rows land at 49–52 on
   VIP-Seg's schedule), but the D-12 probe (`experiments/run_b1.sh`) is still the direct test.
2. **"About 3.4 of the first stage's +7.07 is normalisation" was never measured.** It transfers the
   `l2norm_point_proto` gain (one seed, best-on-valid over 5 checkpoints; best 52.44 but last 50.19)
   onto Eq.21's LayerNorm by analogy. At the measured seed spread (sd 0.010–0.031) the +3.36 itself
   has an uncertainty of several points. Keep it as a hypothesis.
3. **Which row is "the paper's baseline"?** `results/phase15_full/SUMMARY.md` and 15j say it
   corresponds to Baseline + L2 (increment +4.71), while the report compares against Baseline (49.08,
   +8.07). Pick one; D-17 supports the literal Baseline.
4. **D-20 was rejected at zero training.** VIP-Seg's features were trained to be read through
   LayerNorm'd prototypes, not through a raw dot product with a mean prototype, so a zero-shot probe
   is biased low. The rejection is still right, now for a better reason: the author's DyPolySeg
   ablation measures the same construct on the same recipe at 49–52.
5. **"Independent validation of our training"** (report §3.2) compares an encoder trained for MAP and
   read by MAP with an encoder trained for PEM/PDM and read zero-shot by MAP. That is not
   like-for-like; the external ablations are the better validation.
6. **"Area 5 would only make it harder"** holds only if training stays episodic on the training
   classes. The other reading of §4.1 — ordinary supervised training on Areas 1–4, 6 with every
   class, testing episodes on Area 5 — shows the test classes during training. That is the D-21
   family, not a low-prior case.
7. **Protocol.** The paper states 600 random episodes; we report `fixed100`. `results/phase14_p1/
   summary_random600.md` exists; the difference is a few points at most, but the report should state
   it.

## 4. What can produce the paper's level: ranked hypotheses

The gap is present on the baseline, which contains none of the added modules, and is roughly
uniform across Table 4. So the cause must sit on the baseline's path or outside the model.

| # | hypothesis | prediction that would confirm it | status |
| :--- | :--- | :--- | :--- |
| H1 | **Test classes seen in training**: folds swapped between training and evaluation, or training on all classes (the supervised reading of "Area 5") | an S0-trained checkpoint scored on fold-1 classes (its own training classes) reaches ≈ 80; D-21 reaches ≈ 80 | untested; D-21 scripted (`experiments/run_leak.sh`), not run |
| H2 | **Query labels reach the prediction** in the authors' code (prototypes, GMMN or routing fed query ground truth) | oracle prototypes pooled from the query's own mask score ≈ 85–90 on our encoder | untested; not verifiable without their code |
| H3 | Our baseline is too weak | a fix on the baseline's path moves it above 60 | **refuted by the literature** (five independent baselines at 48–55) |
| H4 | D-12 schedule (batch 4) | batch 1 × 24,000 steps moves the baseline by > 10 | unlikely by the same evidence; `run_b1.sh` tests it |
| H5 | A different mIoU | an alternative definition reaches 80 | **refuted** (report §3.1, max 64.85) |

**A fingerprint that favours H1.** In every source where both folds are reported, S1 scores above
S0 (VIP-Seg 72.20 / 76.09, DyPolySeg 72.02 / 73.82, TaylorSeg-PN without APP 49.42 / 52.67, DyPolySeg
without PCM 52.21 / 54.35): fold 1 holds the large, easy classes (floor, wall, door, table). CascadeProto
reverses the order on every Table 4 row (baseline 82.72 / 79.83, full 88.53 / 84.53). A swapped
evaluation — S0's number measured on fold-1 classes — would give exactly that reversal. This is a
pattern, not a proof: it also fits noise or a different method.

**The cheapest decisive probe (no training).** Score the existing S0 checkpoints on fold 1:

```bash
python eval.py --dataset s3dis --data_path datasets/S3DIS/blocks_bs1_s1 --cvfold 1 --n_way 2 --k_shot 1 \
  --checkpoint log_phase14/s3dis_S0_N2_K1_point_T0/best.pt --eval_protocol fixed100 --extra_metrics true
```

(and the same for `text_T4`). The model is class-agnostic, so this is valid, and the S1 cache is
built on first use. Reading, fixed before the run: **≥ 75 supports H1** (the paper's level is
seen-class performance); **≤ 65 rejects the fold-swap form of H1**, which leaves D-21 and H2.
Then the H2 ceiling (oracle query-mask prototypes) bounds what a label leak could produce.

## 4b. Update after 15w/15x (D-12 and D-21 measured)

**D-12 is cleanly closed.** Batch 1, 24,000 steps, halving every 15 epochs (the log confirms 480
steps/epoch and halvings at epochs 15/30/45): 0.4901 best against 0.4908. The validation curve sits at
0.42–0.50 throughout, matching the old run. This also answers §3.1: the schedule is not a hidden
defect.

**D-21 is not an upper bound, contrary to report §3.5 and CHANGELOG 15x.** The logs confirm the
sampler saw all 12 classes (`train classes [0..11]`), but:

1. **Exposure.** 9,600 episodes × 2 ways over 12 classes = **1,600** way slots per test class. The
   normal S0 run gives each of its own training classes 24,000 × 2 / 6 = **8,000**. A paper that
   trained on the test classes by swapping folds would give them the 8,000, five times what D-21 gave.
2. **Not converged.** Validation rose 0.586 → 0.637 (baseline) and 0.638 → 0.704 (full) between
   epochs 10 and 20, and `best` = `last` = epoch 20. The report says so, then calls the number an upper
   bound anyway.
3. **Diluted.** Half of each D-21 episode's way slots go to the six split classes, so D-21 measures a
   partial leak, not the swapped protocol.

So "leakage explains 12–15 points, not the level" holds only for this partial form. The strong form is
still untested — and it costs no training: `experiments/run_seen.sh` scores the existing S0
checkpoints on fold 1, whose test classes are exactly S0's training classes, seen for the full 50
epochs. Reading rule, fixed before the run: baseline ≥ 0.75 supports the swapped-fold reading, ≤ 0.65
rejects it. VIP-Seg's S0 checkpoint is scored the same way as a reference.

**Result of the seen-class probe (2026-09-22, VM, fixed100 fold 1, 1,500 episodes, test classes
door / floor / sofa / table / wall / window, S0-trained `best.pt` = epoch 20):**

| checkpoint | fold-1 mIoU (classes seen in training) | best alternative (per episode, with bg) | paper S0 |
| :--- | ---: | ---: | ---: |
| baseline | **71.40** | 75.79 | 82.72 |
| full model | **75.99** | 79.14 | 88.53 |
| VIP-Seg released S0 | **79.13** | 80.26 | (72.20 cited) |

* **By the rule fixed before the run the baseline is inconclusive** (71.40 lies between 65 and 75).
  The swapped-fold reading moves the baseline from 49.08 to 71.40, two thirds of the way to the paper,
  but leaves 11.3 points (baseline) and 12.5 points (full) unexplained.
* **The strongest single fact:** VIP-Seg's full model, scored on the very classes it was trained on,
  reaches 79.13, **below the paper's plain baseline (82.72)**. So no class-leakage variant of this
  loader and metric lifts a masked-average-pooling baseline to 82.72 unless it also beats VIP-Seg on
  VIP-Seg's own training classes.
* Seen-class gain of the modules: +4.59 (paper +5.81), one seed each, so within noise.
* **Caveat that makes 71.40 a floor, not a ceiling:** `best.pt` was selected on the S0 test classes
  (unseen generalisation) at epoch 20; training loss kept falling to epoch 50, so `last.pt` fits the
  fold-1 classes more and may score higher on them. Scoring `last.pt` the same way is the remaining
  cheap check of this hypothesis.

**`last.pt` (epoch 50) on fold 1, run 2026-09-22:** baseline **77.32** (per episode with bg 79.89,
point accuracy 89.32). Full model: output truncated, pending. The criterion stated before this run
(baseline `last.pt` ≥ 75 makes the swapped-fold reading the lead hypothesis) is met.

* On the unseen S0 classes `last` and `best` are equal (49.07 / 49.08); on the seen fold-1 classes
  `last` is 5.9 points above `best`. Extra training helps only on the training classes: base-class
  fitting, the same pattern as the falling training loss with flat validation (14e-P1).
* Remaining gap to the paper's S0 baseline: **5.4 points** (82.72 − 77.32), down from 33.6. What is
  left is of the size of one seed's spread plus protocol details: the paper's 600 random episodes
  against our fixed100, and checkpoint choice. A paper that selected its checkpoint on the seen
  classes would pick the best seen-class epoch, not necessarily epoch 50.
* VIP-Seg's released checkpoint (selected on unseen classes) scores 79.13 on fold 1 against its
  cited 72.20 on S0: the same kind of shift, which fits a table mixing VIP-Seg's unseen number with
  the paper's own seen-class numbers.
* **This is consistency, not proof.** The confirming test, with its prediction stated now: train the
  baseline on S1 (its training classes are fold 0), then score fold 0 (`--cvfold 0`). Under the swap
  hypothesis that is what the paper prints as "S1"; prediction: last.pt a few points **below** our fold-1
  number (paper: S1 79.83 is 2.9 below S0 82.72), i.e. roughly 72–77, since fold 0 holds the harder
  classes. That would reproduce the paper's reversed S0 > S1 order. A result ≤ 65 would weaken the
  hypothesis; one above our fold-1 number would contradict the ordering argument.

**Confirming run, S1 baseline (2026-09-22, 50 epochs, seed 0).** Training classes
[0, 3, 4, 8, 10, 11] = fold 0. Validation on its own unseen fold-1 classes: 0.5180 (epoch 40),
0.5224 (epoch 50, best = last). Its `last.pt` scored on fold 0 (seen classes, fixed100, 1,500
episodes): **71.58** (per episode with bg 76.28, point accuracy 86.25).

| classes | unseen (standard protocol) | seen in training | gain from seeing |
| :--- | ---: | ---: | ---: |
| fold 0 (beam, board, bookcase, ceiling, chair, column) | 49.08 (S0 model, test) | **71.58** (S1 model, last) | +22.5 |
| fold 1 (door, floor, sofa, table, wall, window) | 51.91 (S1 model, test) | **77.32** (S0 model, last) | +25.4 |
| difference fold 1 − fold 0 | +2.8 | +5.7 | |

Test-set score of the S1 model on its unseen classes: 51.91 (best = last = epoch 50; validation
52.24). Full model `last.pt` (S0-trained) on fold 1: **81.28** against the paper's S0 full 88.53
(gap 7.25); module gain on seen classes +3.96 (paper +5.81). All results committed in `results/seen/`
(6b48636).

* **Prediction A (stated before the run: 72–77, below 77.32):** 71.58 is below 77.32, so the paper's
  reversed S0 > S1 order is reproduced by the swapped reading; it falls 0.4 below the stated band,
  and far above the ≤ 65 that would have weakened the hypothesis. Mostly confirmed.
* **Prediction B (unseen fold-1 baseline 52–55, as in the author's DyPolySeg/TaylorSeg ablations):**
  51.91 on the test set. Confirmed.
* **Decomposition.** "Easy classes" is worth about 3 points (51.9 vs 49.1 unseen); having seen the
  classes is worth 22–25 points on either fold. The jump is leakage, not class difficulty.
* **Against the paper's baseline row** (S0 82.72 / S1 79.83 / Avg 81.28): swapped reading 77.32 /
  71.58 / 74.45, gaps 5.4 / 8.2 / 6.8. Standard reading 49.08 / 51.91 / 50.50, gap 30.8 on Avg. The
  paper's S0 − S1 difference is 2.9; ours under the swap is 5.7. One seed, fixed100 against the
  paper's random600, checkpoint = last: the residual 5–8 points are unexplained but of the size of
  those differences combined.

**"The module gain matches the paper (+5.70 against +5.81)" is noise-level evidence.** One seed per
row, and the measured seed sd is 1–3 points per run, so the difference of two runs has an sd of about
1.5–4 points. It fits the protocol reading but cannot support it.

**§2's correction** was applied to the report on 2026-09-22 (CHANGELOG 15y).

## 5. What this note does not settle

* Whether the authors' numbers come from any of H1/H2. Without their code (the official repository
  holds only a README) only consistency with a hypothesis can be shown.
* Why VIP-Seg's poster (73.50) and released logs (72.20) differ (external note §1).
* VIP-Seg's own ablation table (OpenReview PDF not retrievable).
