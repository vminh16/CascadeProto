# What Does Few-Shot 3D Point-Cloud Segmentation Measure? Seen-Class Scoring, a Query-Position Shortcut and a Density Cue in the Standard S3DIS Protocol

> **Internal draft, 2026-09-30. Not for submission.** Written from this repository's result files; every number
> carries its source in Appendix A. Author list, related-work citations and figures are placeholders. Claims about
> other groups' work are stated as "consistent with", never as findings about their intent. Before any public
> version: contact the authors of VIP-Seg and CascadeProto, run the checks listed in §9, and verify every number
> marked [snippet] against the original paper.

---

## Abstract

Few-shot 3D point-cloud semantic segmentation is benchmarked almost entirely on the S3DIS and ScanNet episode protocol
introduced by attMPTI. Recent methods report 2-way 1-shot mIoU above 72 on S3DIS without any pre-training. One
reports 88.5.

We re-implement a recent method, CascadeProto, and audit the protocol with the released code and checkpoint of its
base method, VIP-Seg. We find three effects that inflate the headline number without measuring few-shot recognition:

1. **Seen-class scoring.** A faithful re-implementation of CascadeProto reaches 57.15 on novel classes. It reaches
   81.28 when the same checkpoint is scored on classes seen in training, which is consistent with the published
   88.53 (and its baseline 82.72).
2. **A query-position shortcut.** The episode loader always samples query block *b* for way *b + 1*, and VIP-Seg's
   cross-term carries block order into the prediction. Swapping the two query blocks drops VIP-Seg's released
   checkpoint from 75.36 to 0.87 mIoU. The shortcut is worth +19.94 [+18.40, +21.52] mIoU on fold S1. An
   order-equivariant head scores 54.84, which is no better than plain prototype matching (55.74).
3. **A sampling-density cue.** The loader draws the target class twice, so the target is denser than everything else
   in its block. The same object re-sampled densely is found with recall 0.833 against 0.245 when sparse (causal
   fraction φ = 1.001). Under uniform sampling, the order-equivariant model falls from 54.84 to 28.10.

We also measure a checkpoint-selection effect of +1.85 (best-of-validation on test-class episodes against the final
checkpoint), and a representation ceiling. Among the prototype-based heads we test, none closes more than 2.8 points
of the gap. The prototype-matching score extrapolates to about 64 with unlimited shots, and the features collapse to
a participation ratio of 5.56 ≈ C − 1. We release a shortcut-audit evaluation (swapped order, uniform sampling,
unseen-class guard) and recommend it as a standard robustness column.

---

## 1. Introduction

Few-shot point-cloud segmentation asks a model to segment a class it has never been trained on, given one or a few
annotated support examples. The attMPTI protocol (CVPR'21) defines the standard setting:
* S3DIS rooms cut into 1 m × 1 m blocks of 2,048 points;
* the 12 classes split into two folds of 6;
* an encoder trained episodically on one fold and tested on the other;
* mIoU accumulated over 100 fixed episodes per class combination.

Methods in this protocol have risen from the mid-50s (attMPTI, 2021) to 72–76 (VIP-Seg, NeurIPS'25; EDS-Net,
AAAI'26). Several of them explicitly require no pre-training. CascadeProto adds entropy-aware prototype purification,
a cascade and cross-modal prototypes on top of VIP-Seg, and reports 88.53 (S0) and 84.53 (S1) for 2-way 1-shot.

COSeg (CVPR'24) already showed that the protocol leaks foreground through its sampler. Retraining without the leak
dropped earlier methods by 23–36 points. The pre-training-free line of work that followed kept the original
protocol.

**We set out to reproduce CascadeProto and ended up auditing the protocol.** Our contributions:

1. **A verified pipeline.** We use the inherited loader, sampler and metric byte-identically. VIP-Seg's released S0
   checkpoint scores 71.97 in our pipeline against its logged 72.20 (§3).
2. **Seen-class scoring.** It reconciles a faithful CascadeProto with its published table to within 5–7 points,
   while the same weights score 30 points lower on unseen classes (§4).
3. **A query-position shortcut in VIP-Seg's head,** traced to one reshape and to the loader's fixed query order,
   worth about 20 mIoU (§5). We found no prior report of it.
4. **A causal measurement of the density cue** inside a prototype-matching head (§6).
5. **A decomposition of what remains** after both shortcuts are removed: the head adds nothing over prototype
   matching, and the representation is the binding constraint (§7).
6. **An evaluation suite** that exposes all three effects at inference cost (§8).

---

## 2. Setup

**Data and episodes.** S3DIS, blocks `blocks_bs1_s1` (1 m blocks, stride 1, 2,048 points, channels `xyzrgbXYZ`).
The two class folds are inherited: S0 = {beam, board, bookcase, ceiling, chair, column}, S1 = {door, floor, sofa,
table, wall, window}. The loader, sampler and preprocessing files are byte-identical to VIP-Seg's pinned commit
`28aedc5`, checked by test ENV-3.

**Metric.** VIP-Seg's own `evaluate_metric` (restored byte-identical), which works as follows:
* TP, predicted count and ground-truth count are accumulated per test class over all episodes;
* background is excluded;
* the score is reported as mIoU in %.

**Draws.**
* **fixed100:** the protocol's 1,500 cached test episodes.
* **random600:** three independent draws of 600 episodes each (seeds 0, 1, 2).
* **leak-free:** 1,500 episodes in which support and query are sampled uniformly (the loader's own
  `random_sample=True` path, 2,048 points). This is COSeg's "w/o FG" form, not COSeg's full protocol.
* **swap:** fixed100 with the two query blocks of each episode exchanged and labels permuted accordingly.

**Protocol guard.** Our `eval.py` refuses to score a checkpoint on classes of its own training fold unless an
explicit diagnostic flag is set. Every "seen" number in this paper was produced with that flag and is labelled a
diagnostic.

**Models.**

| id | description | training |
| :--- | :--- | :--- |
| CP-base, CP-full | our CascadeProto re-implementation: baseline row and full model (LMA + 4 EPPM stages + ADRM) | S0, 50 epochs × 480 episodes, batch 4 (the paper's schedule) |
| VIP-Seg | released 2-way 1-shot checkpoints, S0 and S1 | authors' |
| E1 | VIP-Seg's head (PEM/PDM) in our pipeline | S1, VIP-Seg's update count (24,000 updates, batch 1) |
| VR | E1 trained with random query order | as E1 |
| CR | the same head with an order-equivariant cross-term (`clean`), random query order | as E1 |

**Statistics.** Differences are paired over identical episodes, with 95 % episode-bootstrap CIs. One training seed
per trained arm; §10 discusses what this does and does not support.

---

## 3. The pipeline reproduces VIP-Seg

| check | ours | reference |
| :--- | ---: | ---: |
| VIP-Seg released S0 checkpoint, our data, metric and fixed100 | 71.97 | 72.20 (log / paper Tab. 6) |
| VIP-Seg's head trained in our loop, S1, best-of-validation (E1 `best`) | 75.05 | 75.36 (released S1 checkpoint, same episodes) |
| E1 `best` − released, fixed100 / random600 × 3 | −0.31 [−0.63, +0.01] / −0.15, −0.32, +0.02 | — |

**Conclusion.** Data, sampler, loss, optimiser and metric agree with VIP-Seg. Any gap to a published number below
does not come from the pipeline.

---

## 4. Finding 1: seen-class scoring reconciles a faithful CascadeProto with its table

We implement every equation of CascadeProto and train each row of its ablation table (Table 4) on S0 with the
paper's schedule. The authors' code is not released. An equation-by-equation audit is in
`docs/research/2026-09-20_paper_vs_code_audit.md`.

**Table 1. CascadeProto, S3DIS S0, 2-way 1-shot, fixed100.** "Unseen" is the standard protocol. "Seen" scores the
same checkpoint on fold 1's classes, which were in its training set (diagnostic only).

| row | ours, unseen (`best` / `last`) | ours, seen (`last`) | paper |
| :--- | ---: | ---: | ---: |
| Baseline | 49.08 / 49.07 | **77.32** | 82.72 |
| + LMA | 49.65 / 48.32 | — | 83.98 |
| + Entropy Gate (T = 1) | 56.72 / 56.72 | — | 85.34 |
| + Cascade (T = 4) | 56.55 / 56.03 | — | 87.89 |
| + ADRM (full) | 57.15 / 56.70 | **81.28** | 88.53 |
| VIP-Seg released S0 | 71.97 | 79.13 | 72.20 (unseen) |

**Observations.**
1. **The unseen level is 30+ points below the paper.** No implementation choice on the baseline path accounts for
   this. The baseline row is essentially VIP-Seg's encoder with prototype matching, and VIP-Seg's own released model
   scores 71.97 in the same pipeline.
2. **Scoring the same weights on seen classes lifts them by 28.2 (baseline) and 24.6 (full).** This lands 5–7
   points below the printed numbers, with the paper's ordering preserved.
3. **Three further features of the published tables fit the same explanation:**
   * The paper's baseline (82.72) sits above VIP-Seg's own published number (72.20), although it contains no
     component VIP-Seg lacks.
   * Its fold ordering is S0 > S1, the reverse of every other method in the same table.
   * Its Table 4 row "T = 1" (85.34) and Table 5 row "T = 1" (85.21) describe the same configuration but disagree.
4. **The relative gain reproduces.** Baseline + L2 → full is +4.71 for us against +5.81 in the paper. Individual
   steps do not: the entropy gate gives +7.07 for us against +1.36 in the paper, and the cascade −0.17 against +2.55.

We do not claim the published numbers were produced by seen-class scoring. We show that seen-class scoring is the
only mechanism we found that brings a faithful implementation within 7 points of them. A partial class leak
(training on all 12 classes for 20 epochs) gives 63.54 / 69.24.

---

## 5. Finding 2: a query-position shortcut in VIP-Seg's head

**Mechanism.** In an N-way episode, the loader samples query block *b* for local class *b + 1*
(`dataloaders/loader.py:181-223`). Block order therefore encodes the class.

VIP-Seg's cross-correlation term reshapes a `[2 queries × 36 tokens]` tensor with `reshape(72, −1)`. After this, the
attention of query *b′* and support slot *w′* is:

softmax_r ( Σ_{r<72} Q′_{⌊r/36⌋, 2(r mod 36)+b′}ᵀ S′_{⌊r/24⌋, 3(r mod 24)+w′} / √128 )

This sums over both queries and every support slot. We re-derived the formula from the inherited module; it matches
to 4.7 × 10⁻¹⁴ (`results/c1/c2_crosscorr.txt`). Changing only the way-2 support moves slot 1's attention by up to
0.45, against a typical entry of 0.0078.

**Table 2. Swapping query order (S1, fixed100).**

| checkpoint | stored order | swapped | share of predictions relabelled by position |
| :--- | ---: | ---: | ---: |
| E1 `last` | 73.20 | 0.92 | 0.912 |
| VIP-Seg released S1 | 75.36 | 0.87 | 0.898 |
| same weights, cross-term made order-equivariant (no retraining) | 37.91 / 35.45 | 37.91 / 35.45 | 0.000 |

Replacing only the cross-term with an equivariant form removes the order dependence exactly. The score of the
unchanged weights falls to 36–38, below plain prototype matching on the same features (49–52). The trained heads
route most of their prediction through the positional path.

**Table 3. 2 × 2: head × training order (S1, one seed per arm; `last`).**

| | fixed100 | random600 s0 / s1 / s2 | leak-free |
| :--- | ---: | :--- | ---: |
| VF = E1 (scrambled head, fixed order) | 73.20 | 72.51 / 74.23 / 75.50 | 17.89 |
| VR (scrambled head, random order) | 53.26 | 50.44 / 56.08 / 53.42 | 24.04 |
| CR (equivariant head, random order) | **54.84** | 51.83 / 56.73 / 54.76 | **28.10** |
| prototype matching on CR's features (U) | 55.74 | 55.55 / 57.85 / 57.18 | 33.11 |

* **Shortcut worth:** VF − VR = **+19.94 [+18.40, +21.52]** on fixed100; +22.07 / +18.15 / +22.09 on random600.
* **Origin and carrier.** VR, trained with random order, keeps no positional dependence (relabel shift +0.004). The
  shortcut therefore needs both the reshape and the fixed training order; neither alone produces it.
* **Without the shortcut, the head adds nothing measurable over prototype matching:** CR 54.84 against 55.74 (U) and
  against 55.02 (the support-prototype rule).
* **Position is also anti-correlated with robustness to density.** On the leak-free draw E1 is the *worst* model
  (17.89).

**Scope.** We tested VIP-Seg's released S1 checkpoint and our retrained heads. Whether other methods sharing this
cross-term inherit the shortcut is untested; see §9.

---

## 6. Finding 3: the density cue, measured causally

**Mechanism.** For a block sampled for class *s* with raw share π_s, the sampler first draws ⌊π_s · 2048⌋ points of
*s*, then 2048 − that from the whole block. The expected share of *s* is π_s(2 − π_s), and every other class stays
at background density.

In a 2-way episode, the other way's class appears in a query block at background density (the "other" condition).
Supports are always in the "own" condition. Measured on 1,500 episodes per fold (`results/c1/`):

**Table 4. Sampling conditions (S1).**

| class | share of its query points in "other" | kNN-16 radius own / other [m] | duplicated points own / other |
| :--- | ---: | :--- | :--- |
| wall | 0.235 | 0.084 / 0.121 | 0.044 / 0.001 |
| floor | 0.192 | 0.072 / 0.130 | 0.060 / 0.001 |
| table | 0.068 | 0.078 / 0.133 | 0.050 / 0.002 |
| door | 0.042 | 0.076 / 0.112 | 0.057 / 0.001 |
| window | 0.032 | 0.084 / 0.117 | 0.042 / 0.002 |
| sofa | 0.007 | 0.080 / 0.102 | 0.049 / 0.000 |

For reference, the background radius is 0.118 m and the support-foreground radius 0.079 m. On S0, ceiling has the
same "other" share as wall (0.235).

**Intervention.** For 1,045 query blocks that contain the other way's class, we re-sample the *same scan*:
* V0 as the protocol does;
* V1 sampled for that class (dense);
* V2 uniformly.

We then measure the class's recall under U on CR's features (`results/phase16_p8/`).

**Table 5. Density intervention (S1, CR features, U).**

| | R_V0 (sparse, as sampled) | R_V1 (same object, dense) | R_V2 (uniform) | R_own (its own blocks) | φ = (R_V1 − R_V0)/(R_own − R_V0) |
| :--- | ---: | ---: | ---: | ---: | :--- |
| other-condition class | 0.245 | 0.833 | 0.540 | 0.833 | **1.001 [0.962, 1.034]** |
| own class of the block, V0 → V2 | 0.837 | — | 0.463 | — | — |

* **Making the object dense recovers all of its missing recall** (φ ≈ 1). Its feature direction moves from cos 0.55
  to 0.86 of the support prototype. Nothing about the object is missing except its sampling density.
* **The own class also relies on density.** Uniform sampling lowers its recall from 0.837 to 0.463.
* **At the benchmark level:** U falls from 55.74 (fixed100) to 33.11 (leak-free), so about **22.6 points** of the
  clean standard-protocol score rest on density. The within-block cosine oracle falls to about 61 on the same draw.
* The three classes with the largest "other" share (wall, floor on S1; ceiling on S0) are also the three worst
  classes of VIP-Seg's released checkpoints across both folds, beam aside: Spearman −0.57, p = 0.051.

**Training for density invariance trades one draw for the other.** Condition-balanced training (half the training
queries thinned to background density) raises other-condition recall from 0.18 to 0.46 and leak-free U by +2.11
[+1.43, +2.80]. It lowers fixed100 by −5.81 [−6.90, −4.74]. Two other density-directed changes (a density-invariant
encoder; base-class prototype alignment) show the same sign pattern. On the standard protocol, a model that stops
using density is penalised.

---

## 7. What remains once the shortcuts are removed

On the equivariant base CR (S1, fixed100), we measured:

**Table 6. Heads and inference rules on CR's features.**

| rule | fixed100 | leak-free |
| :--- | ---: | ---: |
| CR model (trained PEM/PDM head) | 54.84 | 28.34 |
| U, cosine to support prototypes | 55.74 | 33.11 |
| U + background rules ("both") | 57.63 | — |
| U + both + label propagation | **58.55** | 33.15 |
| cosine oracle (query's own class means; a bound, not a result) | 83.38 | ≈ 61 |

**The head is saturated.** We tested about 17 head and inference variants on this and earlier bases:
* test-time EM;
* calibration;
* distillation;
* text priors;
* attention necks;
* optimal transport;
* unmixing;
* a correlation head.

None exceeds +2.8 over U (the best is both + label propagation). Two trained variants lose 3–5 points.

**The gap does not close with more shots.** On 1,500 five-shot episodes, taking the first *k* shots gives
U = 56.32 / 60.22 / 61.63 / 62.74 at *k* = 1 / 2 / 3 / 5. The mIoU gap to the oracle fits 16.06 + 8.45/*k*
(R² = 1.000), so U∞ ≈ 64. The prototype error e(*k*) = 1 − cos(p̂_k, μ) fits a + c/*k* with a = 0.156 (instance
shift) and c = 0.047 (one-shot variance). Even a perfect class-level prototype, from any modality, would stop near
64 on this protocol with these features.

**The features are collapsed.** The participation ratio of CR's point features is 5.56 of 128, close to
C − 1 = 6 for 6 base classes plus background. Every S1 checkpoint we measured reads 3.7–7.3, including VIP-Seg's
released model (6.39).

Raising it with VICReg's variance and covariance terms (to 42.45 and 95.25) lowers U by 5.20 and 11.34. This
spreads the geometry without adding information about novel classes. We read this as a representation limit of
training from scratch on 6 classes, not a head limit.

**Checkpoint selection.** Choosing the best of 13 validations on test-class episodes (VIP-Seg's practice) is worth
**+1.85 [+1.43, +2.30]** for E1 on fixed100, and +1.27 to +1.83 on random600. VIP-Seg's own S1 log reads 72.84 at
its last validation against 75.63 at its best; the published 76.09 is best-of-validation.

---

## 8. A shortcut-audit evaluation

We recommend reporting the following next to the standard fixed100 number. All of them cost inference only.

| column | what it exposes | cost |
| :--- | :--- | :--- |
| **Swap**: fixed100 with query order exchanged | position shortcuts (§5) | one pass |
| **Leak-free**: support and query sampled uniformly | density / foreground leakage (§6) | one pass on a new draw |
| **Unseen-class guard**: refuse to score fold-*k* classes with a fold-*k* checkpoint | seen-class scoring (§4) | none |
| **`last` and `best`** | selection on test-class validation (§7) | none |
| **Prototype-matching baseline U** on the method's own features | whether the head adds anything | one pass |

The implementation is `eval.py` with its protocol guard, `experiments/c3_query_order.py`, and the leak-free scorer
in `experiments/p5_condition_probe.py`.

For method claims we recommend COSeg's corrected protocol, with the columns above as robustness checks.

---

## 9. Before publication: open checks

1. **Other methods.**
   - Run the swap test on other methods of the same family with public code or checkpoints: Seg-PN; TaylorSeg and
     DyPolySeg if checkpoints exist; any method reusing VIP-Seg's cross-term.
   - The paper's scope depends on whether the shortcut is family-wide or VIP-Seg-specific.
2. **S0 for the clean base.** Every result in §5–§7 is S1. CR on S0 is one training run (about 2 GPU-h).
3. **Seeds.**
   - Add a second seed for CR and VR. The shortcut (+19.94) is ten times any seed spread we measured (≤ 1.7 between
     seeds; up to about 3 in earlier phases), but the CR − VR difference (+1.58) is not.
   - Report CIs combining training and episode variance.
4. **Route-A shortcut check.** Run the swap test on our CascadeProto (EPPM) checkpoints. By construction their stage
   is local to each query, so we expect no dependence. This is not yet measured.
5. **Contact the authors of CascadeProto and VIP-Seg** with §4–§5 before any public version. Ask for evaluation
   scripts.
6. **Verify every [snippet] literature number** against the original paper. arXiv, CVF and OpenReview were not
   reachable from our environment.

---

## 10. Limitations

* **Single seed per trained arm.** The large effects (§4: +24–28; §5: +19.94; §6: φ ≈ 1 and −22.6) are an order of
  magnitude above the seed spread we observed. Differences of 1–2 points (CR − VR; the condition-balanced leak-free
  gain) are not established.
* **Mostly one fold.** Phase-16 results are on S1; §4 is on S0. The two folds share scenes and differ in class
  composition (S1 has floor and wall; S0 has ceiling and beam).
* **One method family.** We cannot say how widely the position shortcut exists until §9.1 is done.
* **Our leak-free draw is not COSeg's protocol.** It keeps 2,048-point 1 m blocks and changes only the sampler.
  COSeg also changes voxelisation, point count and backbone pre-training.
* **Oracles are bounds, not reachable targets.** They use the query's own labels and are scored in-sample.
* **We cannot see the CascadeProto authors' code.** §4 shows that seen-class scoring is sufficient to explain the
  published level, not that it was used.

---

## 11. Conclusion

On the standard S3DIS 2-way 1-shot protocol:
* scoring on seen classes lifts a faithful CascadeProto from 57 to 81;
* a query-position shortcut contributes about 20 of VIP-Seg's 75;
* sampling density contributes about 23 of the remaining 56.

Once these are removed, the trained head matches plain prototype matching, and the features trained from scratch on
six base classes cap prototype matching near 64 even with unlimited shots. The honest level of this line of work is
therefore the mid-50s on the standard draw and about 30 on a uniform draw.

That is in line with the 37–45 that methods report on COSeg's corrected protocol. We suggest the field report the
audit columns of §8 and move method claims to protocols without these cues.

---

## Appendix A. Source of every number

All paths are relative to the repository root. "D-nn" is the decision file in `docs/decisions/` that registered the
experiment and its rules before it ran.

| claim | value | file |
| :--- | :--- | :--- |
| VIP-Seg S0 released, our pipeline | 71.97 | `results/phase16_p0/SUMMARY.md` (row "VIP-Seg S0"); 05 §4 sanity check |
| E1 `last` / `best`, VIP-Seg S1 released | 73.20 / 75.05 / 75.36 | `results/phase16_e1/SUMMARY.md` |
| E1 `best` − released; selection +1.85 [+1.43, +2.30] | — | `results/phase16_e1/SUMMARY.md`, paired table |
| VIP-Seg S1 log 72.84 last / 75.63 best | — | D-22 amendment |
| Table 4 rows (ours, unseen) | 49.08 … 57.15 | `results/phase15_full/SUMMARY.md` |
| Baseline seen 77.32 (`last`), 71.40 (`best`) | 0.77324 / 0.71399 | `results/seen/baseline_S0_on_S1_last.json`, `baseline_S0_on_S1.json` |
| Full seen 81.28 (`last`), 75.99 (`best`) | 0.81281 / 0.75987 | `results/seen/full_S0_on_S1_last.json`, `full_S0_on_S1.json` |
| VIP-Seg S0 released, seen | 79.13 | `results/seen/vipseg_S0_on_S1.json` |
| S1 baseline seen / unseen | 71.58 / 51.91 | `results/seen/baseline_S1_on_S0_last.json`, `baseline_S1_unseen_best.json` |
| Partial class leak 63.54 / 69.24 | — | D-21; `docs/research/2026-09-21_reproduction_report.md` §3.5 |
| Cross-term formula match 4.7e-14; attention moves 0.45 vs 0.0078 | — | `results/c1/c2_crosscorr.txt`; D-35 point 5 |
| Swap test 73.20 → 0.92, 75.36 → 0.87; equivariant cross-term 37.91 / 35.45 | — | `results/phase16_d37/SUMMARY.md` (C4 table) |
| 2 × 2 table, shortcut +19.94 [+18.40, +21.52], leak-free levels | — | `results/phase16_d37/SUMMARY.md` (D-37 tables and rules) |
| Sampling-condition table | — | `results/c1/c1_S1.txt`, `results/c1/c1_S0.txt` |
| Density intervention, φ 1.001 [0.962, 1.034], recall 0.245 / 0.833 / 0.540, own 0.837 → 0.463 | — | `results/phase16_p8/SUMMARY.md` (part B) |
| Spearman −0.57, p 0.051 (other share vs IoU) | — | D-35 point 2 |
| U 55.74 / 33.11; U + both + LP 58.55 / 33.15; CR leak-free 28.34 | — | `results/phase16_d46/SUMMARY.md` (CR columns) |
| U + both 57.63; LP +0.92 | — | D-39, D-40 outcomes |
| K-shot curve, fit 16.06 + 8.45/k, a 0.156, c 0.047 | — | `results/phase16_d46/SUMMARY.md` (P10.3) |
| Participation ratio 5.56; census 3.7–7.3; VIP-Seg 6.39; VICReg 42.45 / 95.25; U −5.20 / −11.34 | — | `results/phase16_d45/SUMMARY.md`; D-45 outcome |
| Condition-balanced training: other recall 0.18 → 0.46; leak-free +2.11; fixed100 −5.81 | — | `results/phase16_d49/SUMMARY.md`; D-49 outcome |
| Head/inference variants ≤ +2.8 | — | D-26 … D-29, D-33, D-34, D-38 … D-40, D-48 outcomes; 00 §4 index |
| Paper numbers 82.72 … 88.53, 84.53; Tab. 5 T = 1 85.21 | — | CascadeProto Tables 2, 4, 5 (`experiments/summarize.py` holds the printed values) |

## Appendix B. Reproducing the tables

| table | command (on a GPU machine with the S3DIS blocks) |
| :--- | :--- |
| 1 (unseen) | `experiments/phase14.py` queue, phase-15 rows; `eval.py --protocol fixed100` |
| 1 (seen) | `experiments/run_seen.sh` (the only script that sets `--allow_seen_classes true`) |
| 2, 3 | `experiments/run_d37.sh full` (C4 + D-37 training and evaluation) |
| 4 | `python experiments/c1_sampling_condition.py` |
| 5 | `experiments/run_p8.sh` |
| 6, K-shot, collapse | `experiments/run_d46.sh`, `experiments/run_d45.sh`, `experiments/d49_eval.py` |
