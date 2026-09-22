# CascadeProto: what an unofficial re-implementation reproduces, and what it does not

Status: 2026-09-22, complete for S3DIS 2-way 1-shot (every Table 4 row on fold S0; the baseline also
on fold S1). Every number below is a finished run.

## Verdict

**No implementation bug was found.** The trace went through every layer and each one checks out: the
pipeline reproduces VIP-Seg's published checkpoint to 0.2 points (§1), every equation of spec 02 is
implemented as written (272 tests, and an independent paper-to-code audit found nothing), and every
row of Table 4 was trained on the paper's schedule (§2). What does not reproduce has three causes, in
decreasing order of size:

1. **The paper's absolute numbers are inconsistent with its own base method** (about 30 points),
   and **neither a different mIoU (§3.1) nor VIP-Seg's trained encoder (§3.2) explains them**. Its
   baseline — "a plain VIP-Seg backbone with masked average pooling and single-step prototype
   matching" — is reported at 82.72 on S0, above VIP-Seg's own full model at 72.20 in the same paper.
   A baseline that removes VIP-Seg's prototype modules cannot outscore VIP-Seg with them unless
   something the paper does not state is different. Our pipeline scores VIP-Seg's released checkpoint
   at 0.7197, so it reaches the level VIP-Seg really has; our baseline at 49.08 sits below it, where
   a stripped-down VIP-Seg should, and where the first author's own ablations of TaylorSeg and
   DyPolySeg put the same construct (49–52). No setting described in the paper closes this gap (§3),
   and none of our own decisions on the baseline's path does either (§3.3, including VIP-Seg's
   schedule). **What does come close is scoring on classes seen in training** (§3.6): our baseline
   scores 77.32 / 71.58 (S0 / S1) when each fold is scored on the other fold's classes, against the
   paper's 82.72 / 79.83. That reading cuts the gap on the Avg column from 30.8 to 6.8 points and
   reproduces the paper's S0 > S1 order, which every other method in Table 2 reverses. It is
   consistent with the paper's numbers, not proof of how they were made; the authors' code is not
   released.
2. **The cascade depth does not work as the printed equations define it** (the paper's +2.55 on S0). One
   EPPM stage gains +7.07; stages two to four add −0.17. The mechanism: the only summands Eq.19 adds
   to the prototype are class-poor — `P_diffuse` has no class index at all (Eq.15–18) and `P_cross`
   mixes channels (Eq.14) — so a stage after the first can only pass its residual through; training
   even learns to switch the diffusion branch off (weight 0.47–0.51 → 0.01–0.08). The paper's own
   tables disagree about the T = 1 configuration by 0.63 points (§3), and Eq.12's gate output is not
   consumed by any later equation.
3. **A step the paper does not print** (+3.36). L2-normalising the point prototypes, which VIP-Seg
   does (`models/vipseg.py:142`), is worth +3.36 on the baseline and about half of the first stage's
   gain, since Eq.21's LayerNorm has the same effect. We keep the paper's literal Eq.3 as the default
   and the normalisation as the ablation switch `l2norm_point_proto` (D-10).

What **does** reproduce: the pipeline, the total gain of the added modules (+8.07 against +5.81), and
ADRM's increment (+0.60 against +0.64).


* **Paper.** "CascadeProto", Wang et al., `10069.pdf`. Method in §3 (Eq.1–27), hyper-parameters in
  §4.1, results in Tables 2–6.
* **Reference implementation.** VIP-Seg at the pinned commit, vendored in `models/vipseg.py`; the
  paper names it as the shared encoder (§4.1).
* **This work.** Every equation implemented from the paper, with each ambiguity recorded as a
  numbered decision in `docs/spec/00_SOURCES_AND_DECISIONS.md` (D-01…D-19) and each equation
  transcribed in `docs/spec/02_TENSOR_MATH_SPEC.md`. 272 CPU tests, mutation-checked per sub-phase.
* **Setting for every number below.** S3DIS, fold S0, 2-way 1-shot, 2,048 points per block, one NVIDIA
  L4. Training: 50 epochs × 480 episodes, batch 4, AdamW lr 1e-3 wd 0.1, StepLR halving every 10
  epochs, as §4.1 states. Evaluation: the `fixed100` protocol of D-08, 1,500 cached episodes.

---

## 1. The pipeline is sound

Three independent checks, because a reproduction that fails is only interesting if the harness is
not the reason.

| check | result |
| :--- | :--- |
| VIP-Seg's released S0 checkpoint, scored through **our** `eval.py`, our data and our metric | **0.7197** against its published **0.7220** [PAPER Tab.6] |
| VIP-Seg's own model trained through **our** loop, our data, our loss, 2,400 episodes | **0.6948**, against **0.689** from its own training script at the same point |
| CPU gate | 272 tests, every equation of spec 02 checked against an independent reference implementation |

So the data pipeline, the episode sampler, the loss, the optimiser loop and the mIoU metric all
reproduce VIP-Seg's published behaviour. Whatever the gap below is, it is not the harness.

---

## 2. What the full schedule gives

`fixed100`, 1,500 episodes, `best` checkpoint (D-15). One seed per row.

| configuration | switches | ours best | ours last | paper [Tab.4] |
| :--- | :--- | ---: | ---: | ---: |
| Baseline | `use_lma=false num_stages=0` | 0.4908 | 0.4907 | 0.8272 |
| + LMA | `use_lma=true num_stages=0` | 0.4965 | 0.4832 | 0.8398 |
| + Entropy Gate (T = 1) | `num_stages=1` | 0.5672 | 0.5672 | 0.8534 |
| + Cascade (T = 4) | `num_stages=4 use_adrm=false` | 0.5655 | 0.5603 | 0.8789 |
| + ADRM, the full model | defaults | 0.5715 | 0.5670 | 0.8853 |
| *aside*: baseline, prototypes L2-normalised | `… l2norm_point_proto=true` | 0.5244 | 0.5019 | — |

### The increments, one at a time

| step | ours (S0) | paper (S0) | verdict |
| :--- | ---: | ---: | :--- |
| Baseline → + LMA | +0.57 | +1.26 | same sign, about half the size |
| + LMA → + Entropy Gate (T = 1) | **+7.07** | +1.36 | five times the claim |
| + Entropy Gate → + Cascade (T = 4) | **−0.17** | +2.55 | **not reproduced; depth buys nothing** |
| + Cascade → + ADRM | +0.60 | +0.64 | reproduced almost exactly |
| Baseline → full model | +8.07 | +5.81 | larger than claimed |

The paper's S0 values are the ones Table 4 prints (82.72 / 83.98 / 85.34 / 87.89 / 88.53). An earlier
version of this report listed 83.93 / 85.35 / 87.41 for rows 2–4 and compared our S0 increments with
§4.3's **Avg** increments (+1.21, +1.42, +2.06, +0.56, which sum to +5.25, not +5.81); corrected
2026-09-22 (CHANGELOG 15y). No verdict changes.

**The total is reproduced; the attribution is not.** The paper spreads its +5.81 over four components.
We obtain +8.07, but almost all of it comes from the single first EPPM stage, and going from one stage
to four costs 0.17 points instead of gaining 2.55.

The "+ Entropy Gate" row is `num_stages=1` under D-17, i.e. a whole EPPM stage — gate,
cross-attention, diffusion, fusion, and the LayerNorm of Eq.21. That LayerNorm equalises the
prototype row norms, which is what `l2norm_point_proto` does on its own for +3.36. So of the +7.07,
roughly 3.4 points may be the normalisation the paper never prints and roughly 3.7 the stage's own
refinement. This split is an analogy, not a measurement: it was never run as an ablation, and the
+3.36 itself is one seed (best 52.44, last 50.19).

**What is reproduced:** ADRM's +0.64, to within 0.04. **What is not:** the cascade depth that gives
the paper its largest single increment, and the absolute level — 49.08 against 82.72 for the
baseline, 57.15 against 88.53 for the full model.

---

## 3. Why the absolute level is out of reach

The paper's own numbers are internally inconsistent, which is enough to explain part of the gap and
is documented in full in `2026-09-20_paper_vs_code_audit.md`.

* **The baseline row is above the method it is built on.** Table 4's baseline is 81.28 Avg / 82.72 S0,
  while VIP-Seg's own row in Table 2 is 74.15 Avg and its Table 6 S0 is 72.20. Deleting VIP-Seg's
  entire prototype stack would therefore improve S0 by 10.52. The paper states no protocol,
  retraining or normalisation difference that would account for it, and every §4.3 increment is
  measured from this row.
* **Two tables disagree about the same configuration.** Table 4 row 3 (83.91 Avg) and Table 5 `T=1`
  (83.28 Avg) are the same model under D-17's mapping. This matters more than it looks: our T = 1 row
  is already within 0.4 points of our full model, so which configuration the paper means by
  "+ Entropy Gate" decides whether its cascade increment is being measured from the right place.
* **Eq.14 is dimensionally invalid as printed.** `A ∈ R^{Nq×Ns}` times `ψ(P^{t-1}) ∈ R^{(N+1)×D}` is
  undefined unless `Ns = N+1`, and even then it yields `R^{Nq×D}`, not the `(N+1)×D` that Eq.19 and
  Eq.21 need. Any implementation must discard part of what is printed; D-01 records the choice.
* **Eq.12's output is never used.** `x_gated` appears in Eq.12 and nowhere else; Eq.14 multiplies the
  ungated `ψ(P^{t-1})` and Eq.21's residual is the ungated `P^{t-1}`. Under the prototype reading the
  entropy gate is dead code; under the feature reading it is consumed by Eq.13. Both were measured
  (§4) and neither changes the score.
* **Table 6's parameter budget is unreachable.** Four EPPM stages built exactly from Eq.10–21 cost
  317,580 parameters, already more than the ~0.31M the paper implies for LMA + EPPM + ADRM together,
  and the LMA alone (148,352) exceeds the "+0.12M" of §4.4.
* **Table 6 leaves no room for a stronger backbone.** CascadeProto is listed at 2.88M parameters and
  8.86G FLOPs against VIP-Seg's 2.76M and 8.48G. With LMA + EPPM + ADRM at about 0.47M, what is left
  for the backbone is at most VIP-Seg's own (about 2.57M without PEM/PDM), so an unstated, larger or
  heavier backbone cannot be what lifts the baseline to 82.72.
* **Eq.11 cannot do what the abstract claims.** `g = σ(2(θ − H))` with `H ∈ [0, ln 2]` gives `g < 1`
  for every finite `θ`, so the gate can only attenuate, never "amplify low-entropy foreground
  information".

---

### 3.1 Is it the metric? No.

The paper's VIP-Seg row may have been copied from the VIP-Seg paper while its own rows were scored
differently. We re-scored every full-schedule checkpoint and VIP-Seg's released checkpoint with five
definitions (`pipeline/metrics_alt.py`, `results/rescore/`):

| run | primary (D-08) | accumulated, with bg | per episode, fg | per episode, with bg | point accuracy | paper |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| VIP-Seg, released checkpoint | 71.97 | 72.72 | 74.27 | 75.35 | 85.46 | 72.20 |
| Baseline | 49.08 | 51.88 | 51.12 | 57.15 | 72.81 | 82.72 |
| Baseline + L2 | 52.44 | 55.09 | 54.64 | 60.38 | 75.56 | — |
| + LMA | 49.65 | 53.23 | 50.74 | 58.92 | 76.35 | 83.98 |
| + Entropy Gate (T = 1) | 56.72 | 59.21 | 58.45 | 63.93 | 78.32 | 85.34 |
| + Cascade (T = 4) | 56.55 | 59.52 | 58.04 | 64.63 | 79.84 | 87.89 |
| + ADRM, full model | 57.15 | 59.97 | 58.55 | 64.85 | 79.74 | 88.53 |

* The re-implementation of the primary metric equals it on every row, so the other columns are
  computed from the same counts correctly.
* No mIoU definition moves the full model past 65; the most generous one (per episode, background
  included) adds 7.7 points, a quarter of the gap. Even point accuracy, which is not an mIoU, leaves
  the baseline 9.9 points and the full model 8.8 points short of the paper.
* The ordering is the same under every definition: **VIP-Seg is ahead of every CascadeProto row by
  13–15 points**. Under no scoring does the printed CascadeProto beat the method it is built on,
  which is the paper's central claim.

### 3.2 Was the paper's baseline built on VIP-Seg's trained encoder? No.

VIP-Seg's released S0 checkpoint supplies a trained encoder and feature head. Plugged into our
baseline — masked average pooling and a dot product, exactly the paper's "plain VIP-Seg backbone with
masked average pooling and single-step prototype matching" — and scored with no training at all
(`experiments/vipseg_init_probe.py`, D-20, `results/vipinit/`):

| baseline on … | primary mIoU | best alternative definition |
| :--- | ---: | ---: |
| our encoder, trained from scratch | 49.08 | 57.15 (per episode, with bg) |
| **VIP-Seg's trained encoder** | **47.37** | 56.96 (per episode, with bg) |
| our encoder, L2-normalised prototypes | 52.44 | 60.38 |
| **VIP-Seg's trained encoder, L2-normalised prototypes** | **54.97** | 63.08 |
| VIP-Seg's full model (its PEM/PDM head on the same encoder) | 71.97 | 75.35 |
| paper, Table 4 baseline | 82.72 | — |

* **VIP-Seg's 72 comes from its prototype head, not its encoder.** On VIP-Seg's own trained features,
  prototype matching scores 47–55, the same as on ours; the 17–25 points on top come from PEM/PDM.
  A plain backbone with masked average pooling therefore sits near 50 whether its encoder is trained
  here or by VIP-Seg, and cannot reach 82.72 — 10 points *above* the head it strips away.
* **Independent validation of our training.** Our from-scratch encoder gives the baseline the same
  feature quality as VIP-Seg's released one (49.08 against 47.37; 52.44 against 54.97 with L2).
* The fine-tuning step of D-20 was not run: by the decision rule fixed before the probe (below 60 →
  reject), the hypothesis is rejected at zero training.

### 3.3 Trace back through the method: which of our own decisions can reach the gap?

The gap is already 33.6 points on the **baseline**, which contains none of the added modules. So a
decision can only explain it if it touches the baseline's path: encoder → feature head → masked
average pooling → dot product → cross-entropy → optimiser and schedule. Every decision, sorted by that
criterion:

| decision | on the baseline's path? | status |
| :--- | :---: | :--- |
| D-01, D-02, D-03, D-11, D-14, D-16, D-18, D-19 (EPPM internals) | no | can only move the increments, which reproduce in total (+8.07 vs +5.81) |
| D-04, D-05, D-06, D-13 (LMA, GMMN, CLIP) | no | LMA is +0.57 vs +1.26; the paper's three modalities differ by ≤ 2 points |
| D-07 split, D-08 metric | yes | verified: VIP-Seg's checkpoint scores 71.97 against 72.20; metric equals [34]'s code |
| D-10 logit form | yes | measured: L2 prototypes +3.4, a 1/√D scale −5.5 |
| D-15 model selection | yes | `best` and `last` both reported; they differ by < 2.3 points |
| D-17 what "baseline" means | yes | matches §4.3 word for word: VIP-Seg backbone, masked average pooling, single-step matching |
| D-12 epoch size, batch, LR decay | yes | measured: VIP-Seg's schedule gives 49.01 against 49.08 (below) |

Code on the baseline's path, checked against VIP-Seg's own:

* **Encoder and feature head** — the same modules, names and layer order; ENC-6/ENC-7 on the GPU show
  our features equal VIP-Seg's forward pass on the same weights.
* **Masked average pooling** — the same background and foreground means as `models/vipseg.py:108-130`
  (CP tests), and on VIP-Seg's own trained encoder it scores 47.37 with no training (§3.2), so the
  pooling is not what keeps the baseline near 50.
* **Optimiser** — AdamW, lr 1e-3, weight decay 0.1 on every parameter, as VIP-Seg's learner
  (`models/vipseg_learner.py:16-21`). VIP-Seg's own model trained through our loop reaches 0.6948
  after 2,400 episodes, where its own script reaches 0.689.
* **The one difference in schedule.** VIP-Seg trains 24,000 steps at batch 1 and halves the learning
  rate every 7,000 steps (`scripts/vipseg_s3dis.sh`). D-12 keeps its 24,000 *episodes* but at the
  paper's batch 4, so we take 6,000 steps and halve every 1,200: a quarter of the updates, the last
  third of them at a learning rate below 1.25e-4. That is our decision, not the paper's; the paper
  gives only "batch size 4, 50 epochs, halve every 10 epochs".

The probe for it: the baseline on VIP-Seg's exact schedule (batch 1, 24,000 steps, halving every 15
epochs ≈ 7,200 steps), `train.py --batch_size 1 --lr_step_epochs 15`. It scores **49.01** (best,
epoch 40) and 47.88 (last) on fixed100, against 49.08 and 49.07 on D-12's schedule, and its validation
curve stays within noise of the old one at every checkpoint (`results/b1/`, CHANGELOG 15x). Four times
the updates and a slower decay change nothing.

**The trace back is closed.** Every decision on the baseline's path is verified or measured, and the
code on it matches VIP-Seg's. Nothing we decided ourselves can account for the 33.6 points.

### 3.4 Are the cited numbers and the metric the ones we use? Yes.

* **The metric of the protocol the paper follows.** The paper states "We follow the standard N-way
  K-shot episodic protocol [34]" (§4.1), [34] being AttMPTI. AttMPTI's `evaluate_metric`
  (`runs/eval.py` of github.com/Na-Z/attMPTI) accumulates TP, predicted and ground-truth counts per
  class over all test episodes and averages IoU over the classes **excluding background** — the same
  computation as VIP-Seg's copy, which we call unchanged (VIP-Seg only adds 0.001 to the
  denominator). So our primary metric is the standard one the paper names.
* **The VIP-Seg row is VIP-Seg's own released logs.** The released log folders of VIP-Seg are
  `log_S0_N2_K1_0.722026`, `log_S1_N2_K1_0.760875`, `log_S0_N2_K5_0.764836`, `log_S1_N2_K5_0.775364`,
  `log_S0_N3_K1_0.687951`, `log_S1_N3_K1_0.683496`, `log_S0_N3_K5_0.681455`, `log_S1_N3_K5_0.704371`
  (github.com/changshuowang/VIP-Seg_NeurIPS2025, `log_s3dis_VIPSeg/`). Table 2's VIP-Seg row, 72.20 /
  76.09 | 76.48 / 77.54 | 68.80 / 68.35 | 68.15 / 70.44, is exactly those values rounded. They were
  computed with the AttMPTI metric on the loader we use, and our pipeline scores the S0 checkpoint at
  71.97.
* **The jump the paper claims.** In Table 2's 2-way 1-shot S0 column, every earlier method improves on
  its predecessor by 0.2–5.6 points (DGCNN 36.34 … DyPolySeg 72.02, VIP-Seg 72.20). CascadeProto (Text)
  reports 88.53, +16.3 over VIP-Seg, and its own Table 4 baseline (82.72) would already be +10.5 over
  the method it is built on. Since the cited rows and the metric are verified to be on the scale we
  measure, the paper's own rows are the only ones not on it.
* **One protocol statement not tested.** §4.1 also says "using Areas 1, 2, 3, 4, 6 for training and
  Area 5 for testing under two category splits S0 and S1", which contradicts the class-only split of
  [34] that the cited rows use (D-07). Restricting the test rooms to an unseen area would make the task
  harder, not 30 points easier, so its prior is low; it is recorded as untested (the `area5` flag of
  D-07 was never implemented). The stronger form of the protocol hypothesis was tested instead
  (§3.5).

### 3.5 Were the test classes seen in training? Partial leakage: 12–15 points.

The gap is nearly uniform across Table 4 (28.6–34.3 points), the pattern of a data or protocol
difference rather than a model difference. The largest such difference the paper's wording admits is
that the category split did not take effect and the S0 test classes were trained on. D-21 trains on
all 12 classes (`--train_classes all`, 20 epochs, past the divergence point) and scores the usual
fixed100 test episodes. It is a **partial** leak, not an upper bound (an earlier version of this
report said upper bound; corrected in CHANGELOG 15y): each test class fills 9,600 × 2 / 12 = 1,600
way slots, against 24,000 × 2 / 6 = 8,000 for a class the model is trained on normally, and the runs
had not converged. §3.6 measures the full form.

| row | class split (50 epochs) | test classes seen (20 epochs) | change | paper |
| :-- | --: | --: | --: | --: |
| baseline | 49.08 | **63.54** | +14.46 | 82.72 |
| full model | 57.15 | **69.24** | +12.09 | 88.53 |
| full − baseline | +8.07 | +5.70 | | +5.81 |

Leakage lifts both rows by 12–15 points and leaves the gain of the added modules close to the paper's
+5.81, which is what a protocol difference would do. But the leaked full model stays 13.5 points below
the paper's *baseline* and 19.3 below its full model. The leaked runs had not plateaued at 20 epochs
(validation rose 5–7 points between epochs 10 and 20), so the share leakage explains could be somewhat
larger. One seed per row (`results/leak/`, CHANGELOG 15x).

### 3.6 Scored on classes seen in training: the paper's level, within 5–8 points

The full form of the leakage reading needs no new model: S0's training classes are exactly fold 1's
test classes, so an S0-trained checkpoint scored with `--cvfold 1` is scored on classes it saw for 50
epochs, and symmetrically for an S1-trained one on fold 0. If the paper's training and evaluation folds
were swapped, these are the numbers it would print. Each prediction below was written down before its
run (`docs/research/2026-09-21_gap_diagnosis.md` §4b); fixed100, 1,500 episodes, one seed
(`results/seen/`, CHANGELOG 15y).

**Baseline.**

| classes scored | not seen in training (standard) | seen in training (`last.pt`) | effect of having seen them |
| :--- | ---: | ---: | ---: |
| fold 0: beam, board, bookcase, ceiling, chair, column | 49.08 (S0 model) | **71.58** (S1 model) | +22.5 |
| fold 1: door, floor, sofa, table, wall, window | 51.91 (S1 model) | **77.32** (S0 model) | +25.4 |

| baseline | S0 | S1 | Avg | gap to the paper (Avg) |
| :--- | ---: | ---: | ---: | ---: |
| paper, Table 4 | 82.72 | 79.83 | 81.28 | — |
| standard protocol | 49.08 | 51.91 | 50.50 | 30.8 |
| scored on seen classes | 77.32 | 71.58 | 74.45 | **6.8** |

**Other checkpoints on fold 1 (seen classes).** Baseline `best.pt` (epoch 20) 71.40; full model
`best.pt` 75.99 and `last.pt` **81.28** (paper S0 full 88.53); VIP-Seg's released S0 checkpoint 79.13
(its cited, unseen-class number is 72.20).

* **The jump is having seen the classes, not easier classes.** Unseen, fold 1 is only 2.8 points
  easier than fold 0 (51.91 against 49.08; the first author's own ablations give 52.67 and 54.35 for
  S1). Having seen the classes is worth 22–25 points on either fold.
* **It reproduces the paper's fold order.** Under the standard protocol S1 > S0 for us (51.91 > 49.08)
  and for every method in Table 2 with both folds (VIP-Seg 76.09 > 72.20). The paper reports S0 > S1 on
  every row (82.72 > 79.83). Scored on seen classes, ours is S0 > S1 as well (77.32 > 71.58).
* **It explains the baseline above VIP-Seg.** A baseline scored on seen classes (77.32) next to VIP-Seg's
  cited unseen-class number (72.20) produces exactly the ordering of §3's first bullet.
* **Extra training helps only on seen classes.** On the unseen S0 classes `last` and `best` agree
  (49.07 / 49.08); on the seen fold-1 classes `last` is 5.9 points higher (77.32 / 71.40).
* **What remains: 5–8 points** (S0 5.4, S1 8.2; full model 7.3). That is the size of one seed's spread
  (1–3 points per run), the paper's 600 random episodes against our fixed100, and an unknown checkpoint
  choice combined; none of these was measured separately.
* **Consistent, not proven.** Without the authors' code this shows that the paper's numbers match
  seen-class scoring and do not match the protocol it states, not how they were produced.

## 4. Ambiguities we resolved by measurement

Each was implemented behind a switch, with the default unchanged, and measured on three seeds.

| ambiguity | readings | outcome |
| :--- | :--- | :--- |
| D-02: what Eq.10–12 gates | the prototype, or the features | No measurable difference (t = +0.08). The feature reading is the only one that consumes Eq.12, so it is the better reading of the text, but it changes nothing. |
| D-10: Eq.23's "scaled dot-product matching" | no scale, as the equation prints, or `1/√D` | **No scale.** `1/√D` costs 5.5 points (t = −2.90) and its training loss plateaus at 0.446–0.463 against 0.33–0.39, because it flattens the softmax. |
| D-18: the scale inside Eq.14's softmax | as printed, or standardised projections | As printed. The LayerNorm probe costs 0.9 points and pins the attention near the uniform end. |

Caveat on all three: they were measured with the short diagnostic harness, whose budget was too small
(§5). The D-10 result has a budget-independent mechanism and is likely to survive; the other two
should be treated as untested at the real schedule.

---

## 5. A methodological error of our own

The diagnostic harness trained for 600 optimiser steps, five epochs of the real schedule. The P1
training logs show the full model sitting at the baseline's level at epoch 10 (valid 0.4942 against
0.4642) and separating only between epoch 10 and epoch 20 — between 1,200 and 2,400 steps. Every
conclusion drawn from that harness was therefore drawn before the effect existed, including the claim
we held for several hours that "the cascade adds nothing measurable". The full-schedule run of §2
overturns it. The harness default is now 2,400 steps.

The loop is also not bit-reproducible on CUDA: the identical command at a fixed seed gave 0.5218,
0.5223 and 0.5316 for one configuration and 0.5164, 0.5346 and 0.5029 for another, so differences
below about two points carry no information on one seed.

---

## 6. What is not established

* Every full-schedule number is **one seed**. At the short budget the seed spread was 0.010–0.031.
  The increments that survive that scale are the +7.07 of the first stage and the +8.07 total; the
  +0.57 of LMA, the −0.17 of cascade depth and the +0.60 of ADRM are all inside one standard
  deviation of it, so "ADRM reproduces the paper's +0.64" and "depth buys nothing" are both stated
  with that caveat.
* Only 2-way 1-shot; fold S1 only for the baseline (§3.6). Table 2's other columns and ScanNet were
  never run.
* Table 6 (parameters, FLOPs, time) is not an acceptance criterion here (D-09), and its FLOPs are a
  lower bound because fvcore does not count the custom CUDA kernels.
* How the paper's numbers were produced is not settled. The encoder's training, the schedule (D-12)
  and our own decisions are ruled out (§3.2, §3.3). Scoring on classes seen in training reproduces
  them to within 5–8 points and reproduces their fold order (§3.6); the residual was not decomposed
  (seeds, random600, checkpoint choice). Only the authors' code or protocol can confirm the mechanism.
