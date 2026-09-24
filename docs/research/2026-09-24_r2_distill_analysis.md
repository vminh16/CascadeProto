# R2 (D-29) analysed: why the distillation transferred but mIoU did not, and where the 5-point r0 − VIP-Seg gap sits

Status: 2026-09-24, phase 16, after R2 (`results/phase16_r2/`, commit `4a2ec60`). Desk analysis only: the repository,
the R2 result files, the two local R2 checkpoints (inspected on CPU), VIP-Seg's released training and evaluation logs
at the pinned commit (fetched from GitHub, not previously in the repo), and the QGE and DPA papers (fetched from
arXiv HTML). No GPU run, no code change. Tags: **[measured: file]** a number read from a result file or log,
**[verified]** read from code or a primary source, **[inferred]** a reading or prediction.

## Summary

* **The logit-pair cosine is the wrong proxy for decisions.** It is an energy-weighted, uncentred correlation: points
  far from the boundary dominate it, and sign errors near the boundary barely move it (bound in §1.1). d29 raised it
  on novel classes from 0.771 to 0.843, above VIP-Seg's 0.807, with mIoU unchanged (+0.06). That fits a student that
  reshaped logit *magnitudes* on easy points [measured: `results/phase16_r2/test_S1_fixed100.json`; mechanism inferred].
* **On training episodes the oracle rule adds no decision information beyond CE.** The labels already give CE every
  sign, and CE alone had already brought the base-class pair cosine to 0.933 [measured: `train_r0_S1.log`]. The P0
  headroom (+14.43 for both r0 and d29) is a **transductive information gap**: it needs the query labels at test time.
  It is not a training-signal gap, and R2 leaves it exactly unchanged [measured; reading inferred].
* **QGE and DPA are weak precedents.** QGE distils AttMPTI's *support prototypes* with a KL at temperature 15. That is
  one run (+3.27), 1-way, on a pretrained DGCNN, in the foreground-leak setting where COSeg's retraining drops QGE by
  27.8. DPA's "distillation" is **self**-distillation between its own stages and uses no query labels (+1.73 S0 /
  +3.58 S1, single runs) [verified: arXiv 2308.03177 §3.2, T5; 2401.16051 §3.4, Eq.8, T3].
* **The D-29 implementation is sound.** β is 0.0 / 1.0 in the checkpoints, the teacher is detached, and the masks are
  right. `last.pt` was scored, the diagnostic is the loss function itself, and the loss was optimised (0.067 → 0.023).
  Only stale comments were found (§2).
* **New evidence: VIP-Seg's released S1 training log** (12 validations). It reads **68.52 at 2,000 updates, 70.07 at
  6,000, 72.84 at the last (24,000) and 75.63 at best (22,000)**. Our r0 reached **70.26 after 6,000 updates**, the
  same level at the same number of updates. The 4.98-point valid gap (75.24 vs 70.26) splits into **≈ 2.8 points of
  best-of-12 selection on test-class episodes** and **≈ 2.6 points of training length** (24,000 vs 6,000 updates)
  [measured: VIP-Seg `log_vipseg.txt`; split inferred, one run each].
* **On S0, VIP-Seg's own run peaks at update 4,000** (72.94 valid) and ends at 68.97. The mean of its 12 validations
  is 69.29. The published 72.20 is therefore ≈ 3–4 points of selection above that run's typical level. Also, both
  published numbers (76.09 / 72.20) come from a *second* test evaluation of the best checkpoint; the training log's
  own test of the same checkpoint gave 74.65 / 72.58 [measured: VIP-Seg logs].
* **Next.** (1) r0 on VIP-Seg's schedule (batch 1, 24,000 updates, LR halved every 7,200, ~13 validations), about
  2.3 GPU-h. It discriminates schedule from everything else. (2) Weight averaging of the last-stage checkpoints, which
  captures the checkpoint-to-checkpoint swing without selecting on test classes. (3) The text prior (the paper's
  multimodal idea) on top of (1), with 2 seeds per arm. The evidence puts this family's honest `last.pt` ceiling near
  VIP-Seg's own *typical* level (≈ 73 S1 / ≈ 69–70 S0 valid), below its published numbers.

---

## 1. Why the objective transferred but mIoU did not

### 1.1 What the uncentred logit-pair cosine measures

Take one query b and one pair c < c'. Over the points I = {i : y_i ∈ {c, c'}}, the model's pair function is
a_i = L_ic − L_ic' and the teacher's is b_i = T_ic − T_ic' [verified: `models/oracle_distill.py:53-66`, spec 02 §14].
Both are linear in the feature: a_i = ⟨f_i, u⟩ with u = M_eff,c − M_eff,c', and b_i = ⟨f_i, v⟩ with v = O_c − O_c'
(`L_final = F^q M_effᵀ`, `models/cascadeproto.py:205-216`). So

  cos = uᵀ S v / √(uᵀ S u · vᵀ S v),  S = Σ_{i∈I} f_i f_iᵀ = n (C + m mᵀ),

a cosine between prototype *differences* in the metric of the pair's own second-moment matrix [inferred, algebra].
Three consequences:

1. **Energy weighting.** Each point enters with weight ∝ b_i² (or a_i²). A point far from the oracle boundary (large
   |b_i|) counts quadratically more than a point near it.
2. **A sign-error bound.** With â = a/‖a‖ and b̂ = b/‖b‖, 1 − cos = ½‖â − b̂‖². Let D be the points where a and b
   disagree in sign. On D, (â_i − b̂_i)² ≥ b̂_i², so **1 − cos ≥ ½ Σ_{i∈D} b̂_i²**. The loss penalises a sign error only
   through the teacher's energy share on that point, and the *fraction* of points in D is unconstrained [inferred,
   algebra].
3. **Mean-direction dominance.** Post-ReLU features share a large mean m (D-27's revision measured a high raw cosine
   of every point to the centre, `00_SOURCES_AND_DECISIONS.md:669-675`). So uᵀSv carries a term n(u·m)(v·m): the
   pair's mean logit offset, set mostly by the pair's majority class, usually background. Much of the cosine is a
   per-query balance term, not a boundary term [inferred].

**Worked example** [inferred, arithmetic]. 80 % of the pair's points are easy (|b| = 1) and 20 % are near the boundary
(|b| = 0.1).
* Student A equals the teacher on the easy points and has the **wrong sign on every near point**. Then cos =
  (0.8 − 0.002)/(0.8 + 0.002) = **0.995**, with 20 % of points misclassified relative to the teacher.
* Student B has **the correct sign everywhere** but constant magnitude, a_i = sign(b_i). Then cos = (0.8 + 0.02) /
  √(1 · 0.802) = **0.916**.

The loss prefers A to B. A rise from 0.771 to 0.843 can therefore come entirely from reshaping magnitudes on easy
points, without moving any decision. The per-class IoUs agree: d29 − r0 is within ±2 points on every class (door
−0.7, floor +2.0, sofa −1.2, table −0.6, wall +1.2, window −0.2) and the headroom is identical (+14.43 / +14.43)
[measured: `test_S1_fixed100.json`, `class_iou`, `paired`].

VIP-Seg illustrates the same point from the other side. Its logit-pair cosine is *lower* (0.807) and it is 5.1
points better, with the gains exactly on the classes where the cosine says nothing: floor +11.7, sofa +8.0, table
+9.1 over r0 [measured: same file]. The metric ranks the three models d29 > VIP-Seg > r0; mIoU ranks them
VIP-Seg ≫ d29 ≈ r0.

### 1.2 What distilling the oracle rule can add beyond CE

* **On a training episode CE already knows every correct sign.** The oracle rule is a *worse* labeller: on test
  classes the equal-norm oracle scores 85.0–86.3 mIoU, not 100 [measured: `eval_S1.log`]. So distillation cannot add
  decision information. It can add only (i) a magnitude prior (the pair logits should be proportional to projections
  on O_c − O_c'), and (ii) gradient on correctly classified points, where CE's gradient has vanished (CE ≈ 0.11 at
  the end). Neither carries information about novel classes [inferred].
* **CE alone already produces most of that structure on base classes.** r0's L_distill, computed without gradient,
  falls from 0.160 to 0.067, a pair cosine of 0.933. d29 reaches 0.023 (0.977) at the cost of a slightly higher
  CE (0.146 − 0.023 = 0.122 vs r0's 0.117 at epoch 50) [measured: `train_r0_S1.log:18,75`,
  `train_d29_S1.log:18,75`]. The +0.044 on base classes became +0.072 on novel classes, so the structure transfers;
  the decisions do not.
* **The P0 headroom is transductive.** Replacing the prototypes by O uses the *query's labels at test time*: it
  measures how far the support-derived class direction is from the query's own. That is the one-shot intra-class
  shift. A head trained on base classes can at best learn E[O | support, unlabelled query]. The residual is the part
  of O that those inputs do not determine. D-26…D-28 tried to estimate exactly that part from the unlabelled query
  and failed (P0: +0.37 / −1.21 on VIP-Seg; P1, P2 closed) [measured: `00_SOURCES_AND_DECISIONS.md:626-633,711-715,
  758-763`]. R2 leaves the headroom exactly where it was (r0 +14.43, d29 +14.43 kept-norm; +14.81 / +15.24 equal-norm)
  [measured: `eval_S1.log`]. Reading: **the headroom is an inference gap, not a missing training signal**, so no
  training-only loss toward O should be expected to close it [inferred].

### 1.3 QGE and DPA: what they did and why it need not carry over

| | QGE (Ning et al., arXiv 2308.03177) | DPA (Liu et al., arXiv 2401.16051) |
| :--- | :--- | :--- |
| student | support prototypes of AttMPTI; cosine logits × t = 15 | early-stage prototypes |
| teacher | "optimal query prototypes", mask-pooled with the query labels | its own deeper-stage prototypes (**no query labels**) |
| loss | KL on softmaxed logits (Eq.7), training only (§3.2) | KL on prototype distributions (Eq.8), γ = 0.1 (Eq.11) |
| base model | AttMPTI, DGCNN pretrained 100 epochs (§4.2) | DGCNN pretrained 150 epochs, 40,000 iterations (§4.2) |
| setting | S3DIS **S0, 1-way 1-shot**, 2,048 points | S3DIS 2-way 1-shot |
| gain | HR(KL) 66.27 → 69.54 (+3.27); L1 +1.64, L2 +1.89 (T5) | +PD: 64.35 → 66.08 (S0, +1.73), 70.72 → 74.30 (S1, +3.58) (T3) |
| seeds | single run (no variance reported) | single run (no variance reported) |

[verified: arXiv HTML of both papers, fetched 2026-09-24]

Why they may have helped there and not here [inferred]:
1. **QGE's student has no query-conditioned module.** AttMPTI scores with fixed support prototypes, so the KL is
   effectively a feature-learning signal that pulls support and query features of a class together. VIP-Seg's
   PEM/PDM already condition every prototype on the query (`models/vipseg.py:235-310,338-405`), and CE already trains
   the encoder through them.
2. **QGE's loss is decision-level and calibrated** (softmax at t = 15). D-29's uncentred cosine is not (§1.1).
3. **The QGE effect is small against its own noise.** Its three loss forms differ by 1.6 points. It was measured once,
   in the setting whose foreground-density leak COSeg showed inflates QGE by 27.8 points (74.05 → 46.27 when
   retrained without it) [verified: `2026-09-21_external_sources_on_gap.md:95-99`]. Whether HR survives the
   correction is unknown.
4. **DPA is not evidence for an oracle target at all.** D-29 cites it correctly as stage self-distillation
   (`00_SOURCES_AND_DECISIONS.md:781-782`), but it shares only the word "distillation".
5. **Both base models sit at 66–70 with a pretrained backbone**, where the prototypes are further from saturation
   than VIP-Seg's head.

### 1.4 How much power R2's null has

D-29 assumed sd ≈ 0.5 for a difference of two single runs, from two seeds of a 2,400-step run
(`00_SOURCES_AND_DECISIONS.md:829-831`). At the end of a full run the checkpoint-to-checkpoint swing is larger:
* d29 valid goes 71.60 → 70.19 between epochs 40 and 50 [measured: `train_d29_S1.log:65,77`].
* VIP-Seg's own S1 run swings by −0.87, +2.35, +1.44 and −2.79 between consecutive validations at LR ≤ 5e-4
  [measured: VIP-Seg `log_vipseg.txt`, below].

A `last.pt` level sd of ≈ 1 is therefore more realistic [inferred]. The difference of two runs then has sd ≈ 1.4, and
a true +1.0 gain would read below +0.5 about a third of the time. The stop verdict still stands, because the mechanism
moved (cosine +0.072) while nothing else did, but R2 cannot exclude a +0.5 to +1 effect [inferred].

## 2. Bug audit of D-29

| check | finding | evidence |
| :--- | :--- | :--- |
| β in the checkpoints | r0: config and args `distill_beta` 0.0; d29: 1.0 | CPU load of `log_r2/.../{last,best}.pt` [verified] |
| loss enters the objective | `episode_loss` adds β·L_distill when β > 0 and raises if it is missing | `pipeline/model_api.py:37-40` [verified] |
| gradient acted | d29's L_distill falls to 0.023 vs r0's 0.067 without gradient; test logit-pair cosine 0.843 vs 0.771 | `train_*_S1.log:75`, `eval_S1.log` [measured] |
| weights differ | encoder ‖Δ‖ 47.2 of 369.7; stages 3.7 / 4.8 / 4.6 / 6.9 of 13–17; routing 0.52 of 1.14. Not proof by itself: a same-seed CUDA rerun also diverges (AGENTS §6) | CPU state-dict comparison [measured] |
| ADRM `w_g` | row norms r0 [0.604, 0.599, 0.572, 0.508], d29 [0.603, 0.603, 0.560, 0.466]. Both are near the default-init scale √(1/3) ≈ 0.577, so d29 did not satisfy the loss by re-routing stages | CPU [measured]; init scale [inferred] |
| teacher detached | `oracle_logits` detaches `f_q` before computing O and T; the student gradient flows through `L_final` only | `models/oracle_distill.py:46-50,75` [verified] |
| point and pair masks | `points` = 1 on c ∪ c' for c ≠ c'; the diagonal is excluded by `triu(1)`; pairs need both classes present | `models/oracle_distill.py:59,64-66` [verified] |
| no label leak at eval | forward returns before the loss in eval mode | `models/cascadeproto.py:195-196` [verified] |
| arms scored from `last.pt` | `run_r2.sh` passes `last.pt` for both; the JSON records those paths, β, and protocol "clean" | `experiments/run_r2.sh:52,59`; `test_S1_fixed100.json` `models` [verified] |
| diagnostic = loss | the evaluation calls the same `pair_logit_cosine(logits, oracle_logits(...))`. It pools over all (episode, query, pair) instead of the loss's per-episode mean, a weighting difference only | `experiments/r2_distill_eval.py:126,130` [verified] |
| decide rule | the mechanism check reads `logit_pair`, not the first form's `cos(M_eff, O)` | `experiments/r2_distill_eval.py:270-271` [verified] |
| best vs last | r0 `best.pt` is bit-identical to `last.pt` (epoch 50). d29 `best.pt` is epoch 40 (valid 71.60) and was not tested (D-22) | CPU [measured] |

**Stale text, no effect on results:**
* `experiments/run_r2.sh:2` still describes "distillation of the effective prototype", and `:14` describes the go
  rule as "a higher cos(M_eff, O)". Both are the first form of D-29.
* `train.py:81` help says the loss acts "on the effective prototype".

Worth correcting when the code is next touched. **No implementation bug was found.**

## 3. The r0 vs VIP-Seg gap

### 3.1 New evidence: VIP-Seg's own training curves

VIP-Seg validates every 2,000 iterations on valid episodes of the **test classes** and keeps the best
(`runs/training.py:89-100`, `scripts/vipseg_s3dis.sh:20-24`). Its released logs at the pinned commit give the whole
curve. Values are valid mIoU at updates 2k … 24k [measured: `log_s3dis_VIPSeg/log_S1_N2_K1_0.760875/log_vipseg.txt`
lines 22–174 and `log_S0_N2_K1_0.722026/log_vipseg.txt` lines 22–167, github.com/changshuowang/VIP-Seg_NeurIPS2025
@28aedc5]:

| updates (k) | 2 | 4 | 6 | 8 | 10 | 12 | 14 | 16 | 18 | 20 | 22 | 24 |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| LR (×1e-3) | 1 | 1 | 1 | .5 | .5 | .5 | .5→.25 | .25 | .25 | .25 | .125 | .125 |
| **S1** | 68.52 | 69.50 | 70.07 | 71.04 | 71.82 | 72.22 | 71.90 | 72.71 | 71.84 | 74.19 | **75.63** | 72.84 |
| **S0** | 72.06 | **72.94** | 66.55 | 65.31 | 69.66 | 69.60 | 67.06 | 69.74 | 68.94 | 72.11 | 68.54 | 68.97 |
| our r0 (S1) | 68.27 @1.2k | 69.49 @2.4k | 68.85 @3.6k | 69.91 @4.8k | 70.26 @6k | | | | | | | |

In both logs the "Test Classes" line and a final TEST follow. Training-log TEST of the saved best checkpoint: **74.65
(S1) / 72.58 (S0)**. The published folder values 0.760875 / 0.722026 come from a separate `log_vipseg_eval.txt`
(10 lines each) [measured]. We score the same released S1 checkpoint at 75.36 on fixed100 and 74.49 / 75.82 / 77.21
on the three random600 draws [measured: `eval_S1.log`]. So the published 76.09 is best-of-12 on test-class valid
episodes, plus the upper of two test draws the repo shows.

**Reading** [inferred unless tagged]:
* **At equal updates the two loops agree on S1.** VIP-Seg 70.07 at 6k vs r0 70.26 at 6k, and 68.5 vs 68.3 in the first
  2k/1.2k. This matches CHANGELOG 14e/15a on S0: VIP-Seg's model in our loop 69.48 at 600 steps against 68.9 from its
  own script at 2,000 [measured: `docs/CHANGELOG.md:521`, `2026-09-21_reproduction_report.md:67`].
* **VIP-Seg keeps improving on S1 after 6k updates**, to 72.84 at `last` (mean of its last five validations 73.44).
  Our schedule stops at 6k updates with LR 6.25e-5 for the last fifth; VIP-Seg is still at 5e-4 until update 14,000.
  The integrated step size Σ lr is 1,200 × 1.9375e-3 = **2.33** for us against 7,000 × 1.75e-3 + 3,000 × 1.25e-4 =
  **12.6** for VIP-Seg. The AdamW decay factor exp(−wd Σ lr) is 0.79 against 0.28, so VIP-Seg's weights also travel
  and shrink far more [arithmetic from `train.py:286-287`, `vipseg_learner.py:16-21,43`].
* **Selection is worth about 2.8 points on S1 and 4.0 on S0** (best − last in VIP-Seg's own run). The S0 run peaks at
  update 4,000, at LR 1e-3, and never recovers. On S0, longer training did not help VIP-Seg at all; the published
  72.20 is a selected checkpoint about 3 points above its run's mean (69.29).
* **Decomposition on our valid draw.** VIP-Seg best 75.24 (75.63 on its own draw) − r0 70.26 = 4.98 ≈ selection
  2.8 (75.63 − 72.84) + training length 2.6 (72.84 − 70.26) − draw offset 0.4. Single runs, and different valid
  draws for the VIP-Seg column. This decomposition is a hypothesis to test, not a measurement.

### 3.2 Every difference between the two training paths

| # | difference | ours | VIP-Seg | plausibility as a cause of the 5 points |
| :-- | :--- | :--- | :--- | :--- |
| 1 | checkpoint selection | `last.pt` headline (D-22); 5 validations | best of 12 on test-class valid (`runs/training.py:89-100`) | **high, ≈ 2.8 on S1 / 4.0 on S0** (§3.1) |
| 2 | optimizer updates, decay points, Σ lr | 6,000 at batch 4; halve every 1,200 (`train.py:235,287`; `pipeline/episodes.py:38-42`) | 24,000 at batch 1; halve every 7,000, stepped per iteration (`vipseg_learner.py:21,43`; `vipseg_s3dis.sh:20-24`) | **high on S1, ≈ 2.6**: curves agree at equal updates and VIP-Seg keeps rising. **Low on S0** (its curve does not rise) |
| 3 | seed / run luck | 1 run | 1 run | medium. `last.pt` level sd ≈ 1 [inferred, §1.4] |
| 4 | gating bias | `nn.Linear(128, T, bias=False)` (`models/adrm.py:25`) | `nn.Linear(128, T)` with bias (`models/vipseg.py:190`) | low. Only a constant preference per step, and `w_g` stays near init in both arms (§2) |
| 5 | query layout inside PEM/PDM | P^0 expanded to B_q = 2 (`cascadeproto.py:177`), same module call (`vip_stage.py:51`) | `feature_memory.repeat(N_way)` (`vipseg.py:145`) with B_q = N_way · n_queries = 2 | **ruled out**: identical tensors, since `N_QUERIES = 1` (`pipeline/episodes.py:24`). The cross-query reshape (`vipseg.py:285-291`) sees the same B_q = 2, way = 3 |
| 6 | prototype formation | MAP + L2 (`prototypes.py:37-45`, `cascadeproto.py:165-166`) | same, including the 0.1 background fallback (`vipseg.py:108-142`) | ruled out |
| 7 | encoder and feature head | same modules; ENC-6/7 equal on the same weights | — | ruled out (`2026-09-21_reproduction_report.md:220-226`) |
| 8 | data, sampler, augmentation | `WAY_RATIO/NUM`, shift 0.1, rot, jitter (`pipeline/episodes.py:22-27`) | same (`vipseg_s3dis.sh:12-13,28`; `main.py:55-66`) | ruled out |
| 9 | loss | CE (+0 · GMMN, +0 · distill in r0) | CE (`vipseg.py:182`) | ruled out |
| 10 | BatchNorm batches | one forward per episode (`train.py:163`) | one per episode | ruled out |
| 11 | gradient averaging over 4 episodes | yes | no (noisier, 4× more steps) | part of #2 |

**Does D-12's null transfer to the head? No** [inferred]. The batch-1 probe changed nothing for the *headless*
baseline, whose valid curve is flat and noisy from epoch 10 (0.469 / 0.469 / 0.422 / 0.498 / 0.485)
[measured: `results/b1/b1.log:29-77`]. MAP has nothing more to fit. VIP-Seg's head keeps gaining ≈ 4.5 points on S1
between updates 6k and 22k (§3.1). R1's statement that the curves plateau by epoch 20 was made on EPPM-family S0
runs (`results/phase16_r1/SUMMARY.md`) and does not cover this head.

### 3.3 Cheapest discriminating experiments

One full-schedule run cost 1.67 h on the L4 (r0: 4,999 s training + 5 validations of ~210 s,
`train_r0_S1.log:75`). The batch-1 baseline took 4,675 s for the same 24,000 episodes (`results/b1/b1.log:75`), so
batch 1 costs no more per episode.

**E1 — r0 on VIP-Seg's schedule** (no code change needed):

    train.py <r0 flags of run_r2.sh> --batch_size 1 --lr_step_epochs 15 --valid_every 4 --seed 0

* 24,000 updates, LR halved every 7,200 (the value 15x used), 13 validations. The run directory gets `_b1`.
* About 1.4 h of training plus 13 × 3.5 min ≈ **2.2–2.4 GPU-h**.
* Score `last.pt` and `best.pt` on the same four draws as R2, next to r0 and VIP-Seg.

Pre-registered reading [inferred expectations]:

| outcome on S1 fixed100 | reading | next |
| :--- | :--- | :--- |
| `last` ≥ 73.0 and valid curve still rising after 6k | training length explains ≈ 2.5 points; R2.0's −5.16 was mostly schedule + selection | adopt the b1 schedule for every route-B run (a new decision amending D-12 for `stage_type=vip`) |
| `best` (VIP-Seg's selection rule) − `last` ≥ 2 | selection on test-class episodes is a real ≈ 2–3 point effect in our loop too | report both numbers in every table |
| `last` ≤ 71.0 | schedule is not the cause | E2: second seed of r0 at batch 4 (luck), then `bias=True` in ADRM (#4, needs a flag) |
| in between | one run cannot separate it | second seed of E1 |

**E1′ (optional, same cost)**: batch 4 with Σ lr matched: 6,000 steps, lr 2e-3, halve every 1,750. It separates
"number of updates / gradient noise" from "integrated step size". Run it only if E1 lands in the first row.

**E1 on S0 is not needed first.** VIP-Seg's own S0 run does not gain from length, so S0 will mainly test the
selection effect [inferred].

## 4. What next

Constraints:
* Training-free fixes are closed (D-26…D-28). D-29 is closed (R2.2).
* The goal is to beat VIP-Seg 72.20 / 76.09 and EDS-Net 73.32 / 74.67 honestly (D-22: unseen classes, `last.pt`
  headline).
* Text (CLIP + LMA) is the idea to keep (`models/lma.py`). Our LMA measured +0.57 on the printed model; the paper
  claims +1.26 (`2026-09-21_reproduction_report.md:92`).

**The honest ceiling** [inferred from §3.1]. VIP-Seg's *typical* level in its own run is ≈ 72.8–73.4 on S1 (last /
mean of the last five validations) and ≈ 69 on S0 (last 68.97, mean 69.29). The published numbers add ≈ 3 points of
test-class checkpoint selection and a favourable test draw. A `last.pt` result beating 76.09 / 72.20 therefore needs
roughly +3 over VIP-Seg's own typical level. Nothing measured in this repository on top of a trained head comes close:
D-26 ≤ +0.37, D-27 < 0, D-28 = D-26, D-29 +0.06, LMA +0.57. Per-stage and oracle distillation are now also
measured null. Two honest options:
* (a) Claim "beats VIP-Seg retrained under the identical protocol and seeds" (E1 as the reference). This is
  achievable.
* (b) Compare with the published numbers under VIP-Seg's own selection rule, disclosed, with `last.pt` also reported.
  That needs a maintainer decision amending D-22's headline.

Claiming to beat 76.09 with `last.pt` is unlikely with this family on present evidence.

| rank | direction | evidence | expected size | cost | pre-registered rule |
| :-- | :--- | :--- | :--- | :--- | :--- |
| 1 | **E1: route-B reference on VIP-Seg's schedule** (§3.3) | curves agree at equal updates; VIP-Seg +2.8 after 6k on S1 | +2 to +3 over r0 (S1 `last`), 0 to +1 on S0 | 2.3 GPU-h | §3.3 table |
| 2 | **Weight averaging of the last-stage checkpoints** (SWA-style; BN re-estimated on training episodes, never on valid/test) | VIP-Seg's consecutive validations swing up to 2.8 points at LR ≤ 2.5e-4, so `last` is a noisy draw from a band whose top is what best-of-12 selects; SWA's flat-optimum argument (Izmailov et al. 2018, arXiv 1803.05407; classification, not measured here) | +0.5 to +1.5 over `last` [not measured]; also halves the run-to-run noise that limited R2 | ≈ 0 extra training if E1 saves its validated checkpoints (needs a small `train.py` change and a decision) | adopt if SWA − `last` ≥ +1.0 on fixed100 with a CI above 0 and positive on all three random600 draws; SWA must never be selected on valid scores |
| 3 | **Text prior on top of the E1 recipe** (the paper's multimodal claim): LMA text prototype as a support-validated gated prior (MM-FSS Eq.9–10 form), since fixed fusion weights gained only +0.1 to +0.9 | LMA +0.57 here; paper +1.26; MM-FSS text +3.3 / +3.7 (corrected setting, ScanNet 1-way); gated vs fixed +1.9 / +2.0 vs +0.1…+0.9 (`00_SOURCES_AND_DECISIONS.md:613-617`; `2026-09-21_external_sources_on_gap.md:158`) | +0.5 to +2 | 2 arms × 2 seeds, E1 is arm A seed 0: ≈ 7 GPU-h more | go: mean gain over 2 seeds ≥ +1.0 on fixed100 and on the random600 mean, both seeds > 0 → S0 once per arm. Stop: mean < +0.5. Needs its own decision first (00:613-617), incl. how the unnormalised `P_modal` combines with VIP-Seg's L2 prototype input (`cascadeproto.py:165-172`) |
| 4 | Trained base-class calibration (COSeg Eq.9–12) | P1.4 "in between": AUC 0.716 (S1) / 0.649 (S0) on VIP-Seg (`00_SOURCES_AND_DECISIONS.md:711-715`); COSeg +3.44 in its setting | 0 to +1 [inferred] | 2 seeds × 2.3 h + code | only after 1–3; go ≥ +1.0 as in 3 |

Not recommended: further D-29 variants (KL with temperature, per-stage oracle supervision). The distillation target
already transferred (cosine 0.843 > VIP-Seg's 0.807) and left both decisions and headroom unchanged (§1.2), which is
the failure mode those variants share [inferred].

**Seeds.** Every go/stop above uses at least 2 seeds per arm. The `last.pt` sd ≈ 1 of §1.4 means one run per arm
resolves only effects of ≥ 3 points [inferred].

## Sources

Repository (commit `4a2ec60`):
* `docs/spec/00_SOURCES_AND_DECISIONS.md` (D-26 557–635, D-27 639–715, D-28 719–763, D-29 767–856)
* `docs/spec/02_TENSOR_MATH_SPEC.md` §14 (347–381)
* `models/oracle_distill.py`, `models/cascadeproto.py`, `models/vip_stage.py`, `models/adrm.py`,
  `models/prototypes.py`, `models/vipseg_backbone.py`, `pipeline/model_api.py`, `pipeline/episodes.py`
* `train.py`, `experiments/r2_distill_eval.py`, `experiments/run_r2.sh`
* Inherited VIP-Seg code: `models/vipseg.py`, `models/vipseg_learner.py`, `runs/training.py`,
  `scripts/vipseg_s3dis.sh`, `main.py`
* R2 results: `results/phase16_r2/{test_S1_fixed100.json, eval_S1.log, train_r0_S1.log, train_d29_S1.log}`
* Other results: `results/phase16_r2_pre/SUMMARY.md`, `results/phase16_r1/SUMMARY.md`, `results/b1/b1.log`
* Checkpoints `log_r2/s3dis_S1_N2_K1_point_T4_vip{,_distill1}/{last,best}.pt` (CPU inspection)
* Notes: `docs/CHANGELOG.md` (14e/15a 505–530, 15k 737–760, 15x 935–960, 16f 1140–1183, 16k 1303–1326,
  16l 1327–1347); `docs/research/2026-09-21_reproduction_report.md`;
  `docs/research/2026-09-21_external_sources_on_gap.md`; `docs/research/2026-09-23_gap_and_upgrade_research.md`

External:
* [VIPSEG-LOG] VIP-Seg released logs, github.com/changshuowang/VIP-Seg_NeurIPS2025 @28aedc5093c0d386d526864c49505ae6921b1600,
  `log_s3dis_VIPSeg/log_S1_N2_K1_0.760875/{log_vipseg.txt, log_vipseg_eval.txt}` and
  `log_s3dis_VIPSeg/log_S0_N2_K1_0.722026/{log_vipseg.txt, log_vipseg_eval.txt}` (fetched 2026-09-24)
* [QGE] Ning et al., *Boosting Few-shot 3D Point Cloud Segmentation via Query-Guided Enhancement*,
  https://arxiv.org/abs/2308.03177 (§3.2, Eq.7, §4.2, Tables 1, 5)
* [DPA] Liu et al., *Dynamic Prototype Adaptation with Distillation for Few-shot Point Cloud Segmentation*,
  https://arxiv.org/abs/2401.16051 (§3.4, Eq.8, Eq.11, §4.2, Table 3)
* [COSeg] An et al., CVPR 2024, https://arxiv.org/abs/2403.00592 (Table 1, via the external note)
* [SWA] Izmailov et al., *Averaging Weights Leads to Wider Optima and Better Generalization*,
  https://arxiv.org/abs/1803.05407
