# The prototype gap re-diagnosed: sampling condition, class presence and gauge (phase-16 review, with directions)

Status: 2026-09-24, after N2 (docs v2.0). Desk research plus CPU measurements on the local S3DIS blocks; no GPU run,
no change to any model code. Sources: the repository (code, result files, decision log), VIP-Seg's inherited code,
`experiments/c1_sampling_condition.py` (new, CPU, inherited loader only) with its outputs `results/c1/`, and primary
literature fetched by three search agents (COSeg, MM-FSS, GFS-VL, EDS-Net, DPA, Seg-PN, QGE, and 2-D FSS / transductive
few-shot papers). Tags: **[measured: file]** read from a result file, **[verified]** read from code or a primary
source, **[inferred]** reasoning or prediction, **(not measured)** an estimate.

## 0. Summary

1. **The oracle is not a prototype oracle.** Every oracle rule in the repository replaces the rows of the classes
   *present* in a query block and leaves the absent classes on the head's own rows, in a different gauge
   (all-positive unit-feature means against LayerNorm-centred rows). With non-negative features the absent class is
   expected to rarely win, so the oracle effectively knows **which classes each query block contains** [verified:
   `experiments/r2_distill_eval.py:112-125`, `models/oracle_distill.py:34-43`; the suppression is inferred from the
   sign structure]. Part of the +12.7 (E1) gap would then be block-level presence information that no prototype can
   carry. Not yet measured how much; P5-A measures the oracle's absent-class false positives directly.
2. **P4's row-wise oracle split is a gauge artifact.** "Background row only −26, foreground rows only −9, all rows
   +13.7" replaces some rows by `‖m‖·O_c` (all-positive, large common component) and keeps the others centred; the
   replaced rows win or lose everywhere for that reason alone. The conclusion "the error is a joint,
   query-conditioned shift of every prototype", which motivated D-33/D-34, does not follow [verified:
   `experiments/p4_background_probe.py:97-104`; reading inferred].
3. **The benchmark's sampler creates two conditions for every class, and the support only ever shows one.** The
   inherited sampler over-samples the class a block is sampled for (density factor 2 − π) and under-samples every
   other class (1 − π). In a 2-way episode the query block of way k is sampled for class k, so the *other* episode
   class, when present, is at background density. Supports are always in the over-sampled condition. Measured on
   1,500 seeded episodes per fold (`results/c1/`): **floor 19.2 %, wall 23.5 % (S1) and ceiling 23.5 % (S0) of their
   query points are in the "other" condition; every other class 0.7–6.8 %.** Other-condition points have the
   background's kNN-16 radius (0.11–0.13 m, vs 0.07–0.08 m for own-condition points and support foreground) and no
   duplicates (own: 4–9 % duplicated points). COSeg measured that removing the over-sampling costs old-protocol
   models **24–27 mIoU** at the same 2,048 points [verified: COSeg Tab.1] — the cue is worth over a third of their
   score. COSeg did not examine the N-way asymmetry, and no later paper reports it [verified by search].
4. **This explains the planar-class pattern with a mechanism that can be tested at test time.** The three classes
   with the largest other-condition share are the three worst IoUs of VIP-Seg's released checkpoints (ceiling 64.4,
   floor 65.3, wall 69.7), Spearman(other share, IoU) = −0.57, p = 0.051 over 12 classes [measured: `results/c1/`,
   `results/phase16_p0/test_vipseg_S*.json`]. The earlier reading ("always background in training", ρ = −0.55) is
   confounded with it (both follow class ubiquity) and could never be tested; this one can (§4). It does **not**
   explain S0's beam / board / bookcase gaps (other shares 0.9–6.6 %, oracle gaps 15–19): S0 has a second error
   type.
5. **VIP-Seg's head cannot fix it and plausibly amplifies it.** Its cross-correlation is episode-global and
   class-agnostic (the `reshape(72, -1)` mixes both queries and all support slots; §1.4), so its only
   query-conditioned, class-specific path is absent; the query gate is shared by all rows and computed from
   max-pooled tokens that favour the densest class. Consistent with P3: the head *lowers* floor recall from 0.874
   (support prototype, no head) to 0.719 [measured: `results/phase16_p3/SUMMARY.md`].
6. **Every stopped phase-16 mechanism is predicted to fail by this model** (§2.5): each acts on the prototype's
   semantics, none on the condition or on presence.
7. **Two untested, cheap confounds of the "query-conditioned shift"**: BatchNorm runs on per-set batch statistics
   in training (support batch and query batch separately) and on running statistics at test; and the encoder
   divides local offsets by one `torch.std` over the whole batch, so blocks in one forward are coupled even in
   `eval()` [verified: `models/encoder.py:182-189`, `models/vipseg_backbone.py:86-98`].
8. **Next step: one inference-only probe (P5, ~1–1.5 GPU-h, no training)** that splits E1's errors by condition and
   presence, intervenes on the sampling of the same scans, tests test-time batch statistics, a leak-free test draw
   and a training-free dual-condition prototype. Its counterfactuals decide between five method lines (§5).
9. **On 80+:** with the condition and presence errors isolated, the plausible S1 range is 76–78 `last` (not
   measured); 80 on `last` has no supporting evidence yet. The honest paper story that this opens is stronger than
   "+x on S1": a diagnosis of the old benchmark's two conditions plus a method that gains on both the standard
   protocol and a leak-free draw.

---

## 1. Errors and blind spots in the phase-16 experiments

### 1.1 The oracle leaks class presence [code verified; suppression inferred; size not measured]

`oracle_replaced` (R2, E1, N1, N2) writes `norm · O_c` into the rows of classes present in query block b and keeps
`M_eff` for absent ones; `oracle_directions` returns zero rows for absent classes (used as-is by the D-29 teacher).
`O_c` is the normalised mean of unit post-ReLU features, so `⟨f_i, O_c⟩ ≥ 0` and large for every point, while an
`M_eff` row is a LayerNorm output (centred before its affine map), whose dot product with a non-negative feature is a
contrast of either sign. The absent class is therefore expected to lose at most points: the oracle can hardly make a
false positive of a class the block does not contain (P5-A counts them). In the 2-way protocol the other way's class is absent from the
other query block in 6 % (floor), 27 % (wall), 74 % (table), 88 % (door), 93 % (window) and 98 % (sofa) of episodes
[measured: `results/c1/c1_S1.json`, blocks own/other]. Sofa's E1 error is mostly precision (recall 0.946, IoU 75.4,
so precision ≈ 0.79 [inferred from `results/phase16_p3/SUMMARY.md`]); how much of it sits in blocks without sofa is
exactly what the oracle hides. **Fix:** a presence-fair oracle that keeps a competing row for absent classes in the
oracle's gauge (e.g. the support prototype's unit-feature direction with the common norm), reported next to the
current one.

### 1.2 The row-wise oracle split measures gauge, not error location [verified; reading inferred]

`row_oracle_logits` replaces the selected rows with `‖m_c‖ · O_c` and keeps `M_eff` elsewhere. Logits are invariant to
adding one vector to all rows, not to adding it to some rows: the replaced rows gain the common positive component
of `O`, i.e. a class-wide bias. "Background only → everything background (−26)", "foreground only → too much
foreground (−9)" is what that bias predicts regardless of where the error is. A gauge-consistent split first aligns
the oracle to the head (`min_{s>0,v} Σ_c ‖s O_c + v − m_c‖²` per block, then replace rows of `s O + v`), or splits the
decision directly: the background-vs-foreground axis `m_0 − (m_1 + m_2)/2` against the foreground axis `m_1 − m_2`.
Consequence: D-33's premise ("a joint, query-conditioned shift of every prototype, not a contaminated row") is not
established. P4's `clean_bg` result (+0.09) stands; it tests contamination, not the row's condition.

### 1.3 The sampling condition was never examined [measured: `results/c1/`]

Inherited sampler [VIPSEG dataloaders/loader.py:37-58], COSeg's Algorithm 1: for a block of raw class shares π_k
sampled for class s, the expected counts are

  n_s = 2048 · π_s (2 − π_s),   n_k = 2048 · (1 − π_s) π_k  (k ≠ s),

so the sampled-for class is at density factor ρ_own = 2 − π_s and every other class, *including the other episode
class*, at ρ_other = 1 − π_s; ρ_own/ρ_other = (2 − π_s)/(1 − π_s) ≥ 2 for every block. Some points of the
sampled-for class are drawn twice (exact duplicates). C1, 1,500 seeded test episodes per fold:

| class (fold) | other share of query points | other-way blocks containing it | kNN-16 radius own / other [m] | duplicated own / other |
| :--- | ---: | ---: | :--- | :--- |
| wall (S1) | 0.235 | 363 / 500 | 0.084 / 0.121 | 0.044 / 0.001 |
| ceiling (S0) | 0.235 | 473 / 500 | 0.072 / 0.123 | 0.064 / 0.001 |
| floor (S1) | 0.192 | 472 / 500 | 0.072 / 0.130 | 0.060 / 0.001 |
| table (S1) | 0.068 | 132 / 500 | 0.078 / 0.133 | 0.050 / 0.002 |
| bookcase (S0) | 0.066 | 91 / 500 | 0.082 / 0.119 | 0.052 / 0.000 |
| door, chair, window, beam, column, board, sofa | 0.007–0.042 | 11–101 / 500 | 0.068–0.084 / 0.10–0.13 | 0.04–0.09 / ≤ 0.005 |

Support foreground radius 0.079 (S1) / 0.077 (S0) m, support and query background 0.116–0.119 m. Other-condition
occurrences are also smaller fragments (floor: median 152 sampled points vs 723 own; [measured]). Share of all
foreground query points in the other condition: 11.8 % (S1), 7.3 % (S0). The S1 **training** classes are S0's list:
7.3 % of their foreground query points are in the other condition, so ~93 % of the foreground supervision can be met
by "dense ⇒ foreground".

The density is visible to the network: every DyPowerConv divides kNN offsets by one scalar `torch.std` over the batch
[verified: `models/encoder.py:182-189`], so local point spacing survives normalisation; the duplicates put zero-distance
neighbours into own-class neighbourhoods.

### 1.4 VIP-Seg's cross-correlation is episode-global and class-agnostic [verified: index algebra + CPU check]

In PEM/PDM, `que [B, 72, 128].reshape(72, −1)` has rows r = (query b = ⌊r/36⌋, projected rows 2(r mod 36), +1) and
`sup [N+1, 72, 128].reshape(72, −1)` has rows r = (slot w = ⌊r/24⌋, projected rows 3(r mod 24) … +2). After
`reshape(B, 128, N+1, 128)` the output index "query b′" is the *parity* of the projected row and "slot w′" its residue
mod 3, so

  A[b′, w′] = softmax( Σ_{r=0}^{71} Q′_{⌊r/36⌋, 2(r mod 36)+b′}ᵀ S′_{⌊r/24⌋, 3(r mod 24)+w′} / √128 ):

every "per-query, per-class" attention matrix sums over **both queries and all three support slots**; the slot index
only selects a subset of the shared projection filters. `experiments/c2_vipseg_crosscorr_check.py` rebuilds the inherited
module: the formula equals it to 4.7e-14; changing only way 2's support moves slot 1's attention by up to 0.45 and
changing only query 2 moves query 1's by 0.075, against a typical entry of 0.0078 [measured:
`results/c1/c2_crosscorr.txt`]. The class-specific, query-conditioned part of the head is therefore
only the self-gate `σ(W₃′ G_s,c)` from class c's own support slot; the query enters through `σ(W₃ G_q)`, one gate
shared by every row, computed from 64 max-pooled tokens of 32 random points, which the densest (own) class
dominates. D-01's "clean" per-slot form (our EPPM) and D-23's "pooled S′" were compared against a head whose
cross-term is neither; R1's +15.5 measures VIP-Seg's whole PEM (class-specific self-gates plus an episode-level
channel mixing), not per-class attention.

### 1.5 BatchNorm regime and batch coupling [verified; effect not measured]

Training calls the encoder twice per episode, on the N·K support blocks and on the B_q query blocks
[verified: `models/vipseg_backbone.py:86-98`, as VIP-Seg `models/vipseg.py:85-89`], with BatchNorm in the encoder
(`models/encoder.py:230,317,320,483`) and in the feature head using **each set's own batch statistics**. At test the
same layers use running averages over alternating support and query batches. The network is trained on per-set
standardised activations; at test a support batch and a query batch keep their own offsets, which is a query- and
episode-dependent shift of exactly the kind the oracle absorbs. Transductive batch statistics at test (support and
query forwards separately, momentum 0) match training and use no label; TaskNorm (Bronskill et al., ICML 2020)
documents the size of such effects in few-shot learning. One evaluation pass (§4, arm C). Separately, `torch.std`
over the batch couples the two query blocks of an episode in `eval()` too, so "our stages keep the queries
separate" does not hold at the encoder.

### 1.6 Statistics and protocol [verified]

* E1's validations swing between 70.9 and 75.2 from epoch 28 to 50 [measured: `results/phase16_e1/train_e1_S1.log`], larger than
  every effect since D-30. Weight averaging (EMA/SWA) over the last stage, recommended in the R2 analysis (§ Next, 2)
  and never run, lowers the `last` noise every later comparison pays.
* The paired bootstrap resamples episodes as independent; episodes share scans (and class pairs), so the CIs are
  too narrow. A bootstrap over class pairs, then episodes, is the conservative version.
* One training run per arm against thresholds of +1.0 with a training sd of ~0.5–1 point: fine as a screen, but a
  negative result (N2 −0.66) and a null (D-29 +0.06) are not distinguishable from each other by one run.
* N1's warm start dropped the optimiser moments (noted in its summary); the "neck never opened" verdict is confounded
  with it, which D-34 addressed only by retraining.

### 1.7 D-26 tested frozen EM, not trained self-support [verified: 2-D literature via agents]

The literature that motivated D-26 gains when the refinement is trained in the loop: IPMT's intermediate prototype
improves with trained iterations (62.5 → 64.1 → 66.8, +2.7 from the iterations alone, PASCAL-5ⁱ 1-shot); SSP's
self-support is trained end-to-end with its own loss; DPA's prototype-to-query attention +3.05 and QGE's background
prototype adaptation +5.09 (S3DIS, old protocol) are trained modules. P0 measured a frozen EM on checkpoints never
trained to expect it; its failure says nothing about the trained form. Under §2's model it also fails for a specific
reason: other-condition points get low foreground responsibility, so the M-step never sees them (confirmation).

## 2. The problem, re-derived

### 2.1 Causal graph [inferred]

```
sampler(s) ──► density ρ, duplicates ──┐
class Y ──► appearance A ──────────────┼──► encoder features F = g(A, ρ, context) ──► head ──► logits
block composition ──► context ─────────┘         ▲
                                     support: always ρ_own     query: ρ_own (sampled-for) or ρ_other
```

The support prototype estimates E[F | Y = c, own condition, support block]. The query needs E[F | Y = c, condition of
this block, this block]. The per-block oracle is computed downstream of the condition and of presence, so it absorbs
both; a prototype computed from the support cannot.

### 2.2 Why training makes it worse: shortcut learning and gradient starvation [inferred]

With CE, once a feature separates most training points with margin, their gradients vanish (p → 1) and the learning
of other predictive features is starved (Pezeshki et al., NeurIPS 2021). Density separates ~93 % of the foreground
query points of S1 training episodes and all support foreground from background (§1.3). The remaining gradient comes
from other-condition occurrences (mostly ceiling in S1 training) and background points that resemble foreground. Prediction: E1 recognises a
class much better in its own condition than in the other, and the gap is largest where the model never had to learn
the class without density — testable in P5-A. COSeg's Tab.1 (−24 to −27 mIoU for these methods when the 2,048 points are
sampled uniformly; whether they were retrained per setting was not checked here) gives the size of the cue.

### 2.3 Gradient flow through the head [verified from code; consequences inferred]

* **Support side.** `P⁰_c = n(mean_{i∈M_c} f_i)`: every foreground support point of class c receives the same gradient
  `(I − P⁰_c P⁰_cᵀ) ∂L/∂P⁰_c / (|M_c| ‖μ_c‖)`. Support points get no per-point discriminative signal, only a
  collective shift of their mean, plus the sparse gradient of the max-pool argmax points in the head's support slot.
* **Query side.** Per-point CE gradients `Σ_c (p_ic − y_ic) M_eff,c` plus the max-pool argmax points of `G_q`, which
  are mostly own-class (dense) points: the query gate's gradient is own-condition driven as well.
* **LayerNorm at the end of every module** projects the gradient orthogonally to the mean and the radial direction:
  modules rotate rows, norms are fixed by γ and grow only through PDM's outer residual; step logits have different
  scales, which the gating network absorbs.
* **The encoder's fusion of the 9 input channels** is fixed and early: `PosE` mixes normalised-XYZ and RGB embeddings
  0.8 / 0.2, `LoConv` averages knn features, xyz and rgb embeddings with weights 1/3 [verified:
  `models/encoder.py:136,269,356`]. The metric xyz channels (0–2) are **not read at all** [verified: `models/encoder.py:645`]; the model sees height only
  min-max normalised per block. RGB, the one density-independent stream, is fused before any density-sensitive
  aggregation can be separated from it.

### 2.4 Error taxonomy (what P5 measures)

| type | definition | who removes it |
| :--- | :--- | :--- |
| E-a | FN of other-condition occurrences (sparse, often small fragments) | per-block oracle; condition-robust features or prototypes |
| E-b | FP of a class in a query block that does not contain it | oracle by construction (presence); presence estimation |
| E-c | FN/FP of own-condition occurrences (instance and context shift, boundaries) | oracle; trained query self-support |
| E-d | FP on base-class objects (chair → sofa, board/wall confusions) | base-class knowledge (BAM-type) |

### 2.5 Why each stopped mechanism was predicted to fail [inferred]

| decision | acts on | why it misses E-a / E-b |
| :--- | :--- | :--- |
| D-26 EM | query-side prototype from the model's own posterior | other-condition points have low responsibility; frozen, never trained to expect the update |
| D-27/28 base calibration | E-d only, training-free, AUC 0.65–0.72 | no effect on E-a; weak on E-d |
| D-29 distillation | logit geometry on training episodes | CE already has the labels; the pair cosine is dominated by easy (dense) points |
| D-31 text | fg-vs-fg ranking | E1's loss is fg-vs-bg; text is density-agnostic but carries only a weak fg-vs-fg signal (0.52–0.62) |
| D-32 clean background | contamination of the background row | the background row is already in the *same* (sparse) condition as other-condition foreground; removing contaminating points does not change that |
| D-33/34 neck | support points attend to query points they resemble | they resemble own-condition (dense) query points; no other-condition prototype is created |

## 3. Literature (verified by the agents; old = AttMPTI protocol, new = COSeg-corrected)

| evidence | number | protocol |
| :--- | :--- | :--- |
| COSeg Tab.1, same 2,048 points, over-sampled → uniform | AttMPTI 65.52 → 41.41, QGE 73.83 → 47.02, QGPA 61.95 → 38.34 (S3DIS 1-way 1-shot) | both |
| DPA Tab.3 (S3DIS 2w1s) | prototype rectification +13.70, prototype-to-query attention +3.05, stage distillation +2.65 | old |
| Seg-PN / QUEST (S3DIS S0 2w1s) | cross-correlation alone +15.40 over a training-free encoder | old |
| QGE (S3DIS S0 1w1s) | background prototype adaptation +5.09, holistic rectification +3.27 | old |
| COSeg Tab.3 (S3DIS 1-way) | base prototype calibration +3.44 / +2.06 | new |
| MM-FSS (S3DIS 2w1s) | 44.30 vs COSeg† 38.07 on the same 2-D-aligned backbone; on S3DIS the 2-D-aligned weights are transplanted from ScanNet, no S3DIS images are used | new |
| MM-FSS Tab.3e (ScanNet 1-way) | TACC (support-IoU-weighted text prior) +1.90 / +2.03 | new |
| EDS-Net Tab.5 (S3DIS 2w1s, same first author as VIP-Seg) | all three CascadeProto-like components +1.08 Avg (72.92 → 74.00) | old |
| IPMT (PASCAL-5ⁱ 1-shot) | trained prototype iterations +2.7 | 2-D |
| BAM (PASCAL-5ⁱ 1-shot) | base learner +1.51, with ensemble + adjustment +6.80 | 2-D |
| PMMs (PASCAL-5ⁱ 1-shot) | mixture prototypes +2.70 | 2-D |
| TIM / LaplacianShot / PT-MAP | transductive objectives +3 to +17 (classification, frozen features) | 2-D cls |

Gaps in the literature: no FS-PCS paper measures own- vs other-condition performance inside N-way queries, reports
per-class floor/wall/ceiling errors, or uses a query-mean oracle as a diagnostic [verified by search; "not found",
not proof of absence].

## 4. Proposed next step: P5, inference only (to be recorded as D-35 before any code)

Checkpoints: E1 `last.pt` (and VIP-Seg's released S1 checkpoint for arms A and C). S1 only; S0 held out.

* **A. Condition and presence split** (fixed100). For every query block tag each episode class as own / other-present
  / absent. Per class and pooled: recall own vs other; FP in own, other-present and absent blocks. The same for the
  current oracle and for a **presence-fair oracle** (§1.1). Counterfactuals from the counts: **cf-a** = E1 with each
  class's other-condition recall raised to its own-condition recall (FP unchanged); **cf-b** = E1 without FP in
  absent blocks.
* **B. Condition intervention** (fresh seeded draw with scan names, P4's episode machinery, 100 episodes per pair).
  For every query block sampled for a that contains the other episode class c: V0 the protocol block, V1 the same
  scan sampled for c (inherited `sample_pointcloud`), V2 the same scan sampled uniformly; same support. Recall and IoU
  of c under V0/V1/V2 and of a under V0/V2; episode-bootstrap CIs. Also: raw class of every false positive
  (base / clutter / other novel), which sizes E-d.
* **C. Transductive batch statistics**: every BatchNorm in batch-statistics mode, support and query forwards separate,
  running statistics frozen; fixed100 and random600 seed 0.
* **D. Leak-free draw**: support and query sampled uniformly (the inherited `random_sample=True` path, 2,048 points,
  COSeg's "w/o FG" form), 100 episodes per pair: E1, the support rule without head, the oracle.
* **E. Dual-condition prototypes, training-free**: a sparse view of each support block (keep each foreground point
  with probability (1 − r̂)/(2 − r̂), r̂ = 1 − √(1 − f) from the block's foreground share f, i.e. the inverse of
  f = r(2 − r); refill to 2,048 with copies of uniformly drawn remaining points jittered by the training jitter,
  σ = 0.01 m); the head re-run on the sparse views; foreground logit = max(dense, sparse). Control: the dense logits
  with a per-episode constant added to the foreground columns so that the count of foreground predictions equals the
  dual arm's.

**Rules, to fix before the run** (thresholds are the repository's: +1.0 = twice the single-run sd, the smallest gain
worth a training run; +0.5 = the smallest training-free effect worth a follow-up):

* P5.1 E-a matters: cf-a − E1 ≥ +1.0. Then P5.1a *sampling is causal*: V1 − V0 recall of c closes at least half of
  (recall_own − recall_other), CI above 0 → M1 and M2. P5.1b *fragment/context, not density*: V1 − V0 < 20 % of it →
  M1 only (uniform queries also present small fragments at background density), M2 dropped.
* P5.2 E-b matters: cf-b − E1 ≥ +1.0 → M3 gets a decision.
* P5.3 batch statistics: C − E1 ≥ +1.0 with CI above 0 on fixed100 and > 0 on random600 seed 0 → the test-time rule
  of every later arm, reported as transductive, with and without.
* P5.4 dual prototypes: E − control ≥ +0.5 with CI above 0 → M2 trained.
* P5.5 reported, no rule: the presence-fair oracle's gain, the head's gain over the support rule with and without
  the leak (D), the FP composition (B).

Cost (not measured): A ≈ 2 × 10 min, B ≈ 15 min, C ≈ 10 min, D ≈ 10 min, E ≈ 15 min on an L4.

## 5. Method lines, each tied to the error it targets

| line | targets | mechanism | evidence | cost | CascadeProto idea |
| :--- | :--- | :--- | :--- | :--- | :--- |
| M0a | noise | EMA of weights over the last LR stage, evaluated as `last_ema` | E1 swings 71–75; R2 note | ~0 (flag) | — |
| M0b | shift | transductive batch statistics (P5-C) | §1.5 | 0 | — |
| M1 | E-a | condition-balanced training: each training query block sampled uniformly with probability p (supports unchanged), so the dense-support → background-density-query pairing of the test's other condition is trained | COSeg Tab.1: the over-sampling is worth 24–27 mIoU to old-protocol methods at 2,048 points | 1 run, 2.3 GPU-h | — (data) |
| M2 | E-a | dual-condition prototypes: sparse support views as extra "ways" of the unchanged inherited modules (rows [bg, dense₁…N, sparse₁…N], class logit = max or logsumexp of its two rows), trained on the same views of base classes | P5-E; PMMs +2.70 (multi-prototype) | 1 run, ~1.3× compute | condition-aware cascade rows |
| M3 | E-b | presence estimation per (query block, class): a small head on pooled posterior and prototype-similarity statistics trained with BCE on base episodes (the other way's class is present or absent naturally), or block-level prior EM (Saerens et al. 2002) | P5-A cf-b | post-hoc on frozen E1: minutes | entropy of the block posterior as evidence |
| M4 | E-c | trained posterior-feedback cascade: stage t builds class-conditional query statistics with the previous stage's posterior and entropy weights, `q_c = Σ_i w_i r_ic f_i / Σ_i w_i r_ic`, `w_i = 1 − H(r_i)/ln(N+1)` (r stop-grad), feeds `G_q,c` to PEM in place of the class-shared `G_q`, zero-initialised gate, per-stage CE | IPMT +2.7 (trained iterations), DPA +3.05, QGE +5.09; D-26 was frozen | 1 run | the paper's "entropy-aware purification" and "cascade", on the right variable |
| M5 | E-d | BAM-type base learner: auxiliary head (base classes + other) on raw base labels of training blocks, ensembled into the background logit with a learned weight | BAM +6.80 (2-D); COSeg BPC +3.44 (3-D, new); D-27 training-free AUC 0.65–0.72 | 1 run | — |
| M6 | E-a (floor, ceiling) | a density-invariant modality from the unused metric xyz: height above the block's lowest point, \|n_z\| and planarity from fixed-radius PCA; class-conditional histograms from the support; fused as a log-likelihood ratio weighted by its support IoU (MM-FSS's TACC) and by the head's posterior entropy | TACC +1.9 / +2.0; floor and ceiling are defined by exactly these attributes | training-free probe, minutes | "multimodal" with per-point information |
| M7 | evaluation | the leak-free draw (P5-D) reported next to fixed100 for every arm | COSeg Tab.1 | +10 min per arm | — |

Order after P5: M0a on every run; M0b if P5.3; then whichever of M1/M2 (P5.1), M3 (P5.2) has the larger
counterfactual; M4 as the cascade contribution; M5 before any S0 claim (S0's beam/board/bookcase gaps are not
condition errors, §0.4); M6 as a cheap probe any time. S0 has never been run for route B and must be before any claim.

### 5.1 Why M1 is not "exploiting the leak"

M1 trains the model to recognise classes *without* the density cue; it cannot increase the use of the cue. Its risk
is the opposite: less reliance on density may cost own-condition points. P5-D and M7 report the leak-free level, so
a gain that came from the cue would show up as a gain on fixed100 without one on the leak-free draw. M3 needs the same
check, because a presence head could learn "densest class = present".

## 6. On the 80+ target

* The other condition holds 11.8 % of S1's foreground query points (floor 19 %, wall 24 %). If E-a on floor and wall
  were removed at their FP level, their IoUs would rise by about 7 points each [inferred from E1's per-class counts
  under an assumed own/other recall split, not measured] → about +2.4 mIoU. E-b is unknown until P5-A. Trained
  self-support (M4) is worth +2 to +3 in comparable settings. A plausible S1 range is therefore 76–78 on `last` and
  78–80 on best-of-validation (not measured).
* 80 on `last` would need most of the oracle gap closed without presence information, which no evidence supports
  yet. The oracle's 85.9 itself includes presence (§1.1); a presence-fair oracle is expected to be lower.
* EDS-Net, the same author's CascadeProto-like model, measures +1.08 Avg for the three ideas together on this
  protocol [verified]. A paper that diagnoses the two conditions, shows where VIP-Seg-type heads fail, and improves
  both the standard protocol and the leak-free draw is a stronger contribution than a larger S1 number.

## Sources

* Repository: `models/vipseg.py:235-310,338-405` (PEM/PDM), `models/encoder.py:136,182-189,230,269,317,320,483,645`,
  `models/vipseg_backbone.py:70-98`, `experiments/r2_distill_eval.py:112-125`, `models/oracle_distill.py:34-50`,
  `experiments/p4_background_probe.py:97-104`, `dataloaders/loader.py:31-89`.
* Results: `results/c1/c1_S0.json`, `results/c1/c1_S1.json` (new), `results/phase16_p0/test_vipseg_S0.json`,
  `test_vipseg_S1.json`, `results/phase16_e1/`, `results/phase16_p3/`, `results/phase16_p4/`, `results/phase16_n2/`.
* Literature: COSeg arXiv 2403.00592 (Tab.1–3, App. A); MM-FSS arXiv 2410.22489 (Eq.9–10, Tab.1, Tab.3e, App. B); GFS-VL
  arXiv 2503.16282 (no S3DIS results); EDS-Net AAAI 2026 (Tab.5); DPA arXiv 2401.16051 (Tab.3); Seg-NN/Seg-PN arXiv
  2404.04050; QGE arXiv 2308.03177; IPMT arXiv 2210.06780; SSP arXiv 2207.11549; BAM arXiv 2203.07615 (Tab.4); PMMs
  arXiv 2008.03898; TIM arXiv 2008.11297; LaplacianShot arXiv 2006.15486; PT-MAP arXiv 2006.03806; Saerens,
  Latinne, Decaestecker, Neural Computation 2002 (prior adjustment by EM); Pezeshki et al., "Gradient Starvation",
  NeurIPS 2021; Bronskill et al., "TaskNorm", ICML 2020.
