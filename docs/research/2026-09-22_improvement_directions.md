# Improving CascadeProto past the reproduction: a mathematical diagnosis and a ranked plan

Status: 2026-09-22. Desk research against primary sources plus CPU-only numerical checks of the repository's own
modules (Appendix A). No GPU run and no code change. Numbers of our own runs are quoted from
`2026-09-21_reproduction_report.md` (below: *report*), `2026-09-21_gap_diagnosis.md`,
`2026-09-21_external_sources_on_gap.md` (*external note*), `docs/CHANGELOG.md` and the training logs under `results/`.
Everything said about what a new design will score is a **prediction**, marked as such, with a decision rule fixed
before the run.

## Verdict

* **The paper's absolute level is not a legitimate target.** Under the protocol the paper states, the best published
  S3DIS 2-way 1-shot results are 73.32 on S0 (EDS-Net, by CascadeProto's first author [EDS T2]) and 76.09 on S1
  (VIP-Seg, report §3.4). The paper's 82.72 (baseline) and 88.53 (full model) match scoring on classes seen in
  training (report §3.6). "Close to the paper" can only mean its relative claims, and the one worth having is the
  qualitative one: that CascadeProto's ideas improve on the VIP-Seg level. **The realistic goal is to beat VIP-Seg
  72.20 / 76.09 and EDS-Net 73.32 / 74.67 (S0 / S1) under the standard protocol.** Our full model is at 57.15 on S0.
* **The printed EPPM cannot close the 15-point gap to VIP-Seg, for structural reasons, most of them exact** (§3). Every
  summand Eq.19 adds to a prototype is either common to all classes, which provably cannot change a prediction, or a
  convex mixing of the class's own channels driven by label-free statistics. The entropy gate is a fixed, even,
  pointwise function of the prototype value with one scalar. No stage reads the previous stage's prediction, and at
  initialisation a stage is numerically close to an isometry on LayerNorm'd input, so Eq.28's contraction does not
  hold and depth has nothing to add. Eq.9 fuses text and points with a fixed 1:1 weight that training calibrates on
  the base classes, where the text prototype becomes almost equal to the point prototype, and that applies unchanged
  to novel names.
* **The active ingredient of the modules this paper descends from is known** (§4). Seg-PN's QUEST, TaylorSeg's APP and
  VIP-Seg's PEM/PDM are the same code (checked token by token), and their own ablations put the gain in the channel
  cross-correlation branch (+15.4 and +15.2 on S0 alone). In their official code that branch is computed after a
  reshape that makes the correlation **shared by all class rows of the episode**: every correlation matrix depends on
  both queries and on all three class slots (checked by differentiation). DPA writes the shared form explicitly, one
  correlation with the mean of all support features, and measures +13.7 for it alone [DPA T3]. Our D-01 computes one
  correlation *per class slot* from whole, unmasked support blocks; D-01 itself records that a single `S′` from all
  support blocks is the more literal reading of Eq.13 (`00` D-01 item 5). **This is the leading hypothesis for the gap.
  Testing it costs one flag and about seven GPU-hours** (R1, §6.2).
* **CascadeProto's ideas can be given a sound mathematical form** (§5): entropy-aware purification as entropy-weighted,
  class-selective aggregation of query points (a MAP-EM step under a von Mises–Fisher model with the support/text
  prototype as prior); the cascade as unrolled EM with per-stage supervision and a training-only oracle-prototype
  target; diffusion as label propagation, or as the cross-class shift that Eq.15's channel means support; cross-modal
  enrichment as a text prior whose weight is validated on the labelled support. Each has primary-source evidence of
  +1 to +5 points in few-shot segmentation, and the first can be tested on existing checkpoints with zero training.
* **Ranked directions, predicted S0 effect under the standard protocol** (§5.8): (1) episode-shared correlation and a
  stripped stage: 57 → 64–71; (2) entropy-aware feedback stages: +1 to +4 on top; (3) base-class calibration: +0.5 to
  +2; (4) the text prior done right: +0.3 to +1.5, bounded by theory (§2.1). Beating VIP-Seg on S0 is plausible if (1)
  recovers most of the gap; beating it on S1 (76.09) is harder. If (1) fails, build on VIP-Seg's own head (route B).
* **An independent calibration by the paper's own first author** (§4.6). EDS-Net (AAAI 2026) is CascadeProto's design
  under other names: a noise-conditioned CLIP generator with a GMMN loss, four refinement "generations" and GAP-softmax
  routing. Under the standard protocol it measures **+1.30 on S0 (+1.08 Avg)** for these components on DyPolySeg,
  where CascadeProto's Table 4 credits them with +5.81. Expect about +1 from the paper's ideas in their printed form
  on a strong head. Anything more has to come from the transductive stage (direction 2).

---

## 1. Targets

| S3DIS, 2-way 1-shot, 2,048-point protocol | S0 | S1 | source |
| :--- | ---: | ---: | :--- |
| EDS-Net (AAAI 2026, CascadeProto's first author) | 73.32 | 74.67 | [EDS] Table 2 |
| VIP-Seg, released logs (NeurIPS poster: 73.50 / 74.92) | 72.20 | 76.09 | report §3.4; external note §1 |
| DyPolySeg | 72.02 | 73.82 | [DyPoly] Table 1 |
| DPA (pre-trained DGCNN) | 66.08 | 74.30 | [DPA] Tables 1, 3 |
| VIP-Seg's released checkpoint through our pipeline | 71.97 | — | report §1 |
| ours: baseline / full model | 49.08 / 57.15 | 51.91 / — | report §2, §3.6 |
| paper, Table 4 baseline / full model | 82.72 / 88.53 | 79.83 / 84.53 | [P] Tables 2, 4 |
| ours scored on classes seen in training: baseline / full (`last.pt`) | 77.32 / 81.28 | 71.58 / — | report §3.6 |

* The last row explains the paper's level. **It must never be reported as a few-shot result**: it scores classes the
  model was trained on.
* DPR-Net (ICML 2026, the paper's reference [24]) could not be retrieved. If it reports more, the target moves.
* Every row shares the loader's foreground over-sampling (`dataloaders/loader.py:39-51`), which lifted AttMPTI, QGE
  and QGPA by 23 to 36 points in COSeg's re-training [COSeg §3.2, Table 1]. A claim of beating the state of the art
  here should eventually be repeated in COSeg's setting, where S3DIS S0 2-way 1-shot is at 37–42 (external note §3.2).

## 2. Where a one-shot prototype's error comes from

### 2.1 Support-side and query-side error

Write the class mean as μ_c, the support object's mean as s = μ_c + δ_s, and the query object's own class mean, the
*oracle prototype* the query's labels would give, as q* = μ_c + δ_q, with independent deviations of variance τ² per
channel. A prototype p is effectively scored against q*:

| prototype | E‖p − q*‖² / (Dτ²) | what reduces it |
| :--- | ---: | :--- |
| one support shot, p = s | 2 | — |
| K support shots | 1 + 1/K | more support |
| one shot fused with an unbiased class prior as informative as one shot (e.g. text) | 1.5 | Bayesian shrinkage |
| a perfect class prior, p = μ_c | 1 | nothing more from the support side |
| the query's own class mean, p = q* | 0 | only the query can supply it |

Within-object sampling noise, τ_w²/n with n in the hundreds of points, is negligible. Measurements size the two halves:

* **Support side: 1.5 to 4.3 points.** Four extra shots (1 → 5) are worth +4.28 / +1.45 (S0 / S1) to VIP-Seg (report
  §3.4, released log names) and +3.97 / +1.50 to DyPolySeg [DyPoly T1]. A text prior can at best act like extra
  shots, so its gain is a fraction of this. The measured text gains agree: our LMA +0.57 (report §2), EDS-Net's text
  module +0.38 on a 72.9 model [EDS T5], and on images AM3's gain falls from +8.7 (1-shot) to +0.9 (5-shot) to +0.2
  (10-shot) [AM3 T1], as shrinkage predicts.
* **Query side: about 25 points.** Replacing support prototypes by the query's own class means lifts S3DIS S0 1-way
  1-shot from 66.40 to 93.89 [QGE T1]; in 2-D, 58.2 to 83.0 [SSP T2]. Knowing only the query's foreground proportion
  is worth 11–14 points to RePRI [RePRI §4.1].

**Consequence.** The large lever is query-conditioned adaptation, of the metric (the channel-correlation modules, §4)
or of the prototype itself (transductive aggregation, §4.5). Cross-modal enrichment addresses the smaller term.

### 2.2 What can change a prediction

With logits L_c(f) = ⟨f, p_c⟩ (Eq.23), adding a vector v to every prototype row adds the same ⟨f, v⟩ to every logit
of a point: **argmax and softmax are unchanged**, and only the contrasts p_c − p_c′ matter. A class-common
*multiplicative* channel gate is different: ⟨f, a ⊙ p_c⟩ = ⟨a ⊙ f, p_c⟩ is a query-conditioned diagonal metric, and it
does change predictions. LayerNorm with (γ, β) on the prototypes is such a metric: β is class-common and cancels, γ
reweights channels. Checked in Appendix A: a class-common additive vector changes 0 % of the predictions of 2,048
points, a class-common sigmoid gate changes 33.5 %, and LN(p) gives exactly the argmax of diag(γ) on standardised p.

## 3. Diagnosis of the printed method

### 3.1 Entropy gate (Eq.10–12)

* g(x) = σ(2(θ − h(σ(x)))), h the binary entropy, is **even in x** (|g(x) − g(−x)| ≤ 9e-16), because
  h(σ(−x)) = h(1 − σ(x)). It is a fixed function of |x| with one learnable scalar θ per stage.
* g < 1 always (report §3); at θ = 0.5, g ∈ [0.4046, 0.7310]. The largest contrast it can apply,
  g(∞)/g(0) = σ(2θ)/σ(2θ − 2 ln 2), is 1.81 at θ = 0.5 and tends to 4 only as θ → −∞.
* As a map, x ↦ x·g(x) has slope in (0, 1) on x ≥ 0 (0.40–0.83 at θ = 0.5), so it is a contraction. But its
  attenuation differs by at most the factor g(∞)/g(0) between small and large values, and Eq.21's LayerNorm is
  scale-invariant, so the common part of the attenuation is undone one step later.
* H_i depends on x_i alone and is not the entropy of any distribution the paper describes. "The background is
  heterogeneous" is a property of a *set* of points, which a mean vector does not retain. On a synthetic mixture the
  per-channel "entropy" of a heterogeneous background mean is barely above that of a homogeneous class mean (0.668
  against 0.646 nats).
* On features (`gate_target=features`; F ≥ 0 by D-14), x·g(x) is monotone, so it **commutes exactly with the
  32-point max-pool** that feeds Eq.13 (maximum deviation 0.0): that reading applies a fixed monotone function to the
  64 pooled tokens. This is why D-02's two readings are indistinguishable (t = +0.08, CHANGELOG 15h).
* A fixed channel transform can help under task shift, but the one that does is concave, φ_k(λ) = 1/ln^k(1/λ + 1),
  and boosts weak channels [Luo Eq.1–2]. x·g(x) has the opposite curvature: it suppresses weak channels and
  reinforces the channel emphasis learned on the base classes.

**The quantity with the meaning the paper gives to entropy** is the predictive entropy of a query point's class
posterior, H(r_i) = −Σ_c r_ic log r_ic: it is high exactly where the model cannot tell which class a point belongs to.
§5.2 uses it that way.

### 3.2 Cross-attention (Eq.13–14)

* Every row of A is a softmax, so each channel of P_cross[b,c] = A ψ(P[b,c]) is a convex combination of the channels
  of ψ(P[b,c]) (verified, Appendix A). The operator can reorder and smooth a prototype's channels; it cannot import
  class evidence the prototype does not already hold.
* Its inputs are the pooled tokens of whole support blocks and of the query (`models/eppm.py:105-111`). **The support
  masks enter the cascade only through Eq.3**, and no operation of EPPM aggregates query points selectively by class,
  so the cascade cannot estimate the query-side prototype that §2.1 identifies as the dominant error.
* **Per class slot or shared.** With one matrix A_b shared by all class rows, the contrast of two classes is
  fᵀA_bW(p_c − p_c′): an episode-adaptive bilinear metric applied to the support-derived contrast, adding no
  class-specific term of its own. With one matrix per class slot (D-01) the contrast also contains
  (A_{b,c} − A_{b,c′})ψ(·), where A_{b,c} is the channel co-activation of the query with the *whole block* sampled for
  way c, background included. Class c's prototype is then pulled
  toward the query whenever its support *scene* resembles the query: a class bias that no label supports. D-01 records
  that the maintainer chose class slots on 2026-09-19, and that one `S′` from all support blocks "would follow Eq.13
  more literally" (`00` D-01 item 5). §4.4 shows what the published code computes.

### 3.3 Diffusion (Eq.15–18)

P_diffuse has no class index and is broadcast to every row (D-16), so by §2.2 it can move a prediction only through
the nonlinearities of Eq.20–21 and the background weight 0.8. Training switches it off: its fusion weight goes from
0.47–0.51 to 0.008–0.082 (CHANGELOG 15e). The channel means Eq.15 computes have one sound use, on the *query* side:
BD-CSPN's cross-class-bias shift f̃ = f + (μ_s − μ_q), μ the mean support and query features [BDCSPN Eq.7–8], changes
⟨f̃, p_c⟩ by ⟨μ_s − μ_q, p_c⟩, a class-dependent correction. §5.3 gives it together with a class-aware diffusion.

### 3.4 Fusion, SE, class weights, output (Eq.19–21)

* The per-query fusion weights and the SE gate are common to all classes. The SE gate is multiplicative, so it acts as
  a query-conditioned diagonal metric on the update branch (§2.2), but it is driven by a class-mean of the prototypes,
  not by second-order feature statistics as VIP-Seg's self-correlation gate is.
* Eq.21's LN equalises the row norms of P^t, and its γ is a learned metric. `l2norm_point_proto` alone is worth +3.36
  (report §2), which suggests that about half of the first stage's +7.07 is this effect (an analogy, not a measurement).
* On synthetic post-ReLU features one stage leaves 3.8–4.2 % of the prototype energy in the class-varying part,
  against 14.2–14.9 % in P^0 and 6.9–8.9 % after VIP-Seg's PEM (CHANGELOG 15i). The stage dilutes the only part that
  can change a prediction.

### 3.5 The cascade and Eq.28

* Stage t receives (P^{t−1}, F^s, F^q) and never L^{t−1}; F^s and F^q are identical at every stage. The cascade is a
  function of P^0 and fixed pooled statistics, so stages 2 to T receive no information stage 1 lacked.
* Eq.28 states that "each EPPM applies an entropy gate that acts as a contraction mapping on the noise component",
  under "mild regularity conditions" and without a proof ([P] §3.7). The printed stage has an identity path, Eq.21's
  residual. At initialisation, its Jacobian on LayerNorm'd input, which is what stages 2–4 receive, has singular values
  with median 0.996–1.004 and maximum 1.02–1.03 (three seeds, Appendix A); stages 2–4 composed have spectral norm
  1.00–1.04 (the four-stage variant of the same check). The stage is close to an isometry, not a contraction.
* That is the measured behaviour: going from T = 1 to T = 4 changes the score by −0.17 (report §2).
* **What gives depth a meaning** is an input that changes from stage to stage: the previous posterior. Then each stage
  is one step of an iterative estimator (EM, label propagation, or a gradient step with a learned step size, the
  unrolling of LISTA [LISTA §1]), depth is the number of iterations, and per-stage supervision makes every iterate a
  usable prediction. Even then the gain saturates quickly: soft k-means refinement did not improve beyond one step
  [Ren §3.1.1]. T = 2–3 is the realistic depth.

### 3.6 LMA and GMMN (Eq.4–9)

* **Set-MMD on tiny sets.** The background term compares sets of one sample: MMD = 12 − 2Σ_σ exp(−d²/2σ²), a bounded
  distance (0.99 at d = 2, 10.5 at d = 80). The foreground term treats the N rows as one set and is invariant to
  permuting them: with the two foreground rows of P_modal swapped, L_GMMN is exactly 0, and Eq.9 would then make both
  foreground rows of P^0 identical (Appendix A). Within one episode only cross-entropy distinguishes the matchings.
  Across episodes the same name meets different partners, which forces a consistent map, and PAP3D trains the same
  kind of set-MMD (bandwidths {2, 5, 10, 20, 40, 60}) successfully [PAP3D §III-E Eq.14, §IV-A]. So this is a weak
  signal, not a failure. GMMN matched minibatches of 1,000 samples [GMMN §4.2, §5]; with one or two samples the kernel
  is a bounded pairwise distance rather than a distance between distributions.
* **A fixed 1:1 fusion calibrated on the wrong classes.** P^0 = P_point + P_modal is, up to scale, a 50/50 shrinkage of
  the support prototype toward a name-dependent prior. In every LMA run L_GMMN falls from 0.47–0.84 at epoch 1 to
  0.008–0.026 at epoch 50 (`results/phase15_full/log_phase15_t4rows/*/log_train.txt`,
  `results/phase14_p1/phase14/*/log_train.txt`); for one-sample sets 0.008 corresponds to a distance of about 0.16.
  On the base classes the text prototype therefore ends almost equal to the point prototype, and the network learns
  with P^0 ≈ 2·P_point. On a novel name the text half is an extrapolation of unknown quality, still weighted 1:1.
  The weight that shrinkage calls for depends on how reliable the prior is *for this class*, which only something
  evaluated on the episode, such as the support labels, can tell.
* **Seven anchors.** P_modal depends only on the prompt ("This point cloud represents the {name}." and one fixed
  background sentence, `models/clip_text.py:15,21-23`) and z = 0 at test (D-06). The adapter and generator see six
  base names and the background sentence. PAP3D shows that such a map can transfer: projected CLIP prototypes alone
  give 61.09 zero-shot on S3DIS 2-way [PAP3D Table VII], used *instead of* support prototypes, not added to them,
  and inflated like every number here by the foreground over-sampling.

### 3.7 Logits (Eq.23)

⟨f, p_c⟩ = ‖f‖‖p_c‖cos θ_c. ‖f‖ is common to the classes of a point; ‖p_c‖ is not and acts as a class-dependent
temperature, which is what `l2norm_point_proto` removes (+3.36, report §2). The opposite error is a scale that is too
small: with normalised logits at scale s, cross-entropy over n classes cannot fall below log(1 + (n−1)e^{−ns/(n−1)})
[NormFace Prop.2]. For n = 3 that floor is 0.37 at s = 1 and 0.45 at s = 0.84. The `logit_scale=sqrt_D` runs
plateaued at 0.446–0.463 (CHANGELOG 15h), the signature of an effective scale below 1. Normalised prototypes with a
learned scale remove both errors; at s ≥ 5 the floor is 0.001.

### 3.8 Training labels

The loader labels every query point outside the episode's ways 0 (`dataloaders/loader.py:85-88`). With cvfold 0 the
test classes are beam, board, bookcase, ceiling, chair and column, the training classes door, floor, sofa, table,
wall and window, and clutter is never a way (`dataloaders/s3dis.py:20-31`). Test-class objects inside training blocks
are trained as background, the "feature undermining" that MiningFSS counters by mining latent classes without labels
[Mining, abstract]. At test time base-class objects in the query are false-positive candidates, which BAM and COSeg
suppress with base-class knowledge [BAM, abstract; COSeg §4.3]. Legitimate remedies: latent-class mining and
base-class calibration (§5.6). **Not legitimate: masking test-class points out of the training loss**, which uses
test-class labels.

## 4. What the literature identifies as the active ingredient

### 4.1 One lineage of code

VIP-Seg's PDM is token-identical to Seg-PN's QUEST (`models/quest.py` of the Seg-NN repository), and VIP-Seg's PEM
is TaylorSeg's APP (`models/app.py`) up to the order of two independent statements (comments and docstrings stripped,
tokens compared). DyPolySeg's PCM prints the same two branches [DyPoly Eq.15–21]; its code is not public. EPPM is
the fifth member. The three with public code feed **L2-normalised** MAP prototypes into the module
(`models/vipseg.py:142`) and score with a plain dot product against LayerNorm'd outputs, as Eq.23 does.

### 4.2 The modules side by side

| module | correlation | support side | channel gate | output |
| :--- | :--- | :--- | :--- | :--- |
| EPPM (Eq.13–21, D-01) | softmax(Q′_bᵀS′_c/√72) | one per class slot: whole block c; background = way-mean | σ(2(θ−H(σ(P)))) ⊙ P, pointwise | LN(W_out ReLU(w_cls ⊙ a ⊙ (w₁P_cross + w₂P_diffuse)) + P) |
| QUEST = PDM | softmax(reshape(que)ᵀ reshape(sup)/√128) | all slots and queries mixed by the reshape | σ(r(G_q − G_s)/√128) ⊙ ψ(P) | LN(fc(P_cross + P_self) + P) |
| APP = PEM | same | same | LN(fc(σ(r_q(G_q)) ⊙ ψ(P)) + fc(σ(r_s(G_s)) ⊙ ψ(P))) | LN(fc(P_cross + P_self) + P) |
| PCM | softmax(F_qᵀF_s/√D) | F_s = MaxPool of the support set | σ(U G)/√D ⊙ V | V_self + A_inter + V |
| DPA, PR | softmax(f_qᵀ f̄_s) | f̄_s = mean of **all** N·K support features | — | p + A(p) |

G = Q′ᵀQ′ is the channel covariance of the projected pooled tokens; r is a Linear(D→1). Sources: `models/eppm.py`,
`models/vipseg.py:197-404`, [SegPN Eq.7–11], [Taylor Eq.9–16], [DyPoly Eq.15–21], [DPA Eq.4–5].

### 4.3 Component ablations (S3DIS 2-way 1-shot unless noted)

| paper, table | from → to | S0 | S1 | mean |
| :--- | :--- | ---: | ---: | ---: |
| Seg-PN T6 | no QUEST (47.32 / 50.05 / 48.68) → cross-correlation only | 62.72 | 67.16 | 64.94 |
| Seg-PN T6 | → cross + self (QUEST) | 64.84 | 67.98 | 66.41 |
| TaylorSeg T4 | no APP (49.42 / 52.67 / 51.05) → CAP (cross) only | 64.64 | 68.54 | 66.59 |
| TaylorSeg T4 | → APP (ACP alone: 51.02 / 55.23 / 53.13) | 67.12 | 71.11 | 69.12 |
| DyPolySeg T5 | no PCM (49.17 / 52.32 / 50.75) → IEM (cross) only | 68.44 | 71.66 | 70.05 |
| DyPolySeg T5 | → SEM (self) only 70.15 / 71.03 / 70.59; both | 72.02 | 73.82 | 72.92 |
| DPA T3 | ProtoNet (50.45 / 51.12 / 50.79) → + PR (shared correlation) | 60.42 | 68.55 | 64.49 |
| DPA T3 | → + prototype-to-query attention | 64.35 | 70.72 | 67.54 |
| DPA T3 | → + stage self-distillation | 66.08 | 74.30 | 70.19 |
| PAP3D T1 | ProtoNet (52.17 / 57.85 / 55.01) → + QGPA channel attention | 56.77 | 61.19 | 58.98 |
| QGE T5, 1-way | AttMPTI 66.27 → + background adaptation 71.36; + oracle-prototype distillation 69.54; both | 74.30 | — | — |
| ours, report §2 | baseline 49.08 → T = 1 (includes Eq.21's LN) | 56.72 | — | — |

* The correlation branch alone is worth +13.7 to +19.3 on the mean in every paper that isolates it. The self branch is
  worth +1.6 to +1.7 on S0 in Seg-PN and TaylorSeg but +21.0 in DyPolySeg, so the family's ablations disagree with
  each other and cannot rank the ingredients by themselves; only a controlled run in our loop can (R1).
* Seg-PN's encoder is non-parametric and runs under `torch.no_grad()` (`models/seg_pn.py` of its repository); only a
  feature head and QUEST are trained, and QUEST still adds 17.5 points on S0. The gain is a property of the head, not
  of an encoder co-adapting with it, which agrees with report §3.2 (MAP on VIP-Seg's own encoder: 47.37).

### 4.4 What the reshape computes

In QUEST, APP, PEM and PDM the correlation is `que.reshape(72, -1)ᵀ @ sup.reshape(72, -1)`, reshaped back to
`[B, W, 128, 128]` (`models/vipseg.py:285-291,387-391`). D-01 rejected it as an interleaving of classes and filters.
Differentiating each output matrix with respect to each input (Appendix A; B_q = 2 queries, W = 3 slots, the 2-way
episode of spec 02 §0): **every A[b′, c′] depends on both queries and on all three class slots**, through the same
pairing for every (b′, c′). The 72 summed projection rows pair (query 0, background slot) 24 times, (query 0, slot 1)
12 times, (query 1, slot 1) 12 times and (query 1, slot 2) 24 times; b′ and c′ only choose which rows enter. The
published modules therefore apply to each class row an *episode-level* correlation, six row-subsampled views of one
statistic, not a per-class one. Whatever the authors intended, the gains of §4.3 were measured on this class-shared,
query-mixed operator, and DPA obtains +13.7 with the shared form written cleanly. D-01's per-slot form is the one
member of the family whose correlation is class-specific.

### 4.5 Transductive and cross-modal evidence

* **Class-selective aggregation of query points works.** Soft k-means refinement [Ren Eq.4]; confidence-weighted
  pseudo-label prototypes [BDCSPN Eq.5–6]; self-support prototypes from confident query regions (thresholds 0.7 for
  foreground, 0.6 for background), +3.1 / +4.0 on PASCAL 1- / 5-shot [SSP §3.2, Table 13]; DPA's prototype-to-query
  attention, +3.05 [DPA T3]; AttMPTI's label propagation Z = (I − αS)⁻¹Y over prototypes and query points
  [AttMPTI Eq.5], MPTI 52.27 against ProtoNet 48.39 [AttMPTI T1].
* **Entropy is the right regulariser for it, with a marginal term.** TIM maximises I(X;Y) = H(Y) − H(Y|X) on the
  query and reaches 73.6 against SimpleShot's 62.6 in mini-ImageNet 1-shot [TIM T1]. Without H(Y), conditional-entropy
  minimisation assigns all queries of many tasks to one class, and 1-shot accuracy falls from 60 (cross-entropy alone)
  to 35–36 [TIM §3.4, Table 4]. For segmentation, where background dominates, RePRI replaces H(Y) by a KL term toward
  an estimated foreground proportion [RePRI Eq.1].
* **Training-only query labels give a purification target.** QGE distils support prototypes toward the "optimal query
  prototypes" (KL on logits), +3.3 [QGE T5]; DPA distils early stages toward late ones, +2.65 [DPA T3].
* **Text works as a gated prior.** AM3 mixes p′_c = λ_c p_c + (1 − λ_c) w_c with λ_c = σ(h(w_c)) [AM3 §3.2]. MM-FSS
  weighs its text branch by γ, the IoU the text-only prediction achieves on the labelled support [MMFSS Eq.9–10]:
  +1.9 / +2.0 over no text calibration, where fixed weights give +0.1 to +0.9 [MMFSS T3e]. Text adds +3.3 / +3.7 in
  MM-FSS's setting [MMFSS T3d] and +0.38 in EDS-Net [EDS T5].
* **Base-class calibration** (EMA prototypes of the base classes, momentum 0.995, raise the background score where the
  query matches them): +3.44 [COSeg §4.3, Table 3].

### 4.6 The closest published sibling: EDS-Net

EDS-Net [EDS], by CascadeProto's first author, is CascadeProto's architecture under other names. It is measured under
the standard protocol, on DyPolySeg's backbone with VIP-Seg's schedule (AdamW 1e-3 halved every 7,000 iterations,
one episode per batch, G = 4) [EDS §4.2]:

| CascadeProto | EDS-Net |
| :--- | :--- |
| LMA generator `G([E_CLIP; z])`, noise z (Eq.6) | semantic projection network `P_fake = SPN(F_sem, z)`, CLIP 512-d, noise z [EDS Eq.13, §4.2] |
| `L_GMMN` between modal and point prototypes (Eq.7–8) | "domain bridging loss" `L_GMMN(V_visual, P_fake)`, a squared MMD [EDS Eq.19–20] |
| T = 4 EPPM stages (Eq.22) | G = 4 "generations" of `V^(g+1) = V_self + A_inter + V^(g)`, PCM's two branches [EDS Eq.12, 21] |
| ADRM `softmax(W_g AvgPool(F^q))` over stage logits (Eq.24–25) | `α^(g) = softmax(FC(GAP(F_q)))` over generation prototypes [EDS Eq.22–23] |

With a linear readout, weighting prototypes (EDS-Net) and weighting logits (CascadeProto) give the same prediction.
EDS-Net's module ablation [EDS Table 5, S3DIS 2-way 1-shot, S0 / S1 / Avg]:

| configuration | S0 | S1 | Avg |
| :--- | ---: | ---: | ---: |
| no SEM, VSBM or Multi-Gen (= DyPolySeg, its Table 1 row) | 72.02 | 73.82 | 72.92 |
| + SEM only | 72.85 | 74.31 | 73.58 |
| + VSBM (the text branch) only | 72.47 | 74.12 | 73.30 |
| + Multi-Gen (depth) only | 72.34 | 74.05 | 73.20 |
| all three | 73.32 | 74.67 | 74.00 |

Its Fig. 2 puts the Avg at 72.92 with one generation, 74.00 with four, 73.67 with five.

* **This is the empirical size of the paper's ideas under the honest protocol.** The same author implements the same
  three ideas (noise-conditioned CLIP generator with a GMMN loss, a four-step cascade, GAP-softmax routing) on a
  72-level head and measures **+1.30 on S0 and +1.08 on the Avg**. CascadeProto's Table 4 credits the same components
  with +5.81 on S0. EDS-Net is therefore an independent calibration, by the paper's own first author, and it agrees
  with our measurements (LMA +0.57, depth −0.17) rather than with the paper.
* **What it means for §5.8.** Route B's increments should be expected near EDS-Net's +1 for the paper's ideas in their
  printed form. The upper end of route B (77) assumes the transductive stage of §5.2 adds what EDS-Net does not have.
  Only P0.2 and R2 can show whether it does.

## 5. A redesign that keeps CascadeProto's four ideas

Shapes for one episode: F^q [B_q, 2048, D], F^s [N, K, 2048, D], D = 128, d = 72; f̂ = f/‖f‖.

### 5.1 Stage 0: episode-shared channel correlation ("EPPM-S")

```text
F̃q_b = MaxPool32(Fq_b)                      [64, D]      F̄s = (1/NK) Σ_{n,k} MaxPool32(Fs_{n,k})   [64, D]
Q′_b = φ(F̃q_b),  S̄′ = φ(F̄s)                 [d, D]       A_b = softmax_row(Q′_bᵀ S̄′ / √d)          [D, D]
P_cross[b,c] = A_b ψ(p̂_c)                   [D]          p̂_c = P_point,c / ‖P_point,c‖
P_self[b,c]  = σ(r(Q′_bᵀQ′_b − S̄′ᵀS̄′)/√D) ⊙ ψ(p̂_c)       [D]
P¹[b,c]      = LN(W(P_cross + P_self) + p̂_c)                [D]
```

* This is Eq.13 with a single S′, the literal option D-01 records (§3.2), in the form DPA states and the published code
  effectively computes (§4.4), without the reshape's mixing of queries: each query keeps its own A_b.
* Removed from EPPM: P_diffuse (class-common, §3.3), the pointwise entropy gate (§3.1), the ReLU before W_out, the
  class-pooled SE and w_cls. Added: the L2 input of VIP-Seg (`models/vipseg.py:142`) and QUEST's channel-preserving
  term.
* Any class-specific statistic should be computed from the **masked** support points of class c, so that it rests on a
  label, never from a whole block.
* Parameters: φ 4,608; ψ 16,512; r 128; W 16,384; LN 256; about 37.9 k per module, against 79,395 per EPPM stage
  (CHANGELOG 15i).

### 5.2 Entropy-aware purification with feedback: the principled EPPM

For stage t = 1..T, from the previous logits L^{t−1} [B_q, 2048, N+1] and prior prototypes m_c (stage 0, or support
fused with text, §5.5):

```text
E-step    r_ic   = softmax_c(L^{t-1}_ic / T_t)             [B_q, 2048, N+1]
weights   w_i    = 1 − H(r_i) / ln(N+1)  ∈ [0, 1]           [B_q, 2048]
M-step    u_c    = (1/2048) Σ_i w_i r_ic f̂_i                 [B_q, N+1, D]
update    μ^t_c  = normalise(κ0 m_c + κ_t u_c)              [B_q, N+1, D]
logits    L^t_ic = s_t ⟨f̂_i, μ^t_c⟩                          [B_q, 2048, N+1]
```

* **Derivation.** Model f̂_i as a mixture of von Mises–Fisher components with a common concentration and mean
  directions μ_c, and put a vMF prior on μ_c centred on m_c. The posterior over classes is a softmax of scaled cosines,
  the E-step. The log-posterior of μ_c is κ₀ m_cᵀμ_c + κ Σ_i r_ic f̂_iᵀμ_c + const, maximised on the unit sphere by the
  normalised sum, the update. Without the weights w_i, and with the E-step temperature tied to the concentration
  (T_t = 1, s_t = κ), this is exact MAP-EM, which never decreases its objective
  [EM, 1977]: the monotonicity Eq.28 claims for the printed stage, which the printed stage does not have. The weights
  make it a robust weighted variant for which that guarantee is no longer exact. Because u_c grows with the confident
  mass Σ_i w_i r_ic, a large confident region pulls harder, as more data should.
* **Entropy where it carries meaning.** w_i down-weights points whose class is uncertain, a soft version of SSP's
  thresholds (which can replace it as an ablation). The background prototype is purified too: μ_0 becomes the query's
  own confident background, the largest single gain in QGE [QGE T5].
* **Against collapse.** Entropy-driven updates can hand everything to the majority class [TIM §3.4]. The prior term
  κ₀m_c keeps each prototype anchored to its support; a class-proportion term log π^t_c with π^t estimated from r is
  RePRI's alternative.
* **Cascade, supervision and routing.** T = 2–3 unrolled steps, each with its own (κ₀, κ_t, T_t, s_t): four scalars per
  stage, the LISTA pattern. Loss Σ_t λ_t CE(L^t, Y^q) + β Σ_t KL(softmax L* ‖ softmax L^t), where L* uses the
  **oracle prototypes** normalise(Σ_{i: y_i = c} f̂_i) of the training query, with a stopped gradient. This is QGE's
  distillation target [QGE T5]; it is legitimate because the labels of training episodes are base-class labels. ADRM
  (Eq.24–25) stays as the weighting over iterates; an entropy-aware variant weights stage t by
  exp(−β · mean_i H(softmax L^t_i)).
* **Zero-training version.** With κ, T and s fixed, the stage runs on any trained checkpoint (P0.2), which makes it the
  cheapest test of whether entropy-aware cascaded purification works at all on these features.

### 5.3 Diffusion, re-derived

* **Label propagation** of the posteriors on the query's k-NN graph: R ← αSR + (1 − α)R⁰ with S = D^{−1/2}WD^{−1/2}
  converges to (1 − α)(I − αS)⁻¹R⁰, and (I − αS)⁻¹ is a diffusion kernel [Zhou §2]. With the paper's α = 0.5, ten
  iterations leave an error below 0.5¹⁰ ≈ 1e-3. No parameters. AttMPTI uses the same operator with α = 0.99
  [AttMPTI §4.2].
* **Cross-class shift** f̃ = f + (μ_s − μ_q), from the channel means Eq.15 already computes [BDCSPN Eq.7–8].

### 5.4 A support-conditioned channel gate (optional)

Per way k and channel d, a Fisher ratio J_d = (μ_fg,d − μ_bg,d)² / (σ²_fg,d + σ²_bg,d + ε) from the masked support
points, and g_d = 1 + a·tanh(b(J_d − median J)): class-specific, grounded in the masks, able to amplify (Eq.11 cannot),
two parameters. Channel emphasis matters under task shift [Luo, abstract] and a gate changes predictions (§2.2), but
there is no 3-D evidence. Predicted 0 to +2.

### 5.5 Cross-modal prior

* Fuse by a weight validated on the episode: λ = IoU of the text-only prediction on the support block against its mask
  (MM-FSS's γ [MMFSS Eq.10]), either on prototypes, m_c = normalise(p̂_c + λ t_c) with t_c = normalise(G(A(E_CLIP,c))),
  or on logits as MM-FSS does, L = L_point + λ L_text. No parameters. AM3's learned λ_c is the alternative (about 8 k
  parameters).
* Keep the moment matching, which transfers in PAP3D, but make it per class: a paired loss between t_c and an EMA
  memory of each base class's point prototype, so the ambiguity of §3.6 disappears; optionally an auxiliary CE with
  text prototypes alone.
* Ensemble several prompt templates for the class names and replace the single background sentence.
* Predicted +0.3 to +1.5: the bound of §2.1, EDS-Net's +0.38, MM-FSS's +2 to +3.7 in another setting.

### 5.6 Base-class calibration

EMA prototypes b_j of the six base classes (momentum 0.995) from the masks of each training episode's ways, leaving out
the current ways as COSeg does; at test the background logit gains ω·max_j cos(f_i, b_j) [COSeg Eq.9–12]. One
parameter plus buffers; only base-class labels are used. Predicted +0.5 to +2 here (+3.44 in COSeg's setting).

### 5.7 Logits

L_c = s·cos(f, p_c) with s learned from 10; NormFace's floor is 1e-6 at s = 10 for n = 3. One parameter.

### 5.8 Two routes and the predicted gains

* **Route A, CascadeProto repaired**: our encoder → 5.1 → 5.2 (+ 5.3) → 5.5 → 5.6. It keeps the paper's structure and
  starts at 57.
* **Route B, CascadeProto on VIP-Seg**: VIP-Seg's own PEM/PDM head, imported read-only from `models/vipseg.py`, as the
  stage-0 purifier, then 5.2–5.6. It starts from the proven 72 (71.97 released; 0.6948 through our loop after 600
  steps).
* R1 decides: if EPPM-S (R1 c) comes within two points of VIP-Seg's module (R1 d), and later of R0 on the full
  schedule, route A keeps the paper's structure at no cost; otherwise take route B. Modules 5.2–5.6 are the same in
  both.

| step (S0, fixed100), predictions | route A | route B | evidence |
| :--- | :--- | :--- | :--- |
| stage 0 | 64–71 | 71–72 | §4.3; report §1 |
| + feedback stages (5.2) | +1 to +4 | +1 to +3 | §4.5 |
| + base-class calibration (5.6) | +0.5 to +2 | +0.5 to +2 | [COSeg T3] |
| + text prior (5.5) | +0.3 to +1.5 | +0.3 to +1.5 | §2.1 |
| total, allowing for sub-additivity | 66–76 | 73–77 | — |

These are ranges, not promises; gains measured on weaker baselines usually shrink on stronger ones. On a 72-level
head, EDS-Net measures +1.08 Avg for the paper's own three ideas (§4.6). Route B's range above that assumes §5.2 works.

## 6. Experiment plan

Ordered by information per GPU-hour. Unless noted: S3DIS S0, 2-way 1-shot, fixed100 (1,500 episodes), the metric of
D-08. Hyper-parameters of test-time procedures are tuned on the **other fold** (the S1 baseline checkpoint, scored on
its own test classes) and applied unchanged to S0, never tuned on the classes being scored.

### 6.1 Zero-training probes (minutes of GPU each)

* **P0.1 Oracle ceiling.** On the baseline, full-model and VIP-Seg released checkpoints, replace the final prototypes
  by the mean of F^q_b over the points with Y^q_b = c (every class present in the query), rescaled to the model's
  prototype norm; also with the oracle background only and the oracle foreground only. *Prediction:* 75–92 on every
  checkpoint (QGE: 93.89 in 1-way) and oracle-background-only ≥ +8. *Rule:* oracle − model ≥ 20 → prototype-side
  purification has headroom, proceed with 5.2; 10–20 → proceed, expecting the low end of §5.8; < 10 → the encoder is
  the bound, drop 5.2 and spend the budget on route B.
* **P0.2 Test-time entropy-weighted EM** (5.2 with κ₀ = 1, κ ∈ {0.25, 0.5, 1}, T ∈ {1, 2, 3, 5}, with and without
  w_i) on the same three checkpoints. *Prediction:* +1 to +4 at T = 1 and at most +1 more for T = 2–3. *Rule:* a T = 1
  gain ≥ 1.5 on at least two checkpoints → train 5.2 (R2); T = 3 − T = 1 ≥ 0.5 → depth is real, otherwise keep T ≤ 2
  and drop the depth claim.
* **P0.3 Label propagation and cross-class shift** (5.3) on the same checkpoints. *Prediction:* 0 to +2. *Rule:* ≥ 1 →
  keep as the diffusion branch.
* **P0.4 Text diagnostic** on the full checkpoint: mIoU with P_modal alone (z = 0), and the fraction of episodes in
  which P_modal of a way's name is nearer to that way's P_point than to the other way's. *Prediction:* assignment
  50–65 % and mIoU well below the baseline. *Rule:* ≥ 70 % and ≥ 35 mIoU → the text transfers, fuse it as in 5.5;
  otherwise 5.5 is expected to add at most one point.
* **P0.5 Is VIP-Seg's trained correlation class-shared?** On the released checkpoint, the mean cosine similarity
  between A[b′, c′] and A[b′, c″] over the class index, and the same for A_{b,c} of our full model. *Prediction:*
  ≥ 0.9 for VIP-Seg, lower for ours. *Rule:* confirms or weakens the reading of §4.4 before any training.

### 6.2 Training runs

* **R0 VIP-Seg through our loop on the full schedule** (3 seeds, about 1.5 h each). The loop has trained it only for
  600 steps (0.6948); route B and R1 need the full-schedule number. This needs a `train.py` path for VIP-Seg's model;
  the diag harness builds it (`experiments/diag_short.py:61-68`) but trains without LR decay.
* **R1 Which difference costs the 15 points** (T = 1, 9,600 episodes = 2,400 steps, 3 seeds each, about 0.55 h per run):
  (a) EPPM as specified; (b) EPPM with only the shared S̄′ of 5.1; (c) EPPM-S, 5.1 complete; (d) one VIP-Seg PEM on our
  prototypes; optionally (e) EPPM with the literal reshape. *Prediction:* (b) − (a) ≥ +3 and (c) within 2 of (d).
  *Rule:* (b) − (a) ≥ 3 with t > 3 → the per-slot correlation of D-01 is a main cause, revise D-01; (b) ≈ (a) but
  (c) ≫ (a) → the cause is in the additions of Eq.19–21, remove ReLU, SE, P_diffuse, w_cls and the raw input one at a
  time; (c) ≈ (a) and (d) ≫ (a) → take route B.
* **R2 Feedback stages** (5.2; T ∈ {1, 2, 3}; with and without the oracle-distillation term) on the R1 winner,
  screening at 2,400 steps, then the full schedule, 3 seeds. *Prediction:* +1 to +4. *Rule:* keep a term only if its
  gain exceeds two seed standard deviations.
* **R3 Base-class calibration** (5.6) and **R4 text prior** (5.5), 3 seeds each (3.3 h at 2,400 steps, 9 h on the full
  schedule). *Predictions:* +0.5 to +2 and +0.3 to +1.5.
* **R5 Final.** The best configuration on S0 and S1, 3 seeds, fixed100 and random600, `best` and `last`, against
  VIP-Seg 72.20 / 76.09 and EDS-Net 73.32 / 74.67, reported as seed means with standard deviations.

### 6.3 Budget and repository

* About 40–45 GPU-hours on the L4: P0 about 1 h, R0 4.5 h, R1 6.6 h, R2 about 11 h, R3 and R4 3.3–9 h, R5 about
  9 h. A full-schedule run is 82 min of training plus validations (4,916 s at epoch 50 for T = 1,
  `results/phase15_full/log_phase15_t4rows/s3dis_S0_N2_K1_text_T1/log_train.txt`).
* The project is closed (AGENTS.md §1); reopening it is the maintainer's decision. If it is reopened, 5.1 is a new
  reading of an ambiguous equation, so it goes through a revision of D-01 with this evidence, then spec 02 §5.2, then
  code, behind a flag (for example `cross_attn_support = {class_slots, pooled}`) whose default keeps the reproduction.
  5.2–5.7 are beyond the paper: new PROPOSED decisions from D-22 on, flags off by default as for D-19, spec lines tagged
  [DECISION D-nn], tests with mutation checks per spec 05, results under `results/phase16_*`, one commit and one
  CHANGELOG entry per sub-phase. VIP-Seg's modules are imported, never edited (AGENTS.md §4, guardrail 2). The probes
  P0.x belong in `experiments/`, next to `vipseg_init_probe.py`.

## 6.4 Outcome of R1 (2026-09-23)

Measured on S1, T = 1, no LMA, 2,400 steps, two seeds (`results/phase16_r1/SUMMARY.md`, CHANGELOG 16f);
these are S1 valid-draw numbers, not the S0 fixed100 scale used elsewhere in this note.

| variant | mean | vs no stage (`baseline_l2` 0.5602) |
| :--- | ---: | ---: |
| the printed stage | 0.5305 | −2.97 |
| + Eq.13's single `S′` (§5.1, the leading hypothesis) | 0.5339 | −2.63 |
| EPPM-S, stripped (§5.1) | 0.4927 | −6.75 |
| one VIP-Seg PEM (route B's stage 0) | **0.6852** | **+12.50** |

* **§4.4's hypothesis is refuted.** The shared correlation is worth **+0.33 points** (t = +0.65), not the
  +13.7 to +15.4 that the family's own ablations credit to that branch. Whatever those ablations
  measured, it is not the support reading on its own.
* **§5.1's prediction (57 → 64–71) fails.** Stripping the stage of the parts §3 proved inert made it
  worse than the stage it replaces and worse than using no stage at all. The proofs in §2.2 and §3
  stand as statements about what those parts *cannot* do; they do not license the conclusion that a
  stage without them works.
* **§5.8's route B is on track.** One VIP-Seg module reaches 0.6852 at 40 % of the schedule against a
  cited 0.7609 for VIP-Seg's S1 checkpoint, and it is +15.47 over the printed stage inside the same
  pipeline (t = +27.78).
* **Not a budget artefact.** The same run resolves +12.50 with t = +7.86, and the repository's
  full-schedule curves move by at most 1.1 points between epoch 20 and epoch 50.
* **What this changes for §5.2–5.6.** The transductive stage, the base-class calibration and the text
  prior are unaffected — they were always meant to sit on top of a working stage 0, and that stage 0 is
  now VIP-Seg's module rather than one of ours. The open question moves to *why* PEM wins (§6.4 of the
  summary lists the three candidates), and the query-mixing candidate would be direct evidence for the
  transductive reading of §5.2.

## 7. Risks, falsifiers, and what not to do

* **Falsifiers.** An oracle below 65 in P0.1 (the encoder, not the prototype, is the limit); R1 with (b) ≈ (a) and
  (d) ≈ (a) (the gap is not in the head at all); P0.2 flat at T = 1 (transductive purification does not help on these
  features).
* **Collapse.** Entropy-driven updates can swallow a small foreground. Watch the per-class IoU, not only the mean, and
  keep the prior term.
* **Foreground over-sampling.** Query-side methods may exploit the loader's denser foreground more than support-side
  ones do. `pipeline/episodes.py:25` fixes the loader's `random_sample` to False; scoring the final model with it on
  would show how much of a gain depends on density.
* **Batch coupling.** The published reshape makes one query's prediction depend on the other query of the episode.
  Keep the per-query form of 5.1. (The encoder already couples blocks within one call, spec 02 §2.)
* **Do not** chase 82–88; report seen-class scores as few-shot results; tune new hyper-parameters on the test classes,
  since the inherited validation set is built from them (D-15); use test-class labels in training; add stages or
  capacity to the printed EPPM (its stage Jacobian is already close to the identity and T = 1 → 4 gives −0.17);
  strengthen the GMMN (it already converges, and its optimum makes text redundant on the base classes); judge
  differences below two points on one seed, or at budgets below 2,400 steps (CHANGELOG 15k).

## Appendix A. CPU checks

Run from the repository root with `PYTHONPATH=. .venv/Scripts/python` (torch 2.5.1, float64). The script imports the
repository's modules read-only and writes nothing.

```python
import math, torch
from models.eppm import EntropyGate, CrossAttention, EPPMStage, pool_tokens
from loss.gmmn_loss import gmmn_loss
torch.set_default_dtype(torch.float64); torch.manual_seed(0)
g = EntropyGate().double(); x = torch.linspace(-12, 12, 24001)
with torch.no_grad():
    print("gate even:", float((g.gate(x) - g.gate(-x)).abs().max()), "range:", float(g.gate(x).min()), float(g.gate(x).max()))
    print("g(inf)/g(0) at theta=0.5, -6:", [float(torch.sigmoid(torch.tensor(2*t)) / torch.sigmoid(torch.tensor(2*t - 2*math.log(2)))) for t in (0.5, -6.0)])
    F = torch.relu(torch.randn(3, 2048, 128))
    print("gate commutes with MaxPool on F>=0:", float((pool_tokens(g(F)) - g(pool_tokens(F))).abs().max()))
fq, P = torch.relu(torch.randn(2048, 128)), torch.randn(3, 128)
base = (fq @ P.t()).argmax(-1)
print("argmax changed by class-common add:", float(((fq @ (P + 5 * torch.randn(128)).t()).argmax(-1) != base).double().mean()),
      "by class-common gate:", float(((fq @ (P * torch.sigmoid(2 * torch.randn(128))).t()).argmax(-1) != base).double().mean()))
xa, fs, fq2, Pb = CrossAttention().double(), torch.relu(torch.randn(2, 1, 2048, 128)), torch.relu(torch.randn(2, 2048, 128)), torch.randn(2, 3, 128)
with torch.no_grad():
    pc, v = xa(Pb, fs, fq2), xa.psi(Pb)
print("P_cross inside channel range of psi(P):", bool(((pc <= v.amax(-1, keepdim=True) + 1e-9) & (pc >= v.amin(-1, keepdim=True) - 1e-9)).all()))
Pt = torch.randn(3, 128); swap = Pt[[0, 2, 1]]
print("L_GMMN with fg rows swapped:", float(gmmn_loss(swap, Pt)), "P0 fg rows equal:", bool(torch.allclose((Pt + swap)[1], (Pt + swap)[2])))
que, sup = torch.randn(2, 72, 128, requires_grad=True), torch.randn(3, 72, 128, requires_grad=True)
A = (que.reshape(72, -1).t() / 128 ** 0.5 @ sup.reshape(72, -1)).reshape(2, 128, 3, 128).permute(0, 2, 1, 3)
for b, c in ((0, 1), (1, 2)):
    gq, gs = torch.autograd.grad(A[b, c].sum(), (que, sup), retain_graph=True)
    print(f"reshape A[{b},{c}] depends on queries", [int(gq[i].abs().sum() > 0) for i in range(2)], "slots", [int(gs[i].abs().sum() > 0) for i in range(3)])
for seed in range(3):
    torch.manual_seed(seed); st = [EPPMStage().double().eval() for _ in range(2)]
    fs, fq3 = torch.relu(torch.randn(2, 1, 2048, 128)), torch.relu(torch.randn(2, 2048, 128))
    m = (torch.rand(2, 1, 2048) < 0.3).double()
    p0 = torch.cat([(torch.einsum("nkp,nkpd->d", 1 - m, fs) / (1 - m).sum())[None],
                    torch.einsum("nkp,nkpd->nd", m, fs) / m.sum(dim=(1, 2))[:, None]])[None].expand(2, -1, -1).contiguous()
    with torch.no_grad(): p1 = st[0](p0, fs, fq3)
    sv = torch.linalg.svdvals(torch.autograd.functional.jacobian(lambda p: st[1](p, fs, fq3), p1).reshape(768, 768))
    print(f"seed {seed}: stage-2 Jacobian singular values median {float(sv.median()):.3f} max {float(sv[0]):.3f}")
```

Output:

```text
gate even: 8.881784197001252e-16 range: 0.4046096848277171 0.7310271768559261
g(inf)/g(0) at theta=0.5, -6: [1.8068242641099854, 3.999981567476194]
gate commutes with MaxPool on F>=0: 0.0
argmax changed by class-common add: 0.0 by class-common gate: 0.33544921875
P_cross inside channel range of psi(P): True
L_GMMN with fg rows swapped: 0.0 P0 fg rows equal: True
reshape A[0,1] depends on queries [1, 1] slots [1, 1, 1]
reshape A[1,2] depends on queries [1, 1] slots [1, 1, 1]
seed 0: stage-2 Jacobian singular values median 0.996 max 1.022
seed 1: stage-2 Jacobian singular values median 1.004 max 1.030
seed 2: stage-2 Jacobian singular values median 0.997 max 1.032
```

A longer version of the same script also gave: slope of x·g(x) on x ≥ 0 in [0.40, 0.83] at θ = 0.5; the swapped
L_GMMN is 5.87 with `fg_mode=per_class`; MMD between one-sample sets 0.99 at distance 2 and 10.52 at distance 80;
per-channel Eq.10 "entropy" 0.646 nats for the mean of one synthetic class against 0.668 for the mean of seven; stages
2–4 composed have spectral norm 1.00–1.04. The code-identity claims of §4.1 come from tokenising the downloaded
`models/quest.py` (Seg-NN) and `models/app.py` (TaylorSeg) and the vendored `models/vipseg.py`, stripping comments and
docstrings, and comparing the token streams.

## Sources

Repository: `docs/research/2026-09-21_reproduction_report.md`, `2026-09-21_gap_diagnosis.md`,
`2026-09-21_external_sources_on_gap.md`, `docs/spec/00_SOURCES_AND_DECISIONS.md` (D-01, D-02, D-10, D-15, D-16, D-19),
`docs/spec/02_TENSOR_MATH_SPEC.md`, `docs/CHANGELOG.md` (15e, 15h, 15i, 15k), `models/eppm.py`, `models/vipseg.py`,
`models/clip_text.py`, `loss/gmmn_loss.py`, `dataloaders/loader.py`, `dataloaders/s3dis.py`, `pipeline/episodes.py`,
`experiments/diag_short.py`, and the training logs under `results/phase14_p1/` and `results/phase15_full/`.

* [P] Wang et al., *CascadeProto*, local copy `10069.pdf`: §3.3–3.7 (Eq.1–28), Tables 2, 4, 5, 6. Official repository
  https://github.com/changshuowang/CascadeProto (README only).
* [SegPN] Zhu et al., *No Time to Train: Empowering Non-Parametric Networks for Few-shot 3D Scene Segmentation*,
  https://arxiv.org/abs/2404.04050 (§3.2, §4 Eq.7–11, Tables 1, 4, 6); code https://github.com/yangyangyang127/Seg-NN
  (`models/quest.py`, `models/seg_pn.py`).
* [Taylor] Wang et al., *TaylorSeg*, AAAI 2025, https://arxiv.org/abs/2504.02454 (§3.4 Eq.9–16, Tables 1, 4); code
  https://github.com/changshuowang/TaylorSeg (`models/app.py`, `models/seg_pn.py`).
* [DyPoly] Wang, Fang, Tiwari, *DyPolySeg*, ICML 2025,
  https://raw.githubusercontent.com/mlresearch/v267/main/assets/wang25aa/wang25aa.pdf (§3.4 Eq.15–21, Tables 1, 3, 5).
  Code repository https://github.com/changshuowang/DyPolySeg not found.
* [VIP] VIP-Seg, https://github.com/changshuowang/VIP-Seg_NeurIPS2025, vendored as `models/vipseg.py`.
* [EDS] Wang et al., *EDS-Net: Biologically-Inspired Evolutionary Domain Symbiosis for Few-shot and Zero-shot Point
  Cloud Semantic Segmentation*, AAAI 2026, https://ojs.aaai.org/index.php/AAAI/article/view/37929, PDF
  https://ojs.aaai.org/index.php/AAAI/article/view/37929/41891 (Eq.12–23, §4.2, Tables 2, 5, Fig. 2).
* [DPA] Liu et al., *Dynamic Prototype Adaptation with Distillation for Few-shot Point Cloud Segmentation*, 3DV 2024,
  https://arxiv.org/abs/2401.16051 (Eq.4–8, §4.2, Tables 1, 3).
* [PAP3D] He et al., *Prototype Adaption and Projection for Few- and Zero-shot 3D Point Cloud Semantic Segmentation*,
  IEEE TIP 2023, https://arxiv.org/abs/2305.14335 (Eq.4–5, 14; §III-E, §IV-A; Tables I, VII).
* [QGE] Ning et al., *Boosting Few-shot 3D Point Cloud Segmentation via Query-Guided Enhancement*,
  https://arxiv.org/abs/2308.03177 (Tables 1, 3, 5).
* [AttMPTI] Zhao, Chua, Lee, *Few-shot 3D Point Cloud Semantic Segmentation*, https://arxiv.org/abs/2006.12052
  (§3.2.3 Eq.5, §4.2, Table 1).
* [COSeg] An et al., *Rethinking Few-shot 3D Point Cloud Semantic Segmentation*, CVPR 2024,
  https://arxiv.org/abs/2403.00592 (§3.2, §4.3 Eq.9–12, Tables 1, 3).
* [MMFSS] An et al., *Multimodality Helps Few-shot 3D Point Cloud Semantic Segmentation*, ICLR 2025,
  https://arxiv.org/abs/2410.22489 (§3.5 Eq.9–10, Table 3).
* [SSP] Fan et al., *Self-Support Few-Shot Semantic Segmentation*, https://arxiv.org/abs/2207.11549 (§3.2, Tables 2, 13).
* [RePRI] Boudiaf et al., *Few-Shot Segmentation Without Meta-Learning: A Good Transductive Inference Is All You Need?*,
  https://arxiv.org/abs/2012.06166 (Eq.1, §4.1, Tables 1–2).
* [TIM] Boudiaf et al., *Transductive Information Maximization for Few-Shot Learning*, https://arxiv.org/abs/2008.11297
  (Eq.3, §3.4, Tables 1, 4).
* [BDCSPN] Liu et al., *Prototype Rectification for Few-Shot Learning*, https://arxiv.org/abs/1911.10713 (Eq.4–8).
* [Ren] Ren et al., *Meta-Learning for Semi-Supervised Few-Shot Classification*, https://arxiv.org/abs/1803.00676
  (§3.1.1, Eq.4).
* [AM3] Xing et al., *Adaptive Cross-Modal Few-Shot Learning*, https://arxiv.org/abs/1902.07104 (§3.2, Table 1).
* [Luo] Luo et al., *Channel Importance Matters in Few-Shot Image Classification*, https://arxiv.org/abs/2206.08126
  (Eq.1–2, abstract).
* [Mining] Yang et al., *Mining Latent Classes for Few-shot Segmentation*, https://arxiv.org/abs/2103.15402 (abstract).
* [BAM] Lang et al., *Learning What Not to Segment: A New Perspective on Few-Shot Segmentation*,
  https://arxiv.org/abs/2203.07615 (abstract).
* [NormFace] Wang et al., *NormFace: L2 Hypersphere Embedding for Face Verification*, https://arxiv.org/abs/1704.06369
  (Proposition 2).
* [GMMN] Li, Swersky, Zemel, *Generative Moment Matching Networks*, https://arxiv.org/abs/1502.02761 (§4.2, §5).
* [Zhou] Zhou et al., *Learning with Local and Global Consistency*,
  https://proceedings.neurips.cc/paper_files/paper/2003/file/87682805257e619d49b8e0dfdc14affa-Paper.pdf (§2).
* [LISTA] Gregor, LeCun, *Learning Fast Approximations of Sparse Coding*, ICML 2010,
  https://icml.cc/Conferences/2010/papers/449.pdf (§1).
* [EM] Dempster, Laird, Rubin, *Maximum Likelihood from Incomplete Data via the EM Algorithm*, JRSS-B 1977; standard
  result, not re-read for this note.

Not retrieved or not consulted: DPR-Net (ICML 2026, the paper's [24]); Hu et al., TCSVT 2023 (the paper's QGPA [9],
56.30 in its Table 6); VIP-Seg's paper PDF (OpenReview browser challenge, external note §1); DyPolySeg's code.
