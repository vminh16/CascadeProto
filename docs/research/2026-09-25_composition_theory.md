# Why modules that look right fail when composed, and the causal chain a module must satisfy

Research note, 2026-09-25, phase 16 after D-39. Numbers are S1 fixed100 unless stated; sources in brackets.
Literature claims are marked with the reference list of §9; anything not verified there is flagged.

## 0. Summary

1. Every failure of phase 16 has the same shape: a module was built on an assumption about the quantity it acts on,
   and in the composed pipeline that assumption was false. The assumption was never measured on the input the
   module actually receives. §2 lists them; §3 gives the contract every module must satisfy before it is built.
2. Self-support (P6, D-39) fails on the foreground for a computable reason: its fixed point is pulled toward the
   contaminants inside the predicted region, and the foreground predicted regions are 20–45 % contaminated (precision
   0.55–0.80), while the background's is 12 % (0.88). §4 derives the improvement condition; it predicts the sign of
   every P6 arm.
3. Label propagation on the query's own graph brings information outside the support prototype's cone (spatial
   continuity), but it has its own preconditions (edge homophily, seed accuracy, expansion). §5 derives a vote
   condition that can be **measured with labels before building anything**, which bounds the gain from above and
   the damage from below.
4. D-39's A0 result (training without the head gives 2.2 worse features for the same rule, while the head is
   useless at inference) matches the projection-head effect of contrastive learning. §7 states it as a testable
   hypothesis with a cheap architectural consequence.
5. §8 proposes P7 as a two-stage probe: P7a measures the preconditions (homophily, vote bound, contamination,
   cone geometry) with labels; P7b runs propagation only if P7a's bound clears the go threshold.

## 1. The objects

Per query block, unit query features u_i = n(f_i), logits L_ic = f_i · r_c with unit rows r_c (the U geometry of
D-38). Support direction s_c, oracle direction o_c = n(Σ_{i∈Ω_c} u_i) where Ω_c is the set of query points of class
c. The prediction is ŷ_i = argmax_c f_i · r_c. Measured on CR's features: U 55.74, oracle 80.86, U + both 57.63
[`results/phase16_d39/SUMMARY.md`].

Two invariances matter for composition:

* **Common gauge.** argmax_c f_i · r_c is invariant to r_c → λ r_c for one λ > 0 shared by all rows, and to adding
  the same vector to every row; it is **not** invariant to per-row scales. Any module that changes one row's norm
  (or mixes rows from two geometries) changes decisions for a reason unrelated to its direction. P4's row oracle
  violated this (D-35 point 3).
* **Query-order equivariance.** Every per-query operation commutes with permuting the query blocks; VIP-Seg's
  reshape does not (D-36). A module that breaks it can learn the loader's order.

## 2. The failures of phase 16, read as broken assumptions

| module | assumption it needed | what was true in the composed pipeline | measured by |
| :--- | :--- | :--- | :--- |
| D-26 EM, D-31 text, D-33/34 neck (on E1) | the prediction is driven by the prototypes they modify | E1's prediction was driven by the query position (swap: 73.20 → 0.92) | C3, D-36 |
| P4 row oracle | all rows in one gauge | replaced rows gained a norm the others lacked (bg predictions +48 %) | D-35 point 3 |
| D-29 oracle distillation (on E1) | the logit geometry is the one being scored | the shortcut path decided the logits | C3 |
| P6 foreground self-support | the predicted foreground region is mostly the class | precision 0.55–0.80 | P6, §4 |
| D-39 A1 (trained self-support) | training fixes the pseudo-labels' contamination | α_fg stayed 0.70 and A1 − A0 = −0.66 | D-39 |
| D-39 A0 (no head) | the head only computes the inference output | the head also shapes the features in training (A0 −2.2) | D-39, §7 |
| ssp_bg + km3 | independent error sets (additive) | +1.19 and +1.37 give +1.89, not +2.56: 26 % overlap | D-39, §6 |

None of these assumptions was false in general; each was false **for the input the module received in that
pipeline**. That is the common cause.

## 3. The contract: a causal chain for every module

A module M acting on a quantity Q to reduce an error E is admissible only when each link is measured, in order:

* **L0 target.** E is measured on the current base, with its ceiling (an oracle that fixes E exactly, in the same
  gauge). If the ceiling is small, stop.
* **L1 mechanism.** A statement "if assumption A holds, M reduces E by at least δ(A)", derived, not intuited.
* **L2 precondition on data.** The quantities in A are measured on the base's actual outputs, *before* M is built
  (a label-using diagnostic is allowed here; it is a measurement, not a method).
* **L3 interface.** The same quantities are measured on the input M receives **after the upstream modules** of the
  composed pipeline (a module tuned in isolation meets a shifted input distribution when composed).
* **L4 effect.** Inference-only probe first (frozen on valid, tested once), then a trained arm; paired bootstrap;
  thresholds fixed before the run (+1.0 for a training run, +0.5 for a training-free follow-up).
* **L5 mechanism check.** The intermediate quantity M claims to move (e.g. cos(r_c, o_c)) moves in the predicted
  direction. A gain without it is treated as unexplained (the R2.4 rule of D-29 generalised).
* **Confounds, every time.** Leaks (position, density, class presence), gauge, selection on test, one training seed.

Composition f ∘ g is admissible when post(g) ⇒ pre(f): the distribution g outputs satisfies f's assumption. L3 is
exactly this check.

## 4. Self-support: the fixed point and when it helps

Let A_c(r) = {i : ŷ_i = c under rows r} be the predicted region, π_c its precision (share of true c points in it),
μ_c the unit-feature sum over its true-c points and ν_c over its contaminants. One step gives

    S_c = n(π_c μ̄_c + (1 − π_c) ν̄_c),     r_c ← n(α r_c + (1 − α) S_c)

(μ̄, ν̄ the corresponding means). Two forces act on the new row:

* **Truncation.** μ̄_c is the mean of the class points *inside* the cone around r_c. For a unimodal class around o_c
  (e.g. a von Mises–Fisher law) it lies between r_c and o_c: this part moves r_c toward the oracle.
* **Contamination.** ν̄_c points toward the classes that leak into A_c. For a foreground row the main contaminant is
  the background (heterogeneous, far from o_c); its weight is 1 − π_c.

r_c' lies on the arc between r_c and S_c, so for small 1 − α the step improves the row (cos(r_c', o_c) >
cos(r_c, o_c)) exactly when

    S_c · o_c > r_c · o_c,     S_c = n(π_c μ̄_c + (1 − π_c) ν̄_c):

the self-support direction must already be better aligned with the oracle than the row it corrects. The
contaminant mean ν̄_c has small ν̄_c · o_c (it is made of other classes), so as 1 − π_c grows S_c swings toward the
contaminants and the condition fails.

Consequences that match the measurements [`results/phase16_p6/SUMMARY.md`, test JSONs]:

* Background: π_bg = 0.88 and the background region is the largest set (small truncation variance): helps (+1.19).
* Foreground: π = 0.55 (table) … 0.80 (window): the contaminant term dominates; every foreground arm ≤ U on valid.
* ρ < 1 (entropy selection) raises π slightly but shrinks the region to the cone's core, where μ̄_c ≈ r_c: the
  truncation term vanishes, so the step does nothing. Every ρ < 1 arm was ≈ 0 (P6).
* Iterating converges to a fixed point r* = n(α r* + (1 − α) S_c(r*)) that sits between the support direction and
  the contaminant mean, not at o_c; the recall errors (true c points outside every cone) never enter S_c.
* Training (A1) cannot change the sign unless it raises π_c on **novel** classes; it learned α_fg ≈ 0.70 on base
  classes and lost 0.66 on novel ones.
* The self-training theory of Wei et al. [2] gives the same threshold from another side: its denoising bound needs
  the pseudo-labeller's error below 1/3 (with expansion c > 3). The U rule's foreground pseudo-label error (1 − π) is
  0.45 (table), 0.38 (door), 0.34 (sofa), 0.29 (floor), 0.28 (window), 0.20 (wall): half the classes are outside the
  regime where denoising is guaranteed, the background (0.12) is inside it.
* A1 was not a faithful SSP [7]: SSP trains with an additional self-support loss that drives the self-support matching
  itself, while A1's auxiliary term only trained the support-only prediction. The negative result is about A1's form,
  not about every trained self-support.

L5 check available now: the diagnostic can compute cos(r_c, o_c) before and after one step per class; the theory
predicts a decrease for π_c below a threshold that depends on μ̄ · o and ν̄ · o, which the same pass measures.

## 5. Label propagation on the query graph

**Operator.** kNN graph on the query's points, W_ij = k_x(‖x_i − x_j‖) · k_f(u_i · u_j) for j ∈ kNN(i),
S = D^{−1/2} W D^{−1/2}. Label spreading: F* = (1 − β)(I − βS)^{−1} Y₀, the limit of F ← βSF + (1 − β)Y₀, whose
convergence Zhou et al. prove [1]. AttMPTI [6] uses the same closed form on a graph over support multi-prototypes and
query points; its paper does not isolate the gain of the propagation step (not found in its ablation), and TPN [5]
reports that transduction by propagation beats per-query inference in few-shot image classification.
With K = (1 − β)(I − βS)^{−1} (a non-negative smoother, rows dominated by short walks), F*_i = Σ_j K_ij Y₀_j: point
i's new score is a walk-weighted vote of the initial predictions around it. A random-walk / harmonic-function reading
of the same operator is in Zhu, Ghahramani and Lafferty (ICML 2003, not verified here).

**Why it can do what self-support cannot.** Self-support only reuses Y₀ through the prototype (a global average);
propagation uses the graph, i.e. spatial and feature continuity **inside this query**. A class point outside the
support cone can be recovered if its walk-neighbourhood votes for its class. This is information s_c does not carry.

**Vote condition (binary, class c against the rest).** For a point i of class c let p_i be the K-mass of its
neighbourhood on class-c points (local homophily), a the recall of Y₀ on those neighbours and b the rate at which
non-c neighbours are predicted c. Its expected vote for c is p_i a + (1 − p_i) b, so propagation labels it c when

    p_i a + (1 − p_i) b > 1/2     ⇔     p_i > (1/2 − b) / (a − b).

With the U-rule's foreground recall a ≈ 0.7–0.85 and b ≈ 0.05–0.1 [P6 recall/precision], p_i must exceed about
0.6–0.7. The same condition read for a correctly predicted point with a heterophilous neighbourhood gives the
damage: propagation flips it when its neighbourhood is dominated by other classes. Both counts are computable with
labels on the actual predictions; their difference bounds the net gain from above (with the true K) — an L2
measurement.

**Where the theory says it breaks.**

* **Seed errors cluster.** a is not uniform: missed points sit next to other missed points (the recall errors are
  regions, not salt-and-pepper). The local a around missed points is what matters, and it is lower than the global
  recall. Measure it; the global number overstates the gain.
* **Heterophily at boundaries and in sparse regions.** Points of a class sampled at background density (the "other"
  condition of D-35) have neighbours farther away, often of other classes: p_i drops exactly where the recall
  errors are.
* **Over-smoothing.** As β → 1, F* tends to the degree-weighted stationary vote of the connected component: small
  classes are absorbed by the majority. β must be selected, and the break count grows with β.
* **Degree bias from density.** Dense (oversampled) regions have larger degrees; the symmetric normalisation limits
  but does not remove their pull. This is a density-leak path (D-35): propagation may raise the standard-protocol
  score through the density cue while doing nothing leak-free. The leak-free draw is the control.
* **Composition with the background rules.** Y₀ should be the output of U + both (the best rule), so its a, b differ
  from U's; the preconditions must be measured on that Y₀ (L3), not on U's.

**Theory of when propagation denoises pseudo-labels.** The expansion assumption of Wei et al. (ICLR 2021) and Cai et
al. (ICML 2021) — every small subset of a class has a proportionally larger neighbourhood inside the same class —
is the condition under which propagating from a noisy labeller provably reduces error [2, 3]; [2] also
needs the labeller's error below 1/3. On a point cloud it is
the statement that each object is a connected, well-sampled region of the kNN graph. It is plausible for dense,
own-condition objects and doubtful for sparse, other-condition ones; the diagnostic measures it per condition.

## 6. Composition algebra

* **Additivity.** Two rules add when they fix disjoint error sets and neither breaks what the other fixes. Measured:
  ssp_bg +1.19 and km3 +1.37 give +1.89, 74 % of the sum. Both act on the background row and fix overlapping
  errors. A new module stacked on U + both must be judged against U + both, never against U.
* **Covariate shift of the downstream module.** A module tuned on the base's outputs sees a different input when
  placed after another module. P6's frozen α (0.25) was tuned on U's rows of CR; on A0's rows the same rule gave
  +0.2. Re-select on valid for every composed pipeline.
* **Training-time coupling.** A module present in training changes the features through its gradients. Two
  observed forms: the aux / self-support gradient did not help novel classes (A1), and the head's gradient helped the
  backbone (A0). Conflicting objectives can be detected directly as negative cosine between the task gradients
  (the PCGrad diagnostic [9]). Homophily below the vote threshold is the graph analogue of [10].
* **Selection.** Choosing among many arms on valid inflates the valid gain (winner's curse); a fresh test removes the
  bias. Measured: self-support valid +1.56 → test +1.19, k-means +1.27 → +1.32.
* **Leaks.** Any gain must survive the swapped order (position) and be reported on the leak-free draw (density) and
  with presence-fair oracles.

## 7. The head as a projection head (hypothesis, from D-39)

Observations: on CR's backbone, the U rule (55.74) beats the head's own prediction (54.84); a backbone trained
without the head (A0) is 2.2 worse under the same rule (53.52). In SimCLR the representation *before* the nonlinear
projection head beats the one after it by more than 10 points in linear evaluation, because the head, trained for the
contrastive loss, discards information useful downstream [8]; Xue et al. give a theoretical account of why training
with a head and discarding it helps in self-supervised, supervised-contrastive and supervised settings [8b]. The reading: PEM/PDM absorb episode-specific, base-class-specific adjustments
during training, so the backbone does not have to; at inference on novel classes those adjustments do not transfer.

Testable predictions: (i) a plain MLP head used only in training and discarded at inference reproduces CR's backbone
benefit over A0; (ii) the head's gain over A0 is larger on novel than on base classes. If (i) holds, the architecture
simplifies to "train with a disposable head, infer with prototypes + graph", which is a clean paper claim; it costs
one training run.

## 8. P7, proposed (not recorded as a decision yet)

* **P7a diagnostics, labels allowed, CR + U + both as Y₀, S1 valid and fixed100.** Per class and per sampling
  condition (own / other, D-35): local homophily p_i on the xyz and feature graphs (k ∈ {8, 16}); local seed recall
  around missed points; the vote bound (fixed − broken counts under the true K for β ∈ {0.5, 0.8, 0.9}); for
  self-support, cos(r_c, o_c) before and after one step with π_c, μ̄ · o, ν̄ · o (L5 of §4).
* **Gate.** Only if the vote bound predicts ≥ +1.0 mIoU on valid for some (k, β) does P7b run.
* **P7b propagation.** Label spreading on Y₀ with (k, β, graph weights) selected on valid, tested once on fixed100,
  random600 × 3, leak-free; the go rule of P6.
* **Separately, one training run** for §7's hypothesis (a disposable MLP head), if the maintainer wants the
  architecture claim tested.

## 9. References (verified against primary sources on 2026-09-25 unless marked)

1. D. Zhou, O. Bousquet, T. N. Lal, J. Weston, B. Schölkopf. Learning with local and global consistency. NIPS 2003.
   S = D^{−1/2}WD^{−1/2}, F(t+1) = αSF(t) + (1 − α)Y, convergence to F* = (1 − α)(I − αS)^{−1}Y.
2. C. Wei, K. Shen, Y. Chen, T. Ma. Theoretical analysis of self-training with deep networks on unlabeled data.
   ICLR 2021, arXiv:2010.03622. (a, c)-expansion (Def. 3.1); Thm 4.3: under expansion with c > 3 and pseudo-labeller
   error below 1/3, the self-trained classifier's error is bounded by (2/(c − 1)) Err(G_pl) plus a consistency term.
3. T. Cai, R. Gao, J. D. Lee, Q. Lei. A theory of label propagation for subpopulation shift. ICML 2021,
   arXiv:2102.11203. Expansion between source and target subpopulations; propagation from a source teacher provably
   improves on it in the target.
4. E. Arazo, D. Ortego, P. Albert, N. O'Connor, K. McGuinness. Pseudo-labeling and confirmation bias in deep
   semi-supervised learning. IJCNN 2020. Naive pseudo-labelling overfits its own wrong labels (confirmation bias).
5. Y. Liu et al. Learning to propagate labels: transductive propagation network for few-shot learning. ICLR 2019,
   arXiv:1805.10002. Gains of about 4.1 (1-shot) and 1.7 (5-shot) points on 5-way miniImageNet over prior work.
6. N. Zhao, T.-S. Chua, G. H. Lee. Few-shot 3D point cloud semantic segmentation (AttMPTI). CVPR 2021,
   arXiv:2006.12052. Label propagation over support multi-prototypes and query points with the closed form of [1];
   an ablation of the propagation step alone was not found.
7. Q. Fan, W. Pei, Y.-W. Tai, C.-K. Tang. Self-support few-shot semantic segmentation (SSP). ECCV 2022,
   arXiv:2207.11549. Self-support prototypes from confident query predictions, adaptive self-support background
   prototypes, trained with an additional self-support loss.
8. T. Chen, S. Kornblith, M. Norouzi, G. Hinton. SimCLR. ICML 2020, arXiv:2002.05709. Representation before the
   projection head beats the one after by > 10 points (linear evaluation); the head removes information such as
   colour or orientation. 8b. Y. Xue, E. Gan, J. Ni, S. Joshi, B. Mirzasoleiman. Investigating the benefits of
   projection head for representation learning. ICLR 2024, arXiv:2403.11391. (Also K. Gupta et al.,
   arXiv:2212.11491, 2022.)
9. T. Yu et al. Gradient surgery for multi-task learning (PCGrad). NeurIPS 2020, arXiv:2001.06782. Conflicting
   gradients: negative cosine.
10. J. Zhu et al. Beyond homophily in graph neural networks. NeurIPS 2020, arXiv:2006.11468. Edge homophily
    h = |{(u, v) ∈ E : y_u = y_v}| / |E|; performance degrades as h falls.
11. Z. An et al. Rethinking few-shot 3D point cloud semantic segmentation (COSeg). CVPR 2024, arXiv:2403.00592.
    "Foreground leakage" from non-uniform point sampling (density disparity); sparse 2,048-point sampling.

Not verified here: Zhu, Ghahramani, Lafferty (ICML 2003) harmonic functions; the exact COSeg corrected numbers
(taken in D-35 from the paper's Table 1).
