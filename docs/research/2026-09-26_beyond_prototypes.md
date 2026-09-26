# Beyond prototype refinement: what the remaining gap is made of, and rules derived for it

Research note, 2026-09-26, phase 16 after D-40 (P7), written while D-41 (P8) runs. S1, 2-way 1-shot, CR's features
(D-37's clean base). Numbers without a source are derived here; literature claims are marked in §9 as re-checked or
agent-reported.

## 0. Summary

1. **The ceiling we have been chasing is a ceiling of one rule, not of the features.** 80.86 is the presence-fair
   oracle of the cosine rule `argmax_c u·o_c` (unit geometry, one direction per class and block). Any rule that also
   uses second-order statistics (a metric) has its own, possibly higher, ceiling. With the target at 80+, the question
   is not only "how close can s_c get to o_c" but "which rule's ceiling is above 80 and how much of it is reachable
   without labels".
2. **Every query-side method tried so far is a function of the first prediction and cannot repair region errors.**
   §2 states this as a lemma: EM, self-support and label propagation all move a region only if some of its points
   already vote for the right class; P7a measured that around missed points 91–96 % of the same-class neighbour
   weight is also missed. The limit is structural, not a tuning issue.
3. **Three new levers, each derived with a measurable precondition (§3–§6):**
   * **Nuisance-subspace removal learned on base classes** (the WCCN / NAP idea of speaker verification, where one
     enrolment must match a test utterance of another session): estimate on base classes the covariance of the
     block-to-block shift of a class direction, and score in a metric that shrinks it. If the support–query shift of
     novel classes lies in the same subspace, it is removed (Woodbury, §3); the precondition is one measurable ratio.
   * **Transductive LDA with the query's own total covariance**: for any number of classes, the Fisher discriminant
     directions computed with the total covariance T equal those computed with the within-class covariance W
     (§4, a two-line proof), and T needs no label. This raises the rule's ceiling, not only its level.
   * **Density**: dual-condition prototypes (training-free) and a density-normalised aggregation in the encoder
     (trained; an importance-weighting identity makes the aggregated feature independent of the sampling density).
4. **Text, measured now: weak with 6 base classes.** With CLIP ViT-B/16 and the repository's prompts, the differences
   between novel-class embeddings have only 11 % of their energy (1–22 %) in the span of the 6 base embeddings plus
   the background prompt, and novel-minus-background 23–44 %. A text adapter trained on base anchors can only carry
   that projected part (§7). It is not a route to 80 under our guardrails.
5. **Proposed next probe P9** (inference only): the bias oracle (how much of the gap is calibration), the LDA oracle
   (does a metric raise the ceiling above 80.86), the nuisance-capture ratio, and the two label-free metric rules,
   each with a rule fixed before the run (§8).
6. **Architecture (§9)**: VIP-Seg block by block with the measured weakness of each, and a stacked head (nuisance
   projection, multi-head subspaces, multi-component prototypes, transductive metric, head aggregation, query
   refinement) in which each block covers an error the others provably cannot; its composition contract and order of
   validation.

## 1. A model of the gap

Unit features u ∈ S^{D−1}, D = 128. For a class c seen in block β (support or query),

    u = n(μ_c + η_{c,β} + ε),     η_{c,β} = a_β + b_{c,β},

μ_c the class-invariant part, η the block shift, split into a part a_β shared by every class of the block (density,
scale, colour, local geometry of the sampled region) and a class-specific part b_{c,β} (this object's shape and
appearance), ε point noise. The support direction is s_c ≈ n(μ_c + η_{c,s}), the oracle o_c ≈ n(μ_c + η_{c,q}); the
gap is Δ_c = o_c − s_c ≈ η_{c,q} − η_{c,s}.

The decision between class c and the background for a point of the query is sign(uᵀ(s_c − s_0) + b), with the
oracle's d* = o_c − o_0 and the rule's d = s_c − s_0 = d* − (Δ_c − Δ_0). Four quantities decide the error, each with
an oracle that isolates it:

| part | oracle that removes it | what it says |
| :--- | :--- | :--- |
| bias b (calibration) | per block and class, the additive logit offset that maximises the block's IoU | how much a transductive threshold (class-mass OT, EM on the bias) can give |
| direction d vs d* | the cosine oracle (80.86) | the prototype gap measured so far |
| metric | the LDA oracle with the block's within-class covariance | whether second-order structure raises the ceiling |
| condition (density) | P8's `oracle_other` | how much the "other" points cost |

Only the second has been measured. P9 measures the other two.

## 2. Why rules that start from the first prediction cannot repair a region

**Lemma.** Let Ŷ⁰ be the first prediction and let an update produce Ŷ¹ where each point's new score is a non-negative
combination of (i) its old score and (ii) scores or labels of other points weighted by a kernel K ≥ 0 (self-support
and EM: through a prototype; label spreading: through (I − βS)^{−1}). Let R be a set of class-c points all predicted
b ≠ c, whose K-mass outside R is small (a region). Then Ŷ¹ = b on R unless the K-mass on correctly predicted
class-c points reaching R exceeds the mass on points predicted b.

*Proof sketch.* The update of a point i ∈ R is a weighted vote; inside R every vote is for b, so only the boundary
mass can overturn it; the condition is the vote condition of the composition note §5 evaluated at the region level.

**Measured consequence (P7a, fixed100, feature graph k 16).** Around a missed own-condition foreground point, the local
seed recall is 0.049 (0.04–0.09 over the six graphs): ≥ 91 % of the same-class neighbour weight votes wrong. The lemma
then says no kernel of this family fixes it, which is what P6, D-39 and P7 found (+1.2, −0.7, +0.9, mostly precision).
Any further gain must change the **scores themselves** (metric, prototype, features), not how they are propagated.

## 3. Nuisance-subspace removal learned on base classes

**Estimate on base classes (labels allowed, training blocks).** For each training block β and base class c with at
least n_min points, o_{c,β} = n(Σ u_i) over its points; ō_c the mean over blocks. The shift covariance

    Σ_η = (1 / N) Σ_{c,β} (o_{c,β} − ō_c)(o_{c,β} − ō_c)ᵀ = V Λ Vᵀ.

**Rule.** Score in the metric M_λ = (Σ_η + λ τ I)^{−1}, τ = tr Σ_η / D, in the whitened cosine form so that every row
stays in one gauge (D-38):

    L_ic = n(M_λ^{1/2} u_i) · n(M_λ^{1/2} s_c).

**Why it removes the shift (Woodbury).** Suppose the shared shift lies in an r-dimensional subspace, Σ_η = B Λ_B Bᵀ
with BᵀB = I, Λ_B ≫ λτ. Then

    M_λ = (λτ)^{−1} [ I − B Λ_B (Λ_B + λτ I)^{−1} Bᵀ ],

so a component along B is scaled by λτ / (Λ_k + λτ) → 0 and the complement is left as is: in the limit M_λ is
(λτ)^{−1} times the projector P⊥ = I − BBᵀ (Nuisance Attribute Projection). If Δ_c − Δ_0 ∈ span B, then
P⊥d = P⊥d*: the rule's discriminant equals the oracle's on the complement, and the whole shift is gone.

**Precondition, measurable on valid with labels (L2).** With B the top-r eigenvectors of the base Σ_η:

    κ = ‖P_B(Δ_c − Δ_0)‖² / ‖Δ_c − Δ_0‖²   (share of the novel shift inside the base nuisance subspace)
    ρ = ‖P_B d*‖² / ‖d*‖²                    (share of the oracle discriminant lost by removing it)

and the direct test cos(P⊥d, P⊥d*) > cos(d, d*). The rule can only help when κ > ρ: the base-class nuisance must
capture more of the novel shift than of the novel discriminant. The density shift is a candidate for the shared part
a_β: base classes also appear in both sampling conditions during training, so its direction enters Σ_η.

**What it cannot do.** The class-specific part b_{c,β} of a novel class is, by construction, not in the base
statistics; after removal it is the irreducible part of the one-shot gap. P9 measures what remains.

## 4. Transductive LDA: the query's total covariance gives the within-class directions

**Lemma (Fisher directions with T or W).** Let W be the within-class and B the between-class covariance of the query
block's points, T = W + B the total covariance. If B v = γ W v then B v = γ/(1 + γ) T v.

*Proof.* B v = γ (T − B) v ⇒ (1 + γ) B v = γ T v. ∎

So the discriminant subspace (the generalised eigenvectors of (B, W)) is the same as that of (B, T), and T is the
covariance of the unlabelled query: it needs no label. For two classes this is the familiar T^{−1}(μ_1 − μ_0) ∝
W^{−1}(μ_1 − μ_0) (Sherman–Morrison).

**Rule (label-free).** With the query's mean ū and total covariance T̂ (2,048 points, D = 128), shrunk
T_λ = (1 − λ) T̂ + λ (tr T̂ / D) I, and the support class means m_c of the unit features:

    L_ic = m_cᵀ T_λ^{−1} (u_i − ū) − ½ (m_c − ū)ᵀ T_λ^{−1} (m_c − ū)      (LDA form, one shared metric)

The first term weights directions by the inverse of the query's own spread: a direction along which the query
varies a lot (e.g. its height, its density gradient, the instance's large-scale shape) counts less; a direction
that is quiet in the query but separates the support means counts more.

**Its ceiling.** The LDA oracle (block means and W from the query's labels) bounds this family; if it is well above
80.86, the target becomes reachable in principle on these features. If it is not, no metric helps and the features
must change.

**Caveat.** The support means m_c carry the shift Δ; the metric changes how Δ is weighted, not Δ itself. §3 and §4
compose: M from base classes (shared shift), T from the query (its own spread).

## 5. Density

P7a: other-condition recall 0.16 against 0.77; a third of the missed foreground. P8 measures the bound (g_other) and
the cause (φ). Two remedies, one per level:

* **Training-free: dual-condition prototypes** (D-35's arm E, never run on a clean head). Build a sparse view of each
  support block (P5's `sparse_view`: foreground thinned to background density), encode it, and score
  L_ic = max(u_i·s_c^dense, u_i·s_c^sparse) for foreground rows. It helps iff cos(s_c^sparse, o_c^other) >
  cos(s_c^dense, o_c^other), measurable in the same pass.
* **Trained: density-normalised aggregation.** For a neighbourhood aggregation g(x) = Σ_j K(x_j − x) h_j over points
  sampled with density ρ, the importance-weighted form

      g̃(x) = Σ_j K(x_j − x) h_j / ρ̂(x_j)  /  Σ_j K(x_j − x) / ρ̂(x_j)

  satisfies E[Σ_j K(x_j − x) h(x_j)/ρ(x_j)] = n ∫ K(y − x) h(y) dy for any ρ, so g̃ estimates the kernel average of
  the underlying field independently of the sampling density (ρ̂ a kernel density estimate; PointConv's inverse
  density scale is this idea, §9). VIP-Seg's encoder instead divides the kNN offsets by one batch-global std
  [VIPSEG models/encoder.py:182-189], which keeps density in the features (D-35). A density-normalised variant would
  remove the other-condition deficit at its source. **Risk:** the standard protocol rewards density (the own class
  is oversampled); an invariant encoder may lower own-condition recall. P8's V2 arm (own class sampled uniformly)
  measures how much own recall rests on density.

## 6. Collapse: 6 base classes span few directions

Cross-entropy training drives class means towards a simplex of C − 1 directions with vanishing within-class
spread (neural collapse, §9). With C = 7 (6 base classes and the background), the features are pushed to encode
≈ 6 discriminative directions well; directions that separate novel classes but not base ones are not rewarded and may
be suppressed (the transfer / class-separation trade-off, §9). Two measurements, cheap on CR's features: the
participation ratio PR = (Σ_k λ_k)² / Σ_k λ_k² of the point-feature covariance (effective dimension), and the share
of the novel oracle discriminant d* inside the span of the base class means, ‖P_base d*‖² / ‖d*‖². A low PR with a
small share would point at training: a variance / covariance regulariser against dimensional collapse, or episodes
with pseudo-classes (geometric segments) to raise task diversity. Both are one training run each; neither is proposed
before the measurement.

## 7. Text with six base classes, measured

CLIP ViT-B/16, the repository's prompts ("This point cloud represents the {name}." and the background prompt), unit
embeddings, S1 (base: ceiling, beam, column, chair, bookcase, board; novel: door, floor, sofa, table, wall, window):

| quantity | value |
| :--- | :--- |
| mean cosine between novel embeddings | 0.898 |
| energy of a novel embedding in span(base + background) | 0.855–0.926 |
| energy of a **difference** of two novel embeddings in that span | mean 0.113 (0.013–0.216) |
| energy of novel − background in that span | 0.234–0.440 |
| a random 512-d vector (7-dimensional span) | 0.013 |

The embeddings share a large common component (cosine 0.90), which is why a single embedding looks well explained
by the base span; the part that discriminates novel classes is mostly outside it. An adapter trained with base
anchors only is constrained on span(base): for a linear map G trained from zero (or with weight decay) on base
embeddings, G t = G P t + G (I − P) t with G (I − P) untrained, so only the projected part of a novel embedding
carries learned information. CascadeProto's LMA is such a map. Text can therefore add at most a weak class prior
here; MM-FSS's gain came with 2D vision-language pretraining over many categories, which our guardrails exclude.

## 8. P9, proposed (inference only, CR's features; after P8)

* **Oracles (bounds):** bias oracle per block and class (grid of additive offsets, labels); LDA oracle (block means
  and pooled within-class covariance from labels, shrinkage λ from a grid, the best on valid).
* **Measurements:** κ, ρ and cos(P⊥d, P⊥d*) for r ∈ {4, 8, 16} with Σ_η from base-class training blocks; PR and the
  base-span share of d*; for dual prototypes, cos(s^sparse, o^other) against cos(s^dense, o^other).
* **Label-free rules, frozen on valid, tested once:** the nuisance metric (r or λ from a small grid), transductive
  LDA (λ grid), dual-condition prototypes; each on U, then composed with both + LP.
* **Rules (proposal):** a rule joins the pipeline if it holds at +0.5 (fixed100 gain, CI > 0, all random600 > 0); an
  oracle below +1.0 over the current best closes its direction; if the LDA oracle is not above 80.86 + 1.0, the
  metric family cannot reach the target and the next step is training (§5, §6).

## 9. Architecture: VIP-Seg's blocks, the weakness each one carries, and a stacked replacement

The sections above treat the failures one at a time. The maintainer's point (2026-09-26): an upgrade must start from
the architecture, locate the weakness of each block, and replace it by blocks designed to work together, as a
transformer needs multi-head attention, residuals, normalisation and the feed-forward layer, each covering a limit of
the others. This section does that.

### 9.1 Anatomy of VIP-Seg as used here, with the measured weakness of each block

| block | what it computes [source] | assumption | measured weakness |
| :--- | :--- | :--- | :--- |
| B1 encoder | DyPowerConv kNN aggregation, offsets divided by one batch-global std; Mamba [VIPSEG models/encoder.py:182-189] | features depend on geometry, not on sampling density | density is kept: "other" points recall 0.16 vs 0.77 (P7a), a third of the missed foreground |
| B2 feature head + loss | per-point features, CE on 6 base classes + background | base supervision yields features that separate unseen classes | at most 6 constrained class directions (task span, Tripuraneni et al.); collapse carries to new classes only with many i.i.d. source classes (Galanti et al.); stronger base separation transfers worse (Kornblith et al.) |
| B3 prototype | masked mean per class, one background prototype | one direction summarises a class across instances and densities | region-level instance shift: cosine oracle 80.86 vs 58.55; a multi-component background gives +1.3 (P6), one vector is too little |
| B4 PEM / PDM | channel-correlation attention: Q' S'ᵀ over 72 projected rows gives a D × D row-stochastic A; prototype ← LN(fc(A p + σ(self-corr) ⊙ p) + p) [VIPSEG models/vipseg.py:235-405] | a per-episode channel mixing of the prototype transfers to novel classes | at inference no gain over U (54.84 vs 55.74); in training it helps the features (A0 −2.2, the projection-head effect) |
| B5 logit | dot product f · p, no metric, no calibration | the Euclidean geometry is the right one | untested; the bias and LDA oracles (§8) measure it |

B4 is already a second-order block: it builds D × D channel correlations. It uses them to **re-mix the prototype**
(a vector), learned on base classes, and never to **change the metric** in which query points are compared. That is
the gap the stacked design fills.

### 9.2 The stacked head (proposal): nuisance-projected, multi-head, multi-component metric prototypes

Notation: u_i the unit query feature, U_c^s the support points of class c (c = 0 background).

1. **N: nuisance projection** (fixed per model, from base statistics; §3). P⊥ = I − B Bᵀ, or the soft form M_λ^{1/2},
   with B from the base-class shift covariance Σ_η. Removes the shift directions shared by all classes (density,
   scale, local geometry), which no per-episode block can see from one support.
2. **H: multi-head subspaces** (learned). An orthogonal R = [W_1; …; W_H] ∈ O(D), W_h ∈ R^{d_h × D}; per head the unit
   features u_i^h = n(W_h P⊥ u_i). The analogue of multi-head attention: each head compares in its own subspace, and
   the per-head normalisation makes a head's score independent of the other heads' norms.
3. **C: multi-component prototypes per head** (support side). For each class and head, M components
   s_c^{h,m} = n(Σ_{j ∈ K_m} u_j^h) from spherical k-means of U_c^s, plus a sparse-view component for density (the
   dual-condition prototype of §5). Class score in a head: a_ic^h = max_m u_i^h · s_c^{h,m}.
4. **T: transductive per-head metric** (query side, label-free; §4). The whitened cosine under T_h, the shrunk
   covariance of u^h over the query block: a_ic^h = max_m n(T_h^{−1/2} u_i^h) · n(T_h^{−1/2} s_c^{h,m}). By the Fisher
   lemma of §4, T_h gives the within-class discriminant directions without labels.
5. **A: head aggregation.** L_ic = Σ_h ω_h a_ic^h, ω = softmax(θ) learned on base episodes (a block-diagonal metric in
   the rotated basis); a trimmed mean over heads if the shift is concentrated in few heads.
6. **Q: query refinement** (kept). Background rules (self-support, k-means) and label propagation (D-40) on L.

Without the max, the per-head normalisations and T, the stack is L = U P⊥ Rᵀ Ω R P⊥ Sᵀ: a learned metric in a learned
basis composed with a fixed nuisance projection. The per-head normalisation, the component max and the query
whitening are the non-linear parts that one metric cannot express.

### 9.3 Why a stack and not one of its blocks

| error (measurement) | N | H | C | T | Q (LP) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| shift shared by all classes: density, scale (P8, κ) | removes if κ > ρ | reweights | sparse view | partial | no (§2 lemma) |
| class-specific instance shift, concentrated in some subspaces (e_h, g_h) | no, not in base statistics | down-weights corrupted heads | no | no | no |
| multi-modal class, parts (k-means gain) | no | no | components | no | no |
| this block's own spread (LDA oracle) | no | no | no | whitens | no |
| isolated point errors (P7) | no | no | no | no | smooths |

No single block covers two rows fully, which is the argument for the stack. The order matters: N before H so that the
heads are learned on nuisance-free features; T after C so that the query metric is estimated in each head's subspace;
Q last because it can only smooth.

### 9.4 Composition contract (L3 of the composition note)

* After N, the discriminant share ρ lost must stay small, measured on valid before H is trained.
* H helps only if the shift is spread unevenly over subspaces: for a basis partition, the per-head shift share
  e_h = ‖W_h Δ‖² / ‖Δ‖² against the discriminant share g_h = ‖W_h d*‖² / ‖d*‖²; head weights can gain only when
  g_h / e_h varies across heads. Measurable on CR's features with a PCA or random partition before training.
* C needs multi-modal supports: the gain of k-means foreground components over the single mean on valid.
* T needs a within-class covariance that is far from isotropic: the LDA oracle above the cosine oracle.
* Every block keeps one gauge for all rows (D-38) and acts per query block (order-free, D-37).

### 9.5 Training

Training the stack as the model's head, in place of PEM / PDM, lets the backbone learn features for this comparison
rule. Two training terms follow from §6: the episode CE on base classes, and VICReg's variance and covariance terms
(v(Z) = (1/d) Σ_j max(0, γ − √(Var z^j + ϵ)), c(Z) = (1/d) Σ_{i≠j} C(Z)_{ij}², Bardes et al. Eqs. 1, 4) on the point
features against dimensional collapse. PEM / PDM can stay as a training-only auxiliary head, to keep the
projection-head benefit (A0 vs CR), and be dropped at inference. Encoder-level density normalisation (§5) is a
separate arm, decided by P8.

### 9.6 Order of validation

1. P8 (running): density's share and cause.
2. P9 (inference, CR's features): bias and LDA oracles; κ, ρ; the head-shift spread e_h, g_h; k-means foreground
   components; dual-condition prototypes; the label-free blocks N, C, T and their stack, each against the current best
   (58.55). Rules fixed before the run.
3. One training run of the stacked head with the blocks P9 kept, against CR; a second arm adds the VICReg terms; a
   third, the density-normalised encoder, only if P8 finds density causal.

## 10. References, re-checked

An Opus research agent collected the sources; I re-opened the PDFs for the items marked re-checked.

* **WCCN** (Hatch, Kajarekar, Stolcke, Interspeech 2006): kernel xᵀW^{−1}x with W the expected within-class
  covariance, smoothed (1 − α)Ŵ + αI; estimated on one set of speakers and applied to a disjoint set; up to 22 % EER
  and 28 % DCF gain (**re-checked**: disjoint speaker splits, 22 % / 28 %). **NAP** (Solomonoff, Quillen, Campbell,
  Odyssey 2004): P = I − wwᵀ chosen to minimise within-speaker distances, applied to data other than its training set
  (agent-reported). i-vector WCCN, and the caveat that a WCCN estimated on one corpus did not transfer to another
  (Dehak et al., TASLP 2011, agent-reported).
* **PLDA**: x_ij = µ + F h_i + G w_ij + ε_ij (El Shafey et al., TPAMI 2013, Eq. 1; agent-reported).
* **Simple CNAPS** (Bateni et al., CVPR 2020): Mahalanobis distance with Q_k = λ_k Σ_k + (1 − λ_k) Σ_task + βI,
  λ_k = |S_k| / (|S_k| + 1); one shot gives Q_k = 0.5 Σ_k + 0.5 Σ_task + βI with Σ_k = 0 (**re-checked**); +4.2
  in-domain over CNAPS on Meta-Dataset (agent-reported).
* **PT-MAP** (Hu, Gripon, Pateux, arXiv 2006.03806): Sinkhorn assignment with equal class marginals (agent-reported).
  **Veilleux et al.** (NeurIPS 2021): with Dirichlet class proportions PT-MAP drops 16.8–18.2 points (miniImageNet
  RN-18, **re-checked**), below the inductive baseline. Class-mass priors are therefore not used here: the class
  proportions of a query block are unknown.
* **Kornblith et al.** (NeurIPS 2021): class separation R² against linear transfer, Spearman ρ = −0.93; softmax R²
  0.349, cosine softmax 0.641, squared error 0.845 (**re-checked**).
* **Galanti, György, Hutter** (arXiv 2112.15121): collapse on source classes carries over to new classes when source
  and target classes are drawn from the same distribution, with a bound that vanishes as the number of source classes
  grows (**re-checked**); with 6 source classes the bound gives no guarantee.
* **Tripuraneni, Jin, Jordan** (arXiv 2002.11684): task diversity ν = σ_r(AᵀA / t) > 0 needs the tasks to span all r
  directions (agent-reported).
* **Jing et al.** (ICLR 2022), dimensional collapse; **VICReg** (Bardes, Ponce, LeCun, ICLR 2022), Eqs. 1, 4, 6
  (agent-reported).
* **PointConv** (Wu, Qi, Fuxin, CVPR 2019): inverse density scale from a KDE through an MLP; ScanNet Table 5, xyz,
  0.5 m: 61.0 with density, 60.3 without, 60.1 with raw density and no MLP (**re-checked**). About one point, and raw
  density without the MLP is worse than none: a density term must be learned, not fixed.
* **Few-shot 3D segmentation**: no paper found that uses a Mahalanobis / covariance-aware prototype or optimal
  transport on the query (agent search over 8 papers; absence of evidence). AttMPTI's propagation: old protocol
  53.77 / 55.94, corrected 31.09 / 29.62. COSeg's base-prototype calibration is trained, not test-time
  (agent-reported). QHP (arXiv 2512.08253) builds query-aware prototypes from a support–query bipartite graph,
  corrected protocol 38.86 / 37.84 (agent-reported).
