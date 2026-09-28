### D-48 — A correlation architecture with base-class exclusion, multi-prototypes and per-episode modality weights, replacing the PEM/PDM head · `PROPOSED`, beyond the paper, maintainer request 2026-09-28

* **Problem.**
  - On the clean base CR, the trained head adds nothing on novel classes. U, which has no parameters, beats it
    (55.74 against 54.84 on fixed100). The head recovers −0.01 of the novel-class gap (P3 part A, D-46) against 0.24
    of the base-class gap (74.23 vs U 70.68 vs oracle 85.44; P10.1).
  - Every learned module tried since D-37 (M1, M2, M5 alignment; the P10.4 base-class covariance) behaves the same
    way: it fits the 6 base classes and does not transfer.
  - The inference stack reaches 58.55 (U + both + LP).
  - D-46 split the novel gap. Own pairs carry no bias; their error is instance variance. The other condition carries a
    K-invariant bias. The background rows add up to 8.4 points. No single term exceeds 13.6 points, so no single
    module can close the gap. The maintainer asked for one composed architecture instead of one measured module at a
    time.
* **Literature** (read 2026-09-28; the numbers are in their own settings and backbones, not comparable to ours):
  - **COSeg** (CVPR 2024).
    - Correlation instead of feature optimisation: +9.26 for 1-way 1-shot.
    - Hyper Correlation Augmentation: +3.84.
    - Base Prototypes Calibration: +3.44.
    - It also identifies foreground leakage, the density shortcut that D-41 measured as φ 1.00.
  - **BAM** (CVPR 2022): a base learner that marks what not to segment.
  - **MM-FSS** (ICLR 2025): Test-time Adaptive Cross-modal Calibration weights a text prediction per episode by its
    IoU on the support.
  - **DA-FSS** (2026): late, decoupled fusion of modalities.
* **Earlier work in this repository** (why this is not a repeat):
  - **D-27 and D-28.** Training-free base calibration on untrained features (AUC 0.49–0.63), run on the pre-D-37
    base. Here a base learner is trained with full base-class labels.
  - **D-33 and D-34.** A feature-level attention neck on the pre-D-37 base. Its gate never opened, and P9.8 found
    missed-point retrieval purity 0.109. Here the neck reads correlations, not features.
  - **P6 and P9.6.** Fixed k-means components at inference: background +1.32 (kept in "both"), foreground 53.86
    below U. Here multi-prototype correlations are consumed by a trained neck.
  - **D-31 and D-46 P3.** Text with a fixed weight gives +0.41; the oracle weight gives +3.31. Here the weight is
    estimated per episode from the support.
* **Architecture** (`stage_type=corr`; the encoder and the 128-d feature head are CR's, trained from scratch as CR):
  * **[1] Encoder and feature head.** Unchanged. Unit features f = n(F).
  * **[2] Base learner.**
    - An MLP (128 → 128 → |C_base| + 1) on the query and support point features. The output is a softmax over the 6
      base classes of the fold plus "none". "None" covers test classes and clutter, which are background in every
      training episode anyway.
    - It is trained with CE against the full per-point labels of the sampled blocks. These come from the inherited
      loader, called through a wrapper with the base classes as `sampled_classes` for the label vector only. The
      point sampling depends only on `sampled_class`, so the sampled points are the episode's own (to be verified by
      a test).
    - During training, the episode's own target classes are excluded when the base score is formed (as COSeg Eq. 11).
    - The base score is g(x) = max over the non-excluded base classes of p_base(x).
  * **[3] Multi-prototypes.**
    - Per way: M_fg = 16 foreground prototypes, formed by farthest-point sampling of seeds inside the support mask
      (over the K shots) followed by one assignment step. Each prototype is the mean unit feature of its cell.
    - Background: M_bg = 32 prototypes, formed the same way over every mask-0 point of every way.
    - With fewer masked points than M, M is reduced to the number of points. There are at least 101 per block
      (10a).
    - M is a convention, not measured. COSeg used 100 over about ten times more points.
  * **[4] Correlation.**
    - C[x, c, m] = ⟨f_x, n(p_{c,m})⟩ for every query point x, class row c and prototype m.
    - The descriptor is the top-T values of C[x, c, ·] sorted in descending order (T = 8), plus their mean. This
      makes it invariant to the order of the prototypes and to any orthogonal rotation of the feature space.
  * **[5] Correlation neck.**
    - L = 2 transformer layers over the points of each query block, 4 heads, width 64. They take the
      [2048, N+1, T+1] descriptors embedded per class row by one shared linear map, with positional encoding from xyz.
    - Attention runs across points, with each class row processed by the same weights; one cross-class mixing layer
      follows.
    - Output: logits ℓ[x, c].
    - Background calibration: ℓ[x, bg] += ψ · logit(g(x)), with ψ a learned scalar initialised at 0.
  * **[6] Late modality fusion, at inference, no parameters.**
    - Per modality m ∈ {text, image, audio} (D-47), a modality logit t_m[x, c] = ⟨R f_x, e_{m,c}⟩ / τ. R is P3's
      ridge map fitted on base classes (frozen, as in D-46's P10.2).
    - Its weight is γ_{e,m} = IoU on the episode's support of the argmax of t_m over {bg, c} for each support block.
    - ℓ ← ℓ + Σ_m γ_{e,m} t_m.
    - With γ = 0 this is the identity.
  * **[7] LP.** D-40's rule at inference, unchanged.
* **Loss.** L = CE(ℓ_final) + (1/L) Σ_l CE(ℓ_l) + λ_b CE(p_base, y_base), with λ_b = 1 (convention).
  - ℓ_l are the logits read after neck layer l (deep supervision, as for D-33's unopened gate).
  - No alignment loss (D-46 M5) and no VICReg (D-45).
  - K = 1 for training, as CR, so that the schedule and the test are comparable. The K-curve of D-46 P10.3 is scored
    at test.
* **Input assumptions, stated so that they can be attacked** (each one a reason the architecture could fail):
  - A1. Base-class points are a sizeable share of the novel episodes' query background, and the base learner
    separates them from novel foreground on test blocks. The risk is novel-to-base confusion, for example sofa →
    chair on S1.
  - A2. The correlation descriptor of a novel class has the same distribution as that of a base class, so the neck
    transfers. This is the reason for choosing correlations.
  - A3. With 2,048 points and at least 101 foreground points per support block, 16 cells per way are populated
    enough to be less noisy than one mean.
  - A4. The ridge map R fitted on base classes carries novel-class text information (P3.0 holds on CR: accuracy
    0.693), and support IoU predicts query usefulness per episode.
  - A5. The neck does not relearn the density shortcut of D-41. It sees xyz positions and correlation magnitudes,
    both of which carry density.
  - A6. Deep supervision and the base CE do not dominate the episodic CE on 6 classes.
* **P11, a pre-check** (inference, CR `last.pt`, valid for decisions, fixed100 reported; about 1 h, not measured):
  * **P11.1 exclusion ceiling.** Set to background every query point whose true label is a base class, on top of
    U + both + LP. This gives Δ_base, the gain.
    - Also reported: the base share of background points, and the share of novel foreground points whose nearest base
      prototype (P1's bank) is closer than their own support prototype.
    - **Rule:** if Δ_base < 1.0 on valid, layer [2] is dropped from both arms (c).
  * **P11.2 per-episode text weight.** [6] with text only, on U.
    - **Rule:** holds if the TACC gain over U is ≥ +0.5 with CI above 0 on valid. The arms then use [6] at test;
      otherwise [6] is reported and switched off.
  * **P11.3 multi-prototype correlation without a neck.** Max over the top-T of [4], on U's decision rule, with the
    background row from "both". This is descriptive: it shows whether [3]–[4] alone lose to U, as P9.6's components
    did.
* **Training** (after P11; two arms on the 3090, CR's schedule: batch 1, 24,000 updates, LR halved every 7,200,
  seed 0):
  - **A:** [1]–[5], with [2] unless P11.1 dropped it.
  - **B:** as A, with [3]–[5] replaced by CR's four `vip_clean` stages on mean prototypes, keeping [2] through the
    same ψ calibration of the background logit.
  - A − B isolates the correlation path. B − CR isolates the base learner.
* **Test** (as D-45 and D-46): `d43_eval.py test` on CR, A and B, `last.pt` and `best.pt`.
  - Draws: fixed100, random600 seeds 0–2, and the leak-free draw. The leak-free draw is reported for every arm, as the
    benchmark's density leakage (COSeg) makes it the second setting of the paper.
  - The K-curve k = 1, 2, 3, 5 on A.
  - **Inference ablations of A, without retraining:**
    - ψ = 0 (no exclusion);
    - γ = 0 (no modality);
    - M_fg = M_bg = 1 (mean prototypes);
    - no LP.
* **Rules, fixed before the run.** "Holds at g" means: fixed100 gain ≥ g with a paired CI above 0, and > 0 on all three
  random600 draws, `last.pt`.
  - **D48.1 new base.** A's full stack ([6] and [7] as P11 set them) − CR's U + both + LP holds at +1.0. A becomes the
    base.
  - **D48.2 correlation path.** A − B, both with the same [6] and [7], holds at +1.0. The correlation neck is
    credited.
  - **D48.3.** D48.1 fails, B's stack − CR's stack holds at +1.0. B becomes the base, and the correlation path is
    dropped.
  - **D48.4.** Neither A nor B holds. The architecture stops. The inference ablations and A1–A6's reported
    measurements decide what is re-examined, in a new decision.
  - **D48.5, reported.** The ablations; the leak-free draw (with no rule, since the paper's comparisons are on the
    standard setting); the K-curve; ψ; the base learner's accuracy on base classes and its novel-to-base confusion on
    test blocks.
* **Cost** (not measured). P11 about 1 h. Two trainings together, about 3 h: the neck attends over 2,048 points per
  class row. Tests about 1.5 h.
* **Affects.**
  - `models/base_learner.py` (new), `models/correlation.py` (new: [3]–[5]), `models/modality_fusion.py` (new: [6]);
  - `models/cascadeproto.py` (`stage_type=corr`, `base_learner`, `n_proto_fg`, `n_proto_bg`), `pipeline/episodes.py`
    (the label wrapper), `pipeline/model_api.py` (`loss_reg` gains the base CE and deep supervision);
  - `train.py`;
  - `experiments/p11_precheck.py`, `experiments/run_d48.sh`;
  - `tests/test_correlation.py`, `tests/test_base_learner.py`; 05, new section.
