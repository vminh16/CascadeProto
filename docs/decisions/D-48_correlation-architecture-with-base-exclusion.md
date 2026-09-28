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
* **Amendment 1 (2026-09-28, after a mathematical review, before any code).**
  * **Review.** Maintainer request, agent on Opus. The report is `docs/research/2026-09-28_d48_math_debate.md`, with
    its CPU scripts and output in `results/phase16_d48/review/`. The main claims were re-checked against the sources:
    - P3's support weight is in `models/text_prior.py:102-115`.
    - P8's recalls: U 0.245 → U + both 0.166 for the other condition, and 0.837 → 0.463 under uniform sampling
      (`results/phase16_p8/SUMMARY.md`).
    - D-45 arm B's stack is 27.84 against U 44.40.
    - The CPU composition numbers are in `results/phase16_d48/review/d48_cpu_checks.json`.
  * **New measurements (CPU, 600 seeded episodes per fold side, S1).**
    - At test, the query background is 0.430 base, 0.129 clutter and 0.441 novel classes outside the episode. In
      training it is 0.252 non-target base and 0.633 "none"; the "none" part is mostly wall and floor, S1's novel
      classes.
    - The densely sampled class is always a base class in training and always novel at test.
    - 95 % of test-class blocks also occur in training.
    - fixed100 support foreground: minimum 187 points, median 764.
    - Removing every false positive would lift U from 55.74 to 77.02 and U + both + LP from 58.55 to 72.60 (their mean
      recalls).
    - If base points caused false positives in proportion to their share of the background, removing them would give
      +4.68 on U and +3.35 on the stack. This is an estimate, not measured.
    - A one-call wrapper with `support=False` and the class list [targets…, other base classes] reproduced 20 of 20
      augmented episodes bit for bit. A second sampler call does not.
  * **Changes.**
    1. **[2] is detached from the encoder.**
       - The base CE trains only the base MLP (stop-gradient on its input), as BAM's base learner is trained apart.
       - Why: a "none" target spanning all novel points puts collapse pressure on exactly the test classes. M5 already
         showed a base-side loss lowering the novel oracles (80.84 → 78.03 / 73.72).
    2. **The exclusion and the background term are redefined.**
       - In training, g is computed from a softmax renormalised over {non-target base classes, none}. This
         leave-target-out form lets training see novel-like foreground with its true g.
       - g is clamped to [10⁻⁴, 1 − 10⁻⁴].
       - In arm A, g enters the background row of the correlation descriptor as one more channel before the neck (as
         COSeg Eq. 12).
       - In arm B, and as A's inference ablation, the term is one-sided: ℓ[x, bg] += ψ · (−log(1 − g)).
       - Why: the symmetric ψ · logit(g) lowered the background logit wherever g < 0.5, which covers 57 % of the test
         background.
    3. **P11.1 becomes P11.1b, a realistic probe.**
       - The base MLP is trained on frozen CR features on the training episodes, through the one-call wrapper.
       - It is applied with the one-sided term, and ψ is selected on a seeded validation draw that carries raw labels
         (P5's `draw_episodes`), since the stored valid and fixed100 episodes keep no raw labels.
       - Reported: the net gain after recall loss, and g on own-condition (dense) novel foreground against
         other-condition and background points.
       - **Rule.** [2] enters the arms if the net gain is ≥ +1.0 and at most 10 % of dense novel foreground has
         g > 0.5 (c). Otherwise [2] is dropped. The oracle P11.1 is kept as a reported ceiling (estimated +3.35 on the
         stack; it could not fail the old bar).
    4. **P11.4, a transfer probe, is added and gates arm A.**
       - A small MLP is trained on the [3]–[4] descriptors of base classes (training episodes, frozen CR features) and
         scored against U on novel valid episodes, on the leak-free draw, and by own / other recall.
       - **Rule.** Arm A is trained only if the probe's novel-class score − U ≥ 0 on valid (c). If it beats U on base
         classes and loses on novel ones, A2 fails as it did for CR's head, and A is not trained.
       - Its own / other recall is the density reading for A5.
    5. **[6] is rewritten.**
       - **History corrected.** P3 already weighted every arm per episode from the support (γ_e = max(0, 2·acc − 1)
         on the support foreground). The +0.41 is that support-weighted prior, not a fixed weight as the Problem
         paragraph says.
       - **Why [6] cannot add a direction.** The ridge prior is a linear read-out of the 6 base-prototype similarities.
         Every modality of D-47 passes through the same bank, so adding modalities re-weights the same 6 numbers and
         cannot add a direction. On sofa and table the text rewards chair-likeness, which [2] sends to background; on
         door, window and wall it duplicates [2].
       - **Consequences.**
         - The rules D48.1–D48.3 are scored with [6] off (γ = 0).
         - [6] is reported as an inference add-on, with κ selected on each arm's valid and the bank and ridge refitted
           on each arm's features.
         - The background text row must be defined before [6] is coded.
         - Modalities are combined by one coefficient vector per class selected on valid, not by summing independent
           γ's.
         - Audio is reported only as D-47's identity check.
       - P11.2 reports corr(γ_e, κ*_e) against the per-episode oracle weight.
    6. **"both" and LP are re-selected on each arm's valid**, with "none" allowed. The CR selections failed on other
       features before: −16.6 on D-45's arm B, and the loss of M1's leak-free gain in D-43.
    7. **P11.3** reports M_bg ∈ {8, 16, 32}, raw against centred correlations (features are non-negative; the mean
       unit-feature norm² is 0.56), and the share of background cells dominated by the other way's class.
    8. **Readings.**
       - The ψ = 0 and M = 1 inference ablations are co-adaptation readings, not "the model without the component".
       - The base learner's confusion on test blocks is labelled in-sample, since 95 % of those blocks were training
         blocks.
       - The neck's cross-class mixing layer must be equivariant to the order of the foreground ways (a test).
    9. **Seeds.** An arm that passes D48.1 or D48.2 by less than +2.0 gets a second seed before adoption (AGENTS §6:
       several seeds below about two points).
    10. **Logging.** Each loss term's gradient norm at the feature head is logged. The novel cosine oracle on valid is
        reported for every arm.
  * **Order and stop.** P11.1b, P11.2, P11.3 and P11.4 run first (inference plus minutes of GPU for the two probes).
    - If P11.1b and P11.4 both fail, no arm is trained. D-48 reduces to [6] and [7], re-selected on CR, and a new
      decision reads the results.
    - If only P11.1b holds, only arm B is trained.
    - If only P11.4 holds, arm A is trained without [2].
* **Amendment 2 (2026-09-28, maintainer critique `debate.md`, before any code).**
  * **What was checked.** The critique proposes:
    - truncated subspace whitening;
    - adaptive spherical k-means with at least 32 points per cell;
    - semi-relaxed prototype-to-point optimal transport (OT);
    - a max-combined base background logit.

    Each proposal was checked against the evidence. The full list of fixes is in
    `docs/research/2026-09-28_d48_fix_register.md` (F11–F15, F19, F20).
  * **[3] Adaptive cell count** (adopted from the critique, F11).
    - M_c = clamp(⌊n_c / 32⌋, 2, 16) per way over its masked support points, and the same for the background over
      its points.
    - Cells are formed by spherical k-means (5 iterations from farthest-point seeds).
    - On fixed100 this gives M_fg between 5 and 16: the minimum support is 187 points, the median 764.
    - The descriptor of [4] becomes (max, mean of the top 2, mean) per class row, defined for every M ≥ 2. The
      review found the top 8 nearly flat.
  * **Descriptor space** (from the critique and the review, F12, F13, F20).
    - P11.3 and P11.4 compare four spaces:
      - raw unit features;
      - features centred on the base mean μ_base;
      - centred and projected on the top r ∈ {6, 8} eigenvectors of the pooled base-class covariance;
      - centred and truncated-whitened, (Λ_r + εI)^{−1/2} U_rᵀ(f − μ_base).
    - The space with the best P11.4 score on valid is frozen for arm A.
    - Full whitening is never used.
    - Reported: the energy share of the top r, and the share of the oracle discriminant d* inside the top-r span.
    - Recorded before the run: the evidence leans against whitening. The P10.4 base-class metric adds nothing over the
      isotropic rule, the query-whitened LDA scores 31–35 against U's 55.96, and the nuisance shares the discriminant's
      directions (κ < ρ at every r, P9).
  * **P11.5, OT assignment at inference** (adapted from the critique, F14).
    - Unbalanced entropic OT between the cell prototypes of every row (background included) and the 2,048 query points.
      - Cost 1 − cos.
      - Entropic regularisation ε ∈ {0.05, 0.1}.
      - Both marginals relaxed by KL with weight ρ ∈ {0.1, 1}.
      - Prototype masses ∝ the cell sizes, query masses uniform.
    - Logit ℓ[x, c] = log Σ_{m ∈ c} T_{m x}.
    - Selected on valid, frozen, then scored on fixed100 and random600 against U + both.
    - **Rule:** P11.5 holds at +0.5 (P9's bar for a label-free rule), and the leak-free draw is not below U + both.
      The second clause is needed because mass priors can encode object size, the leakage cue (median own share 0.38).
    - If it holds, it enters the stack before LP, re-selected per arm.
    - Reported: the share of transported mass that lands on true foreground (the critique's Gate 2).
    - Why the critique's semi-relaxed form is replaced: with Σ_j T_mj = 1/M and a column cap κ/P, the mass must reach at
      least P/κ = 1,024 points, which again forces support mass into the background of a small object.
  * **Not adopted.**
    - max(L_OT_bg, ψ · logit g). It mixes scales (ln-mass ≤ ln(2/2048) ≈ −6.9), so it fires below g = 0.5; the
      one-sided additive term of amendment 1 is kept (F15).
    - The additive expectation of 66–70 (F19).
    - Gate 1, which holds by construction (F20).
  * **Answers to the critique's questions.**
    - OT is solved between prototypes and query points only, not point to point: P9.8's missed-point purity is 0.109.
    - The covariance is pooled from the base classes. Per-block query covariance was measured and loses (label-free
      LDA 31–35).
  * **Order unchanged.**
    - P11.1b, P11.2, P11.3, P11.4 and P11.5 run first on frozen CR features.
    - The gates of amendment 1 decide the arms.
    - P11.5, if it holds, is an inference module for every arm and for CR.
* **Amendment 3 (2026-09-28, maintainer approval, before any code): the experimental design for attribution.**
  * **[2] is fully decoupled.**
    - It is trained apart, on frozen features, with the stop-gradient of amendment 1.
    - It enters every head through the one-sided additive term ψ · (−log(1 − g)). The input-channel form in arm A is
      dropped.
    - [2] is therefore a post-hoc, inference-switchable block on any checkpoint. Its effect contains no co-adaptation.
  * **Arm B is dropped.** With [2] decoupled, B is CR with [2] attached post hoc. The training slot goes to a
    **second seed of arm A** (seeds 0 and 1). The only trained factor left is the head: PEM (CR) against the
    correlation neck (A).
  * **The factorial.**
    - Four inference blocks, each on or off: [2] exclusion, [6] modality prior, P11.5 OT (if it held) and [7] LP. That
      gives 2⁴ = 16 combinations.
    - The baseline is U + both on the same cached features.
    - The 16 combinations are scored for every head (CR, A seed 0, A seed 1). Draws: valid for selection; fixed100 and
      random600 seeds 0–2 for the rules; leak-free as the second setting.
    - Each block's parameters (ψ; κ; ε and ρ; k and β) are selected once per head on valid, in the full combination,
      and frozen across the 16 combinations.
  * **Quantities per block i**, with m(S) the mIoU of the set S:
    - add-one: Δᵢ⁺ = m(base + i) − m(base);
    - leave-one-out: Δᵢ⁻ = m(full) − m(full − i);
    - exact Shapley φᵢ over the 16 combinations, which satisfies Σφᵢ = m(full) − m(base);
    - pairwise interaction Iᵢⱼ = m(base+i+j) − m(base+i) − m(base+j) + m(base).
    - All come with paired bootstrap CIs over episodes.
    - The neck's effect is A − CR in each of the 16 combinations.
  * **Pre-registered contrasts, the only ones read as conclusions.** Everything else is descriptive.
    - Add-one and leave-one-out for every block and for the neck.
    - Five interactions: ([2], LP), ([2], [6]), (OT, LP), (neck, LP), (neck, [2]).
  * **Rules, replacing D48.1–D48.3.**
    - **D48.1' block kept.** An inference block is kept if φᵢ ≥ +0.5, with a paired CI above 0 on fixed100 and
      φᵢ > 0 on all three random600 draws.
    - **D48.2' neck kept.** A − CR in the full combination holds at +1.0 for **both** seeds. A difference between the
      seeds larger than the effect voids the claim.
    - **D48.3' new base.** The best kept combination on the kept head becomes the base.
    - **D48.4' stop.** No block and no neck kept.
  * **Mechanism readings** (a gain without its mechanism is reported as unexplained):
    - [2]: the drop in false positives on points whose true label is a base class, with the recall change;
    - [6]: the rank correlation of γ_e with the episode's gain;
    - OT: own / other recall and the transported mass on true foreground;
    - LP: precision against recall;
    - neck: own / other recall and the leak-free change (density relearning).
  * **Affects.** `experiments/p11_precheck.py` (stages for the factorial and the attribution), and arm B removed from
    `run_d48.sh`.
* **Amendment 4 (2026-09-28, implementation notes, written with the P11 code and before any run).** These notes fix
  choices that amendments 1–3 left open. No threshold changes.
  * **Order inside a combination:** U + both → [2] → [6] → OT → LP. The logits of U + both are taken on unit query
    features (cosine units), which keeps D-39's decisions; this is checked on the first 20 episodes of every draw.
  * **[2].**
    - The MLP (128 → 128 → 7) is trained by plain CE on the unit features of 1,050 seeded training episodes (seed 12,
      70 per base pair), 256 points per block. Raw labels come from P5's checked copy of the sampler (one sampler call
      per block, fix F18).
    - ψ is never trained inside episodes, so the leave-target-out softmax (F3) is implemented and tested but not
      needed at test.
    - Grid ψ ∈ {0.01, 0.03, 0.1, 0.3, 1} (cosine units).
  * **[6].**
    - Text only until D-47 lands. It uses P3's ridge source with the "descriptions" prompts, its bank, and its
      support weight γ_e.
    - The prior re-ranks the foreground ways only (`text_prior.apply_prior`), so no background text row is needed
      (F9).
    - Grid κ ∈ {0.01, 0.03, 0.1, 0.3, 1, 3}.
  * **OT (P11.5).**
    - The rows are the classes of the current combination, not the cells of [3], so the OT effect is not confounded
      with the cells.
    - Class masses come from the support shares: way w gets its foreground share / N.
    - The columns are the points of both query blocks as one set.
  * **Selection.**
    - ψ, κ and (ε, ρ) are selected by coordinate ascent (two rounds, ties to the earlier value) on valid, in the full
      combination without LP.
    - LP is P7's frozen arm on CR. It is re-selected for other heads (amendment 1, change 6).
  * **Gates.**
    - P11.1b and P11.5 use the decision's "holds at g" (fixed100 CI above 0, and > 0 on the three random600 draws).
    - P11.1b's density clause is read on valid_raw, a seeded test draw with raw labels (seed 11).
    - P11.2 and P11.4 read valid, as registered.
    - The CR factorial always scores all four blocks. The gates decide which blocks go on to the arms.
  * **Descriptor spaces.**
    - μ and Σ are the moments of the base-labelled points only.
    - Truncated whitening uses δ = 10⁻³ λ₁.
    - The P11.4 probe is equivariant across the ways: one MLP for the foreground rows and one for the background row.
* **Amendment 5 (2026-09-28, maintainer request, while P11 runs and before its results are read): P11.6, context
  unmixing, and a new inference base.**
  * **Problem.** P10.5 puts the K-invariant bias in the other condition (b̂ 0.4158, bound g_other 5.48 at k = 1). P9
    left one candidate untested: the class composition of each neighbourhood.
    - The decoder of VIP-Seg's encoder interpolates coarse features from the 3 nearest coarse points [VIPSEG
      models/encoder.py], which is a linear mixture.
    - So to first order, a point of class c sitting sparsely in a context reads u_x ≈ a_x m_c + (1 − a_x) m_ctx(x) + ε_x.
      There is an MLP after the mixture, so this is an approximation to be checked.
  * **The rule (label-free).**
    - Context: c(x) = the mean unit feature of the k spatial neighbours of x in its query block (all neighbours, no
      prediction read, so no circularity).
    - Per class row c, with p_c its support row, the 2 × 2 least squares u_x ≈ α p_c + γ c(x) gives
      α_c(x) = G⁻¹[⟨u, p_c⟩, ⟨u, c⟩]₀, where G is the Gram matrix of (p_c, c(x)). α_c is the class evidence once the
      shared context is removed.
    - **Gate (identifiability).** det G ∝ 1 − cos²(p_c, c(x)): when the context equals the class (a dense, own-condition
      point), α is not identified and its variance grows as 1/(1 − cos²). The unmixed score replaces the plain score
      of row c only where cos(p_c, c(x)) < τ.
    - The unmixed score is α_c(x)·‖p_c‖², in the plain score's units, so the argmax across the rows stays comparable.
    - Everything is computed in the centred space (u − μ_base, from P11's fit), which removes the common component of
      non-negative features (‖mean‖² 0.56) and lowers the between-class cosines that make G ill-conditioned.
    - Grid k ∈ {16, 32}, τ ∈ {0.7, 0.8, 0.9} (conventions), selected on valid.
  * **New inference base, which combines only blocks with a measured gain on CR:**
    - the isotropic Euclidean mean rule (P10.4, +1.11, positive on every random600 draw);
    - "both" (D-39, +1.89);
    - LP (D-40, +0.92).

    The mean rule is the global form of unmixing: "a > ½" against a global background is "nearer to p_c than to the
    background mean" in Euclid. It has never been measured together with "both" and LP.
    - **Not combined, for lack of evidence:** an anti-collapse term (D-45 lost 5–11 points) and a density-invariant
      encoder (D-43 lost on the standard draw). Their only candidates are this unmixing and a new representation,
      which needs its own decision.
  * **P11.6.**
    1. **Oracle check of the model** (valid_raw).
       - For other-condition foreground points, regress u_x on {the query's own class mean, c(x)}, centred space.
       - Report R² and the distribution of α.
       - If the median R² < 0.5 (c), the linear mixture is wrong and the unmixing stops.
    2. **The new base:** mean rule + both + LP, against CR's U + both + LP (58.55), on every draw.
    3. **Unmixing on the new base, a switchable block** (before LP), scored on every draw with own / other recall.
  * **Rule P11.6.** Unmixing is kept if all of the following hold:
    - it holds at +0.5 over the new base (fixed100 CI above 0, and > 0 on the three random600 draws);
    - the leak-free draw does not fall;
    - other-condition recall rises while own-condition recall falls by at most 0.01 (c).

    A gain without the recall mechanism is reported as unexplained, and a trained head is not built on it. If the
    rule holds, a trained head with the same structure (τ and a context weight learned, shared by all classes) goes
    into a later decision.
  * **Affects.** `models/unmix.py` (new), `experiments/p11_precheck.py` (stage `unmix`), `tests/test_p11.py`.
