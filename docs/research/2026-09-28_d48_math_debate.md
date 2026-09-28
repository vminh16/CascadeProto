# D-48 adversarial review: correlation architecture with base exclusion (2026-09-28)

Scope: `docs/decisions/D-48_*.md` and `D-47_*.md`, checked against the CR evidence (D-37 onward), the code, COSeg and MM-FSS. Labels used throughout: **[proof]** means it follows from the definitions or the code. **[evidence]** means a measured number, with its file. **[conjecture]** means not measured. New CPU numbers come from two scratch scripts, `d48_cpu_checks.py` (output in `d48_cpu_checks.json`) and `d48_ridge_overlap.py`. Both are in this scratchpad. They read `datasets/S3DIS/blocks_bs1_s1` (column 6 is the label), the stored fixed100 episodes and `results/phase16_d46/test_S1_fixed100_counts.npz`. No GPU was used and no repository file was changed.

## 0. New CPU measurements (S1)

| quantity | value | source |
| :--- | ---: | :--- |
| Wrapper with `support=False` and class list `[targets…, other base]`, `pc_augm` on, 20 seeded training episodes | 20/20 bit-identical to the inherited episode | check 1 |
| Naive wrapper that calls the sampler a second time without restoring the RNG | different points | check 1 |
| Test query blocks (600 seeded episodes, the loader's sampler): background share of points | 0.508 | check 2 |
| Share of that background by origin: base / clutter / novel classes not in the episode | **0.430** / 0.129 / **0.441** | check 2 |
| Base share by class, as a share of the background: ceiling / bookcase / column / chair / beam / board | 0.265 / 0.066 / 0.034 / 0.034 / 0.021 / 0.009 | check 2 |
| Test query blocks holding ≥ 20 base points | 0.973 | check 2 |
| Training query blocks: share of the background by origin: non-target base / novel ("none") / clutter | 0.252 / **0.633** / 0.116 | check 3 |
| Densely sampled class in training blocks: always a base class; median share of its block | 0.38 | check 3 |
| Test-class blocks that also appear in some training class list | **0.950** (sofa and table 1.000) | ridge script |
| fixed100 support foreground points: min / 5th percentile / median | 187 / 260 / 764 | check 4 |
| Points per cell at M_fg = 16: min / median | 11.7 / 47.8; 9.6 % of blocks < 20 | check 4 |
| Support background points per episode: min / median | 189 / 2,362 | check 4 |
| P11.1 estimate on U + both + LP, assuming false positives come from base points in proportion to their share | 58.55 → 61.90 (**+3.35**); on U +4.68 | check 5, *estimate, not measured* |
| Mean recall of U + both + LP, the bound if every foreground false positive were removed | 72.60 | check 5 |
| E[max of M iid N(0,1)] for M = 1 / 16 / 32 | 0 / 1.77 / 2.07 sd | ridge script |

## 1. Layer by layer

### [1] Encoder and head. **Keep, with one caveat.**
Unit features are non-negative (BN + ReLU, D-42 fact 4) and have ‖mean unit feature‖² = 0.56 (`results/phase16_p9/SUMMARY.md`). Every cosine therefore lies in [0, 1], and two random points already have a mean cosine of about 0.56. The informative part of every correlation in [4] is a narrow band near the top of [0, 1]. Centring by a base mean μ, as `text_prior.centred_unit` does, widens that band. P11.3 should compare raw and centred correlations; not measured.

### [2] Base learner. **Modify, and gate on a realistic probe instead of the oracle.**
1. **The label wrapper works, under one condition [proof and evidence].** `sample_pointcloud` consumes the RNG the same way whatever `sampled_classes` and `support` are, because both are read only after sampling (`dataloaders/loader.py:31-58, 82-89`). A subclass of `MyDataset` that overrides `generate_one_episode` works if it makes one call per block, with `support=False` and the list `[episode targets…, other base classes]`. The episode labels are then `gt ≤ N`, the support mask is `gt == way`, and the base labels are the full `gt`. This reproduced 20 of 20 augmented episodes bit for bit. The natural implementation fails: a second call for the labels shifts every later draw. `support=True` also ignores `sampled_classes`, so it cannot return base labels. Put the one-call form in D-48 and in `tests/test_base_learner.py`.
2. **Target exclusion creates a training blind spot [proof].** In training, g on target foreground is max over non-target base classes of p_base. If the learner fits, the probability mass sits on the (excluded) target, so g ≈ 0 on every foreground point. The pair (g high, y = foreground) never occurs in training, so ψ is never penalised for sending high-g points to background. At test nothing is excluded, g is a max over 6 classes instead of 4, and novel foreground gets g equal to whatever mass leaks to the nearest base class. COSeg's exclusion is on cosines, which do not renormalise. With a softmax, "exclude" must mean **leave-target-out renormalisation**: softmax over {non-target base, none}. Only then does training simulate a novel foreground point. D-48 does not say which it is.
3. **Density confound [proof plus evidence].** Every training block is sampled for a base class (check 3), so "densely sampled ⇒ base class" holds with probability 1 in training, and "none" points are never dense. At test the dense class is always novel. P8 showed that CR's features encode density: the same object has feature cosine 0.74 dense vs sparse (`results/phase16_p9/SUMMARY.md`, A1), and density moves recall from 0.245 to 0.833 (`results/phase16_p8/SUMMARY.md`). A base learner on these features can learn "dense ⇒ base", which would raise g exactly on the own-condition novel foreground, where U gets its recall. COSeg does not face this because its input is uniformly sampled. *Not measured.* It must be measured before training (§3).
4. **Semantic novel→base confusion is a minor term by volume [evidence].** Chair is 3.4 % of test background and under 2.9 % of sofa blocks. Board is 0.9 %. The base background is mostly ceiling (0.265 of the background, 10–17 % of every test block). The risk the proposal names (sofa → chair) matters less than the density term above.
5. **"None" puts pressure to collapse the novel classes [proof of the pressure; the size is conjecture].** In training, "none" is 63 % of the background and consists mostly of wall and floor, the S1 novel classes (check 3). A CE term with one "none" target on every novel point, back-propagated into the shared encoder, pulls door, wall, floor, window, table and sofa toward one direction. The features already sit in about 6 directions (participation ratio 5.56, D-45). M5 showed that a base-side loss can lower the novel oracles (valid cosine oracle 80.84 → 78.03 / 73.72, `results/phase16_d46/SUMMARY.md`). **Fix:** stop the gradient from the base CE into the encoder (BAM freezes its base learner: Lang et al., CVPR 2022, https://arxiv.org/abs/2203.15712), or report the novel cosine oracle on valid as a stop rule.
6. **The reported confusion numbers are in-sample [evidence].** 95 % of test-class blocks are also training blocks, where their novel points carried the "none" label. D48.5's "novel-to-base confusion on test blocks" is therefore close to training accuracy (at training density) and must be labelled as such.

### ψ · logit(g). **Replace with a one-sided or input-side term.**
- **Sign [proof].** logit(g) < 0 wherever g < 0.5. With ψ > 0 the term *lowers* the background logit on every point the learner calls "none". At test that is the clutter and the non-episode novel classes, 0.129 + 0.441 = 57 % of the query background (check 2). For example, wall is 15–24 % of door, window, table and sofa blocks and is background in those episodes. So the term creates false positives on more background than it removes (43 % base).
- **Training cannot resolve this.** In training the "none" background (75 %) pushes ψ down and the base background (25 %) pushes it up, so ψ settles at a compromise. The neck cannot compensate point by point because it never sees g.
- **Scale [proof].** logit is unbounded. A softmax g of 1e-7 gives −16 per unit of ψ, on logits whose scale the neck sets freely. Clamp g to [ε, 1 − ε] at least.
- **Fix.** Use ψ · softplus(logit g) or ψ · g, or feed g into the background row before the neck as COSeg Eq. 12 does (`coseg.txt` §4.3), so the neck learns the non-linearity.
- **Ablation ψ = 0.** An inference ablation measures co-adaptation damage, not the contribution of [2], because the neck's background logit was trained with the term present. The same holds for the ablation M_fg = M_bg = 1: a mean prototype makes all T sorted entries equal, which is out of the neck's training distribution. Neither ablation reads as "the model without the component".

### [3] Multi-prototypes. **Modify (fewer background cells, measure contamination) and let P11.3 decide M.**
- **A3, population [evidence].** Cells hold 11.7 points at the minimum and 47.8 at the median. The "≥ 101" floor is loose; the actual minimum on fixed100 is 187. Population is not the problem.
- **Max bias from unequal M [proof, iid model].** Background max over 32 cells vs foreground max over 16 gains about 0.30 sd from the count alone (2.07 vs 1.77). Background cells are also noisier: at the episode minimum they hold 189/32 ≈ 6 points. The bias is fixed while M is fixed, so the neck can learn an offset, but it pushes toward background exactly on low-similarity points.
- **Interaction with the "other" condition [evidence plus conjecture].** Other-condition points sit at cos 0.540 to their support direction, own-condition points at 0.889 (P8 part C). A richer background model wins these points first. The direct precedent: "both", a 3-component background, lowered other-condition recall from 0.245 to 0.166, and the recall of a class sampled uniformly from 0.463 to 0.324 (`results/phase16_p8/SUMMARY.md`, part B). Thirty-two cells is a stronger version of the same mechanism.
- **Contamination becomes a prototype [conjecture from evidence].** Floor is in 95 % of the other way's support blocks (18 % of their background) and wall in 64 % (26 %) (CHANGELOG 16s, C0). With a mean background row this was harmless: +0.09 when removed with labels (P4, measured on E1, so mechanism only). With 32 cells, floor and wall points form their own background cells. When floor or wall is a target, the query's floor points then match a background cell at own-condition cosine. P11.3 should report the share of background cells dominated by the other way's class, using support labels read as in P4.
- **The components precedent [evidence].** Foreground k-means gave 53.86 (k 2) and 52.93 (k 3) against U 55.96 (valid, `results/phase16_p9/SUMMARY.md`), falling with k. A max over 16 cells tends toward the kNN retrieval whose purity on missed points is 0.109. Everything therefore rests on [5].

### [4] Sorted top-T descriptor. **Modify the claim; keep the descriptor only if P11.4 (below) passes.**
- **The invariance claim [proof].** Every inner product is invariant to an orthogonal rotation, so that half of the claim is automatic. Sorting adds invariance to the order of the prototypes. Neither property implies that the distribution of the descriptor is the same across classes, which is what A2 needs.
- **Information content [evidence plus conjecture].** Within-class spread on base classes is tr Σ_w = 0.0717, so a point is at 1 − cos ≈ 0.036 from its block's class mean (P10.4 stats, `results/phase16_d46/SUMMARY.md`), with participation ratio 11.1. The 16 cells of one object differ along a few directions only, so the top-8 is close to flat and adds little beyond (max, mean). The informative dimension is the *level*, and the level differs by condition: own pairs sit at e(1) = 0.113, other pairs at 0.456 (P10.5). That is the density bias again, in the only coordinate the neck can read.

### [5] Correlation neck. **Keep only behind a cheap transfer probe; treat A5 as expected to fail.**
- **A2.** For the other condition, A2 is already false. For own pairs it is open: novel own e(1) = 0.113, against base block shifts of tr Σ_η = 0.138 (which pools both conditions). At the head level the evidence is negative: CR's head recovers 0.24 of the base-class gap and −0.01 of the novel gap (P10.1 / P3 part A). U scores 70.68 on base classes against 55.74 on novel ones, so the descriptor distributions the neck trains on are measurably easier.
  - **Honest counterweight:** COSeg's correlation decoder beat prototype matching, a U-like "feature optimisation" baseline, by +9.26 on novel classes (`coseg.txt` Tab. 3).
  - That result comes with a pretrained Stratified Transformer, 20,480 uniformly sampled points, linear attention and 100 ordered prototypes per class. Being uniform, it carries no density cue, so it transfers only partly.
- **A5 [argument; not measured].** Attention over 2,048 points can compute, per class row, the share of points in a high-correlation band. That share is the dense class's share of the block (median 0.38). The standard protocol rewards it: own recall 0.837 → 0.463 when density is removed (P8). Nothing in the loss opposes it, so the neck should be expected to relearn the density shortcut. This helps own pairs on fixed100 and hurts other pairs and the leak-free draw. xyz positional encoding adds a class-layout prior learned on base classes (ceiling on top, and so on); it is also not measured.
- **What the neck can fix.** On missed points the local evidence is wrong (purity 0.109, D-44), and so is the context (local seed recall 0.04–0.09 around missed points, P7a, D-40). The neck's plausible ceiling is precision and boundaries, like LP's, not the 13.6-point own term.
- **Permutation.** The cross-class mixing layer must be equivariant to the order of the foreground ways (no per-row embedding except a background/foreground type). Test it, given D-35's position shortcut.

### [6] Late modality fusion. **Modify substantially.**
1. **Already measured [evidence].** P3 on CR already used a per-episode support weight in every arm: γ_e = max(0, 2·acc − 1), from text accuracy on the support's foreground (`models/text_prior.py:102-115`, `experiments/p3_probe.py:140-148`). The +0.41 is that support-weighted prior, not "a fixed weight". The +3.31 bound is a per-episode κ chosen with query labels. D-48's "Problem" paragraph misstates this. The novelty of [6] reduces to IoU instead of accuracy, application to U, and a background column.
2. **The background text row is undefined [proof].** "argmax of t_m over {bg, c}" needs a background direction. The ridge bank holds only the 6 base classes (`test_cr.json` bank: classes [0,3,4,8,10,11]), and `prompts_for("background", "descriptions")` raises. It must be specified before code.
3. **The ridge prior is a linear read-out of base similarities, and every modality shares it [proof].**
   - The text prior is t̂_c = n(Bᵀa_c), with B the 6 base prototypes. So t_{m,c}(x) = ⟨s(x), a_{m,c}⟩ / ‖Bᵀa_{m,c}‖, with s(x) = B·n(f − μ) ∈ ℝ⁶.
   - Every modality of D-47 goes through the same bank, so Σ_m γ_m t_{m,c} is **one** linear functional of the same 6 numbers per class. Image and audio can re-weight 6 base similarities; they cannot add a direction.
   - If the transcript is exact, audio equals text, γ_audio = γ_text, and the sum doubles the text weight: double counting, by construction. More generally, summing independent γ's over-weights whatever the modalities share. A precision-weighted combination would need their covariance, which one support block cannot estimate.
4. **[6] works against [2] [evidence, ridge coefficients, "descriptions" prompts, λ 1e-3].**
   - Sofa = +0.50 chair (other terms −0.12 to −0.21).
   - Table = +0.46 chair.
   - Floor = +0.50 ceiling.
   - Door, window and wall have every coefficient ≤ +0.04 (door −0.60 ceiling, −0.84 chair; window −0.51, −0.79), so they read "unlike every base class".
   - So on sofa and table the text rewards chair-likeness, which [2] sends to background. On door, window and wall the text is itself a base-exclusion signal, which double-counts [2]. Door and window have nearly parallel coefficient vectors, so the text cannot separate that pair.
   - [2] and [6] are therefore not separately attributable, and A and B both contain [6].
5. **The TACC analogy fails on its premise.** MM-FSS's G_q comes from a 2D-aligned head frozen during meta-learning, "much less bias towards the training categories" (`mmfss.txt` §3.5). Here R is fitted on the base prototypes, which is the most base-biased source available.
6. **γ as an IoU [proof plus conjecture].**
   - For a fixed classifier, the IoU rises with the class's prevalence π: IoU = TPR·π / (π + FPR·(1 − π) + …). Support blocks are sampled dense (own condition), so γ grows with the target's support share and is measured on dense objects only. It is biased upward for other-condition query points.
   - γ is *not* in-sample with respect to the text classifier, which does not use the support. That half of the attack does not hold.
   - The variance of γ over one 2,048-point block is not measured. P11.2 should report the rank correlation of γ_e with the oracle-optimal κ_e per episode. That is the only direct test of "support IoU predicts usefulness".
7. **Scale and transfer.** t/τ has no κ, and a trained neck sets its own logit scale, so κ must be selected on each arm's valid. "R frozen from CR" is wrong for A and B, whose features differ: the bank and the ridge must be refitted on each arm's training episodes.

### [7] LP. **Keep, but re-select per arm.**
LP's gain is precision: +0.92 on fixed100, false positives removed, other-condition recall lowered (D-40). A point-attention neck and a false-positive-removing [2] both act on the same errors, so the gains will not add. LP selected on CR is known to fail on other features: B's U + both + LP fell 17 points in D-45, and "both" and LP removed M1's leak-free gain in D-43. Re-select k and β on each arm's valid, and allow "no LP".

### Loss
- **Weights.** Deep supervision with L = 2 gives the final CE 1 + ½ and the intermediate CE ½, so the episodic weight is effectively 2 against λ_b = 1 on 7 classes over roughly 8,192 labelled points (query and support). If the base CE reaches the encoder, see [2].5. The readout for ℓ_l is unspecified: shared or per layer?
- **K = 1** matches CR. Fine.
- **Rule for A6.** Log the three loss terms and the gradient norm each sends to the feature head, and stop if the base CE share exceeds the episodic share for more than 10 % of training (convention).

## 2. Hidden input assumptions not in A1–A6
- **H1. Train/test composition shift of the background.** Base is 25 % of the background in training and 43 % at test; "none" is 75 % in training and 57 % at test (checks 2 and 3). ψ and the neck's background calibration are fitted on the first mix.
- **H2. Density.** Training has P(dense ⇒ base) = 1; at test the dense class is always novel ([2].3).
- **H3. In-sample blocks.** 95 % of the test blocks were seen in training with novel points labelled "none" ([2].6).
- **H4. Coupled encoding.** Blocks are not encoded independently: batch-global std in DyPowerConv and the decoder (D-42 fact 1). The size of this effect on CR is not measured (P9's 2.4e-4 was for M1). The base learner and the neck inherit the coupling of each block with the other query block.
- **H5. Per-block axis normalisation.** XYZ is divided per axis by the block's maximum (D-42 fact 2), so an xyz positional encoding carries a block-dependent stretch. Metric xyz (channels 0–2) is rotated about z in training and not at test.
- **H6. Collapse.** In about 6 effective directions ([1], [4]), correlation descriptors carry roughly two numbers per class row.
- **H7. Raw labels for P11.1.** Stored valid and fixed100 episodes (h5) have no scan names and no raw labels (`dataloaders/loader.py:286-307`). P11.1 therefore needs either a label recovery step or a fresh seeded draw with P5's `draw_episodes` (which returns `query_raw`), which is not the registered "valid". This is not in the plan or the cost.

## 3. The rules and P11
- **P11.1 cannot fail.** The oracle removes base false positives at zero recall cost. The proportional estimate is +3.35 on the stack (+4.68 on U), against a bar of 1.0. It is also not an upper bound for a *learned* [2] in a retrained model, whose features and U change.
  - **Replace it with P11.1b (inference plus minutes of GPU).** Train the base learner as a probe on frozen CR features, on the training episodes through the one-call wrapper, with leave-target-out renormalisation. Apply it with the one-sided term, and pick ψ on a seeded validation draw.
  - Report Δ net of recall loss, and g on own-condition (dense) novel foreground against other-condition and background points. That measures H2.
  - Gate [2] on the net Δ ≥ +1.0 and on "novel foreground with g > 0.5 ≤ 10 %" (c).
- **P11.2.** Keep, but rename it for what it adds over P3 (IoU weighting, application to U). Report corr(γ_e, κ_e*), and κ selected on valid. Given P3's +0.41, expect it to fail the +0.5 bar (conjecture).
- **Add P11.4, a transfer probe for A2 and A5 (minutes of GPU, inference only).** Fit a small MLP on the [3]–[4] descriptors of *base* classes (training episodes, frozen CR features), score it on novel valid episodes against U, and on the leak-free draw. If it loses to U on novel classes while beating U on base classes, A2 fails as it did for CR's head, and arm A should not be trained. Report its own/other recall to detect density use.
- **Confounds in the arms.**
  - B − CR mixes the calibration with the change of representation caused by the base CE. Fix: detach the base learner, or add P11.1b as the frozen-feature reading.
  - [6] and [2] share the base subspace (§1 [6].4), so any rule crediting [2] must be read with [6] off. Score every arm's rule on the γ = 0 stack as well.
  - [6] and [7] "as P11 set them" were set on CR's U. Re-select them on each arm's valid, or the D48.1 comparison inherits the D-43/D-45 transfer failure.
- **Seeds.** One seed per arm. The paired CI resamples episodes, not training runs. With the repository's own single-run sd of about 0.5 (the rationale for +1.0), A − B has a seed sd of about 0.7, so +1.0 is about 1.4 sd. Either use two seeds for the arm that passes, or state that D48.2 cannot distinguish a +1 effect from seed noise (conjecture from the sd figure, not re-measured here).

## 4. Verdict: changes before code, ranked
1. **Detach the base CE from the encoder, or add a novel-oracle stop rule.** A "none" class spanning all novel points puts collapse pressure on exactly the classes being tested ([2].5; M5 precedent).
2. **Replace ψ·logit(g) with a one-sided or input-side term, clamp g, and define exclusion as leave-target-out renormalisation.** The symmetric term pushes 57 % of the test background toward foreground, and the non-renormalised exclusion never trains the (g high, foreground) case ([2].2, ψ).
3. **Replace P11.1 with P11.1b** (frozen-feature probe, net Δ, g on dense novel foreground). The oracle rule is estimated at +3.35 against a bar of 1.0, so it cannot fail (§3).
4. **Add P11.4, the descriptor-transfer probe, before training arm A.** The only admissible evidence for A2 is negative at the head level; the evidence for it is COSeg's, in a setting without density (§1 [5]).
5. **Rewrite [6].**
   - Correct the P3 history: the support weight was already used, +0.41.
   - Define the background text row, add κ selected on valid, and refit the ridge per arm.
   - Combine modalities by one coefficient vector per class, not a sum of independent γ's. Report audio only as a D-47 identity check.
   - State that [6] and [2] act on the same 6 base similarities.
6. **Re-select LP (and "both", if used) on each arm's valid.**
7. **[3]:** add M_bg ∈ {8, 16, 32} and centred-vs-raw correlations to P11.3, report background-cell contamination by the other way's class, and document the real support minimum (187).
8. **Label the ψ = 0 and M = 1 inference ablations as co-adaptation readings, and D48.5's confusion as in-sample.**
9. **Put the one-call wrapper form in the spec and its test** (verified 20/20), and cost the raw-label step of P11.1 (H7).

**What P11 must measure to decide the rest:**
- the net Δ of a realistic [2] (P11.1b) and g on dense novel foreground;
- descriptor transfer from base to novel classes and its own/other split (P11.4);
- corr(γ_e, κ_e*) (P11.2);
- max-over-cells against U, for M_bg ∈ {8, 16, 32}, with background-cell contamination (P11.3).

If P11.1b and P11.4 both fail, D-48 reduces to a re-selected [6] + [7] on CR, and no training run is warranted.
