### D-46 — P10 and M5: does training leave a base-class prototype gap to align, does text carry novel-class information on the clean base, and prototype alignment training if it does · `PROPOSED`, beyond the paper

* **Problem.** On CR, U is 55.96 on valid against the cosine oracle 80.84 (P9). The oracle is the ceiling of every
  rule that changes only the prototypes on these features, so 80+ needs δ = p̂_support − μ_query close to zero.
  - P8 splits the +24.9: foreground own condition +13.71 (instance shift, cos(s_c, o_c) 0.889), other condition
    +5.25 (density, φ 1.00), and the background.
  - Inference-time rules without labels cannot estimate δ (P6, P9: EM, LDA, projection, components, LP).
  - Changing the encoder's density handling (D-43) and the feature spectrum (D-45) both lost to CR.
  - P8.4 registered instance-alignment training as the first training arm (bound +13.71); it has not run on the
    clean base.
  - **Two earlier results bear on it; both ran on the position-shortcut base (E1), so they are mechanisms, not
    numbers for CR:**
    - D-29 (pairwise logit alignment to the oracle): the objective moved (logit-pair cosine 0.771 → 0.843) and mIoU
      did not. The reading: on a training episode the labels already give CE every decision the oracle could.
    - D-31 (text prior, P3): 0 of 192 arms gained; the best text-only foreground accuracy was 0.62.
* **Part A — P10, no training, CR `last.pt`** (`experiments/p10_align_probe.py`; `experiments/p3_probe.py`
  unchanged). Two measurements, run in parallel:
  * **P10.1 base-class gap.** On 1,000 seeded training episodes (P1's `training_episodes`, no augmentation, the base
    classes as the scored classes): U, the cosine oracle (the query's own unit class directions in every present row)
    and the model. Also reported: the mean of the prototype-alignment loss below at τ = 0.1, and the gap on valid.
    If CE already aligns the base-class prototypes, as D-29's reading says, this gap is small and an alignment loss has
    nothing to act on.
  * **P10.2 text on CR.** P3's select (192 arms on valid) and test (the frozen arm and its sibling on fixed100) on
    CR, with D-31's rules P3.0–P3.3 unchanged.
* **Part B — M5, prototype alignment training** (`models/proto_align.py`, `align_weight` and `align_tau` in the
  configuration; beyond the paper). It runs only if P10.1 holds; run_d46.sh launches it automatically.
  * **Loss.** The training loss becomes CE + λ L_align. For each query block b and each class c present in it:
    - μ_bc = n(Σ_{i: y_bi = c} n(f_bi)), the query's own unit class direction (D-29's `oracle_directions`);
    - p̂_c' = n(Σ n(f_s)) over the support's points of class c' (background = every mask-0 point of every way), U's
      rows (`p5.support_directions`);
    - L_align = mean over (b, c) of −log softmax_{c'}(⟨μ_bc, p̂_c'⟩ / τ)[c].
  * **How it differs from D-29.**
    - D-29 compared decision functions point by point, which CE already constrains on base classes.
    - L_align asks the whole query class mean to sit on its own support prototype and away from the others: a
      statement about δ itself, which point-level CE does not make.
    - The softmax over the other prototypes keeps the trivial solution (every feature on one direction) from lowering
      the loss.
    - Gradients reach both sides, with no stop-gradient.
  * **Weights.** τ = 0.1, the temperature commonly used for supervised contrastive losses (a convention, not
    measured). λ ∈ {0.25, 1.0}: arms M5-A and M5-B, λ = 1 weighting the two losses equally (a convention, as D-45's
    division by 25).
  * **Everything else is CR's** configuration and schedule: `vip_clean`, four stages, no LMA, L2 point prototypes,
    random query order, batch 1, 24,000 updates, LR halved every 7,200, 13 validations, seed 0. One seed each; the
    two arms train together on one GPU. No monitor early stop: D-45's stops on the participation ratio, which this
    loss does not target.
* **Test** (as D-45): `d43_eval.py test` on CR and each arm, `last.pt` and `best.pt`; model / U / U + both /
  U + both + LP on fixed100, random600 seeds 0–2 and the leak-free draw of this machine's data. P9 part B on each
  arm's `last.pt`.
* **Rules, fixed before the run.** "Holds at g" means: fixed100 gain ≥ g with a paired CI above 0, and > 0 on all
  three random600 draws, `last.pt`.
  * **P10.1 (gates part B).**
    - Holds: base-class gap (cosine oracle − U) ≥ 5.0 points on the training episodes. Part B runs.
    - Fails: gap < 5.0. CE already aligns the base-class prototypes, an alignment loss has nothing to act on, and
      part B does not run.
    - The threshold is a convention: one fifth of the novel gap (24.88), below which the loss would act on few
      episodes.
  * **P10.2 (text).** D-31's rules on CR's test. P3.1 go gives a trained text prior its own decision. P3.2 or P3.0
      failing stops text on the clean base.
  * **D46.1 mechanism (per arm).** The arm's valid gap (cosine oracle − U, part B) ≤ CR's 24.88 − 2.0; a smaller
    change is not an alignment effect (a convention). An arm that fails has its score change reported as unexplained.
  * **D46.2 base.** An arm with D46.1 whose U − CR's U holds at +1.0 becomes the base (the higher valid U if both).
    The stack is re-selected on it before use, as D-45.
  * **D46.3.** D46.1 holds and D46.2 fails: prototypes align on valid without a score gain; reported, and the next
    decision reads part B.
  * **D46.4.** Both arms fail D46.1: base-class alignment does not transfer to novel classes at these weights; the
    alignment line stops.
  * **D46.5 reported.** Every rule of the stack on every draw, the leak-free draw, `best.pt`, the base-class gap of
    each arm on the training episodes, the per-condition split of P8 is not re-run.
* **Cost** (not measured). Part A: about 30 min on the 3090, P3 took 22 min on the L4. Part B: two trainings about
  2.5 h together (D-45's took 2 h 24 min), then about 1.5 h of tests. If P10.1 fails, the run ends after part A.
* **Affects.** `experiments/p10_align_probe.py` (new), `models/proto_align.py` (new), `models/cascadeproto.py`
  (`align_weight`, `align_tau`; λ L_align is added to D-45's `loss_reg`, so the model contract is unchanged),
  `train.py` (`--align_weight`, `--align_tau`, run tag `_align<λ>`), `experiments/run_d46.sh`,
  `tests/test_proto_align.py`, 05 §3.8w.
* **Amendment 1 (2026-09-28, maintainer request after an external review, before any GPU run): two more
  measurements in part A, and the goal read per shot count.**
  * **Why.** The maintainer does not optimise for one shot alone.
    - Model K-shot prototypes as s_k = μ_c + b + ε_k, with b the systematic support–query shift (sampling
      condition, background composition) and ε_k the instance, of covariance Σ_η. Then
      E‖δ_K‖² = ‖b‖² + tr(Σ_η)/K.
    - At K = 1 the instance term cannot be removed by any rule that sees one support example and unlabelled query
      points; only an informative prior on the novel class can (the Bayes estimate needs m₀, Σ₀ of an unseen class).
    - So a 1-shot score at the cosine oracle asks for foundation-model-level prior knowledge.
    - A method is worth training if it removes b, which helps at every K. The instance term shrinks with K by itself.
  * **Review of the external critique** (`debate.md`, checked against the code and results):
    - The EVT argument against D-43's max-pool is right in principle: a sparse set is a subset of the dense one, so
      its max is lower. P9 measured that pathway at ψ −0.003 / −0.008, so it does not explain D-43.
    - The VICReg critique agrees with D-45's outcome. Its proposed Tr Σ_w − γ log det Σ_b would deepen the collapse
      (NC1; Σ_b of C base classes has rank ≤ C − 1) and is not adopted.
    - The EPPM (D-01, D-18) and Eq. 9 points concern route A, which the clean base does not use.
    - The 70–77 % projection sums guessed ranges and oracle bounds; it is not a measurement.
    - One proposal is new and unmeasured: a metric from base-class statistics (P10.4). P9's label-free LDA used the
      query's own total covariance, and its projection is the hard limit of this rule.
  * **P10.3 — the K-shot curve** (CR, inference; a diagnostic, not a benchmark number, since CR is trained at
    K = 1).
    - Episodes: 1,500 seeded 2-way 5-shot test episodes (`build_eval_dataset`, K = 5, seed 0, 100 per pair). Each is
      scored with its first k shots, k ∈ {1, 2, 3, 5}: the same queries, nested supports.
    - Per k: U, the model, the cosine oracle and P10.4's frozen rule (with Σ_η / k).
    - The prototype error e(k) = mean over (episode, foreground class present in a query block) of 1 − cos(p̂_k, μ),
      with μ the query's own unit class direction.
    - Fit e(k) = a + c / k by least squares. Bias share β = max(a, 0) / e(1).
    - **Bands, fixed now (conventions):**
      - β ≥ 0.5, bias-dominated: the 1-shot gap is mostly systematic, and training that removes it is the lever.
      - β ≤ 0.25, variance-dominated: the 1-shot gap is mostly the single support instance. The goal is read per
        shot count and no further 1-shot-specific prototype fix is proposed.
      - Otherwise mixed.
  * **P10.4 — a metric from base-class statistics** (no training).
    - Statistics: on P10.1's 1,000 training episodes, each block's base-class foreground (at least 16 points) gives
      the mean m of its unit features.
      - Σ_w is the pooled within-class covariance of the unit features around their block's m.
      - Σ_η is the pooled covariance of the m of one class across blocks around that class's mean.
    - Rule: `p9.lda_logits` with the support's unit-feature means (`p9.support_means`) and
      C = shrink(Σ_w + Σ_η / K, λ), λ ∈ {0.1, 0.3, 0.5, 0.7, 0.9, 1.0}.
      - With Gaussian features and prototypes this is the Bayes rule for a prototype estimated from K instances.
      - λ = 1 is the isotropic control, which separates the metric from the switch to means.
    - λ is selected on valid among λ < 1. The frozen arm is tested on fixed100 and random600 seeds 0–2 against U.
    - **P10.4 holds** when frozen − U holds at +0.5: fixed100 gain ≥ 0.5, CI above 0, and > 0 on all three random600
      draws. +0.5 is P9's bar for a label-free rule. The metric then enters the next decision's stack.
    - **It fails otherwise**, and with P9's results the metric-head line (D-42 M3) closes.
  * **Unchanged.** P10.1 still gates part B; P10.3 and P10.4 run next to P10.1 and P10.2 on the same GPU.
  * **Affects.** `experiments/p10_align_probe.py` (stages `stats`, `metric`, `kcurve`), `experiments/run_d46.sh`,
    `tests/test_proto_align.py` (PA-11…14).
* **Amendment 2 (2026-09-28, maintainer request, after P10.3 and part B were read and before this measurement ran):
  the K-curve split by sampling condition, and a correction of amendment 1's model.**
  * **Correction.** The cosine oracle compares with the query block's own class mean μ_q = μ_c + ε_q, which is itself
    one instance. So δ_K = b + ε̄_K − ε_q, and the error is E‖δ_K‖² = ‖b‖² + tr(Σ_η)(1 + 1/K), not
    ‖b‖² + tr(Σ_η)/K.
    - The fitted intercept a is ‖b‖² + tr(Σ_η); only a − c estimates ‖b‖² (in 1 − cos units, first order).
    - P10.3's registered bias share a / e(1) = 0.77 overstates the bias. The corrected share is (a − c) / e(1) = 0.54.
      It falls in the same band, and the registered reading stands, with the corrected number beside it.
  * **Why this measurement.** Part B lowered the alignment loss (0.320 → 0.224 / 0.210) without moving the base-class
    gap (14.76 → 14.26 / 14.43). P10.3's e(k) averages over (query block, present foreground class) pairs, and P8
    measured the other-condition pairs far off their support prototype (cos 0.54 against 0.889). Every support shot
    is sampled for its class, so a density shift would appear as a b that no K removes.
  * **P10.5 (descriptive; CR, the same 1,500 5-shot episodes, k ∈ {1, 2, 3, 5}).**
    - e(k) is split by condition. A pair is own when block b was sampled for class c (c = b + 1, `p8.own_mask`), and
      other otherwise.
    - Fit a + c / k per condition, with b̂ = a − c.
    - P8's condition oracles are scored per k (`p8.condition_oracle_rows`, own / other), giving the mIoU bounds g_own(k)
      and g_other(k) over U(k), with pair counts.
    - **Reading, fixed now (conventions):**
      - "Density-driven bias" if b̂_other ≥ 2 b̂_own and b̂_own / e_own(1) < 0.25.
      - "Instance bias in both" if b̂_own / e_own(1) ≥ 0.25.
      - "Mixed" otherwise.
      - The mIoU bounds are reported per k. No rule; the next decision reads them.
  * **Affects.** `experiments/p10_align_probe.py` (stage `kcond`), `tests/test_proto_align.py` (PA-15).
