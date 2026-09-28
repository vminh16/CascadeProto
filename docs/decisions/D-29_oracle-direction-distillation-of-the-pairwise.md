### D-29 — Oracle-direction distillation of the pairwise decisions during training · `PROPOSED`, beyond the paper

* **Problem, from the measurements of phase 16.**
  1. **The headroom is in the directions of the prototypes the head outputs.** VIP-Seg's scoring rule is
     `L = F^q M_effᵀ` with `M_eff = Σ_t w_t M^t` [VIPSEG models/vipseg.py:152-174]. Replacing the
     direction of `M_eff` by the query's own class mean of unit features, the norm kept, gains +8.42 (S1)
     and +15.13 (S0) on VIP-Seg's released checkpoints and +20.46 / +21.76 on ours (P0, fixed100,
     `results/phase16_p0/SUMMARY.md`); with one common norm, +10.95 / +14.12 (revision below). The
     features carry the information; the head does not extract it.
  2. **No fixed rule recovers it at test time.** The model's own posterior (D-26: +0.37 S1, −1.21 S0),
     a base-class margin (D-27: −0.16 to −0.03) and their combination (D-28: the filter was never
     selected, r = 0 froze) all fail on the same checkpoints.
  3. **The head is never told where the target is.** It is trained by CE on the final logits only
     (02 §7). On a training episode the target of point 1 is computable, because the query's labels
     are base-class labels of the training fold (D-22 is not touched). QGE distils toward "optimal
     query prototypes" (+3.3, T5, 1-way) and DPA distils earlier stages toward later ones (+2.65, T3).
* **What it does.** During training only, a loss pulls each pairwise decision function of the model
  toward that of the oracle-direction rule of the training query. Spec 02 §14.
  * `O_bc = normalise(Σ_{i: y_bi = c} f_bi / ‖f_bi‖)`, stop-gradient: the direction of P0's oracle
    replacement (`ORACLE_REPLACE` in `experiments/p0_em_probe.py`, κ → ∞ in 02 §11).
  * Teacher logits `T = F^q Oᵀ`, stop-gradient: the oracle rule with one common norm for the classes.
  * For each query b and each pair of classes c < c' present in it, over the points labelled c or c':
    `cos_bcc' = cos_i(L_bic − L_bic', T_bic − T_bic')`, uncentred, with `L = L_final`.
  * `L_distill = mean over (b, c < c') present of 1 − cos_bcc'`; `L_total = CE + λ L_GMMN + β L_distill`,
    β = `distill_beta`, default 0 (the reproduction unchanged).
  * `L_distill = 0` exactly when every pairwise decision function of the model is a positive multiple of
    the oracle rule's on those points, so the two rules predict the same class between c and c'.
  * Nothing changes at evaluation: `L_distill` is computed only in `train()` mode, and the logits never
    read `query_y` (test DIS-5).
* **Revised before any run (smoke run and measurement of 2026-09-23).** The first form of this decision
  (commit `660906a`, never trained) was `1 − cos(M_eff_c, O_c)` on the effective prototype
  `M_eff = Σ_t w_t P^t`. The smoke run's diagnostics refuted it:
  1. **The loss saw what the prediction cannot.** Adding one vector v to every prototype of a query adds
     `⟨f_i, v⟩` to every class logit of point i, so softmax, CE and argmax are unchanged, and components
     of `M` orthogonal to the features change no logit at all. `cos(M_c, O_c)` depends on both. Measured on
     VIP-Seg's released checkpoints (fixed100, `results/phase16_r2_pre/`): `cos(M_eff, O)` is 0.43 / 0.30
     (background / foreground) on S1 and 0.10 / 0.13 on S0, `cos(M_c − M_c', O_c − O_c')` 0.53 / 0.44, while
     the normalised support prototypes that the head starts from score 0.83 / 0.83 raw and 0.63 / 0.66
     pairwise. The trained head moves its prototypes away from the oracle by both measures, and the
     first measure differs by a factor of three to four between the folds while the head scores 75.36
     and 71.97 on them: neither tracks what decides the prediction. The logit-space form above is invariant to both effects.
  2. **The target is the directions, not the norms.** The same measurement scored the oracle rule with
     one common norm for the present classes at +10.95 (S1) / +14.12 (S0), against +8.41 / +15.13 with
     each class's norm kept (P0's rule, reproduced to 0.01). The teacher therefore carries no norm and no
     temperature, which also removes the scale choice a KL toward softmax would need.
* **Why these choices, each from evidence.**
  * **Logit space, pairwise, cosine.** Point 1 of the revision: the only quantities that decide the
    prediction between two classes are the signs of `L_c − L_c'` on the points; a cosine over points
    compares their orientation and ignores the common shift, the scale and the feature-orthogonal part.
  * **Points of the two classes only.** The pairwise decision between c and c' decides the prediction
    on points of c or c'; on a third class's points it is irrelevant when that class wins.
  * **β = 1.** No value of β has been measured here. 1 is the weight of the paper's other auxiliary term
    (λ = 1, [PAPER Eq.26]); `1 − cos` lies in [0, 2] and the CE of a full-schedule run ends near 0.13
    (`results/phase15_full/baseline_l2/log_train.txt`), so the two terms have the same order. Not tuned:
    one value, because each arm is trained once (maintainer, 2026-09-23).
* **R2, the measurement.**
  * **Arms.** R0 = VIP-Seg's four alternating modules in our loop (`stage_type=vip`, `num_stages=4`,
    `l2norm_point_proto=true`, `use_lma=false`, ADRM), the `r1_vip4` configuration of R1; D29 = R0 with
    β = 1. R0 is the reference that R1 lacked: route B's base trained by our loop.
  * **Schedule.** The full schedule of D-12 (50 × 480 episodes, batch 4, StepLR), seed 0, because R1 has
    no learning curve for VIP-Seg's head and R0 has to be a level comparable with VIP-Seg's released
    S1 checkpoint (75.36 fixed100, P0), not a 40 % screen. `last.pt` is the headline (D-22).
  * **One training run per arm** (maintainer). Training-seed noise is therefore not measured by R2; the
    estimate is R1's: `r1_vippem` seeds 0.6827 / 0.6878, sd 0.36, so the difference of two single runs
    has sd ≈ 0.36 · √2 ≈ 0.5 [inferred from two seeds].
  * **Test, three seeds.** S1 first (D-22): fixed100 (the table's protocol, one cached draw) and three
    independent `random600` draws (seeds 0, 1, 2), with R0, D29 and VIP-Seg's released S1 checkpoint
    scored on identical episodes in each draw; paired bootstrap over episodes per draw.
  * **Mechanism diagnostics**, labels used for diagnostics only: the logit-pair cosine of 02 §14 on the
    test episodes, the prototype-space cosines per step (descriptive only), the mIoU of both oracle rules
    per model (the headroom it leaves), and `L_distill` per epoch in training for both arms (R0 computes
    it without gradient).
* **Rules, fixed before the run.**
  * R2.0 reference: R0 − VIP-Seg released on S1 fixed100 is reported. Below −2, our loop trains the head
    worse than VIP-Seg's own code and a gain over R0 is not a gain over VIP-Seg.
  * R2.1 go: D29 − R0 ≥ +1.0 on fixed100 with a paired CI above 0, positive on all three random600
    draws, and the mean logit-pair cosine on the test episodes higher for D29 than for R0: the trained
    objective transfers to the novel classes. +1.0 is twice the
    estimated sd of a single-run difference. Then S0: both arms once, same test; the S0 claim needs the
    same rule and "beats VIP-Seg" needs D29 > 72.20 on S0 fixed100.
  * R2.2 stop: fixed100 gain < +0.5, or the mean of the three random600 gains < +0.5.
  * R2.3 in between: anything else. One run per arm cannot separate it from training noise; report and
    ask the maintainer for a second training seed per arm.
  * R2.4 mechanism: a gain without a higher logit-pair cosine is not a distillation result and is
    treated as R2.3.
  * R2.5 collapse watch: a gain with any test class below R0 by more than 3 IoU points is reported.
* **Reporting.** VIP-Seg's head trained with an oracle-direction loss, labelled that way; not
  CascadeProto. The text prior (the paper's modality branch) is the next addition on top of the
  winner of R2, not part of it.
* **Affects.** `models/oracle_distill.py` (new), `models/cascadeproto.py` (`distill_beta`, `cascade` and
  `effective_prototype` for the diagnostics), `pipeline/model_api.py` (`EpisodeOutput.loss_distill`, `distill_weight`), `train.py`
  (`--distill_beta`, run-dir suffix `_distill<β>`, `L_distill` in the log), `experiments/r2_distill_eval.py`
  and `experiments/run_r2.sh` (new), 02 §14, 05 §3.8j (DIS-1…).
* **Outcome: R2.2 stop (2026-09-24, `results/phase16_r2/SUMMARY.md`,
  `docs/research/2026-09-24_r2_distill_analysis.md`).** d29 − r0 on S1: fixed100 +0.06 [−0.26, +0.39],
  random600 −0.04 / −0.04 / −0.21. The objective transferred to the novel classes (logit-pair cosine
  0.771 → 0.843, above VIP-Seg's 0.807) and the mIoU did not move: the cosine weights points by their
  squared teacher margin, so it is dominated by easy points and barely sees sign errors at the boundary,
  and on training episodes the labels already give CE every decision the oracle rule could. The oracle
  headroom (+14.43) is a transductive information gap, not a missing training signal. No implementation
  bug (β 0 / 1 in the checkpoints, teacher detached, masks right, the loss optimised 0.067 → 0.023).
  R2.0: r0 is 5.16 [4.72, 5.64] below VIP-Seg's released checkpoint on fixed100 with equally informative
  features (oracle rules 84.6 / 85.0 against 83.8 / 86.3). VIP-Seg's released S1 log reads 70.07 valid at
  6,000 updates, where r0 ends (70.26), then 72.84 at its last update (24,000) and 75.63 at its selected
  best (22,000); its S0 log peaks at update 4,000 (72.94) and ends at 68.97. The gap is training length
  plus best-of-12 selection on test-class episodes [inferred from one run each]. Closed.
