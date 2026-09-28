### D-26 — Query-side, entropy-weighted EM refinement of the prototypes · `PROPOSED`, beyond the paper

* **Problem.** Route B (D-25, R1) gives a head that works; what is added on top of it has to be new and
  has to address the error that remains. Three pieces of evidence point at the query side:
  1. **Where the one-shot error is.** Replacing the support prototypes by the query's own class means
     lifts S3DIS S0 1-way 1-shot from 66.40 to 93.89 [QGE T1]; four extra support shots are worth only
     +1.5 to +4.3 (VIP-Seg's released logs, DyPolySeg T1). The support side is bounded below by the class
     mean, the query side is not (research note 2026-09-22 §2.1).
  2. **Class-selective aggregation of query points works elsewhere.** SSP's self-support prototypes
     +3.1 / +4.0 (PASCAL 1- / 5-shot) [SSP §3.2, T13]; DPA's prototype-to-query attention +3.05 [DPA T3];
     AttMPTI's label propagation 52.27 against ProtoNet's 48.39 [AttMPTI T1] (research note §4.5).
  3. **The printed method's entropy is on the wrong object.** Eq.10–12's per-channel entropy is an even
     function of the prototype value that can only attenuate (D-02, research note §3.1). The quantity
     that is high where the model cannot tell the classes apart is the entropy of a query point's class
     posterior, H(r_i).
* **What it does.** `models/transductive.py`: on top of a model's own scoring rule `L = F^q M^T`,
  T steps of `r = softmax(L)`, `w_i = 1 − H(r_i)/ln(N+1)`, `u_c = (1/P) Σ_i w_i r_ic f_i/‖f_i‖`,
  `μ_c = ‖m_c‖ · normalise(m_c/‖m_c‖ + κ u_c)`, `L = F^q μ^T`. Spec 02 §11.
  * The update is the MAP mean direction of a von Mises–Fisher mixture with a vMF prior on each mean
    centred on the model's prototype (research note §5.2); the prior term is the anchor against the
    majority collapse of pure entropy minimisation [TIM §3.4, T4]. `‖u_c‖` is at most the confident mass
    fraction of class c, so a class the query barely contains barely moves.
  * `‖m_c‖` is kept: the trained rule is an unnormalised dot product [VIPSEG models/vipseg.py:162-164]
    [PAPER Eq.23] whose prototype norms are part of the decision. `κ = 0` or `T = 0` returns the
    model's own logits exactly (EM-1).
  * Arms of the weight: `entropy` (this decision), `none` (its ablation), `ssp` (SSP's hard thresholds,
    0.7 foreground / 0.6 background [SSP §3.2]) and `oracle` (the query labels: an upper bound, never a
    result).
* **P0, the probe, before any training** (`experiments/p0_em_probe.py`, `experiments/run_p0.sh`). No
  parameter is trained, so an arm is compared with the model **on the same episodes and weights**: the
  seed noise of training does not enter, and a paired bootstrap over episodes (2,000 resamples) gives
  the uncertainty. This answers the objection that a gain of 1–3 points is indistinguishable from
  training noise (CHANGELOG 16f: two seeds of `r1_baseline_l2` differ by 3.1).
  * Checkpoints: VIP-Seg's released S1 and S0 checkpoints (route B's head, trained by its authors) and
    our S1 and S0 baselines (`num_stages = 0`), so the effect is measured on two different feature
    extractors. Other configurations of ours raise rather than being approximated.
  * **Selection** on the S1 `valid` draw of the S1 checkpoints only [DECISION D-15] [DECISION D-22]:
    weight {entropy, none, ssp} × κ {0.5, 1, 2, 4, 8, 16} × T {1, 2, 3}. κ extends the research note's
    {0.25, 0.5, 1} because the pull on a class is at most κ·π_c, π_c its confident mass fraction: a
    foreground object that covers a tenth to a third of a query block (an estimate, **not measured**)
    would move by a few percent at κ ≤ 1. The probe records the measured π_c (`confident_mass`), so the
    grid can be checked against it before the test stage is read. T ≤ 3
    because soft k-means refinement saturates after one step [Ren §3.1.1].
  * **Test** of the frozen setting once on the fixed100 test draw: S1 checkpoints on S1, S0 checkpoints
    on S0, which is looked at here for the first time. Every checkpoint passes `eval.py`'s protocol guard.
  * **Rules, fixed before the run.** P0.1 go: gain ≥ +1.5 on both S1 checkpoints and ≥ +1.0 on both S0
    checkpoints. P0.2 stop: < +0.5 on both S1 checkpoints; the direction is dropped and base-class
    calibration is next (research note 2026-09-23 §5, direction 4). P0.3 otherwise: train with learned
    κ, expect the low end. P0.5: "the entropy weight helps" is claimed only if the frozen setting beats
    the same setting with `weight = none` with a 95 % CI above 0 on both S0 checkpoints. P0.6: an oracle
    gain below 10 on VIP-Seg's checkpoints means its features, not its prototypes, bound it. Collapse
    watch: a positive mean gain with a class falling by more than 3 points is reported (TIM §3.4). The
    go/stop thresholds are those of research note 2026-09-23 §7.1.
  * The probe also reports how the points split by weight (w < 0.5, 0.5–0.9, ≥ 0.9) and how accurate
    each bin is. That is the direct test of the claim that the posterior entropy singles out reliable
    points, which the paper asserts for its own entropy and never measures.
* **Not in P0: the text prior.** VIP-Seg's checkpoints carry no text branch, and a text prototype
  needs a trained adapter, so the text prior cannot be probed without training. It enters with R2 as
  the prior `m_c`, weighted by how well the text prototype alone predicts the support mask, the form
  that gained +1.9 / +2.0 in MM-FSS where fixed text weights gained +0.1 to +0.9 [MMFSS Eq.9–10, T3e].
  It needs its own decision before code.
* **Bug found while implementing (2026-09-23).** A first M-step normalised `u_c` to unit length before
  weighting it by κ. EM-3 (a uniform posterior must leave every prototype unchanged) failed: an entropy
  weight of 10⁻¹⁶ still produced a unit update. Such a form hands a class with a vanishing confident mass
  a full-weight update, contradicting the vMF derivation above; it was replaced by the unnormalised
  mean of unit features before any run.
* **Reporting.** A number with this refinement is VIP-Seg's head (or our baseline) plus D-26, not
  CascadeProto, and is labelled that way. Nothing trained here yet; R2 (training with the refinement
  inside the loop and per-step supervision) follows only on P0.1 or P0.3.
* **Outcome: P0.2 stop (2026-09-23, `results/phase16_p0/SUMMARY.md`).** Selection froze `ssp_k0.5_T1`
  (mean valid gain +0.28; every pseudo-label arm worsens as κ or T grows). Test, fixed100: VIP-Seg S1
  +0.37 [+0.24, +0.50], ours S1 +0.06 [−0.04, +0.16], VIP-Seg S0 **−1.21 [−1.57, −0.87]**, ours S0 −0.16
  [−0.28, −0.03]. With the query labels the same update gains +8.42 to +21.76, so the query-side headroom
  is real but the model's own posterior cannot reach it. The posterior entropy does rank points by
  reliability (accuracy 52–73 % at w < 0.5 against 88–95 % at w ≥ 0.9 on all four checkpoints), yet the
  soft entropy weight lost to SSP's hard thresholds and P0.5 is not claimable. R2 is not run; the next
  direction is base-class calibration. The module and probe stay, off every default path.
* **Affects.** `models/transductive.py` (new), `experiments/p0_em_probe.py` (new),
  `experiments/run_p0.sh` (new), 02 §11, 05 §3.8g (EM-1…14).
