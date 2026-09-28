### D-27 — Base-class calibration of the background · `PROPOSED`, beyond the paper

* **Problem.** In a novel-class episode the background of a query block holds the fold's base classes
  (S3DIS S0 tests beam, board, bookcase, ceiling, chair, column against a background of floor, wall,
  window, door, table, sofa and clutter). A model trained on the base classes is drawn to them: COSeg
  shows false activations of base classes such as wall and door in novel-class queries [COSeg §4.3,
  Fig.5]. Evidence that correcting this pays:
  1. COSeg's Base Prototypes Calibration adds **+3.44 / +2.06** (1- / 5-shot) on top of its correlation
     model [COSeg T3], with the momentum of its base prototypes insensitive between 0.99 and 0.999
     (47.21 / 46.91 / 47.40) [COSeg T6]. Measured in COSeg's corrected setting, on a weaker base level;
     research note 2026-09-22 §5.6 predicts +0.5 to +2 at the 72 level.
  2. It uses only base-class labels, which training episodes carry anyway (research note §5.6), and it
     works on the side P0 did not touch: D-26 changed the foreground prototypes from the query's own
     pseudo-labels and failed (P0.2); this decision uses labels the model was trained on.
* **What it does.** `models/base_calibration.py`, spec 02 §12:
  * **Geometry.** Base, support and query features are compared after centring on the mean feature of
    the base training episodes and L2 normalisation, `f̃ = normalise(f − μ)`: SimpleShot's CL2N, which it
    reports to improve nearest-centroid few-shot classification over unnormalised and L2-normalised
    features [SimpleShot §3].
  * **Bank.** `b_j = normalise(mean_o normalise(Σ_{i∈o} f̃_i))` over the occurrences o of base class j
    (support and query masks) in training episodes: the limit of COSeg's per-episode masked average
    under its EMA [COSeg Eq.9–10] when the features are frozen.
  * **Calibration.** The episode's support prototypes `u_c` are built the same way. The base margin of a
    point the model assigns to foreground class c is `m_i = max_j cos(f̃_i, b_j) − cos(f̃_i, u_c)`; per
    query, the points with `m_i > 0` among the top fraction q of its foreground predictions by `m_i` move
    to the background. Background predictions and foreground logits are never changed. COSeg adds the
    base guidance through a trained layer (Eq.12), which a probe cannot do.
* **Revised before any real run (2026-09-23).** The first version added `ω · max_j s⟨f_i, b_j⟩` (raw
  unit-mean base directions at the foreground prototypes' norm) to the background logit of the model's
  own rule, arguing from P0's oracle that such directions are valid prototypes of that rule. The P1
  smoke run (5 episodes) refuted it: VIP-Seg fell from 84 to about 0 mIoU at every ω ≥ 0.8, everything
  turned background, and the AUC of the base similarity was 0.14–0.39, below chance. Two causes:
  post-ReLU features are non-negative, so any point's cosine with a raw mean direction is high; and
  P0's oracle had replaced *all* prototypes, background included, so it never set the two geometries
  against each other. The revision compares like with like (CL2N; base against the episode's own
  support prototypes) and only ever removes foreground. The bank records the mean raw cosine of the
  query features to the centre, which quantifies the shared component.
* **Second revision before any real run (same day).** The second smoke run (20 episodes) of the CL2N
  form ran cleanly but showed the absolute margin grid mis-scaled: δ = 0 / 0.05 / 0.1 / 0.2 moved
  73 / 63 / 51 / 33 % of VIP-Seg's S1 foreground predictions to the background, while only 3.7 % were
  false. Any flip beyond the false share removes mostly true foreground, so the grid became the fraction
  q of foreground predictions, anchored on that measured share and independent of the margin's scale.
* **P1, the probe** (`experiments/p1_bpc_probe.py`, `experiments/run_p1.sh`), with P0's machinery:
  the same four checkpoints, scoring rules read through hooks and checked against the model's logits on
  every episode, paired bootstrap over episodes, eval.py's protocol guard.
  * **Bank:** 1,000 seeded training episodes of the checkpoint's **own fold**, no augmentation, two
    passes (the centre μ, then the prototypes), at least 100 occurrences per base class (else it
    raises); 1,000 episodes is five times the memory of COSeg's EMA, 1/(1 − 0.995) = 200 updates
    [COSeg T6]. The bank must not contain a scored class (it raises).
  * **Selection** of q ∈ {0.5, 1, 2, 4} % on the S1 valid draw of the S1 checkpoints, a range around the
    share of false foreground measured in the second smoke run (3.7 % of VIP-Seg's S1 foreground
    predictions). **Test** of the frozen q once on fixed100: S1 checkpoints on S1, S0 checkpoints on S0.
    The probe reports, per checkpoint, the share of false foreground and the share of foreground
    predictions with a positive margin, which bound what any threshold on the margin can do.
  * **Diagnostic.** Among the model's foreground predictions, the AUC of the base margin for separating
    false foreground (ground truth background) from true foreground. It measures whether the base
    margin carries the signal at all, independently of the threshold.
* **Rules, fixed before the run.** P1.1 go: gain ≥ +0.5 with a 95 % CI above 0 on all four checkpoints,
  the low end of the predicted +0.5 to +2 and resolvable at the CI half-widths P0 measured (0.10–0.43).
  P1.2 stop (of the training-free form): no S1 checkpoint gains with a CI above 0. P1.3 otherwise.
  P1.4: AUC ≥ 0.70 on both VIP-Seg checkpoints means the base similarity flags false foreground and a
  trained calibration (COSeg Eq.12) is justified even if P1.2 fires; AUC < 0.60 on both drops D-27
  entirely; in between, report only (0.7 is the conventional threshold of acceptable discrimination,
  0.5 is chance). P1.5 collapse watch: a positive gain with a test class falling by more than 3 points,
  the expected failure when a novel class resembles a base class (board and column against wall).
* **Not in P1:** training. R3, the trained form (EMA bank during training, momentum 0.995, COSeg Eq.9–12,
  exclusion of the current ways' base prototypes during training), follows only on P1.1, P1.3 or P1.4.
* **Reporting.** A number with this calibration is the checkpoint's model plus D-27, labelled that way.
* **Affects.** `models/base_calibration.py` (new), `experiments/p1_bpc_probe.py` (new),
  `experiments/run_p1.sh` (new), `experiments/p0_em_probe.py` (the scoring rules keep `F^s`), 02 §12,
  05 §3.8h (BPC-1…10).

* **Outcome: P1.2 stop, P1.4 weak (2026-09-23, `results/phase16_p1/SUMMARY.md`).** Frozen q = 0.5 %.
  Test gains: VIP-Seg S1 −0.16 [−0.17, −0.15], ours S1 −0.07, VIP-Seg S0 −0.07 [−0.09, −0.06], ours S0
  −0.03, all CIs below 0. AUC of the base margin for false against true foreground: VIP-Seg 0.716 / 0.649
  (S1 / S0), ours 0.494 / 0.629; false share of foreground predictions 10–27 %. The training-free form is
  closed; the signal is kept as D-28's filter.
