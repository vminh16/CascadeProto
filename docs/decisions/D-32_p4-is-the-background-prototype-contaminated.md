### D-32 — P4: is the background prototype contaminated by the episode's own classes, causally? · `PROPOSED`, beyond the paper

* **Problem, from P3 and C0.** E1's dominant error is foreground → background on floor and wall (recall 0.72 /
  0.73, oracle 0.99 / 0.95; background precision 0.861, oracle 0.977) (`results/phase16_p3/SUMMARY.md`).
  VIP-Seg's background prototype pools the mask-0 points of every support block
  [VIPSEG models/vipseg.py:108-116], and in S3DIS the other way's support block often contains this way's
  class labelled background. Measured on the raw block labels (`experiments/c0_background_contamination.py`,
  3,000 S1 test episodes): floor is present in 95 % of the other way's support blocks (18 % of their
  background), wall in 64 % (26 %), the other four classes in 3–21 % (0.5–3 %); across the 15 class pairs of
  fixed100, the pooled contamination correlates with E1's oracle gap (Spearman +0.70, p = 0.004) and
  background precision falls from 0.95 (clean pairs) to 0.69 (floor + wall). That is a correlation; pairs
  with floor or wall may be hard for other reasons. P4 intervenes.
* **Episodes with the other way's labels.** The inherited `sample_pointcloud` with `support=False` labels a
  block's points by their index among the episode's classes, with the same point sampling as the support
  call [VIPSEG dataloaders/loader.py:31-87]; calling it on the support scans gives, for every support point,
  whether it is this way's class, the other way's class, or neither. The scans are chosen as in
  `generate_one_episode` (query then support per way, no scan reused) [VIPSEG dataloaders/loader.py:174-218].
  The inherited files are called, never edited (guardrail 2). Two seeded draws of the S1 test classes,
  100 episodes per class pair each: **A** (seed 1) selects, **B** (seed 2) tests.
* **Arms, E1 `last.pt`, no training.** The head is re-run from a modified `P^0` (the head's support slots take
  whole-block features, not masks, so `P^0`'s background row is where the masks enter
  [VIPSEG models/vipseg.py:244,347]); an identity check reproduces E1's logits from the unmodified `P^0`.
  * `clean_bg`: background row of `P^0` from the mask-0 support points minus the other way's points (labels):
    the causal intervention.
  * `purify_q`: without labels, drop from way j's mask-0 points the fraction q with the highest cosine to
    way k's (k ≠ j) foreground prototype, q ∈ {0.05, 0.1, 0.2, 0.3}; q frozen on draw A.
  * Diagnostics: the purifier's AUC for other-way points among mask-0 points; background-only and
    foreground-only oracle replacements (the query's own background or foreground direction, one common
    norm), which split the oracle gap between the two kinds of row.
* **Rules, fixed before the run (draw B, paired bootstrap).**
  * P4.1 causal: `clean_bg` − E1 ≥ +1.0 with a CI above 0 and floor and wall recall both higher → the
    contamination causes part of the gap. `clean_bg` − E1 < +0.5 → it does not; stop this line.
  * P4.2 label-free fix: the frozen `purify_q` ≥ +0.5 with a CI above 0 → a training-free candidate.
  * P4.3 neck: P4.1 holds and `clean_bg` − `purify_q` ≥ +1.0 → a label-free rule leaves at least a point that
    a learned purification (trainable, supervised by the base classes' raw labels in training) could take;
    if `purify_q` is within 1.0 of `clean_bg`, the rule suffices and no neck is warranted by this evidence.
  * +1.0 and +0.5 are the thresholds of R2/E1 and P0–P3 (twice the single-run sd; the smallest effect worth a
    training run).
* **Affects.** `experiments/p4_background_probe.py`, `experiments/run_p4.sh`, 05 §3.8l (BG-…).
* **Outcome: P4.1 not causal, the line stops (2026-09-24, `results/phase16_p4/SUMMARY.md`).** The background row
  rebuilt without the other way's points (labels) gains +0.09 [+0.04, +0.14]; floor recall 0.724 → 0.733,
  wall 0.737 → 0.738. The label-free purification gains +0.02. Row-wise oracles: background only −26.07,
  foreground only −9.05, all rows +13.71: the oracle's gain is a joint, query-conditioned shift of every
  prototype, not a contaminated row. C0's correlation was confounded; no purification neck is warranted by
  this evidence (P4.3 not reached).
