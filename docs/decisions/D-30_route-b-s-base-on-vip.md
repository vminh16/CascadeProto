### D-30 — Route B's base on VIP-Seg's update count (E1) · `PROPOSED`, beyond the paper

* **Problem.** R2.0 measured our loop's route-B base r0 5.16 [4.72, 5.64] points below VIP-Seg's released
  S1 checkpoint (fixed100) with features as informative as VIP-Seg's (oracle rules 84.6 / 85.0 against
  83.8 / 86.3) (`results/phase16_r2/SUMMARY.md`). VIP-Seg's released S1 log reads 70.07 valid at 6,000
  updates, where r0 ends after its 6,000 (70.26), and 72.84 at its last update, 24,000
  [VIPSEG log_s3dis_VIPSeg/log_S1_N2_K1_0.760875/log_vipseg.txt]. D-12's null (CHANGELOG 15x) was
  measured on a headless baseline whose curve is flat from epoch 10; it does not cover the head.
* **What it does.** E1 trains r0 unchanged except for the schedule: `--batch_size 1 --lr_step_epochs 15`,
  i.e. 24,000 updates with the learning rate halved every 7,200 (VIP-Seg: 24,000 and 7,000
  [VIPSEG scripts/vipseg_s3dis.sh]), the same 24,000 training episodes as D-12, and `--valid_every 4`,
  13 validations (VIP-Seg: 12 every 2,000 updates) so that `best.pt` follows D-22's amended rule 1.
  No code change to the model; S1, seed 0, one run.
* **Test.** `last.pt` and `best.pt` of E1, `last.pt` and `best.pt` of r0, `best.pt` of d29, and VIP-Seg's
  released S1 checkpoint, on the identical episodes of fixed100 and random600 seeds 0, 1, 2
  (`experiments/r2_distill_eval.py`), paired bootstrap per draw.
* **Rules, fixed before the run** (`r2_distill_eval.py decide_e1`), on fixed100:
  * E1.1 adopt: E1 `last` ≥ 73.0, the level VIP-Seg's own run reaches at its last update (72.84):
    VIP-Seg's update count becomes route B's schedule for every later arm.
  * E1.2 not the schedule: E1 `last` ≤ 71.0, within the ≈ 1-point spread of our `last.pt` runs above
    r0 (70.20): the gap lies elsewhere; next candidates in the analysis §3 (gating bias, selection).
  * E1.3 partial: in between; adopt the schedule only if E1 − r0 has a paired CI above 0 on fixed100
    and is positive on all three random600 draws.
  * Reported, not ruled: E1 `best` against VIP-Seg released (both best-of-validation), and the
    selection gain `best − last` of E1 and r0.
* **Affects.** `experiments/run_r2.sh` (`train e1`, `eval_e1`), `experiments/r2_distill_eval.py`
  (named pairs, `decide_e1`), 05 §3.8j.
* **Outcome: E1.1 adopt (2026-09-24, `results/phase16_e1/SUMMARY.md`).** E1 `last` 73.20 on S1 fixed100,
  +2.99 [+2.62, +3.39] over r0 and +2.60 to +3.11 on every random600 draw. E1 `best` (best of 13
  validations, VIP-Seg's protocol) 75.05, against VIP-Seg's released 75.36: −0.31 [−0.63, +0.01], and
  −0.32 to +0.02 on the random600 draws. Selection gain `best − last` +1.85 [+1.43, +2.30]. Our loop
  reproduces VIP-Seg's head; R2.0's gap was training length (≈ 3.0) plus selection (≈ 1.9). VIP-Seg's
  update count is route B's schedule for every later arm.
