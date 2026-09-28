### D-28 — EM refinement with a base-margin filter on the foreground M-step · `PROPOSED`, beyond the paper

* **Problem.** D-26 and D-27 failed for opposite reasons, measured on the same checkpoints:
  1. D-26's M-step averages the points the model assigns to a class; a false share ε of that mass drags
     the prototype toward the model's own errors, and the entropy weight comes from the same posterior,
     so its errors correlate with the model's. Every larger step (κ, T) was worse (P0).
  2. D-27's base margin is computed from information the posterior does not use (base-class labels and
     the episode's support), but it is too weak to decide single points: AUC 0.65–0.72 on VIP-Seg while
     10–18 % of its foreground predictions are false (P1).
* **Why the combination can work where the parts did not.** A weak signal loses when it *classifies*
  (one wrong flip costs one point of IoU) but can win when it *filters a mean*: the prototype is an
  average over a few hundred confident points, so excluding a true point costs variance ∝ 1/n, while
  excluding a false one reduces the bias ∝ ε. Pseudo-labels improve when a second view's errors are not
  those of the model (co-training, Blum and Mitchell 1998); the base margin is such a candidate view.
  Whether it cleans the M-step in practice is exactly what P2 measures first (P2.0).
* **What it does.** `models/transductive.py::em_step(fg_keep=...)`: at every EM step, the foreground
  M-step drops the points in the top fraction r of each query's foreground-assigned points by base
  margin (D-27's margin, recomputed from that step's logits, positive margins only); the background
  M-step and the E-step are unchanged. `r = 0` is D-26 exactly. Spec 02 §13.
* **P2, the probe** (`experiments/p2_fused_probe.py`, `experiments/run_p2.sh`), P0's and P1's machinery,
  P1's banks reused.
  * **Grid**, each value from P0/P1: weight {ssp (P0's winner), entropy (D-26's own)} × κ {0.5, 1, 2, 4, 8}
    (P0's grid without 16, which lost everywhere) × T {1, 2} (P0's best arms) × r {0, 0.1, 0.2, 0.3}
    (around P1's measured false shares, 10–27 %).
  * **Selection** on the S1 valid draw by the gain on the **VIP-Seg** S1 checkpoint only: route B builds
    on VIP-Seg's head (D-25), and our baseline's features carry no base signal (P1, AUC 0.49). Our
    checkpoints are scored for the report. **Test** of the frozen arm and of the same arm with r = 0.
  * **Mechanism diagnostic.** The false share ε of the first foreground M-step, weighted by the
    responsibilities, with and without the filter, from the query labels (diagnostic only).
* **Rules, fixed before the run.** P2.0 mechanism: at the frozen (weight, r), ε must fall to ≤ 0.75 of its
  unfiltered value on both VIP-Seg checkpoints, the reduction the expected +0.5 to +2 was estimated from;
  otherwise stop, whatever the mIoU. P2.1 go: P2.0 holds and the frozen arm gains ≥ +0.5 with a CI above 0
  on VIP-Seg S1 and S0: train the fused refinement (R2, the EM steps unrolled in training with per-step
  supervision and COSeg's EMA bank). P2.2 stop: VIP-Seg S1 has no CI above 0, or P2.0 fails. P2.3
  otherwise. P2.4: "the filter is what helps" is claimed only if frozen − unfiltered has a CI above 0 on
  VIP-Seg S0. P2.5 collapse watch as P0/P1.
* **Reporting.** VIP-Seg's head plus D-28, labelled that way; not CascadeProto.
* **Affects.** `models/transductive.py` (`fg_keep`), `experiments/p2_fused_probe.py` (new),
  `experiments/run_p2.sh` (new), 02 §13, 05 §3.8i (FUS-1…8).
* **Outcome: P2.0 fails, P2.2 stop (2026-09-23, `results/phase16_p2/SUMMARY.md`).** Selection froze the
  unfiltered arm (`ssp_k0.5_T1_r0`, +0.51 valid); the best arm per r falls monotonically, +0.51 / +0.50 /
  +0.49 / +0.47 for r = 0 / 0.1 / 0.2 / 0.3. The false share of VIP-Seg S1's first foreground M-step falls
  only to 0.85× at r = 0.3 (0.095 → 0.080) with `ssp` and not at all with `entropy` (0.116 → 0.114). The
  test reproduces P0's frozen numbers exactly. A base margin of AUC 0.65–0.72 removes false and true
  mass at nearly the same rate, so the filter's asymmetry never applies. Closed.
