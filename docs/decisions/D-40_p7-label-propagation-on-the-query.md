### D-40 — P7: label propagation on the query's own point graph, with its preconditions measured · `PROPOSED`, beyond the paper

* **Problem.** After D-39 the best honest rule is CR's features with U + both: 57.63 on S1 fixed100 against the
  presence-fair oracle's 80.86 (`results/phase16_d39/SUMMARY.md`). The remaining gap is a foreground recall gap: U +
  both has foreground recall 0.62–0.84 and precision 0.65–0.82 per class, the oracle recall 0.94–0.99
  (`results/phase16_d39/test_S1_fixed100.json`). Every query-side rule tried so far (P0 EM, P6 self-support, D-39
  trained self-support) re-estimates a prototype from the query's predicted region; the research note
  `docs/research/2026-09-25_composition_theory.md` §4 shows why that cannot restore recall: a missed point lies outside
  every cone and never enters the average, and the foreground regions are 20–45 % contaminated. Label propagation on
  the query's own graph acts per point: a missed point is relabelled when its neighbourhood votes for its class, i.e.
  it uses spatial and feature continuity inside the query, which the prototypes do not carry. Its preconditions are
  known and measurable (note §5, §10): the missed points' neighbourhoods must be homophilous above a vote threshold
  (0.54–0.69 per class from U's global recall and false-positive rate on fixed100, higher locally), correct seeds must
  be reachable inside the class, and β must not over-smooth; the (a, c)-expansion of Wei et al. [ICLR 2021] is the
  population form of the same condition.
* **What it does** (`experiments/p7_propagation_probe.py`, inference only, CR `last.pt`, S1).
  * **Seeds.** Y₀ = one-hot of the U + both prediction (D39.1's rule on U's rows, as in D-39), `[P, N+1]` per query
    block. Hard seeds, so the vote condition of the note applies as derived and no temperature is introduced.
  * **Graphs**, per query block (P = 2,048 points), never across blocks (order-free, as D37-T5): the k nearest
    neighbours of each point, itself excluded; affinity a_ij for j ∈ NN_k(i), W = A + Aᵀ (symmetric, zero diagonal),
    S = D^{−1/2} W D^{−1/2} with D = diag(W 1) and zero rows for isolated points [Zhou et al., NIPS 2003; Iscen et al.,
    CVPR 2019, Eq. 9 and §3, re-read]. Three graphs: `xyz`, neighbours in the block's metric coordinates (channels
    0–2 of `query_x`, metres from the block minimum [VIPSEG dataloaders/loader.py:65-66]), a_ij = 1; `xyzf`, the same
    neighbours with a_ij = [u_iᵀu_j]₊^γ, u the unit features, γ = 3 (Iscen's weight on a spatial graph: edges across a
    feature boundary are down-weighted); `feat`, neighbours by the cosine of u with a_ij = [u_iᵀu_j]₊^3 (Iscen's graph
    as published there with k = 50, γ = 3, α = 0.99).
  * **Propagation.** Z = (1 − β)(I − βS)^{−1} Y₀, the fixed point of F ← βSF + (1 − β)Y₀ [Zhou et al.; Iscen Eq. 6,
    10], solved exactly (Cholesky in float64; I − βS is positive definite for β < 1); prediction argmax_c Z_ic.
    Isolated points keep their seed.
  * **Grid**: graph {xyz, xyzf, feat} × k {8, 16} × β {0.5, 0.8, 0.9, 0.99}, 24 arms. k: the note's plan; D-35
    measured kNN-16 radii of 0.07–0.13 m, one object part. β: expected walk length β/(1 − β) of 1, 4, 9 and 99 steps;
    0.99 is Zhou's and Iscen's value.
  * **P7a diagnostics** (labels read; measurements, never used for selection), for each of the six (graph, k), per
    test class and sampling condition (own / other of D-35; the background as one set): recall of Y₀; weighted
    homophily p_i = Σ_j W_ij [y_j = y_i] / Σ_j W_ij of hit and of missed points; around missed points, the local seed
    recall a_i (share of i's same-class neighbour weight predicted y_i) and the local false rate b_i (share of its
    other-class neighbour weight predicted y_i); the one-hop vote argmax_c Σ_j W_ij [ŷ_j = c]: missed points it would
    fix, hit points it would break; reachability: missed points whose same-class connected component holds a
    correctly predicted seed. Per propagation arm: points fixed and broken, with the mean homophily of each set.
  * **Selection.** The arm with the largest mIoU gain over Y₀ on the S1 `valid` draw (1,500 episodes of the test
    classes, D-15/D-22) is frozen; ties go to the earlier arm of the grid.
  * **Test.** Only if P7.1 passes: the frozen arm on Y₀ (`lp`) and, reported, the same arm on U's prediction (`lp_u`,
    composition), with the model, U and U + both, on fixed100, random600 seeds 0–2 and the leak-free draw (seed 4);
    paired bootstrap over episodes. The P7a diagnostics are computed on every draw whether or not the gate passes
    (they read Y₀ only).
  * **Checks; a failure stops the run.** Model identity (P0's check) every episode; the solve's residual
    ‖(I − βS)Z − (1 − β)Y₀‖_∞ ≤ 10⁻⁸ every block; on fixed100 the model's count-based mIoU equals VIP-Seg's metric
    and D-37's CR (54.84), U and U + both equal D-39's `cr:base` (55.74) and `cr:both` (57.63), each within 0.01
    points.
* **Rules, fixed before the run** (paired bootstrap over episodes; "holds at g" = fixed100 gain ≥ g with the CI
  above 0 and > 0 on all three random600 draws; the reference is Y₀ = U + both):
  * P7.1 gate (valid): frozen arm − Y₀ ≥ +0.5 on valid → test. Otherwise P7 stops without scoring any propagation
    arm on a test draw, and P7a says which precondition fails.
  * P7.2 adopt: `lp` − Y₀ holds at +0.5 (the training-free threshold of P0–P6) → propagation joins the inference
    pipeline (CR + U + both + LP), the reference of every later arm.
  * P7.3 trained form: holds at +1.0 (twice the single-run sd, as P6.2) → a training run with the propagation in the
    loop is admissible for the next decision.
  * P7.4 stop: `lp` − Y₀ < +0.5 on fixed100. Between the two: reported, no adoption.
  * P7.5 density control, with its reading fixed now: if P7.2 holds and the leak-free gain is ≤ 0, the gain is
    recorded as protocol-dependent (it may pass through the density cue of D-35) and not claimed as a recall
    mechanism.
  * P7.6 mechanism (L5 of the note): on fixed100 the frozen arm fixes more foreground points than it breaks, and the
    fixed points' mean homophily exceeds the broken points'. If P7.2 holds without this, the gain is recorded as
    unexplained.
  * P7.7 reported: P7a per graph, class and condition; recall and precision per class and condition (P5's split
    counts); `lp_u` − U next to `lp` − Y₀ (overlap with the background rules).
* **Why this design.** The note (§8) proposed a gate on a label-based vote bound; that bound needs the same solve as
  the method, so it saves nothing. The gate is the method's own valid gain instead, which keeps the test draws unread
  when propagation fails. P7 is one inference pass, no training (about 30 min on an L4 for valid and 20 min for the
  test draws, not measured).
* **Known limits.** Duplicated points (4–9 % of own-condition points, D-35) are their own nearest neighbours in `xyz`
  and `feat`; they are kept as the loader produces them. Hard seeds discard confidence; a soft variant would add a
  temperature that no evidence fixes.
* **Affects.** `experiments/p7_propagation_probe.py`, `experiments/run_p7.sh`, `tests/test_propagation_probe.py`,
  05 §3.8r.
* **Outcome (2026-09-26, `results/phase16_p7/SUMMARY.md`).** P7.1 gate pass (`lp_feat_k16_b0.99`, +0.91 on valid).
  P7.2 adopt: `lp` − Y0 +0.92 [+0.72, +1.13] on fixed100 (58.55), random600 +0.91 / +1.03 / +1.42; P7.3 not reached.
  P7.5 leak-free +0.17. P7.6 unexplained: foreground fixed 53,173 against broken 41,824, but the fixed points' mean
  homophily (0.944) is not above the broken points' (0.947); the gain is mostly precision (false positives removed),
  not recall. P7.7: fully additive with the background rules (`lp_u` − U +0.92). P7a: homophily is high (0.93–0.96
  around missed points) but the seeds around a missed own-condition point are almost all wrong (local seed recall
  0.04–0.09): recall errors are whole regions. Other-condition foreground points have recall 0.162 (own 0.774) and are
  32.6 % of the missed foreground points while 11.5 % of the foreground; propagation lowers their recall further.
