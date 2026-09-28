### D-35 — P5: split E1's oracle gap by sampling condition and class presence, before any new module · `PROPOSED`, beyond the paper

* **Problem.** D-26…D-34 all acted on the *semantics* of the prototypes and all left the oracle gap where it was
  (12.7–14.4 points). Four facts measured or read after N2 show that the gap was never decomposed correctly, so no
  mechanism could be aimed at it (research note `docs/research/2026-09-24_condition_presence_gauge.md`):
  1. **The benchmark puts every class in two sampling conditions, and the support shows only one.** The inherited
     sampler draws `int(ratio · 2048)` points of the class a block is sampled for, then `2048 − that` points from the
     whole block [VIPSEG dataloaders/loader.py:37-58] (COSeg's "Algorithm 1"). For raw class shares π the expected
     counts are `2048 · π_s(2 − π_s)` for the sampled-for class and `2048 · (1 − π_s)π_k` for every other class, so
     the sampled-for class is at least twice as dense as anything else in its block. In an N-way episode the query
     block of way k is sampled for class k: the other way's class, when present, sits at background density
     ("other" condition). Supports are always "own". Measured on 1,500 seeded test episodes per fold
     (`experiments/c1_sampling_condition.py`, `results/c1/c1_S{0,1}.json`):

     | class (fold) | share of its query points in the other condition | kNN-16 radius own / other [m] | duplicated points own / other |
     | :--- | ---: | :--- | :--- |
     | wall (S1) | 0.235 | 0.084 / 0.121 | 0.044 / 0.001 |
     | ceiling (S0) | 0.235 | 0.072 / 0.123 | 0.064 / 0.001 |
     | floor (S1) | 0.192 | 0.072 / 0.130 | 0.060 / 0.001 |
     | the nine other test classes | 0.007–0.068 | 0.068–0.084 / 0.10–0.13 | 0.04–0.09 / ≤ 0.005 |

     Background radius 0.116–0.119 m (support and query), support foreground 0.077–0.079 m: other-condition points
     look like background by density. All foreground query points: 11.8 % other (S1), 7.3 % (S0); the second number
     is also S1's *training* classes, so 92.7 % of S1's foreground supervision is own-condition. The encoder keeps
     density: every DyPowerConv divides its kNN offsets by one scalar `torch.std` of the batch
     [VIPSEG models/encoder.py:182-189]. COSeg measures what the cue is worth at the same 2,048 points: AttMPTI
     65.52 → 41.41, QGE 73.83 → 47.02, QGPA 61.95 → 38.34 without the over-sampling (S3DIS 1-way 1-shot)
     [COSeg Tab.1]; it does not examine the N-way asymmetry, and no later paper was found that does.
  2. **The three classes with the largest other share are the three worst.** VIP-Seg's released checkpoints score
     ceiling 64.4, floor 65.3, wall 69.7, the lowest IoUs of both folds bar beam (67.2); over the 12 test classes,
     Spearman(other share, IoU) = −0.57, p = 0.051 (`results/c1/`, `results/phase16_p0/test_vipseg_S{0,1}.json`).
     P4's follow-up read the same pattern as "always background in training" (ρ = −0.55), which the protocol can never
     test; the other-condition reading can be tested at test time. Neither explains S0's beam, board, bookcase (other
     shares 0.009–0.066, oracle gaps 15–19).
  3. **P4's row-wise oracle split measured gauge, not error location.** `row_oracle_logits` writes `‖m_c‖ · O_c` into
     the chosen rows [`experiments/p4_background_probe.py:97-104`]; `O_c` is a mean of unit non-negative features, so the
     replaced rows gain a large common positive component that the other (LayerNorm) rows lack. The saved counts show
     exactly that bias (`results/phase16_p4/test_e1_counts.npz`, draw B): background row only → background predictions
     4.59 M against 3.10 M true (+48 %), foreground predicted at 0.51× its true count, every foreground recall down
     (floor 0.72 → 0.38); foreground rows only → background predictions 1.56 M (−50 %), foreground at 1.51×, table FP
     1.26× its true count. "−26 / −9 / +13.7" therefore says nothing about *where* the error is, and D-33's premise
     ("a joint shift of every prototype") is not established. `clean_bg` (+0.09) stands: it tests contamination only.
  4. **The oracle rules also carry block-level presence.** `oracle_replaced` replaces only the rows of classes present
     in a query block and keeps `M_eff` for absent ones [`experiments/r2_distill_eval.py:112-125`]; the D-29 teacher
     leaves absent rows at zero [`models/oracle_distill.py:34-43`]. Absent rows compete from another gauge, so the
     oracle can hardly predict a class the block lacks. The other way's class is absent from the other query block in
     6 % (floor), 27 % (wall), 74 % (table), 88 % (door), 93 % (window), 98 % (sofa) of episodes (`results/c1/c1_S1.json`).
     E1's precision errors sit on exactly the rarely co-occurring classes (FP/GT door 0.215, sofa 0.255, table 0.208
     against wall 0.042), and the oracle roughly halves door and sofa FP (0.104, 0.098) but *raises* table's (0.355)
     (`results/phase16_e1/test_S1_fixed100_counts.npz`): presence is part of the oracle's gain, not all of it. Its size
     is unknown.
  5. **VIP-Seg's cross-term is not class-specific.** After `reshape(72, −1)` the attention of "query b′, slot w′" is
     `softmax(Σ_{r<72} Q′_{⌊r/36⌋, 2(r mod 36)+b′}ᵀ S′_{⌊r/24⌋, 3(r mod 24)+w′} / √128)`: it sums over both queries and
     all support slots; the slot only selects projection filters. `experiments/c2_vipseg_crosscorr_check.py` rebuilds the
     inherited module (parsed, not edited): the formula matches it to 4.7e-14; changing only way 2's support moves slot
     1's attention by up to 0.45 and changing only query 2 moves query 1's by 0.075, against a typical entry of 0.0078
     (`results/c1/c2_crosscorr.txt`). The only class-specific query-conditioned path is the support self-gate; the query
     gate is shared by all rows and computed from max-pooled tokens. P3 measured the head *lowering* floor recall from
     0.874 (support prototype, no head) to 0.719 [`results/phase16_p3/SUMMARY.md`].
  6. **An untested source of support/query shift.** Training runs every BatchNorm of the encoder
     [VIPSEG models/encoder.py:230,317,320,483] and feature head on the support batch and the query batch separately,
     each with its own statistics [`models/vipseg_backbone.py:86-98`]; evaluation uses running statistics. The oracle,
     computed inside each query block, is immune to such a shift; E1 is not.
* **What it does.** P5 (`experiments/p5_condition_probe.py`, `experiments/run_p5.sh`): no training, no change to any
  model; E1 `last.pt` (`log_r2/s3dis_S1_N2_K1_point_T4_vip_b1/last.pt`) and, for arms A and C, VIP-Seg's released S1
  checkpoint. S1 only (D-22). Every arm passes `eval.py`'s protocol guard.
  * **Tags.** Query block b was sampled for local class b + 1 [VIPSEG dataloaders/loader.py:174-225]. In block b, local
    class k is *own* if k = b + 1, *other* if k ≠ b + 1 and some point carries k, *absent* otherwise.
  * **A. Condition and presence split** (fixed100). Per class, accumulated over episodes as VIP-Seg's metric: TP and GT
    for own and other points; FP in own, other and absent blocks. For E1, VIP-Seg, the current oracle (common norm, R2's
    `oracle_unit`) and a **presence-fair oracle**: absent rows replaced by the support prototype's unit-feature direction
    `n(Σ_{i∈M_c} f_i/‖f_i‖)` in the same common norm, so that absent classes compete in the oracle's gauge.
    Counterfactuals (bounds, never results): **cf-a**, each class's other-condition TP raised to
    `recall_own · GT_other`, predicted count raised by the same amount, all FP unchanged; **cf-b**, each class's FP in
    absent blocks removed.
  * **B. Condition intervention** (a fresh draw, seed 3, P4's episode machinery with scan names, 100 episodes per
    class pair). For every query block sampled for a whose scan holds ≥ 100 raw points of the other episode class c:
    V0 the protocol block; V1 the same scan sampled for c (inherited `sample_pointcloud`, `sampled_class = c`); V2 the
    same scan sampled uniformly (`random_sample = True`); the rest of the episode unchanged, each version drawn with its
    own seed. Measured on block b only: recall of c under V0 / V1 / V2, recall of a under V0 / V2, and R_own(c), c's
    recall in the blocks of the same draw sampled for c. Causal fraction `φ = (R_V1 − R_V0) / (R_own − R_V0)`. Also
    the raw class of every false positive (base / clutter / other novel), which sizes base-class confusion.
  * **C. Transductive batch statistics.** Every BatchNorm in batch-statistics mode with momentum 0 (running statistics
    unchanged), support and query forwards separate as in training; fixed100 and random600 seed 0.
  * **D. Leak-free draw.** Support and query both sampled uniformly (the inherited `random_sample = True` path, 2,048
    points: COSeg Tab.1's "w/o FG" form), seed 4, 100 episodes per pair: E1, the support rule without head
    (`F^q n(P_point)ᵀ`, P3's rule) and the oracle.
  * **E. Dual-condition prototypes, training-free.** For each support block with foreground share f, r̂ = 1 − √(1 − f)
    (the inverse of f = r(2 − r)); keep each foreground point with probability (1 − r̂)/(2 − r̂), which puts the
    foreground at background density; refill to 2,048 points with copies of uniformly drawn kept points plus the
    training jitter (σ = 0.01 m, clip 0.05 [VIPSEG dataloaders/loader.py:110-112]). The head is re-run on the sparse
    views (their `P^0` and their support slots); foreground logit of class k = max(dense, sparse). **Control:** the
    dense logits plus one constant on the foreground columns, set per episode by bisection so that the number of
    foreground predictions equals the dual arm's.
  * **Checks, on every episode of the smoke run (5 episodes per arm) and of the full run.** Every scoring rule
    reproduces its model's logits (P0's identity check), arm E's re-run head included; the index copy of the sampler
    that gives B its raw labels equals the inherited `sample_pointcloud` on every block it draws; A's own + other +
    absent counts add up to the pooled counts and the pooled mIoU equals VIP-Seg's `evaluate_metric`; batch
    statistics change the logits of every episode and leave every running statistic unchanged.
* **Rules, fixed before the run** (+1.0 = twice the single-run sd, the smallest gain worth a training run in R2/E1/N1;
  +0.5 = the smallest training-free effect worth a follow-up in P0–P4):
  * **P5.1 E-a (other-condition misses) matters:** cf-a − E1 ≥ +1.0 on fixed100. Then **P5.1a sampling is causal:**
    φ ≥ 0.5 with the episode-bootstrap CI of R_V1 − R_V0 above 0 → a decision for condition-balanced training (M1)
    and dual-condition prototypes (M2). **P5.1b not density:** φ < 0.2 → M1 only (uniform queries also show small
    fragments at background density), M2 dropped. Otherwise report.
  * **P5.2 E-b (presence) matters:** cf-b − E1 ≥ +1.0 → a decision for presence estimation (M3).
  * **P5.3 batch statistics:** C − E1 ≥ +1.0 with a paired CI above 0 on fixed100 and > 0 on random600 seed 0 → the
    test-time rule of every later arm, reported as transductive, always next to the `eval()` number.
  * **P5.4 dual prototypes:** E − control ≥ +0.5 with a paired CI above 0 → M2 is trained.
  * **P5.5 reported, no rule:** the presence-fair oracle's gain over E1, the head's gain over the support rule with and
    without the leak (D), the FP composition (B), the same split for VIP-Seg's checkpoint.
  * Neither P5.1 nor P5.2 → the condition and presence readings are dropped; the gap is own-condition (instance shift),
    the target of a trained self-support cascade (research note §5, M4).
* **Why no training first.** Each of M1–M4 costs one 2.3-GPU-h run and targets a different error; P5 costs about
  1–1.5 GPU-h (not measured) and says which error is large enough to be worth a run. It also gives the phase-16 claims
  in points 3–4 their measured size.
* **Reporting.** P5 is a diagnostic of E1; nothing in it is a result against VIP-Seg. The counterfactuals are bounds.
* **Affects.** `experiments/c1_sampling_condition.py`, `experiments/c2_vipseg_crosscorr_check.py` (new, CPU, done),
  `experiments/p5_condition_probe.py`, `experiments/run_p5.sh`, `tests/test_condition_probe.py` (new), 05 §3.8n
  (P5-1…15).

* **Outcome of the smoke run and C3 (2026-09-24): the full P5 is not run as registered.** The smoke run (5 episodes
  per arm, `results/phase16_p5/*_smoke.json`) passed every check but showed E1 at other-condition recall 0 / 1,153
  points, VIP-Seg at 1 / 1,153, and arm B's V1 (the other class made dense in its own scan) still at recall 0. A
  diagnostic found why: E1 labels the dense region of query block b with class b + 1 *whatever it is* (floor made
  dense in the "wall" block: 421 / 671 points predicted wall, 0 floor; the support rule without head: 668 / 671
  floor). C3 (`experiments/c3_query_order.py`, `results/phase16_p5/c3_query_order.json`, all 1,500 fixed100
  episodes, the two query blocks swapped, labels moving with their blocks) measured it:

  | model | mIoU, stored order | mIoU, swapped | own-class points kept → after swap | relabelled by position | support rule (both orders) |
  | :--- | ---: | ---: | :--- | ---: | ---: |
  | E1 `last` | 73.20 | **0.92** | 0.935 → 0.001 | 0.913 | 49.27 |
  | VIP-Seg released | 75.36 | **0.87** | 0.937 → 0.000 | 0.898 | 51.57 |

  VIP-Seg's head names the foreground by the **position** of the query block, which the loader fixes (the block
  sampled for class k is appended k-th [VIPSEG dataloaders/loader.py:181-222], in training and in the cached test
  episodes). Arms B and E presuppose a head that names classes by their prototypes, so their rules cannot be read on
  these checkpoints; arms A, C, D measure what they were built for but on a model whose errors are those of the
  shortcut. Every phase-16 decision measured on route B (D-25…D-34), and the oracle gap itself, was measured on
  heads with this shortcut. D-36 and D-37 trace it before any new module.
