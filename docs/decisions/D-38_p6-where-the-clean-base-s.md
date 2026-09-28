### D-38 — P6: where the clean base's prototype gap lies, and whether a training-free query adaptation recovers it · `PROPOSED`, beyond the paper

* **Problem.** D-37 fixed the base: the clean head trained with random query order (CR, `stage_type=vip_clean`),
  S1 fixed100 54.84, order-free. On the same checkpoint the plain support rule `F^q n(P_point)ᵀ` scores 55.02 and
  the oracle (the query's own class directions) 83.38; on the leak-free draw 28.10 / 32.15 / 61.10
  (`results/phase16_d37/SUMMARY.md`). So the backbone's features allow 83 on the standard protocol, the head adds
  nothing over the support prototypes (−0.2), and the loss is in the prototypes (+28.4). The oracle mainly restores
  recall (0.60–0.78 → 0.95–0.99; precision 0.64–0.82 → 0.71–0.93). Two things are unknown and decide the design
  of the next trained module: whether the gap sits in the background row or the foreground rows, and whether a rule
  that does not read query labels recovers any of it. Every earlier attempt at either (P0 EM, P4 row oracle, D-31
  text, D-33/34 neck) was measured on shortcut heads, P4 also in a mixed gauge, so none of them answers this.
* **What it does** (`experiments/p6_prototype_probe.py`, inference only, CR `last.pt`, S1).
  * **One geometry for every row.** All rules are `L = F^q Rᵀ` with unit rows R in the oracle's geometry (normalised
    sum of unit features): the support rows are `support_directions` [D-35], the oracle rows `oracle_directions`
    [D-29]. Rows of classes absent from a query block keep their support direction (the presence-fair form of D-35),
    so no row is advantaged by its norm (the P4 artefact, D-35 point 3). The base `U` is this rule with support rows
    only; the model and the raw support rule (55.02) are reported next to it.
  * **(a) Row oracle.** U with the oracle direction in the background row only (`oracle_bg`), in the present
    foreground rows only (`oracle_fg`), in every present row (`oracle_all`). Bounds, never results.
  * **(b) Entropy-gated self-support** (SSP's idea [Fan et al., ECCV 2022] with CascadeProto's entropy selection),
    per query block, no labels: predict with R; for each class c take the points predicted c whose entropy of
    `softmax(L)` is within the lowest fraction ρ of them (at least 16 points, otherwise the row is kept); S_c = the
    normalised sum of their unit features; `R_c ← n(α R_c + (1 − α) S_c)` for the rows in the set r; repeat T times.
    Grid ρ ∈ {0.25, 0.5, 1.0}, α ∈ {0.25, 0.5, 0.75}, T ∈ {1, 2}, r ∈ {background, foreground, all}: 54 arms.
  * **(c) Multi-component background.** Spherical k-means (k-means++ seeding, fixed seed, 20 iterations) on the
    unit features of all support background points; background logit = max over the k centroid directions,
    k ∈ {3, 5}; foreground rows unchanged.
  * **Selection.** (b) and (c) each freeze the arm with the largest mIoU gain over U on the S1 `valid` draw (1,500
    episodes of the test classes, D-15/D-22); ties go to the earlier arm of the grid. The frozen arms, U, the model,
    the raw support rule and the three oracles are then scored once on fixed100, random600 seeds 0–2 and P5's
    leak-free draw (seed 4); paired bootstrap over episodes.
  * **Checks, every episode; a failure stops the run.** The model's logits are reproduced by its scoring rule
    (P0's identity check); on fixed100 the model's count-based mIoU equals VIP-Seg's `evaluate_metric` and D-37's
    CR number (54.84) within 0.01.
* **Rules, fixed before the run** (+1.0 = twice the single-run sd, the smallest gain worth a training run; +0.5 =
  the smallest training-free effect worth a follow-up, as in P0–P4):
  * P6.1 location (fixed100, bounds): with g_x = oracle_x − U, **foreground** if g_fg ≥ ⅔ g_all, **background** if
    g_bg ≥ ⅔ g_all, **joint** otherwise (both ≥ ⅔: either row suffices, reported as such). The same split on the
    leak-free draw is reported.
  * P6.2 self-support go: frozen (b) − U ≥ +1.0 on fixed100 with a paired CI above 0 and > 0 on all three random600
    draws → D-39 trains an entropy-gated self-support cascade on the clean base (CascadeProto's entropy and cascade
    ideas placed where the gap is). P6.3 multi-background go: the same test for (c) → D-39 includes it.
  * P6.4 stop: both (b) and (c) below +0.5 on fixed100 → training-free query adaptation does not recover the gap;
    D-39 must be a trained module, and P6.1 says which rows it acts on.
  * Otherwise (between +0.5 and the go rule): reported; D-39 decides with P6.1.
  * P6.5 reported, no rule: every arm on the leak-free draw, per-class IoU, recall and precision, the share of the
    oracle gap closed, the frozen parameters.
* **Why no training first.** The trained module of D-39 costs about 2 GPU-h per arm; P6 costs about 45 minutes on
  an L4 (not measured) and fixes which rows it must act on and whether query-side selection is the mechanism.
* **Affects.** `experiments/p6_prototype_probe.py`, `experiments/run_p6.sh`, `tests/test_prototype_probe.py`,
  05 §3.8p.
* **Outcome (2026-09-25, `results/phase16_p6/SUMMARY.md`).** P6.1 joint on both protocols (fixed100: oracle bg
  +6.14, fg +16.31, all +25.12 over U 55.74). P6.2 go: `ssp_bg_T2_a0.25_r1` +1.19 [+0.85, +1.52], random600 +1.34 /
  +1.58 / +1.13; P6.3 go: km3 +1.32 [+0.95, +1.69], +1.05 / +0.96 / +1.04. Each closes ~5 % of the gap and neither
  moves the leak-free draw. Every self-support arm that helps acts on the background row only, with ρ = 1 (no
  entropy selection); foreground self-support lowers the score (best −0.35 on valid). U alone beats the trained
  model by +0.9.
