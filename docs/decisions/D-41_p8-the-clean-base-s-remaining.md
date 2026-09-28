### D-41 — P8: the clean base's remaining gap split by sampling condition, with D-35's intervention re-run on it · `PROPOSED`, beyond the paper

* **Problem.** After D-40 the best rule scores 58.55 on S1 fixed100 against the presence-fair oracle's 80.86 on the
  same features; the whole gap is the mismatch between the support direction s_c and the query's class direction o_c
  (D-38/D-40), and P7a showed it is a region-level error (local seed recall 0.04–0.09 around missed points), which no
  query-side smoothing can repair. Two causes of the mismatch are known and call for different trained remedies:
  (i) **sampling condition**: a class sampled at background density in a block sampled for the other way ("other",
  D-35) has recall 0.16 against 0.77 for "own" points, and is 32.6 % of the missed foreground while 11.5 % of it
  (`results/phase16_p7/test_S1_fixed100.json`); the support is always "own" and the encoder keeps density (D-35);
  (ii) **instance shift**: support and query show different objects of the class. D-35 registered a causal test of (i)
  (arm B) and its reading, but it was never run on a head that names classes by prototypes (D-35 outcome: E1 used the
  query position). P8 re-runs that test and splits the oracle gap by condition on the clean base, before choosing the
  next training run.
* **What it does** (`experiments/p8_condition_probe.py`, inference only, CR `last.pt`, S1; rules on U, the unit
  geometry of D-38, as P6).
  * **A. Condition oracles** (bounds, never results). U with the query's own direction in the foreground rows of the
    blocks where the class is "own" only (`oracle_own`), "other" only (`oracle_other`), both (`oracle_fg`, P6's arm);
    rows of absent classes and the background row keep their support direction (presence-fair, one gauge). Gains
    g_own = oracle_own − U, g_other = oracle_other − U. Decisions on the S1 `valid` draw (1,500 episodes, D-15/D-22);
    fixed100 reported.
  * **B. Condition intervention** (D-35 arm B unchanged: a fresh draw, seed 3, 100 episodes per class pair, P5's
    episode machinery with scan names). For every query block sampled for a whose scan holds ≥ 100 raw points of the
    other episode class c: V0 the protocol block; V1 the same scan sampled for c (inherited `sample_pointcloud`,
    `sampled_class = c`: the same object made dense); V2 the same scan sampled uniformly; the rest of the episode
    unchanged. On block b, with U's prediction on CR's features: recall of c under V0 / V1 / V2, recall of a under
    V0 / V2 (a loses its oversampling), R_own(c) (c's recall in the blocks of the same draw sampled for c), and
    `φ = (R_V1 − R_V0) / (R_own − R_V0)` [D-35]; also cos(s_c, o_c) under each version (the mechanism: does density
    move the class direction). U + both reported beside U.
  * **C. Alignment**, valid and fixed100: cos(s_c, o_c) per query block and present class, by condition, with the
    block's recall of c under U; its distribution per class and condition and its rank correlation with recall.
  * **Not measured: text.** A text prior needs an adapter from CLIP's 512-d space to the 128-d features; the 6 base
    classes of S1 cannot fit such a map without training, so P8 says only whether the instance-shift part (the one a
    class-level prior could address) is large.
  * **Checks; a failure stops the run.** Model identity every episode; on fixed100 the model, U and U + both equal
    D-37's CR and D-39's `cr:base` / `cr:both`, and `oracle_fg` equals P6's (within 0.01 points); on valid, model and U
    equal P6's; `oracle_own` and `oracle_other` together change exactly the rows `oracle_fg` changes; P5's sampler copy
    equals the inherited sampler on every re-sampled block.
* **Rules, fixed before the run** (+1.0 = twice the single-run sd, the smallest gain worth a training run; φ bands
  those of D-35 P5.1):
  * P8.1 other-condition bound: g_other ≥ +1.0 on valid → the other-condition error is worth a training run.
  * P8.2 density is causal: φ ≥ 0.5 with the episode-bootstrap CI of R_V1 − R_V0 above 0 (D-35 P5.1a); φ < 0.2 → not
    density (P5.1b); otherwise partial.
  * P8.3 own-condition bound: g_own ≥ +1.0 on valid → the instance-shift error is worth a training run.
  * P8.4 next training run (the next decision), fixed now: **condition-balanced training** (D-35's M1: training
    queries in which the target class also appears at background density) is admissible iff P8.1 and P8.2 is causal
    or partial; **instance-alignment training** (D-29's alignment objective on the clean base, and the place where a
    text prior is tested, being instance-independent) is admissible iff P8.3. If both are admissible, the one with the
    larger bound on valid runs first; if neither, the prototype gap is not the lever and the next decision revisits
    the backbone.
  * P8.5 reported, no rule: the three oracles on fixed100; recall of a under V0 / V2; φ under U + both; cos(s_c, o_c)
    by class and condition, and its rank correlation with recall.
* **Why this design.** D-35 fixed arm B and its bands before any run; re-using them keeps the reading pre-registered.
  The oracles give the size of each error, the intervention its cause; together they choose between two one-run
  training arms without spending either. About 1 h on an L4 (not measured).
* **Affects.** `experiments/p8_condition_probe.py`, `experiments/run_p8.sh`, `tests/test_condition_split.py`,
  05 §3.8s.
* **Amendment (2026-09-26, before arm B's result was read).** The first full run stopped in arm B on an episode of the
  seed-3 draw whose two support blocks hold no background point: the background prototype is then undefined for
  every rule (U's row is the zero vector, the 3-component background has no point). Such episodes are skipped and
  counted (`skipped_no_background`); part A (both draws) had none and is unchanged. Arm B alone is re-run
  (`run_p8.sh intervene`).
* **Outcome (2026-09-26, `results/phase16_p8/SUMMARY.md`).** P8.1 holds (g_other +5.25 on valid), P8.3 holds
  (g_own +13.71; g_fg +16.12). P8.2 density causal: φ 1.001 [0.962, 1.034] over 1,045 blocks (1 episode skipped);
  the same object re-sampled dense is found as often as an own-condition class (R_V1 0.833 = R_own 0.833, from
  R_V0 0.245), and its direction moves from cos 0.55 to 0.86 of the support's. Uniform sampling lowers the own class's
  recall from 0.837 to 0.463: the model finds foreground largely by density. P8.4: both training arms admissible;
  instance-alignment training first by the rule (bound +13.71), condition-balanced second. Alignment: own cos 0.889,
  other 0.540 (Spearman with recall 0.79). Reading for D-42: density invariance is the condition for every later
  block, and the own bound is measured on points that density helps.
