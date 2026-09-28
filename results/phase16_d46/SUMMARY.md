# Phase 16 D-46: P10 (base-class gap, text on CR, K-shot curve, base-class metric, condition split) and M5 prototype alignment [DECISION D-46]

Run of 2026-09-28 on a rented RTX 3090 (vast.ai), local data (D-44 amendment 3). The full run started at 04:49 UTC
at commit `f7659be` and ended at 09:10 UTC (`d46_full.log`). Amendment 1's stages (`stats`, `metric`, `kcurve`) ran
next to P10.1 and P3 (`p10_amend.log`). Amendment 2's `kcond` (commit `e2fafc0`) ran on the idle GPU after the
trainings (`p10_kcond.log`, 09:09 UTC). One seed per arm; both arms trained together on the one GPU, exit 0 at
07:19 UTC. The results were copied back with matching md5 sums, and the instance was then stopped.

**Checks.** CR's references hold on this GPU: fixed100 model 54.84, U 55.74, U + both 57.63, U + both + LP 58.55;
valid U 55.96 (P9's value).

**Verdicts.**
- **P10.1 holds** (part B ran): base-class gap +14.76 [+13.40, +16.04], bar 5.
- **P10.2:** P3.0 holds, P3.3 in between (+0.41), no go for text.
- **P10.3 bias-dominated:** registered share 0.77, corrected share 0.54 (amendment 2).
- **P10.4 holds** at λ 0.9 (+1.46), but the covariance adds nothing measurable over the isotropic control.
- **P10.5 density-driven bias.** Own pairs have no bias (b̂ ≈ 0); other pairs are 91 % bias.
- **Part B: D46.4.** Both arms fail D46.1, and the alignment line stops.

## Part A

### P10.1 base-class gap (1,000 training episodes, base classes scored)

| checkpoint | model | U | cosine oracle | gap (oracle − U) [95 % CI] | alignment loss (τ 0.1) |
| :--- | ---: | ---: | ---: | :--- | ---: |
| CR | 74.23 | 70.68 | 85.44 | +14.76 [+13.40, +16.04] | 0.320 |
| M5-A (λ 0.25) | 74.10 | 68.41 | 82.66 | +14.26 [+13.14, +15.30] | 0.224 |
| M5-B (λ 1) | 71.84 | 65.52 | 79.95 | +14.43 [+13.38, +15.45] | 0.210 |

CE leaves a base-class gap as large as three fifths of the novel one (24.88). M5 lowers the loss it trains
(0.320 → 0.224 / 0.210) and leaves this gap unchanged. It gets there by lowering the oracle as much as U: the query
class means move toward the support prototypes, and both lose separability.

### P10.2 text on CR (P3, D-31's rules)

| rule | reading |
| :--- | :--- |
| select (192 arms, valid) | frozen `ridge_descriptions_one_k2` (+0.28 on valid) |
| P3.0 mechanism | text accuracy 0.693 (bar 0.6), alignment +0.454 [+0.429, +0.477]: holds |
| P3.3 | frozen +0.41 [+0.23, +0.60] on fixed100 (model 54.84 → 55.24); oracle-γ +3.31: in between |
| P3.4 | not claimable (selection froze a ridge arm) |
| part A | support rule 55.02, model 54.84, oracle 83.38; fixable points 0.175, boundary enrichment 1.29 |

On the clean base the text does carry class information, which it did not on E1 (accuracy 0.693 against 0.62). But
the best prior it gives still moves the score by only +0.41; an oracle mixing weight would give +3.31.

### P10.3 K-shot curve (1,500 seeded 2-way 5-shot test episodes, first k shots)

| k | 1 | 2 | 3 | 5 |
| :--- | ---: | ---: | ---: | ---: |
| e(k) = 1 − cos(p̂_k, μ) | 0.2029 | 0.1787 | 0.1710 | 0.1655 |
| model | 55.33 | 59.24 | 61.12 | 62.52 |
| U | 56.32 | 60.22 | 61.63 | 62.74 |
| P10.4 metric (Σ_η / k) | 57.40 | 61.29 | 62.74 | 63.90 |
| cosine oracle | 80.83 | 80.48 | 80.53 | 80.48 |
| gap oracle − U | 24.51 | 20.26 | 18.90 | 17.75 |

Fit e(k) = a + c/k: a 0.1556, c 0.0471, R² 0.999.
- The registered share a / e(1) is 0.77, which reads bias-dominated.
- The corrected share (a − c) / e(1) is 0.54 (amendment 2), in the same band.
- The mIoU gap fits 16.06 + 8.45 / k (R² 1.000). Five shots close 6.8 of the 24.5 points; about 16 would remain
  with infinitely many.

### P10.4 metric from base-class statistics

Statistics from 1,000 training episodes (3.46 M points, 677–960 instances per base class):
- tr Σ_w 0.0717, tr Σ_η 0.1382;
- participation ratios 11.1 (Σ_w) and 6.1 (Σ_η).

| draw | U | λ 0.1 | 0.3 | 0.5 | 0.7 | **0.9** | 1.0 (isotropic) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| valid | 55.96 | 54.24 | 55.65 | 56.39 | 56.98 | **57.56** | 57.20 |
| fixed100 | 55.74 | 53.74 | 55.26 | 56.02 | 56.59 | 57.21 | 56.85 |
| random600 seed 0 | 55.55 | 52.77 | 54.01 | 54.77 | 55.45 | 56.29 | 56.48 |
| random600 seed 1 | 57.85 | 55.15 | 56.79 | 57.62 | 58.34 | 59.16 | 59.06 |
| random600 seed 2 | 57.18 | 54.20 | 55.76 | 56.50 | 57.12 | 58.11 | 58.40 |

**Frozen λ 0.9 − U:** fixed100 +1.46 [+1.07, +1.85]; random600 +0.74 / +1.31 / +0.93. The rule holds at +0.5.

**λ 0.9 − λ 1.0** (paired; computed for this summary, not a registered rule):
- fixed100 +0.36 [−0.01, +0.72]; valid +0.35 [−0.05, +0.76];
- random600 −0.18 / +0.10 / −0.29, all CIs containing 0.

So the gain comes from scoring the support's unit-feature means with a Euclidean rule instead of U's unit directions
(+1.11 isotropic). The base-class covariance adds nothing that can be distinguished from 0. More weight on it
(λ < 0.7) costs up to 3 points.

### P10.5 K-curve split by sampling condition (amendment 2; the same 1,500 episodes)

| condition | pairs | e(1) | e(2) | e(3) | e(5) | a | c | b̂ = a − c | bias share (a − c) / e(1) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| own (block sampled for the class) | 3,000 | 0.1126 | 0.0838 | 0.0744 | 0.0675 | 0.0558 | 0.0567 | −0.0010 | ≈ 0 |
| other (class sparse in another class's block) | 1,068 | 0.4564 | 0.4453 | 0.4424 | 0.4406 | 0.4360 | 0.0201 | 0.4158 | 0.91 |

Reading (rule fixed in amendment 2): **density-driven bias.** b̂_other ≥ 2 b̂_own, and b̂_own / e_own(1) < 0.25.

The own pairs match the corrected model with ‖b‖² = 0: the intercept equals the slope (a 0.0558, c 0.0567). The
error that remains at any K is the query block's own instance, tr(Σ_η) ≈ 0.057. Nothing systematic separates an own
query from its support. The bias of P10.3 comes entirely from the other pairs:
- their error is four times the own error at one shot;
- it barely moves with K (0.456 → 0.441).

mIoU bounds per k: condition oracle − U(k). The fit to a + c/k is computed for this summary.

| bound | k 1 | k 2 | k 3 | k 5 | fit a (k → ∞) | fit c |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| own | 13.62 | 10.44 | 9.30 | 8.25 | 7.03 | 6.63 |
| other | 5.48 | 5.24 | 4.95 | 4.81 | 4.70 | 0.82 |
| foreground (both) | 16.12 | 13.06 | 11.98 | 11.17 | 9.93 | 6.19 |
| full oracle − foreground oracle (background rows) | 8.40 | 7.20 | 6.92 | 6.58 | 6.13 | 2.25 |

Notes:
- The foreground oracle reaches 72.44 at k = 1 and 73.90 at k = 5.
- The bounds are not additive: own + other exceeds the foreground bound.
- a and c are 4-point least-squares fits. Values at k → ∞ are extrapolations, not measurements.

## Part B: M5 prototype alignment

Test, mIoU %, identical episodes, `last.pt` unless marked:

| draw | CR model | CR U | CR U+both+LP | A model | A U | A U+both+LP | A best model | B model | B U | B U+both+LP | B best model |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed100 | 54.84 | **55.74** | 58.55 | 50.15 | 52.09 | 54.42 | 52.68 | 51.09 | 49.37 | 51.01 | 49.97 |
| random600 seed 0 | 54.59 | 55.55 | 58.37 | 49.88 | 52.10 | 54.14 | 52.56 | 49.17 | 49.13 | 50.40 | 48.97 |
| random600 seed 1 | 57.07 | 57.85 | 60.36 | 51.72 | 53.24 | 55.29 | 53.12 | 50.89 | 50.64 | 51.71 | 51.18 |
| random600 seed 2 | 56.11 | 57.18 | 59.99 | 51.21 | 53.02 | 55.41 | 52.13 | 50.71 | 49.49 | 51.29 | 51.68 |
| leak-free | 28.34 | 33.11 | 33.15 | 24.97 | 35.67 | 33.97 | 26.74 | 25.84 | 34.13 | 33.24 | 24.52 |

Paired comparisons, computed for this summary (`p0.paired_bootstrap`). The rules only read D46.1:

| comparison | fixed100 [95 % CI] | random600 seeds 0 / 1 / 2 | leak-free |
| :--- | :--- | :--- | :--- |
| A U − CR U | −3.65 [−4.52, −2.83] | −3.45 / −4.61 / −4.16 | +2.56 [+1.99, +3.12] |
| B U − CR U | −6.37 [−7.22, −5.60] | −6.43 / −7.21 / −7.69 | +1.02 [+0.41, +1.64] |
| A model − CR model | −4.68 [−5.70, −3.59] | −4.71 / −5.36 / −4.90 | −3.38 [−4.04, −2.72] |
| B model − CR model | −3.75 [−4.77, −2.63] | −5.42 / −6.18 / −5.40 | −2.50 [−3.16, −1.87] |
| A best model − CR model | −2.15 [−3.34, −1.02] | −2.03 / −3.96 / −3.98 | −1.61 [−2.26, −0.98] |
| B best U − CR U | −9.27 [−10.13, −8.49] | −9.27 / −10.28 / −10.08 | −4.56 [−5.23, −3.84] |

Part B on valid (P9's reader):

| quantity | CR | A | B |
| :--- | ---: | ---: | ---: |
| model | 55.17 | 51.02 | 50.48 |
| U | 55.96 | 52.88 | 49.97 |
| cosine oracle | 80.84 | 78.03 | 73.72 |
| gap (D46.1, bar 22.88) | 24.88 | 25.15 | 23.75 |
| LDA oracle (0.1) | 96.04 | 96.02 | 95.48 |
| label-free LDA (0.5) | 35.44 | 33.57 | 35.83 |
| projection (4) | 43.92 | 41.65 | 36.63 |

- **D46.1 fails for both arms.** The novel-class gap does not shrink by 2 points: A's widens (+0.27), and B's
  narrows by 1.13 only because its oracle falls 7.1 points, more than its U (−6.0).
- **D46.4: stop.** Base-class alignment does not transfer to novel classes at these weights.
- On the leak-free draw, U of both arms is above CR's (+2.56 / +1.02), while the model is below. This is reported,
  and no rule reads it.

## Reading

1. **P10.5 answers the question amendment 2 asked.** At one shot the prototype gap has three parts:
   - **Own, 13.6 points.** Pure instance difference, with no systematic shift (b̂ ≈ 0). Half of it shrinks with shots
     (c 6.6). The other half is the query's own instance (a 7.0), which no support rule can remove.
   - **Other, 5.5 points.** Almost pure bias, which does not shrink with K. A class seen sparsely in another class's
     block sits far from its support prototype (1 − cos 0.44), whatever the shot count.
   - **Background rows, 8.4 points.** Between the full and the foreground oracle.
2. **M5 failed because it acted where the gap is not.** Its loss averages all present (block, class) pairs, and
   74 % of the foreground pairs of P10.5's episodes are own (3,000 of 4,068). On own pairs, the difference from the support is instance variance, and aligning the
   means of such pairs can only shrink the between-class spread. The part B oracles fell with U (82.66 / 79.95 on
   base classes, 78.03 / 73.72 on valid). The systematic part (other pairs) is a minority of the loss and was not
   removed.
3. **A 1-shot target near the cosine oracle is not reachable by prototype fixes alone.** Even exact foreground
   prototypes reach 72.44 at one shot. That is the full own and other correction, and the own part includes the
   query's own instance.
4. **What remains addressable without a novel-class prior:**
   - the other-condition bias, a systematic and K-invariant shift of known cause (density; P8's φ 1.00), worth up
     to 5.5 points;
   - the background rows, up to 8.4;
   - the mean-rule switch of P10.4, +1.1 on U.

   Text carries class information on CR, but only +0.41 of it reaches the score.
