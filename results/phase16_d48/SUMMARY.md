# Phase 16 D-48: P11 on frozen CR features, the 2⁴ inference factorial [DECISION D-48, amendments 1–4]

The run was on 2026-09-28, on a rented RTX 3090 (vast.ai) with local data, at commit `cfc3589`.

- The GPU smoke at `f522abe` found the P11.4 probe underfitting in the raw space: it scored 0 on held-out base
  episodes. The fix, standardised probe inputs, was committed as `cfc3589` before the full run.
- The full run went from 14:33 to 16:05 UTC (`d48_full.log`, `factorial_*.log`).
- No training. Every number is on CR `last.pt`, S1, in mIoU %.

**Checks.** U + both (the "base" combination) and U + both + LP reproduce D-39's and D-40's values on every draw:
- fixed100 57.63 and 58.55;
- random600 57.20 / 59.35 / 59.10 and 58.37 / 60.36 / 59.99.

The pipeline reads CR's features as before.

**Selected on valid** (coordinate ascent, full combination without LP): ψ 0.01, the smallest value of the grid, so
valid wants the exclusion off; κ 0.1; ε 0.05; ρ 0.1.

## The factorial (mIoU %)

| draw | base (U + both) | + excl | + text | + OT | + LP | all four |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| valid | 58.09 | 57.87 | 58.19 | 58.41 | 59.00 | 59.66 |
| fixed100 | 57.63 | 57.41 | 57.70 | 57.92 | 58.55 | **59.24** |
| random600 seed 0 | 57.20 | 56.92 | 57.34 | 58.04 | 58.37 | 59.42 |
| random600 seed 1 | 59.35 | 59.09 | 59.62 | 59.77 | 60.36 | 60.85 |
| random600 seed 2 | 59.10 | 58.91 | 59.16 | 58.44 | 59.99 | 60.01 |
| leak-free | 33.05 | 32.92 | 33.63 | 32.09 | 33.15 | 32.25 |
| valid_raw (seed 11) | 57.90 | 57.66 | 58.09 | 57.93 | 58.68 | 59.43 |

**Attribution on fixed100** (paired bootstrap), with the random600 values alongside:

| block | Shapley [95 % CI] | random600 | add-one | leave-one-out | D48.1' |
| :--- | :--- | :--- | ---: | ---: | :--- |
| excl | −0.17 [−0.24, −0.11] | −0.20 / −0.18 / −0.14 | −0.22 | −0.15 | not kept |
| text | +0.07 [−0.03, +0.17] | +0.13 / +0.19 / +0.13 | +0.07 | +0.04 | not kept |
| OT | +0.57 [+0.03, +1.14] | +0.99 / +0.40 / −0.30 | +0.29 | +0.81 | not kept |
| LP | +1.15 [+0.96, +1.33] | +1.31 / +1.10 / +1.22 | +0.92 | +1.34 | **kept** |

- Interactions: excl × LP +0.02 [−0.04, +0.07]; excl × text +0.07 [+0.01, +0.13]; OT × LP +0.53 [+0.29, +0.79].
- All four − base: +1.62 [+1.04, +2.24] (59.24 against 57.63). Against LP alone that is +0.69.
- On the leak-free draw all four lose 0.80 against base.

## Gates

| gate | verdict | reading |
| :--- | :--- | :--- |
| P11.1b exclusion | **fails** | excl − base on fixed100 −0.22 [−0.31, −0.14], random600 −0.27 / −0.26 / −0.19. **62 % of dense novel foreground has g > 0.5** (limit 10 %) |
| P11.2 text | **fails** | valid +0.11 [−0.03, +0.26]; spearman(γ_e, κ*_e) +0.30 |
| P11.3 cells (descriptive) | every space and cap below U + both | best raw bg32 −1.40 on valid; projected and whitened spaces −6 to −10 |
| P11.4 transfer probe | **holds** (raw space): arm A is trained by the rule | probe − U on valid +2.07 (58.04 vs 55.96), but −0.05 against U + both; leak-free 27.32 vs 33.05; other-condition recall 0.077 against U + both's 0.105 |
| P11.5 OT | **fails** | fixed100 +0.29 [−0.20, +0.79], random600 +0.84 / +0.42 / −0.65, leak-free −0.96; mass on true foreground 0.910 |

## Mechanism readings (valid_raw, raw labels)

The base learner's score g by the kind of point (mean, and the share with g > 0.5):

| point kind | points | mean g | share g > 0.5 |
| :--- | ---: | ---: | ---: |
| novel foreground, own condition (dense) | 2.68 M | 0.572 | **0.620** |
| novel foreground, other condition (sparse) | 0.33 M | 0.133 | 0.070 |
| background, base class | 1.34 M | 0.693 | 0.699 |
| background, clutter | 0.41 M | 0.269 | 0.171 |
| background, novel outside the episode | 1.38 M | 0.137 | 0.076 |

- The learner's train accuracy is 0.86 (base classes 0.70–0.96, "none" 0.90).
- Base-origin false positives: 86,878 → 84,490 with the exclusion, while true positives fall 2,144,951 → 2,127,593.
  The exclusion removes 2.4 k false positives and 17.4 k true positives.
- Background cells dominated by an episode class: 0.113.

Descriptor spaces:
- Energy share of the top r of the base covariance: 0.866 (r 6), 0.906 (r 8).
- Oracle contrast (o_c − o_bg) share inside the top-r span: 0.814 (r 6), 0.835 (r 8).

## Reading

1. **The base learner reads density, not class.**
   - g is as high on dense novel foreground (0.572) as on base background (0.693).
   - It is low on the same novel classes when they are sparse (0.133).
   - Every training block is sampled dense for a base class, so "dense ⇒ base" is what CE learns. This is the trap
     of the review's [2].3, now measured.
   - Excluding "what not to segment" therefore deletes the target itself. The effect COSeg reports (+3.44 with
     uniform sampling) is not available on this benchmark's features.
2. **Inference rules on CR's features are saturated.**
   - Only LP is kept.
   - All four blocks add +0.69 over LP on fixed100 and lose on the leak-free draw.
   - Every gain that shows on the standard draws disappears or reverses without the density cue.
3. **The correlation probe passes its registered gate but carries the density warning of A5.**
   - It beats U (+2.07) but not U + both.
   - It loses 5.7 points on the leak-free draw.
   - It lowers other-condition recall further (0.077).
   - By the rule, arm A is trained; the leak-free draw is reported with it.
4. **Text through the base bank adds almost nothing** (+0.07 Shapley), as F9 predicted.
5. **81–84 % of the oracle's discriminant lies in the top 6–8 base directions.** The novel classes are separated
   inside a space shaped by the base classes (collapse), and nothing the head does at inference changes that space.

## Notes

- Fit's held-out base scores (probe 34.30 against U 32.85) are on the last 20 % of the training episodes. These hold
  only 3 of the 6 base classes, so both numbers are about half of a per-present-class mIoU; the comparison is fair.
  Later runs should hold out every fifth episode.

## P11.6: context unmixing on a mean-rule base (amendment 5, commit `58d01f9`, 16:29–16:58 UTC)

| draw | CR U + both + LP | mean rule + both | new base: mean rule + both + LP | unmix k 32, τ 0.7 (no LP) | unmix k 16, τ 0.9 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| valid | 59.00 | 58.40 | 59.26 | 55.89 | 40.08 |
| fixed100 | 58.55 | 57.91 | 58.77 | 54.79 | 39.07 |
| random600 seed 0 | 58.37 | 56.96 | 57.74 | 54.71 | 38.89 |
| random600 seed 1 | 60.36 | 59.80 | 60.75 | 57.31 | 40.93 |
| random600 seed 2 | 59.99 | 59.52 | 60.46 | 56.03 | 37.29 |
| leak-free | 33.15 | 33.00 | 33.02 | 29.37 | 16.77 |

- **New base against CR's stack:** fixed100 +0.22 [−0.08, +0.53], random600 −0.63 / +0.39 / +0.47, leak-free −0.13.
  Not an improvement.
- **Unmixing: not kept.** The best configuration is k 32, τ 0.7, and it loses everywhere:
  - with LP against the new base: fixed100 −2.86 [−3.57, −2.14], random600 −1.72 / −2.30 / −3.24, leak-free −4.15;
  - own recall 0.778 → 0.715, other recall 0.074 → 0.049.

**Oracle check of the linear mixture** (valid_raw, k 16). The fit is v_x ≈ α·(the query's own class mean) + γ·c(x),
per point, in the centred space.

| points | median R² | α quantiles (10 / 25 / 50 / 75 / 90 %) |
| :--- | ---: | :--- |
| other condition | 0.990 | −0.254 / −0.112 / **−0.018** / 0.083 / 0.263 |
| own condition | 0.996 | — |

Gate share (τ 0.8): own 0.43, other 0.93, background 1.00.

**Reading.**
- The linear model fits: R² 0.99. But the class's own coefficient on sparse (other-condition) points is **zero at the
  median**: their features are their neighbourhood, and the sparse class leaves no measurable trace in them.
- Unmixing can only re-weight a signal that is present, so it removes the context and finds nothing under it.
- The gate cannot separate the cases either. It passes 43 % of own points, and removing their context removes their
  class, since their neighbours are their own class. That is where the large losses come from.
- The other-condition error is therefore **not recoverable by any head on CR's features**. The information is lost
  before the head, in the encoder's neighbourhood aggregation (P9's untested candidate, now measured from the other
  side).
- A fix must change what the representation keeps for sparse points: a representation trained or distilled so that a
  minority point keeps its own class signal, or an extra per-point modality.

> **Correction (2026-09-29, D-49).** The third and fourth points above over-read α. D-49's evaluation measured α for
> own-condition points as a control: the median is the same (CR −0.018), although the mean rule labels 83 % of those
> points correctly — the 2 × 2 fit cannot split the class mean from a same-class context. α ≈ 0 is therefore not
> evidence that the information is lost, and condition-balanced training raised other-condition recall from 0.18 to
> 0.46 (`results/phase16_d49/SUMMARY.md`). What stands: unmixing loses, and inference rules on CR's features do not
> recover the sparse points.
