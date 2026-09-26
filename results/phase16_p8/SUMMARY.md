# Phase 16 P8: the clean base's gap split by sampling condition, and D-35's intervention re-run [DECISION D-41]

VM runs of 2026-09-26 (NVIDIA L4): part A 05:03–05:15 UTC (commit `f4c3593`); part B stopped on an episode whose
supports hold no background point, was amended (D-41 amendment, commit `f220854`) and re-run alone 07:37–07:52 UTC
(1 of 1,500 episodes skipped). CR `last.pt`, S1, inference only, rules on U. Checks passed: model identity, the
oracle partition, VIP-Seg's metric, D-37's CR, D-39's U and U + both, P6's oracle_fg and valid numbers.

## A. Condition oracles (mIoU %; bounds, never results)

| draw | U | oracle_own | oracle_other | oracle_fg | g_own | g_other |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| valid | 55.96 | 69.68 | 61.21 | 72.08 | **+13.71** | **+5.25** |
| fixed100 | 55.74 | 69.49 | 60.91 | 72.05 | +13.74 | +5.17 |

Other-condition points are 11.5 % of the foreground points but carry a +5.2 bound; the gain is concentrated on wall
(IoU 58.8 → 70.8 on valid) and floor.

## C. Alignment cos(s_c, o_c) (valid; fixed100 agrees)

| condition | blocks | mean cos (quartiles) | pooled recall under U | Spearman(cos, block recall) |
| :--- | ---: | :--- | ---: | ---: |
| own | 3,000 | 0.889 (0.865 / 0.937 / 0.971) | 0.824 | 0.42 |
| other | 1,060 | 0.540 (0.361 / 0.540 / 0.707) | 0.273 | 0.79 |

## B. Condition intervention (seed 3, 1,045 eligible blocks)

| rule | R_V0 (other, as sampled) | R_V1 (same object, dense) | R_V2 (uniform) | R_own | φ [95 % CI] | own class a: V0 → V2 |
| :--- | ---: | ---: | ---: | ---: | :--- | :--- |
| U | 0.245 | 0.833 | 0.540 | 0.833 | **1.001 [0.962, 1.034]** | 0.837 → 0.463 |
| U + both | 0.166 | 0.791 | 0.385 | 0.788 | 1.005 [0.967, 1.040] | 0.787 → 0.324 |

Per class under U: φ 0.86 (door, floor) to 1.16 (window); sofa 0.023 → 0.870, table 0.109 → 0.786. Mean
cos(s_c, o_c) of the re-sampled class: V0 0.547, V1 0.861, V2 0.696.

## Rules

* **P8.1 holds** (g_other +5.25), **P8.3 holds** (g_own +13.71).
* **P8.2 density causal**: φ 1.001 with R_V1 − R_V0 CI [+0.552, +0.621].
* **P8.4**: both training arms admissible; instance-alignment training runs first (bound +13.71 > +5.25),
  condition-balanced training second.

## What it establishes

1. **The other-condition deficit is entirely density.** The same object, re-sampled dense in its own scan, is found
   exactly as often as a class in its own block (0.833 = 0.833), and its direction moves from cos 0.55 to 0.86 of the
   support's. Nothing about the object is missing; its features change with the sampling density.
2. **The own-condition recall also rests on density.** A class in its own block drops from 0.837 to 0.463 recall
   when the block is sampled uniformly (0.324 with U + both). The model finds foreground largely by density, the
   "foreground leakage" COSeg describes; the standard protocol rewards it.
3. Consequence for the architecture (D-42): density invariance (M1) is not an option but the condition for every
   later block. It will raise the other-condition points and remove the density prop under the own ones, so its net
   effect on the standard protocol is not predictable from these numbers and must be reported on both the standard
   and the leak-free draws; the +13.7 own bound is measured on dense points and overstates the instance-shift part
   that survives without density.
