# Phase 16 D-43: M1, a density-invariant encoder, against the clean base CR [DECISION D-43]

VM run of 2026-09-26 (NVIDIA L4), commit `8504462`: training 12:29–14:33 UTC (24,000 updates, E1's schedule, seed 0),
evaluation 14:33–16:20 UTC. CR's numbers are re-computed in the same run and equal D-40's (U + both + LP 58.55,
U + both 57.63 on fixed100) and P8's (arm B: R_V0 0.245); checks passed. The rules are those of D-43, read by
`experiments/d43_eval.py decide` (re-run locally on the copied files: identical lines).

## Training

Validation mIoU (S1 valid, every 4 epochs): 43.97, 41.43, 42.60, 42.40, 43.18, 41.99, 43.12, 48.55, 44.75, 47.82,
50.55, **51.63** (epoch 48, `best.pt`), 48.17 (epoch 50, `last.pt`). CR at epoch 48: 55.82.

## Test (mIoU %, S1, 2-way 1-shot; stack = model head / U / U + both / U + both + LP with P7's frozen arm)

| draw | CR | M1 `last` | M1 `best` |
| :--- | :--- | :--- | :--- |
| fixed100 | 54.84 / 55.74 / 57.63 / 58.55 | 47.94 / 49.42 / 48.54 / 49.24 | 51.50 / 51.72 / 50.72 / 51.56 |
| random600 seed 0 | 51.83 / 53.18 / 54.84 / 55.75 | 46.49 / 48.43 / 47.64 / 48.25 | 49.25 / 49.59 / 49.15 / 50.03 |
| random600 seed 1 | 56.73 / 57.55 / 59.54 / 60.57 | 49.70 / 50.87 / 49.59 / 50.28 | 51.93 / 52.76 / 51.79 / 52.62 |
| random600 seed 2 | 54.76 / 57.10 / 58.93 / 60.35 | 46.85 / 49.48 / 47.28 / 47.82 | 50.12 / 51.53 / 49.68 / 50.32 |
| leak-free (seed 4) | 28.10 / 33.02 / 33.14 / 33.31 | 30.17 / 34.57 / 32.93 / 33.01 | 30.98 / 34.93 / 33.68 / 33.85 |

Paired bootstrap over episodes, M1 − CR per stack level (the rules use only the last column; the others are
descriptive and were not pre-registered):

| draw | checkpoint | model | U | U + both | U + both + LP |
| :--- | :--- | :--- | :--- | :--- | :--- |
| fixed100 | `last` | −6.89 [−8.43, −5.32] | −6.33 [−7.53, −5.10] | −9.09 [−10.44, −7.73] | **−9.31 [−10.82, −7.85]** |
| fixed100 | `best` | −3.34 [−4.78, −1.84] | −4.02 [−5.25, −2.81] | −6.91 [−8.22, −5.61] | −6.99 [−8.41, −5.57] |
| leak-free | `last` | +2.07 [+1.21, +2.93] | +1.55 [+0.79, +2.30] | −0.21 [−1.15, +0.77] | **−0.30 [−1.41, +0.85]** |
| leak-free | `best` | +2.88 [+2.10, +3.69] | +1.91 [+1.17, +2.63] | +0.54 [−0.42, +1.45] | +0.53 [−0.56, +1.63] |

random600, U + both + LP, `last`: −7.50, −10.28, −12.52.

## Mechanism (P8's measurements on M1 `last`, under U)

| quantity | CR (P8) | M1 | D43.1 limit |
| :--- | ---: | ---: | ---: |
| arm B R_V0 (other, as sampled) | 0.245 | 0.276 | |
| arm B R_V1 (same object, dense) | 0.833 | 0.770 | |
| arm B R_own | 0.833 | 0.708 | |
| other-condition deficit R_own − R_V0 | 0.588 | **0.432** | ≤ 0.294 |
| φ [95 % CI] | 1.001 [0.962, 1.034] | 1.143 [1.087, 1.200] | |
| own class a: R(V0) → R(V2 uniform) | 0.837 → 0.463 | 0.789 → 0.358 | |
| own-class uniform drop | 0.374 | **0.431** | ≤ 0.187 |
| cos(s_c, o_c) of the re-sampled class V0 / V1 / V2 | 0.547 / 0.861 / 0.696 | 0.721 / 0.825 / 0.750 | |
| condition oracles (valid): U, g_own, g_other, g_fg | 55.96, +13.71, +5.25, +16.12 | 49.53, +18.59, +4.83, +22.25 | |
| alignment own / other: mean cos, pooled recall | 0.889 / 0.540; 0.824 / 0.273 | 0.842 / 0.720; 0.743 / 0.300 | |

1,045 intervention blocks from 890 episodes (1 episode skipped for a support without background, as in P8).

## Rules

* **D43.1 mechanism fails.** The other-condition deficit falls from 0.588 to 0.432 (limit 0.294); the own-class drop
  under uniform sampling rises from 0.374 to 0.431 (limit 0.187). The density dependence of the decision is not
  removed.
* **D43.2 leak-free fails.** U + both + LP: −0.30 [−1.41, +0.85].
* **D43.4 stop.** M1 is not adopted; CR stays the base. fixed100 −9.31 [−10.82, −7.85].
* D43.5: reported above.

## What it establishes

1. **The four encoder changes move the features but not the decision.** The direction gap between the sparse and the
   dense version of the same object shrinks (cos V0 → V1: 0.55 → 0.86 on CR, 0.72 → 0.83 on M1), and the other
   condition's mean alignment rises from 0.54 to 0.72; yet the recall of that object stays at 0.28 as sampled and
   0.77 dense (φ 1.14). Density therefore still enters the decision after these four pathways are closed, through a
   path the direction cosine does not show.
2. **Residual pathways (candidates, not measured).** (a) A metric ball with fewer than 16 distinct points is padded
   with duplicates, and a max-pool over n distinct samples of a surface grows with n, so the neighbourhood feature
   still counts points below K; r₁ = 0.1 m was chosen to hold about 16 points at background density, so sparse
   objects are exactly where this happens. (b) The per-block statistics are taken over points, so a dense region
   weighs more in the block's mean and std. (c) FPS centres, the Mamba sequence order and the feature head were not
   changed. D-43's statement that a max over a fixed metric region "does not count points" holds only when the
   region is sampled to at least K points; that assumption was wrong for the sparse objects the change targets.
3. **Without the density cue, the standard score falls and the honest score rises at the head.** The model head and
   U are better than CR's on the leak-free draw (+2.07 and +1.55, CIs above 0, `best` +2.88 and +1.91), and worse on
   fixed100 (−6.9 and −6.3). This is the trade D-43.3 anticipated (the part of the standard score that rests on the
   density leak); it was not a pre-registered rule and does not change the verdict.
4. **The training-free rules were tuned on CR's features and do not transfer.** "both" (support-side background plus
   k-means) adds +1.9 on CR's fixed100 and −0.9 on M1's; on the leak-free draw it removes M1's +1.55 at U. Every rule
   of the stack must be re-selected on the base it is applied to, which is D-42's composition failure L3 observed.
5. **M1 trains slower.** Validation reaches CR's level of epoch 12 only at epoch 44, and `best` (epoch 48) beats
   `last` by 3.6 on fixed100; the schedule copied from CR may be short for a new encoder (not measured).
