# Phase 16 D-45: M2, VICReg's variance and covariance terms on the query features [DECISION D-45]

Run of 2026-09-27 on a rented RTX 3090 (vast.ai), local data (D-44 amendment 3). Started 14:00 UTC at commit
`d235c79`. The host rebooted twice (about 14:34 and 15:32 UTC, not requested). Each time the script was rerun with
`train.py --resume` from the last finished epoch (commits `6521fdd` and `05ff77a` only let the rerun skip finished
steps and keep the monitor's decisions). Trainings ended 16:24 UTC (both exit 0); tests and part B ended 17:53 UTC.
One seed per arm, both arms trained together on the one GPU.

**Checks.** CR's references hold on this GPU: fixed100 model 54.84, U 55.74, U + both 57.63, U + both + LP 58.55, the
D-43 values. Neither arm was stopped by the monitor.

**Verdict: D45.3.** Both arms un-collapse the features (D45.1 holds: participation ratio 42.45 and 95.25, limit 12).
Both lose to CR on every draw, by 5 points (A) and 11 points (B) on U (D45.2 fails). By the rule, the space widened
and U does not use it. Part B shows the widening added no class information: every oracle is at or below CR's.

## Test (mIoU %, identical episodes; `last.pt` unless marked)

| draw | CR model | CR U | CR U+both | CR U+both+LP | A model | A U | A best model | B model | B U | B best model |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed100 | 54.84 | **55.74** | 57.63 | 58.55 | 48.52 | 50.54 | 52.66 | 48.64 | 44.40 | 52.40 |
| random600 seed 0 | 54.59 | 55.55 | 57.20 | 58.37 | 48.67 | 50.66 | 51.88 | 47.67 | 44.49 | 51.26 |
| random600 seed 1 | 57.07 | 57.85 | 59.35 | 60.36 | 49.96 | 51.52 | 53.79 | 49.77 | 45.76 | 51.59 |
| random600 seed 2 | 56.11 | 57.18 | 59.10 | 59.99 | 49.41 | 50.03 | 52.83 | 48.90 | 43.42 | 51.84 |
| leak-free | 28.34 | 33.11 | 33.05 | 33.15 | 24.59 | 30.47 | 22.74 | 23.58 | 24.11 | 22.02 |

The stack selected on CR hurts both arms: A's U + both + LP is 48.28 on fixed100 and B's is 27.84.

The random600 draws are drawn from this machine's `class2scans`, which differs from the old VM's (D-44 amendment
3). CR's random600 values therefore differ from D-43's (seed 0: 54.59 here, 51.83 there). Every comparison below
re-scores CR on the same episodes; fixed100 uses stored episodes and matches D-43 exactly.

| paired comparison | fixed100 [95 % CI] | random600 seeds 0 / 1 / 2 |
| :--- | :--- | :--- |
| A U − CR U (D45.2) | −5.20 [−6.02, −4.34] | −4.89 / −6.32 / −7.14 |
| B U − CR U (D45.2) | −11.34 [−12.34, −10.36] | −11.07 / −12.09 / −13.76 |
| A best U − CR U | −5.40 [−6.22, −4.54] | −5.34 / −6.43 / −7.37 |
| A model − CR model | −6.31 [−7.36, −5.17] | −5.92 / −7.11 / −6.70 |
| A best model − CR model | −2.17 [−3.25, −1.07] | −2.71 / −3.28 / −3.28 |
| B best model − CR model | −2.43 [−3.44, −1.39] | −3.33 / −5.48 / −4.27 |

## Part B on each arm (valid, P9's reader; CR from P9)

| quantity | CR | A (μ 1, ν 0.04) | B (μ 4, ν 0.16) |
| :--- | ---: | ---: | ---: |
| participation ratio (of 128) | 5.56 | **42.45** | **95.25** |
| share of d* in the base-class span (median) | 0.857 | 0.498 | 0.506 |
| retrieval purity, hit / missed points | 0.886 / 0.109 | 0.862 / 0.228 | 0.755 / 0.260 |
| nuisance κ / ρ at r 8 | 0.71 / 0.86 | 0.37 / 0.49 | 0.38 / 0.51 |
| heads CV (PCA, H 4) | 0.13 | 0.23 | 0.21 |
| model | 55.17 | 48.40 | 48.38 |
| U | 55.96 | 51.12 | 45.43 |
| cosine oracle | 80.84 | 79.74 | 79.27 |
| LDA oracle (0.1) | 96.04 | 94.43 | 92.73 |
| label-free LDA (0.5) | 35.44 | 35.25 | 34.25 |
| projection (4) | 43.92 | 37.83 | 30.26 |
| components, k-means 2 | 53.86 | 50.62 | 45.07 |
| best composite | 36.55 | 36.58 | 34.38 |

## Monitor (first 300 valid episodes; participation ratio / U)

| epoch (A / B) | 4 / 5 | 12 / 13 | 20 / 21 | 24 / 25 | 32 / 33 | 40 / 41 | 48 / 49 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| A | 13.9 / 60.5 | 23.0 / 60.6 | 23.8 / 62.6 | 31.6 / 57.9 | 36.1 / 61.5 | 39.0 / 61.3 | 42.3 / 59.9 |
| B | 45.6 / 56.0 | 58.0 / 54.6 | 82.1 / 58.2 | 87.6 / 54.7 | 98.7 / 58.9 | 93.1 / 62.2 | 91.5 / 60.2 |

On these 300 episodes CR's final checkpoint reads 5.34 / 67.18. The ratio rises from the first epochs and keeps
rising; U stays 5 to 10 points below CR's at every reading.

Validation of `train.py` (1,500 episodes, the model's output), every 4 epochs:
- **A:** 48.5, 46.0, 49.0, 48.4, 49.7, 44.9, 43.2, **53.0**, 50.2, 49.3, 47.2, 52.7, 48.4 (end).
- **B:** 44.4, 47.1, 44.1, 49.3, 49.2, 45.2, 48.6, 49.3, 51.1, **51.7**, 49.8, 48.1, 48.4 (end).
- **CR, same epochs:** 48.6, 49.2, 49.9, 52.3, 51.1, 52.5, 53.3, 55.2, 55.2, 54.3, 51.9, 55.8, 55.2 (end).

## Collapse census (first 300 valid episodes, inference)

| checkpoint | participation ratio | U | model |
| :--- | ---: | ---: | ---: |
| CR (vip_clean, random order) | 5.34 | 67.18 | 68.79 |
| VR (vip, random order) | 5.60 | 66.39 | 65.38 |
| E1 (vip, fixed order) | 5.94 | 58.56 | 68.62 |
| r0 (vip, batch 4) | 7.31 | 61.81 | 66.77 |
| D-29 distilled | 6.86 | 62.94 | 67.26 |
| N1 | 6.03 | 61.62 | 67.40 |
| A0 (no stage, unit) | 3.70 | 65.53 | 65.53 |
| A1 (A0 + self-support + aux) | 3.67 | 66.55 | 66.48 |
| M1 (density) | 6.66 | 59.39 | 60.61 |
| VIP-Seg released S1 | 6.39 | 63.45 | 70.50 |

N1 with attention and N2 are not read (`p5.support_features` does not support the neck, D-35).

These 300 episodes are easier than the full valid split: CR's U reads 67.18 here against 55.96 on the full split.

## What it establishes

1. **The collapse belongs to this training, not to one run or to our code.** Every checkpoint of every head reads
   3.7 to 7.3, including VIP-Seg's released model. With seven training classes, CE on episodes settles at about
   C − 1 directions whatever the head.
2. **VICReg on the 128-d features undoes the collapse.** A reaches 42.45 and B 95.25, and the monitor shows the
   ratio rising from the first epochs. The regulariser works as a regulariser: the mechanism of D45.1 holds.
3. **The directions it adds carry no class information, and they cost accuracy.**
   - Every oracle on the arms is at or below CR's: cosine 79.7 / 79.3 against 80.8, LDA 94.4 / 92.7 against 96.0.
   - U falls 5.2 points for A and 11.3 points for B on fixed100, and B, which widens more, loses more.
   - The share of d* in the base span halves (0.86 to 0.50) and the missed points' retrieval purity doubles (0.11
     to 0.23–0.26), but these re-arrange the same information rather than add to it.
   - CR is collapsed and still reaches the cosine oracle 80.84, so the collapse is not what bounds the head.
4. **The CR-selected stack does not transfer.** It lowers both arms, B's LP by 17 points, as D-43 found for M1.
5. **Selection matters more on the arms than on CR.** `best.pt` recovers about 4 points of the model's score
   (A 48.52 to 52.66) but not U's. The arms' valid curves swing 5 to 10 points between validations, against CR's
   2 to 4.

## Limits

- One seed per arm.
- The weights (1 / 0.04 and 4 / 0.16) are conventions derived from the paper's 25 : 25 : 1 ratio; a much smaller
  weight was not tried.
- The terms act on the 128-d features themselves, not on an expander as in VICReg (§4 of the paper). In VICReg the
  representation used downstream sits before the expander, so the regulariser shapes it only indirectly; here it
  acts on the prototypes' space directly. This deviation may be what costs accuracy; it was not measured.
- The host reboots cost about 10 minutes of training in total and did not change any setting: a resumed run
  continues the same episodes, optimiser state and random states.
