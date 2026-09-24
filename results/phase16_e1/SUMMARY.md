# Phase 16 E1: route B's base on VIP-Seg's update count [DECISION D-30]

VM run of 2026-09-24 (NVIDIA L4), commit `0837680`, `experiments/run_r2.sh train e1 && eval_e1`:
training 00:55–02:37 UTC, evaluation 02:37–03:23. r0 unchanged except the schedule: batch 1, 24,000
updates, LR halved every 7,200, 13 validations on the 1,500 `valid` episodes of S1's test classes
(D-22 amended rule 1). S1, seed 0, one run. All `protocol: clean`.

**Verdict: E1.1 adopt.** E1 `last` scores 73.20 on fixed100 (rule: ≥ 73.0), +2.99 [+2.62, +3.39]
over r0, and +2.60 to +3.11 on each random600 draw. VIP-Seg's update count is route B's schedule from
now on. E1 `best` (best-of-validation, VIP-Seg's protocol) matches VIP-Seg's released checkpoint:
−0.31 [−0.63, +0.01] on fixed100, −0.32 to +0.02 on the random600 draws.

## Test (mIoU %, identical episodes, paired bootstrap over episodes)

| draw | E1 last | E1 best | r0 last (= r0 best) | d29 best | VIP-Seg released |
| :--- | ---: | ---: | ---: | ---: | ---: |
| fixed100 | **73.20** | **75.05** | 70.20 | 71.68 | 75.36 |
| random600 seed 0 | 72.51 | 74.34 | 69.40 | 70.50 | 74.49 |
| random600 seed 1 | 74.23 | 75.50 | 71.40 | 72.39 | 75.82 |
| random600 seed 2 | 75.50 | 77.23 | 72.91 | 74.14 | 77.21 |

| paired comparison | fixed100 [95 % CI] | random600 seeds 0 / 1 / 2 |
| :--- | :--- | :--- |
| E1 last − r0 last | +2.99 [+2.62, +3.39] | +3.11 / +2.83 / +2.60 |
| E1 last − VIP-Seg | −2.16 [−2.52, −1.83] | −1.97 / −1.59 / −1.71 |
| E1 best − VIP-Seg | −0.31 [−0.63, +0.01] | −0.15 / −0.32 / +0.02 |
| E1 best − E1 last (selection) | +1.85 [+1.43, +2.30] | +1.83 / +1.27 / +1.73 |

r0's best validation was its last epoch, so r0 best = r0 last. d29's best (epoch 40) is 1.48 above
r0 on fixed100 while its `last` equals r0's (R2): selection alone moves a single run by 1.4–1.9 points.

Per-class IoU on fixed100 (door, floor, sofa, table, wall, window): E1 last 73.9 / 62.1 / 75.4 / 69.8 /
69.5 / 88.6; E1 best 74.7 / 65.1 / 81.5 / 73.9 / 67.1 / 88.0; r0 74.7 / 53.6 / 72.4 / 64.5 / 68.6 /
87.3; VIP-Seg 74.8 / 65.3 / 80.4 / 73.6 / 69.7 / 88.4. The update count recovers floor (+8.5), table
(+5.3) and sofa (+3.0).

## Validation curve (S1 valid draw, 1,500 episodes, as in R2's training logs)

| epoch | 4 | 8 | 12 | 16 | 20 | 24 | 28 | 32 | 36 | 40 | 44 | 48 | 50 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E1 | 63.76 | 68.17 | 68.05 | 71.31 | 72.46 | 71.09 | **75.21** | 71.69 | 71.64 | 72.39 | 70.94 | 72.68 | 73.04 |

The same shape as VIP-Seg's released S1 log (valid 68.52 → 75.63 at best → 72.84 last): a swing of
about 4 points between validations, the best one well above the last.

## What it establishes

1. **Our loop reproduces VIP-Seg's head.** With VIP-Seg's update count the `last` checkpoint reaches
   VIP-Seg's own last-update level (73.20 fixed100 against 72.84 valid in its log), and the
   best-of-validation checkpoint matches the released one within 0.3 points on every draw. The 5-point
   gap of R2.0 was 3.0 points of training length and 1.9 points of checkpoint selection.
2. **The route-B base is now VIP-Seg-level**: 73.20 `last` / 75.05 `best` on S1 fixed100. Every
   later arm is trained on this schedule and reported both ways (D-22 amended).
3. **Selection is worth ~1.3–1.9 points on one run**, the size of every effect phase 16 has chased;
   claims against the published table must say which checkpoint they rest on.
