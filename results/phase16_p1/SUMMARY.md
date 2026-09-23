# Phase 16 P1: base-class calibration of the background on trained checkpoints [DECISION D-27]

VM run of 2026-09-23 (12:47–13:21 UTC, NVIDIA L4), commit `e8801dd`, `experiments/run_p1.sh`. No
training; paired bootstrap over episodes as in P0. Banks: 1,000 seeded training episodes of each
checkpoint's own fold, 673–966 occurrences per base class. All four checkpoints scored `protocol: clean`.

**Verdict: P1.2 stop (training-free form); P1.4 weak.** Moving to the background even 0.5 % of the
foreground predictions, those nearest to a base class, costs mIoU on every checkpoint.

Three smoke runs preceded the real run and changed the design before it (D-27, "Revised" and "Second
revision"): the first form collapsed VIP-Seg to background, the second's absolute margin grid flipped
33–73 % of the foreground predictions.

## Selection (S1 valid, 1,500 episodes)

| q (flipped share of foreground predictions) | 0.5 % | 1 % | 2 % | 4 % |
| :--- | ---: | ---: | ---: | ---: |
| VIP-Seg S1 (model 75.24) | 75.07 | 74.86 | 74.40 | 73.41 |
| ours S1 (model 52.24) | 52.17 | 52.07 | 51.84 | 51.34 |

Frozen: q = 0.5 % (mean valid gain −0.12).

## Test (fixed100, 1,500 episodes each)

| checkpoint | model | frozen | gain [95 % CI] | AUC false vs true fg | false share of fg | positive-margin share |
| :--- | ---: | ---: | :--- | ---: | ---: | ---: |
| VIP-Seg S1 | 75.36 | 75.20 | −0.16 [−0.17, −0.15] | 0.716 | 0.100 | 0.692 |
| ours S1 | 51.91 | 51.84 | −0.07 [−0.08, −0.06] | 0.494 | 0.194 | 0.623 |
| VIP-Seg S0 | 71.97 | 71.89 | −0.07 [−0.09, −0.06] | 0.649 | 0.180 | 0.578 |
| ours S0 | 49.07 | 49.04 | −0.03 [−0.04, −0.02] | 0.629 | 0.272 | 0.685 |

## What it establishes

1. **The base margin is a real but weak signal on VIP-Seg's features** (AUC 0.65–0.72) and none on our
   S1 baseline's (0.49). It is far too weak to decide single points: 10–27 % of the foreground predictions
   are false, 58–69 % have a positive margin, and flipping the top 0.5 % already removes more true than
   false foreground.
2. **Raw post-ReLU features share a strong common component** (mean cosine 0.58–0.62 of a query feature
   with the bank's centre), which is why the first form collapsed and why the CL2N centring is needed.
3. Used as a **classifier**, a weak signal loses one IoU point per wrong flip. Used as a **filter** of a
   prototype's mean, a wrong exclusion costs only variance: that asymmetry motivates D-28 (P2).
