# Phase 16 N2: the support → query attention neck trained from scratch on E1's schedule [DECISION D-34]

VM run of 2026-09-24 (NVIDIA L4), commit `daf95d7`, `AUTOSTOP=1 experiments/run_n2.sh`: training 11:13–12:58 UTC,
evaluation 12:58–13:37. The model of D-33 (`--neck sq_attn`) trained from scratch, batch 1, 24,000 updates,
LR 1e-3 halved every 7,200, 13 validations, seed 0, α initialised to 0.1; scored against E1's existing
checkpoints on identical episodes. S1. `protocol: clean`.

**Verdict: N1.2 stop.** N2 − E1 on fixed100 `last` −0.66 [−1.03, −0.29], and −0.52 / −1.08 / −0.94 on the three
random600 draws; `best` −0.81 [−1.21, −0.44]. This time **the neck was used**: α ended at 0.060 (0.041 at the
selected best, epoch 28), from 0.1.

## Test (mIoU %, identical episodes)

| draw | E1 last | N2 last | E1 best | N2 best | VIP-Seg released |
| :--- | ---: | ---: | ---: | ---: | ---: |
| fixed100 | 73.20 | 72.54 | 75.05 | 74.24 | 75.36 |
| random600 seed 0 | 72.51 | 71.99 | 74.34 | 73.75 | 74.49 |
| random600 seed 1 | 74.23 | 73.16 | 75.50 | 74.63 | 75.82 |
| random600 seed 2 | 75.50 | 74.57 | 77.23 | 76.35 | 77.21 |

Per-class IoU on fixed100 (door, floor, sofa, table, wall, window): E1 last 73.9 / 62.1 / 75.4 / 69.8 / 69.5 /
88.6; N2 last 74.0 / 58.4 / 75.9 / 68.8 / 70.2 / 87.9. Oracle gap (N1.4): E1 +12.73, N2 +13.48.

## Validation curve (S1 valid draw)

| epoch | 4 | 8 | 12 | 16 | 20 | 24 | 28 | 32 | 36 | 40 | 44 | 48 | 50 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| N2 | 65.70 | 66.10 | 67.12 | 70.54 | 71.84 | 71.55 | **74.91** | 73.08 | 72.49 | 72.18 | 70.24 | 74.10 | 72.18 |
| E1 | 63.76 | 68.17 | 68.05 | 71.31 | 72.46 | 71.09 | **75.21** | 71.69 | 71.64 | 72.39 | 70.94 | 72.68 | 73.04 |

## What it establishes

1. **Point-level support → query attention before the prototypes does not improve VIP-Seg's head** at this
   budget: the neck opened (α 0.060) and the model is 0.5–1.1 points below E1 on every draw, `last` and `best`.
   The gap to the oracle did not shrink (12.73 → 13.48).
2. The comparison is between two independent runs (training noise about 1 point); the paired CI covers the
   episodes only. The sign is the same on all four draws and both checkpoints, so a hidden +1.0 gain is not
   plausible; a second seed would not change the verdict of N1.2.
3. With D-26…D-33, every mechanism phase 16 has tested to move E1's prototypes toward the query — fixed rules,
   a label-derived loss, text, base classes, background purification and a learned point-level attention —
   leaves the oracle gap where it was.
