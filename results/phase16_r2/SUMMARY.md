# Phase 16 R2: oracle-direction distillation on VIP-Seg's head [DECISION D-29]

VM runs of 2026-09-23 (NVIDIA L4), commit `4a2ec60`, `experiments/run_r2.sh`: r0 trained 14:50–16:12
UTC, d29 16:12–17:39, evaluation 17:40–18:01. S3DIS **S1**, 2-way 1-shot, full schedule of D-12
(50 × 480 episodes, batch 4), seed 0, **one training run per arm** (maintainer). `last.pt` of each arm,
scored with VIP-Seg's released S1 checkpoint on identical episodes. All `protocol: clean`.

* **r0**: VIP-Seg's four alternating modules in our loop (`stage_type=vip`, T = 4, L2 prototypes, no
  LMA, ADRM).
* **d29**: r0 + β = 1 × `L_distill` (spec 02 §14, the revised pairwise logit form).

**Verdict: R2.2 stop.** d29 equals r0 on every draw. R2.0 flags our loop: r0 is 5.16 points below
VIP-Seg's released checkpoint.

## Test (mIoU %, paired bootstrap over episodes, 2,000 resamples)

| draw | r0 | d29 | VIP-Seg released | d29 − r0 [95 % CI] | r0 − VIP-Seg [95 % CI] |
| :--- | ---: | ---: | ---: | :--- | :--- |
| fixed100 | 70.20 | 70.26 | 75.36 | +0.06 [−0.26, +0.39] | −5.16 [−5.64, −4.72] |
| random600 seed 0 | 69.40 | 69.37 | 74.49 | −0.04 [−0.47, +0.41] | −5.09 [−5.82, −4.31] |
| random600 seed 1 | 71.40 | 71.36 | 75.82 | −0.04 [−0.51, +0.43] | −4.42 [−5.16, −3.68] |
| random600 seed 2 | 72.91 | 72.70 | 77.21 | −0.21 [−0.61, +0.21] | −4.30 [−4.96, −3.68] |

## Mechanism and headroom (fixed100)

| | r0 | d29 | VIP-Seg |
| :--- | ---: | ---: | ---: |
| logit-pair cosine to the oracle rule (what D-29 trains) | 0.771 | **0.843** | 0.807 |
| oracle rule, norms kept | 84.63 (+14.43) | 84.69 (+14.43) | 83.77 (+8.41) |
| oracle rule, one common norm | 85.01 (+14.81) | 85.50 (+15.24) | 86.31 (+10.95) |
| `cos(M_eff, O)` background / foreground (descriptive) | 0.071 / 0.088 | 0.132 / 0.112 | 0.428 / 0.303 |
| pairwise prototype cosine, steps 0–4 (descriptive) | .59 .17 .27 .27 .30 | .61 .30 .32 .25 .30 | .63 .59 .58 .36 .27 |

Per-class IoU (door, floor, sofa, table, wall, window): r0 74.7 / 53.6 / 72.4 / 64.5 / 68.6 / 87.3;
d29 74.0 / 55.6 / 71.2 / 63.9 / 69.8 / 87.1; VIP-Seg 74.8 / 65.3 / 80.4 / 73.6 / 69.7 / 88.4.

## Training (S1 valid draw, 300 episodes)

| epoch | 10 | 20 | 30 | 40 | 50 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| r0 | 68.27 | 69.49 | 68.85 | 69.91 | 70.26 |
| d29 | 65.62 | 70.42 | 68.97 | 71.60 | 70.19 |

VIP-Seg's released checkpoint scores 75.24 on this draw. `L_distill` on the training (base) classes,
logged for both arms: r0 0.160 → 0.067 without any gradient from it (CE alone brings the pairwise
logit cosine to about 0.93 on base classes); d29 lower throughout.

## What it establishes

1. **The objective transferred, the metric did not.** d29 raised the logit-pair cosine on the novel
   test classes from 0.771 to 0.843, above VIP-Seg's 0.807, and gained nothing; VIP-Seg, with a lower
   cosine, is 5 points better. Agreement with the oracle rule in this sense does not predict mIoU.
2. **Our loop trains VIP-Seg's head 4.3–5.2 points below VIP-Seg's own code, on every draw, with
   features that are as informative**: the oracle rules on r0's features score 84.6 / 85.0 against
   VIP-Seg's 83.8 / 86.3. The loss is in how the head uses the features (floor −11.7, table −9.1,
   sofa −8.0 IoU), not in the encoder. This gap is larger than any effect phase 16 has measured.
3. Training-seed noise is not measured (one run per arm); the four test draws agree to within 0.25
   points on d29 − r0, so the null result is not a draw artefact.

The analysis of why, and of what to try next: `docs/research/2026-09-24_r2_distill_analysis.md`.
