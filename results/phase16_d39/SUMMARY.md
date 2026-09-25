# Phase 16 D-39: trained self-support prototypes on the clean base, and the combined background rule

VM run of 2026-09-25 (NVIDIA L4), commit `5fbd441`, `AUTOSTOP=1 bash experiments/run_d39.sh full`: A0 training
07:02–08:35, A1 08:35–10:10, evaluation 10:10–11:34 UTC. S1, seed 0, one run per arm, E1's schedule, random query
order. Checks passed (identity every episode, A0 rows = U, VIP-Seg's metric, CR = 54.84). [DECISION D-39]

## Test (mIoU %, `last.pt`; identical episodes)

| checkpoint | rule | fixed100 | random600 s0 / s1 / s2 | leak-free |
| :--- | :--- | ---: | :--- | ---: |
| CR (clean VIP head) | model | 54.84 | 51.83 / 56.73 / 54.76 | 28.10 |
| CR | U (support directions) | 55.74 | 53.18 / 57.55 / 57.10 | 33.02 |
| CR | U + ssp_bg | 56.93 | 54.52 / 59.13 / 58.22 | 33.01 |
| CR | U + km3 | 57.11 | 54.34 / 58.60 / 58.24 | 33.12 |
| **CR** | **U + both** | **57.63** | **54.84 / 59.54 / 58.93** | 33.14 |
| A0 (no head, trained) | model | 53.52 | 50.96 / 53.97 / 55.05 | 30.34 |
| A0 | both | 53.71 | 51.31 / 54.47 / 55.07 | 31.17 |
| A1 (A0 + trained self-support) | model | 52.86 | 50.54 / 53.67 / 54.00 | 30.34 |
| A1 | both | 53.50 | 50.96 / 54.35 / 54.69 | 31.81 |

`best.pt`: A0 55.83, A1 53.72 on fixed100 (model). Oracle (presence-fair, on each model's features): CR 80.86,
A0 78.38, A1 79.55. A1's learned mixing weights: α_bg 0.430, α_fg 0.699 (initialised 0.25 / 0.5).

## Rules

* **D39.1 background → both**: CR both − CR km3 +0.51 [+0.24, +0.78], random600 +0.50 / +0.94 / +0.68.
* **D39.2 self-support stop**: A1 − A0 −0.66 [−1.30, +0.02], random600 −0.42 / −0.30 / −1.05.
* **D39.3 CR stays (+ both)**: candidate A0 + both − CR model −1.12 [−2.12, −0.13].

## What it establishes

1. **The best honest rule so far is CR's features with U + both** (support directions, background adapted to the
   query and split into three components): 57.63 on S1 fixed100, +2.79 over CR's own head, and +3.0 / +2.8 / +4.2 on
   the random600 draws; inference only, no training.
2. **Trained self-support does not help** (A1 − A0 −0.66): even learned, the foreground rows keep α_fg ≈ 0.70 and the
   score falls. Query self-support is not the mechanism for the foreground gap.
3. **Training without the head gives worse features for prototype matching**: A0 53.52 against 55.74 for the same
   rule on CR's features. VIP-Seg's clean head is useless at inference (U beats it) but helps as a training
   signal; why is not measured.
4. The foreground gap is still ≈ +23 (oracle 80.86 against 57.63). The leak-free level moves only through U
   (28.10 → 33.02); the background rules do not change it.
