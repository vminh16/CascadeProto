# Phase 16 P6: the clean base's prototype gap, and training-free query adaptation [DECISION D-38]

VM run of 2026-09-25 (NVIDIA L4), commit `5fdda57`, `AUTOSTOP=1 bash experiments/run_p6.sh full`, 00:54–01:25 UTC.
CR `last.pt` (D-37's base), S1, inference only. Checks passed (model identity every episode; fixed100 model 54.84 =
VIP-Seg's metric = D-37's CR). Three `_counts.npz` files (select, random600 seed 2 and its smoke) failed to copy
and stay on the VM; every JSON and the log are here.

## Test (mIoU %, identical episodes)

| draw | model | support rule | U | oracle bg | oracle fg | oracle all | ssp_bg_T2_a0.25_r1 | km3 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed100 | 54.84 | 55.02 | 55.74 | 61.89 | 72.05 | 80.86 | **56.93** | **57.06** |
| random600 seed 0 | 51.83 | 52.56 | 53.18 | 59.41 | 71.13 | 80.52 | 54.52 | 54.23 |
| random600 seed 1 | 56.73 | 56.92 | 57.55 | 63.99 | 73.09 | 82.05 | 59.13 | 58.51 |
| random600 seed 2 | 54.76 | 56.66 | 57.10 | 62.21 | 73.60 | 81.57 | 58.22 | 58.14 |
| leak-free (seed 4) | 28.10 | 32.15 | 33.02 | 40.03 | 45.35 | 56.12 | 33.01 | 33.05 |

U = support directions in the oracle's geometry (unit rows, sums of unit features); it already beats the trained
model by +0.9. The presence-fair oracle here (80.86) is lower than R2's `oracle_unit` (83.38) because absent classes
keep competing.

## Selection (S1 valid, gain over U)

Self-support: every arm that beats U acts on the **background row only**; the best foreground arm is −0.35, the best
"all" arm −0.10. Frozen `ssp_bg_T2_a0.25_r1` (+1.56): ρ = 1 (all predicted background points, no entropy
selection), α = 0.25 (mostly the query's own background), two steps. Next: same with α 0.5 (+1.53), one step
(+1.25). Every ρ < 1 arm is ≤ +0.05. Background k-means: km3 +1.27, km5 +1.23.

## Rules

* **P6.1 location: joint** on both protocols. fixed100: oracle bg +6.14, fg +16.31, all +25.12 (fg share 0.65, just
  under ⅔); leak-free: +7.02 / +12.33 / +23.10. The two rows interact: separately +22.45, together +25.12.
* **P6.2 self-support go**: +1.19 [+0.85, +1.52], random600 +1.34 / +1.58 / +1.13.
* **P6.3 multi-background go**: +1.32 [+0.95, +1.69], random600 +1.05 / +0.96 / +1.04.
* **P6.5**: each closes about 5 % of the oracle gap (4.7 %, 5.3 %); neither moves the leak-free draw (33.01, 33.05
  against U 33.02). Both raise precision (table 0.55 → 0.58 / 0.64) and background recall, and cost a little
  foreground recall.

## What it establishes

1. The gap is mostly in the **foreground** rows (+16 of +25) but no label-free rule tested here reaches it: query
   self-support on the foreground rows lowers the score. Pseudo-labels drawn from the same support prototype
   reinforce its misses.
2. What a training-free rule can recover lies in the **background**: adapting the background row to the query
   (self-support) or splitting it (k-means) each gives about +1.2–1.3 on every draw of the standard protocol.
3. **Entropy selection did not help** (every ρ < 1 arm ≈ 0); the gain came from the cascade (two steps) and the query
   weight, not from choosing confident points.
4. The leak-free level is untouched by either mechanism: the density dependence lies in the features.
