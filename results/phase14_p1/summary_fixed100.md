## Phase-14 results (fixed100, mIoU %)

`best` = best validation checkpoint, `last` = last epoch (D-15); Avg = mean of S0 and S1.

### Table 4 - components (2-way 1-shot)

| Row | S0 best | S0 last | S1 best | S1 last | Avg best | Paper S0 / S1 / Avg | Avg diff (best) |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- | ---: |
| baseline | 49.08 | 49.07 | - | - | - | 82.72 / 79.83 / 81.28 | - |
| lma | - | - | - | - | - | 83.98 / 80.99 / 82.49 | - |
| gate | - | - | - | - | - | 85.34 / 82.48 / 83.91 | - |
| cascade | - | - | - | - | - | 87.89 / 84.05 / 85.97 | - |
| full | 57.15 | 56.70 | - | - | - | 88.53 / 84.53 / 86.53 | - |

### Table 5 - cascade depth T (2-way 1-shot)

| Row | S0 best | S0 last | S1 best | S1 last | Avg best | Paper S0 / S1 / Avg | Avg diff (best) |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- | ---: |
| T1 | - | - | - | - | - | 85.21 / 81.34 / 83.28 | - |
| T2 | - | - | - | - | - | 86.43 / 82.67 / 84.55 | - |
| T3 | - | - | - | - | - | 87.78 / 83.91 / 85.85 | - |
| T4 | 57.15 | 56.70 | - | - | - | 88.53 / 84.53 / 86.53 | - |
| T5 | - | - | - | - | - | 88.51 / 84.50 / 86.51 | - |
| T6 | - | - | - | - | - | 88.37 / 84.31 / 86.34 | - |

### Table 2 - CascadeProto (Text), all settings

| Row | S0 best | S0 last | S1 best | S1 last | Avg best | Paper S0 / S1 / Avg | Avg diff (best) |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- | ---: |
| N2K1 | 57.15 | 56.70 | - | - | - | 88.53 / 84.53 / 86.53 | - |
| N2K5 | - | - | - | - | - | 88.57 / 84.78 / 86.68 | - |
| N3K1 | - | - | - | - | - | 83.07 / 79.04 / 81.06 | - |
| N3K5 | - | - | - | - | - | 79.95 / 77.94 / 78.95 | - |

### Seed spread - full model, S0 2-way 1-shot (best)

seed 0: 57.15, seed 1: -, seed 2: -; needs at least two finished seeds. Paper S0: 88.53.

