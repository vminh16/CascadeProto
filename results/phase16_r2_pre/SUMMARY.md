# Phase 16 R2, before training: what the first form of D-29 would have measured [DECISION D-29]

VM run of 2026-09-23 (NVIDIA L4), `experiments/r2_distill_eval.py test --draws fixed100` on VIP-Seg's
released S1 and S0 checkpoints only, with the diagnostics of commit `ddd986e` plus the pairwise
prototype cosine and the equal-norm oracle rule (the working copy that became the revision of D-29).
No training. Every episode passed the logit identity check; fixed100 matched VIP-Seg's metric.

## Measured

| | S1 | S0 |
| :--- | ---: | ---: |
| model (fixed100) | 75.36 | 71.97 |
| oracle rule, each class's norm kept (P0's `ORACLE_REPLACE`) | +8.41 | +15.13 |
| oracle rule, one common norm for the present classes | **+10.95** | **+14.12** |
| `cos(M_eff, O)`, background / foreground | 0.428 / 0.303 | 0.099 / 0.132 |
| `cos(M_c − M_c', O_c − O_c')` | 0.530 | 0.443 |
| the same two for the normalised support prototypes (step 0) | 0.834 / 0.633 | 0.826 / 0.661 |

Per step (step 0 = normalised support prototypes, then PEM, PDM, PEM, PDM), raw / pairwise:

| step | 0 | 1 | 2 | 3 | 4 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| S1 | .834 / .633 | .250 / .586 | .350 / .576 | .294 / .363 | .244 / .268 |
| S0 | .826 / .661 | .089 / .510 | .027 / .182 | .240 / .255 | .272 / .315 |

## What it changed

1. **The headroom is in the directions.** An oracle that uses only the directions `O` (one common norm)
   gains as much as P0's rule with each class's norm kept: +10.95 / +14.12 against +8.41 / +15.13
   (the latter reproduce P0 to 0.01). The teacher of D-29 needs no norm and no temperature.
2. **Prototype-space cosines do not measure what decides the prediction.** A common shift of all
   prototypes and any component orthogonal to the features leave every logit difference unchanged,
   and the trained head moves its prototypes far from `O` by both measures (raw 0.83 to 0.10–0.43).
   The first form of D-29, `1 − cos(M_eff, O)`, would have trained and judged those invisible parts.
   D-29 now compares the pairwise logit differences with the oracle rule's (spec 02 §14).
