# Phase 16 N1: a zero-initialised support → query attention neck, warm-started from E1 [DECISION D-33]

VM run of 2026-09-24 (NVIDIA L4), commit `d7ec930`, `experiments/run_n1.sh`: ctl trained 08:32–09:02 UTC,
neck 09:02–09:32, evaluation 09:32–10:19. Both arms warm-started from E1 `last.pt`, seed 0 (identical
episodes), batch 1, constant LR 1.25e-4, 7,200 updates, 3 validations. S1. `protocol: clean`.

**Verdict: N1.2 stop.** neck − ctl on fixed100 −0.17 [−0.22, −0.13], and −0.07 / −0.20 / −0.16 on the random600
draws. **But the neck was never used**: its gate ended at α = 0.0023, so the test measured grafting a closed
neck onto a converged model, not point-level support → query attention (D-34 prepares that test).

## Test (mIoU %, identical episodes)

| draw | E1 (start) | ctl | neck | neck − ctl |
| :--- | ---: | ---: | ---: | :--- |
| fixed100 | 73.20 | 71.15 | 70.97 | −0.17 [−0.22, −0.13] |
| random600 seed 0 | 72.51 | 70.04 | 69.97 | −0.07 |
| random600 seed 1 | 74.23 | 72.41 | 72.21 | −0.20 |
| random600 seed 2 | 75.50 | 73.00 | 72.84 | −0.16 |

ctl − E1 −2.05 [−2.34, −1.78] and neck − E1 −2.22 [−2.51, −1.95]: the continued training made both arms worse
than the checkpoint they started from. Best-of-validation equals last for both arms (their best validation was
the last one). Oracle gaps: ctl +14.31, neck +14.44 (N1.4: unchanged).

## Training (S1 valid draw)

| epoch | 5 | 10 | 15 |
| :--- | ---: | ---: | ---: |
| ctl | 58.91 | 70.17 | 70.95 |
| neck | 59.23 | 70.45 | 70.80 |

E1 ended at 73.04 on this draw.

## What it establishes

1. **A zero-initialised gate on a converged model does not open at a small learning rate.** The neck's
   projections receive gradient only through α, and α receives a gradient whose sign is not consistent while
   the neck's output direction is random: α moved to 0.0023 in 7,200 updates [measured α; mechanism inferred].
2. **Restarting AdamW on a converged model costs about 14 points at first and 2 points at the end** (both arms:
   73.04 → ~59 at epoch 5 → ~71), although E1 itself trained at this learning rate in its last stage without a
   dip. The warm start carried the weights, not the optimiser's moments; `resume.pt` holds them and a warm
   start should load them.
3. N1 says nothing about point-level support → query attention itself; D-34 (N2) trains the neck from scratch
   with α = 0.1.
