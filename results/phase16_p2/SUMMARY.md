# Phase 16 P2: D-26's EM with D-27's base margin filtering the foreground M-step [DECISION D-28]

VM run of 2026-09-23 (13:38–14:02 UTC, NVIDIA L4), commit `4dc1ebd`, `experiments/run_p2.sh`. No
training; P1's banks reused; paired bootstrap over episodes as in P0 and P1. All four checkpoints
scored `protocol: clean`.

**Verdict: P2.0 fails, P2.2 stop.** The filter never helps: selection froze the unfiltered arm, which
is P0's frozen setting, and the test reproduces P0's numbers exactly.

## Selection (S1 valid, 1,500 episodes, VIP-Seg S1 decides)

Best arm per filter fraction r, gain over the model (75.24) in mIoU points:

| r | 0 | 0.1 | 0.2 | 0.3 |
| :--- | ---: | ---: | ---: | ---: |
| best arm (all `ssp`, κ 0.5, T 1) | **+0.51** | +0.50 | +0.49 | +0.47 |

The gain falls monotonically with r. Frozen: `ssp_k0.5_T1_r0`.

## Mechanism: the false share ε of the first foreground M-step

| checkpoint | weight | r = 0 | 0.1 | 0.2 | 0.3 | ratio at 0.3 |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| VIP-Seg S1 | ssp | 0.095 | 0.089 | 0.085 | 0.080 | 0.85 |
| VIP-Seg S1 | entropy | 0.116 | 0.113 | 0.113 | 0.114 | 0.98 |
| ours S1 | ssp | 0.268 | 0.265 | 0.263 | 0.260 | 0.97 |
| ours S1 | entropy | 0.297 | 0.296 | 0.296 | 0.298 | 1.00 |

P2.0 needed ≤ 0.75 on both VIP-Seg checkpoints. Removing 30 % of the foreground-assigned points by base
margin removes barely more false mass than a random 30 % would (a random filter keeps ε unchanged).
With the entropy weight the filter does nothing at all, because the points it removes carry little
weight already.

## Test (fixed100, frozen arm = P0's frozen setting)

| checkpoint | model | frozen | gain [95 % CI] | ε at r = 0 |
| :--- | ---: | ---: | :--- | ---: |
| VIP-Seg S1 | 75.36 | 75.73 | +0.37 [+0.24, +0.50] | 0.093 |
| ours S1 | 51.91 | 51.97 | +0.06 [−0.04, +0.16] | 0.268 |
| VIP-Seg S0 | 71.97 | 70.76 | −1.21 [−1.57, −0.87] | 0.130 |
| ours S0 | 49.07 | 48.91 | −0.16 [−0.28, −0.03] | 0.337 |

Identical to P0's test to the last digit: the probe reproduces itself across two separate runs.

## What it establishes

1. **The weak base signal does not clean the M-step either.** The asymmetry argument of D-28 (a filter
   of a mean loses less than a classifier) holds only if the filter removes false mass faster than true
   mass. At an AUC of 0.65–0.72 (P1) it removes them at nearly the same rate.
2. **Three training-free routes are closed** on the same checkpoints: the model's posterior (D-26), a
   base-class margin (D-27) and both together (D-28). What remains measured and unexploited is P0's
   oracle headroom (+8.4 / +15.1 on VIP-Seg), which is the target of D-29, a loss applied during
   training.
