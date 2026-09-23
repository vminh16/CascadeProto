# Phase 16 P0: query-side EM refinement on trained checkpoints [DECISION D-26]

VM run of 2026-09-23 (11:16–11:36 UTC, NVIDIA L4), commit `50898a9`, `experiments/run_p0.sh`. No training:
every arm is compared with the model on the same episodes and weights, so the uncertainty below is the
paired bootstrap over episodes (2,000 resamples), not training-seed noise. All four checkpoints were
scored on their own fold's test classes (`protocol: clean`). VIP-Seg's S1 checkpoint was fetched from the
pinned commit and matches its git blob `2eda6e2a0411d5c7a25746e9e818548ec8156d32`.

**Verdict: P0.2 stop.** Re-estimating the prototypes from the query's own confident points does not help
on these features, although the same update with the query labels gains +8 to +22 points.

## Selection (S1 valid draw, 1,500 episodes, S1 checkpoints only)

Gain over the model in mIoU points, best arm of each weight and the trend with κ and T:

| arm | VIP-Seg S1 (model 75.24) | ours S1 (model 52.24) |
| :--- | ---: | ---: |
| entropy, best (κ 0.5, T 1) | +0.35 | −0.26 |
| none, best (κ 0.5, T 1) | +0.45 | −0.65 |
| ssp, best (κ 0.5, T 1) | **+0.51** | +0.04 |
| κ 16, T 3 (entropy / none / ssp) | −6.20 / −6.66 / −3.56 | −7.17 / −10.03 / −1.45 |
| oracle, best κ | +9.75 (κ 8) | +19.62 (replace) |

Every pseudo-label arm gets worse as κ or T grows; the smallest step of the grid wins. Frozen:
`ssp_k0.5_T1`, mean valid gain +0.28 (`selection.json`).

## Test (fixed100, 1,500 episodes each; S0 looked at for the first time)

| checkpoint | model | frozen | gain [95 % CI] | ssp − none | oracle, same κ | oracle, replace |
| :--- | ---: | ---: | :--- | ---: | ---: | ---: |
| VIP-Seg S1 | 75.36 | 75.73 | +0.37 [+0.24, +0.50] | +0.03 | +3.05 | +8.42 |
| ours S1 | 51.91 | 51.97 | +0.06 [−0.04, +0.16] | +0.45 | +6.34 | +20.46 |
| VIP-Seg S0 | 71.97 | 70.76 | **−1.21 [−1.57, −0.87]** | −0.19 | +9.73 | +15.13 |
| ours S0 | 49.07 | 48.91 | −0.16 [−0.28, −0.03] | +0.86 | +7.01 | +21.76 |

The model rows reproduce the known levels: VIP-Seg S0 71.97 (its log 72.20; the 05 §4 sanity check
measured 71.97), our S0 baseline 49.07 (report 49.08), our S1 baseline `last.pt` 51.91 (`best.pt` scored
51.91 in `results/seen/baseline_S1_unseen_best.json`).

## The rules, as fixed before the run (`p0_em_probe.py decide`)

| rule | measured | verdict |
| :--- | :--- | :--- |
| P0.1 go (S1 ≥ +1.5 both, S0 ≥ +1.0 both) | S1 +0.37 / +0.06 | not met |
| **P0.2 stop (S1 < +0.5 both)** | S1 +0.37 / +0.06 | **met: drop the direction** |
| P0.4 held-out gain | VIP-Seg S0 −1.21, ours S0 −0.16, both CIs below 0 | the S1 gain does not transfer |
| P0.5 entropy weight | selection froze `ssp`, not `entropy` | not claimable |
| P0.6 oracle < 10 on VIP-Seg | S1 +8.42, S0 +15.13 | mixed: holds on S1 only, not conclusive |
| collapse watch | only positive-gain rows are checked; none has a class below −3 | not triggered (VIP-Seg S0, a negative row, loses 4.37 on `chair`) |

The first `decide` on the VM printed "does not exclude 0" for the two S0 CIs, which lie entirely below 0;
the check only looked at the lower bound. Fixed after the run (EM-13 now covers a CI below 0); the rule
verdicts were not affected.

## What the probe establishes

1. **The query-side headroom is real.** With the query labels, the same update gains +8.4 to +21.8
   points; at the frozen κ = 0.5 it already gains +3.1 to +9.7. The error that support prototypes leave
   is recoverable in principle from the query's own points (QGE T1 measured the same on its model).
2. **The model's own posterior cannot recover it.** The best pseudo-label arm gains +0.5 on the selection
   draw, and every larger step costs more. An update toward the points the model already assigns to a
   class reinforces its current decision boundary instead of correcting it: the confirmation bias that
   self-training methods guard against. The selection gain on S1 does not transfer to S0 (−1.21 on
   VIP-Seg, CI entirely below 0).
3. **The posterior entropy does single out reliable points, which is the premise the paper asserts for its
   own entropy and never measures.** Accuracy by weight bin, w < 0.5 / 0.5–0.9 / ≥ 0.9:

   | checkpoint | fraction of points | accuracy |
   | :--- | :--- | :--- |
   | VIP-Seg S1 | 9.9 / 29.7 / 60.5 % | 62.4 / 80.1 / 94.7 % |
   | ours S1 | 27.7 / 34.5 / 37.8 % | 55.1 / 77.1 / 90.2 % |
   | VIP-Seg S0 | 31.7 / 47.1 / 21.2 % | 73.4 / 89.3 / 94.8 % |
   | ours S0 | 29.2 / 31.0 / 39.8 % | 52.1 / 72.8 / 87.7 % |

   Reliable points are identified, but 5–12 % of the most confident ones are still wrong, and a soft
   entropy weight is beaten by SSP's hard thresholds on both S1 checkpoints. Identifying reliable points
   is not enough to purify a prototype with them.
4. **Confident masses** (entropy weight): foreground 0.155–0.202, background 0.327–0.446 per query block,
   inside the 0.1–0.3 foreground range D-26 had estimated without measuring.

## Consequences

* D-26's direction is closed under its own rule P0.2; R2 (training with the refinement in the loop) is
  not run. What P0 does not test: a refinement **trained** end to end with per-step supervision, whose
  step sizes could learn to stay below the harmful range. Only the oracle gap argues for it; nothing in
  P0 does, so it is not pursued without new evidence.
* Next by D-26's own rule: base-class calibration (research note 2026-09-23 §5, direction 4).
* The per-bin accuracies in point 3 are a publishable measurement in their own right: the entropy that
  identifies reliable query points is the class posterior's, not Eq.10–12's per-channel one.
