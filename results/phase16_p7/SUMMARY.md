# Phase 16 P7: label propagation on the query's own point graph [DECISION D-40]

VM run of 2026-09-25 (NVIDIA L4), commit `b9c64a9`, `AUTOSTOP=1 bash experiments/run_p7.sh full`: valid selection
17:32–18:14, test draws 18:14–19:21 UTC. CR `last.pt` (D-37's clean base), S1, inference only. Checks passed: model
identity every episode, the solve's residual every block, VIP-Seg's metric, D-37's CR (54.84), D-39's U (55.74) and
U + both (57.63) on fixed100, P6's valid model (55.17) and U (55.96).

## Selection (S1 valid, gain over Y0 = U + both 58.09, points)

| graph | k | β 0.5 | 0.8 | 0.9 | 0.99 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| xyz | 8 | 0.00 | +0.18 | +0.21 | +0.02 |
| xyz | 16 | 0.00 | +0.22 | +0.19 | −0.49 |
| xyzf | 8 | 0.00 | +0.20 | +0.27 | +0.31 |
| xyzf | 16 | 0.00 | +0.27 | +0.34 | +0.21 |
| feat | 8 | 0.00 | +0.13 | +0.22 | +0.69 |
| feat | 16 | 0.00 | +0.18 | +0.31 | **+0.91** |

Frozen: `lp_feat_k16_b0.99` (Iscen's graph on the query's unit features, k 16, β 0.99). Gate P7.1 passes. With hard
seeds, β = 0.5 never overturns a point's own seed (every arm exactly 0.00).

## Test (mIoU %, identical episodes)

| rule | fixed100 | random600 s0 / s1 / s2 | leak-free |
| :--- | ---: | :--- | ---: |
| CR model | 54.84 | 51.83 / 56.73 / 54.76 | 28.10 |
| U | 55.74 | 53.18 / 57.55 / 57.10 | 33.02 |
| U + propagation (`lp_u`) | 56.67 | 54.07 / 58.25 / 58.32 | 33.49 |
| U + both (Y0) | 57.63 | 54.84 / 59.54 / 58.93 | 33.14 |
| **U + both + propagation (`lp`)** | **58.55** | **55.75 / 60.57 / 60.35** | 33.31 |

## Rules

* **P7.1 gate pass**: +0.91 on valid.
* **P7.2 adopt**: `lp` − Y0 +0.92 [+0.72, +1.13] on fixed100; random600 +0.91 / +1.03 / +1.42. **P7.3 not reached**
  (+1.0 needed).
* **P7.5 leak-free reported**: +0.17 (not ≤ 0, so not recorded as protocol-dependent; small).
* **P7.6 mechanism unexplained**: foreground points fixed 53,173 against broken 41,824 (net +11,349), but the fixed
  points' mean homophily (0.944) is not above the broken points' (0.947). The registered recall mechanism is not what
  happens (below).
* **P7.7 composition**: `lp_u` − U +0.92, `lp` − Y0 +0.92: propagation and the background rules add fully.

## What it establishes

1. **Propagation is a small, consistent, additive gain (+0.9), and it works as a denoiser, not as a recall
   mechanism.** Foreground precision rises (floor 0.819 → 0.838, sofa 0.759 → 0.775, table 0.660 → 0.676), recall
   barely (+0.1 to +1.2 per class); in the background column it fixes 44,164 points and breaks 28,551 (false positives
   removed). The best graph is the feature graph with long walks, not the spatial graph: the gain comes from
   feature-manifold smoothing, not from spatial continuity.
2. **Why recall is not restored (P7a, fixed100, feature graph k 16; the six graphs agree).** Homophily is high
   everywhere (missed foreground points 0.93–0.96), so the graph is not the problem. The seeds are: around a missed
   own-condition foreground point only 5 % of its same-class neighbour weight is predicted correctly (a_miss 0.049;
   0.04–0.09 over the graphs), and only 2–5 % of missed points are fixable by a one-hop vote. **Recall errors are whole
   regions, not scattered points**; smoothing cannot overturn a region whose seeds are all wrong. This is the
   "seed errors cluster" failure mode the note predicted (§5).
3. **The other sampling condition is the largest single error source.** Y0's foreground recall is 0.774 on
   own-condition points and **0.162** on other-condition points (classes sampled at background density, D-35).
   Other-condition points are 11.5 % of the foreground points (343,162 of 2,983,699) but 32.6 % of the missed ones
   (287,651 of 883,481). Propagation makes them worse (recall_other falls for every class, e.g. floor 0.151 → 0.120):
   sparse points are absorbed by the surrounding background, as the note predicted for heterophilous sparse regions.
4. Leak-free: propagation breaks more than it fixes (56,992 fixed, 77,656 broken) and gains only +0.17; the standard
   protocol's +0.92 is larger than the leak-free one.
