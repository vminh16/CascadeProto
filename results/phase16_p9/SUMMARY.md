# Phase 16 P9: where density still enters M1, and which module the one remaining run can use [DECISION D-44]

Run of 2026-09-27 on a rented RTX 3090 (vast.ai), commit `2060472`, 12:53–13:16 UTC (23 min; step 0 and part B in
parallel, part A after step 0 next to part B). Local copy of the blocks (D-44 amendment 3): the seeded draw gives
1,053 events in 888 episodes (P8/D-43's VM draw: 1,045). Checks passed: U on valid 55.96 (P8's split), step 0 and
part A drew the same events, a block encoded alone within 2.4e-4 of the model's encoding, and on the first five
events the uncapped grouping, the statistics recording and the own-statistics injection reproduced the encoding bit
for bit. Peak VRAM 0.4 GiB per process.

## Step 0: distinct points per ball (mean m, share below 16)

| stage (radius) | c in V0 (sparse) | c in V1 (dense) | a in V0 (protocol) | a in V2 (uniform) |
| :--- | :--- | :--- | :--- | :--- |
| 1 (0.1 m) | 11.81, 69 % | 15.13, 32 % | 15.22, 30 % | 13.45, 52 % |
| 2 (0.2 m) | 15.44, 14 % | 15.85, 4 % | 15.91, 3 % | 15.74, 7 % |
| 3 (0.4 m) | 15.93, 2 % | 15.96, 1 % | 15.99, 0 % | 15.98, 1 % |

Identical to the local preview (same data). D44.0: sparse balls at stage 1, the cap arms ran.

## Part A: M1, 1,053 events

References of this draw (D-43's VM draw in brackets): R_V0 0.281 (0.276), R_V1 0.760 (0.770), R_a(V0) 0.786 (0.789),
R_a(V2) 0.372 (0.358).

ψ = share of the density gap an intervention reproduces, pooled, 95 % CI over episodes:

| arm | other (V1 made sparse) | own (V0 made uniform) |
| :--- | :--- | :--- |
| cap stage 1 | −0.006 [−0.019, +0.007] | −0.006 [−0.014, +0.003] |
| cap stage 2 | +0.001 [−0.000, +0.003] | −0.001 [−0.002, −0.000] |
| cap stage 3 | −0.000 [−0.001, +0.000] | +0.000 [−0.000, +0.000] |
| cap all | −0.003 [−0.015, +0.009] | −0.008 [−0.015, −0.000] |
| block statistics | −0.006 [−0.012, −0.001] | +0.001 [−0.005, +0.006] |
| caps + statistics | +0.012 [−0.002, +0.026] | +0.006 [−0.003, +0.015] |

A1, cosine of the same raw points' features in the two versions (class points / background points):

| pair | embedding | stage 1 | stage 2 | stage 3 | 128-d feature |
| :--- | ---: | ---: | ---: | ---: | ---: |
| V0–V1, class c | 1.000 | 0.989 | 0.973 | 0.898 | 0.740 |
| V0–V1, background | 1.000 | 0.988 | 0.977 | 0.951 | 0.922 |
| V0–V2, class a | 1.000 | 0.991 | 0.979 | 0.911 | 0.796 |
| V0–V2, background | 1.000 | 0.989 | 0.978 | 0.956 | 0.928 |

## Part B: CR, valid (1,500 episodes; 1,500 base-class training episodes for Σ_η)

| arm | mIoU | | arm | mIoU |
| :--- | ---: | --- | :--- | ---: |
| model | 55.17 | | proj r 4 / 8 / 16 | 43.92 / 44.86 / 39.64 |
| U | 55.96 | | k-means k 2 / 3 | 53.86 / 52.93 |
| cosine oracle | 80.84 | | label-free LDA λ 0.1 / 0.3 / 0.5 | 31.11 / 32.79 / 35.44 |
| LDA oracle λ 0.1 / 0.3 / 0.5 | 96.04 / 94.83 / 93.82 | | best composite (r 0, λ 0.5, k 3) | 36.55 |

Nuisance: κ / ρ medians 0.44 / 0.69 (r 4), 0.71 / 0.86 (r 8), 0.87 / 0.93 (r 16); projecting lowers cos(d, d*)
(median −0.16 to −0.21). Heads: CV 0.13 / 0.23 (PCA, H 4 / 8), 0.03 / 0.11 (random). Collapse: participation ratio
**5.56** of 128; median share of d* in the span of the six base-class means and the background mean **0.857**;
‖mean unit feature‖² 0.56. Retrieval purity (16 nearest support points): hit 0.886, missed **0.109**.

## Rules

* D44.0 sparse balls: cap arms run.
* **D44.1 and D44.1b: the encoder branch is closed.** Neither the ball count nor the block statistics carry any of
  the density gap (every ψ within ±0.02).
* D44.2: **P9.3 metric admissible** (only through the oracle clause: LDA oracle 96.04 against the cosine oracle
  80.84; the label-free LDA is 20 points below U). P9.4 nuisance, P9.5 heads, P9.6 components, P9.7 representation,
  P9.3c composite and P9.8 neck fail.
* D44.3: the one remaining run is chosen in a new decision.

## What it establishes

1. **The density dependence is not in the counts or the statistics that D-43 and amendment 2 targeted.** The
   features of the same points stay close through the encoder (stage-3 cosine 0.90) and separate in the 128-d
   feature (0.74 for the class, 0.92 for the background). Where the encoder does diverge, at stage 3, every ball is
   full (m ≈ 16), so the count cannot be the reason; a candidate that no arm tested is the class composition of
   each neighbourhood (a sparse object's 0.4 m ball holds relatively more background points), which any
   neighbourhood aggregation inherits. Not measured.
2. **The features are collapsed to about six directions.** A participation ratio of 5.56 with seven training
   classes (six base classes and the background) is what neural collapse predicts (class means spanning C − 1
   directions, within-class spread shrinking); the novel discriminant d* lies 86 % inside the base-class span, and
   the nuisance and the discriminant share the same directions (κ < ρ at every r), so no projection separates them.
   This explains why every prototype, projection, component and propagation rule saturates: the novel classes must
   be told apart inside a space shaped for the base classes.
3. **The information is in the query block but not reachable from one support.** With the query's own class
   statistics a linear metric separates the block almost perfectly (LDA oracle 96, in-sample and therefore
   optimistic), but the missed points' nearest support points carry the wrong label 89 % of the time, and every
   label-free estimate of the metric (query whitening, projection, components, their composition) loses to U.
4. **P9.7 was specified the wrong way round for this regime.** Its rule reads a large base-span share as "base
   supervision already covers the novel discriminant". With a participation ratio of 5.6 the share is large because
   every direction lies in the base span; the research note (§6) pre-registered an interpretation only for a low
   ratio with a small share. The rule's verdict (fail) is reported as registered; reading the collapse as a reason
   to train against it would be a post-hoc interpretation and needs its own decision. (Before the run this rule was
   expected to pass trivially; it did not.)
