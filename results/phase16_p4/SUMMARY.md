# Phase 16 P4: is E1's background prototype contaminated by the episode's own classes? [DECISION D-32]

VM run of 2026-09-24 (07:37–07:44 UTC, NVIDIA L4), commit `ef19410`, `experiments/run_p4.sh`. No training.
E1 `last.pt`, S1. Two seeded draws of the S1 test classes (100 episodes per pair each): A (seed 1)
selects q, B (seed 2) tests. The support points carry the other way's labels (inherited
`sample_pointcloud(support=False)`, BG-9). The re-run head reproduced E1's logits on every checked episode.

**Verdict: P4.1 not causal — the contamination hypothesis is refuted; the line stops (P4.3 not reached).**

## Test (draw B, paired bootstrap)

| arm | gain over E1 [95 % CI] |
| :--- | :--- |
| `clean_bg` — background row without the other way's points (labels), the intervention | **+0.09 [+0.04, +0.14]** |
| `purify_q0.1` — label-free, frozen on draw A (+0.00 there) | +0.02 [−0.01, +0.05] |
| background row replaced by the query's own direction only | −26.07 [−27.34, −24.76] |
| foreground rows replaced only | −9.05 [−9.99, −8.11] |
| all rows replaced (the oracle) | +13.71 [+12.73, +14.70] |

Floor recall 0.724 → 0.733 and wall 0.737 → 0.738 under `clean_bg`. Contamination of the pooled
background: 10.0 % of its points (draw B), as C0 measured; the label-free purifier ranks them with AUC 0.789.

## What it establishes

1. **Removing every contaminating point, with the labels, changes nothing that matters** (+0.09): the
   correlation of C0 (Spearman +0.70 over pairs) was confounded. Pairs with floor or wall are hard for
   another reason than the background row.
2. **The oracle's gain is joint, not row-wise.** Replacing only the background row costs 26 points,
   only the foreground rows 9, all rows together gains 13.7: the query's own directions share its common
   component, and only the full set is consistent. The error is a query-conditioned shift of every
   prototype at once, the transductive gap of D-26, not one contaminated row.
3. A learned background-purification neck is not warranted by this evidence (P4.3).

## A follow-up reading, not a rule (CPU, raw block labels)

The classes that are everywhere — ceiling in S0 (98 % of training blocks), floor (96 %) and wall (59 %) in
S1 — are always background during training, because they are the other fold's test classes. On VIP-Seg's
released checkpoints (P0) they are among the lowest IoUs (64.4, 65.3, 69.7) with large oracle gaps (20.9,
18.1, 15.8). Over the 12 test classes of both folds, the share of training-block points a class covers
correlates with its IoU at Spearman −0.55 (p = 0.06) and with its oracle gap at +0.27 (p = 0.40): suggestive,
not significant. A causal test would need those classes' labels during training, which the protocol
forbids (D-21, D-22); it could only run as a diagnostic, never as a method.
