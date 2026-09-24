# Phase 16 P3: where E1's prototype error sits, and a training-free text prior [DECISION D-31]

VM run of 2026-09-24 (04:55–05:17 UTC, NVIDIA L4), commit `c3cbb7e`, `experiments/run_p3.sh`; the VM shut
itself down at 05:17 (`autostop.log`). No training. E1 `last.pt` (route B's base, D-30), S1 only. Bank of
base prototypes from 1,000 training episodes (673–966 occurrences per class). `protocol: clean`.

**Verdicts.** Part B: **P3.0 fails** (text on this feature space stops); no arm of 192 gains on the valid draw,
and even a per-episode weight chosen with the query labels gains only +0.70 [+0.52, +0.88]. Part A: the head
recovers 65 % of the support → oracle gap; the fixable points are enriched at boundaries (1.54×, the
"boundary-enriched" band) but 83.5 % of them are interior, they do not depend on support size, and they are
mostly **floor and wall points predicted as background**.

## Part B — text prior (selection on S1 valid, 1,500 episodes; test on fixed100)

| | value |
| :--- | :--- |
| arms with a positive valid gain | **0 of 192** (best −0.001, the smallest κ; worst −11.9) |
| frozen arm | `ridge_bare_entropy_k0.25`, fixed100 −0.00 [−0.00, +0.00] |
| oracle-weight upper bound (κ per episode chosen with the labels) | **+0.70 [+0.52, +0.88]** |
| text-only fg-vs-fg accuracy on valid queries, per source × prompt | 0.52–0.62 (best: retrieval τ 30 + ensemble 0.624, bare 0.614; ridge + descriptions 0.612) |
| alignment with the oracle's correction | −0.01 to +0.19; CI above 0 for 9 of 12 combinations |

P3.0 is applied to the frozen combination (accuracy 0.552, below 0.60). The best combinations clear the bar
(0.61–0.62, alignment CI above 0), yet no weight of their prior raises the mIoU on valid: text carries a weak,
real fg-vs-fg signal, too weak to move decisions the head already gets right.

## Part A — the gap (fixed100, 1,500 episodes)

| rule | mIoU | door | floor | sofa | table | wall | window |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| support prototype, no head | 49.27 | 42.5 | **66.0** | 51.8 | 35.9 | 51.6 | 47.8 |
| E1 | 73.20 | 73.9 | 62.1 | 75.4 | 69.8 | 69.5 | 88.6 |
| oracle, one common norm | 85.93 | 86.8 | 86.7 | 90.0 | 71.9 | 88.0 | 92.2 |

* **Head recovery** (E1 − support) / (oracle − support) = **0.65**.
* **Fixable points** (E1 wrong, oracle right) = 10.8 % of all points; boundary share 16.5 % against 10.8 % of
  all points (enrichment 1.54); **83.5 % of fixable points are interior**.
* **Support size does not matter**: fixable share by support-mask tercile 0.331 / 0.336 / 0.333; Spearman of the
  per-episode oracle gain with the support foreground fraction +0.02.
* **What the oracle fixes** (recall, E1 → oracle): floor 0.719 → 0.990, wall 0.725 → 0.954, door 0.898 → 0.957,
  sofa 0.946 → 0.988; background precision 0.861 → 0.977. E1 under-predicts floor (pred/gt 0.88) and wall (0.77).
* **The head makes floor worse than no head**: support rule floor IoU 66.0 and recall 0.874, E1 62.1 and 0.719.

## What it establishes

1. **Text cannot add usable information on this feature space** in the training-free form (P3.0 per rule; no
   arm positive; upper bound +0.70). The prior re-ranks foreground ways only, and fg-vs-fg is not where E1
   loses.
2. **E1's dominant error is foreground → background on the large planar classes.** 14 % of E1's background
   predictions are foreground (precision 0.861); the oracle removes most of that (0.977).
3. **The error does not come from support coverage** (flat across support-size terciles), and the head, which
   recovers 65 % of the gap overall, loses floor points the plain support prototype kept.
4. Against the pre-registered bands: enrichment 1.54 falls in "boundary-enriched", but by mass the fixable
   points are interior region-level errors (83.5 %). The band was defined on enrichment alone; the mass reading
   is recorded here as the more relevant one for mIoU and is an inference, not a pre-registered rule.

## Open question this raises (next decision)

Floor and wall appear in almost every S3DIS block. In an episode, the other way's support block labels them
background, and the background prototype pools the mask-0 points of all support blocks
[VIPSEG models/vipseg.py:108-116]; the head then adapts that row toward the query, whose blocks are dominated by
the same surfaces. A background prototype contaminated by the episode's own foreground classes would produce
exactly this fg → bg pattern. Not yet measured: the share of background support points that belong to another
way's class, and the part of the gap that replacing only the background prototype recovers.
