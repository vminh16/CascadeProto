# Phase 16 R1: where the 15 points between our stage and VIP-Seg's sit

VM run of 2026-09-23, `experiments/run_r1.sh`. S3DIS **S1** (the screening fold of [DECISION D-22];
S0 stays held out), 2-way 1-shot, T = 1, no LMA, 9,600 training episodes = 2,400 steps at batch 4, no
LR decay, scored on 300 valid episodes of the fold's test classes. **Two seeds per variant.**

These numbers are not comparable with the S0 fixed100 numbers of the reproduction report: different
fold, different episode draw, 40 % of the schedule.

| variant | what differs from `r1_eppm` | seeds | mean | sd | vs `r1_baseline_l2` |
| :--- | :--- | :--- | ---: | ---: | ---: |
| `r1_baseline_l2` | no stage at all, L2-normalised prototypes | 0.5445 / 0.5759 | 0.5602 | 0.0222 | — |
| `r1_eppm` | the stage as the paper prints it (D-01 class slots) | 0.5256 / 0.5355 | 0.5305 | 0.0070 | −2.97 (t = −1.80) |
| `r1_pooled` | one `S′` for the episode [DECISION D-23] | 0.5349 / 0.5328 | 0.5339 | 0.0015 | −2.63 (t = −1.67) |
| `r1_eppms` | the stripped stage + pooled `S′` + L2 [DECISION D-24] | 0.5008 / 0.4845 | 0.4927 | 0.0115 | −6.75 (t = −3.82) |
| `r1_vippem` | one VIP-Seg PEM + L2 [DECISION D-25] | 0.6827 / 0.6878 | **0.6852** | 0.0036 | **+12.50 (t = +7.86)** |

Not run: `r1_eppms_slots` (separates stripping from the support reading inside EPPM-S) and `r1_vip4`
(VIP-Seg's four alternating modules).

## The decision rules, fixed before the run

| Rule | Measured | Verdict |
| :--- | ---: | :--- |
| R1.1 `pooled − eppm ≥ +3`, `t > 3` | **+0.33**, t = +0.65 | **not met** — the support reading is not the cause |
| R1.2 `abs(eppms − vippem) ≤ 2` | **−19.26**, t = −22.55 | **not met** — EPPM-S does not reach the reference head |
| R1.3 `eppms − eppm ≥ +3` | **−3.79**, t = −3.97 | **not met, and negative** — stripping made it worse |
| R1.4 `vippem − eppm ≥ +3` | **+15.47**, t = +27.78 | **met** — take route B (research note §5.8) |

**The budget resolves effects of this size.** `r1_vippem` beats `r1_baseline_l2` by 12.50 points with
t = +7.86 at exactly this budget, so a variant that shows nothing here is not hidden by the budget;
it has nothing of that size to show. This is the positive control the queue lacked as an explicit rule.

## Readings

1. **The leading hypothesis is refuted.** D-23 was ranked first because QUEST, APP, PEM/PDM and DPA all
   apply one episode-level correlation while D-01 gives each class slot its own, and because the
   ablations that isolate that branch credit it with +15.4 and +15.2 on S0 (research note §4.3–4.4).
   Measured directly, with everything else fixed, it is worth **+0.33 points**.
2. **Removing the inert parts made the stage worse, not better** (−3.79 against the printed stage,
   −6.75 against no stage at all). The static analysis behind D-24 — that `P_diffuse`, the entropy
   gate, the SE block and Eq.21's ReLU cannot help — does not imply that a stage without them helps.
3. **The stripped stage is below plain prototype matching on this fold and budget** (−6.75, like for
   like: both have the L2 input), while one VIP-Seg module is 12.5 points above it. The 15-point gap is
   now measured **inside one pipeline with only the stage changed**, not between two training scripts
   as in CHANGELOG 15e. For `r1_eppm` and `r1_pooled` the comparison with `baseline_l2` is **not** like
   for like; see the L2 confound below.
4. **Route B is the decision.** Build on VIP-Seg's head (`stage_type=vip`) and put the phase-16
   additions on top of it, rather than repairing the printed stage.
5. **A starting point that is really there.** `r1_vippem` reaches 0.6852 after 2,400 steps on S1, where
   VIP-Seg's released S1 checkpoint is cited at 0.7609. The head is VIP-Seg's contribution, so anything
   published on top of it has to be new above the head, not the head itself.

## Is the budget too short? No — the curves plateau at epoch 20

Validation curves of the full-schedule S0 runs already in the repository (`train.py`, LR decay,
validation every 10 epochs):

| run | ep 10 | ep 20 | ep 30 | ep 40 | ep 50 | change 20 → 50 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | .4642 | .4949 | .4607 | .4942 | .4931 | −0.2 |
| baseline + L2 | .5314 | .5057 | .5200 | .4976 | .5048 | −0.1 |
| + LMA (T = 0) | .4985 | .4809 | .4779 | .4845 | .4807 | 0.0 |
| T = 1 | .5556 | .5670 | .5533 | .5757 | .5767 | +1.0 |
| T = 4, no ADRM | .5129 | .5710 | .5561 | .5634 | .5599 | −1.1 |
| full model | .4942 | .5803 | .5745 | .5723 | .5741 | −0.6 |

Sources: `results/phase14_p1/phase14/*/log_train.txt`, `results/phase15_full/*/log_train.txt`.

No configuration moves by more than 1.1 points after epoch 20, which is the screening budget. A
15-point verdict cannot be a budget artefact; a 0.33-point one cannot become a 3-point one either.
The one result where ±1 point matters is R1.3 (−3.79 could soften to about −3), and its sign does not
turn.

## The L2 confound, and what survives it

`r1_eppm` and `r1_pooled` run with `l2norm_point_proto=false`, the paper's default [DECISION D-10];
`r1_eppms`, `r1_vippem` and `r1_baseline_l2` run with `true`, because VIP-Seg's head is built on
normalised prototypes [VIPSEG models/vipseg.py:142]. L2 alone measured **+3.36** on the S0 baseline
(report §2, one seed), which is the size of R1's own +3 thresholds. Raised in
`docs/research/2026-09-23_gap_and_upgrade_research.md` §4.5 and verified here against
`experiments/diag_short.py`. Per comparison:

| comparison | confounded? | effect on the verdict |
| :--- | :--- | :--- |
| R1.1 `pooled − eppm` (+0.33) | **no**, both without L2 | the refutation of D-23 stands unchanged |
| R1.3 `eppms − eppm` (−3.79) | yes, in EPPM-S's **favour** | like for like EPPM-S is worse by up to ~7, so the negative verdict is strengthened |
| R1.4 `vippem − eppm` (+15.47) | yes, in PEM's favour by up to ~3.4 | the stage-only gap is ~12–15; route B still fires |
| `eppms` vs `baseline_l2` (−6.75) | **no**, both with L2 | a stripped stage is worse than no stage |
| `eppm`/`pooled` vs `baseline_l2` (−2.97 / −2.63) | yes, against the stages | says nothing on its own; needs `r1_eppm_l2` |

**The missing run is `r1_eppm_l2`** (the printed stage with the L2 input): two to three runs, 1.1–1.6
GPU-hours, and it turns every row above into a like-for-like comparison.

## What is not settled

* **Two seeds**, not three. With n = 2 only the 15-point gap is beyond doubt; −3.79 is indicative and
  +0.33 is nothing. The seeds of `r1_baseline_l2` differ by 3.1 points, which is the noise floor.
  At the harness's seed spread three seeds resolve a true difference of about 5 points, not 3, so R1's
  own thresholds are optimistic and R1.2 cannot establish equivalence at all
  (`2026-09-23_gap_and_upgrade_research.md` §4.5). Read R1 as a screen, confirm on the full schedule.
* **No curve for EPPM-S or PEM.** The plateau evidence above comes from the EPPM family and the
  baselines; a different architecture could have a different curve shape.
* **Why PEM wins is not established.** Three candidates, in the order worth testing: the LayerNorm on
  the self term that EPPM-S does not have (CHANGELOG 15i had named it and 16c omitted it), PEM's two
  separate query/support gates against EPPM-S's difference gate, and the query mixing that VIP-Seg's
  reshape performs and our clean forms deliberately avoid. The third matters beyond this comparison:
  if mixing queries is what pays, that is evidence for the transductive direction of research note §5.2.
* **S1 valid draw only.** No fixed100 test number, no S0 number, no full schedule.

## Against the predictions of the research note

| Prediction (note §5.8, 2026-09-22) | Outcome |
| :--- | :--- |
| Route A stage 0: 64–71 | EPPM-S is at 49.3 on this fold and budget; the route-A premise fails |
| Route B stage 0: 71–72 | one PEM reaches 68.5 at 40 % of the schedule; on track |
| D-23 is the leading hypothesis for the gap | refuted, +0.33 |
