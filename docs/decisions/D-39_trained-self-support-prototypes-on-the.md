### D-39 — Trained self-support prototypes on the clean base, and the combined background rule · `PROPOSED`, beyond the paper

* **Problem.** P6 (D-38 outcome) on the clean base CR: the prototype gap is joint (oracle foreground +16.3,
  background +6.1, both +25.1 on S1 fixed100); a label-free adaptation of the background row (query self-support,
  ρ = 1, α = 0.25, two steps) gives +1.19 and a 3-component background +1.32, positive on every draw, while
  self-support on the foreground rows lowers the score (best −0.35 on valid) and the support directions alone (U)
  already beat CR's trained head by +0.9. Two questions follow. (a) Do the two background mechanisms add? (b) Does
  self-support on the foreground work once the features are **trained** with it? Training-free, the pseudo-labels come
  from the prototype whose misses they should repair; trained end-to-end, the features can learn to make the query's
  own confident region a better prototype (SSP [Fan et al., ECCV 2022] trains its self-support prototypes). Since the
  head of VIP-Seg adds nothing on the clean base, the trained arms drop it and put self-support in its place; this is
  the architectural change a paper would claim.
* **What it does.**
  * **Model** (`models/self_support.py`, `models/cascadeproto.py`, beyond the paper). `prototype_rule=unit`: the
    prototypes are the support directions of D-35/D-38 (normalised sums of unit features: background from all
    support background points, each way's foreground), logits `F^q Rᵀ` with unit rows; requires `num_stages=0`,
    `use_lma=false`, `l2norm_point_proto=false`. `self_support_steps=T`: T steps of P6's rule on every row with ρ = 1,
    trained: per query block, the points predicted c (argmax, no gradient through the assignment) give
    `S_c = n(Σ n(f_i))`, and `R_c ← n(α_r R_c + (1 − α_r) S_c)` when at least 16 points are predicted c;
    `α_bg = σ(θ_bg)`, `α_fg = σ(θ_fg)` learned, initialised at 0.25 (P6's frozen background value) and 0.5 (neutral).
    The logits are those of the last step. `support_aux=w`: in training, `+ w · CE(F^q R_0ᵀ)`, the step-0 (support
    only) prediction, so the pseudo-labels are trained too; w = 1, not tuned (not measured). Every operation is per
    query block: order-free by construction (checked as D37-T5).
  * **Arms** (S1, E1's schedule: batch 1, 24,000 updates, LR halved every 7,200, 13 validations, seed 0, random query
    order as D-37's arms): **A0** `prototype_rule=unit`, T = 0 (trained prototype matching, no head); **A1** A0 with
    T = 2 and `support_aux=1`. CR (D-37) is the reference; no new run for it.
  * **Inference background rules** (no training): `+ssp_bg` = P6's frozen background self-support (ρ 1, α 0.25, T 2)
    on the model's rows; `+km3` = background logit the maximum over the model's background row and the 3 spherical
    k-means directions of the support background (P6's k-means); `+both` = `+ssp_bg`, then `+km3` on its rows. On CR
    they act on U's rows (as in P6), on A0 / A1 on the model's own rows.
  * **Test.** CR, A0, A1 (`last.pt` headline, `best.pt` reported) with every inference rule on fixed100, random600
    seeds 0–2 and the leak-free draw (seed 4), plus the presence-fair oracle of P6 on each model's features.
  * **Checks, every episode.** The scoring rule reproduces each model's logits; A0's rows equal U's; fixed100 model
    mIoU equals VIP-Seg's metric, CR's equals D-37's 54.84.
* **Rules, fixed before the run** (paired bootstrap over episodes; "holds" = gain ≥ threshold on fixed100 with CI
  above 0 and > 0 on all three random600 draws):
  * D39.1 background combination (CR, U rows): `+both` − max(`+ssp_bg`, `+km3`) holds at +0.5 → the combined rule is
    the background rule of every later arm; otherwise the better single rule (larger fixed100 gain).
  * D39.2 trained self-support: A1 − A0 holds at +1.0 → trained self-support is the mechanism for the foreground;
    below +0.5 on fixed100 → it is not, and the foreground needs a different trained module.
  * D39.3 new base: the candidate is fixed by the two rules above, not by the test scores: A1 if D39.2 holds, A0
    otherwise, with the background rule of D39.1. Candidate − CR's model holds at +1.0 → it is the base of every later
    arm; otherwise CR stays and the chosen background rule is added at inference.
  * D39.4 reported: learned α_bg, α_fg; oracle gap of each model (P6's presence-fair oracle); leak-free levels;
    recall and precision per class.
* **Why these two arms only.** A0 separates "dropping VIP-Seg's head" from "adding trained self-support"; A1 is the
  one mechanism the evidence points at. About 1.5 GPU-h per arm (no PEM/PDM; not measured) and 1 h of evaluation.
* **Affects.** `models/self_support.py` (new), `models/prototypes.py` (`unit_prototypes`), `models/cascadeproto.py`
  (`prototype_rule`, `self_support_steps`, `support_aux`), `pipeline/model_api.py` (`loss_aux`), `train.py`,
  `experiments/d39_eval.py`, `experiments/run_d39.sh`, `tests/test_self_support.py`, 05 §3.8q.
* **Outcome (2026-09-25, `results/phase16_d39/SUMMARY.md`).** D39.1 → both (+0.51 [+0.24, +0.78] over km3, all
  random600 > 0). D39.2 stop: A1 − A0 −0.66 [−1.30, +0.02] (learned α_bg 0.43, α_fg 0.70). D39.3 CR stays with the
  combined background rule: CR + U + both scores 57.63 on S1 fixed100 (+2.79 over CR's head; random600 +3.0 / +2.8
  / +4.2), inference only. Training without the head (A0) gives features 2.2 worse for the same rule (53.52 against
  55.74 on CR's features).
