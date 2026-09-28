### D-45 — M2: VICReg's variance and covariance terms against the features' collapse, one seed per arm · `PROPOSED`, beyond the paper

* **Problem.** P9 (D-44 outcome) closed the encoder branch and left every inference-time rule below U; on the clean
  base CR the point features have a participation ratio of **5.56** of 128, the novel discriminant d* lies 86 %
  inside the span of the base-class means, the nuisance and the discriminant share the same directions (κ < ρ at
  every r), and the missed points' nearest support points carry the wrong label 89 % of the time, while the query's
  own statistics would separate the block (LDA oracle 96.04). Seven training classes (six base classes and the
  background) and a participation ratio near C − 1 = 6 is the signature of neural collapse (class means spanning
  C − 1 directions, within-class variability shrinking). This decision reads the collapse as the training target.
  **The reading is post hoc**: P9.7, registered to decide M2, failed (share 0.857 > 0.5) because its rule assumed
  a non-collapsed space (D-44 outcome); the rules below are fixed before any training.
* **What it does** (`models/vicreg.py`, `vicreg_var` and `vicreg_cov` in the configuration; beyond the paper). The
  training loss becomes CE + μ v(Z) + ν c(Z) on Z, the query point features of the episode [B_q·P, 128]:
  v(Z) = (1/d) Σ_j max(0, γ − sqrt(Var(z^j) + ε)), c(Z) = (1/d) Σ_{i≠j} C(Z)_ij², γ = 1, ε = 1e-4, C with 1/(n − 1)
  [Bardes, Ponce, LeCun, ICLR 2022, Eqs. 1–4, §4.2; re-checked in the PDF]. The invariance term is not used (the
  episode CE takes its place). Unlike the paper (an 8192-d expander, §4), the terms act on the 128-d features the
  prototypes are built from, the space P9 measured. Nothing changes at evaluation.
* **Arms** (CR's configuration and schedule otherwise: `vip_clean`, four stages, no LMA, L2 point prototypes, random
  query order, batch 1, 24,000 updates, LR halved every 7,200, 13 validations, seed 0; one seed each, run together on
  one GPU):
  * **M2-A**: μ = 1, ν = 0.04. The paper sets λ = μ = 25, ν = 1 (§4.2); dividing by 25 keeps its variance :
    covariance ratio and gives the task term (our CE in place of its invariance term) weight 1. A convention, not
    measured.
  * **M2-B**: μ = 4, ν = 0.16, four times A, in case A's weight is too small against the CE to move the spectrum.
* **Monitor during training** (`experiments/d45_monitor.py`). At every validation epoch each arm's current weights
  are scored on the first 300 valid episodes: the participation ratio of the unit query features and U's mIoU.
  **Early stop**: an arm whose ratio is below 8 at epoch 24 (half the schedule) is stopped, its mechanism not
  engaging (a convention between CR's 5.56 and the target 12).
* **Collapse census** (same script, inference, next to the training): the participation ratio and U on the same 300
  episodes for every S1 checkpoint kept: CR, VR, E1, r0, D-29's distilled run, N1 (both), N2, D-39's A0 and A1, M1
  and VIP-Seg's released S1 model. Reported, no rule: whether the collapse is a property of this training or of one
  run.
* **Test** (after training; `experiments/d43_eval.py test`, the same reader as D-43, CR and both arms): model, U,
  U + both, U + both + LP on fixed100, random600 seeds 0–2 and the leak-free draw of this machine's data (D-44
  amendment 3); P9 part B (`p9_placement_probe.py modules`) on each arm that reaches the end of its schedule.
* **Rules, fixed before the run** ("holds at g" = fixed100 gain ≥ g with a paired CI above 0 and > 0 on all three
  random600 draws, `last.pt`):
  * D45.1 mechanism: participation ratio of the arm on valid (part B's full measurement) ≥ 12 (c). An arm that
    fails does not un-collapse the features; its score change, if any, is reported as unexplained.
  * D45.2 base: an arm with D45.1 and U − CR's U holding at +1.0 becomes the base (the arm with the higher valid U if
    both). U is the rule without parameters selected on CR's features; the stack (both, LP) was selected on CR and,
    as D-43 found, does not transfer, so it is reported and re-selected on a new base before it is used.
  * D45.3 D45.1 holds and D45.2 fails: the space widened and U does not use it; part B's preconditions on the arm
    (N, H, C, metric, neck) decide the next block.
  * D45.4 both arms fail D45.1: VICReg on the features does not undo the collapse at these weights; the M2 line stops.
  * D45.5 reported: every rule of the stack on every draw, the leak-free draw, `best.pt`, the monitor's curves, the
    census.
* **Cost.** Two trainings and the monitor on the rented RTX 3090 (vast.ai), about 3–4 h together (not measured;
  CR's schedule took 2 h alone on the L4), then about 1.5 h of tests.
* **Affects.** `models/vicreg.py` (new), `models/cascadeproto.py` (`vicreg_var`, `vicreg_cov`), `pipeline/model_api.py`
  (`loss_reg`), `train.py` (`--vicreg_var`, `--vicreg_cov`, run tag `_vic<μ>_<ν>`), `experiments/d45_monitor.py`,
  `experiments/run_d45.sh`, `experiments/p9_placement_probe.py` (the valid check of part B only for CR),
  `tests/test_vicreg.py`, 05 §3.8v.
* **Outcome (2026-09-27, `results/phase16_d45/SUMMARY.md`): D45.3.**
  * **Run.** RTX 3090, local data. The host rebooted twice; each time the trainings were resumed from their last
    epoch with nothing else changed. CR's references hold on this GPU (fixed100 54.84 / 55.74 / 57.63 / 58.55).
    Neither arm was stopped early.
  * **D45.1 holds for both arms.** Participation ratio 42.45 (A) and 95.25 (B), against CR's 5.56.
  * **D45.2 fails for both arms.** U − CR's U on fixed100 is −5.20 [−6.02, −4.34] (A) and −11.34 [−12.34, −10.36]
    (B), and it is negative on every random600 draw. `best.pt` and the leak-free draw are also below CR.
  * **Part B: the widened space holds no more class information.**
    - Cosine oracle 79.74 / 79.27 and LDA oracle 94.43 / 92.73, against CR's 80.84 / 96.04.
    - Every label-free rule remains below U.
    - d*'s base-span share falls from 0.857 to 0.50; missed-point purity rises from 0.109 to 0.23 / 0.26.
  * **Census.** Every kept S1 checkpoint reads 3.7–7.3, including VIP-Seg's released model (6.39). The collapse
    belongs to this episodic training and does not bound the cosine head (CR reaches its oracle 80.84 while
    collapsed).
  * **Consequence.** The M2 line (VICReg on the prototypes' feature space) stops at these weights. The next block
    is chosen from part B's preconditions: as on CR, only the metric head is admissible, through the LDA oracle.
    That choice needs its own decision.
