### D-34 — N2: the neck of D-33 trained from scratch on E1's schedule · `PROPOSED`, beyond the paper

* **Problem, from N1.** Warm-started from E1 at a constant 1.25e-4 with α = 0, the neck ended at α = 0.0023
  after 7,200 updates and both arms' validations moved together (ctl 58.91 / 70.17 / 70.95, neck 59.23 /
  70.45 / 70.80); restarting AdamW's moments also dropped both arms from E1's 73.04 to 59 at epoch 5
  (`results/phase16_n1/`). N1 measured that a zero-initialised neck grafted onto a converged model does not
  open; it did not test point-level support → query attention.
* **What it does.** The model of D-33 (`--neck sq_attn`) trained from scratch on E1's schedule (D-30: batch 1,
  24,000 updates, LR 1e-3 halved every 7,200, 13 validations), seed 0, with **α initialised to 0.1**
  (`--neck_alpha_init 0.1`) so that the neck's projections receive gradient from the first update. The
  "identity at initialisation" property of D-33 is dropped: a model trained from scratch has no base to
  preserve. 0.1 is not tuned (not measured); it is the smallest round value that makes the neck's term
  non-negligible while leaving F_s dominant at initialisation.
* **Reference.** E1's existing `last.pt` and `best.pt` (maintainer, 2026-09-24: E1 is not retrained), scored
  with N2 on identical episodes. The comparison is therefore between two independent training runs, whose
  `last.pt` differ by about 1 point from training noise alone (E1's validations swing 71–75, R2's r0/d29
  `last` equal within 0.1 while their `best` differed by 1.5); the paired CI covers only the episode draw.
* **Rules, fixed before the run** (`r2_distill_eval.py decide_n1` with E1 as `ctl` and N2 as `neck`):
  N1.1 go (N2 − E1 ≥ +1.0 on fixed100 `last`, CI above 0, positive on all three random600 draws), N1.2
  stop (< +0.5, or mean random600 < +0.5), N1.3 otherwise; N1.4 oracle gaps reported. A go is confirmed by
  a second seed before S0, because one run cannot separate +1.0 from training noise.
* **Affects.** `models/neck.py` and `models/cascadeproto.py` (`neck_alpha_init`), `train.py`
  (`--neck_alpha_init`), `experiments/run_n2.sh` (new), 05 §3.8m (NECK-10).
* **Outcome: N1.2 stop (2026-09-24, `results/phase16_n2/SUMMARY.md`).** N2 − E1 on fixed100 `last` −0.66 [−1.03, −0.29],
  random600 −0.52 / −1.08 / −0.94, `best` −0.81 [−1.21, −0.44]; the neck was used (α 0.1 → 0.060) and the oracle
  gap did not shrink (12.73 → 13.48). The same sign on every draw and checkpoint makes a hidden +1.0 implausible.
