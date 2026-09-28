### D-33 — N1: a point-level support → query attention neck before the prototypes · `PROPOSED`, beyond the paper

* **Problem, from P3 and P4.** On E1 the oracle rule (the query's own class directions) gains +13.71 when every
  prototype row is replaced, while replacing only the background row costs 26.07 and only the foreground rows
  9.05 (`results/phase16_p4/SUMMARY.md`): the error is a joint, query-conditioned shift of all prototypes, not a
  contaminated row. Training-free fixes (D-26…D-28, D-31, D-32) and a label-derived training target (D-29)
  leave it unchanged. VIP-Seg's head adapts the prototypes to the query only through max-pooled 64-token
  channel statistics [VIPSEG models/vipseg.py:235-311] and recovers 65 % of the support → oracle gap (P3);
  no point-level correspondence between query and support reaches the prototypes
  (`docs/research/2026-09-24_text_integration_independent.md` §9.3).
* **What it does.** Before the prototypes and the head, every support point attends to the episode's query
  points and moves its feature toward what it matches:
  `F_s' = F_s + α · softmax((LN(F_s) W_Q)(LN(F_q) W_K)ᵀ / √d) (LN(F_q) W_V) W_O`, d = 64, one head, all
  support blocks against all query points of the episode; α a learned scalar initialised to 0. The masked
  means of `F_s'` are the new `P^0` and the head receives `F_s'` for its support slots; `F_q` is unchanged.
  At α = 0 the model is E1 exactly. Spec 02 §15.
  * Why the support side: the oracle is the query's class mean; a support point pulled toward the query
    points it resembles moves its class's masked mean toward that class's query mean [inferred].
  * The prediction for one query depends on the other queries of the episode, as it already does through
    VIP-Seg's head (research note §4.4).
* **N1, the measurement.** Two arms warm-started from E1 `last.pt` (our own weights, trained from scratch on
  S1's base classes; guardrail 1 is not involved), identical seed and episodes, 7,200 updates at batch 1 and
  a constant learning rate 1.25e-4 (E1's last stage), validation every 5 epochs (3 validations):
  **ctl** without the neck, **neck** with it. The shared start removes most of the seed difference between
  the arms. One run per arm (maintainer). Test as R2/E1: fixed100 and random600 seeds 0, 1, 2, both arms and
  E1 on identical episodes, `last.pt` and `best.pt` (D-22 amended).
* **Rules, fixed before the run.**
  * N1.1 go: neck − ctl ≥ +1.0 on fixed100 `last` with a paired CI above 0 and positive on all three random600
    draws. Then S0. (At α = 0 the neck arm computes exactly ctl's function, so a gain implies a used neck;
    the final α is reported.)
  * N1.2 stop: neck − ctl < +0.5 on fixed100 `last`, or the mean over the random600 draws < +0.5.
  * N1.3 in between: otherwise; report.
  * N1.4 mechanism: the oracle rule's gain over the model on fixed100, a gap the neck exists to close, is
    reported for both arms; a go with an unchanged gap is flagged.
  * +1.0 and +0.5 are R2's thresholds (twice the single-run sd of 0.5).
* **Affects.** `models/neck.py` (new), `models/cascadeproto.py` (`neck`), `train.py` (`--neck`,
  `--init_checkpoint`), `experiments/r2_distill_eval.py` (`decide_n1`), `experiments/run_n1.sh` (new), 02 §15,
  05 §3.8m (NECK-…).
* **Outcome: N1.2 stop, the neck never opened (2026-09-24, `results/phase16_n1/SUMMARY.md`).** neck − ctl on fixed100
  −0.17 [−0.22, −0.13], −0.07 / −0.20 / −0.16 on random600; the gate ended at α = 0.0023. Both arms lost about
  2 points against E1 (restarting AdamW's moments: validation 73.04 → ~59 at epoch 5 → ~71). The run measured
  that a zero-initialised neck grafted onto a converged model at 1.25e-4 does not open; D-34 tests the neck.
