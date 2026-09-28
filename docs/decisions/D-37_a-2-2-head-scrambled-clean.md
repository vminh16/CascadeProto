### D-37 — A 2 × 2: head (scrambled, clean) × training query order (fixed, random) · `PROPOSED`, beyond the paper

* **Problem.** C3 (D-35 outcome) shows reliance on the query position at test time; it does not show where the
  reliance comes from or what the head is worth without it. The shortcut needs two things at once: an architecture
  that can represent the position (D-36's necessary condition) and training data in which the position predicts the
  class (the loader's fixed order). Only an experiment that varies both separates them.
* **Arms** (each trained once on E1's schedule, D-30: batch 1, 24,000 updates, LR 1e-3 halved every 7,200, 13
  validations, seed 0, S1):

  | | fixed order (the loader's) | random order |
  | :--- | :--- | :--- |
  | **scrambled head** (VIP-Seg's PEM/PDM) | **VF** = E1, existing checkpoints | **VR**, trained |
  | **clean head** (D-36's `clean` form, same parameters) | **CF** = CR, by the lemma below | **CR**, trained |

  *Random order:* every training episode's query blocks are permuted uniformly at random before the forward pass,
  labels moving with their blocks (`train.py --query_order random`); supports, labels and the test protocol are
  unchanged. The cached test and valid episodes keep the loader's order.
* **Amendments (2026-09-24, before any code or run).**
  * **CF is not trained.** Lemma: with the `clean` form every operation of the model is equivariant or invariant
    in the query index during training as well as at evaluation. The encoder and feature head process each block
    alone except BatchNorm's batch statistics and the batch-wide `torch.std` [VIPSEG models/encoder.py:182-189],
    both invariant to the order of the batch; there is no stochastic layer (`drop_path = 0.`
    [VIPSEG models/encoder.py:514], `DropPath` only built for a positive rate [VIPSEG models/mamba_block.py:60]);
    the `clean` PEM/PDM, ADRM and the logits are per query; the loss (mean cross-entropy over all query points,
    no LMA, D-29 off) is invariant. So a permuted episode gives the same loss and the same parameter gradient as
    the stored one, and CF and CR follow the same trajectory up to floating-point rounding, which CUDA already
    makes non-deterministic (AGENTS §6). Training CF would measure that rounding, not the data order. Verified,
    not assumed: D37-T5 (float64, CPU, exact) and D37-T6 (the real model on the GPU) compare the loss and every
    gradient under a query permutation; the scrambled form must fail the same test. The saving is one run.
    Measured on the L4 (2026-09-25, one episode, default initialisation, TF32 off): with `vip_clean` the query swap
    leaves the loss bit-identical and changes the gradients by at most 1.8e-3 of a parameter's largest entry, below
    the 5.4e-3 caused by moving every query coordinate by one float32 ULP; with `vip` the loss moves by 3.0e-3 and
    the gradients by 0.66. With TF32 (cuDNN's default, used in training) 3.6e-2 against a floor of 3.9e-2. The
    differences sit in the first encoder layers, whose gradients cancel over the batch through BatchNorm.
  * **Episode identity.** The permutation comes from its own generator, `np.random.default_rng([seed, 3, i])` for
    training episode i (`pipeline/episodes.py`, `QueryOrder`, after `SeededEpisodes`); it never draws from the
    global RNGs, so VR and CR see E1's episodes (classes, blocks, points, augmentation) and differ from E1 only in
    the query positions and, for CR, in the cross-term. Run directory suffix `_qrandom`.
  * **Evaluation check.** E1 `last` and `best` are re-scored in the same evaluation run; they must reproduce E1's
    summary (73.20 / 75.05 on fixed100) within 0.05, or the evaluation differs from E1's and the run stops.
  * **Queue.** GPU tests and a smoke run → C4 (D-36) → train VR → train CR → evaluation → rules.
* **Test.** Every arm on fixed100 in stored and swapped order (C3), the three random600 draws (stored), and P5's
  leak-free draw (seed 4, arm D); `last.pt` and `best.pt` (D-22 amended). Oracle columns come with R2's scoring.
* **What each contrast measures.**
  * VF vs VR (data, scrambled head): how much of the standard-protocol score the position shortcut is worth.
  * CF vs CR (data, clean head): nil by the lemma; D37.2 checks it at test time.
  * VR vs CR (architecture, no shortcut available): the honest architectural comparison, the one a paper can claim.
  * VF − VR − (CF − CR) = VF − VR (interaction): the shortcut itself, which needs both the architecture and the data.
  * stored vs leak-free for VR and CR: what the density cue is worth once the position is gone.
* **Rules, fixed before the run.**
  * D37.1 origin: VR swap gap |stored − swapped| ≤ 1.0 and relabel shift (D-36) ≤ 0.05 → the fixed order is the
    origin of the shortcut. If VR still relabels by position (shift > 0.5), the carrier is not the training order; stop
    and re-examine.
  * D37.2 equivariance check: CR swap gap ≤ 0.1 and |relabel shift| ≤ 0.01 (by construction); otherwise the clean
    head is wrong and no other rule is read.
  * D37.3 architecture: CR − VR on fixed100 stored ≥ +1.0 with a paired CI above 0 and > 0 on all three random600
    draws → the clean head is the base of every later arm; ≤ −1.0 with a paired CI below 0 and < 0 on all three
    random600 draws → the scrambled head, trained with random order. Otherwise (amended before the run) the clean
    head: at equal accuracy it is the one that cannot re-learn the shortcut under any data order, including the
    standard loader's, and its prediction for one query does not depend on the other queries of the episode (C2).
  * D37.4 reported: the shortcut's worth VF − VR, the leak-free levels, and the oracle gap of the chosen base — the
    first oracle gap in this repository measured on a head that cannot name classes by position, which is the
    quantity the next decision (prototype, head or neck) is built on.
* **Cost.** Two training runs of about 1.7 GPU-h each (E1 took 1 h 42 min on the L4) plus about 1.5 h of
  evaluation (not measured).
* **Affects.** `models/vip_stage.py` (`clean` form), `models/cascadeproto.py` (`stage_type=vip_clean`),
  `pipeline/episodes.py` (`QueryOrder`), `train.py` (`--query_order`), `experiments/d37_eval.py`,
  `experiments/c3_query_order.py` (`--out`), `experiments/run_d37.sh`, 05 §3.8o.

* **Outcome (2026-09-25, `results/phase16_d37/SUMMARY.md`).** C4: checks C4.0a/b pass exactly; with `clean` both
  checkpoints are order-free (E1 37.91 / 37.91, VIP-Seg 35.45 / 35.45, shift 0.000) → C4.1 holds, the reshape is the
  sole carrier. D-37 on `last`: D37.2 holds (CR 54.84 / 54.84); D37.1 origin (VR 53.26 / 52.75, shift +0.004);
  D37.3 clean head (CR − VR +1.58 [+0.50, +2.66], random600 +1.39 / +0.65 / +1.35); D37.4 shortcut worth VF − VR
  +19.94 [+18.40, +21.52], leak-free E1 17.89 / VR 24.04 / CR 28.10, the base's oracle gap +28.54 (unit). On `best`:
  D37.3 tie (+0.73 [−0.32, +1.85]) → clean head by the tie rule. Without the shortcut the head is level with plain
  prototype matching (CR 54.84 against its support rule 55.02). **The clean head (`stage_type=vip_clean`, random
  query order) is the base of every later arm.**
