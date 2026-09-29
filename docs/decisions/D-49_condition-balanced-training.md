### D-49 — Condition-balanced training: training queries in which the target class also appears sparse · `PROPOSED`, beyond the paper, maintainer approval 2026-09-29

* **Problem.** Three independent measurements on the clean base CR point to one cause: the representation learns
  "dense ⇒ target".
  1. **P8 (D-41).** The same object sampled dense is found with recall 0.833 and sparse with 0.245 (φ 1.00). Under
     uniform sampling the own class falls from 0.837 to 0.463.
  2. **P11.1b (D-48).** A base-class learner scores dense novel foreground like base background (g 0.57 against
     0.69; 0.13 when the same classes are sparse). Every training block is sampled dense for its target, so dense
     means "target" during training.
  3. **P11.6 (D-48 amendment 5).** A linear mixture explains sparse points' features (median R² 0.99), but their
     own-class coefficient is −0.018 at the median. The sparse class leaves no trace in CR's features, so no head
     can recover it.

  During training the target of a query block is always dense. The sampler over-samples it (share r(2 − r) for a
  raw share r). The loss therefore never asks the encoder to keep the class of a minority point.
* **Registered earlier.** D-41's P8.4 fixed condition-balanced training ("training queries in which the target
  class also appears at background density") as an admissible arm, second after instance alignment. Instance
  alignment ran as M5 (D-46) and stopped. D-43 changed the encoder instead of the training data and lost on the
  standard draw (−6.89). Condition balance has never run.
* **What it does** (`pipeline/episodes.py::ConditionBalance`; `train.py --condition_balance q`).
  - For each training episode, each query block is thinned, independently with probability q, in its own class
    (block b sampled for local class b + 1). This happens after the loader's augmentation and before D-37's query
    permutation.
  - Each own-class point is kept with probability (1 − r)/(2 − r), where f is the block's own-class share and
    r = 1 − √(1 − f). The own class then holds the raw share r of the kept points (P5's `sparse_view`, D-35). At
    least one own-class point survives.
  - The block is refilled to 2,048 points with copies of uniformly drawn kept points plus the training jitter
    (σ 0.01, clip 0.05). Their labels are the copied points' labels, so every class label follows its point.
  - The xyz channels keep the loader's augmented coordinates. XYZ is recomputed as the loader does:
    (xyz − min) / max per axis [VIPSEG dataloaders/loader.py:67-74].
  - Supports are untouched and stay dense, so a thinned block reproduces the test's other condition: a sparse query
    class against a dense support.
  - The episode seeding is unchanged: the thinning draws from a private generator seeded by (seed, 4, i), so episode
    i is the same with or without the wrapper, except for the thinned blocks.
* **Arm.**
  - CB: CR's configuration and schedule (`vip_clean`, T 4, no LMA, L2 point prototypes, random query order,
    batch 1, 24,000 updates, LR halved every 7,200, 13 validations), seed 0, with q = 0.5 (a convention: half the
    blocks keep the benchmark's density).
  - One seed; a second seed is added if a rule passes by less than +2.0 (AGENTS §6).
* **Test.**
  - `experiments/d49_eval.py`, the same reader for every head, on CR and CB, `last.pt` and `best.pt`.
  - Rows: model, U, U + both, and model + LP and U + both + LP, with LP re-selected on each checkpoint's valid over
    P7's 24 arms (D-48 amendment 1, change 6).
  - Draws: fixed100, random600 seeds 0–2, leak-free.
  - Own / other recall.
  - On valid_raw, P11.6's oracle abundance of other-condition points, in each checkpoint's own centred space (μ
    from 210 base-class training episodes).
* **Rules, fixed before the run.** Comparisons use U, the rule without parameters on each checkpoint's features.
  "Holds at g" is D-48's definition (fixed100 CI above 0, and > 0 on the three random600 draws), `last.pt`.
  - **D49.1 mechanism.** Both of the following, conventions (c):
    - the median own-class abundance of other-condition points is ≥ 0.10 (CR: −0.018);
    - U's other-condition recall on fixed100 is at least CR's + 0.05.

    An arm that fails has its score change reported as unexplained.
  - **D49.2 leak-free.** CB − CR (U) on the leak-free draw ≥ +1.0, with the paired CI above 0.
  - **D49.3 standard.** CB − CR (U) holds at +1.0. CB becomes the base on both protocols.
  - **D49.4.** D49.1 and D49.2 hold and D49.3 fails with a fixed100 change below 0. CB is the base for the leak-free
    protocol, and the protocol of the main table goes to the maintainer (as D43.3).
  - **D49.5.** D49.1 fails. Condition balance does not restore minority points at q = 0.5, and the line stops.
  - **Reported:** every row, `best.pt`, the stacks, and own / other recall.
* **Cost** (not measured). One training next to D-48's arm A (seeds 0 and 1) on the 3090, about 4 h for the three,
  then about 1.5 h of tests.
* **Affects.** `pipeline/episodes.py` (`sparse_query_view`, `ConditionBalance`, `with_condition_balance`),
  `train.py` (`--condition_balance`, run tag `_cb<q>`), `experiments/d49_eval.py` (new), `experiments/run_d49.sh`
  (new), `tests/test_condition_balance.py` (new), 05, new section.
