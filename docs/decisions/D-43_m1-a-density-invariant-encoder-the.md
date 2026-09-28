### D-43 — M1: a density-invariant encoder, the first block of D-42's stack · `PROPOSED`, beyond the paper

* **Problem.** P8 (D-41 outcome) showed by intervention that recall is set by the sampling density, not by the
  object: the same object re-sampled dense is found at 0.833 instead of 0.245 (φ 1.00 [0.96, 1.03]), and a class in
  its own block sampled uniformly falls from 0.837 to 0.463. Its direction moves from cos 0.55 to 0.86 of the support
  prototype when only the density changes, so the density enters the **features**, upstream of every prototype or head
  rule tried in D-26…D-40. Reading the encoder shows where [VIPSEG models/encoder.py]:
  1. neighbourhoods are the k = 16 nearest points (`FPS_kNN`, `knn_point`), so their metric extent grows where points
     are sparse and reaches other surfaces;
  2. DyHiConv feeds the neighbour offsets and their norm ‖p_j − p_i‖ into its kernel generator (lines 365–369), and
     DyPowerConv divides the offsets by one std of the whole batch (lines 182–189): the offset scale is a density
     measurement;
  3. LoConv, DyHiConv and the decoder standardise by a mean and std over the whole batch tensor (lines 182–189, 282–285,
     413–416, 582–584), which also couples the blocks encoded together (D-42 input assumption 1);
  4. the encoder reads the per-axis normalised coordinates (channels 6–8), so no radius is metric (D-42 assumption 2).
  The aggregations themselves are max-pools (lines 101–104, 457), which do not count points: once the neighbourhood is
  a fixed metric region, a max over it depends on the surface, not on how many samples fell on it.
* **What it does** (`models/density_encoder.py`, `encoder=density`; beyond the paper). The VIP-Seg encoder with the
  same modules and parameter shapes (the inherited classes are subclassed, never edited; their weights are the same
  tensors), and four changes, one per pathway above:
  1. **Metric ball neighbourhoods.** For each FPS centre, the neighbours are the points within radius r_s of it, the
     first K = 16 in the block's point order (the loader shuffles points, so this is a uniform subsample of the ball),
     padded with the first one when fewer (PointNet++'s ball query); r_s = 0.1, 0.2, 0.4 m for the three stages.
     r_1: D-35's kNN-16 radii at 2,048 points are 0.068–0.084 m (own) and 0.10–0.13 m (other, background), so 0.1 m
     holds about 16 points at background density; doubling per stage follows PointNet++'s segmentation network (not
     re-checked here). Implemented in torch (distances within the block), not with the CUDA op, so it is testable on
     the CPU.
  2. **Offsets in metric units.** p_j − p_i is divided by r_s (so ‖·‖ ≤ 1), not by a batch std.
  3. **Per-block standardisation.** Every batch-global mean / std becomes the block's own (over all its other axes).
  4. **Metric coordinates.** The encoder reads channels 0–2 (metres from the block minimum [VIPSEG
     dataloaders/loader.py:65-66]) for the embedding, the grouping and the decoder's interpolation.
* **Arm.** M1 = CR's configuration (D-37: `stage_type=vip_clean`, four stages, no LMA, L2 point prototypes, random
  query order, E1's schedule: batch 1, 24,000 updates, LR halved every 7,200, 13 validations, seed 0) with
  `--encoder density`. CR is the reference; no new run for it. One training run (about 2.3 GPU-h, as E1; not measured).
* **Test** (`experiments/d43_eval.py`), CR and M1 `last.pt` (`best.pt` reported), each with its own features:
  * the adopted inference stack of each (model head; U; U + both; U + both + LP with P7's frozen arm) on fixed100,
    random600 seeds 0–2 and the leak-free draw (seed 4);
  * P8's arm B on M1 (seed 3) and P8's condition oracles and alignment on M1 (valid).
* **Checks; a failure stops the run.** Unit tests: the ball grouping against brute force; per-block standardisation;
  the density encoder encodes a block identically alone and inside a batch (GPU test), the VIP-Seg encoder does not
  (the defect is reproduced); the new encoder has the same parameter names and shapes as VIP-Seg's. Evaluation: CR's
  numbers equal D-40's (U + both + LP 58.55, U + both 57.63 on fixed100) and P8's intervention numbers (R_V0 0.245).
* **Rules, fixed before the run** (paired bootstrap over episodes; "holds at g" = fixed100 gain ≥ g with CI above 0
  and > 0 on all three random600 draws; the stack compared is U + both + LP of each checkpoint):
  * D43.1 mechanism (L5): on the seed-3 intervention under U, M1's other-condition deficit R_own − R_V0 is at most half
    of CR's (0.588) **and** its own-class uniform drop R_a(V0) − R_a(V2) at most half of CR's (0.374). Otherwise M1 did
    not remove the density dependence, and any score change is recorded as unexplained.
  * D43.2 honest protocol: M1 − CR on the leak-free draw ≥ +1.0 with a paired CI above 0.
  * D43.3 standard protocol: M1 − CR holds at +1.0 → M1 is the base of every later arm on both protocols. If D43.1 and
    D43.2 hold and the fixed100 gain is negative, M1 is the base for the leak-free protocol, the fixed100 difference
    is reported as the part of the standard score that rests on the density leak, and the protocol of the paper's
    main table goes to the maintainer.
  * D43.4 stop: D43.1 fails → the density pathway is elsewhere (FPS, decoder, head); P9's placement measurements on M1
    locate it before another encoder change.
  * D43.5 reported: every rule of the stack on both checkpoints and draws; P8's g_own, g_other and alignment on M1;
    the learned α-free comparison of the heads (model vs U) on M1.
* **Why one arm.** P8 fixed the target and its cause; the four changes are one mechanism (a density-free
  neighbourhood) and are not separable in a way that a single run could attribute. If D43.1 holds, an attribution
  ablation is a later decision; if it fails, the placement probe (P9) says where density still enters.
* **Affects.** `models/density_encoder.py` (new), `models/vipseg_backbone.py` (encoder choice), `models/cascadeproto.py`
  (`encoder` field), `train.py` (`--encoder`, run tag `_dens`), `experiments/p8_condition_probe.py` (reference checks
  optional), `experiments/d43_eval.py`, `experiments/run_d43.sh`, `tests/test_density_encoder.py`, 05 §3.8t.
* **Outcome (2026-09-26, `results/phase16_d43/SUMMARY.md`).** **D43.1 fails**: other-condition deficit 0.588 → 0.432
  (limit 0.294), own-class drop under uniform sampling 0.374 → 0.431 (limit 0.187); φ 1.14. **D43.2 fails**: leak-free
  U + both + LP −0.30 [−1.41, +0.85]. **D43.4 stop**: M1 is not adopted, CR stays the base; fixed100 −9.31
  [−10.82, −7.85], random600 −7.50, −10.28, −12.52. Descriptive, not pre-registered: M1's model head and U are better
  on the leak-free draw (+2.07 [+1.21, +2.93], +1.55 [+0.79, +2.30]) and worse on fixed100 (−6.89, −6.33); "both" and
  LP, selected on CR's features, remove that gain. The features move (cos of the sparse vs dense object 0.72 / 0.83,
  CR 0.55 / 0.86) but the decision still follows density. Input assumption found wrong: a max over a metric ball does
  not count points only when the ball holds at least K distinct points; r₁ = 0.1 m holds fewer on sparse objects, so
  padding with duplicates keeps a count dependence (candidate, not measured). P9 locates the remaining pathway before
  any other encoder change.
