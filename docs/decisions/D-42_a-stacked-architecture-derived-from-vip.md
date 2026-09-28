### D-42 — A stacked architecture derived from VIP-Seg's weaknesses, and P9: where each module can be placed, checked before any training · `PROPOSED`, beyond the paper

* **Problem.** After D-40 every gain comes from inference arithmetic on frozen features (54.84 → 58.55); only the
  encoder and the feature head learn, through PEM / PDM on 6 base classes. The measurements place the remaining error
  in what is learned: region-level instance shift (2/3 of the missed foreground, oracle gap 22 points, local seed
  recall 0.04–0.09 around missed points, P7a) and density (1/3 of the misses, "other" recall 0.16 against 0.77).
  The research note `docs/research/2026-09-26_beyond_prototypes.md` derives, block by block, a stacked architecture
  (§9): M1 a density-normalised neighbourhood aggregation in the encoder (R1, density invariance); M2 a training
  objective beyond CE on 6 classes (anti-collapse variance / covariance terms, more tasks; R2, instance invariance for
  unseen classes); M3 a metric-prototype head in place of PEM / PDM (nuisance projection, heads, components, query
  whitening; R3, R4); M4 the query graph (R5, kept); a neck and a text head as optional blocks. The maintainer
  approved the direction and asked that the **input assumptions at each insertion point be checked first**, because
  a module fails when its input is not what it assumes (the composition note's L3).
* **Input assumptions found by reading the code** (facts, not measurements):
  1. **Blocks are not encoded independently.** DyPowerConv normalises the kNN offsets and features by `torch.std` over
     the whole batch tensor [VIPSEG models/encoder.py:182-189, 283, 414], and the decoder subtracts a batch-global
     mean and divides by a batch-global std [VIPSEG models/encoder.py:582-584]. The two query blocks are encoded in
     one call and the support blocks in another (`encode_episode`), so each block's features depend on the other
     blocks of the call: order-free (D-37) but not block-independent, contrary to the invariant "each block
     independently" (AGENTS §3).
  2. **The encoder never sees metric coordinates.** It reads channels 6–8 (`pos = x[:, :, 6:]`), the block's
     coordinates divided per axis by the block's own maximum [VIPSEG dataloaders/loader.py:70-74]: objects are
     stretched differently in every block, and the maximum depends on which points the sampler drew.
  3. **Neighbourhoods are by count** (k = 16; FPS 2,048 → 1,024 → 512 → 256): their radius grows where points are
     sparse, and the batch-global std then mixes the scales of dense and sparse blocks.
  4. **Features are non-negative.** The feature head ends in BatchNorm + ReLU [VIPSEG models/vipseg.py:91-97]: every
     unit feature lies in the positive orthant, with a common component that a metric block must remove (centring)
     before whitening.
* **What P9 does** (`experiments/p9_placement_probe.py`, inference only, CR `last.pt`, S1; decisions on `valid`,
  fixed100 reported).
  * **A. Input assumptions.** A1 coupling: each query block encoded alone against the pair (as now); the largest
    relative feature change and U's mIoU for both. A2 density at the insertion points: P5's `sparse_view` thins the
    own class of a query block to background density; for the kept original points, cos(f_dense, f_sparse) at the
    encoder output (900-d, before the head) and at the feature (128-d), and U's recall of the class before and after.
    A3 extent: Spearman correlation between a block's per-axis extent and its class shift 1 − cos(s_c, o_c). A4 common
    component: ‖mean of unit features‖² per block.
  * **B. Preconditions of the modules.** B1 (M3) the bias oracle and the LDA oracle (block means and pooled
    within-class covariance from labels, centred, shrinkage λ ∈ {0.1, 0.3, 0.5}); the label-free transductive LDA
    (query total covariance, same λ grid). B2 (N) κ and ρ with the shift covariance Σ_η of base classes over 1,500
    training episodes, r ∈ {4, 8, 16}, and the projected rule. B3 (H) per-head shift and discriminant shares e_h, g_h
    for a PCA partition into H ∈ {4, 8} blocks; the coefficient of variation of g_h / e_h. B4 (C) foreground k-means
    components k ∈ {2, 3} (max over components) and the dual-condition (sparse-view) component. B5 (M2) the
    participation ratio of the point-feature covariance and the share of the oracle discriminant d* in the span of
    the base-class means. B6 (neck) retrieval purity: for hit and missed query points, the share of their 16 nearest
    support points (cosine) that belong to their class.
  * **Checks; a failure stops the run.** Model identity every episode; on fixed100 U equals D-39's `cr:base`; the
    paired encoding reproduces the model's features exactly.
* **Rules, fixed before the run.** Thresholds marked (c) are conventions, not measured.
  * P9.1 coupling: if encoding blocks alone changes U's valid mIoU by ≥ 0.5 in absolute value or any feature by
    ≥ 10⁻³ relative, the batch-global normalisation is recorded as a defect; M1 replaces it by a per-block one in any
    case, since the invariant requires it, and P9.1 reports its size.
  * P9.2 density placement: if cos(f_dense, f_sparse) at the encoder output is below its value at the feature output,
    density enters in the encoder (M1 belongs there); otherwise in the head.
  * P9.3 metric family (M3): LDA oracle ≥ cosine oracle + 1.0, or a label-free metric rule ≥ +0.5 over U on valid →
    M3 is trained; otherwise its metric blocks are dropped.
  * P9.4 nuisance block N: median κ > median ρ and the projected rule ≥ +0.5 over U on valid → N is kept.
  * P9.5 heads H: coefficient of variation of g_h / e_h ≥ 0.5 (c) → H is kept.
  * P9.6 components C: k-means or dual-condition components ≥ +0.5 over U on valid → C is kept.
  * P9.7 representation M2: share of d* in the base-mean span ≤ 0.5 (c) → the novel discriminant lies mostly outside
    what base supervision constrains, and M2 is trained.
  * P9.8 neck: retrieval purity of missed points ≥ 0.5 (c) → a support–query neck can retrieve the right class and is
    admissible on the clean base; otherwise it waits for M1 / M2.
  * P9.9 reported: every measurement per class and condition; the label-free rules composed with both + LP on
    fixed100 only if one holds on valid.
* **Why before training.** Each module costs a training run; P9 is one inference pass (about 1 h on an L4, not
  measured) and decides which modules enter the first trained architecture and where.
* **Affects.** `experiments/p9_placement_probe.py`, `experiments/run_p9.sh`, `tests/test_placement_probe.py`, 05 §3.8u.
* **Amendment (2026-09-26, after P8, before any P9 code).** P8 found the density dependence causal and located it
  upstream of every head (D-41 outcome), which answers P9.2 and makes the first block of the stack a training
  decision (D-43, M1). The module preconditions P9.3–P9.8 describe the input of M2 / M3 and are therefore measured
  on the features of the base D-43 adopts, not on CR's (a precondition measured on an input the module will not
  receive is the composition failure of the note's L3). P9.1 (block coupling) becomes a unit test of D-43. P9 keeps
  its rules and runs after D-43.
* **Outcome (2026-09-28, maintainer approval; closed without a further run).** Every module of the stack has been
  decided by its own measurement on the clean base. No module is pending, so the plan closes here.
  * **M1, density-invariant encoder (D-43).** D43.4 stop: fixed100 model −6.89 against CR. P9 then closed the encoder
    branch: every ψ is within ±0.02 (D-44).
  * **M2, anti-collapse objective (D-45).** D45.3: the features widen (participation ratio 42.45 / 95.25) and U
    falls by 5.20 / 11.34 points. The line stops.
  * **M3, metric head.**
    - P9.3 holds only through its oracle: LDA oracle 96.04 against cosine oracle 80.84, and the label-free LDA reaches
      35.44 against U's 55.96 (D-44).
    - The base-class covariance adds nothing distinguishable from 0 over the isotropic mean rule: λ 0.9 − λ 1.0 is
      +0.36 [−0.01, +0.72] on fixed100 (D-46, P10.4).
    - A trained M3 would learn that covariance on the base classes, so it is not run. The isotropic mean rule (+1.11
      over U on fixed100) is kept for the stack.
  * **M4, the query graph.** Kept at inference: LP +0.92, giving U + both + LP 58.55 on fixed100 (D-40).
  * **Optional blocks.**
    - The neck fails P9.8 (missed-point purity 0.109).
    - N fails P9.4 (κ 0.71 < ρ 0.86), H fails P9.5 (CV 0.23), and C fails P9.6 (53.86) (D-44).
    - The text head is "in between" (P3.3, +0.41; D-46).
  * **What carries over.** The next architecture is chosen in a new decision from D-46's error decomposition. That
    decomposition is P10.5: the own condition has no bias, the other condition carries a K-invariant bias, and the
    background rows add up to 8.4 points.
