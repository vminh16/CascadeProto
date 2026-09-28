### D-44 — P9 after D-43: where density still enters M1, and which module the one remaining run can use · `PROPOSED`, beyond the paper

* **Problem.** D-43 closed four density pathways of the encoder and the decision still follows density (φ 1.14; the
  sparse object is found at 0.28, the same object dense at 0.77). The features moved (cos of the sparse against the
  dense version 0.72 / 0.83, CR 0.55 / 0.86), so the remaining pathway is one the direction cosine of the final
  feature does not show. D-43's reasoning held that a max over a fixed metric ball "does not count points"; that is
  true only when the ball holds at least K = 16 distinct points. A ball with m < K distinct points is padded with
  duplicates [`models/density_ops.py::ball_group`], and a max over m samples of a surface is smaller in expectation
  than a max over K, so the number of samples (the density) still reaches the neighbourhood feature. r₁ = 0.1 m was
  chosen to hold about 16 points at background density (D-43), so sparse objects are exactly where m < K. This is a
  hypothesis; nothing has measured m. The budget left (about $6, not verified against the bill) pays for this probe
  and one training run (D-43 used about 4 VM-hours), so the probe must say which run that is.
* **What P9 does** (`experiments/p9_placement_probe.py`; inference only; S1; the events of P8's arm B: seed 3, the
  same 1,045 query blocks, V0 as sampled, V1 the same scan sampled for the other class c, V2 sampled uniformly).
  * **Step 0, geometry (CPU, no model).** For each event and version, M1's grouping is replayed on the block: FPS
    2,048 → 1,024 → 512 → 256 (start at index 0, as `furthest_point_sample`), a ball of 0.1 / 0.2 / 0.4 m around each
    centre, the first 16 points inside in point order (`ball_group`). m = the number of distinct coordinates among
    those 16. Reported per stage for the centres labelled c in V0 and V1 and labelled a (the block's own class) in
    V0 and V2: the mean of m, the share with m < 16 and with m ≤ 8.
  * **Part A, the pathway (GPU, M1 `last.pt`).** Each block is encoded alone (M1 encodes blocks independently,
    DE-10), with M1's support features of the episode. (A1) At the raw points present in both V0 and V1, the cosine
    between their features in the two versions, for four slices of the 900-d encoder output (the decoder concatenates
    [embedding 60 | stage 1 120 | stage 2 240 | stage 3 480], [VIPSEG models/encoder.py:595-603] with D-43's
    decoder) and for the 128-d feature; class-c points and background points separately. Reported. (A2) The count
    intervention: M1's grouping with a cap, where a capped centre keeps its first neighbours up to m distinct points
    and pads the rest with the first, as `ball_group` pads. *Other direction*: in V1, each centre labelled c gets an m
    drawn from the m of the c-centres of V0 in the same event and stage; the recall of c under U is R_cap. *Own
    direction*: in V0, each centre labelled a gets an m drawn from the a-centres of V2; the recall of a is R_cap. Arms:
    the cap at stage 1, 2, 3 alone and at all three. The effect is the share of the density gap the cap reproduces,
    ψ = (R_ref − R_cap) / (R_ref − R_low), with (R_ref, R_low) = (R_V1, R_V0) for the other direction and
    (R_a(V0), R_a(V2)) for the own one, pooled over events, with a bootstrap CI over events.
  * **Part B, module preconditions (GPU, CR `last.pt`, the base after D-43).** D-42's P9.3–P9.8 on the valid draw
    (1,500 episodes), unchanged: (B1) cosine oracle (the query's own unit directions in every present row), LDA oracle
    (the query's class means of unit features and its pooled within-class covariance, shrunk λ ∈ {0.1, 0.3, 0.5}, the
    best λ as the bound), label-free transductive LDA (the query's total covariance, same λ, selected on valid);
    (B2) Σ_η from 1,500 training episodes of the fold's base classes (the loader in training mode without
    augmentation, so that the blocks are drawn as at test; one o_{c,β} per block and present class), B its top r
    eigenvectors, r ∈ {4, 8, 16}; κ, ρ per (query block, present class) and U with the features and support
    directions projected by P⊥ = I − BBᵀ and re-normalised; (B3) e_h, g_h for a partition of the eigenbasis of the
    valid query features' covariance into H ∈ {4, 8} equal blocks, and for a random orthogonal basis as a control;
    (B4) U with each class's foreground row the max over k ∈ {2, 3} spherical k-means components of its support
    points (P6's `spherical_kmeans`); (B5) the participation ratio of the valid query features' covariance, and the
    share of d* in the span of the base-class means ō_c (from B2) and the background mean; (B6) for the hit and the
    missed foreground points of U, the share of their 16 nearest support points (cosine, over both ways' blocks)
    carrying their label. D-42's dual-condition component of B4 is dropped: its question (a support-side density
    component) is answered by part A first.
* **Checks; a failure stops the run.** Step 0: the torch `ball_group` and the replay agree on every centre of a
  sample of blocks. Part A: a block encoded alone gives the model's features for that block within 1e-3 relative
  (DE-10's floor is 1.3e-4); the uncapped arm (cap m = 16 on every centre) equals the plain grouping exactly; the
  references R_V0, R_V1, R_own, R_a(V0), R_a(V2) are within 0.005 of D-43's `intervene_m1.json` (per-block encoding
  against the pair can flip a point at the rounding floor). Part B: model identity every episode; U on valid equals
  P8's valid split of CR (U 55.96) within 0.01 points.
* **Rules, fixed before the run.** Thresholds marked (c) are conventions, not measured.
  * D44.0 step 0: if the mean m of the c-centres of V0 is at least 15.5 at every stage, the ball count cannot carry
    the density difference and part A's cap arms are not run (A1 still is).
  * D44.1 pathway (part A, cap at all three stages, D-35's φ bands): ψ ≥ 0.5 in the other **or** the own direction →
    **the ball count carries the density dependence**; a count-free neighbourhood (M1b) is the candidate for the run,
    at the stages whose single-stage ψ ≥ 0.5 (all three if none alone reaches it). ψ ≤ 0.2 in both directions → the
    count is not the pathway; the encoder branch is closed for this budget. Otherwise undecided; the encoder branch
    is not trained on this budget.
  * D44.2 modules (part B, D-42's rules on CR): P9.3 metric (LDA oracle ≥ cosine oracle + 1.0, or label-free LDA ≥ U
    + 0.5); P9.4 nuisance (median κ > median ρ at the r whose projected U gains most, and that gain ≥ +0.5); P9.5 heads
    (coefficient of variation over heads of ḡ_h / ē_h ≥ 0.5 (c) on the PCA partition); P9.6 components (a k ≥ U +
    0.5); P9.7 representation (median base-span share of d* ≤ 0.5 (c)); P9.8 neck (retrieval purity of missed points
    ≥ 0.5 (c)). A block that fails is not a candidate.
  * D44.3 the choice: P9 decides admissibility only. The one training run is chosen in a new decision from D44.1 and
    D44.2, with the measured size of each candidate's bound, before any code for it.
  * D44.4 reported: every measurement per stage, class and direction; for label-free rules that pass on valid, their
    fixed100 score against U, the rule alone (P9.9's composition with both + LP needs the rules re-selected on the
    rule's own logits, D-43 outcome, and is left to the decision that adopts it).
* **Cost.** Parts A and B in one VM session, about 1–1.5 h on an L4 (not measured; part A is about 11 single-block
  encodings per event, part B two passes of 1,500 episodes). Step 0 runs on the VM's CPU while part B uses the GPU.
* **Amendment (2026-09-27, before any GPU run).** Step 0 was first run on the local machine and drew other events:
  1,053 events in 888 episodes and no support without background, against P8's 1,045, 890 and 1. The seeded draw
  chooses scans from the loader's class-to-scan lists (`class2scans_100.pkl`, written once per machine
  [VIPSEG dataloaders/s3dis.py:36-45]), whose order differs between the two machines, so a draw is reproducible only
  on the machine that made it. Step 0 therefore runs on the VM, where P8 and D-43 ran; part A checks its event count
  against D-43's (1,045). The local run is kept as a preview (`results/phase16_p9/geometry_local.json`): stage 1,
  mean m of the c-centres 11.81 as sampled against 15.13 dense (69 % against 32 % of the centres below 16), stages 2
  and 3 at 15.4–16.0.
* **Amendment 2 (2026-09-27, maintainer request, before any GPU run): a block-statistics arm in part A.** The ball
  count is not the only way the sampling reaches M1's features. When a class is sampled dense it also takes a larger
  share of the block, and M1 standardises by the **block's** mean and std at twelve places (LoConv, DyHiConv and the
  DyPowerConv feature scale at each of three stages, the decoder at three levels [`models/density_encoder.py`], D-43
  change 3): every feature of the object then depends on what the rest of the block holds. The cap arm keeps these
  statistics as they are, so ψ_cap is the count's share only; if it is small the probe would not say where the rest
  enters. Step 0 hints at this: the own class loses 1.8 distinct points per ball under uniform sampling and 0.43 of
  its recall, the other class 3.3 points and 0.49 (local preview, not a rule input). Two arms are added, both
  directions, same events:
  * **stats**: the reference version encoded with the statistics (mean and std at each of the twelve places, in call
    order) recorded while encoding the low version of the same event (other: V1 with V0's; own: V0 with V2's);
  * **caps + stats**: the cap at all three stages and the swapped statistics together.
  ψ_stats and ψ_caps+stats as ψ_cap. The swap is exact bookkeeping: a recording pass returns the plain features bit
  for bit, and injecting a block's own statistics reproduces them (checked on the first five events; a failure
  stops the run). The encoder gains a statistics tap that is off by default (`stats_tap = None` runs the code of
  D-43 unchanged); the Mamba blocks mix the whole block too and are not tested by either arm.
  * **D44.1b** (D-35's bands): ψ_stats ≥ 0.5 in a direction → the block statistics carry the density dependence;
    statistics that do not depend on the block's composition become a candidate (M1c). ψ_caps+stats ≤ 0.2 in both
    directions → neither the count nor the statistics carry it; the remaining paths (Mamba's mixing, the geometry of
    the sparse surface itself) are not reachable on this budget and the encoder branch is closed. D44.1 is unchanged;
    M1b and M1c can both be candidates; D44.3 chooses.
* **Amendment 3 (2026-09-27, maintainer request, before any GPU run): new machine, local data, a composite arm.**
  * *Machine and data.* The GCP project is closed. P9 runs on a rented RTX 3090 (vast.ai) with the maintainer's local
    copy of the blocks: the same points as the old VM's, in another row order, and another `class2scans_100.pkl`
    (checked: every `.npy`, checkpoint and stored episode folder transferred with identical md5; the stored S1 valid
    and fixed100 folders are byte-identical to the old VM's). Checked on the 3090 before any P9 code ran: CR scores
    54.8353 on fixed100 (D-37's 54.84 within 0.01) and U 55.7431 (D-39's 55.74). The seeded draws (P8's events,
    random600, leak-free) are therefore other samples of the same procedure: step 0 draws 1,053 events in 888
    episodes (the local preview). The checks that compared the event count and part A's reference recalls with
    D-43's files are replaced: the references R_V0, R_V1, R_a(V0), R_a(V2) are measured on M1 in the same run and
    reported next to D-43's; step 0 and part A must draw the same events (their counts must agree). The rules use
    only quantities of this run, so they are unchanged.
  * *Composite arm in part B.* A block can fail alone and help after another (the nuisance projection may be what
    makes a query metric useful). Label-free, at inference: features and support projected by P⊥ (r ∈ {0, 4, 8, 16},
    r = 0 no projection), the query-whitened LDA of part B (λ ∈ {0.1, 0.3, 0.5}), each way's foreground as k ∈ {1, 2,
    3} components (spherical k-means of its projected support units; a component's mean is the mean of its members;
    a class scores the max over its components). With r = 0 and k = 1 it is part B's label-free LDA. **P9.3c**: the
    best composite on valid ≥ U + 0.5 → the composite is admissible; if it is also ≥ the best single label-free arm
    (label-free LDA, projected U, components) + 0.5, the blocks interact and are reported as such.
  * *Scheduling.* Step 0 (CPU) and part B run together; part A starts when step 0 ends, next to part B.
* **Outcome (2026-09-27, `results/phase16_p9/SUMMARY.md`).** RTX 3090, local data, 1,053 events. D44.0 sparse balls
  (stage 1 mean m 11.81 against 15.13 dense). **D44.1 and D44.1b: encoder branch closed**: every ψ within ±0.02
  (cap all −0.003 / −0.008, block statistics −0.006 / +0.001, both +0.012 / +0.006, other / own). The same points'
  features stay close through the encoder (stage-3 cosine 0.90) and separate in the 128-d feature (0.74 class, 0.92
  background). **D44.2: only P9.3 is admissible, through the oracle clause** (LDA oracle 96.04 against the cosine
  oracle 80.84; label-free LDA 35.44 against U 55.96); P9.4 (κ 0.71 < ρ 0.86 at r 8), P9.5 (CV 0.23), P9.6 (53.86),
  P9.7 (base-span share 0.857), P9.3c (36.55) and P9.8 (missed purity 0.109) fail. Participation ratio 5.56 of 128:
  the features are collapsed to about C − 1 directions for C = 7 training classes. P9.7's rule does not fit this
  regime (a large share because every direction is in the base span); its verdict stands as registered and any
  reading of the collapse as a training target needs its own decision.
* **Affects.** `experiments/p9_placement_probe.py`, `experiments/run_p9.sh`, `tests/test_placement_probe.py`,
  05 §3.8u.
