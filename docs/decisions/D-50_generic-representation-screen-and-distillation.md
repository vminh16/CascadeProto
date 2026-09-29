### D-50 — Generic representation: a frozen-feature screen (P12), then distillation into the light encoder (M6) · `PROPOSED`, beyond the paper, maintainer approval 2026-09-29

* **Problem.** Every clean-base experiment so far changed the head, the inference rule, the loss or the sampling, and
  never the *information source* of the encoder. The encoder is always trained from scratch on the 6 base classes
  plus background of one fold (evidence ledger, `docs/research/2026-09-29_first_principles_layer_audit_vi.md` §3.3).
  Four measurements put the binding constraint there:
  1. **The head is saturated.** The clean head equals plain prototype matching (CR 54.84, U 55.74, support rule
     55.02; D-37, D-38), and about 17 head and inference variants top out at +2.8 (D-39, D-40, D-48).
  2. **The features collapse onto the base classes.** The participation ratio is 5.56 of 128 ≈ C − 1. Every S1
     checkpoint reads 3.7–7.3, including VIP-Seg's release (6.39) (D-45).
  3. **The error does not shrink with shots.** The K-shot curve fits a K-invariant instance shift a = 0.156 against
     a 1-shot variance c = 0.047. U = 56.32 / 60.22 / 61.63 / 62.74 at K = 1 / 2 / 3 / 5 extrapolates to
     U∞ ≈ 64.4 (D-46; `results/phase16_d46/SUMMARY.md`).
  4. **Geometry without information does not help.** VICReg raised the participation ratio to 42–95 and lowered U by
     5.2–11.3 (D-45). By the data-processing inequality, no function of features trained on the 6 base labels adds
     information about the novel classes.

  Outside this repository, frozen generic 3D features carry class structure that base-class training does not
  (`docs/research/2026-09-29_field_ceiling_and_paradigm_vi.md` §3):
  * **LAM3C** is pretrained only on 49k point clouds reconstructed from web room-tour videos, with no real scans and
    no S3DIS. Its S3DIS Area 5 linear probe is 65.7 (PTv3-Base) / 69.5 (PTv3-Large) [LAM3C README, read 2026-09-29].
  * **D-DITR** distils DINOv2 into PTv3 and needs no images at test time. It reaches S3DIS 75.0 against PTv3's 73.6
    [DITR README, read 2026-09-29].
  * Sonata and Concerto report ScanNet linear probes of 72.5 and 77.3 [search snippets, not verified].

  **Nobody has measured these features on this repository's few-shot episodes.** Three unknowns remain:
  * whether they transfer from support to query better than CR's features;
  * whether the 2,048-point blocks put them out of distribution;
  * whether any gain comes from having seen S3DIS scenes unlabelled.
* **Guardrail change (maintainer approval 2026-09-29).** AGENTS guardrail 1 and 01 §2.1 forbid pre-trained
  point-cloud weights. This decision admits them in exactly two roles:
  - a frozen feature extractor in the P12 screen (`experiments/` only);
  - a frozen teacher during M6 training.

  **The evaluated model never loads them.** M6's student is the VIP-Seg encoder and feature head initialised from
  scratch, with the parameter count of CR; the distillation projector is dropped after training. Every number that
  depends on a pretrained point-cloud model is reported in a separate table labelled "label-free pretrained teacher".
  The label says which corpus the teacher saw:
  - LAM3C: no S3DIS;
  - Sonata, Concerto: S3DIS Areas 1–6 unlabelled, and 2D for Concerto.

  Such numbers never enter the main table and never compete with the paper's or VIP-Seg's rows. All three models'
  weights are CC-BY-NC 4.0: research use only.

#### Part P12 — frozen-feature screen (inference only, no training)

* **Feature sources** (each yields per-point features for every support and query block of an episode):

  | Arm | Features | Notes |
  | :--- | :--- | :--- |
  | F0 | CR `last.pt`, 128-d head output | reference: U 55.74 fixed100, 33.11 leak-free, 55.96 valid (`results/phase16_d46/SUMMARY.md`) |
  | F1 | CR `last.pt`, 900-d encoder output after its L2 norm, before `bn`/`fc` (`models/vipseg_backbone.py:76-77`) | in-guardrail; tests whether the head's projection causes the collapse |
  | F2 | LAM3C PTv3-Base | clean control: no S3DIS in pretraining |
  | F2L | LAM3C PTv3-Large | only if F2 runs within the memory of the GPU at hand |
  | F3 | Sonata | saw S3DIS unlabelled |
  | F4 | Concerto (the smallest released size) | saw S3DIS unlabelled, and 2D |

* **Input to F2–F4.**
  - Take the episode's own 2,048 points:
    - metric coordinates, from columns 0–2, which the loader shifts to the block minimum and leaves in metres at
      evaluation [VIPSEG dataloaders/loader.py:60-66];
    - colour, from columns 3–5;
    - normals by PCA over the 16 nearest neighbours.
  - Preprocess exactly as each model's released inference example does: grid size, normalisation, feature channels.
    Take features at every input point through the grid's inverse map.
  - **Context variant `+ctx`** (F2 and F3 only): encode every raw point of the block file, then gather features at
    the episode's 2,048 sampled indices. A wrapper returns the indices; the loader stays untouched. `+ctx` is a
    protocol deviation: it also removes the density cue from the features. It is reported as a diagnostic, never
    as a result.
* **Rules scored per arm:**
  - U, P3's rule `F^q n(P_point)ᵀ` on unit features;
  - U + both + LP, with LP re-selected on the arm's valid as in D-48 amendment 1;
  - the cosine oracle (common norm) and the LDA oracle, as bounds, never results;
  - the participation ratio of unit query features on valid's first 300 episodes;
  - the K-shot fit a, c on D-46's 1,500 five-shot episodes;
  - own/other recall (P8).

  No arm has a trained head, so query order cannot enter; the swap check is not needed.
* **Draws.** valid (selection only), fixed100, random600 seeds 0–2, leak-free (P5 part D). S1 screens. S0 is scored
  and reported for every arm and is never used to select. Every run passes `eval.py`'s protocol guard (D-22).
* **Rules, fixed before any run.** "Holds at g" means: fixed100 gain ≥ g, paired episode-bootstrap CI above 0, and
  > 0 on all three random600 draws.
  * **P12.1 information source (clean control).** F2 or F2L U − F0 U holds at +5.0 on fixed100, **or** its leak-free
    U − F0 leak-free U ≥ +5.0 with the paired CI above 0. Then M6 is run with the LAM3C teacher.
  * **P12.2 exposure.** P12.1 fails, but F3 or F4 passes the same bar. Then the gain is not separable from having
    seen S3DIS unlabelled. M6 may run with that teacher, and its numbers carry the label "teacher saw the test scenes
    unlabelled".
  * **P12.3 stop.** Every arm F2–F4 is ≤ +1.0 over F0 on fixed100 and on leak-free. Then the information source is
    not binding at 1 shot on this protocol, M6 is not run, and the method line closes. The benchmark write-up
    (research note §7) becomes the main output.
  * **Otherwise:** report, no training.
  * **P12.4 pre-head** (diagnostic, no training): F1 U − F0 U holds at +2.0. Then the pre-head feature becomes the
    prototype space of later inference rules. This is inside the guardrail and needs no further decision.
  * **P12.5 density** (diagnostic): F2+ctx − F2 ≥ +3.0 on fixed100. Then M6's teacher sees raw blocks (cached, see
    below), not the 2,048 sampled points.
  * **Reported, no rule:** the fitted a against CR's 0.156 and c against 0.047; the participation ratio against 5.56;
    the oracles; S0.
* **Checks.**
  - F0 reproduces CR's U on fixed100 to within 0.01.
  - Each teacher's features are identical when the same block is encoded twice.
  - The inverse map covers every input point.
  - Extraction does not change any global RNG state that the episode sampler uses.

#### Part M6 — distillation into the light encoder (only after P12.1 or P12.2)

* **Student.** CR's configuration and schedule, unchanged: `vip_clean`, T 4, no LMA, L2 point prototypes, random
  query order, batch 1, 24,000 updates, LR halved every 7,200, 13 validations. Encoder and feature head are
  initialised from scratch.
* **Loss.** `L = CE(L_final, Y_q) + λ · L_KD`, where:
  - `L_KD = mean over every support and query point of (1 − cos(g(h_i), t_i))`;
  - `h_i` is the student's 900-d encoder output at point i;
  - `g` is a linear projector 900 → d_T, used in training only and dropped after;
  - `t_i` is the frozen teacher's feature, detached.

  λ = 1.0 is a convention; it is not tuned. `L_KD` uses no labels. The CE term is unchanged, so the episode labels
  the student sees are CR's.
* **Teacher features.** Computed online on the episode's points if P12.5 does not hold. Otherwise they are cached
  once per raw block, reduced to 64 dimensions by a PCA fitted on base-class training blocks only, stored in fp16,
  and gathered at the sampled indices. The cache size is not measured; the estimate is tens of GB.
* **Arm.** M6 with the teacher chosen by P12, seed 0. A second seed is added if a rule passes by less than +2.0
  (AGENTS §6).
* **Rules, fixed before the run.** Compare M6 `last` against CR `last`, each with U + both + LP selected on its own
  valid.
  * **D50.1 adopt:** M6 − CR holds at +2.0 on fixed100, and M6 − CR on leak-free ≥ 0. Then M6 becomes the base of
    later arms, in the separate table.
  * **D50.2 mechanism**, reported alongside D50.1: the student's participation ratio ≥ 12, and the fitted a below
    0.156. If D50.1 holds but D50.2 fails, the gain is reported as unexplained.
  * **D50.3 stop:** M6 − CR < +1.0 on fixed100 and leak-free U not above CR's. Then the teacher's information does
    not survive distillation into this encoder at 2,048 points. The line closes.
  * **Otherwise:** report; one more seed before any claim.
  * **Reported, no rule:** `best.pt`, S0, the student's own U against its teacher's U (the distillation gap), and the
    parameter count and FLOPs of the deployed student (identical to CR by construction).
* **Cost (not measured).**
  - Environment 1–2 h: spconv, torch_scatter, and the teacher's own package; weights fetched on the VM.
  - P12: about 3–4 GPU-h for all arms and draws, K-shot included.
  - M6: one CR-length run (about 1.7–2.3 GPU-h) plus the teacher's forward passes, or the cache build.
* **Affects** (nothing written yet):
  - `experiments/d50_features.py` (new): extractors behind one interface, with the `+ctx` wrapper.
  - `experiments/d50_eval.py` (new): P12 scoring, reusing P3/P6/P7/P8/D-46 code paths; the rules P12.1–P12.5 and
    D50.1–D50.3.
  - `experiments/run_d50.sh` (new).
  - `train.py` (`--kd_teacher`, `--kd_lambda`, run tag `_kd<teacher>`).
  - `models/kd_projector.py` (new).
  - `tests/test_d50_*.py` (new): extraction determinism, inverse-map coverage, F0 = CR's U, the KD loss against a
    written-out reference, the projector absent from the saved student, and the guardrail (the student's
    `state_dict` loads no teacher tensor).
  - AGENTS guardrail 1, 01 §2.1, and a new section in 05.
