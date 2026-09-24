# CONTEXT — what this project is trying to do, and where it stands

Docs version 2.0, 2026-09-24 (end of phase 16's experiments on fold S1). Read this first; the numbers below
are copied from result files, each with its source. Nothing here is claimed beyond what those files show.

---

## 1. Goal

**Improve VIP-Seg on few-shot 3-D point-cloud semantic segmentation (S3DIS, 2-way 1-shot) with the ideas
of CascadeProto — entropy-aware prototype purification, a cascade of refinement stages, and multimodal
(cross-modal) prototypes — under the standard protocol, honestly.**

* **Target:** 80+ mIoU. This is an aspiration, not a result: the best number this repository has measured
  is 75.05 on fold S1 (best-of-validation) and 73.20 (last checkpoint) (§3). Nothing in the repository reaches
  80 today, and no measured mechanism points to a path there yet (§5).
* **Baselines to beat,** as published: VIP-Seg 72.20 (S0) / 76.09 (S1), EDS-Net 73.32 / 74.67.
* **Honestly** means: no scoring on classes seen in training (enforced by `eval.py`, D-22); design choices
  screened on S1, S0 held out; every experiment's decision rule written before it runs; every claim tied to a
  result file; `last.pt` and best-of-validation `best.pt` both reported (D-22 amended).

The project began as a re-implementation of the CascadeProto paper (phases 8–15). That reproduction is closed
(§6). Phase 16 builds on VIP-Seg's own head ("route B", D-25), not on the paper's modules.

## 2. Vocabulary

| Term | Meaning here |
| :--- | :--- |
| episode | N = 2 classes ("ways"), K = 1 support block per class, one query block per class; 2,048 points per block |
| support / query | labelled example blocks / blocks to segment |
| prototype | a class's mean feature (masked average of its support points); background is row 0 |
| `P^0`, `M_eff` | the first prototype matrix; the effective prototype after the head, with `logits = F^q M_effᵀ` |
| head | VIP-Seg's four modules PEM → PDM → PEM → PDM plus a stage-weighting layer (our ADRM ≈ VIP-Seg's gating) |
| oracle rule | the query's own class means used as prototypes (needs query labels: an upper bound, never a result) |
| fixed100 / random600 | the table's 1,500 cached test episodes / three extra independent draws of 600 (seeds 0, 1, 2) |
| `last` / `best` | the final checkpoint / the best of the validations on test-class episodes (VIP-Seg's own selection) |
| E1 | route B's base: VIP-Seg's head trained in our loop on VIP-Seg's update count (D-30) |
| S0 / S1 | the two class folds; S1 is the screening fold, S0 is held out |

## 3. Where things stand (S3DIS 2-way 1-shot, fold S1, fixed100)

| model | `last` | `best` | source |
| :--- | ---: | ---: | :--- |
| VIP-Seg released checkpoint (their best of 12 validations) | — | 75.36 | `results/phase16_e1/SUMMARY.md` |
| **E1** — VIP-Seg's head in our pipeline, 24,000 updates | **73.20** | **75.05** | `results/phase16_e1/SUMMARY.md` |
| r0 — same head, 6,000 updates (our earlier schedule) | 70.20 | 70.20 | `results/phase16_r2/SUMMARY.md` |
| oracle rule on E1's features (upper bound) | 85.93 / 87.43 | — | `results/phase16_p3/SUMMARY.md`, `results/phase16_e1/` |

* E1 matches VIP-Seg's release within 0.32 points on best-of-validation on all four draws (−0.31, −0.15, −0.32,
  +0.02). Our pipeline
  reproduces VIP-Seg; it does not yet improve on it.
* Selection alone is worth +1.3 to +1.9 points on one run (E1 `best` − `last`), and VIP-Seg's own S0 log ends
  at 68.97 while its published 72.20 is the best of 12 validations (`docs/research/2026-09-24_r2_distill_analysis.md`).
* **S0 has not been run for route B.** No S0 number of ours on VIP-Seg's head exists.

## 4. What phase 16 tested (fold S1; every rule fixed before its run)

| decision | idea | outcome | source |
| :--- | :--- | :--- | :--- |
| D-22 | protocol guard: never score seen classes | in force (`eval.py` refuses) | 00 |
| D-23, D-24 | repair the paper's EPPM stage | EPPM is 15.5 points below one VIP-Seg PEM; the stripped stage is worse still | `results/phase16_r1/` |
| D-25 | route B: build on VIP-Seg's head | adopted | 00 |
| D-26 | query-side EM refinement with an entropy weight (test time) | stop: S1 +0.37, S0 −1.21 | `results/phase16_p0/` |
| D-27 | base-class calibration of the background (test time) | stop: −0.03 to −0.16 | `results/phase16_p1/` |
| D-28 | D-26 filtered by D-27's margin | stop: the filter was never selected | `results/phase16_p2/` |
| D-29 | train the head toward the oracle's decisions | stop: +0.06 | `results/phase16_r2/` |
| D-30 | train for VIP-Seg's update count (E1) | adopted: +2.99 over r0, reproduces VIP-Seg | `results/phase16_e1/` |
| D-31 | training-free text prior (four prompt sets, two maps) + gap decomposition | text stops: 0 of 192 arms gain, upper bound +0.70 | `results/phase16_p3/` |
| D-32 | background contaminated by the episode's own classes | refuted by intervention: +0.09 with a perfectly clean background | `results/phase16_p4/` |
| D-33 | zero-init point-level support → query attention neck, warm start | stop: the neck never opened (α 0.0023) | `results/phase16_n1/` |
| D-34 | the same neck trained from scratch | stop: the neck opened (α 0.060) and cost 0.5–1.1 points | `results/phase16_n2/` |

Research notes behind these: `docs/research/2026-09-22_improvement_directions.md`,
`2026-09-23_gap_and_upgrade_research.md`, `2026-09-24_r2_distill_analysis.md`,
`2026-09-24_text_integration_math.md`, `2026-09-24_text_integration_independent.md`.

## 5. What is established, and what is not

**Established by measurements in this repository (S1):**

1. The features are not the bottleneck at this level: with the query's own class means as prototypes, the same
   features and decoding reach 85.9–87.4 against E1's 73.2 (P3).
2. The error is a joint, query-conditioned shift of every prototype: replacing only the background row with
   the oracle costs 26 points, only the foreground rows 9, all rows together gains 13.7 (P4).
3. E1's dominant errors are foreground points predicted as background on the large planar classes (floor,
   wall recall 0.72 / 0.73; oracle 0.99 / 0.95); they do not depend on support size (P3).
4. The existing text path (LMA + GMMN) is class-blind by construction on route B: template prompts at cosine
   ~0.9, GMMN reduces to matching one foreground mean, six training names per fold (text notes, T0).
5. None of the mechanisms in §4 closes any measurable part of the oracle gap; training length (E1) is the only
   change that moved the level.

**Not established (do not claim):**

* Anything on S0 for route B, and any average over folds.
* That text or any other modality cannot help in general: only the class-name text path, as built here, was
  tested, on one feature space, with six training names per fold.
* That point-level query attention cannot help: one design, one seed, one budget (D-33, D-34).
* Why floor, wall and ceiling (the classes present in most blocks, always background during training) have the
  lowest IoUs in both folds: suggestive (Spearman −0.55, p = 0.06, 12 classes), not causal (P4 follow-up); a
  causal test needs those classes' labels in training, which the protocol forbids.
* Effects below about 1.5–2 points from single training runs (training noise ≈ 1 point on `last`).
* Anything about 80+: no measured mechanism has produced a gain above E1.

## 6. The earlier reproduction (phases 8–15, closed 2026-09-22)

Every equation of CascadeProto was implemented and every row of its Table 4 trained on the full schedule. The
pipeline reproduces VIP-Seg's released checkpoint (0.7197 against 0.7220); the paper's absolute level does not
reproduce (baseline 49.08 / full model 57.15 against 82.72 / 88.53) and is matched only by scoring classes seen in
training. The analysis of why, and the equation audit: `docs/research/2026-09-21_reproduction_report.md`,
`docs/research/2026-09-20_paper_vs_code_audit.md`. In the same pipeline the paper's EPPM stage is 15.5 points below one
VIP-Seg module (R1); a structural reading of why (entropy of a channel magnitude, a class-blind diffusion
term, class-shared fusion) is in `docs/research/2026-09-24_text_integration_independent.md` §6 — an analysis,
not a measurement.

## 7. Open directions, with the evidence each has

| direction | evidence for | cost |
| :--- | :--- | :--- |
| E1 on S0 (the held-out fold's reference) | needed for any claim; no gain expected by itself | ~2.3 GPU-h |
| a second modality that carries **per-point** information (S3DIS's 2-D images through a frozen 2-D vision-language model, as MM-FSS does on its data) | the only listed direction that brings information the support, query and class names do not already hold; not measured here; needs a decision on the "no pre-training" comparison | days of engineering, then GPU-h |
| a second seed of N2 or of any future arm | only to confirm a gain of 1–2 points | ~2.3 GPU-h per run |
| the "always background in training" diagnostic | would explain the planar-class errors; only as a leakage-style diagnostic (like D-21), never a method | ~2 GPU-h |

## 8. How work is done here

* Decisions before code: `docs/spec/00_SOURCES_AND_DECISIONS.md` (D-01…D-34), each with its problem, evidence,
  rules and outcome; specs `01`–`05`; `AGENTS.md` for the guardrails; `docs/CHANGELOG.md` for every step.
* Every experimental choice cites prior evidence; estimates are marked "not measured".
* One training run per arm, several test draws; a short GPU smoke before every long run; the VM shuts itself
  down after unattended runs (`AUTOSTOP=1` in `experiments/run_n2.sh`, on-VM watchers).
