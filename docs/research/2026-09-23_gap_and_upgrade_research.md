# Where the reproduction breaks, what the paper's ablations contradict, and which upgrades can beat the baselines honestly

Status: 2026-09-23, phase 16 open, R1 not yet run (`results/phase16_r1/` does not exist). Desk research: the paper
(`10069.pdf`, text extracted with `pdftotext -layout`), the repository at commit `9361587`, the result files under
`results/`, and primary external sources. No GPU run and no code change. The existing notes are cited, not repeated:
*report* = `2026-09-21_reproduction_report.md`, *diagnosis* = `2026-09-21_gap_diagnosis.md`, *external* =
`2026-09-21_external_sources_on_gap.md`, *audit* = `2026-09-20_paper_vs_code_audit.md`, *directions* =
`2026-09-22_improvement_directions.md`. Each claim is marked **[verified]** (read from a primary source or a result
file) or **[inferred]** (a reading or a prediction). Everything new in this note is listed in §0.

## 0. What this note adds to the existing ones

1. **A shot-insensitivity fingerprint in the paper's Table 2** (§3, C4): 1 → 5 shots is worth +0.15 Avg to
   CascadeProto (Text) in 2-way and **−2.11** in 3-way, against +2.86 / +0.72 for VIP-Seg and +2.74 / +1.68 for
   DyPolySeg in the same table. It is the pattern expected when the scored classes are already known to the model.
2. **Table 4 row 3 against Table 5 `T = 1` is contradictory under both readings, not only under D-17's** (§3, C2),
   because ADRM with one stage is the identity.
3. **R1 has a confound and too little statistical power for two of its rules** (§4.5): variants (a) and (b) run
   without the L2 normalisation that (c), (c′), (d) get, and L2 alone was worth +3.36, the same size as the +3
   thresholds of R1.3 and R1.4. At the harness's seed spread, `t > 3` with three seeds needs a true difference of
   about 5 points.
4. **Backbone evidence from the corrected-protocol literature** (§6): in MM-FSS, COSeg retrained on MM-FSS's stronger
   2D-aligned backbone gains +1.12 (S3DIS) and +0.25 (ScanNet) on 2-way 1-shot, while MM-FSS's head on the same
   weights adds +6.2 and +10.2. In the old protocol a masked-average-pooling (MAP) baseline scores 47–52 on every
   encoder that has been tried.
5. **2025–2026 state of the art in the corrected protocol** (QHP, DA-FSS) and the target's own run-to-run variance
   (VIP-Seg's poster against its released logs, ±1.2 points).
6. **VIP-Seg's released S1 checkpoint** is at the pinned commit (`log_s3dis_VIPSeg/log_S1_N2_K1_0.760875/checkpoint.pt`),
   so test-time procedures can be tuned on S1 and applied unchanged to S0, as D-22 requires.

## 1. TL;DR

* **Q1, why the reproduction misses.** No bug has been found. Report §1–§3.3 closed the pipeline, the metric, the
  encoder, the schedule and every decision of ours. **(i) The absolute level**, about 31 points on the Avg column:
  the paper's numbers match scoring on **classes seen in training**. Our baseline scored on seen classes gives
  77.32 / 71.58 against the paper's 82.72 / 79.83, which removes 24 of the 30.8 points. This is consistency, not
  proof. **(ii) The attribution:** cascade depth adds −0.17 for us against the paper's +2.55. By construction, stages
  2–4 receive no new information and are close to an isometry (directions §3.5). **(iii) An unprinted L2
  normalisation step** worth +3.36.
* **Q2, the ablations.** They are self-contradictory in at least eight ways (§3). The four strongest: the plain
  baseline scores 10.5 points above the VIP-Seg model it is stripped from; Table 4 and Table 5 disagree about the
  same one-stage model under every reading; 5 shots do no better than 1 shot (3-way: −2.11); and the paper's own
  first author measures the same three ideas at +1.30 on S0 in EDS-Net, against +5.81 here.
* **Q3, the phase-16 directions.** D-22 is *needed but not sufficient*: it is the protocol guard every result depends
  on. D-23 (pooled `S′`) is the right first test, but weak on its own: at N = 2 it changes only which support
  statistic enters Eq.13, and the ablation evidence for "shared correlation" was measured with a different operator.
  D-24 (EPPM-S) is **strong** as a diagnostic. D-25 (VIP-Seg stage) is **needed** as the in-loop reference. **R1 as
  scripted** is useful, with two fixes. Add an L2-normalised printed stage (`r1_eppm_l2`), and put `r1_vip4` in the
  default queue. Read R1.1 and R1.2 as screening, not as tests: with 3 seeds and sd ≈ 2, the rule `t > 3` needs
  about 5 points. **None of D-22…D-25 can beat VIP-Seg alone.** At best they bring the head back to VIP-Seg's
  level, about 72.
* **Q4, which components to change.** In order of expected gain per GPU-hour:
  1. the correlation head (R1);
  2. **query-side, entropy-weighted transductive purification** (the principled form of "entropy-aware prototype
     purification"): the oracle headroom is about 25 points, and the literature measures +1 to +4;
  3. per-stage supervision with oracle-prototype distillation (+2.7 to +3.3 in DPA and QGE);
  4. base-class calibration (+3.44 in COSeg's setting);
  5. text as a support-validated gated prior (+0.3 to +2);
  6. multi-prototypes (+1.4 to +3).

  Replace or remove: the pointwise entropy gate, `P_diffuse`, depth without feedback, and the fixed 1:1 text fusion.
* **Q4, the backbone.** The backbone is **not** the ceiling at our level (57) and probably not at 72 either. MAP
  scores 47–52 on five different encoders (non-parametric, pretrained DGCNN, Taylor, DyPoly, VIP-Seg Mamba). The
  same head family gains about 7 points from a non-parametric encoder to the Mamba encoder (Seg-PN 64.84 → VIP-Seg
  72.20), but 17–25 points over MAP on a fixed encoder. In the corrected protocol a stronger backbone gives COSeg
  +0.25 to +1.12. A pretrained PTv3 or Sonata would also break guardrail 1 and the comparability with the
  from-scratch baselines.
* **Q5, the single most worthwhile next step.** After R1 settles the head, **entropy-weighted query-side EM
  purification** (directions §5.2): first as a zero-training probe, tuned on VIP-Seg's released **S1** checkpoint and
  our S1 checkpoint and applied frozen to S0, then trained with per-stage and oracle-prototype supervision on the
  winning head. The plan and decision rules are in §7.
* **What "beating the benchmark" requires statistically.** VIP-Seg's own numbers move by ±1.2 between its poster
  and its logs, and ours by 1–3 points per seed. A claim of beating 72.20 / 76.09 needs three seeds per fold and a
  mean at least about +2 above the target. The fair comparison is VIP-Seg retrained through our loop with the same
  seeds and `last.pt` (R0), not its best-of-12 released log.

---

## 2. The reproduction gap, ranked

Three different gaps are often mixed. They have different causes.

| # | gap | size (S3DIS S0, 2w1s) | cause | status |
| :--- | :--- | ---: | :--- | :--- |
| G1 | absolute level, ours → paper | baseline 49.08 → 82.72 (33.6); full 57.15 → 88.53 (31.4) | numbers consistent with scoring on classes seen in training | **[verified] consistent**: 77.32 / 71.58 seen-class baseline, 81.28 full (`results/seen/*_last.json`); residual 5–8 points not decomposed |
| G2 | attribution: depth | T = 1 → 4: −0.17 (paper +2.55) | a stage after the first receives no new input (no `L^{t−1}`); Jacobian ≈ isometry | **[verified]** measured (report §2); mechanism CPU-checked (directions App. A) |
| G3 | ours → VIP-Seg, the honest target | 57.15 → 71.97 (14.8) | head design: correlation per class slot (D-01) versus the family's episode-shared form, plus the inert parts of Eq.15–21 | **open**; R1 is the test |

**G1, ruled out, with the evidence that closes each item** (report §3.1–§3.3, diagnosis §4b). Each is **[verified]**:
* A different mIoU: the maximum under any definition is 64.85.
* VIP-Seg's trained encoder: MAP on it scores 47.37.
* Schedule D-12: batch 1 × 24,000 steps gives 49.01.
* Partial class leakage (D-21): +12 to +15, not the level.
* Our decisions on the baseline's path: each one traced.
* The harness: VIP-Seg's checkpoint scores 71.97 against 72.20.

**G1, what remains.** The residual of 5.4 (S0) and 8.2 (S1) points under the seen-class reading has three candidate
sources: one seed, fixed100 against the paper's 600 random episodes, and checkpoint choice. **[inferred]** A new,
independent fingerprint supports the seen-class reading (§3, C4): if the classes are known, extra support shots
should add almost nothing, and in the paper they add nothing. A zero-training check would score our S0 `last.pt` on
fold 1 at 1 shot and 5 shots (with `--allow_seen_classes true`, labelled as a diagnostic). Under the reading, the
seen-class 5-shot gain is ≤ 1 point, against +3 to +4 on unseen classes (VIP-Seg +4.28, DyPolySeg +3.97 on S0). This
is optional: the project's verdict does not depend on it.

**G2** is settled for the printed equations. Depth can only have a meaning if each stage reads the previous posterior
(directions §3.5). That is also why EDS-Net's four "generations" add only +0.28 Avg [EDS Fig. 2, via directions §4.6].

**G3** is what phase 16 is about, and the only gap that matters for beating the baselines. What is known
**[verified]**:
* Our first EPPM stage gives +7.07 over the baseline.
* VIP-Seg's head on its own encoder gives +17 to +25 over MAP (report §3.2).
* VIP-Seg's whole model through our loop reached 0.6948 after only 600 steps (report §1).

So the loop can train a 70-level head; the missing points are in the stage design. How they split between D-01's
per-slot correlation, the missing L2 input, the missing channel-preserving term (D-19) and Eq.19–21's additions is
not known. That split is exactly what R1 measures, if the confound of §4.5 is fixed.

## 3. Contradictions in the paper's tables

Paper numbers from `10069.pdf` Tables 2, 4, 5, 6 and §4.2–4.4 [verified from the PDF text; the CascadeProto rows
also match `experiments/summarize.py:21-28`].

| # | where | what the paper prints | why it is inconsistent | kind |
| :--- | :--- | :--- | :--- | :--- |
| C1 | Tab.4 row 1 vs Tab.2 / Tab.6 VIP-Seg | baseline "plain VIP-Seg backbone with masked average pooling" 82.72 / 79.83 / 81.28; VIP-Seg 72.20 / 76.09 / 74.15 | removing VIP-Seg's whole prototype stack would gain +10.5 on S0. On VIP-Seg's own trained encoder MAP scores 47.37 (report §3.2), and every published MAP baseline scores 47–52 (external §4) | arithmetic / baseline mismatch **[verified]** |
| C2 | Tab.4 row 3 vs Tab.5 `T = 1` | 85.34 / 82.48 / **83.91** against 85.21 / 81.34 / **83.28** | ADRM with one stage is `softmax` over one logit set, i.e. weight 1 (Eq.24–25), so Table 5's `T = 1` equals a one-stage model with or without ADRM. **Reading (i)**, row 3 = one EPPM stage (D-17): the same model is printed twice, 0.63 apart. **Reading (ii)**, row 3 = gate only (audit §11): adding cross-attention and diffusion to the gate *lowers* Avg by 0.63, which contradicts "each component helps". No reading is consistent | internal contradiction **[verified]** (new argument) |
| C3 | Tab.5 vs Tab.4 increments | Tab.5 `T = 1 → 4` +3.25 Avg (with ADRM); Tab.4 cascade +2.06, ADRM +0.56 | C2 carried forward: the depth gain implied by Table 5 without ADRM (3.25 − 0.56 = 2.69) differs from Table 4's 2.06 | derived from C2 |
| C4 | Tab.2, 1 → 5 shots | Text: 2w 86.53 → 86.68 (+0.15; S0 88.53 → 88.57); **3w 81.06 → 78.95 (−2.11; S0 83.07 → 79.95, −3.12)**. Image 3w 80.23 → 79.31 (−0.92); Audio 3w 80.50 → 80.48 | in the same table VIP-Seg gains +2.86 (2w) / +0.72 (3w) and DyPolySeg +2.74 / +1.68; elsewhere, five shots are worth +1.5 to +4.3 (directions §2.1). More labelled support that hurts by 3 points on S0 has no mechanism in a prototype method, **except** that the scored classes are already known, so the support carries little information | fingerprint of seen-class scoring **[inferred]** (new). ScanNet shows the same sign: Text 2w 79.57 → 79.25 (VIP-Seg +0.51) |
| C5 | Tab.2 / Tab.4 fold order | S0 > S1 on every CascadeProto row (82.72 / 79.83 … 88.53 / 84.53) | every other method with both folds has S1 > S0 (VIP-Seg 72.20 / 76.09; DyPolySeg 72.02 / 73.82; our unseen 49.08 / 51.91). Swapped-fold scoring reproduces S0 > S1 (77.32 / 71.58) | fingerprint **[verified]** (diagnosis §4, report §3.6) |
| C6 | Tab.4 vs EDS-Net Tab.5 (same first author) | LMA + cascade + ADRM: +5.81 S0 / +5.25 Avg | EDS-Net implements the same three ideas (noise-conditioned CLIP generator with a GMMN loss, four refinement generations, GAP-softmax routing) on a 72-level head under the standard protocol: +1.30 S0 / +1.08 Avg | external inconsistency **[verified]** (directions §4.6) |
| C7 | Tab.6 / §4.4 | 2.88 M = VIP-Seg 2.76 M + 0.12 M | four EPPM stages from Eq.10–21 cost 317,580 parameters, and LMA alone costs 148,352 (audit F5) | arithmetic **[verified]** |
| C8 | Eq.11 and Eq.28 vs the claims | the gate "amplifies low-entropy foreground"; the cascade is a contraction | `g ∈ [0.405, 0.731]`: it can only attenuate (audit F7). The stage Jacobian on LayerNorm'd input has median singular value 0.996–1.004 (directions App. A) | theory vs mechanism **[verified]** |
| C9 | §4.1 protocol vs Table 2's cited rows | "Areas 1–4, 6 train / Area 5 test", "600 random episodes" | the cited VIP-Seg row is its released logs (class split over all areas, 100 fixed episodes per combination; report §3.4). The table mixes protocols, or the stated protocol is not the one used | protocol mismatch **[verified]** |
| C10 | §4.3 vs Table 5 | "Table 5 reports performance and inference time", "≈16 ms per stage" | Table 5 has no time row | missing data **[verified]** (audit F11) |

**How to characterise it.**
* C1, C4 and C5 are the pattern of a protocol difference that lifts every row: a near-uniform offset, a reversed fold
  order, no benefit from extra support. Seen-class scoring produces all three **[inferred]**.
* C2, C3, C7 and C10 are internal bookkeeping errors, independent of protocol.
* C6 and C8 say that even the *relative* claims do not survive under the honest protocol.

Note that the modality rows (Audio / Image / Text) sit within 0.9 Avg of each other on 2w1s, and LMA is credited with
only +1.21. So the paper's own numbers do not attribute its level to the modality either.

## 4. The phase-16 directions

### 4.1 D-22, the protocol guard

* **Effect size:** 0 by design. It prevents the 22–25-point seen-class inflation (report §3.6).
* **Risk:** none.
* **Verdict: needed, not sufficient.** Its reporting rules have two consequences worth stating in any write-up:
  * **Asymmetric selection.** VIP-Seg's 72.20 / 76.09 are the best of 12 validations on test-class episodes
    (external §1), while our headline is `last.pt`. This is conservative for us, by an unknown amount. Our own
    `best` − `last` gaps are ≤ 2.3 points.
  * **S1 is not held out.** Designs are screened on S1's `valid` draw, so the S1 test number is on classes the
    design was chosen on (different episodes, same classes). Only S0 is clean. A claim to beat VIP-Seg on S1
    (76.09, the harder target) inherits that caveat. D-22 rule 2 already says so; keep it in every table.

### 4.2 D-23, one pooled `S′`

* **Mechanism.** With one `A_b` shared by all class rows, the contrast between two classes is an episode-adaptive
  bilinear metric on the support-derived contrast. With one `A` per slot, a class is pulled toward the query whenever
  its support *scene* resembles the query (directions §3.2).
* **Evidence for.** The shared form is what QUEST/APP/PEM/PDM effectively compute and what DPA writes (+13.7) (D-23).
* **Evidence against a large effect** **[inferred]**:
  * (i) The +13.7 to +19.3 of §4.3 in *directions* is correlation versus *no correlation*, not shared versus
    per-slot. No source ablates that specific difference.
  * (ii) At N = 2, K = 1 the pooled `S̄′` is the mean of two support blocks. The per-slot matrices already differ
    only by which block enters, and at initialisation Eq.14 sits at the uniform end (attention width 127.9998 of
    128, CHANGELOG 16b).
  * (iii) The printed stage around it still carries the inert parts (§3 of *directions*).
* **Expected:** 0 to +3.
* **Verdict: weak alone, but the cheapest clean test.** Keep it as R1(b).

### 4.3 D-24, EPPM-S

* **Mechanism.** It strips everything proved inert and adds the channel-preserving self term and the L2 input. It is
  the CascadeProto-shaped member of the QUEST/APP family.
* **Evidence.** The family's ablations give +15 to +20 for this kind of stage over MAP (directions §4.3). D-19's
  measurement shows the self term was missing.
* **Expected:** 57 → 64–71 at T = 1 (the directions prediction).
* **Risk.** Moderate. It is no longer "the paper's method", and must be labelled as outside the paper (D-24 does).
* **Verdict: strong as a diagnostic and as route A's stage 0.** It still ends at the VIP-Seg level, not above it.

### 4.4 D-25, VIP-Seg's PEM/PDM as a stage

* **Mechanism.** The reference head inside our pipeline.
* **Verdict: needed.** Without it, R1 has no in-loop 72-level anchor. `r1_vip4` (the whole four-module head) is not
  in the default queue of `experiments/run_r1.sh` (`VARIANTS` defaults to five variants without it). It should be:
  it is the number every later increment is measured from under route B.
* **Risk.** The query mixing of the reshape (D-25) is a transductive coupling between the two queries of an episode.
  It is legitimate at test time but not present in our stages; report it.

### 4.5 R1 as scripted: two design issues and the statistics

1. **The L2 confound [verified from `experiments/diag_short.py:53-60`].** `r1_eppm` (a) and `r1_pooled` (b) run
   with `l2norm_point_proto=false` (the default, `train.py:58`); `r1_eppms` (c), `r1_eppms_slots` (c′) and
   `r1_vippem` (d) run with `true`. L2 alone measured +3.36 on the baseline (report §2, one seed). So:
   * R1.3 ((c) − (a) ≥ +3) and R1.4 ((d) − (a) ≥ +3) can fire from normalisation alone. The +3.36 may be partly
     absorbed by Eq.21's LayerNorm in a stage, but that was never measured.
   * **Fix:** add `r1_eppm_l2` (and optionally `r1_pooled_l2`). Then compare the stages against each other with
     L2 on everywhere, and against `r1_baseline_l2`.
2. **The harness has no LR decay** [verified, `diag_short.py:6-7,164`]. The divergence point that justifies 2,400
   steps (CHANGELOG 15k: epochs 10–20) was observed on the full schedule, where the LR halves at step 1,200. The
   ranking at a constant 1e-3 is probably, but not certainly, the same **[inferred]**. Confirm the winner on the full
   schedule before building on it (directions R1 → R0 step).
3. **Power.** The harness seed sd is 0.010–0.031 (report §6), plus the noise of 300 valid episodes. With three seeds
   per arm, the standard error of a difference is 0.82 × sd:

   | seed sd (points) | SE of a difference | difference needed for `t > 3` | expected t at a true +3 |
   | ---: | ---: | ---: | ---: |
   | 1.0 | 0.82 | 2.45 | 3.67 |
   | 2.0 | 1.63 | 4.90 | 1.84 |
   | 3.1 | 2.53 | 7.59 | 1.19 |

   Welch's df at n = 3 per arm is about 2–4.
   * R1.1 (`≥ +3 with t > 3`) will usually be **inconclusive** for a true effect of 3–4 points.
   * R1.2 (`|c − d| ≤ 2`) cannot establish equivalence: the 95 % interval of the difference is about ±4.5 at sd 2.
   * Treat R1 as a screen. Promote any variant within 3 points of the best to the full schedule, where the
     variance is lower. Going from 3 to 5 seeds shrinks the SE only by a factor √(5/3) ≈ 1.29, so the full
     schedule is the better use of the hours.
4. **What R1 cannot show.** Whether anything goes *above* 72. Every R1 variant is a support-side stage. By directions
   §2.1, support-side error is the small term (1.5–4.3 points in total), and the large term (~25 points) is
   query-side. R1 decides the base, not the gain.

**Overall verdict on phase 16 so far.** Correct order and hygiene (guard first, one variable at a time, pre-registered
rules). Aimed at recovering VIP-Seg's level, which is **necessary but not sufficient**. The directions note itself
predicts route A at 64–71 after stage 0. Beating 72.20 / 76.09 needs something the R1 variants do not contain.

## 5. Upgrade directions, ranked by expected gain per GPU-hour

Gains are for S3DIS 2w1s under the standard (old, 2,048-point) protocol. Literature numbers are the source's own and
usually shrink on a stronger base **[inferred]**.

| rank | direction | mechanism | evidence (primary) | expected on a 72 base | cost (L4) | fits the repo rules? |
| :---: | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | **Head = episode-shared correlation stage** (R1 winner, or VIP-Seg's head) | query-conditioned channel metric on the prototype contrast | Seg-PN +17.5 S0, TaylorSeg +17.7, DyPolySeg +22.9 over MAP; DPA shared correlation +13.7 mean (directions §4.3) | brings 57 → ~70–72; nothing above | R1 ≈ 6.6 h + full-schedule confirm 4.5 h | yes (D-23/24/25) |
| 2 | **Query-side entropy-weighted EM purification** (feedback stages) | E-step posterior `r`, entropy weight `w = 1 − H/ln(N+1)`, M-step query prototype, prior-anchored update; T = 2–3 | oracle query prototypes: 66.40 → 93.89 (QGE T1, 1-way); SSP self-support +3.1/+4.0 (PASCAL); DPA prototype-to-query attention +3.05; MPTI label propagation 52.27 vs ProtoNet 48.39 | **+1 to +4** (directions §5.2) | probe < 1 h; training 3 seeds × 2 folds ≈ 10 h | yes; new PROPOSED decision (D-26), flag off by default |
| 3 | Per-stage supervision + oracle-prototype distillation (training only) | CE on each stage's logits + KL toward the logits of the query's own class means (base-class labels only) | QGE +3.3 (T5, 1-way); DPA stage self-distillation +2.65 (T3) | +1 to +2.5 | small, same runs as 2 | yes; query labels of *training* episodes are base-class labels |
| 4 | Base-class calibration (BPC) | EMA prototypes of base classes; raise background logits where the query matches them | COSeg +3.44 (T3, corrected setting) | +0.5 to +2 | 3 seeds ≈ 3–9 h | yes; only base-class labels |
| 5 | Normalised logits with learned scale | removes `‖p_c‖` as a class temperature; NormFace floor | L2 prototypes +3.36 on our baseline (report §2); `1/√D` −5.5 (D-10) | 0 on a VIP-Seg-like head (already normalised); +3 on the printed model | trivial | yes (D-10 switch exists) |
| 6 | Multi-prototype support (k-means / hub points per class) | covers multi-modal classes such as clutter and background | AttMPTI multi-prototype 52.27 vs 48.39; QHP hub prototypes +1.44, +2.96 with PDO over COSeg (1w1s S0, corrected setting) | +0.5 to +2 | medium | yes |
| 7 | Text as a support-validated gated prior | weight `λ` = IoU of the text-only prediction on the support (MM-FSS TACC) instead of a fixed 1:1 | TACC +1.9/+2.0 over no calibration; text branch +3.3/+3.7 (MM-FSS T3, corrected setting); EDS-Net text +0.38 on a 72.9 model | +0.3 to +1.5 | small | yes (text is the priority modality) |
| 8 | Base-class supervised pretraining of **our own** encoder on the training fold | standard two-stage recipe (AttMPTI, COSeg, MM-FSS) | no clean old-protocol ablation found; VIP-Seg and DyPolySeg reach 72 **without** it (external §1) | 0 to +2 **[inferred]** | +2–4 h per fold | **needs a new decision.** Guardrail 1 forbids pre-trained point-cloud *weights*; own-fold pretraining loads none, but the paper and VIP-Seg state "no pretraining", so it changes the comparison |
| 9 | Pretrained large backbone (PTv3, Sonata, Stratified Transformer with released weights) | better features | COSeg† on MM-FSS's 2D-aligned backbone: +1.12 S3DIS, +0.25 ScanNet 2w1s (MM-FSS T1/T2) | small at this head level; incomparable with from-scratch rows | large | **violates guardrail 1**; would need a new decision and a separate comparison table |
| — | *drop*: pointwise entropy gate, `P_diffuse`, deeper printed cascade, stronger GMMN | proved inert or harmful (directions §3, §7) | T = 1 → 4 −0.17; diffusion weight → 0.01–0.08 | 0 | — | — |

**Transductive, but legitimate.** Directions 2–4 use the query's *features* at test time, never its labels. At
training time they use base-class labels only. Hyper-parameters of test-time procedures are tuned on the S1 fold
and applied to S0 unchanged (D-22 rule 3).

**Protocol caveat for a paper.** All the old-protocol numbers share the loader's foreground over-sampling, which
inflated AttMPTI, QGE and QGPA by 23–36 points in COSeg's re-training (external §3.1). The 2025–2026 methods report
only the corrected setting. For S3DIS 2w1s mean: COSeg 36.95, QHP 38.35, MM-FSS 44.30, DA-FSS 44.62. A method that
claims SOTA should eventually show both settings, or at least score its final model with `random_sample=True`
(directions §7) to show that its gain does not depend on density.

## 6. Is the backbone the ceiling?

**No, not at our level. At VIP-Seg's level it is a co-factor, not the binding one.** Evidence:

1. **A fixed head on different encoders gives the same number** [verified]. MAP scores:
   * ProtoNet on pretrained DGCNN: 48.39
   * Seg-NN, non-parametric: 49.45
   * TaylorSeg-PN without APP: 49.42
   * DyPolySeg without PCM: 49.17–52.21
   * VIP-Seg's own trained Mamba encoder, zero-shot: 47.37
   * our encoder: 49.08

   (external §1, §4; report §3.2.)
2. **A fixed encoder with different heads gives very different numbers** [verified]. On VIP-Seg's encoder: MAP 47.37,
   L2 MAP 54.97, VIP-Seg's head 71.97 (report §3.2). On a frozen non-parametric encoder, QUEST alone adds +17.5
   (Seg-PN; directions §4.3).
3. **Better encoders under the same head family** [verified, Table 2 of the paper and the sources]: Seg-PN 64.84
   (frozen, non-parametric) → TaylorSeg-PN 67.12 → DyPolySeg 72.02 → VIP-Seg 72.20. That is about 7 points spread
   across encoders, and the heads are not identical either. The encoder therefore matters, but less than the head.
4. **Corrected-protocol evidence** [verified, MM-FSS Tables 1–2]. COSeg retrained on the 2D-aligned backbone that
   MM-FSS uses gains +1.12 (S3DIS 2w1s mean 36.95 → 38.07) and +0.25 (ScanNet 28.78 → 29.03). MM-FSS's head and
   text branch on the *same* weights add +6.2 / +10.2. The MM-FSS authors state that the backbone weights alone do
   "not significantly improve" COSeg.
5. **Headroom inside the current features** [verified in the literature, not yet measured here]. Oracle query
   prototypes on DGCNN features reach 93.89 against 66.40 (QGE T1, 1-way). The features hold far more than
   support prototypes extract. Probe P0.1 of the directions note would measure this on our encoder in minutes.
   It is still unrun, and it is the direct test of this section's claim.

**Conclusion.**
* At 57 the limit is the head (G3).
* At 72, the remaining error is dominated by the query-side term (directions §2.1), which a backbone does not
  address.
* A pretrained large backbone would break guardrail 1, make the comparison with the from-scratch rows of Table 2
  unfair, and, on 2,048-point, 1 m blocks, has no published evidence of a large gain.
* Spend the budget on the head and on query-side purification. Revisit the encoder only if P0.1's oracle comes out
  below about 75 on the 72-level checkpoint.

## 7. Recommended next experiment after R1

### 7.0 Before running R1 (cheap amendments)

* Add `r1_eppm_l2` (and optionally `r1_pooled_l2`) to `experiments/diag_short.py::VARIANTS`.
* Put `r1_vip4` in the default `VARIANTS` of `experiments/run_r1.sh`.
* Add a rule R1.6 to `summarize_r1.py`: the (c)/(a) and (d)/(a) comparisons are also reported against `r1_eppm_l2`.

These touch only `experiments/` and `tests/test_phase16.py` (R1-1 checks the variant list).

### 7.1 The experiment: entropy-weighted query-side EM purification (R2 of the directions note, probe first)

**Why this one.**
* It is the only direction that addresses the query-side term, which is the large term (≈ 25 points of oracle
  headroom against 1.5–4.3 support-side).
* It is the mathematically sound form of the paper's own central idea, "entropy-aware purification", so the paper's
  story survives.
* It has a zero-training version, so the go/no-go costs minutes.
* It works on top of *either* R1 outcome (route A or route B).

**Stage (directions §5.2).** For t = 1..T:
* `r = softmax(L^{t−1}/T_t)`, weight `w_i = 1 − H(r_i)/ln(N+1)`
* `u_c = Σ_i w_i r_ic f̂_i / 2048`
* `μ_c^t = normalise(κ₀ m_c + κ_t u_c)`
* `L^t = s_t ⟨f̂, μ^t⟩`

At training time the loss is `Σ_t CE(L^t) + β Σ_t KL(softmax L* ‖ softmax L^t)`, with L* from the oracle
prototypes of the training query, stop-gradient.

**Files it would touch** (after a new decision D-26 `PROPOSED` in `00`, a spec 02 section tagged `[DECISION D-26]`,
and tests with mutation checks per 05):
* `models/transductive.py`: new, the EM stage.
* `models/cascadeproto.py`: `refine = {none (default), em}`, `refine_steps`, `refine_weight = {entropy, none}`,
  applied after the last stage and before ADRM, with ADRM weighting the iterates.
* `train.py`: flags and a run-dir suffix `_em`.
* `pipeline/model_api.py::episode_loss`: optional per-stage CE and oracle KL, `distill_beta`, default 0.
* `eval.py`: passes the flags through.
* `experiments/p0_em_probe.py`: new, zero-training, modelled on `experiments/vipseg_init_probe.py`, wrapping VIP-Seg's
  released model read-only for its final prototypes and features.

**Measurement plan.**
1. **P0, zero training, about 1 GPU-hour.**
   * Checkpoints: VIP-Seg released **S1** (`log_s3dis_VIPSeg/log_S1_N2_K1_0.760875/checkpoint.pt`, pinned commit)
     and our S1 baseline `last.pt` (`results/seen/` run).
   * Grid: `κ_t ∈ {0.25, 0.5, 1}`, `T ∈ {1, 2, 3}`, `w ∈ {entropy, none}`, on the S1 `valid` draw.
   * Also P0.1, the oracle, on the same checkpoints.
   * Freeze the best setting, then score once on the S1 test draw and on the **S0** released checkpoint and our S0
     `last.pt` (fixed100, 1,500 episodes).
2. **R2, training, if P0 passes.**
   * Base: the R1 winner, or `stage_type=vip` with four stages under route B.
   * Arms: `refine=em`, T ∈ {1, 2}, `distill_beta ∈ {0, 0.5}`.
   * Screening: S1, 3 seeds, 9,600 episodes.
   * Final: full schedule (50 epochs × 480, batch 4) on S1, then frozen and run on **S0**, 3 seeds each, `last.pt`
     headline, `best.pt` beside it, fixed100 and random600.
   * Reference arm: R0, VIP-Seg through our loop, same seeds.

**Decision rules, fixed now.**
* **P0: go.** EM gain on the S1 test draw ≥ +1.5 on both checkpoints, and the frozen setting's gain on S0 ≥ +1.0 on
  both.
* **P0: stop.** Gain < +0.5 on both. Drop the direction and move to direction 4, base-class calibration.
* **P0: in between.** Train anyway, since learned `κ` and `s` usually beat a fixed grid, but expect the low end.
* **P0.1 oracle.** Oracle − model < 10 on the VIP-Seg checkpoints means the features, not the prototypes, bound
  performance; revisit §6.
* **R2: keep a term.** Seed-mean gain > 2 × SE of the difference **and** ≥ +1.0.
* **R2: depth.** Claim depth only if T = 2 − T = 1 ≥ +0.5 with the same test.
* **R2: headline.** Claim "beats VIP-Seg" only if the 3-seed S0 mean exceeds both 72.20 and R0's S0 mean by ≥ 2 × SE.
  Otherwise report "matches VIP-Seg".
* **R2: collapse.** Watch the per-class IoU: a gain carried by `ceiling` / `floor`-like classes while the small
  classes (board, column) fall is majority collapse (TIM §3.4). Remedy: the prior term `κ₀`, or a RePRI-style
  proportion term.

**Predicted outcome [inferred].** Route B + EM + distillation: S0 72 → 73–76, S1 76 → 76–78. Beating EDS-Net's 73.32
on S0 is plausible. Beating VIP-Seg's 76.09 on S1 by a significant margin is the hardest part, and S1 is not a
held-out fold under D-22.

## 8. Sources

**Repository** (commit `9361587`):
* `docs/research/2026-09-20_paper_vs_code_audit.md` (F1–F11, §11), `2026-09-21_reproduction_report.md`,
  `2026-09-21_gap_diagnosis.md`, `2026-09-21_external_sources_on_gap.md`, `2026-09-22_improvement_directions.md`
* `docs/spec/00_SOURCES_AND_DECISIONS.md` (D-01, D-10, D-15, D-17, D-22…D-25), `docs/CHANGELOG.md` (15e, 15k, 15x,
  15y, 16.0–16e)
* `experiments/run_r1.sh`, `experiments/summarize_r1.py`, `experiments/diag_short.py:6-7,53-60,164-183`,
  `experiments/summarize.py:21-28`, `train.py:58-68`
* `results/seen/`, `results/rescore/`, `results/vipinit/`, `results/b1/`, `results/leak/`,
  `results/phase15_full/SUMMARY.md`

**The paper:** Wang et al., *CascadeProto*, local copy `C:\Users\USER\Downloads\10069.pdf`: §3.5–3.7 (Eq.24–28),
§4.1–4.4, Tables 2–6. Official repository https://github.com/changshuowang/CascadeProto (README only).

**VIP-Seg:** https://github.com/changshuowang/VIP-Seg_NeurIPS2025 at `28aedc5`. Tree listing via the GitHub API:
released checkpoints for every S3DIS setting, including `log_s3dis_VIPSeg/log_S1_N2_K1_0.760875/checkpoint.pt`.
NeurIPS poster https://neurips.cc/virtual/2025/poster/115811 (73.50 / 74.92).

**Literature:**
* [EDS] EDS-Net, AAAI 2026, https://ojs.aaai.org/index.php/AAAI/article/view/37929 (Table 5, Fig. 2; via the
  directions note §4.6)
* [COSeg] An et al., CVPR 2024, https://arxiv.org/abs/2403.00592 (§3.2, Tables 1–3)
* [MM-FSS] An et al., ICLR 2025, https://arxiv.org/abs/2410.22489 (Tables 1–3; COSeg† on 2D-aligned backbone, §4.2)
* [QHP] Zhou et al., *Query-aware Hub Prototype Learning for Few-Shot 3D Point Cloud Semantic Segmentation*,
  arXiv Dec 2025, https://arxiv.org/abs/2512.08253 (COSeg setting, 20,480 points, backbone pretrained per fold for
  100 epochs; Table 1: 2w1s S3DIS 38.86 / 37.84 / 38.35 against COSeg 37.17 / 37.03 / 37.10; Table 3: HPG +1.44,
  PDO +2.96, 1w1s S0)
* [DA-FSS] Bian, Xu, *Rethinking Multimodal Few-Shot 3D Point Cloud Segmentation: From Fused Refinement to
  Decoupled Arbitration*, arXiv 2601.01456, https://arxiv.org/html/2601.01456v2 (S3DIS 2w1s 44.62 against MM-FSS
  44.35; Stratified Transformer; +0.5 to +1.2 over MM-FSS)
* [AttMPTI] Zhao et al., https://arxiv.org/abs/2006.12052 (§4.2 pretraining of the feature extractor, 100 epochs;
  Table 1)
* [QGE] https://arxiv.org/abs/2308.03177 (Tables 1, 5)
* [DPA] https://arxiv.org/abs/2401.16051 (Table 3)
* [Seg-PN] https://arxiv.org/abs/2404.04050
* [TaylorSeg] https://arxiv.org/abs/2504.02454
* [DyPolySeg] https://raw.githubusercontent.com/mlresearch/v267/main/assets/wang25aa/wang25aa.pdf
* [SSP] https://arxiv.org/abs/2207.11549
* [TIM] https://arxiv.org/abs/2008.11297
* [RePRI] https://arxiv.org/abs/2012.06166

**Searched and not established.**
* A published FS-3DSeg result using PTv3 or Sonata as the backbone in the episodic 2,048-point protocol: none found.
  GFS-VL (CVPR 2025) addresses *generalized* few-shot segmentation, a different task, and was not read in full.
* DPR-Net (ICML 2026, the paper's [24]): not retrieved.
