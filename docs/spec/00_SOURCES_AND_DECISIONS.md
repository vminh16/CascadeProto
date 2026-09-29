# 00_SOURCES_AND_DECISIONS: Source Hierarchy & Implementation Decision Log

* **Status:** Created 2026-09-17 (Phase 0). Specs `01`–`05`, `AGENTS.md` and `README.md` were rewritten against this file on 2026-09-17 (Phases 1–7). The code has not been updated yet.
* **Audit that motivated this file:** [`docs/research/paper_vs_repo_audit.md`](../research/paper_vs_repo_audit.md).
* **Rule:** Every normative statement in `docs/spec/*` must carry a source tag (Section 2.3). A statement with no traceable source must be deleted, not kept "just in case".

---

## 1. Why this file exists

The pre-rewrite specs mixed paper content with invented formulas (e.g. the two-hop cross-attention, weighted cross-entropy, a `sqrt` MMD, class splits the paper never lists). The code followed the specs, so the code inherited those inventions. This file fixes the order of authority and records every point where the paper is silent, ambiguous or self-contradictory, together with the chosen interpretation and its evidence.

---

## 2. Source hierarchy

### 2.1 Levels of authority

| Level | Source | Pinned version | Authoritative for |
| :--- | :--- | :--- | :--- |
| **L1** | Paper: *CascadeProto: Cascaded Cross-Modal Prototype Purification via Entropy-Aware Learning for Few-Shot 3D Point Cloud Segmentation* (Wang et al.) | Local file `10069.pdf` (not committed; ask the maintainer for a copy) | Everything the paper states explicitly |
| **L2** | Official VIP-Seg repository `github.com/changshuowang/VIP-Seg_NeurIPS2025` | Commit `28aedc5093c0d386d526864c49505ae6921b1600` (2026-02-09) | (a) Inherited parts: backbone, dataloaders, preprocessing, evaluation metric, run-script conventions. (b) Interpretive evidence where L1 is ambiguous, because the paper states "The shared encoder is VIP-Seg [23]" (§4.1) and both papers share the first author |
| **L3** | Decision log: one file per decision in `docs/decisions/`, indexed in Section 4 | `docs/decisions/` | Points where L1 is silent/contradictory and L2 does not settle them |

### 2.2 Conflict rules

1. **L1 beats L2 beats L3.** A decision may never contradict an explicit L1 statement unless L1 contradicts itself; in that case the decision must quote both conflicting L1 statements.
2. **L2 is evidence, not gospel.** Known defects in L2 are not copied (Section 5.3). Where we deviate from L2 for the sake of L1 fidelity, the decision says so.
3. **Not sources:** the pre-rewrite versions of `docs/spec/01`–`05`, `README.md`, `AGENTS.md`, and any code in this repo. They are derived artefacts.
4. **Official CascadeProto code is unavailable.** `github.com/changshuowang/CascadeProto` contains only a README reading "We will release it soon." (checked 2026-09-17). When it is released, re-check every `PROPOSED` and `LOCKED` decision against it.

### 2.3 Source tags

Use exactly one or more of these tags at the end of every normative line in `docs/spec/*`:

| Tag | Meaning | Example |
| :--- | :--- | :--- |
| `[PAPER §x]`, `[PAPER Eq.n]`, `[PAPER Tab.n]`, `[PAPER Fig.n]` | Stated in L1 | `D = 128 [PAPER Eq.2]` |
| `[VIPSEG path:line]` | Taken from L2 at the pinned commit | `MaxPool1d(32) [VIPSEG models/vipseg.py:213]` |
| `[DECISION D-nn]` | Interpretation recorded in Section 4 | `A ∈ R^{(N+1)×D×D} [DECISION D-01]` |

---

## 3. Paper-explicit facts that override the pre-rewrite specs

These need no decision. They are listed because the old specs or code contradicted them.

| Topic | Paper statement | Old spec / code said |
| :--- | :--- | :--- |
| Segmentation loss | "The segmentation loss is the standard cross-entropy applied to the dynamically routed final logits L_final" [PAPER Eq.27] | CE weighted by `w_cls` (spec 05) |
| Where `w_cls` lives | "we apply class-specific weights w_cls = [0.8, 1.0, …, 1.0], yielding P_weighted = P_attended ⊙ w_cls" [PAPER §3.4] | Also applied inside CE |
| MMD form | "MMD(P, Q) = ‖(1/\|P\|)Σφ(x) − (1/\|Q\|)Σφ(y)‖²_H" — the **squared** RKHS norm [PAPER Eq.7] | `sqrt(max(·, ε))` (spec 03), per-class average (spec 05) |
| GMMN weights, kernel | `L_GMMN = 0.1·MMD(P_modal^bg, P_point^bg) + 1.0·MMD(P_modal^fg, P_point^fg)`, σ ∈ {2,5,10,20,40,80} [PAPER Eq.7–8] | Unchanged |
| Projection φ | "Q′ = φ(F^q) …, S′ = φ(F^s) …, where φ is a 1 × 1 convolution" — one symbol, one shared operator [PAPER Eq.13] | Two separate projections |
| Support mask | "M_fg^(k) = {i \| Y_i^s = 1, class(i) = k}", binary mask Y^s [PAPER Eq.3, §3.1] | `mask == k` on a flattened mask |
| Encoder | "The shared encoder is VIP-Seg [23] with feature dimension D = 128" [PAPER §4.1] | Silent fallback to a `TransBlock` when `mamba_ssm` is missing |
| Input | "S3DIS contains RGB point clouds … 2048 randomly sampled points per block" [PAPER §4.1] | xyz only, RGB padded with zeros |
| Hyper-parameters | "T = 4, … d = 72, … each LMA projects CLIP embeddings from R^512 to R^128 via a two-layer MLP. We train with AdamW at an initial learning rate of 10^−3, weight decay 0.1, and a StepLR scheduler that halves the rate every 10 epochs. The batch size is 4 episodes, trained for 50 epochs on S3DIS and 30 epochs on ScanNet." [PAPER §4.1] | Batch size ignored; ScanNet used S3DIS classes and 50 epochs |
| Gate constants | ε = 10⁻⁸, θ learnable, initialised to 0.5, `g_i = σ(2(θ − H_i))` [PAPER Eq.10–11]; each EPPM gates independently [PAPER §3.5] | Unchanged |
| Diffusion constants | τ = 0.5, α = 0.5 [PAPER Eq.15–18] | Unchanged |
| Routing | `w_gate = softmax(W_g · AvgPool(F^q)) ∈ R^T`, `W_g ∈ R^{T×D}` [PAPER Eq.24–25] | Unchanged |
| Loss balance | λ = 1.0 [PAPER Eq.26] | Unchanged |

---

## 4. Decision log (index)

Each decision is one file in [`docs/decisions/`](../decisions/README.md), kept verbatim: problem, evidence, what it
does, rules fixed before the run, amendments and outcome. This section only maps the IDs that `[DECISION D-nn]`
tags cite to their files, with a one-line preliminary result; the file is the source (L3), not this table.
A new decision is a new file plus one row here, written before any code (AGENTS.md §2).


**Status legend**

| Status | Meaning |
| :--- | :--- |
| `LOCKED` | Confirmed by the maintainer on 2026-09-17. Change only with new L1/L2 evidence. |
| `PROPOSED` | Default chosen during Phase 0, awaiting maintainer review. Specs may use it but must keep the ablation flag. |

Each ablation flag named below is a requirement on the future CLI/config, not an existing option.


| ID | Decision | Status | Preliminary result | File |
| :--- | :--- | :--- | :--- | :--- |
| D-01 | Cross-attention refinement (Eq.13-14) | LOCKED | Interprets the paper's cross-attention equations as VIP-Seg's channel-channel correlation (not point-point attention), since the literal point matrix product is dimensionally invalid; scale by √d = √72 as printed in Eq.14, differing from VIP-Seg's own √128. | [D-01](../decisions/D-01_cross-attention-refinement-eq-13-14.md) |
| D-02 | Target of entropy gating | LOCKED | Per the paper audit, gating applies to point-level features (F^s, F^q), not the prototype. `gate_target=features` is implemented and no longer raises, but the default stays `prototype` until a VM run separates them. | [D-02](../decisions/D-02_target-of-entropy-gating.md) |
| D-03 | Shared phi module for query and support | LOCKED (paper-explicit) | Query and support use the same phi module, as printed in Eq.13; folded into D-01 and kept only as an ID that the audit refers to. | [D-03](../decisions/D-03_shared.md) |
| D-04 | GMMN sample sets and gradient flow | LOCKED | Uses the squared MMD of Eq.7. P^fg is the N foreground rows taken jointly as one set, not pairwise comparisons; P^bg is the single background row. P_point is not detached, so GMMN gradients reach the backbone. | [D-04](../decisions/D-04_gmmn-sample-sets-and-gradient-flow.md) |
| D-05 | Fuses both sources (section 3.2) | LOCKED | One modality per run: E_fused is the adapted embedding of the selected modality; "fuses both sources" is read as combining point and modal prototypes via Eq.9. Generator input is z concatenated with E_fused (2D=256) through a three-layer MLP. | [D-05](../decisions/D-05_fuses-both-sources-in-3-2.md) |
| D-06 | Noise z at inference | LOCKED | Training samples z~N(0,I) per forward pass; evaluation fixes z=0, making predictions deterministic. | [D-06](../decisions/D-06_noise-z-at-inference.md) |
| D-07 | Train/test split protocol | LOCKED | Primary protocol is the inherited loader's class-only split, not an area-based split, since VIP-Seg's released S0 result (72.20) was produced with the class-only loader. | [D-07](../decisions/D-07_train-test-split.md) |
| D-08 | Evaluation protocol and metric | LOCKED | Primary metric is VIP-Seg's protocol: 100 fixed episodes per class combination, TP/FP/FN accumulated per class, mIoU over foreground classes only; validated against VIP-Seg's own 72.20 in Table 6. | [D-08](../decisions/D-08_evaluation-protocol-and-metric.md) |
| D-09 | Params/FLOPs target (Table 6) | LOCKED | Table 6 (2.88M/8.86G vs VIP-Seg's 2.76M/8.48G) is not a hard acceptance criterion; only the inherited encoder (2.37M ± 0.01M) and encoder+head (~2.57M) are enforced. | [D-09](../decisions/D-09_params-flops-target-table-6.md) |
| D-10 | Logit form (Eq.23) | LOCKED | Uses a plain dot product exactly as printed in Eq.23, with no temperature and no L2-normalisation of P_point, even though VIP-Seg L2-normalises its initial prototypes. | [D-10](../decisions/D-10_logit-form-eq-23.md) |
| D-11 | Shape of the fusion weight w | LOCKED | Uses one weight pair per query, not per class: w = softmax(f_fusion(mean over classes of [P_cross; P_diffuse])), a two-layer MLP 2D to D to 2, matching the printed w in R^2. | [D-11](../decisions/D-11_shape-of-the-fusion-weight-w.md) |
| D-12 | Epoch definition and augmentation | LOCKED (epoch) / LOCKED (augmentation) | Matches VIP-Seg's 24,000 total training episodes: S3DIS 50 epochs x 480 episodes/epoch (120 steps, batch 4); ScanNet 30 x 800 (200 steps); StepLR halves every 10 epochs. Measured: VIP-Seg's own schedule gives baseline 0.4901 vs 0.4908 under this decision. | [D-12](../decisions/D-12_epoch-definition-augmentation.md) |
| D-13 | Modality scope, CLIP variant, prompts | LOCKED | Implements text first; image and audio are deferred and must fail loudly. CLIP variant defaults to ViT-B/16, embeddings L2-normalised and cached. Foreground prompt is fixed; background prompt is an assumption not in the paper. | [D-13](../decisions/D-13_modality-scope-clip-variant-prompts.md) |
| D-14 | ReLU before EPPM, diffusion degeneracy | LOCKED | Keeps VIP-Seg's ReLU feature head and implements Eq.15-18 literally, documenting that non-negative features make P_diffuse degenerate to entries in [0.25, 0.5] with no class index. | [D-14](../decisions/D-14_relu-before-eppm-and-diffusion-degeneracy.md) |
| D-15 | Model selection | LOCKED | Primary selection validates every 10 epochs on the valid episode set and keeps the best checkpoint, matching VIP-Seg's protocol for comparability; the last-epoch checkpoint's test mIoU is always also logged. | [D-15](../decisions/D-15_model-selection.md) |
| D-16 | Layer details not given in the paper | LOCKED | Fixes layer details absent from the paper: adapter Linear(512 to D) with Dropout(p=0.1); SE block reduction r=4; entropy probability clamped to [10⁻⁷, 1 − 10⁻⁷]; stage parameters not shared across the T stages. | [D-16](../decisions/D-16_layer-details-not-given-in-the.md) |
| D-17 | Meaning of the ablation switches (Table 4-5) | LOCKED | Maps Table 4's cumulative rows to four switches (use_lma, num_stages, use_gate, use_adrm). Observed increments +1.21 (LMA), +1.42 (gate), +2.06 (cascade), +0.56 (ADRM) match the text, but the paper's baseline (81.28 Avg) sits far above VIP-Seg's own 74.15 Avg, unexplained. | [D-17](../decisions/D-17_meaning-of-the-ablation-switches-table.md) |
| D-18 | Degeneracy of the cross-attention output | LOCKED (as an ablation flag) | Cross-attention escapes the uniform-softmax limit on its own but adds almost nothing: full is 1.2 points above baseline_l2 (full-baseline_l2=-0.0033, t=-0.36 across seeds), while VIP-Seg's PEM/PDM add about 17 points; the layernorm probe (0.5134) is rejected as default. | [D-18](../decisions/D-18_degeneracy-of-the-cross-attention-output.md) |
| D-19 | Channel-preserving term in Eq.19 | PROPOSED, beyond the paper | Adds a channel-preserving term (psi of the gated prototype) to P_combined, like VIP-Seg's proto_self; CPU analysis did not confirm it — class-varying energy share moves only from 3.9% to 4.2%, versus VIP-Seg's PEM at 6.9-8.9%. Off by default, an experiment not a claim. | [D-19](../decisions/D-19_a-channel-preserving-term-in-eq.md) |
| D-20 | Initialising from VIP-Seg's trained encoder | PROPOSED, diagnostic only | Diagnostic switch --init_from_vipseg copies VIP-Seg's trained encoder before training, testing whether the paper's baseline (82.72 on S0, above VIP-Seg's own 72.20) could stem from non-scratch init (ours from scratch scores 49.08); contradicts the paper's pre-training-free claim, so diagnostic only. | [D-20](../decisions/D-20_initialising-from-vip-seg-s-trained.md) |
| D-21 | Test classes seen during training | PROPOSED, diagnostic only | Diagnostic (train_classes=all) leaks test classes into training. Scoring a checkpoint on its own training classes gives last.pt 77.32 (S0) / 71.58 (S1) against the paper's 82.72 / 79.83, matching within 5-8 points; diagnostic only, not a result configuration. | [D-21](../decisions/D-21_test-classes-seen-during-training.md) |
| D-22 | Protocol guard against seen-class scoring | PROPOSED | eval.py now scores a checkpoint only on the test classes of its own training fold. Amendment: VIP-Seg's published 76.09/72.20 are best of 12 validations (S1 log 72.84 last vs 75.63 selected, S0 log 68.97 vs 72.94); every run now reports both best.pt and last.pt. | [D-22](../decisions/D-22_protocol-guard-against-seen-class-scoring.md) |
| D-23 | One S-prime per episode vs per class slot | PROPOSED, to be measured | Tests a pooled single support correlation vs class_slots (default), matching the literature (DPA +13.7 mIoU; ablations +15.4/+15.2). Measured on S1: pooled 0.5339±0.0015 vs class_slots 0.5305±0.0070, +0.33 points (t=+0.65), not significant — the support reading is not the cause of the gap to VIP-Seg's head. | [D-23](../decisions/D-23_one-for-the-episode-instead-of.md) |
| D-24 | EPPM-S: a stripped purification stage | PROPOSED, beyond the paper | Negative (R1): EPPM-S scores 0.4927 against the printed stage's 0.5305 and no-stage's 0.5602 — −3.79 and −6.75 points, 19.26 below one VIP-Seg PEM. Kept only as an off-by-default ablation. | [D-24](../decisions/D-24_eppm-s-a-stripped-purification-stage.md) |
| D-25 | VIP-Seg's PEM/PDM as cascade stages | PROPOSED, reference only | One VIP-Seg PEM on our prototypes scores 0.6852 on S1, +15.47 over the printed stage and +12.50 over no stage. Rule R1.4 fires: phase 16 continues on route B, adding to VIP-Seg's head. | [D-25](../decisions/D-25_vip-seg-s-pem-pdm-as.md) |
| D-26 | Query-side entropy-weighted EM prototype refinement | PROPOSED, beyond the paper | P0.2 stop: frozen ssp_k0.5_T1 gains only +0.37 on VIP-Seg S1 and loses −1.21 on VIP-Seg S0; with query labels the same update gains +8.42 to +21.76, real but unreachable headroom. Next direction: base-class calibration. | [D-26](../decisions/D-26_query-side-entropy-weighted-em-refinement.md) |
| D-27 | Base-class calibration of the background | PROPOSED, beyond the paper | P1.2 stop: training-free base calibration loses on all four checkpoints (VIP-Seg S1 −0.16, S0 −0.07); AUC 0.716/0.649 (VIP-Seg) vs 0.494/0.629 (ours), false-foreground share 10–27%. Training-free form closed; signal kept as D-28's filter. | [D-27](../decisions/D-27_base-class-calibration-of-the-background.md) |
| D-28 | EM refinement with base-margin filter | PROPOSED, beyond the paper | P2.0 fails, P2.2 stop: the false share needed to fall to ≤ 0.75 of baseline but only reached 0.85×; best mIoU falls monotonically with r (+0.51 to +0.47); AUC 0.65–0.72 removes true and false mass at nearly the same rate. Closed. | [D-28](../decisions/D-28_em-refinement-with-a-base-margin.md) |
| D-29 | Oracle-direction distillation of pairwise decisions | PROPOSED, beyond the paper | R2.2 stop: d29 − r0 is +0.06 on fixed100 (CI crosses 0) and negative on random600 draws, despite the objective transferring (logit-pair cosine 0.771→0.843). The oracle headroom (+14.43) is a transductive information gap, not a training-signal gap. Closed. | [D-29](../decisions/D-29_oracle-direction-distillation-of-the-pairwise.md) |
| D-30 | Route B's base on VIP-Seg's schedule (E1) | PROPOSED, beyond the paper | E1.1 adopt: matching VIP-Seg's update schedule lifts E1 last to 73.20 on S1 fixed100 (+2.99 over r0); best is 75.05 vs VIP-Seg's released 75.36 (−0.31). VIP-Seg's update count becomes route B's schedule for all later arms. | [D-30](../decisions/D-30_route-b-s-base-on-vip.md) |
| D-31 | P3: prototype error location and text prior | PROPOSED, beyond the paper | P3.0 fails: text prior stops, 0 of 192 arms gain on valid; frozen arm −0.00 on fixed100, oracle-weighted arm only +0.70, best text-only accuracy 0.62. Part A: support 49.27 → E1 73.20 → oracle 85.93 (head recovers 0.65 of the gap). | [D-31](../decisions/D-31_p3-where-the-prototype-error-sits.md) |
| D-32 | P4: background prototype contamination causality | PROPOSED, beyond the paper | P4.1 not causal: removing the other way's points from the background row gains only +0.09 (below the +1.0 bar); label-free purification gains +0.02. Row-wise oracles: bg-only −26.07, fg-only −9.05, all rows +13.71 — a joint shift, not row contamination. Line stops. | [D-32](../decisions/D-32_p4-is-the-background-prototype-contaminated.md) |
| D-33 | N1: point-level support-query attention neck | PROPOSED, beyond the paper | N1.2 stop: the neck never opened (gate ended at α = 0.0023); neck − ctl −0.17 on fixed100, negative on random600 too. Restarting AdamW's moments dropped both arms about 2 points versus E1 (73.04 → ~59 at epoch 5). | [D-33](../decisions/D-33_n1-a-point-level-support-query.md) |
| D-34 | N2: neck retrained from scratch on E1's schedule | PROPOSED, beyond the paper | N1.2 stop: trained from scratch with α initialised at 0.1, N2 − E1 is −0.66 on fixed100 last (negative on all random600 draws); α settled at 0.060 and the oracle gap did not shrink (12.73 → 13.48). A hidden +1.0 gain is implausible. | [D-34](../decisions/D-34_n2-the-neck-of-d-33.md) |
| D-35 | P5: split E1's oracle gap by condition | PROPOSED, beyond the paper | Full P5 not run: C3 found a position shortcut. Swapping query block order collapses mIoU: E1 73.20→0.92, VIP-Seg 75.36→0.87; relabelled-by-position share 0.913 (E1). Every D-25–D-34 result rests on this shortcut; D-36/D-37 trace it. | [D-35](../decisions/D-35_p5-split-e1-s-oracle-gap.md) |
| D-36 | C4: is cross-term reshape sole position carrier | PROPOSED, diagnostic | No outcome recorded in this file. It designs a check of whether VIP-Seg's cross-term reshape is the sole carrier of the query-position shortcut (rule C4.1), to run before D-37; its measured result (C4.1 holds) is reported in D-37's outcome. | [D-36](../decisions/D-36_c4-is-the-cross-term-reshape.md) |
| D-37 | 2x2: scrambled/clean head x query order | PROPOSED, beyond the paper | C4.1 holds (reshape is the sole position carrier). D37.3: the clean head beats scrambled by +1.58 on fixed100 (CR 54.84 vs VR 53.26); the shortcut is worth +19.94 mIoU. The clean head with random query order becomes the base for all later arms. | [D-37](../decisions/D-37_a-2-2-head-scrambled-clean.md) |
| D-38 | P6: clean base's prototype gap location | PROPOSED, beyond the paper | P6.1 joint (oracle bg +6.14, fg +16.31, all +25.12 over U 55.74). P6.2/P6.3 both go: background self-support +1.19, 3-component k-means background +1.32; foreground self-support hurts (−0.35). U alone beats the trained model by +0.9. | [D-38](../decisions/D-38_p6-where-the-clean-base-s.md) |
| D-39 | Trained self-support prototypes on the clean base | PROPOSED, beyond the paper | D39.1: the combined background rule beats km3 alone (+0.51). D39.2 stop: trained self-support does not help (A1 − A0 −0.66). D39.3: CR stays the base; CR + U + both scores 57.63 on fixed100 (+2.79 over CR's head), inference only. | [D-39](../decisions/D-39_trained-self-support-prototypes-on-the.md) |
| D-40 | P7: label propagation on query point graph | PROPOSED, beyond the paper | P7.2 adopt: label propagation gains +0.92 on fixed100 (58.55), positive on all random600 draws. P7.6 unexplained: the gain is mostly precision (removing false positives) — fixed points' homophily (0.944) is not above broken points' (0.947). | [D-40](../decisions/D-40_p7-label-propagation-on-the-query.md) |
| D-41 | P8: remaining gap split by sampling condition | PROPOSED, beyond the paper | P8.1 and P8.3 both hold (g_other +5.25, g_own +13.71, g_fg +16.12); P8.2 density is causal (φ 1.001). Both training arms are admissible; instance-alignment training runs first (larger bound +13.71), condition-balanced second. | [D-41](../decisions/D-41_p8-the-clean-base-s-remaining.md) |
| D-42 | Stacked architecture from VIP-Seg's weaknesses | PROPOSED, beyond the paper | Closed 2026-09-28 without a further run: M1 stops (D-43), M2 stops (D-45), M3 is kept only as the isotropic mean rule (+1.11; base-class covariance adds nothing, D-46), M4 LP is kept (+0.92, D-40); the neck, N, H and C fail P9. | [D-42](../decisions/D-42_a-stacked-architecture-derived-from-vip.md) |
| D-43 | M1: density-invariant encoder, first stack block | PROPOSED, beyond the paper | D43.1 fails (other-condition deficit 0.588 → 0.432, limit 0.294; φ 1.14) and D43.2 fails (leak-free stack −0.30 [−1.41, +0.85]). D43.4 stop: CR stays the base; stack on fixed100 −9.31, model −6.89. | [D-43](../decisions/D-43_m1-a-density-invariant-encoder-the.md) |
| D-44 | P9 after D-43: where density still enters | PROPOSED, beyond the paper | D44.1/D44.1b: the encoder branch is closed, every ψ within ±0.02 — the ball count is not the density pathway. D44.2: only P9.3 (metric head) is admissible, via LDA oracle 96.04 vs cosine oracle 80.84 vs label-free LDA 35.44; other modules fail their preconditions. | [D-44](../decisions/D-44_p9-after-d-43-where-density.md) |
| D-45 | M2: VICReg variance/covariance vs feature collapse | PROPOSED, beyond the paper | D45.3: VICReg un-collapses the features (participation ratio 42.45/95.25 vs CR's 5.56) but D45.2 fails both arms — U worsens by −5.20 (A) and −11.34 (B) on fixed100. The collapse is shared by every S1 checkpoint, including VIP-Seg's (6.39). M2 line stops. | [D-45](../decisions/D-45_m2-vicreg-s-variance-and-covariance.md) |
| D-46 | P10/M5: base-class prototype gap and alignment | PROPOSED, beyond the paper | P10.1 holds (base-class gap +14.76) but M5 fails D46.1 in both arms (valid gaps 25.15 / 23.75, bar 22.88): D46.4 stop. P10.4 holds (+1.46; isotropic +1.11). P10.5: density-driven bias (own b̂ ≈ 0, other 0.4158); foreground oracle 72.44 at k = 1. | [D-46](../decisions/D-46_p10-and-m5-does-training-leave.md) |
| D-47 | Image and audio modalities at the class level (Fig. 1) | ACCEPTED, maintainer request | Implemented and built (inputs only, no score): audio 10/14 transcripts exact (background, ceiling, wall, clutter misheard; = text elsewhere); image rows 0.25–0.31 from text rows (CLIP gap), 12/14 nearest their own text row, more spread than text (0.821 vs 0.900). | [D-47](../decisions/D-47_image-and-audio-modalities-class-level.md) |
| D-48 | Correlation architecture: base-class exclusion, multi-prototypes, correlation neck, per-episode modality weights | PROPOSED, beyond the paper | P11 on CR: only LP kept (Shapley +1.15); exclusion, text, OT fail (the base learner reads density: g 0.57 on dense novel foreground); P11.4 holds by rule (+2.07 over U) with leak-free −5.7; P11.6 unmixing −2.86 (own-class abundance of sparse points −0.018 at the median). Arm A pending the maintainer. | [D-48](../decisions/D-48_correlation-architecture-with-base-exclusion.md) |
| D-49 | Condition-balanced training (queries whose target is also sparse) | PROPOSED, beyond the paper | not run: CB = CR + each query block thinned in its own class with q 0.5; rules on the sparse points' own-class abundance, leak-free and standard draws. | [D-49](../decisions/D-49_condition-balanced-training.md) |

---

## 5. Official VIP-Seg files: restore, reuse, avoid

### 5.1 Reference implementations restored from L2

Restored byte-identical on 2026-09-18 (phase 8a); their git blob SHA-1s match the pinned tree. CascadeProto does not import them.

| File at pinned L2 commit | Purpose |
| :--- | :--- |
| `runs/training_free.py` (`evaluate_metric`, `test_few_shot`) | Evaluation metric of D-08 |
| `runs/evaluate.py` | Evaluation entry point pattern |
| `runs/training.py` | Training loop, validation and checkpointing pattern (D-15) |
| `main.py` | CLI argument names and defaults (augmentation, `pc_npts`, `pc_attribs`) |
| `scripts/vipseg_s3dis.sh`, `scripts/vipseg_scannet.sh`, `scripts/vipseg_eval_*.sh` | Reference hyper-parameters |
| `models/vipseg.py` (replaces a 7-line local stub) | Baseline sanity run (05 §4) and source of D-01 / D-10 evidence |

`.gitignore` no longer ignores `runs/`; `log_*/` and checkpoints stay ignored.

### 5.2 Inherited files present locally

Every file below is byte-identical to L2 (CRLF normalised), checked by test ENV-3 [05 §3.1]: `dataloaders/{loader,s3dis,scannet}.py`, `preprocess/{collect_s3dis_data,collect_scannet_data,room2blocks}.py`, `utils/{checkpoint_util,cuda_util,logger}.py`, `models/{encoder,mamba_block,model_utils,vipseg_learner}.py` and the Python files of §5.1. The local `TransBlock` and Python-FPS fallbacks and the `.cuda()`→`device` edits in `models/{encoder,mamba_block,model_utils}.py` were removed on 2026-09-18 (phase 8b); the model runs on a single CUDA GPU.

**Environment deviation.** L2 vendors `mamba_ssm` 2.0.4 [VIPSEG mamba/mamba_ssm/__init__.py:1] and documents PyTorch 1.13.1 + CUDA 11.7 [VIPSEG README.md]. This repository targets PyTorch 2.7.1 + CUDA 12.8 (RTX 50-series, [PAPER §4.1]) with `mamba-ssm` 2.2.6.post3; the encoder only uses `mamba_simple.Mamba`. Test ENC-1 (2.37M parameters) checks that the encoder is unchanged.

### 5.3 Known L2 defects and traps not to copy

| Item | Evidence | Rule |
| :--- | :--- | :--- |
| Cross-correlation reshape interleaves classes and projection filters | [VIPSEG models/vipseg.py:288-291]; numerical check 2026-09-17 | Use the clean form of D-01 |
| `TransBlock` uses `nn.MultiheadAttention` with default `batch_first=False` on `[B, G, C]` input | [VIPSEG models/mamba_block.py:20] | Never fall back silently; require `mamba_ssm` |
| Loader defaults `num_point=4096`, `pc_attribs='xyz'` | [VIPSEG dataloaders/loader.py:118,232] | Always pass `num_point=2048`, `pc_attribs='xyzrgbXYZ'` explicitly |
| Validation episodes are drawn from test classes | [VIPSEG runs/training.py] | Allowed only under D-15, always logged alongside `last` |
| `requirements.txt` omits `mamba_ssm`, `pointnet2_ops`, CLIP | [VIPSEG requirements.txt] | List every runtime dependency explicitly |
| `pointnet2_ops_lib/setup.py` sets `TORCH_CUDA_ARCH_LIST="3.7+PTX;…;9.0"` | [VIPSEG pointnet2_ops_lib/setup.py:19] | nvcc 12.x rejects `compute_37` and the list lacks `sm_120`; fix the arch list when building on CUDA 12.8 (phase 14) |

---

## 6. Traceability: pre-rewrite spec errors → fix

IDs `S1`–`S17` refer to Section 4 of the audit.

| Audit ID | Resolved by | Fixed in phase |
| :--- | :--- | :--- |
| S1 invented S0/S1 class lists | D-07 | 3 (spec 04), 7 (README) |
| S2 "≥ 50 points" threshold | L2 `max(int(N·0.05), 100)` [VIPSEG dataloaders/s3dis.py:55] | 3 |
| S3 two-hop cross-attention | D-01 | 1 (spec 02), 2 (spec 01) |
| S4 weighted CE | Section 3 | 1, 5 (spec 05) |
| S5 three MMD definitions | Section 3, D-04 | 1, 4 (spec 03), 5 |
| S6 probability clamp presented as paper math | Tag as implementation detail | 1 |
| S7 xyz-only input | Section 3 | 2, 3, 5 |
| S8 ScanNet block count misread | L1 §4.1 wording | 3, 7 |
| S9 per-episode mIoU | D-08 | 3, 5 |
| S10 CLIP variant / Whisper claims | D-13 | 4 |
| S11 background prompt stated as fact | D-13 | 4 |
| S12 meaningless notes column | Delete | 3 |
| S13 "Area 5" attributed to `s3dis.py` | D-07 | 3, 6 (AGENTS), 7 |
| S14 dry run "on real S3DIS dataloader" | Rewrite verification plan | 5, 6 |
| S15 "never rewrite inherited evaluation routines" that were never copied | Section 5.1 | 6 |
| S16 "Official PyTorch implementation" | Unofficial re-implementation | 7 |
| S17 Table 6 omits QGPA row | Restore row | 7 |

---

## 7. Change log

| Date | Change |
| :--- | :--- |
| 2026-09-17 | Created. D-01, D-03, D-06, D-07, D-08 and the epoch part of D-12 locked by the maintainer; D-01 revised from "two-hop" to "channel correlation" after reading L2; D-12 wording corrected (480 episodes, not steps, per epoch). All other decisions `PROPOSED`. |
| 2026-09-17 | Phase 1: added D-16 (layer details not given in the paper). |
| 2026-09-17 | Phase 2: added D-17 (meaning of the ablation switches). |
| 2026-09-17 | Verification run (executable reference of spec 02; pinned VIP-Seg loader and metric on a synthetic dataset): corrected D-14 (conditions for `c_unique = 0`; class-independence comes from Eq.15–18 themselves), D-01 (shot averaging is our choice; class slots are a VIP-Seg construct, flagged), D-07 (baseline protocol evidence), D-12 (shift augmentation is a no-op for the encoder input), D-16 (reading of `dim=1`). |
| 2026-09-17 | Phases 3–7: specs 04, 03, 05, AGENTS.md and README.md rewritten; D-09 corrected with VIP-Seg's logged parameter breakdown (encoder 2.37M, VIP module 0.19M); D-13 extended (L2-normalised embeddings, verbatim class names); §5.1 notes the `.gitignore` conflict for `runs/`. |
| 2026-09-18 | Phase 8: §5.1 files restored from L2; §5.2 rewritten (all inherited files identical, fallbacks removed, `mamba-ssm` version deviation recorded); §5.3 adds the `pointnet2_ops` arch-list trap. |
| 2026-09-18 | Gate G0 passed on a GCP VM (NVIDIA L4 sm_89, driver 580, Ubuntu 22.04, Python 3.10, torch 2.7.1+cu128, mamba-ssm 2.2.6.post3, transformers 4.57.1): 27/27 in `tests/test_environment.py`, 34/34 in G1. `pointnet2_ops` built with the arch list patched locally on the VM (not committed). |
| 2026-09-18 | Phase 9: `pipeline/` (episodes, model contract, VIP-Seg metric), `train.py`/`eval.py` on real episodes, `preprocess/prepare_s3dis.py`; D-08 gains the shared episode-cache tags (04 §6.1); EVAL tests move to the GPU gate (05 §3.7). |
| 2026-09-18 | S3DIS prepared with `preprocess/prepare_s3dis.py`: 272 rooms, 7,547 blocks (the AttMPTI/VIP-Seg count). DATA-0…4 pass on it, including the 1,500 cached S0 2-way 1-shot test episodes. |
| 2026-09-18 | Pipeline sanity check (05 §4): VIP-Seg's released S0 2-way 1-shot checkpoint scores 0.719687 through `eval.py --model vipseg` on our data and metric (VIP-Seg log 0.722026, [PAPER Tab.6] 72.20). |
| 2026-09-19 | D-05 locked by the maintainer (one modality per run, `E_fused := E_adapted^(m)`, generator input `[E_fused; z]` of width 2D). D-06: `eval_noise=mean_of_M` raises until M is chosen. |
| 2026-09-19 | D-01 step 5 closed by the maintainer: one `A` per (query, class slot, shot) as in VIP-Seg. D-01, D-02, D-14: the flags `two_hop`, `gate_target=features`, `diffusion_input=pre_relu` raise until implemented. D-17: `num_stages = 1` gives `L^1` with or without ADRM. |
| 2026-09-20 | Paper audit (`docs/research/2026-09-20_paper_vs_code_audit.md`): no implementation bug; 12 findings, 10 of them defects of the paper. D-02 revised — `gate_target=features` implemented, default unchanged, because `x_gated` is otherwise never consumed. Table 4 row 3 and Table 5 T=1 are the same configuration under D-17 yet differ by 0.63; recorded as an open conflict. |
| 2026-09-20 | D-18 closed on three seeds per variant: `full` 0.5171 ± 0.0123 against `baseline_l2` 0.5205 ± 0.0104, i.e. the cascade adds nothing (t = −0.36). Every stage diagnostic is healthy, and the Eq.19 weight on the class-blind `P_diffuse` falls to 0.008–0.082 by itself, so D-16 is not the cause either. The `layernorm` probe stays an ablation flag, never a default. |
| 2026-09-21 | D-12 measured against VIP-Seg's batch-1 schedule: no difference (0.4901 vs 0.4908). D-21 measured: leakage lifts baseline and full model by 12–15 points, not to the paper's level. |
| 2026-09-22 | D-21 corrected: the 20-epoch run is a partial leak. Scored on classes seen in training, the baseline reaches 77.32 / 71.58 (S0 / S1) against the paper's 82.72 / 79.83. |
| 2026-09-22 | Phase 16 (improvement research, beyond the paper) opened by the maintainer. D-22: `eval.py` refuses to score seen classes; phase-16 reporting rules (last.pt, screening on S1, S0 held out). |
| 2026-09-23 | D-26 (maintainer request): query-side entropy-weighted EM refinement on top of route B, probed on trained checkpoints (P0) before any training; rules fixed before the run. |
| 2026-09-23 | D-26 closed by its rule P0.2. D-27 (maintainer request): base-class calibration of the background, probed on the same checkpoints (P1) before any training; rules fixed before the run. |
| 2026-09-23 | D-27 closed by P1.2 (P1.4 weak). D-28 (maintainer request): the two combined, the base margin filtering the foreground M-step of D-26, probed as P2; rules fixed before the run. |
| 2026-09-23 | D-28 closed by P2.0 / P2.2. D-29 (maintainer request): oracle-direction distillation during training, on route B's head (revised before any run to the logit-space pairwise form); one training run per arm, three test draws (maintainer); rules fixed before the run. |
| 2026-09-24 | D-29 closed by R2.2; R2.0 shows our loop's route-B base 5.16 below VIP-Seg's released checkpoint, which VIP-Seg's own logs put down to training length and checkpoint selection. |
| 2026-09-24 | D-22 rule 1 amended by the maintainer: `best.pt` under VIP-Seg's disclosed selection rule is reported with `last.pt`. D-30: route B's base on VIP-Seg's update count (E1); rules fixed before the run. |
| 2026-09-24 | D-30 closed by E1.1: route B's base on VIP-Seg's update count reaches 73.20 `last` / 75.05 `best` on S1 fixed100 (VIP-Seg released 75.36). |
| 2026-09-24 | D-31 (maintainer request): P3, a training-free gap decomposition and text-prior probe on E1; interpretation bands and rules fixed before the run. |
| 2026-09-24 | D-31 measured: text stops (P3.0); E1's dominant error is floor/wall → background, flat across support sizes. |
| 2026-09-24 | D-32 (maintainer request): P4, a causal test of background contamination by the episode's own classes (clean-background intervention, label-free purification); rules fixed before the run. |
| 2026-09-24 | D-32 measured: background contamination is not causal (clean background +0.09); the oracle gain is joint across rows. |
| 2026-09-24 | D-33 (maintainer request): N1, a zero-initialised point-level support → query attention neck, warm-started from E1 against a control arm; rules fixed before the run. |
| 2026-09-24 | D-33 measured: N1.2 stop with α = 0.0023; the neck was never used. |
| 2026-09-24 | D-34 (maintainer request): N2, the neck trained from scratch on E1's schedule with alpha initialised to 0.1, against E1's existing checkpoints; script prepared, not run. |
| 2026-09-24 | D-34 measured: N1.2 stop; the neck opened (α 0.060) and costs 0.5–1.1 points on every draw. |
| 2026-09-24 | D-35 (maintainer request): P5, an inference-only split of E1's oracle gap by sampling condition and class presence, with a sampling intervention, test-time batch statistics, a leak-free draw and training-free dual-condition prototypes; evidence from C1/C2 and the saved P4/E1 counts; rules fixed before the run. |
| 2026-09-24 | D-35 outcome: the smoke run passed its checks but showed E1 naming the foreground by query position; C3 measured it on all of fixed100 (E1 73.20 → 0.92, VIP-Seg released 75.36 → 0.87 with the two query blocks swapped). The full P5 is not run as registered. |
| 2026-09-24 | D-36 (maintainer request): C4, VIP-Seg's cross-term re-run in a per-query form on the trained weights; D-37 (maintainer request): the 2 × 2 of head × training query order. Amended before any code: CF = CR by a lemma checked by tests, the relabel shift replaces the raw relabel share, E1 re-scored as an evaluation check, a tie in D37.3 goes to the clean head. |
| 2026-09-25 | D-36 measured: C4.1, the cross-term reshape is the sole carrier. D-37 measured: the fixed query order is the origin (D37.1); the clean head is the base (D37.3); the shortcut is worth +19.9 on S1 fixed100, and the oracle gap of the clean base is +28.5. |
| 2026-09-25 | D-38 (maintainer request): P6, an inference-only probe on the clean base: row-wise oracle in one geometry, entropy-gated self-support and a multi-component background, selected on valid, rules fixed before the run. |
| 2026-09-25 | D-38 measured: the gap is joint (foreground +16, background +6); training-free background adaptation (self-support, k-means) gives +1.2–1.3 on every draw, foreground self-support hurts, entropy selection adds nothing. |
| 2026-09-25 | D-39 (maintainer request): trained self-support prototypes on the clean base (A0 prototype matching, A1 + two self-support steps) and the combined background rule; rules fixed before the run. |
| 2026-09-25 | D-39 measured: combined background rule adopted (CR + U + both 57.63, S1 fixed100); trained self-support stops (−0.66); training without the head costs 2.2. |
| 2026-09-25 | D-40 (maintainer request): P7, label propagation on the query's own point graph (xyz, xyz with feature weights, feature kNN) seeded by U + both, with its preconditions measured (P7a); gate on the valid gain, rules fixed before the run. |
| 2026-09-26 | D-40 measured: propagation adopted at inference (CR + U + both + LP 58.55, S1 fixed100, +0.92), as a denoiser, not a recall mechanism (P7.6 unexplained); recall errors are whole regions, and other-condition points (recall 0.16) are a third of the missed foreground. |
| 2026-09-26 | D-41 (maintainer request): P8, the clean base's oracle gap split by sampling condition (own / other oracles) and D-35's condition intervention re-run on it with its φ bands; rules choose the next training run. |
| 2026-09-26 | D-42 (maintainer request): the stacked architecture of the 2026-09-26 research note as the direction; P9 checks the input assumptions at each insertion point (block coupling, density, extent, common component) and each module's precondition before any training; rules fixed before the run. |
| 2026-09-26 | D-41 measured: other-condition bound +5.25, own +13.71; density causal (φ 1.00); uniform sampling halves the own class's recall (0.84 → 0.46). |
| 2026-09-26 | D-42 amended: P9's module preconditions are measured on the base D-43 adopts. D-43 (maintainer request): M1, a density-invariant encoder (metric ball neighbourhoods, offsets in metres, per-block standardisation, metric coordinates), one training run against CR; rules on the mechanism, the leak-free and the standard protocol fixed before the run. |
| 2026-09-26 | D-43 measured: D43.1 and D43.2 fail, stop (D43.4); CR stays the base. M1's head gains on the leak-free draw (+2.07) and loses on fixed100 (−6.89); the decision still follows density (φ 1.14). |
| 2026-09-26 | D-44: P9 after D-43. Step 0 replays M1's ball grouping on P8's events (distinct points per ball); part A locates the remaining density pathway in M1 by a ball-count intervention; part B measures D-42's module preconditions on CR. P9 decides admissibility; the one remaining run is chosen in a new decision. |
| 2026-09-27 | D-44 amended (statistics arm; 3090 with local data; composite arm) and measured: encoder branch closed (no ψ above 0.02); only the metric head is admissible, by its oracle; the features are collapsed (participation ratio 5.56). |
| 2026-09-27 | D-45 (maintainer request): M2, VICReg's variance and covariance terms on the query point features against the collapse P9 measured (post hoc, rules fixed before training); two weights, one seed each, a monitor with an early stop, a collapse census of every kept checkpoint. |
| 2026-09-27 | D-45 measured: D45.3. VICReg widens the features (participation ratio 42 / 95) but U falls 5.2 / 11.3 points below CR, and no oracle rises. The collapse is shared by every checkpoint, VIP-Seg's included. The M2 line stops. |
| 2026-09-28 | D-46: P10 (the base-class prototype gap gates M5 alignment training; P3's text probe on CR) and M5 (prototype alignment, two weights). Amendment 1 (maintainer request after an external review): P10.3, the K-shot curve that splits the 1-shot gap into bias and variance; P10.4, a metric from base-class statistics. |
| 2026-09-28 | D-46 amendment 2 (maintainer request): P10.5, the K-curve split by sampling condition; amendment 1's error model corrected (the oracle's query mean is one instance), P10.3's bias share 0.77 → 0.54. |
| 2026-09-28 | D-46 measured: P10.1 holds (+14.76) but M5 fails D46.1 in both arms, D46.4 stop; P10.4 holds (+1.46, isotropic +1.11); text +0.41 (in between). P10.5: own pairs carry no bias, other pairs 91 % bias; exact foreground prototypes reach 72.44 at one shot. |
| 2026-09-28 | D-42 closed (maintainer approval): every module decided by its own measurement (D-43, D-44, D-45, D-46); the next architecture is chosen from D-46's error decomposition. |
| 2026-09-28 | D-47 (maintainer request): image and audio modalities at the class level, as in Fig. 1. D-48 (maintainer request): one composed architecture replacing the PEM/PDM head (base learner, multi-prototype correlation neck, per-episode modality weights, LP), with a pre-check P11 and two arms; rules fixed before the run. |
| 2026-09-28 | D-48 amendment 1 (after a mathematical review by an Opus agent, re-checked against the sources): base CE detached from the encoder; leave-target-out exclusion with a one-sided or input-side background term; P11.1b and P11.4 gate the arms; the text prior's history corrected (P3 already weighted per episode) and scored off in the rules; LP and "both" re-selected per arm; second seed below +2.0. |
| 2026-09-28 | D-48 amendment 2 (maintainer critique): adaptive cell count (≥ 32 points per cell) with a (max, top-2 mean, mean) descriptor; four descriptor spaces compared (raw, centred, top-r projection, truncated whitening); P11.5 unbalanced OT assignment at inference with a leak-free guard; max-combined base logit and the semi-relaxed OT form not adopted. Fix register: `docs/research/2026-09-28_d48_fix_register.md`. |
| 2026-09-28 | D-48 amendment 3 (maintainer approval): [2] fully decoupled and attached post hoc; arm B replaced by a second seed of A; a 2⁴ inference factorial per head with add-one, leave-one-out, exact Shapley and pre-registered interactions; rules D48.1'–D48.4'. |
| 2026-09-28 | D-48 amendment 4 (implementation notes before any run): block order, grids, coordinate-ascent selection, the gates' draws, class-level OT rows, text only until D-47, base-labelled moments. P11 coded and tested on the CPU (P11-1…23, mutation 17/17); not run on the GPU. |
| 2026-09-28 | D-48 amendment 5 (maintainer request, before P11's results were read): P11.6, label-free context unmixing (2 × 2 least squares on the class row and the spatial-neighbour mean, gated by cos(p_c, c(x)) < τ, centred space) with an oracle check of the linear mixture; a new inference base from measured gains only (mean rule + both + LP). |
| 2026-09-28 | D-48 P11 and P11.6 measured: inference rules on CR saturated (only LP kept); the base learner reads density; the correlation probe passes its gate with a leak-free loss; unmixing fails because sparse points carry no own-class signal in CR's features. |
| 2026-09-29 | D-48 amendment 6 (arm A as built: correlation head, raw-space descriptors, 2 layers, deep supervision, seeds 0 and 1) and D-49 (condition-balanced training, D-41's registered second arm, never run until now); both trained together on one GPU. |
| 2026-09-29 | D-47 implemented (amendment 1: manifest in `assets/modality/`, images chosen by eye from Commons contact sheets, audio settings) and built: audio 10/14 exact transcripts, image rows 12/14 nearest their own text row. |
| 2026-09-19 | D-17: no `W_g` for T = 1 (identical prediction, no dead parameter). D-16 biases of `W_1`, `W_2`, `W_out` kept although Eq.20–21 print none (maintainer decision). |
| 2026-09-28 | Decision log split: §4 moved verbatim into one file per decision in `docs/decisions/` (checked: every line of the old section is in exactly one file); §4 is now an index with a one-line preliminary result per decision (drafted by two agents, every number checked against the files). |
