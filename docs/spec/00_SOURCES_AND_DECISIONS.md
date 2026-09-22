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
| **L3** | Decision log (Section 4 of this file) | This file | Points where L1 is silent/contradictory and L2 does not settle them |

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

## 4. Decision log

**Status legend**

| Status | Meaning |
| :--- | :--- |
| `LOCKED` | Confirmed by the maintainer on 2026-09-17. Change only with new L1/L2 evidence. |
| `PROPOSED` | Default chosen during Phase 0, awaiting maintainer review. Specs may use it but must keep the ablation flag. |

Each ablation flag named below is a requirement on the future CLI/config, not an existing option.

### D-01 — Cross-attention refinement (Eq.13–14) · `LOCKED`

* **Problem.** Eq.14 is dimensionally invalid as printed: "A = softmax(Q′S′ᵀ/√d) ∈ R^{Nq×Ns}, P_cross = A · ψ(P^{t−1})" multiplies an `Nq×Ns` point–point matrix by an `(N+1)×D` prototype matrix. No implementation can honour both halves literally.
* **Evidence (L2).** VIP-Seg implements the same construct as a **channel–channel** correlation, which makes `A · ψ(P)` valid:
  * `self.maxpool = nn.MaxPool1d(32, stride=32)` — 2048 points → 64 tokens [VIPSEG models/vipseg.py:213]
  * `self.map = nn.Conv1d(64, self.proj_dim, 1, bias=False)` with `proj_dim = 72`, one module shared by query and support [VIPSEG models/vipseg.py:219]
  * `self.proto_map = nn.Linear(128, 128)` (ψ) [VIPSEG models/vipseg.py:222]
  * `crosscor = … softmax(dim=-1)` of shape `[B, N+1, 128, 128]`, then `proto_cross = crosscor @ new_proto` [VIPSEG models/vipseg.py:288-296]
  * The printed symbols φ = 1×1 conv, d = 72, softmax, ψ = linear and `P_cross = A·ψ(P)` all match this construct; only the annotation `∈ R^{Nq×Ns}` does not. The Table 6 overhead of +0.38 GFLOPs for all four stages is compatible with a 128×128 matrix and not with a 2048×(N·K·2048) matrix per stage.
* **Options considered.** (1) Channel correlation as in VIP-Seg, keeping `P_cross = A·ψ(P)` literal. (2) Two-hop point attention from the pre-rewrite spec (`F_qs = A·F_s`, then a second attention with ψ(P) as query), keeping only the `Nq×Ns` annotation and inventing an extra attention.
* **Decision: option 1**, specified in its mathematically clean form:
  1. `F̃^q = MaxPool_32(F^q) ∈ R^{64×D}`, `F̃^s_{c,k} = MaxPool_32(F^s_{c,k}) ∈ R^{64×D}` for class slot `c ∈ {0..N}` and shot `k`. The background slot uses the mean of the way-wise support features, `F^s_{0,k} = mean_n F^s_{n,k}` [VIPSEG models/vipseg.py:244].
  2. `Q′ = φ(F̃^q) ∈ R^{d×D}`, `S′_{c,k} = φ(F̃^s_{c,k}) ∈ R^{d×D}`, with φ = `Conv1d(64 → d=72, kernel 1, bias=False)` shared across query and support [PAPER Eq.13] [VIPSEG models/vipseg.py:219].
  3. `A_{c,k} = softmax_row(Q′ᵀ S′_{c,k} / √d) ∈ R^{D×D}` — scale by **√d = √72** as printed in Eq.14 (VIP-Seg uses √128; L1 wins).
  4. `P_cross[c] = (1/K) Σ_k A_{c,k} · ψ(P^{t−1}_gated[c]) ∈ R^D`, ψ = `Linear(D, D)` [PAPER Eq.14] [VIPSEG models/vipseg.py:222,296]. See D-02 for the gated input. VIP-Seg averages its **whole module output** over shots [VIPSEG models/vipseg.py:308]; averaging only `P_cross` is our choice, because Eq.21 applies the residual and LayerNorm once per stage.
  5. **Class slots are a VIP-Seg construct, not paper notation.** Eq.13 writes a single `S′ = φ(F^s)` without class slots; one `A` per class slot follows VIP-Seg [VIPSEG models/vipseg.py:244,288-296]. The background slot averages support features point by point across ways; because the loader shuffles point order [VIPSEG dataloaders/loader.py:58], this pairs unrelated points and survives max-pooling only as a rough mixture. A single `A` per (query, shot) computed from all support blocks would follow Eq.13 more literally; the maintainer chose the class slots of VIP-Seg on 2026-09-19, because Eq.14 applies `A · ψ(P)` to each class row and the paper builds on VIP-Seg.
* **Deliberate deviation from L2.** VIP-Seg computes the correlation after `reshape(proj_dim, -1)` on batched tensors [VIPSEG models/vipseg.py:288-291]. A numerical check (2026-09-17) showed this does **not** equal the per-(query, class) product `Q′ᵀS′_c`, even for batch size 1: rows of different classes and projection filters are interleaved. We implement the clean per-(query, class, shot) form above. A compatibility flag may reproduce the VIP-Seg reshape for debugging only.
* **Ablation flags.** `cross_attn = {channel (default), two_hop}`; `cross_attn_scale = {sqrt_d (default), sqrt_D}`. `two_hop` is option 2 above, which was rejected; it raises `NotImplementedError`.
* **Affects.** Spec 01 §3 sub-module 2, spec 02 §4.2, spec 05 EPPM tests, `models/eppm.py`.

### D-02 — Target of entropy gating · `LOCKED`

* **Problem.** Eq.10 gates "a feature vector x ∈ R^D", §1 speaks of suppressing "high-entropy background features", but the next sentence says "After entropy gating, we apply cross-attention between the query and support to further refine the prototype", and Eq.14/Eq.21 only reference `P^{t−1}`.
* **Decision.** Gate the incoming prototype channel-wise: `P^{t−1}_gated = P^{t−1} ⊙ g(P^{t−1})`, with one scalar θ per stage [PAPER §3.5 "each EPPM applies entropy gating independently"]. `P^{t−1}_gated` is the input to ψ in D-01. The residual in Eq.21 uses the **ungated** `P^{t−1}`, as printed.
* **Rationale.** Channel-wise gating composes naturally with the channel–channel attention of D-01, and it is the only reading consistent with Eq.14 and Eq.21 using `P^{t−1}`.
* **Known limitation.** With θ = 0.5 and natural log, `g ∈ [0.405, 0.731]`, so the gate is weak at initialisation (audit H4).
* **Outcome of the paper audit (2026-09-20, `docs/research/2026-09-20_paper_vs_code_audit.md`).** The
  rationale above does not hold. Both readings are consistent with Eq.14 and Eq.21 referencing
  `P^{t-1}`: if the gate applied to `F^q`/`F^s`, Eq.13's `φ(F^q)` would simply consume gated features
  and Eq.14 would be unchanged. Worse, under `gate_target = prototype` the printed Eq.14 multiplies
  `ψ(P^{t-1})`, the **ungated** prototype, and Eq.21's residual is also `P^{t-1}`, so `x_gated` — a
  symbol that appears in Eq.12 and **nowhere else in the paper** — would never be consumed and Eq.10–12
  would be dead. Our code only makes the gate matter by feeding ψ the gated prototype, which is a
  silent departure from Eq.14. The paper's own wording leans the other way: Eq.10 says "for a **feature
  vector** x ∈ R^D", Eq.12 says "the **gated feature** is", and §3.2 says each module "suppresses
  high-entropy background **features**".
* **Revised decision (2026-09-20).** `gate_target = features` is implemented and no longer raises: the
  gate is applied per point to `F^s` and `F^q` before Eq.13, and ψ receives the ungated `P^{t-1}`. The
  default stays `prototype` until a VM run separates them, because changing it changes every trained
  checkpoint's meaning. The diffusion branch of Eq.15–18 is ungated under both readings, since those
  equations name `F^q` and `F^s` with no mention of gating.
* **Ablation flag.** `gate_target = {prototype (default), features}`. Neither value raises.

### D-03 — Shared φ · `LOCKED` (paper-explicit)

* Query and support use the same φ module [PAPER Eq.13] [VIPSEG models/vipseg.py:219]. Folded into D-01; kept as an ID because the audit refers to it.

### D-04 — GMMN sample sets and gradient flow · `LOCKED`

* **Problem.** Eq.7 defines MMD between sets, Eq.8 applies it to `P^bg` and `P^fg` without defining the sets. The paper does not say whether `P_point` is detached.
* **Decision.**
  1. Use the squared MMD of Eq.7 (paper-explicit, Section 3).
  2. `P^fg` = the N foreground rows taken **jointly** as one set of N samples, not N separate 1-vs-1 comparisons. `P^bg` = the single background row (1-vs-1).
  3. `P_point` is **not** detached; gradients from `L_GMMN` reach the backbone, since Eq.26 optimises all parameters jointly and the paper does not mention stopping gradients.
* **Known limitation.** For N = 2 or 3 the MMD estimate is very noisy; a 1-vs-1 MMD reduces to `2·(6 − k(x, y))`.
* **Ablation flags.** `gmmn_fg_mode = {joint (default), per_class}`; `gmmn_detach_point = {false (default), true}`.

### D-05 — "Fuses both sources" in §3.2 · `LOCKED`

* **Conflict inside L1.** §3.2: "Learnable Modality Adapters (LMA) project CLIP text and image embeddings into the point cloud feature space. A GMMN-based distribution matching module then fuses both sources". Abstract: "enabling flexible single-modality semantic enrichment"; Tables 2–3 report separate Text / Image / Audio rows; Eq.4 defines one adapter per modality.
* **Decision.** One modality per run. `E_fused := E_adapted^(m)` for the selected modality m [PAPER Eq.4, Eq.6]. "Both sources" is read as the point prototype and the modal prototype combined by Eq.9.
* **Sub-decision (generator input).** Eq.6 writes `G(E_fused, z)` without a noise dimension; use `z ∈ R^{(N+1)×D}` concatenated with `E_fused` (input 2D = 256) followed by a three-layer MLP [PAPER Eq.6 "three-layer MLP generator"].

### D-06 — Noise z at inference · `LOCKED`

* **Problem.** Eq.9 says "The fused initial prototype **used for training** is P^0 = P_point + P_modal"; §3 promises an "inference strategy" that is never described.
* **Decision.** Training samples `z ~ N(0, I)` per forward pass. Evaluation uses `z = 0`, making predictions deterministic.
* **Ablation flag.** `eval_noise = {zero (default), sample, mean_of_M}`. `mean_of_M` needs a value of M that L1 does not give; it raises until one is chosen.

### D-07 — Train/test split · `LOCKED`

* **Conflict between L1 and the protocol it cites.** L1: "using Areas 1, 2, 3, 4, 6 for training and Area 5 for testing under two category splits S0 and S1" [PAPER §4.1], but also "We follow the standard N-way K-shot episodic protocol [34]" [PAPER §4.1]. The loader VIP-Seg releases (inherited from AttMPTI [34]) splits by **class only**, over all areas [VIPSEG dataloaders/s3dis.py:20-31], and VIP-Seg's released S0 result (72.20) was produced with it [VIPSEG log_s3dis_VIPSeg/log_S0_N2_K1_0.722026/log_vipseg_eval.txt:7-9]. The other Table 2 baselines are cited, not re-run, so their protocol is assumed to be the same.
* **Decision.** Primary protocol = class-only split with the fold lists of the inherited loader:
  * S3DIS S0 = `beam, board, bookcase, ceiling, chair, column`; S1 = `door, floor, sofa, table, wall, window` [VIPSEG dataloaders/s3dis.py:20-21].
  * ScanNet S0 = `bathtub, bed, bookshelf, cabinet, chair, counter, curtain, desk, door, floor`; S1 = `otherfurniture, picture, refridgerator, shower curtain, sink, sofa, table, toilet, wall, window` [VIPSEG dataloaders/scannet.py:21-22].
  * The paper itself never lists the classes of S0/S1; the old spec's lists were invented.
* **Ablation flag.** `split_protocol = {class_only (default), area5}`. `area5` restricts training scans to Areas 1–4, 6 and test scans to Area 5 (S3DIS only).

### D-08 — Evaluation protocol and metric · `LOCKED`

* **Problem.** L1 gives no mIoU formula: "Performance is reported as mean IoU (mIoU) averaged over S0 and S1 across 600 randomly sampled episodes" [PAPER §4.1].
* **Decision.** Primary = the VIP-Seg protocol:
  * Test set = `MyTestDataset`, 100 episodes per class combination, fixed and cached as `.h5` [VIPSEG dataloaders/loader.py:230-267] [VIPSEG scripts/vipseg_eval_s3dis.sh `N_TEST_EPISODES=100`], `n_queries = 1`.
  * Metric = TP/FP/FN accumulated over **all** test episodes per global test class, IoU per class, mean over foreground classes only (background excluded) [VIPSEG runs/training_free.py:56-59].
  * Report S0, S1 and their average separately, as in Tables 2–3.
  * L2 evidence that this reproduces the paper's baseline numbers: VIP-Seg log directory `log_s3dis_VIPSeg/log_S0_N2_K1_0.722026` matches VIP-Seg's 72.20 in Table 6.
* **Ablation flag.** `eval_protocol = {vipseg_fixed (default), random600}`.

### D-09 — Params/FLOPs target (Table 6) · `LOCKED`

* **Problem.** Table 6 states CascadeProto 2.88M / 8.86G against VIP-Seg 2.76M / 8.48G [PAPER Tab.6], described as "a marginal overhead of 0.12M parameters and 0.38 GFLOPs over the VIP-Seg backbone" [PAPER §4.4].
* **Evidence (L2).** VIP-Seg's 2.76M is the **whole** VIP-Seg model: encoder 2.37M + VIP module 0.19M + feature head/gating [VIPSEG log_s3dis_VIPSeg/log_S0_N2_K1_0.722026/log_vipseg.txt:2-11]. The encoder plus the 900→196→128 feature head is therefore ≈ 2.57M (head = 204,260 parameters). CascadeProto replaces the VIP module with LMA + EPPM + ADRM, so the paper implies ≈ 0.31M for those modules; the layer choices of D-16 give 466,444.
* **Decision.** Table 6 is **not** a hard acceptance criterion. Measure and report with a fixed tool and configuration (fvcore, 2-way 1-shot, one query, 9 channels). Acceptance for the inherited part only: encoder 2.37M ± 0.01M and encoder + head ≈ 2.57M once `mamba_ssm` is used. The added modules are reported separately.

### D-10 — Logit form (Eq.23) · `LOCKED`

* **Problem.** Eq.23 is called "scaled dot-product matching" but prints no scale: `L^t = F^q (P^t)ᵀ`.
* **Evidence (L2).** VIP-Seg uses a plain dot product with no temperature [VIPSEG models/vipseg.py:158], but L2-normalises the **initial** prototypes [VIPSEG models/vipseg.py:142]. L1 Eq.3 and Eq.9 show no normalisation.
* **Decision.** Plain dot product exactly as printed; no temperature; no L2 normalisation of `P_point`.
* **Ablation flags.** `logit_scale = {none (default), sqrt_D}`; `l2norm_point_proto = {false (default), true}`.

### D-11 — Shape of the fusion weight w (Eq.19) · `LOCKED`

* **Problem.** Eq.19 states `w ∈ R²` while `P_cross`, `P_diffuse ∈ R^{(N+1)×D}`.
* **Decision.** One weight pair per query: `w = softmax(f_fusion(mean_c [P_cross; P_diffuse]))`, where `f_fusion` is a two-layer MLP `2D → D → 2`, pooled over the class dimension, as printed `w ∈ R²`.
* **Ablation flag.** `fusion_weight = {per_query (default), per_class}`.

### D-12 — Epoch definition, augmentation · `LOCKED` (epoch) / `LOCKED` (augmentation)

* **Problem.** L1 gives epochs, batch size and StepLR period but not the number of episodes per epoch or any augmentation. L2 trains for `NUM_ITERS=24000` episodes with batch size 1 on both datasets [VIPSEG scripts/vipseg_s3dis.sh, scripts/vipseg_scannet.sh].
* **Decision (LOCKED).** Match VIP-Seg's total of **24,000 training episodes**, with L1's batch size and epochs:
  * S3DIS: 50 epochs × 480 episodes/epoch = 24,000 episodes = 120 optimiser steps/epoch (batch 4).
  * ScanNet: 30 epochs × 800 episodes/epoch = 24,000 episodes = 200 optimiser steps/epoch (batch 4).
  * StepLR(step = 10 epochs, γ = 0.5) [PAPER §4.1]. Episodes per epoch exposed on the CLI.
  * *Correction (2026-09-17):* the option text confirmed earlier said "≈480 steps/epoch"; 480 is the number of **episodes** per epoch, i.e. 120 steps at batch 4.
  * *Measured (2026-09-21):* VIP-Seg's own schedule (batch 1, 24,000 steps, halving every 7,200) gives the baseline 0.4901 on fixed100 against 0.4908 under this decision, so the choice of batch and decay is not a source of the gap to the paper (`train.py --batch_size`, report §3.3, CHANGELOG 15x).
* **Decision (LOCKED, augmentation).** Use VIP-Seg's training augmentation: `pc_augm` on, `shift 0.1`, `rot 1`, `jitter 1`, `scale 0`, `mirror 0`, `color 0` [VIPSEG scripts/vipseg_s3dis.sh] [VIPSEG main.py:56-66]. No augmentation at test time. Measured effect (2026-09-17): the encoder reads only the normalised `XYZ` and `rgb` columns [VIPSEG models/encoder.py:645], and `XYZ` is recomputed after subtracting the minimum [VIPSEG dataloaders/loader.py:70-74], so the shift augmentation does not change the model input; rotation and jitter do.

### D-13 — Modality scope, CLIP variant, prompts · `LOCKED`

* **Problem.** L1 names "CLIP [17]" and Fig.1 shows "Audio → Whisper → CLIP" and "Image → CLIP", but gives no CLIP variant, no background prompt, no image or audio source, and no Whisper usage details. The only prompt shown is "This point cloud represents the chair." [PAPER Fig.1].
* **Decision.**
  1. Implement **text first**; image and audio are deferred and must fail loudly if selected before they exist.
  2. CLIP variant fixed by config, default `ViT-B/16`, logged with every run. Embeddings are L2-normalised (standard CLIP usage; not stated in L1), computed once per class and cached; CLIP is never reloaded per episode.
  3. Foreground prompt `"This point cloud represents the {class}."` [PAPER Fig.1]. Background prompt `"This point cloud represents the background."` (assumption; not in L1). `{class}` is the class name exactly as written in the dataset's class-name file (e.g. `shower curtain`, `otherfurniture`, `refridgerator`).
  4. Audio (when implemented): Whisper transcription → CLIP text encoder, following the arrow order of Fig.1.

### D-14 — ReLU before EPPM and diffusion degeneracy · `LOCKED`

* **Problem.** The VIP-Seg feature head ends in `BatchNorm1d + ReLU` [VIPSEG models/vipseg.py:46-53], so features are ≥ 0 and `σ(mean) ≥ 0.5`. With τ = 0.5 a channel is active in a branch exactly when its mean is **strictly** positive. When every channel has a positive mean in both branches (the usual case; measured at initialisation, audit M2), `m_common = 1` and `c_unique = 0`, so `P_diffuse = (q_ch + s_ch)/4` with every entry in [0.25, 0.5]. A channel whose ReLU output is zero on all points of one branch but not the other still yields a non-zero `c_unique` (verified numerically 2026-09-17). Independently of ReLU, `P_diffuse` is the same for every class row, because Eq.15–18 contain no class index.
* **Decision.** Keep the VIP-Seg head unchanged (L1 says the encoder is VIP-Seg) and implement Eq.15–18 literally. Document the degeneracy.
* **Ablation flag.** `diffusion_input = {post_relu (default), pre_relu}`. `pre_relu` needs the feature head's pre-ReLU output and raises `NotImplementedError`.

### D-15 — Model selection · `LOCKED`

* **Problem.** L1 is silent. L2 evaluates every 2,000 iterations on a `MyTestDataset(mode='valid')` built from the **test classes** and keeps the best checkpoint [VIPSEG runs/training.py], i.e. model selection on the test classes.
* **Decision.** Primary = VIP-Seg-compatible selection, so numbers are comparable with Tables 2–3 (validate every 10 epochs on the `valid` episode set, keep the best). Always also log the **last-epoch** checkpoint's test mIoU so the selection effect is visible.
* **Ablation flag.** `model_selection = {best_valid (default), last}`.

### D-16 — Layer details not given in the paper · `LOCKED`

* **Problem.** L1 names several layers without widths, rates or placement. The pre-rewrite specs stated values for them as if they were paper content.
* **Decision.** Use the following values, always tagged `[DECISION D-16]`:
  1. **Adapter** (Eq.5, "two-layer MLP with LayerNorm and Dropout" [PAPER §3.3]): `Linear(512→D) → LayerNorm(D) → ReLU → Dropout(p = 0.1) → Linear(D→D)`. Dropout rate and position are not in L1.
  2. **Generator G** ("three-layer MLP" [PAPER Eq.6]): `Linear(2D→D) → ReLU → Linear(D→D) → ReLU → Linear(D→D)`, input `[E_fused; z]` per D-05.
  3. **Fusion network** ("two-layer MLP" [PAPER Eq.19]): `Linear(2D→D) → ReLU → Linear(D→2)`.
  4. **SE block** (Eq.20): reduction ratio r = 4, `W_1 ∈ R^{D/r×D}`, `W_2 ∈ R^{D×D/r}`; AvgPool over the class dimension.
  5. **Output projection** (Eq.21): `W_out = Linear(D→D)`; `LN = LayerNorm(D)`.
  6. **Entropy stability:** clamp `p` to `[10⁻⁷, 1 − 10⁻⁷]` before Eq.10. L1 only specifies ε = 10⁻⁸; the clamp does not change results for |x| < 16.
  7. **Diffusion pooling** (Eq.15): `mean(F, dim=1)` is read as the mean over the point axis of a batched `[B, N_points, D]` tensor, the only reading that yields the "channel-level activations" the text describes (on an unbatched `[N_points, D]` tensor `dim=1` would average over channels). `s_ch` is the channel mean over all support points of all ways and shots; `q_ch` is computed per query. The vector `P_diffuse ∈ R^D` is broadcast to all N+1 class rows before Eq.19.
  8. **Stage parameters** are not shared across the T stages [PAPER §3.5 "each EPPM applies entropy gating independently"].

### D-17 — Meaning of the ablation switches (Table 4–5) · `LOCKED`

* **Problem.** Table 4 "cumulatively adds one component": LMA, Entropy Gate, Cascade (T = 4), ADRM, "starting from a plain VIP-Seg backbone with masked average pooling and single-step prototype matching" [PAPER §4.3]. L1 does not say what the model looks like between rows, e.g. what "Entropy Gate" without "Cascade" is.
* **Observation.** The row increments in Table 4 match the text: +1.21 (LMA), +1.42 (gate), +2.06 (cascade), +0.56 (ADRM) [PAPER Tab.4] [PAPER §4.3]. The baseline row (81.28 Avg) is far above VIP-Seg's own 74.15 Avg in Table 2; L1 does not explain the gap.
* **Decision.** Four switches, mapped to the rows of Table 4:

  | Row | `use_lma` | `num_stages` | `use_gate` | `use_adrm` | Prediction |
  | :--- | :---: | :---: | :---: | :---: | :--- |
  | Baseline | false | 0 | – | – | `F^q P_pointᵀ` |
  | + LMA | true | 0 | – | – | `F^q (P^0)ᵀ` |
  | + Entropy Gate | true | 1 | true | – | `L^1` |
  | + Cascade (T = 4) | true | 4 | true | false | `L^4` |
  | + ADRM (full) | true | 4 | true | true | `L_final` |

  `use_gate = false` sets `g ≡ 1`. `use_lma = false` sets `P_modal ≡ 0` and drops `L_GMMN`. Without ADRM the prediction is the last stage's `L^T`. With `num_stages = 1` ADRM is a softmax over one stage, i.e. weight 1, so `L_final = L^1` whether `use_adrm` is on or off; `W_g` is then not built, because it could never receive a gradient (maintainer decision 2026-09-19). Table 5 varies `num_stages` ∈ {1..6} with every other switch on [PAPER Tab.5].

---

### D-18 — Degeneracy of the cross-attention output (Eq.14) · `LOCKED` (as an ablation flag)

* **Symptom (L0, measured).** On the VM, with the same loop, data, encoder and metric, 2,400 training
  episodes and 300 valid episodes (2026-09-20): `baseline_l2` 0.5218, `full_l2` 0.5150, `full` 0.5164,
  `baseline` 0.4416, VIP-Seg's own model 0.6948. Four EPPM stages plus ADRM plus LMA move the score by
  −0.7 points against a plain L2-normalised prototype. The +7.5 of `full` over `baseline` is the
  LayerNorm of Eq.21 equalising the prototype norms, which `l2norm_point_proto` does on its own (+8.0).
* **Structural cause, independent of any number.** `P_cross[b,c] = A[b,c] ψ(P_gated[b,c])` with the
  rows of `A` a softmax over the **channel** axis [DECISION D-01]: output channel `i` is a convex
  combination of the channels of `ψ(P)`. Both limits of that softmax destroy the channel structure —
  saturated, every row copies the same channel; uniform, every row copies the channel mean — and Eq.14
  has no learnable term that controls where between the two it sits. The other summand of Eq.19,
  `P_diffuse`, carries no class index at all [DECISION D-16]. So a stage can add class-discriminative
  structure only through the residual `P^{t-1}` of Eq.21. Measured: the update `P^1 − P^0` keeps
  87.5–99.6 % of its energy in a single singular value across every feature distribution tried, and in
  the distribution of the phase-11 fixtures the class rows leave the stage **more** similar than they
  entered (max pairwise cosine 0.9906 → 0.9964).
* **Where on that axis does the module sit?** The regime depends on the per-channel statistics of the
  encoder features, which are not reproducible on the CPU. Measured on synthetic post-ReLU features,
  `β` = the per-channel offset a trained BatchNorm supplies (one shared φ, one seed, `|mean_D|`-relative
  channel variation of `P_cross`):

  | features | `none`: softmax width (uniform = 128) | `none`: channel variation | `layernorm`: width | `layernorm`: channel variation |
  | :--- | ---: | ---: | ---: | ---: |
  | β = 0 (BatchNorm at initialisation) | 112.8 | 2.3e-01 | 127.5 | 1.3e-01 |
  | β std 0.5 | 8.1 | 6.8e-01 | 125.6 | 1.2e-01 |
  | β std 1.0 | 10.7 | 2.2e+00 | 114.0 | 1.8e-01 |
  | β std 2.0 | 27.2 | 1.9e+00 | 96.1 | 5.1e-01 |
  | positive-only offset, `relu(N(0,1))` per channel | 1.02 | 2.6e-04 | 127.4 | 1.5e-01 |

  The last row is the fully saturated case: all 128 rows of `A` select the same channel, the winner
  takes weight 0.998, `P_cross` is constant along `D`, and φ's relative gradient falls to 2.8e-04 of
  ψ's (at three times that feature scale it is exactly 0, so the stage can never leave the state).
  **Which row describes the real encoder is not established** and is the measurement of 15c.
* **Why VIP-Seg does not hit this.** Its `que.reshape(72, -1)` [VIPSEG models/vipseg.py:284] is not a
  transpose: it folds the batch axis into the filter axis, so the matrix it builds is not the channel
  correlation it is annotated as (max abs difference 513.99 against the clean form on the same inputs,
  same φ, same scale). The scrambling decorrelates the logits — row std 1.191 against 4.543 — and keeps
  the softmax usable. D-01 deliberately did not copy it. VIP-Seg also carries a channel-preserving term
  `proto_self = σ(A_s) ⊙ ψ(P)` [VIPSEG models/vipseg.py:270-274]; Eq.19 fuses only `P_cross` and
  `P_diffuse`, so CascadeProto has no equivalent.
* **Decision.** Add the switch `cross_attn_norm = {none (default), layernorm}` as a **probe, not a
  fix**. `layernorm` standardises `Q'` and `S'` along the projection axis `r` with one shared LayerNorm
  before the correlation, which makes `A` exactly invariant to the scale of the features and gives each
  stage 144 parameters (`γ`, `β`) with which to choose its own sharpness — control that the fixed
  `1/√d` does not provide. It removes the saturated regime but not the rank-1 update, and in the
  β std 0.5–2.0 rows above it has **less** channel variation than `none`, so it is not established as
  an improvement. The default stays `none`, the literal Eq.14.
* **Outcome on the VM (15c, real encoder, 2,400 train / 300 valid episodes, 2026-09-20).** The
  saturation hypothesis is **refuted** and `layernorm` is **rejected as a default**.

  | variant | valid mIoU | `attn_width` init → end | `P_cross` channel variation init → end |
  | :--- | ---: | :--- | :--- |
  | `baseline_l2` | 0.5223 | — | — |
  | `full` | 0.5346 | 127.9997 → 9.59 | 6e-04 → 0.369 |
  | `full_norm` | 0.5134 | 92.14 → 109.31 | 0.649 → 0.012 |

  Readings. (i) The real encoder puts Eq.14 at the **uniform** end at initialisation, not the
  saturated one: width 127.9997 of 128, `P_cross` channel variation 6e-04, i.e. collapsed, but by the
  other limit than the CPU probe suggested. (ii) The stage **escapes on its own**: after 600 steps the
  width is 9.59 and the channel variation 0.369, so φ does learn and the module is not dead. (iii) It
  still buys almost nothing — `full` is 1.2 points above `baseline_l2`, and the same `full`
  configuration scored 0.5164 on an earlier run of the identical command, so the run-to-run spread of
  the loop is larger than the effect. (iv) `layernorm` makes it **worse**: it holds the attention near
  the uniform end (92 → 109 instead of 128 → 9.6), the channel variation ends at 0.012 instead of
  0.369, and the score drops to 0.5134.
* **Decision.** `cross_attn_norm` stays in the code as an ablation flag with default `none`, the
  literal Eq.14. Neither value raises. It is not a fix and must not become the default.
* **Three seeds per variant (15e, 2026-09-20, `results/phase15_diag/`).** The cascade contributes
  nothing measurable, and the module is not broken.

  | variant | seeds | mean | sd | `attn_width` init → end | `P_cross` chan_var init → end | `w_diffuse` init → end |
  | :--- | :--- | ---: | ---: | :--- | :--- | :--- |
  | `baseline_l2` | .5316 / .5187 / .5111 | 0.5205 | 0.0104 | — | — | — |
  | `full` | .5029 / .5244 / .5241 | 0.5171 | 0.0123 | 128.00 → 29–54 | 0.0006–0.0019 → 0.37–0.61 | 0.47–0.51 → 0.008–0.082 |

  `full − baseline_l2 = −0.0033`, standard error 0.0093, `t = −0.36`. A 95 % interval is about
  ±2.6 points, so the +4.04 that Table 4 attributes to gate + cascade + ADRM lies outside it.
  Meanwhile every stage diagnostic is healthy on all three seeds: the attention leaves the uniform
  collapse (128.00 → 29–54), `P_cross` gains channel structure (three orders of magnitude), and the
  fusion learns to shut the class-blind branch off.
* **That closes two candidates.** (a) `P_diffuse` having no class index [DECISION D-16] is **not** the
  cause: the Eq.19 weight on it falls from ~0.49 to 0.008–0.082, i.e. training removes the branch by
  itself, and removing it earlier by hand would change nothing. (b) The cross-attention is **not**
  stuck: it is healthy after 600 steps on every seed.
* **What is left.** The module behaves as specified and adds nothing, while VIP-Seg's own PEM/PDM add
  about 17 points over the same `baseline_l2` through the same loop. The one structural difference
  left is the channel-preserving self-correlation term `proto_self = σ(A_s) ⊙ ψ(P)`
  [VIPSEG models/vipseg.py:270-274], which Eq.19 of the paper does not have. Adding it would mean
  implementing a module the paper does not describe, so the finding is recorded rather than patched:
  **Table 4's increments are not reproducible from the equations as printed.** See also D-17, where
  the paper's own baseline row (81.28 Avg) already sits above VIP-Seg's published 74.15.

---

### D-19 — A channel-preserving term in Eq.19 · `PROPOSED`, beyond the paper

* **Problem.** Only the part of a prototype that differs between class rows can change
  `argmax_c <f, p_c>`, because `<f, p_c> = <f, μ> + <f, d_c>` with `μ = mean_c p_c` common to every
  class. Measured on synthetic post-ReLU features: `P^0` carries 14.2–14.9 % of its energy in `d`,
  one EPPM stage leaves 3.8–4.2 %, and VIP-Seg's PEM leaves 6.9–8.9 %. Both summands Eq.19 prints are
  class-poor — `P_diffuse` has no class index at all [DECISION D-16], and `P_cross` is a row-stochastic
  mixture over **channels**, near rank 1 [DECISION D-18] — so the fused term dilutes `d` rather than
  sharpening it.
* **What the switch does.** `eq19_self = {none (default), gated}` adds `ψ(P^{t-1}_gated)` to
  `P_combined` before Eq.20, reusing the same `ψ` as Eq.14, so the parameter budget is unchanged
  (79,395 per stage). It is the CascadeProto-shaped analogue of VIP-Seg's
  `proto_self = σ(A_s) ⊙ ψ(P)` [VIPSEG models/vipseg.py:270-277], which Eq.19 has no counterpart for
  [audit F1].
* **Honest status of the evidence.** The static CPU analysis that motivated it did **not** confirm the
  fix: adding the term moves the class-varying share only from 3.9 % to 4.2 %, and reproducing
  VIP-Seg's 7–8 % additionally needs a LayerNorm on the new term and the removal of Eq.21's ReLU, the
  SE block and `W_out`'s bias — at which point a third metric (how many point predictions survive a
  stage) moves the other way. Three static metrics disagree, and 15e already showed that a stage whose
  diagnostics all look healthy after training still buys nothing. **This switch is an experiment, not
  a claim**; it is off by default and is not part of any paper table.
* **Ablation flag.** `eq19_self = {none (default), gated}`. Neither value raises. Any run using
  `gated` is outside the paper and must be reported as such.

---

### D-20 — Initialising from VIP-Seg's trained encoder · `PROPOSED`, diagnostic only

* **Problem.** Table 4's baseline, "a plain VIP-Seg backbone with masked average pooling and
  single-step prototype matching" [PAPER §4.3], scores 82.72 on S0, above VIP-Seg's own full model
  at 72.20 [PAPER Tab.6]. Trained from scratch, ours scores 49.08. No mIoU definition closes the gap
  (15q). One reading of "the shared encoder is VIP-Seg" [PAPER §4.1] that could: the encoder was taken
  from VIP-Seg's trained checkpoint, not trained from scratch.
* **Conflict.** That reading contradicts the paper's own "pre-training-free framework … without any
  pre-trained weights" [PAPER §1] and guardrail #1 of this repository. It is therefore a diagnostic,
  never a result configuration.
* **What the switch does.** `train.py --init_from_vipseg <checkpoint>` copies VIP-Seg's trained
  encoder, feature head and fixed projections into `model.features` (strict), before any resume, and
  tags the run directory `_vipinit`. `experiments/vipseg_init_probe.py` scores the baseline on those
  features with no training at all. VIP-Seg's S0 checkpoint saw only the S0 training classes, so the
  S0 test classes are not leaked.
* **Ablation flag.** `--init_from_vipseg` (default: none). Any number obtained with it is outside the
  paper and outside this repository's guardrails and must be reported as such.

---

### D-21 — Test classes seen during training · `PROPOSED`, diagnostic only

* **Problem.** The absolute gap to the paper is almost uniform across Table 4 (33.6, 34.3, 28.6,
  30.9, 31.4 points), the signature of a data or protocol difference rather than a model difference.
  §4.1 says "using Areas 1, 2, 3, 4, 6 for training and Area 5 for testing under two category splits
  S0 and S1" [PAPER §4.1]. If the split was by area and the category split did not take effect, the
  S0 test classes were seen during training and "few-shot" becomes segmentation of learnt classes.
* **What the switch does.** `train.py --train_classes all` lets training episodes sample the fold's
  test classes as well (all 12 classes, clutter excluded as in the loader). The inherited loader is
  unchanged; only the instance's `classes` array, the one thing its sampler reads, is widened
  [VIPSEG dataloaders/loader.py:163]. Run directories are tagged `_leak`. Test episodes are the usual
  fixed100 cache, which draws from every area, so the diagnostic leaks classes *and* possibly blocks:
  it is an upper bound of the area-split scenario.
* **Prediction.** If this is what produced the paper's numbers, the baseline lands at 75-85 and the
  increments between rows stay small.
* **Ablation flag.** `train_classes = {split (default), all}`. `all` is a protocol violation, never a
  result configuration.
* **Result (2026-09-21).** 20 epochs each, fixed100: baseline 0.6354 (+14.46 over the split run),
  full 0.6924 (+12.09); full − baseline +5.70 against the paper's +5.81. Leakage explains part of the
  level but leaves the leaked full model below the paper's baseline (82.72), so it is not the whole
  explanation (report §3.5, CHANGELOG 15x).
* **Correction and full form (2026-09-22).** The 20-epoch run is a partial leak, not an upper bound: each
  test class fills 1,600 way slots against 8,000 for a normally trained class. The full form needs no
  training: an S0-trained checkpoint scored with `--cvfold 1` (and an S1-trained one with `--cvfold 0`) is
  scored on its own training classes. Baseline `last.pt`: 77.32 (S0 model on fold 1) and 71.58 (S1 model
  on fold 0) against the paper's 82.72 / 79.83; unseen 49.08 / 51.91. Seen-class scoring matches the
  paper to 5–8 points and reproduces its S0 > S1 order (report §3.6, CHANGELOG 15y). Diagnostic only.

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
| 2026-09-19 | D-17: no `W_g` for T = 1 (identical prediction, no dead parameter). D-16 biases of `W_1`, `W_2`, `W_out` kept although Eq.20–21 print none (maintainer decision). |
