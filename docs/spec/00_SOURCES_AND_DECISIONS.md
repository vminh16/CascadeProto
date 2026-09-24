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

### D-22 — Protocol guard against seen-class scoring · `PROPOSED`

* **Problem.** Report §3.6 showed that scoring a checkpoint on classes it was trained on lifts the
  baseline by 22–25 points, and that this reading reproduces the paper's level. Nothing in `eval.py`
  prevented it: `--cvfold` was read from the command line, independently of the fold the checkpoint
  was trained on, and S0's training classes are exactly fold 1's test classes
  [VIPSEG dataloaders/s3dis.py:20-31]. Phase 16 compares new designs whose gains are expected to be a
  few points, so one silent fold swap would dominate any result.
* **Decision.** `eval.py` scores a checkpoint only on the test classes of the fold it was trained on.
  The training fold is read from the checkpoint (`args.cvfold`, which `train.py` stores in `best.pt`
  and `last.pt`) or, for checkpoints that record none (VIP-Seg's release saves only the model,
  iteration and IoU [VIPSEG runs/training.py:97-100]), stated with `--checkpoint_cvfold`; a stated
  fold that contradicts the recorded one raises. A different fold, or a checkpoint trained with
  `--train_classes all` [DECISION D-21], raises before any episode is built.
* **Diagnostic escape.** `--allow_seen_classes true` permits such a run; the log and the result JSON
  then carry `protocol_check = "SEEN-CLASS DIAGNOSTIC, not a few-shot result: …"`. Clean runs carry
  `"clean"`. `experiments/run_seen.sh` is the only script that sets the flag.
* **Reporting rules for phase 16** (recorded here so that they bind every later run):
  1. Headline numbers are `last.pt`. The inherited validation episodes are drawn from the test classes
     [VIPSEG runs/training.py:52-66]; `best.pt` is selected on them and is reported only next to
     `last.pt` [DECISION D-15].
  2. Design choices are screened on fold S1 (the `valid` episode draw) and the S0 test episodes are
     not looked at until the design is frozen, so S0 is a held-out fold for every phase-16 decision.
     S1 results are reported with that caveat.
  3. New hyper-parameters are never tuned on the classes being scored.
* **Amendment of rule 1 (maintainer, 2026-09-24).** VIP-Seg's published 76.09 / 72.20 are the best of 12
  validations on episodes of the test classes, re-tested once: its released S1 log reads 72.84 at the
  last update and 75.63 at the selected one, its S0 log 68.97 last and 72.94 selected, mean of the 12
  validations 69.29 [VIPSEG log_s3dis_VIPSeg/log_S{1,0}_N2_K1_*/log_vipseg.txt]
  (`docs/research/2026-09-24_r2_distill_analysis.md` §3). Rule 1 compared our `last.pt` with numbers
  selected that way. From now on every result reports **both**:
  * `best.pt` under VIP-Seg's own disclosed selection rule, the protocol of the published baselines:
    about 12 validations over training (`--valid_every` chosen so), each on the 1,500 `valid` episodes of
    the fold's test classes [DECISION D-15], the best one kept. This is the number compared with the
    published table, labelled "best-of-validation, VIP-Seg's protocol".
  * `last.pt`, the number free of selection on test classes, labelled that way.
  A claim of beating a baseline states which of the two it rests on; a claim on `best.pt` alone is
  marked as resting on selection. Rules 2 and 3 are unchanged.
* **Affects.** `eval.py`, `experiments/run_seen.sh`, `experiments/rescore_alt_metrics.sh`, 04 §6.1,
  05 §3.12 (PROT-1…6), README §5.

---

### D-23 — One `S′` for the episode instead of one per class slot (Eq.13) · `PROPOSED`, to be measured

* **Problem.** D-01 item 5 already recorded that Eq.13 writes a single `S′ = φ(F^s)` with no class
  index, and that one `A` per class slot follows VIP-Seg rather than the paper. Phase 16 found that
  this is the one place where our implementation differs from **every** published member of the
  family it belongs to.
* **Evidence (L2 and literature, 2026-09-22).** VIP-Seg's PDM is token-identical to Seg-PN's QUEST
  (`models/quest.py` of github.com/yangyangyang127/Seg-NN) and its PEM to TaylorSeg's APP
  (`models/app.py` of github.com/changshuowang/TaylorSeg); DyPolySeg's PCM prints the same two
  branches. In all of them the correlation is computed after `que.reshape(72, -1)` and
  `sup.reshape(72, -1)` [VIPSEG models/vipseg.py:285-291,387-391]; differentiating each output block
  with respect to each input shows that **every** `A[b′, c′]` depends on both queries and on all
  three class slots, so the published operator is an episode-level correlation, six row-subsampled
  views of one statistic, not a per-class one (research note §4.4, Appendix A). DPA states the shared
  form cleanly — one correlation against the mean of all support features — and measures +13.7 mean
  IoU for it alone (arXiv 2401.16051, Table 3). The ablations that isolate this branch credit it with
  +15.4 (Seg-PN Table 6) and +15.2 (TaylorSeg Table 4) on S0, which is the size of our gap to VIP-Seg's
  head.
* **Why the difference can matter.** With one `A_b` shared by the class rows, the contrast of two
  classes is `fᵀA_bW(p_c − p_c′)`: an episode-adaptive bilinear metric on the support-derived contrast.
  With one `A_{b,c}` per class slot the contrast also contains `(A_{b,c} − A_{b,c′})ψ(·)`, where
  `A_{b,c}` is the channel co-activation of the query with the **whole block** sampled for way c,
  background included, so a class is pulled toward the query whenever its support *scene* resembles it
  — a class bias no label supports.
* **Decision.** Add the switch `cross_attn_support = {class_slots (default), pooled}`. `pooled` is
  Eq.13's literal single `S′`: the pooled tokens of all N·K support blocks are averaged into one
  `F̄^s`, `S̄′ = φ(F̄^s)`, and `A_b = softmax_row(Q′_bᵀ S̄′ / √d) ∈ R^{D×D}` is applied to every class
  row, `P_cross[b,c] = A_b ψ(P_gated[b,c])`. Unlike the published reshape it keeps the queries
  separate, so one query's prediction never depends on another's. No parameter changes.
* **Status: measured and negative (2026-09-23, R1, `results/phase16_r1/SUMMARY.md`).** On S1, T = 1,
  no LMA, 2,400 steps, two seeds, everything else fixed: `pooled` 0.5339 ± 0.0015 against
  `class_slots` 0.5305 ± 0.0070, i.e. **+0.33 points, t = +0.65**, against the rule's +3 with t > 3.
  The same run resolves a 12.50-point difference between a VIP-Seg module and no stage at all
  (t = +7.86), so the budget is not what hides it. The support reading is **not** the cause of the gap
  to VIP-Seg's head. D-01 item 5 stands; the flag stays as an ablation with default `class_slots`.
* **Affects.** `models/eppm.py` (`CrossAttention`), `models/cascadeproto.py`, `train.py`
  (run directories get `_pooled`), 01 §3, 02 §5.2, 05 §3.4 (XATT-11…15).

---

### D-24 — EPPM-S, a stripped purification stage · `PROPOSED`, beyond the paper

* **Problem.** The printed stage is 15 points behind VIP-Seg's on the same features (CHANGELOG 15e),
  and the research note shows why each of its parts cannot help: `P_diffuse` is common to every class
  and provably cannot change a prediction, so training drives its fusion weight to 0.008–0.082
  (CHANGELOG 15e); the entropy gate is an even pointwise function of the prototype value with one
  scalar [DECISION D-02]; Eq.20's SE gate is pooled over the classes; Eq.21's ReLU truncates the
  update. The family this module belongs to instead carries a **channel-preserving** term beside the
  correlation [VIPSEG models/vipseg.py:262-277], which [DECISION D-19] measured as missing.
* **What the switch does.** `stage_type = {eppm (default), eppm_s, vip}`; `eppm_s` builds
  `models/eppm_s.py::EPPMSharedStage`:
  `P^t = LN(W(P_cross + σ(W_3(Q′ᵀQ′ − S′ᵀS′)/√D) ⊙ ψ(P^{t-1})) + P^{t-1})`, with `P_cross` as in
  Eq.13–14 under either `cross_attn_support` [DECISION D-23]. Dropped: `P_diffuse`, the entropy gate,
  Eq.19's fusion MLP, Eq.20's SE, `w_cls` and the ReLU. 37,888 parameters per stage against 79,395.
  Like VIP-Seg's head it expects L2-normalised prototypes at the top of the cascade
  [VIPSEG models/vipseg.py:142]; supply them with `--l2norm_point_proto true` [DECISION D-10].
* **Honest status.** This is **not** a reading of the paper: it is the CascadeProto-shaped member of
  the QUEST / APP / PEM / PDM family, built to locate the 15 points. Any run using it is outside the
  paper and must be reported as such. It is measured against the printed stage and against
  [DECISION D-25] in R1 (16e).
* **Outcome: negative (2026-09-23, R1, `results/phase16_r1/SUMMARY.md`).** EPPM-S scores 0.4927 ± 0.0115
  against the printed stage's 0.5305 ± 0.0070 and plain L2 prototype matching's 0.5602 ± 0.0222, i.e.
  **−3.79 points against the stage it replaces and −6.75 against using no stage at all**, and 19.26
  below one VIP-Seg PEM on the same run. Removing the parts that provably cannot change a prediction
  does not produce a stage that helps: the static argument of this decision is not sufficient. The
  switch stays as an ablation, off by default. Three candidates for the difference to PEM remain
  untested: the LayerNorm on the self term (named in CHANGELOG 15i, omitted here), PEM's separate
  query and support gates against the difference gate used here, and the query mixing of VIP-Seg's
  reshape, which this stage deliberately avoids.
* **Guard.** With `stage_type ≠ eppm` the EPPM-only switches (`use_gate`, `gate_target`, `eq19_self`,
  `cross_attn_norm`, `fusion_weight`, `diffusion_input`) must stay at their defaults; a non-default
  value raises rather than being ignored, so a run's configuration always describes what it ran.
* **Affects.** `models/eppm_s.py` (new), `models/cascadeproto.py` (`build_stage`), `train.py`
  (run directories get `_eppm_s`), 01 §3–§4, 05 §3.4b (EPS-1…9).

---

### D-25 — VIP-Seg's PEM/PDM as cascade stages · `PROPOSED`, reference only

* **Problem.** "VIP-Seg's head is worth about 17 points more than ours through the same loop"
  (CHANGELOG 15e) was measured by training VIP-Seg's **whole model**, which also differs in its
  prototype normalisation, its logits and its gating. To attribute the gap to the stage itself, the
  same pipeline must run VIP-Seg's stage and ours with everything else held fixed.
* **What the switch does.** `stage_type = vip` wraps the inherited
  `models/vipseg.py::PrototypeEnhancementModule` / `PrototypeDifferenceModule` in
  `models/vip_stage.py::VIPStage`, which reproduces VIP-Seg's alternation (PEM on even steps, PDM on
  odd ones) and the outer residual it adds to the PDM output [VIPSEG models/vipseg.py:154-160]. The
  modules are imported, never edited (AGENTS guardrail 2), and imported lazily, because
  `models.vipseg` needs `pointnet2_ops` (gate G2). `cross_attn_scale` and `cross_attn_support` are
  fixed inside VIP-Seg's code, so a non-default value raises.
* **Known property, kept on purpose.** VIP-Seg's `reshape(72, -1)` makes one query's prediction depend
  on the other queries of the episode (research note §4.4). This is a reference configuration, not a
  design to copy; our own stages keep the queries separate.
* **Reporting.** A number produced with `stage_type=vip` is VIP-Seg's module inside our pipeline, not
  CascadeProto, and must be labelled that way.
* **Measured (2026-09-23, R1, `results/phase16_r1/SUMMARY.md`).** One PEM on our prototypes:
  **0.6852 ± 0.0036** on S1 after 2,400 steps, **+15.47 over the printed stage (t = +27.78)** and
  +12.50 over no stage at all. The 15 points that CHANGELOG 15e measured between two training scripts
  are reproduced inside one pipeline with only the stage changed, with the caveat that the printed
  stage runs without the L2 input that VIP-Seg's head gets, worth up to 3.4 of those points
  (`results/phase16_r1/SUMMARY.md`). Rule R1.4 of `run_r1.sh` therefore
  fires: phase 16 continues on **route B**, adding to VIP-Seg's head rather than repairing the printed
  stage. VIP-Seg's released S1 checkpoint is cited at 0.7609, so this is a real starting point and the
  head remains VIP-Seg's contribution.
* **Affects.** `models/vip_stage.py` (new), `models/cascadeproto.py`, `train.py` (run directories get
  `_vip`), 01 §3, 05 §3.4c (VIPS-1…5).

---

### D-26 — Query-side, entropy-weighted EM refinement of the prototypes · `PROPOSED`, beyond the paper

* **Problem.** Route B (D-25, R1) gives a head that works; what is added on top of it has to be new and
  has to address the error that remains. Three pieces of evidence point at the query side:
  1. **Where the one-shot error is.** Replacing the support prototypes by the query's own class means
     lifts S3DIS S0 1-way 1-shot from 66.40 to 93.89 [QGE T1]; four extra support shots are worth only
     +1.5 to +4.3 (VIP-Seg's released logs, DyPolySeg T1). The support side is bounded below by the class
     mean, the query side is not (research note 2026-09-22 §2.1).
  2. **Class-selective aggregation of query points works elsewhere.** SSP's self-support prototypes
     +3.1 / +4.0 (PASCAL 1- / 5-shot) [SSP §3.2, T13]; DPA's prototype-to-query attention +3.05 [DPA T3];
     AttMPTI's label propagation 52.27 against ProtoNet's 48.39 [AttMPTI T1] (research note §4.5).
  3. **The printed method's entropy is on the wrong object.** Eq.10–12's per-channel entropy is an even
     function of the prototype value that can only attenuate (D-02, research note §3.1). The quantity
     that is high where the model cannot tell the classes apart is the entropy of a query point's class
     posterior, H(r_i).
* **What it does.** `models/transductive.py`: on top of a model's own scoring rule `L = F^q M^T`,
  T steps of `r = softmax(L)`, `w_i = 1 − H(r_i)/ln(N+1)`, `u_c = (1/P) Σ_i w_i r_ic f_i/‖f_i‖`,
  `μ_c = ‖m_c‖ · normalise(m_c/‖m_c‖ + κ u_c)`, `L = F^q μ^T`. Spec 02 §11.
  * The update is the MAP mean direction of a von Mises–Fisher mixture with a vMF prior on each mean
    centred on the model's prototype (research note §5.2); the prior term is the anchor against the
    majority collapse of pure entropy minimisation [TIM §3.4, T4]. `‖u_c‖` is at most the confident mass
    fraction of class c, so a class the query barely contains barely moves.
  * `‖m_c‖` is kept: the trained rule is an unnormalised dot product [VIPSEG models/vipseg.py:162-164]
    [PAPER Eq.23] whose prototype norms are part of the decision. `κ = 0` or `T = 0` returns the
    model's own logits exactly (EM-1).
  * Arms of the weight: `entropy` (this decision), `none` (its ablation), `ssp` (SSP's hard thresholds,
    0.7 foreground / 0.6 background [SSP §3.2]) and `oracle` (the query labels: an upper bound, never a
    result).
* **P0, the probe, before any training** (`experiments/p0_em_probe.py`, `experiments/run_p0.sh`). No
  parameter is trained, so an arm is compared with the model **on the same episodes and weights**: the
  seed noise of training does not enter, and a paired bootstrap over episodes (2,000 resamples) gives
  the uncertainty. This answers the objection that a gain of 1–3 points is indistinguishable from
  training noise (CHANGELOG 16f: two seeds of `r1_baseline_l2` differ by 3.1).
  * Checkpoints: VIP-Seg's released S1 and S0 checkpoints (route B's head, trained by its authors) and
    our S1 and S0 baselines (`num_stages = 0`), so the effect is measured on two different feature
    extractors. Other configurations of ours raise rather than being approximated.
  * **Selection** on the S1 `valid` draw of the S1 checkpoints only [DECISION D-15] [DECISION D-22]:
    weight {entropy, none, ssp} × κ {0.5, 1, 2, 4, 8, 16} × T {1, 2, 3}. κ extends the research note's
    {0.25, 0.5, 1} because the pull on a class is at most κ·π_c, π_c its confident mass fraction: a
    foreground object that covers a tenth to a third of a query block (an estimate, **not measured**)
    would move by a few percent at κ ≤ 1. The probe records the measured π_c (`confident_mass`), so the
    grid can be checked against it before the test stage is read. T ≤ 3
    because soft k-means refinement saturates after one step [Ren §3.1.1].
  * **Test** of the frozen setting once on the fixed100 test draw: S1 checkpoints on S1, S0 checkpoints
    on S0, which is looked at here for the first time. Every checkpoint passes `eval.py`'s protocol guard.
  * **Rules, fixed before the run.** P0.1 go: gain ≥ +1.5 on both S1 checkpoints and ≥ +1.0 on both S0
    checkpoints. P0.2 stop: < +0.5 on both S1 checkpoints; the direction is dropped and base-class
    calibration is next (research note 2026-09-23 §5, direction 4). P0.3 otherwise: train with learned
    κ, expect the low end. P0.5: "the entropy weight helps" is claimed only if the frozen setting beats
    the same setting with `weight = none` with a 95 % CI above 0 on both S0 checkpoints. P0.6: an oracle
    gain below 10 on VIP-Seg's checkpoints means its features, not its prototypes, bound it. Collapse
    watch: a positive mean gain with a class falling by more than 3 points is reported (TIM §3.4). The
    go/stop thresholds are those of research note 2026-09-23 §7.1.
  * The probe also reports how the points split by weight (w < 0.5, 0.5–0.9, ≥ 0.9) and how accurate
    each bin is. That is the direct test of the claim that the posterior entropy singles out reliable
    points, which the paper asserts for its own entropy and never measures.
* **Not in P0: the text prior.** VIP-Seg's checkpoints carry no text branch, and a text prototype
  needs a trained adapter, so the text prior cannot be probed without training. It enters with R2 as
  the prior `m_c`, weighted by how well the text prototype alone predicts the support mask, the form
  that gained +1.9 / +2.0 in MM-FSS where fixed text weights gained +0.1 to +0.9 [MMFSS Eq.9–10, T3e].
  It needs its own decision before code.
* **Bug found while implementing (2026-09-23).** A first M-step normalised `u_c` to unit length before
  weighting it by κ. EM-3 (a uniform posterior must leave every prototype unchanged) failed: an entropy
  weight of 10⁻¹⁶ still produced a unit update. Such a form hands a class with a vanishing confident mass
  a full-weight update, contradicting the vMF derivation above; it was replaced by the unnormalised
  mean of unit features before any run.
* **Reporting.** A number with this refinement is VIP-Seg's head (or our baseline) plus D-26, not
  CascadeProto, and is labelled that way. Nothing trained here yet; R2 (training with the refinement
  inside the loop and per-step supervision) follows only on P0.1 or P0.3.
* **Outcome: P0.2 stop (2026-09-23, `results/phase16_p0/SUMMARY.md`).** Selection froze `ssp_k0.5_T1`
  (mean valid gain +0.28; every pseudo-label arm worsens as κ or T grows). Test, fixed100: VIP-Seg S1
  +0.37 [+0.24, +0.50], ours S1 +0.06 [−0.04, +0.16], VIP-Seg S0 **−1.21 [−1.57, −0.87]**, ours S0 −0.16
  [−0.28, −0.03]. With the query labels the same update gains +8.42 to +21.76, so the query-side headroom
  is real but the model's own posterior cannot reach it. The posterior entropy does rank points by
  reliability (accuracy 52–73 % at w < 0.5 against 88–95 % at w ≥ 0.9 on all four checkpoints), yet the
  soft entropy weight lost to SSP's hard thresholds and P0.5 is not claimable. R2 is not run; the next
  direction is base-class calibration. The module and probe stay, off every default path.
* **Affects.** `models/transductive.py` (new), `experiments/p0_em_probe.py` (new),
  `experiments/run_p0.sh` (new), 02 §11, 05 §3.8g (EM-1…14).

---

### D-27 — Base-class calibration of the background · `PROPOSED`, beyond the paper

* **Problem.** In a novel-class episode the background of a query block holds the fold's base classes
  (S3DIS S0 tests beam, board, bookcase, ceiling, chair, column against a background of floor, wall,
  window, door, table, sofa and clutter). A model trained on the base classes is drawn to them: COSeg
  shows false activations of base classes such as wall and door in novel-class queries [COSeg §4.3,
  Fig.5]. Evidence that correcting this pays:
  1. COSeg's Base Prototypes Calibration adds **+3.44 / +2.06** (1- / 5-shot) on top of its correlation
     model [COSeg T3], with the momentum of its base prototypes insensitive between 0.99 and 0.999
     (47.21 / 46.91 / 47.40) [COSeg T6]. Measured in COSeg's corrected setting, on a weaker base level;
     research note 2026-09-22 §5.6 predicts +0.5 to +2 at the 72 level.
  2. It uses only base-class labels, which training episodes carry anyway (research note §5.6), and it
     works on the side P0 did not touch: D-26 changed the foreground prototypes from the query's own
     pseudo-labels and failed (P0.2); this decision uses labels the model was trained on.
* **What it does.** `models/base_calibration.py`, spec 02 §12:
  * **Geometry.** Base, support and query features are compared after centring on the mean feature of
    the base training episodes and L2 normalisation, `f̃ = normalise(f − μ)`: SimpleShot's CL2N, which it
    reports to improve nearest-centroid few-shot classification over unnormalised and L2-normalised
    features [SimpleShot §3].
  * **Bank.** `b_j = normalise(mean_o normalise(Σ_{i∈o} f̃_i))` over the occurrences o of base class j
    (support and query masks) in training episodes: the limit of COSeg's per-episode masked average
    under its EMA [COSeg Eq.9–10] when the features are frozen.
  * **Calibration.** The episode's support prototypes `u_c` are built the same way. The base margin of a
    point the model assigns to foreground class c is `m_i = max_j cos(f̃_i, b_j) − cos(f̃_i, u_c)`; per
    query, the points with `m_i > 0` among the top fraction q of its foreground predictions by `m_i` move
    to the background. Background predictions and foreground logits are never changed. COSeg adds the
    base guidance through a trained layer (Eq.12), which a probe cannot do.
* **Revised before any real run (2026-09-23).** The first version added `ω · max_j s⟨f_i, b_j⟩` (raw
  unit-mean base directions at the foreground prototypes' norm) to the background logit of the model's
  own rule, arguing from P0's oracle that such directions are valid prototypes of that rule. The P1
  smoke run (5 episodes) refuted it: VIP-Seg fell from 84 to about 0 mIoU at every ω ≥ 0.8, everything
  turned background, and the AUC of the base similarity was 0.14–0.39, below chance. Two causes:
  post-ReLU features are non-negative, so any point's cosine with a raw mean direction is high; and
  P0's oracle had replaced *all* prototypes, background included, so it never set the two geometries
  against each other. The revision compares like with like (CL2N; base against the episode's own
  support prototypes) and only ever removes foreground. The bank records the mean raw cosine of the
  query features to the centre, which quantifies the shared component.
* **Second revision before any real run (same day).** The second smoke run (20 episodes) of the CL2N
  form ran cleanly but showed the absolute margin grid mis-scaled: δ = 0 / 0.05 / 0.1 / 0.2 moved
  73 / 63 / 51 / 33 % of VIP-Seg's S1 foreground predictions to the background, while only 3.7 % were
  false. Any flip beyond the false share removes mostly true foreground, so the grid became the fraction
  q of foreground predictions, anchored on that measured share and independent of the margin's scale.
* **P1, the probe** (`experiments/p1_bpc_probe.py`, `experiments/run_p1.sh`), with P0's machinery:
  the same four checkpoints, scoring rules read through hooks and checked against the model's logits on
  every episode, paired bootstrap over episodes, eval.py's protocol guard.
  * **Bank:** 1,000 seeded training episodes of the checkpoint's **own fold**, no augmentation, two
    passes (the centre μ, then the prototypes), at least 100 occurrences per base class (else it
    raises); 1,000 episodes is five times the memory of COSeg's EMA, 1/(1 − 0.995) = 200 updates
    [COSeg T6]. The bank must not contain a scored class (it raises).
  * **Selection** of q ∈ {0.5, 1, 2, 4} % on the S1 valid draw of the S1 checkpoints, a range around the
    share of false foreground measured in the second smoke run (3.7 % of VIP-Seg's S1 foreground
    predictions). **Test** of the frozen q once on fixed100: S1 checkpoints on S1, S0 checkpoints on S0.
    The probe reports, per checkpoint, the share of false foreground and the share of foreground
    predictions with a positive margin, which bound what any threshold on the margin can do.
  * **Diagnostic.** Among the model's foreground predictions, the AUC of the base margin for separating
    false foreground (ground truth background) from true foreground. It measures whether the base
    margin carries the signal at all, independently of the threshold.
* **Rules, fixed before the run.** P1.1 go: gain ≥ +0.5 with a 95 % CI above 0 on all four checkpoints,
  the low end of the predicted +0.5 to +2 and resolvable at the CI half-widths P0 measured (0.10–0.43).
  P1.2 stop (of the training-free form): no S1 checkpoint gains with a CI above 0. P1.3 otherwise.
  P1.4: AUC ≥ 0.70 on both VIP-Seg checkpoints means the base similarity flags false foreground and a
  trained calibration (COSeg Eq.12) is justified even if P1.2 fires; AUC < 0.60 on both drops D-27
  entirely; in between, report only (0.7 is the conventional threshold of acceptable discrimination,
  0.5 is chance). P1.5 collapse watch: a positive gain with a test class falling by more than 3 points,
  the expected failure when a novel class resembles a base class (board and column against wall).
* **Not in P1:** training. R3, the trained form (EMA bank during training, momentum 0.995, COSeg Eq.9–12,
  exclusion of the current ways' base prototypes during training), follows only on P1.1, P1.3 or P1.4.
* **Reporting.** A number with this calibration is the checkpoint's model plus D-27, labelled that way.
* **Affects.** `models/base_calibration.py` (new), `experiments/p1_bpc_probe.py` (new),
  `experiments/run_p1.sh` (new), `experiments/p0_em_probe.py` (the scoring rules keep `F^s`), 02 §12,
  05 §3.8h (BPC-1…10).

* **Outcome: P1.2 stop, P1.4 weak (2026-09-23, `results/phase16_p1/SUMMARY.md`).** Frozen q = 0.5 %.
  Test gains: VIP-Seg S1 −0.16 [−0.17, −0.15], ours S1 −0.07, VIP-Seg S0 −0.07 [−0.09, −0.06], ours S0
  −0.03, all CIs below 0. AUC of the base margin for false against true foreground: VIP-Seg 0.716 / 0.649
  (S1 / S0), ours 0.494 / 0.629; false share of foreground predictions 10–27 %. The training-free form is
  closed; the signal is kept as D-28's filter.

---

### D-28 — EM refinement with a base-margin filter on the foreground M-step · `PROPOSED`, beyond the paper

* **Problem.** D-26 and D-27 failed for opposite reasons, measured on the same checkpoints:
  1. D-26's M-step averages the points the model assigns to a class; a false share ε of that mass drags
     the prototype toward the model's own errors, and the entropy weight comes from the same posterior,
     so its errors correlate with the model's. Every larger step (κ, T) was worse (P0).
  2. D-27's base margin is computed from information the posterior does not use (base-class labels and
     the episode's support), but it is too weak to decide single points: AUC 0.65–0.72 on VIP-Seg while
     10–18 % of its foreground predictions are false (P1).
* **Why the combination can work where the parts did not.** A weak signal loses when it *classifies*
  (one wrong flip costs one point of IoU) but can win when it *filters a mean*: the prototype is an
  average over a few hundred confident points, so excluding a true point costs variance ∝ 1/n, while
  excluding a false one reduces the bias ∝ ε. Pseudo-labels improve when a second view's errors are not
  those of the model (co-training, Blum and Mitchell 1998); the base margin is such a candidate view.
  Whether it cleans the M-step in practice is exactly what P2 measures first (P2.0).
* **What it does.** `models/transductive.py::em_step(fg_keep=...)`: at every EM step, the foreground
  M-step drops the points in the top fraction r of each query's foreground-assigned points by base
  margin (D-27's margin, recomputed from that step's logits, positive margins only); the background
  M-step and the E-step are unchanged. `r = 0` is D-26 exactly. Spec 02 §13.
* **P2, the probe** (`experiments/p2_fused_probe.py`, `experiments/run_p2.sh`), P0's and P1's machinery,
  P1's banks reused.
  * **Grid**, each value from P0/P1: weight {ssp (P0's winner), entropy (D-26's own)} × κ {0.5, 1, 2, 4, 8}
    (P0's grid without 16, which lost everywhere) × T {1, 2} (P0's best arms) × r {0, 0.1, 0.2, 0.3}
    (around P1's measured false shares, 10–27 %).
  * **Selection** on the S1 valid draw by the gain on the **VIP-Seg** S1 checkpoint only: route B builds
    on VIP-Seg's head (D-25), and our baseline's features carry no base signal (P1, AUC 0.49). Our
    checkpoints are scored for the report. **Test** of the frozen arm and of the same arm with r = 0.
  * **Mechanism diagnostic.** The false share ε of the first foreground M-step, weighted by the
    responsibilities, with and without the filter, from the query labels (diagnostic only).
* **Rules, fixed before the run.** P2.0 mechanism: at the frozen (weight, r), ε must fall to ≤ 0.75 of its
  unfiltered value on both VIP-Seg checkpoints, the reduction the expected +0.5 to +2 was estimated from;
  otherwise stop, whatever the mIoU. P2.1 go: P2.0 holds and the frozen arm gains ≥ +0.5 with a CI above 0
  on VIP-Seg S1 and S0: train the fused refinement (R2, the EM steps unrolled in training with per-step
  supervision and COSeg's EMA bank). P2.2 stop: VIP-Seg S1 has no CI above 0, or P2.0 fails. P2.3
  otherwise. P2.4: "the filter is what helps" is claimed only if frozen − unfiltered has a CI above 0 on
  VIP-Seg S0. P2.5 collapse watch as P0/P1.
* **Reporting.** VIP-Seg's head plus D-28, labelled that way; not CascadeProto.
* **Affects.** `models/transductive.py` (`fg_keep`), `experiments/p2_fused_probe.py` (new),
  `experiments/run_p2.sh` (new), 02 §13, 05 §3.8i (FUS-1…8).
* **Outcome: P2.0 fails, P2.2 stop (2026-09-23, `results/phase16_p2/SUMMARY.md`).** Selection froze the
  unfiltered arm (`ssp_k0.5_T1_r0`, +0.51 valid); the best arm per r falls monotonically, +0.51 / +0.50 /
  +0.49 / +0.47 for r = 0 / 0.1 / 0.2 / 0.3. The false share of VIP-Seg S1's first foreground M-step falls
  only to 0.85× at r = 0.3 (0.095 → 0.080) with `ssp` and not at all with `entropy` (0.116 → 0.114). The
  test reproduces P0's frozen numbers exactly. A base margin of AUC 0.65–0.72 removes false and true
  mass at nearly the same rate, so the filter's asymmetry never applies. Closed.

---

### D-29 — Oracle-direction distillation of the pairwise decisions during training · `PROPOSED`, beyond the paper

* **Problem, from the measurements of phase 16.**
  1. **The headroom is in the directions of the prototypes the head outputs.** VIP-Seg's scoring rule is
     `L = F^q M_effᵀ` with `M_eff = Σ_t w_t M^t` [VIPSEG models/vipseg.py:152-174]. Replacing the
     direction of `M_eff` by the query's own class mean of unit features, the norm kept, gains +8.42 (S1)
     and +15.13 (S0) on VIP-Seg's released checkpoints and +20.46 / +21.76 on ours (P0, fixed100,
     `results/phase16_p0/SUMMARY.md`); with one common norm, +10.95 / +14.12 (revision below). The
     features carry the information; the head does not extract it.
  2. **No fixed rule recovers it at test time.** The model's own posterior (D-26: +0.37 S1, −1.21 S0),
     a base-class margin (D-27: −0.16 to −0.03) and their combination (D-28: the filter was never
     selected, r = 0 froze) all fail on the same checkpoints.
  3. **The head is never told where the target is.** It is trained by CE on the final logits only
     (02 §7). On a training episode the target of point 1 is computable, because the query's labels
     are base-class labels of the training fold (D-22 is not touched). QGE distils toward "optimal
     query prototypes" (+3.3, T5, 1-way) and DPA distils earlier stages toward later ones (+2.65, T3).
* **What it does.** During training only, a loss pulls each pairwise decision function of the model
  toward that of the oracle-direction rule of the training query. Spec 02 §14.
  * `O_bc = normalise(Σ_{i: y_bi = c} f_bi / ‖f_bi‖)`, stop-gradient: the direction of P0's oracle
    replacement (`ORACLE_REPLACE` in `experiments/p0_em_probe.py`, κ → ∞ in 02 §11).
  * Teacher logits `T = F^q Oᵀ`, stop-gradient: the oracle rule with one common norm for the classes.
  * For each query b and each pair of classes c < c' present in it, over the points labelled c or c':
    `cos_bcc' = cos_i(L_bic − L_bic', T_bic − T_bic')`, uncentred, with `L = L_final`.
  * `L_distill = mean over (b, c < c') present of 1 − cos_bcc'`; `L_total = CE + λ L_GMMN + β L_distill`,
    β = `distill_beta`, default 0 (the reproduction unchanged).
  * `L_distill = 0` exactly when every pairwise decision function of the model is a positive multiple of
    the oracle rule's on those points, so the two rules predict the same class between c and c'.
  * Nothing changes at evaluation: `L_distill` is computed only in `train()` mode, and the logits never
    read `query_y` (test DIS-5).
* **Revised before any run (smoke run and measurement of 2026-09-23).** The first form of this decision
  (commit `660906a`, never trained) was `1 − cos(M_eff_c, O_c)` on the effective prototype
  `M_eff = Σ_t w_t P^t`. The smoke run's diagnostics refuted it:
  1. **The loss saw what the prediction cannot.** Adding one vector v to every prototype of a query adds
     `⟨f_i, v⟩` to every class logit of point i, so softmax, CE and argmax are unchanged, and components
     of `M` orthogonal to the features change no logit at all. `cos(M_c, O_c)` depends on both. Measured on
     VIP-Seg's released checkpoints (fixed100, `results/phase16_r2_pre/`): `cos(M_eff, O)` is 0.43 / 0.30
     (background / foreground) on S1 and 0.10 / 0.13 on S0, `cos(M_c − M_c', O_c − O_c')` 0.53 / 0.44, while
     the normalised support prototypes that the head starts from score 0.83 / 0.83 raw and 0.63 / 0.66
     pairwise. The trained head moves its prototypes away from the oracle by both measures, and the
     first measure differs by a factor of three to four between the folds while the head scores 75.36
     and 71.97 on them: neither tracks what decides the prediction. The logit-space form above is invariant to both effects.
  2. **The target is the directions, not the norms.** The same measurement scored the oracle rule with
     one common norm for the present classes at +10.95 (S1) / +14.12 (S0), against +8.41 / +15.13 with
     each class's norm kept (P0's rule, reproduced to 0.01). The teacher therefore carries no norm and no
     temperature, which also removes the scale choice a KL toward softmax would need.
* **Why these choices, each from evidence.**
  * **Logit space, pairwise, cosine.** Point 1 of the revision: the only quantities that decide the
    prediction between two classes are the signs of `L_c − L_c'` on the points; a cosine over points
    compares their orientation and ignores the common shift, the scale and the feature-orthogonal part.
  * **Points of the two classes only.** The pairwise decision between c and c' decides the prediction
    on points of c or c'; on a third class's points it is irrelevant when that class wins.
  * **β = 1.** No value of β has been measured here. 1 is the weight of the paper's other auxiliary term
    (λ = 1, [PAPER Eq.26]); `1 − cos` lies in [0, 2] and the CE of a full-schedule run ends near 0.13
    (`results/phase15_full/baseline_l2/log_train.txt`), so the two terms have the same order. Not tuned:
    one value, because each arm is trained once (maintainer, 2026-09-23).
* **R2, the measurement.**
  * **Arms.** R0 = VIP-Seg's four alternating modules in our loop (`stage_type=vip`, `num_stages=4`,
    `l2norm_point_proto=true`, `use_lma=false`, ADRM), the `r1_vip4` configuration of R1; D29 = R0 with
    β = 1. R0 is the reference that R1 lacked: route B's base trained by our loop.
  * **Schedule.** The full schedule of D-12 (50 × 480 episodes, batch 4, StepLR), seed 0, because R1 has
    no learning curve for VIP-Seg's head and R0 has to be a level comparable with VIP-Seg's released
    S1 checkpoint (75.36 fixed100, P0), not a 40 % screen. `last.pt` is the headline (D-22).
  * **One training run per arm** (maintainer). Training-seed noise is therefore not measured by R2; the
    estimate is R1's: `r1_vippem` seeds 0.6827 / 0.6878, sd 0.36, so the difference of two single runs
    has sd ≈ 0.36 · √2 ≈ 0.5 [inferred from two seeds].
  * **Test, three seeds.** S1 first (D-22): fixed100 (the table's protocol, one cached draw) and three
    independent `random600` draws (seeds 0, 1, 2), with R0, D29 and VIP-Seg's released S1 checkpoint
    scored on identical episodes in each draw; paired bootstrap over episodes per draw.
  * **Mechanism diagnostics**, labels used for diagnostics only: the logit-pair cosine of 02 §14 on the
    test episodes, the prototype-space cosines per step (descriptive only), the mIoU of both oracle rules
    per model (the headroom it leaves), and `L_distill` per epoch in training for both arms (R0 computes
    it without gradient).
* **Rules, fixed before the run.**
  * R2.0 reference: R0 − VIP-Seg released on S1 fixed100 is reported. Below −2, our loop trains the head
    worse than VIP-Seg's own code and a gain over R0 is not a gain over VIP-Seg.
  * R2.1 go: D29 − R0 ≥ +1.0 on fixed100 with a paired CI above 0, positive on all three random600
    draws, and the mean logit-pair cosine on the test episodes higher for D29 than for R0: the trained
    objective transfers to the novel classes. +1.0 is twice the
    estimated sd of a single-run difference. Then S0: both arms once, same test; the S0 claim needs the
    same rule and "beats VIP-Seg" needs D29 > 72.20 on S0 fixed100.
  * R2.2 stop: fixed100 gain < +0.5, or the mean of the three random600 gains < +0.5.
  * R2.3 in between: anything else. One run per arm cannot separate it from training noise; report and
    ask the maintainer for a second training seed per arm.
  * R2.4 mechanism: a gain without a higher logit-pair cosine is not a distillation result and is
    treated as R2.3.
  * R2.5 collapse watch: a gain with any test class below R0 by more than 3 IoU points is reported.
* **Reporting.** VIP-Seg's head trained with an oracle-direction loss, labelled that way; not
  CascadeProto. The text prior (the paper's modality branch) is the next addition on top of the
  winner of R2, not part of it.
* **Affects.** `models/oracle_distill.py` (new), `models/cascadeproto.py` (`distill_beta`, `cascade` and
  `effective_prototype` for the diagnostics), `pipeline/model_api.py` (`EpisodeOutput.loss_distill`, `distill_weight`), `train.py`
  (`--distill_beta`, run-dir suffix `_distill<β>`, `L_distill` in the log), `experiments/r2_distill_eval.py`
  and `experiments/run_r2.sh` (new), 02 §14, 05 §3.8j (DIS-1…).
* **Outcome: R2.2 stop (2026-09-24, `results/phase16_r2/SUMMARY.md`,
  `docs/research/2026-09-24_r2_distill_analysis.md`).** d29 − r0 on S1: fixed100 +0.06 [−0.26, +0.39],
  random600 −0.04 / −0.04 / −0.21. The objective transferred to the novel classes (logit-pair cosine
  0.771 → 0.843, above VIP-Seg's 0.807) and the mIoU did not move: the cosine weights points by their
  squared teacher margin, so it is dominated by easy points and barely sees sign errors at the boundary,
  and on training episodes the labels already give CE every decision the oracle rule could. The oracle
  headroom (+14.43) is a transductive information gap, not a missing training signal. No implementation
  bug (β 0 / 1 in the checkpoints, teacher detached, masks right, the loss optimised 0.067 → 0.023).
  R2.0: r0 is 5.16 [4.72, 5.64] below VIP-Seg's released checkpoint on fixed100 with equally informative
  features (oracle rules 84.6 / 85.0 against 83.8 / 86.3). VIP-Seg's released S1 log reads 70.07 valid at
  6,000 updates, where r0 ends (70.26), then 72.84 at its last update (24,000) and 75.63 at its selected
  best (22,000); its S0 log peaks at update 4,000 (72.94) and ends at 68.97. The gap is training length
  plus best-of-12 selection on test-class episodes [inferred from one run each]. Closed.

---

### D-30 — Route B's base on VIP-Seg's update count (E1) · `PROPOSED`, beyond the paper

* **Problem.** R2.0 measured our loop's route-B base r0 5.16 [4.72, 5.64] points below VIP-Seg's released
  S1 checkpoint (fixed100) with features as informative as VIP-Seg's (oracle rules 84.6 / 85.0 against
  83.8 / 86.3) (`results/phase16_r2/SUMMARY.md`). VIP-Seg's released S1 log reads 70.07 valid at 6,000
  updates, where r0 ends after its 6,000 (70.26), and 72.84 at its last update, 24,000
  [VIPSEG log_s3dis_VIPSeg/log_S1_N2_K1_0.760875/log_vipseg.txt]. D-12's null (CHANGELOG 15x) was
  measured on a headless baseline whose curve is flat from epoch 10; it does not cover the head.
* **What it does.** E1 trains r0 unchanged except for the schedule: `--batch_size 1 --lr_step_epochs 15`,
  i.e. 24,000 updates with the learning rate halved every 7,200 (VIP-Seg: 24,000 and 7,000
  [VIPSEG scripts/vipseg_s3dis.sh]), the same 24,000 training episodes as D-12, and `--valid_every 4`,
  13 validations (VIP-Seg: 12 every 2,000 updates) so that `best.pt` follows D-22's amended rule 1.
  No code change to the model; S1, seed 0, one run.
* **Test.** `last.pt` and `best.pt` of E1, `last.pt` and `best.pt` of r0, `best.pt` of d29, and VIP-Seg's
  released S1 checkpoint, on the identical episodes of fixed100 and random600 seeds 0, 1, 2
  (`experiments/r2_distill_eval.py`), paired bootstrap per draw.
* **Rules, fixed before the run** (`r2_distill_eval.py decide_e1`), on fixed100:
  * E1.1 adopt: E1 `last` ≥ 73.0, the level VIP-Seg's own run reaches at its last update (72.84):
    VIP-Seg's update count becomes route B's schedule for every later arm.
  * E1.2 not the schedule: E1 `last` ≤ 71.0, within the ≈ 1-point spread of our `last.pt` runs above
    r0 (70.20): the gap lies elsewhere; next candidates in the analysis §3 (gating bias, selection).
  * E1.3 partial: in between; adopt the schedule only if E1 − r0 has a paired CI above 0 on fixed100
    and is positive on all three random600 draws.
  * Reported, not ruled: E1 `best` against VIP-Seg released (both best-of-validation), and the
    selection gain `best − last` of E1 and r0.
* **Affects.** `experiments/run_r2.sh` (`train e1`, `eval_e1`), `experiments/r2_distill_eval.py`
  (named pairs, `decide_e1`), 05 §3.8j.
* **Outcome: E1.1 adopt (2026-09-24, `results/phase16_e1/SUMMARY.md`).** E1 `last` 73.20 on S1 fixed100,
  +2.99 [+2.62, +3.39] over r0 and +2.60 to +3.11 on every random600 draw. E1 `best` (best of 13
  validations, VIP-Seg's protocol) 75.05, against VIP-Seg's released 75.36: −0.31 [−0.63, +0.01], and
  −0.32 to +0.02 on the random600 draws. Selection gain `best − last` +1.85 [+1.43, +2.30]. Our loop
  reproduces VIP-Seg's head; R2.0's gap was training length (≈ 3.0) plus selection (≈ 1.9). VIP-Seg's
  update count is route B's schedule for every later arm.

---

### D-31 — P3: where the prototype error sits, and whether a text prior carries novel-class information · `PROPOSED`, beyond the paper

* **Problem.** On E1 (route B's base, D-30) replacing only the prototypes by the query's own class means
  lifts S1 fixed100 from 73.20 to 85.93–87.43 with the same features and the same dot-product decoding
  (`results/phase16_e1/SUMMARY.md`): the error is the prototype that the support and the head produce, not the
  features or the decoding (`docs/research/2026-09-24_text_integration_independent.md` §9.1). A prototype
  can be corrected only from the support, the query's unlabelled points, or prior knowledge (§9.2). Before
  designing a trained module (a point-level query–support neck, §9.4, or a text branch), two things must be
  measured on E1 with no training: which kind of error the oracle fixes, and whether class names carry
  usable information about novel classes.
* **Part A — gap decomposition** (descriptive; it chooses the mechanism of the next decision):
  * rules scored on the same episodes: the support-prototype rule without the head (`F^q n(P_point)ᵀ`), E1,
    and the oracle rule with one common norm (R2-pre); head recovery `r = (E1 − support) / (oracle − support)`;
  * **fixable points** = wrong under E1 and right under the oracle; their share among boundary points (a
    query point whose 16 nearest neighbours in xyz include another label) against interior points; their
    share by support-mask size tercile; Spearman between the per-episode oracle gain (foreground point
    accuracy) and the support foreground fraction.
  * Interpretation bands, stated before the run (not go/stop rules): fixable points enriched ≥ 1.5× at
    boundaries → a local refinement problem; < 1.2× → region-level confusions, i.e. a global prototype
    shift, the target of a point-level query–support neck; Spearman ≤ −0.3 with the support foreground
    fraction → support quality, the target of a prior (text, base classes) or of several prototypes per class.
* **Part B — training-free text prior after the head** (the research agent's T1 plus the entropy weight
  of the independent note §6, `docs/research/2026-09-24_text_integration_math.md` §5):
  * text direction per class `t̂_c` from the base-class bank of the checkpoint (P1's machinery, 1,000 training
    episodes of S1's base classes): **ridge** `t̂_c = n(Bᵀ(ẼẼᵀ + λI)⁻¹ ẽ_c)` with λ = 10⁻³, or **retrieval**
    `t̂_c = n(Σ_j softmax_j(τ cos(e_c, e_j)) B_j)` with τ ∈ {30, 100} (the best values of T0-C: 100 for the ensemble on all three banks, 30 for bare names
    on two);
    `ẽ = n(e − ē)`, ē the mean of the 12 class embeddings (names only);
  * prompts: the repo template, bare names, the 6-template ensemble of T0 and the 12 geometric descriptions
    frozen in the agent's probe (written before any result with them);
  * `T_ic = ⟨n(f_i − μ), t̂_c⟩`; for foreground columns `L′_ic = L_ic + κ γ_e w_i (T_ic − mean_{c′≥1} T_ic′)`;
    background unchanged; `γ_e = max(0, 2 acc_e − 1)` with acc_e the text-only foreground-vs-foreground
    accuracy on the support's foreground points; `w_i = 1` or the normalised entropy of E1's posterior at
    point i (text trusted where the head is unsure);
  * κ ∈ {0.25, 0.5, 1, 2, 4, 8, 16, 32}: the scale of E1's logits is not known in advance, so the grid is
    geometric over two decades;
  * selection on the S1 **valid** draw, test once on fixed100 (P0's protocol, D-22 rules 2 and 3); the
    frozen arm, the same arm with `w = 1`, and an oracle-γ arm (κ per episode chosen with the query labels,
    an upper bound, never a result).
* **Rules, fixed before the run (Part B).**
  * P3.0 mechanism: for the frozen (source, prompt), text-only foreground-vs-foreground accuracy on S1-valid
    query points ≥ 0.60 (the agent's bar; 0.5 is chance), and the alignment of the text direction with the
    oracle's correction (cosine over the points of the two foreground classes between `T_1 − T_2` and
    `(O_1 − O_2) − (L_1 − L_2)`) with an episode-bootstrap CI above 0. Otherwise stop text on this feature
    space.
  * P3.1 go: P3.0 holds, the frozen arm gains ≥ +0.5 on fixed100 with a paired CI above 0, and the oracle-γ
    arm gains ≥ +0.5 → a trained text prior (T2) gets its own decision.
  * P3.2 stop: the oracle-γ arm gains < +0.5 on fixed100 (even a per-episode weight chosen with the labels
    cannot reach the smallest effect worth a training run, P0–P2's +0.5).
  * P3.3 in between: otherwise; report.
  * P3.4 entropy weight: claimed only if the frozen arm with `w = H` beats the same arm with `w = 1` with a
    paired CI above 0 on fixed100.
  * P3.5 collapse watch: a gain with any class below E1 by more than 3 IoU points is reported.
* **Checkpoint.** E1 `last.pt` (`log_r2/s3dis_S1_N2_K1_point_T4_vip_b1/last.pt`); S1 only, S0 held out.
* **Affects.** `models/text_prior.py` (new), `experiments/p3_probe.py` and `experiments/run_p3.sh` (new),
  05 §3.8k (TXT-…).
* **Outcome (2026-09-24, `results/phase16_p3/SUMMARY.md`).** Part B: **P3.0 fails, text stops on this feature
  space**: 0 of 192 arms gain on the valid draw, the frozen arm is −0.00 on fixed100, and a per-episode weight
  chosen with the labels reaches only +0.70 [+0.52, +0.88]; the best text-only fg-vs-fg accuracy is 0.62. Part A:
  support rule 49.27 → E1 73.20 → oracle 85.93 (head recovery 0.65); fixable points are 1.54× enriched at
  boundaries (the pre-registered "boundary-enriched" band) but 83.5 % interior, flat across support sizes
  (Spearman +0.02), and mostly floor and wall predicted as background (recall 0.72 / 0.73 → 0.99 / 0.95 under
  the oracle; background precision 0.861 → 0.977). The head lowers floor below the plain support prototype
  (IoU 66.0 → 62.1).

---

### D-32 — P4: is the background prototype contaminated by the episode's own classes, causally? · `PROPOSED`, beyond the paper

* **Problem, from P3 and C0.** E1's dominant error is foreground → background on floor and wall (recall 0.72 /
  0.73, oracle 0.99 / 0.95; background precision 0.861, oracle 0.977) (`results/phase16_p3/SUMMARY.md`).
  VIP-Seg's background prototype pools the mask-0 points of every support block
  [VIPSEG models/vipseg.py:108-116], and in S3DIS the other way's support block often contains this way's
  class labelled background. Measured on the raw block labels (`experiments/c0_background_contamination.py`,
  3,000 S1 test episodes): floor is present in 95 % of the other way's support blocks (18 % of their
  background), wall in 64 % (26 %), the other four classes in 3–21 % (0.5–3 %); across the 15 class pairs of
  fixed100, the pooled contamination correlates with E1's oracle gap (Spearman +0.70, p = 0.004) and
  background precision falls from 0.95 (clean pairs) to 0.69 (floor + wall). That is a correlation; pairs
  with floor or wall may be hard for other reasons. P4 intervenes.
* **Episodes with the other way's labels.** The inherited `sample_pointcloud` with `support=False` labels a
  block's points by their index among the episode's classes, with the same point sampling as the support
  call [VIPSEG dataloaders/loader.py:31-87]; calling it on the support scans gives, for every support point,
  whether it is this way's class, the other way's class, or neither. The scans are chosen as in
  `generate_one_episode` (query then support per way, no scan reused) [VIPSEG dataloaders/loader.py:174-218].
  The inherited files are called, never edited (guardrail 2). Two seeded draws of the S1 test classes,
  100 episodes per class pair each: **A** (seed 1) selects, **B** (seed 2) tests.
* **Arms, E1 `last.pt`, no training.** The head is re-run from a modified `P^0` (the head's support slots take
  whole-block features, not masks, so `P^0`'s background row is where the masks enter
  [VIPSEG models/vipseg.py:244,347]); an identity check reproduces E1's logits from the unmodified `P^0`.
  * `clean_bg`: background row of `P^0` from the mask-0 support points minus the other way's points (labels):
    the causal intervention.
  * `purify_q`: without labels, drop from way j's mask-0 points the fraction q with the highest cosine to
    way k's (k ≠ j) foreground prototype, q ∈ {0.05, 0.1, 0.2, 0.3}; q frozen on draw A.
  * Diagnostics: the purifier's AUC for other-way points among mask-0 points; background-only and
    foreground-only oracle replacements (the query's own background or foreground direction, one common
    norm), which split the oracle gap between the two kinds of row.
* **Rules, fixed before the run (draw B, paired bootstrap).**
  * P4.1 causal: `clean_bg` − E1 ≥ +1.0 with a CI above 0 and floor and wall recall both higher → the
    contamination causes part of the gap. `clean_bg` − E1 < +0.5 → it does not; stop this line.
  * P4.2 label-free fix: the frozen `purify_q` ≥ +0.5 with a CI above 0 → a training-free candidate.
  * P4.3 neck: P4.1 holds and `clean_bg` − `purify_q` ≥ +1.0 → a label-free rule leaves at least a point that
    a learned purification (trainable, supervised by the base classes' raw labels in training) could take;
    if `purify_q` is within 1.0 of `clean_bg`, the rule suffices and no neck is warranted by this evidence.
  * +1.0 and +0.5 are the thresholds of R2/E1 and P0–P3 (twice the single-run sd; the smallest effect worth a
    training run).
* **Affects.** `experiments/p4_background_probe.py`, `experiments/run_p4.sh`, 05 §3.8l (BG-…).
* **Outcome: P4.1 not causal, the line stops (2026-09-24, `results/phase16_p4/SUMMARY.md`).** The background row
  rebuilt without the other way's points (labels) gains +0.09 [+0.04, +0.14]; floor recall 0.724 → 0.733,
  wall 0.737 → 0.738. The label-free purification gains +0.02. Row-wise oracles: background only −26.07,
  foreground only −9.05, all rows +13.71: the oracle's gain is a joint, query-conditioned shift of every
  prototype, not a contaminated row. C0's correlation was confounded; no purification neck is warranted by
  this evidence (P4.3 not reached).

---

### D-33 — N1: a point-level support → query attention neck before the prototypes · `PROPOSED`, beyond the paper

* **Problem, from P3 and P4.** On E1 the oracle rule (the query's own class directions) gains +13.71 when every
  prototype row is replaced, while replacing only the background row costs 26.07 and only the foreground rows
  9.05 (`results/phase16_p4/SUMMARY.md`): the error is a joint, query-conditioned shift of all prototypes, not a
  contaminated row. Training-free fixes (D-26…D-28, D-31, D-32) and a label-derived training target (D-29)
  leave it unchanged. VIP-Seg's head adapts the prototypes to the query only through max-pooled 64-token
  channel statistics [VIPSEG models/vipseg.py:235-311] and recovers 65 % of the support → oracle gap (P3);
  no point-level correspondence between query and support reaches the prototypes
  (`docs/research/2026-09-24_text_integration_independent.md` §9.3).
* **What it does.** Before the prototypes and the head, every support point attends to the episode's query
  points and moves its feature toward what it matches:
  `F_s' = F_s + α · softmax((LN(F_s) W_Q)(LN(F_q) W_K)ᵀ / √d) (LN(F_q) W_V) W_O`, d = 64, one head, all
  support blocks against all query points of the episode; α a learned scalar initialised to 0. The masked
  means of `F_s'` are the new `P^0` and the head receives `F_s'` for its support slots; `F_q` is unchanged.
  At α = 0 the model is E1 exactly. Spec 02 §15.
  * Why the support side: the oracle is the query's class mean; a support point pulled toward the query
    points it resembles moves its class's masked mean toward that class's query mean [inferred].
  * The prediction for one query depends on the other queries of the episode, as it already does through
    VIP-Seg's head (research note §4.4).
* **N1, the measurement.** Two arms warm-started from E1 `last.pt` (our own weights, trained from scratch on
  S1's base classes; guardrail 1 is not involved), identical seed and episodes, 7,200 updates at batch 1 and
  a constant learning rate 1.25e-4 (E1's last stage), validation every 5 epochs (3 validations):
  **ctl** without the neck, **neck** with it. The shared start removes most of the seed difference between
  the arms. One run per arm (maintainer). Test as R2/E1: fixed100 and random600 seeds 0, 1, 2, both arms and
  E1 on identical episodes, `last.pt` and `best.pt` (D-22 amended).
* **Rules, fixed before the run.**
  * N1.1 go: neck − ctl ≥ +1.0 on fixed100 `last` with a paired CI above 0 and positive on all three random600
    draws. Then S0. (At α = 0 the neck arm computes exactly ctl's function, so a gain implies a used neck;
    the final α is reported.)
  * N1.2 stop: neck − ctl < +0.5 on fixed100 `last`, or the mean over the random600 draws < +0.5.
  * N1.3 in between: otherwise; report.
  * N1.4 mechanism: the oracle rule's gain over the model on fixed100, a gap the neck exists to close, is
    reported for both arms; a go with an unchanged gap is flagged.
  * +1.0 and +0.5 are R2's thresholds (twice the single-run sd of 0.5).
* **Affects.** `models/neck.py` (new), `models/cascadeproto.py` (`neck`), `train.py` (`--neck`,
  `--init_checkpoint`), `experiments/r2_distill_eval.py` (`decide_n1`), `experiments/run_n1.sh` (new), 02 §15,
  05 §3.8m (NECK-…).
* **Outcome: N1.2 stop, the neck never opened (2026-09-24, `results/phase16_n1/SUMMARY.md`).** neck − ctl on fixed100
  −0.17 [−0.22, −0.13], −0.07 / −0.20 / −0.16 on random600; the gate ended at α = 0.0023. Both arms lost about
  2 points against E1 (restarting AdamW's moments: validation 73.04 → ~59 at epoch 5 → ~71). The run measured
  that a zero-initialised neck grafted onto a converged model at 1.25e-4 does not open; D-34 tests the neck.

---

### D-34 — N2: the neck of D-33 trained from scratch on E1's schedule · `PROPOSED`, beyond the paper

* **Problem, from N1.** Warm-started from E1 at a constant 1.25e-4 with α = 0, the neck ended at α = 0.0023
  after 7,200 updates and both arms' validations moved together (ctl 58.91 / 70.17 / 70.95, neck 59.23 /
  70.45 / 70.80); restarting AdamW's moments also dropped both arms from E1's 73.04 to 59 at epoch 5
  (`results/phase16_n1/`). N1 measured that a zero-initialised neck grafted onto a converged model does not
  open; it did not test point-level support → query attention.
* **What it does.** The model of D-33 (`--neck sq_attn`) trained from scratch on E1's schedule (D-30: batch 1,
  24,000 updates, LR 1e-3 halved every 7,200, 13 validations), seed 0, with **α initialised to 0.1**
  (`--neck_alpha_init 0.1`) so that the neck's projections receive gradient from the first update. The
  "identity at initialisation" property of D-33 is dropped: a model trained from scratch has no base to
  preserve. 0.1 is not tuned (not measured); it is the smallest round value that makes the neck's term
  non-negligible while leaving F_s dominant at initialisation.
* **Reference.** E1's existing `last.pt` and `best.pt` (maintainer, 2026-09-24: E1 is not retrained), scored
  with N2 on identical episodes. The comparison is therefore between two independent training runs, whose
  `last.pt` differ by about 1 point from training noise alone (E1's validations swing 71–75, R2's r0/d29
  `last` equal within 0.1 while their `best` differed by 1.5); the paired CI covers only the episode draw.
* **Rules, fixed before the run** (`r2_distill_eval.py decide_n1` with E1 as `ctl` and N2 as `neck`):
  N1.1 go (N2 − E1 ≥ +1.0 on fixed100 `last`, CI above 0, positive on all three random600 draws), N1.2
  stop (< +0.5, or mean random600 < +0.5), N1.3 otherwise; N1.4 oracle gaps reported. A go is confirmed by
  a second seed before S0, because one run cannot separate +1.0 from training noise.
* **Affects.** `models/neck.py` and `models/cascadeproto.py` (`neck_alpha_init`), `train.py`
  (`--neck_alpha_init`), `experiments/run_n2.sh` (new), 05 §3.8m (NECK-10).
* **Outcome: N1.2 stop (2026-09-24, `results/phase16_n2/SUMMARY.md`).** N2 − E1 on fixed100 `last` −0.66 [−1.03, −0.29],
  random600 −0.52 / −1.08 / −0.94, `best` −0.81 [−1.21, −0.44]; the neck was used (α 0.1 → 0.060) and the oracle
  gap did not shrink (12.73 → 13.48). The same sign on every draw and checkpoint makes a hidden +1.0 implausible.

---

### D-35 — P5: split E1's oracle gap by sampling condition and class presence, before any new module · `PROPOSED`, beyond the paper

* **Problem.** D-26…D-34 all acted on the *semantics* of the prototypes and all left the oracle gap where it was
  (12.7–14.4 points). Four facts measured or read after N2 show that the gap was never decomposed correctly, so no
  mechanism could be aimed at it (research note `docs/research/2026-09-24_condition_presence_gauge.md`):
  1. **The benchmark puts every class in two sampling conditions, and the support shows only one.** The inherited
     sampler draws `int(ratio · 2048)` points of the class a block is sampled for, then `2048 − that` points from the
     whole block [VIPSEG dataloaders/loader.py:37-58] (COSeg's "Algorithm 1"). For raw class shares π the expected
     counts are `2048 · π_s(2 − π_s)` for the sampled-for class and `2048 · (1 − π_s)π_k` for every other class, so
     the sampled-for class is at least twice as dense as anything else in its block. In an N-way episode the query
     block of way k is sampled for class k: the other way's class, when present, sits at background density
     ("other" condition). Supports are always "own". Measured on 1,500 seeded test episodes per fold
     (`experiments/c1_sampling_condition.py`, `results/c1/c1_S{0,1}.json`):

     | class (fold) | share of its query points in the other condition | kNN-16 radius own / other [m] | duplicated points own / other |
     | :--- | ---: | :--- | :--- |
     | wall (S1) | 0.235 | 0.084 / 0.121 | 0.044 / 0.001 |
     | ceiling (S0) | 0.235 | 0.072 / 0.123 | 0.064 / 0.001 |
     | floor (S1) | 0.192 | 0.072 / 0.130 | 0.060 / 0.001 |
     | the nine other test classes | 0.007–0.068 | 0.068–0.084 / 0.10–0.13 | 0.04–0.09 / ≤ 0.005 |

     Background radius 0.116–0.119 m (support and query), support foreground 0.077–0.079 m: other-condition points
     look like background by density. All foreground query points: 11.8 % other (S1), 7.3 % (S0); the second number
     is also S1's *training* classes, so 92.7 % of S1's foreground supervision is own-condition. The encoder keeps
     density: every DyPowerConv divides its kNN offsets by one scalar `torch.std` of the batch
     [VIPSEG models/encoder.py:182-189]. COSeg measures what the cue is worth at the same 2,048 points: AttMPTI
     65.52 → 41.41, QGE 73.83 → 47.02, QGPA 61.95 → 38.34 without the over-sampling (S3DIS 1-way 1-shot)
     [COSeg Tab.1]; it does not examine the N-way asymmetry, and no later paper was found that does.
  2. **The three classes with the largest other share are the three worst.** VIP-Seg's released checkpoints score
     ceiling 64.4, floor 65.3, wall 69.7, the lowest IoUs of both folds bar beam (67.2); over the 12 test classes,
     Spearman(other share, IoU) = −0.57, p = 0.051 (`results/c1/`, `results/phase16_p0/test_vipseg_S{0,1}.json`).
     P4's follow-up read the same pattern as "always background in training" (ρ = −0.55), which the protocol can never
     test; the other-condition reading can be tested at test time. Neither explains S0's beam, board, bookcase (other
     shares 0.009–0.066, oracle gaps 15–19).
  3. **P4's row-wise oracle split measured gauge, not error location.** `row_oracle_logits` writes `‖m_c‖ · O_c` into
     the chosen rows [`experiments/p4_background_probe.py:97-104`]; `O_c` is a mean of unit non-negative features, so the
     replaced rows gain a large common positive component that the other (LayerNorm) rows lack. The saved counts show
     exactly that bias (`results/phase16_p4/test_e1_counts.npz`, draw B): background row only → background predictions
     4.59 M against 3.10 M true (+48 %), foreground predicted at 0.51× its true count, every foreground recall down
     (floor 0.72 → 0.38); foreground rows only → background predictions 1.56 M (−50 %), foreground at 1.51×, table FP
     1.26× its true count. "−26 / −9 / +13.7" therefore says nothing about *where* the error is, and D-33's premise
     ("a joint shift of every prototype") is not established. `clean_bg` (+0.09) stands: it tests contamination only.
  4. **The oracle rules also carry block-level presence.** `oracle_replaced` replaces only the rows of classes present
     in a query block and keeps `M_eff` for absent ones [`experiments/r2_distill_eval.py:112-125`]; the D-29 teacher
     leaves absent rows at zero [`models/oracle_distill.py:34-43`]. Absent rows compete from another gauge, so the
     oracle can hardly predict a class the block lacks. The other way's class is absent from the other query block in
     6 % (floor), 27 % (wall), 74 % (table), 88 % (door), 93 % (window), 98 % (sofa) of episodes (`results/c1/c1_S1.json`).
     E1's precision errors sit on exactly the rarely co-occurring classes (FP/GT door 0.215, sofa 0.255, table 0.208
     against wall 0.042), and the oracle roughly halves door and sofa FP (0.104, 0.098) but *raises* table's (0.355)
     (`results/phase16_e1/test_S1_fixed100_counts.npz`): presence is part of the oracle's gain, not all of it. Its size
     is unknown.
  5. **VIP-Seg's cross-term is not class-specific.** After `reshape(72, −1)` the attention of "query b′, slot w′" is
     `softmax(Σ_{r<72} Q′_{⌊r/36⌋, 2(r mod 36)+b′}ᵀ S′_{⌊r/24⌋, 3(r mod 24)+w′} / √128)`: it sums over both queries and
     all support slots; the slot only selects projection filters. `experiments/c2_vipseg_crosscorr_check.py` rebuilds the
     inherited module (parsed, not edited): the formula matches it to 4.7e-14; changing only way 2's support moves slot
     1's attention by up to 0.45 and changing only query 2 moves query 1's by 0.075, against a typical entry of 0.0078
     (`results/c1/c2_crosscorr.txt`). The only class-specific query-conditioned path is the support self-gate; the query
     gate is shared by all rows and computed from max-pooled tokens. P3 measured the head *lowering* floor recall from
     0.874 (support prototype, no head) to 0.719 [`results/phase16_p3/SUMMARY.md`].
  6. **An untested source of support/query shift.** Training runs every BatchNorm of the encoder
     [VIPSEG models/encoder.py:230,317,320,483] and feature head on the support batch and the query batch separately,
     each with its own statistics [`models/vipseg_backbone.py:86-98`]; evaluation uses running statistics. The oracle,
     computed inside each query block, is immune to such a shift; E1 is not.
* **What it does.** P5 (`experiments/p5_condition_probe.py`, `experiments/run_p5.sh`): no training, no change to any
  model; E1 `last.pt` (`log_r2/s3dis_S1_N2_K1_point_T4_vip_b1/last.pt`) and, for arms A and C, VIP-Seg's released S1
  checkpoint. S1 only (D-22). Every arm passes `eval.py`'s protocol guard.
  * **Tags.** Query block b was sampled for local class b + 1 [VIPSEG dataloaders/loader.py:174-225]. In block b, local
    class k is *own* if k = b + 1, *other* if k ≠ b + 1 and some point carries k, *absent* otherwise.
  * **A. Condition and presence split** (fixed100). Per class, accumulated over episodes as VIP-Seg's metric: TP and GT
    for own and other points; FP in own, other and absent blocks. For E1, VIP-Seg, the current oracle (common norm, R2's
    `oracle_unit`) and a **presence-fair oracle**: absent rows replaced by the support prototype's unit-feature direction
    `n(Σ_{i∈M_c} f_i/‖f_i‖)` in the same common norm, so that absent classes compete in the oracle's gauge.
    Counterfactuals (bounds, never results): **cf-a**, each class's other-condition TP raised to
    `recall_own · GT_other`, predicted count raised by the same amount, all FP unchanged; **cf-b**, each class's FP in
    absent blocks removed.
  * **B. Condition intervention** (a fresh draw, seed 3, P4's episode machinery with scan names, 100 episodes per
    class pair). For every query block sampled for a whose scan holds ≥ 100 raw points of the other episode class c:
    V0 the protocol block; V1 the same scan sampled for c (inherited `sample_pointcloud`, `sampled_class = c`); V2 the
    same scan sampled uniformly (`random_sample = True`); the rest of the episode unchanged, each version drawn with its
    own seed. Measured on block b only: recall of c under V0 / V1 / V2, recall of a under V0 / V2, and R_own(c), c's
    recall in the blocks of the same draw sampled for c. Causal fraction `φ = (R_V1 − R_V0) / (R_own − R_V0)`. Also
    the raw class of every false positive (base / clutter / other novel), which sizes base-class confusion.
  * **C. Transductive batch statistics.** Every BatchNorm in batch-statistics mode with momentum 0 (running statistics
    unchanged), support and query forwards separate as in training; fixed100 and random600 seed 0.
  * **D. Leak-free draw.** Support and query both sampled uniformly (the inherited `random_sample = True` path, 2,048
    points: COSeg Tab.1's "w/o FG" form), seed 4, 100 episodes per pair: E1, the support rule without head
    (`F^q n(P_point)ᵀ`, P3's rule) and the oracle.
  * **E. Dual-condition prototypes, training-free.** For each support block with foreground share f, r̂ = 1 − √(1 − f)
    (the inverse of f = r(2 − r)); keep each foreground point with probability (1 − r̂)/(2 − r̂), which puts the
    foreground at background density; refill to 2,048 points with copies of uniformly drawn kept points plus the
    training jitter (σ = 0.01 m, clip 0.05 [VIPSEG dataloaders/loader.py:110-112]). The head is re-run on the sparse
    views (their `P^0` and their support slots); foreground logit of class k = max(dense, sparse). **Control:** the
    dense logits plus one constant on the foreground columns, set per episode by bisection so that the number of
    foreground predictions equals the dual arm's.
  * **Checks, on every episode of the smoke run (5 episodes per arm) and of the full run.** Every scoring rule
    reproduces its model's logits (P0's identity check), arm E's re-run head included; the index copy of the sampler
    that gives B its raw labels equals the inherited `sample_pointcloud` on every block it draws; A's own + other +
    absent counts add up to the pooled counts and the pooled mIoU equals VIP-Seg's `evaluate_metric`; batch
    statistics change the logits of every episode and leave every running statistic unchanged.
* **Rules, fixed before the run** (+1.0 = twice the single-run sd, the smallest gain worth a training run in R2/E1/N1;
  +0.5 = the smallest training-free effect worth a follow-up in P0–P4):
  * **P5.1 E-a (other-condition misses) matters:** cf-a − E1 ≥ +1.0 on fixed100. Then **P5.1a sampling is causal:**
    φ ≥ 0.5 with the episode-bootstrap CI of R_V1 − R_V0 above 0 → a decision for condition-balanced training (M1)
    and dual-condition prototypes (M2). **P5.1b not density:** φ < 0.2 → M1 only (uniform queries also show small
    fragments at background density), M2 dropped. Otherwise report.
  * **P5.2 E-b (presence) matters:** cf-b − E1 ≥ +1.0 → a decision for presence estimation (M3).
  * **P5.3 batch statistics:** C − E1 ≥ +1.0 with a paired CI above 0 on fixed100 and > 0 on random600 seed 0 → the
    test-time rule of every later arm, reported as transductive, always next to the `eval()` number.
  * **P5.4 dual prototypes:** E − control ≥ +0.5 with a paired CI above 0 → M2 is trained.
  * **P5.5 reported, no rule:** the presence-fair oracle's gain over E1, the head's gain over the support rule with and
    without the leak (D), the FP composition (B), the same split for VIP-Seg's checkpoint.
  * Neither P5.1 nor P5.2 → the condition and presence readings are dropped; the gap is own-condition (instance shift),
    the target of a trained self-support cascade (research note §5, M4).
* **Why no training first.** Each of M1–M4 costs one 2.3-GPU-h run and targets a different error; P5 costs about
  1–1.5 GPU-h (not measured) and says which error is large enough to be worth a run. It also gives the phase-16 claims
  in points 3–4 their measured size.
* **Reporting.** P5 is a diagnostic of E1; nothing in it is a result against VIP-Seg. The counterfactuals are bounds.
* **Affects.** `experiments/c1_sampling_condition.py`, `experiments/c2_vipseg_crosscorr_check.py` (new, CPU, done),
  `experiments/p5_condition_probe.py`, `experiments/run_p5.sh`, `tests/test_condition_probe.py` (new), 05 §3.8n
  (P5-1…15).

* **Outcome of the smoke run and C3 (2026-09-24): the full P5 is not run as registered.** The smoke run (5 episodes
  per arm, `results/phase16_p5/*_smoke.json`) passed every check but showed E1 at other-condition recall 0 / 1,153
  points, VIP-Seg at 1 / 1,153, and arm B's V1 (the other class made dense in its own scan) still at recall 0. A
  diagnostic found why: E1 labels the dense region of query block b with class b + 1 *whatever it is* (floor made
  dense in the "wall" block: 421 / 671 points predicted wall, 0 floor; the support rule without head: 668 / 671
  floor). C3 (`experiments/c3_query_order.py`, `results/phase16_p5/c3_query_order.json`, all 1,500 fixed100
  episodes, the two query blocks swapped, labels moving with their blocks) measured it:

  | model | mIoU, stored order | mIoU, swapped | own-class points kept → after swap | relabelled by position | support rule (both orders) |
  | :--- | ---: | ---: | :--- | ---: | ---: |
  | E1 `last` | 73.20 | **0.92** | 0.935 → 0.001 | 0.913 | 49.27 |
  | VIP-Seg released | 75.36 | **0.87** | 0.937 → 0.000 | 0.898 | 51.57 |

  VIP-Seg's head names the foreground by the **position** of the query block, which the loader fixes (the block
  sampled for class k is appended k-th [VIPSEG dataloaders/loader.py:181-222], in training and in the cached test
  episodes). Arms B and E presuppose a head that names classes by their prototypes, so their rules cannot be read on
  these checkpoints; arms A, C, D measure what they were built for but on a model whose errors are those of the
  shortcut. Every phase-16 decision measured on route B (D-25…D-34), and the oracle gap itself, was measured on
  heads with this shortcut. D-36 and D-37 trace it before any new module.

---

### D-36 — C4: is the cross-term reshape the only carrier of the query position? · `PROPOSED`, diagnostic

* **Problem.** C3 shows that E1 and VIP-Seg depend on the query position (D-35 outcome). By the index algebra of
  C2 (`results/c1/c2_crosscorr.txt`), `reshape(72, −1)` gives query b′ the projected rows of parity b′ and slot w′ the
  rows ≡ w′ (mod 3), so the cross-term can learn a different map for every (position, slot) pair: a *necessary*
  condition for the shortcut. Every other operation of the model is symmetric in the query index [verified: the
  encoder and feature head treat each block alike and couple them only through the batch-wide `torch.std` and, in
  training, BatchNorm, both invariant to the order of the batch (`models/encoder.py:182-189`); PEM/PDM's self-gates
  are per query and per slot (`models/vipseg.py:266-277,370-380`); the gating network and the logits are per query
  (`models/vipseg.py:158-176`)]. Whether the reshape is also *sufficient*, i.e. the only carrier, is not measured.
* **What it does** (`experiments/c4_crossterm_ablation.py`, inference only). A re-implementation of VIP-Seg's
  PEM/PDM forward that reads the inherited modules' weights (never edits them) and computes the cross-term in one of
  two forms: `scrambled` (VIP-Seg's, checked to reproduce the inherited module's output on every episode) and
  `clean`, `A[b, w] = softmax(Q′_bᵀ S′_w / √128)` per query b and slot w, D-01's form with VIP-Seg's scale. E1 and
  VIP-Seg's released head re-run with each form on fixed100, stored and swapped order (C3's scoring).
  * **Implementation** (amended 2026-09-24, before any code). `models/vip_stage.py` gains
    `vip_module_forward(module, query, supports, prototype, form)`: PEM (Eq.9–14) and PDM (Eq.15–18) written out
    from the inherited source, reading the module's own submodules (`maxpool`, `map`, `proto_map`, `reweight`,
    `reweight_s`, `fc`, `fc_qs`, `layer_norm_qs`, `layer_norm`); only the cross-term differs between the forms
    [VIPSEG models/vipseg.py:285-296,387-394]. `VIPStage` gains `cross_form ∈ {native, scrambled, clean}`; `native`
    calls the inherited module exactly as before, so E1's configuration and every state-dict key are unchanged.
    On VIP-Seg's released model the entries of `vip_module` are wrapped in the loaded instance only; no class is
    edited (AGENTS guardrail 2).
  * **Passes per checkpoint** (fixed100): native stored and swapped (a repeat of C3), `scrambled` stored,
    `clean` stored and swapped.
  * **Checks, on every episode; a failure stops the run.** C4.0a: `scrambled` reproduces the native logits,
    max |diff| ≤ 1e-4 · max(1, max |native logit|). C4.0b: the native passes reproduce C3's mIoU (73.20 / 0.92,
    75.36 / 0.87) within 0.01.
* **Rules, fixed before the run.**
  * **Relabel shift** (defined 2026-09-24, before any run). After the swap, the label of the position a block moved
    to is the other episode class's label, which a model can also predict by honest confusion. The raw relabel
    share is therefore not zero for an order-free model; the quantity is its shift, the share of own-class points
    predicted as the other episode class in the swapped order minus the same share in the stored order (0 for an
    order-free model; for E1, 0.913 minus a stored-order share that C3 did not record and C4 does).
  * C4.1 sole carrier: with `clean`, |stored − swapped| ≤ 0.1 mIoU and |relabel shift| ≤ 0.01 on both
    checkpoints. The equivariance argument above predicts exact invariance up to floating point, so a larger
    difference means a second carrier exists and must be found before D-37 is read.
  * C4.2 reported: `clean` stored-order mIoU against `scrambled` (73.20 / 75.36) and the support rule (49.27 /
    51.57). The weights were trained with the scrambled form, so the level is not a model's quality; it measures how
    much of the trained head's prediction runs through the positional path.
* **Cost.** Five passes over fixed100 per checkpoint, about 30 minutes on an L4 (not measured).
* **Order.** C4 runs before D-37's training; a C4.1 failure stops D-37's queue, because a second carrier would
  also be present in the clean arm.
* **Affects.** `experiments/c4_crossterm_ablation.py`, `models/vip_stage.py` (the `clean` form is shared with
  D-37), tests (05 §3.8o).

---

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
| 2026-09-19 | D-17: no `W_g` for T = 1 (identical prediction, no dead parameter). D-16 biases of `W_1`, `W_2`, `W_out` kept although Eq.20–21 print none (maintainer decision). |
