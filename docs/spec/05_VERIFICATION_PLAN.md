# 05_VERIFICATION_PLAN: Tests, Gates & Acceptance Criteria

* **Ground truth:** [00_SOURCES_AND_DECISIONS.md](00_SOURCES_AND_DECISIONS.md). Every check traces to a spec section and, through it, to the paper, VIP-Seg or a decision.
* **Scope:** what must be tested, on which inputs, and what counts as passing. Tests verify the specs [01](01_ARCHITECTURE_SPEC.md)–[04](04_DATA_AND_EPISODES.md); they are not a second source of math.
* **Rewritten:** 2026-09-17 (Phase 5).

---

## 1. Principles

1. **Test the contract the real pipeline uses.** Synthetic tensors must have the shapes and dtypes of the inherited loader (04 §4.3): support `[N, K, 9, 2048]`, **binary** int32 masks, query `[N, 9, 2048]`. Masks with values 2, 3, … are forbidden because they hid a prototype bug before [VIPSEG dataloaders/loader.py:82-83,220-223].
2. **One test per claim.** Each formula or decision in 02–04 that can be checked numerically has a test that would fail if the claim were violated.
3. **No network or dataset in unit tests.** CLIP embeddings are replaced by fixed random L2-normalised tensors; tests that need CLIP, CUDA, `mamba_ssm` or data are marked and run in later gates.
4. **Synthetic data only under `tests/`.** `train.py` and `eval.py` never generate random episodes; a "dry run" uses the real loader on real blocks (04 §4).
5. **Fail loudly.** Missing `mamba_ssm`, `pointnet2_ops`, CLIP or an unimplemented modality is a test failure, not a skip, in the gates that require them (01 §2.1).

---

## 2. Gates

Run in order; a gate starts only when the previous one passes.

| Gate | Needs | Command | Purpose |
| :--- | :--- | :--- | :--- |
| G0 Environment | GPU runtime | `pytest tests/test_environment.py -v` | Required packages import; extensions run on the GPU; inherited files unchanged |
| G1 Unit (CPU) | nothing external | `pytest -m "not cuda and not clip and not data" -v` | Math and wiring of 02–03 |
| G2 Unit (GPU) | CUDA, `mamba_ssm`, `pointnet2_ops` | `pytest -m cuda -v` | Encoder batching and parameter counts |
| G3 Integration | G2 + CLIP weights | `pytest -m clip -v` | End-to-end episode with real CLIP embeddings |
| G4 Data | preprocessed S3DIS | `pytest -m data -v`, then `python train.py --dry_run ...` | Real loader, real episodes, one optimiser step |
| G5 Reproduction | full data, compute | training + evaluation runs | Tables 2–5 comparison (§5) |

Markers `cuda`, `clip`, `data` are registered in `pytest.ini`. ENV-3 and ENV-4 carry no marker, so they also run in G1.

---

## 3. Test specifications

### 3.1 `tests/test_environment.py` (G0)

| ID | Check | Source |
| :--- | :--- | :--- |
| ENV-1 | `import mamba_ssm`, `import pointnet2_ops`, `import clip`, `import h5py`, `import transforms3d`, `import timm` succeed (marker `cuda`) | 01 §2.1, 04 §4 |
| ENV-2 | `torch.cuda.is_available()`; `torch.cuda.get_arch_list()` holds a kernel that runs on the GPU (same-major `sm_` with lower or equal minor, or older `compute_` PTX); `furthest_point_sample` and a `Mamba` forward run on it (marker `cuda`) | [PAPER §4.1 "single NVIDIA RTX 5090"] |
| ENV-3 | Inherited files are byte-identical to the pinned VIP-Seg commit: `dataloaders/{loader,s3dis,scannet}.py`, `preprocess/{collect_s3dis_data,collect_scannet_data,room2blocks}.py`, `utils/{checkpoint_util,cuda_util,logger}.py`, `models/{encoder,mamba_block,model_utils,vipseg,vipseg_learner}.py`, `runs/{training_free,training,evaluate}.py`, `main.py` (git blob SHA-1 of the CRLF-normalised file, compared with the pinned tree) | 04 header, [00 §5.2](00_SOURCES_AND_DECISIONS.md) |
| ENV-4 | `models/encoder.py` contains no fallback block and imports `mamba_ssm` unconditionally | 01 §2.1 |

### 3.2 `tests/test_prototypes.py` (G1)

Fixture: `F^s ∈ [N=3, K=2, 2048, 128]` random, binary masks with different fore-/background points per way.

| ID | Check | Source |
| :--- | :--- | :--- |
| PROTO-1 | Row k of `P_point` equals the manual mean of `F^s[k−1]` over its mask-1 points, for every k = 1..N | 02 §3 |
| PROTO-2 | Row 0 equals the mean over all mask-0 points of all ways and shots | 02 §3 |
| PROTO-3 | No row is all-zero; rows 1..N differ pairwise when the masks differ | 02 §3 |
| PROTO-4 | Changing way 2's mask leaves rows 1 and 3 unchanged | 02 §3 |
| PROTO-5 | Equals a line-by-line port of VIP-Seg's loop (without its L2 normalisation) on 5 random episodes | [VIPSEG models/vipseg.py:108-130], D-10 |
| PROTO-6 | Invariant to point order and shot order; permuting ways permutes rows 1..N and keeps row 0 | 02 §3 |
| PROTO-7 | Affine equivariance `P(aF + b) = aP(F) + b` | 02 §3 (means) |
| PROTO-8 | Autograd gradient of row k equals `mask / count` | 02 §3 |
| PROTO-9 | Empty background gives `0.1·1`; an empty foreground raises and names the way | 02 §3 |
| PROTO-10 | Non-binary masks, wrong point count, missing way axis and flattened ways raise; float32 matches float64 to 1e-5 | 02 §1 |

All PROTO checks run in float64 with tolerance 1e-12. Mutation check (2026-09-18): six wrong variants (per-shot means, per-way background means, L2 normalisation, background from one way, audit C2's flattened ways, zero empty background) each fail at least one test.

### 3.2b `tests/test_feature_extractor.py` (G1)

The VIP-Seg encoder is replaced by per-point stand-ins with its contract (`[B, 2048, 9] → [B, 900, 2048]`); float64, tolerance 1e-12.

| ID | Check | Source |
| :--- | :--- | :--- |
| FEAT-1 | Head = `BN(900), ReLU, Conv(900→196), BN, ReLU, Conv(196→128), BN, ReLU`, no in-place ReLU, 204,260 parameters | [VIPSEG models/vipseg.py:45-53] |
| FEAT-2 | Output equals the formula written out with `F.batch_norm`/`F.conv1d` in eval mode; all entries ≥ 0 | 02 §2 |
| FEAT-3 | Invariant to any positive per-point scale of the encoder output (channel L2 norm per point) | [VIPSEG models/vipseg.py:86] |
| FEAT-4 | `encode_episode` keeps way and shot order: `F^s[n,k]` equals encoding block `(n,k)` alone | 02 §2 |
| FEAT-5 | In training mode the support blocks and the queries are separate BatchNorm batches | [VIPSEG models/vipseg.py:79-89] |
| FEAT-6 | Wrong layouts (3 channels, 4096 points, channels-first, 2-D, 4-D) and a wrong encoder output raise | 01 §2.1 |
| FEAT-7 | Gradients reach every encoder and head parameter | 02 §2 |
| FEAT-8 | Fixed projections are saved and restored by `state_dict`; `load_vipseg_weights` copies them from plain attributes and refuses a different encoder | [VIPSEG models/encoder.py:619-620] |

Mutation check (2026-09-18): twelve wrong variants (no or wrong-axis L2 norm, in-place ReLU, missing final ReLU, head order, hidden width, swapped way/shot order, joint support–query batch, no input check, no projection buffers, projections not copied, non-strict loading) each fail at least one test.

### 3.3 `tests/test_lma_gmmn.py` (G1)

| ID | Check | Source |
| :--- | :--- | :--- |
| LMA-1 | Adapter `[N+1, 512] → [N+1, 128]`; generator input width 256; parameter counts 82,432 and 65,920; output equals Eq.5–6 written out with explicit tensor operations (LayerNorm with biased variance, eps 1e-5) | 02 §4.1–4.2, 01 §2.3 |
| LMA-2 | `model.eval()`: two forwards give identical `P_modal` (z = 0) | 02 §4.2, [DECISION D-06] |
| LMA-3 | `model.train()`: two forwards differ (fresh z); an exact replay of the random stream (dropout mask p = 0.1 after the ReLU, then `z ~ N(0, I)` `[N+1, 128]`) reproduces the output; z has mean 0 and standard deviation 1 | 02 §4.2, [DECISION D-16] |
| MMD-1 | `k(x, x) = 6` | 03 §4.1 |
| MMD-2 | `MMD({x}, {y}) = 2·(6 − k(x, y))` to 1e-5 | 03 §4.1 |
| MMD-3 | `MMD(X, X) = 0`; `MMD ≥ −1e-6` on random sets | 03 §4.1 |
| MMD-4 | Result equals the explicit triple-mean formula of 02 §4.3 (no square root) | 02 §4.3 |
| MMD-5 | Foreground term uses rows 1..N as one set: equals `MMD(P_modal[1:], P_point[1:])`, not the mean of per-row MMDs (for N ≥ 2 the two differ on the fixture) | 02 §4.4, [DECISION D-04] |
| MMD-6 | `L_GMMN = 0.1·bg + 1.0·fg` | 02 §4.4 |
| MMD-7 | Gradients of `L_GMMN` are non-zero for adapter, generator **and** `P_point`; the background gradient equals `0.2·Σ_σ exp(−‖x−y‖²/2σ²)(x−y)/σ²`; `gradcheck` passes; `gmmn_detach_point` stops the gradient to `P_point` only | [DECISION D-04] |

All MMD checks run in float64 against references written with Python scalars and explicit loops, to 1e-12. Mutation check (2026-09-19): fourteen wrong variants (square root, clamped result, unbiased estimator, σ instead of σ², missing factor 2, unsquared distance, kernel averaged over bandwidths, a bandwidth dropped, bg weight 1.0, per-class foreground in joint mode, last way dropped, detach always or never, no input check) each fail at least one test. Mutation check for LMA (2026-09-19): fourteen wrong variants (no LayerNorm, LayerNorm after ReLU, no dropout, dropout 0.5, two-layer G, final ReLU in G, z first in the concatenation, z added instead of concatenated, noise in evaluation, no noise in training, uniform or scaled noise, `mean_of_M` accepted, no input check) each fail at least one test. Moving the dropout before the ReLU is an equivalent mutant (a non-negative mask commutes with ReLU) and is not a defect.

### 3.3b `tests/test_clip_text.py` (G1; TXT-6, EP-5, EP-6 in G3 with marker `clip`)

CPU tests use a recording stand-in encoder that returns float16 features, like CLIP on a GPU.

| ID | Check | Source |
| :--- | :--- | :--- |
| TXT-1 | Prompt strings, background prompt, class names verbatim (`shower curtain`), default variant `ViT-B/16` | 03 §2.1, [DECISION D-13] |
| TXT-2 | `E_CLIP` is `[N+1, 512]` float32, equal to `normalize(raw.float())` row by row, row 0 = background | 03 §2.1 |
| TXT-3 | Reordering the ways reorders foreground rows only | 03 §2.1 |
| TXT-4 | Each prompt string is encoded once per process; later episodes reuse the cache | 03 §2.1 |
| TXT-5 | Features of the wrong width raise | 03 §3 |
| TXT-6 | An unknown CLIP variant raises; there is no fallback variant (`clip`) | [DECISION D-13] |
| EP-5 | Two embedders and two episodes load CLIP once; CLIP is in eval mode with no trainable parameter (`clip`) | 03 §2.1 |
| EP-6 | Real embeddings equal a direct `clip.tokenize → encode_text → float → normalize` call bit for bit; distinct prompts give distinct unit vectors (`clip`) | 03 §2.1 |

Mutation check (2026-09-19): twelve wrong variants (background prompt "background clutter", missing period, altered class name, background row last, normalisation in float16, no normalisation, no cache, default `ViT-B/32`, silent fallback variant, reload on every call, CLIP not frozen, no shape check) each fail at least one test.

### 3.4 `tests/test_eppm.py` (G1)

Fixture: `B_q = 2`, N = 2, K = 2, D = 128; `F^s`, `F^q` non-negative (after ReLU) unless stated.

| ID | Check | Source |
| :--- | :--- | :--- |
| GATE-1 | For inputs in [−1000, 1000]: `H ∈ [0, ln 2]`, no NaN/Inf; `H` equals Eq.10 evaluated on Python floats (with the D-16 clamp), is even in x and strictly decreasing in |x| | 02 §5.1, 02 §9 |
| GATE-2 | At θ = 0.5: gate values lie in [0.4046, 0.7311], the lower end reached at x = 0; the output equals `x·σ(2(θ − H(x)))` entry by entry; changing one entry changes only its own gate | 02 §5.1, [DECISION D-02] |
| GATE-3 | θ is a scalar `nn.Parameter` per stage, initial value 0.5; its gradient equals `Σ 2·P·g(1 − g)` for the loss `Σ P_gated`; `gradcheck` passes; `use_gate=false` is the identity with no θ | 02 §5.1, 01 §2.4, D-17 |
| XATT-1 | `A` has shape `[B_q, N+1, K, 128, 128]`; every row sums to 1 | 02 §5.2 |
| XATT-2 | `A[b, c, k]` equals `softmax(Q'[b]ᵀ S'[c, k] / √72)` computed with an explicit loop, VIP-Seg's `MaxPool1d(32)` on the transposed tensor, φ as a matrix and a hand-written softmax (also with `cross_attn_scale=sqrt_D`) | 02 §5.2, [DECISION D-01] |
| XATT-3 | Query and support use the **same** φ module (`id` equal); φ is `Conv1d(64, 72, 1, bias=False)` | 02 §5.2, [PAPER Eq.13] |
| XATT-4 | Background slot of `S'` is built from the way-mean of support features, taken before pooling; permuting the ways keeps slot 0 and permutes slots 1..N | 02 §5.2 |
| XATT-5 | `P_cross` equals the mean over K of `A · ψ(P_gated)` for K = 1, 2, 3 (ψ written out); reordering the shots changes nothing | 02 §5.2 |
| XATT-6 | Permuting queries in the batch permutes outputs identically; one query's output does not depend on the others | 02 §5.2, [DECISION D-01] |
| XATT-7 | `cross_attn_norm` defaults to `none`, which leaves φ untouched; `layernorm` costs 2d = 144 parameters; an unknown value raises | 02 §5.2, [DECISION D-18] |
| XATT-8 | `layernorm` standardises each column of `Q'` along the projection axis (mean 0, variance 1 to the LayerNorm eps) | 02 §5.2, [DECISION D-18] |
| XATT-9 | Scaling every feature by α leaves `A` unchanged to 1e-5 with `layernorm` and moves it by more than 0.1 without | [DECISION D-18] |
| XATT-10 | On features with a per-channel offset at scale 10 the literal Eq.14 collapses: softmax width below 1.1 of 128, `P_cross` exactly constant along D, `‖∇φ‖/‖φ‖` below 1e-12. `layernorm` gives the same three numbers at scale 1 and at scale 10 | [DECISION D-18] |
| DIFF-1 | Mixed-sign features: `P_diffuse` matches Eq.15–18 computed by hand, including a non-zero `c_unique` | 02 §5.3 |
| DIFF-2 | Non-negative features with strictly positive channel means in both branches: `c_unique = 0`, `P_diffuse = (q_ch + s_ch)/4` | 02 §5.3, [DECISION D-14] |
| DIFF-3 | A channel that is zero on all support points but positive in the query gives a non-zero `c_unique` for that channel | 02 §5.3, [DECISION D-14] |
| DIFF-4 | For any input, `P_diffuse` is identical across class rows: it is computed once per query `[B_q, D]` and broadcast; permuting ways or regrouping the same support blocks into one way leaves it unchanged; one query's value does not depend on the others | 02 §5.3 |
| DIFF-5 | The masks use a strict `> τ` (a channel with mean exactly 0 is inactive); gradients reach `F^s` and `F^q` through `q_ch`, `s_ch` | 02 §5.3 |
| (mutation) | 12d (2026-09-19): twenty-three wrong variants of fusion and stage (residual to the gated P, no residual, LayerNorm before the residual, no ReLU before `W_out`, no gate, fusion weights swapped or fixed, max-pooled or one-branch fusion input, sigmoid instead of softmax, one-layer fusion MLP, no SE, SE pooled over channels or with max, SE without ReLU, r = 2, `w_bg` = 1.0, weight on the last row, `w_cls` after `W_out`, diffusion on row 0 only, scaled logits, one query's prototypes for all queries, no shape check) each fail at least one test. "No ReLU before `W_out`" first survived because `P_weighted` was non-negative on the fixture; FUSE-3 now forces negative entries | |
| (mutation) | 12c (2026-09-19): fourteen wrong variants of the diffusion; thirteen fail (no sigmoid, channel mean, max pooling, first way only, query mean over all queries, `≥ τ`, τ = 0.6, α = 0.7, missing `/2` in either term, full masks in `c_unique`, α swapped, detached values). Replacing `m_common` by the union of the masks survives because it is equivalent at α = 0.5: for a channel active in one branch only both give `x/4` | |
| (mutation) | 12b (2026-09-19): eighteen wrong variants of the cross-attention (background slot as way sum, way max, last slot, or mean of pooled ways; average or strided pooling; softmax over the wrong axis; A transposed; a garbled correlation; √D by default; no scale; separate φ for support; φ with bias; shot sum; first shot only; ψ after the attention; no ψ; no point-count check) each fail at least one test | |
| (mutation) | 12a (2026-09-19): thirteen wrong variants of the gate (log₂, one entropy term, no sigmoid, no clamp, ε = 1e-6, θ₀ = 0, no factor 2, `H − θ`, gate replacing P, θ not learnable, θ per channel, gate averaged over channels, disabled gate still gating) each fail at least one test | |
| STAGE-0 | A whole stage equals Eq.10–21 written out from its weights (gate, explicit attention loop, Python-float diffusion, fusion MLP, SE, `w_cls`, `W_out`, LayerNorm by hand) to 1e-11, for K = 1 and 2, with the gate off, with `fusion_weight=per_class` and with `cross_attn_scale=sqrt_D` | 02 §5 |
| FUSE-1 | Fusion weight shape `[B_q, 2]` (`[B_q, N+1, 2]` with `per_class`), rows sum to 1 | 02 §5.4, [DECISION D-11] |
| FUSE-2 | SE vector shape `[B_q, 128]` in (0, 1) | 02 §5.4 |
| FUSE-3 | Row 0 of `P_weighted` equals 0.8 × row 0 of `P_attended`; other rows × 1.0 (checked on the input of `W_out`); `w_cls` is neither a parameter nor a buffer; with ψ shifted negative, `P_weighted` has negative entries and the ReLU of Eq.21 is exercised | 02 §5.4 |
| FUSE-4 | Residual uses the **ungated** `P^{t−1}`: with the weight and bias of `W_out` set to 0, `P^t = LN(P^{t−1})` exactly, whatever the gate does | 02 §5.4, [DECISION D-02] |
| STAGE-1 | `L^t = F^q (P^t)ᵀ` exactly (no scale) | 02 §5.5 |
| STAGE-2 | One EPPM stage has 79,395 parameters (79,394 without the gate), split as in 01 §4; stages do not share parameter tensors | 01 §2.4, 01 §4 |
| STAGE-3 | Permuting the ways permutes rows 1..N of `P^t` and keeps row 0; one query's `P^t` does not depend on the others; every parameter receives a gradient; `gradcheck` in `P^{t−1}` passes | 02 §5 |

### 3.5 `tests/test_adrm_loss.py` (G1)

| ID | Check | Source |
| :--- | :--- | :--- |
| ADRM-1 | `w_gate` shape `[B_q, T]` for T = 2, 3, 4, 6, rows sum to 1, equal to `softmax(W_g · mean_i F^q)` written out | 02 §6 |
| ADRM-2 | `W_g` weight shape `[T, 128]`, no bias | 02 §6 |
| ADRM-3 | `L_final = Σ_t w_gate[t] · L^t` from an explicit loop, to 1e-12; identical stage logits pass through unchanged | 02 §6 |
| ADRM-4 | `∂L_final/∂L^t = w_gate[t]` per query (checked with a random upstream gradient); `W_g` receives a gradient; `gradcheck` passes (on 16 points: ADRM does not depend on the point count); queries do not mix; T = 1 and a wrong stage count raise. Mutation check (2026-09-19): twelve wrong variants (bias, softmax over queries, sigmoid, max pooling, channel pooling, pooling over all queries, uniform weights, reversed stages, last stage only, detached weights, T = 1 allowed, no count check) all fail | 02 §6 |
| LOSS-1 | `L_seg` equals Eq.27 written with Python floats, i.e. `F.cross_entropy(L_final, Y_q)` **without** `weight` (differs from the `w_cls`-weighted CE) | 02 §7, [PAPER Eq.27] |
| LOSS-2 | `L_total = L_seg + 1.0 · L_GMMN` | 02 §7 |
| LOSS-3 | No loss term uses stage logits `L^1..L^{T−1}` directly: the model contract carries only `logits` and `loss_gmmn` | 02 §7 |

### 3.6 `tests/test_ablation_switches.py` (G1)

| ID | Check | Source |
| :--- | :--- | :--- |
| ABL-1 | Each row of the D-17 table runs forward/backward and produces the prescribed prediction tensor (`F^q P_pointᵀ`, `F^q (P^0)ᵀ`, `L^1`, `L^4`, `L_final`), compared with the written-out references to 1e-11 | 01 §3, [DECISION D-17] |
| ABL-2 | `num_stages` ∈ {1..6} gives `w_gate` of width `num_stages` (no ADRM at T = 1). Mutation check of the wiring (2026-09-19): 6/6 wrong variants killed (ADRM never built, `use_adrm` ignored, a stage dropped from the routing, weights from support features, first stage without ADRM, stage logits from P^0) | 01 §3, [PAPER Tab.5] |
| ABL-3 | `modality=image` and `modality=audio` raise `NotImplementedError` | 03 §2.2 |

### 3.6b `tests/test_cascadeproto.py` (G1)

Phase-13 model = all Table 4 rows of D-17, including the full model (CP-7 and CP-8 run for all five). The encoder is the per-point stand-in of 3.2b and CLIP the recording stand-in of 3.3b; feature head, prototypes, adapter, generator, GMMN, logits and loss are the real code. float64, 1e-12. CP-7 and CP-8 run for both rows.

| ID | Check | Source |
| :--- | :--- | :--- |
| CP-1 | Logits `[B_q, 2048, N+1]`; `L_GMMN = 0` when `use_lma=false` | 02 §10, D-17 |
| CP-2 | Logits equal `F^q P_pointᵀ` computed from the extractor and `point_prototypes`, element-wise | [PAPER Eq.23], D-10 |
| CP-3 | Permuting the ways permutes logit channels 1..N and keeps channel 0 | 02 §3, §5.5 |
| CP-4 | Everything after the encoder is per query: with a per-point stand-in encoder, one query's logits do not depend on the other queries (the real encoder couples them through batch statistics, ENC-3) | 02 §2, §5.5 |
| CP-5 | `logit_scale=sqrt_D` divides by √128; `l2norm_point_proto=true` normalises the prototype rows | D-10 |
| CP-6 | Unimplemented switch combinations raise naming their phase; out-of-range switches raise `ValueError` | 01 §3, D-17 |
| CP-7 | The loss reaches every parameter with a non-zero, finite gradient | 02 §7 |
| CP-8 | 30 AdamW steps (lr 1e-3, wd 0.1) on a learnable episode lower the loss by > 20 % and beat chance; `state_dict` round trip is exact | 04 §5 |
| CP-9 | "+ LMA": logits equal `F^q (P_point + P_modal)ᵀ` with `P_modal` written out from the adapter and generator weights (z = 0) | [PAPER Eq.9, Eq.23], D-17 |
| CP-10 | `L_GMMN` equals `0.1·MMD(bg) + 1.0·MMD(fg)` of `P_modal` and `P_point` from the explicit reference; `L_total = CE + L_GMMN`; `gmmn_fg_mode` reaches the loss | [PAPER Eq.8, Eq.26], D-04 |
| CP-11 | The prompts follow the episode's ways in sampling order (checked with a non-sorted order) | 03 §2.1 |
| CP-12 | Evaluation is deterministic (z = 0), training and `eval_noise=sample` are not; `L_GMMN` reaches the feature head unless `gmmn_detach_point` | D-06, D-04 |
| CP-14 | `num_stages = 1, 2, 4` with `use_adrm=false`: logits equal `F^q (P^T)ᵀ` after running P^0 (written out, one copy per query) through the stages in order; stages share no parameter; each adds 79,395 parameters; swapping two stages changes the output; `logit_scale=sqrt_D` scales the cascade output | [PAPER Eq.22–23], D-17 |
| CP-15 | With one stage, `use_adrm` true or false gives identical logits; `use_gate`, `cross_attn_scale`, `fusion_weight` reach every stage | D-17, 01 §3 |
| CP-16 | The cascade keeps way equivariance (row 0 fixed, rows 1..N permuted) and query independence, with and without LMA | 02 §5–6 |
| CP-17 | Full model: logits equal `Σ_t w_gate^(t) L^t` with the stage logits recomputed and `w_gate` written out, for T = 2 and 4; ADRM really mixes the stages; the added modules total 466,444 parameters | [PAPER Eq.24–25], 01 §4 |
| CP-18 | Table 5: T = 1…6 with every switch on train end to end, every parameter receives a gradient; T = 1 has no `W_g` | [PAPER Tab.5], D-17 |
| CP-19 | `cross_attn_norm` reaches every stage and adds 4 × 144 parameters; the default builds no LayerNorm; an unknown value raises | [DECISION D-18] |
| CP-13 | `state_dict` holds only `features.*` and `lma.*` (CLIP excluded); LMA adds exactly 148,352 parameters; `.double()` leaves the CLIP cache in float32; a phase-10 checkpoint configuration still loads | 01 §4, 03 §2.1 |

Mutation check (2026-09-18): eight wrong variants (cosine logits, scale always on, L2 flag ignored or always on, background prototype dropped, non-zero `L_GMMN`, no implementation check, reversed ways) each fail at least one test.

Mutation check for the "+ LMA" row (2026-09-19): twelve wrong variants (`P_modal` not added, used alone or subtracted, GMMN on `P^0`, GMMN dropped, `gmmn_fg_mode` or `gmmn_detach_point` ignored, always detached, `eval_noise` ignored, class names sorted, modality check removed, L2 flag ignored) each fail at least one test. The sorted-names variant first survived because the fixture's classes were already in alphabetical order; CP-11 now uses a non-sorted order.

Mutation check for the cascade rows (2026-09-19): fourteen wrong variants (logits from P^0, mean of stage logits, every stage fed P^0, first stage only, one shared stage, one stage fewer, `use_gate` / scale / fusion flags ignored, cascade started without `P_modal`, ADRM check dropped or applied at T = 1, flag check dropped, `logit_scale` skipped for the cascade) each fail at least one test; the last one first survived and CP-14 now covers `sqrt_D` on the cascade.

### 3.7 `tests/test_eval_metric.py` (G2, marker `cuda`)

The metric is VIP-Seg's `evaluate_metric`, called unchanged; its module imports the VIP-Seg encoder, so these tests need the GPU environment. The reference is an independent implementation of 04 §6.2.

| ID | Check | Source |
| :--- | :--- | :--- |
| EVAL-1 | The project's metric (VIP-Seg's `evaluate_metric`) equals an independent implementation of 04 §6.2 on random predictions over 20 episodes | 04 §6.2 |
| EVAL-2 | A crafted two-episode case where accumulated IoU ≠ mean of per-episode IoU; the project returns the accumulated value | 04 §6.2, [DECISION D-08] |
| EVAL-3 | Background (local label 0) is excluded from the mean | 04 §6.2 |

### 3.8 `tests/test_encoder.py` (G2, marker `cuda`)

| ID | Check | Source |
| :--- | :--- | :--- |
| ENC-1 | Encoder parameter count 2.37M ± 0.01M; feature head 204,260 | 01 §4 |
| ENC-2 | Output `[B, 2048, 128]`, all entries ≥ 0 | 02 §2 |
| ENC-3 | **Batch coupling is VIP-Seg's:** in `eval()` mode, changing the other blocks of one encoder call changes block 0's features (batch-wide mean/std) | 02 §2, [VIPSEG models/encoder.py:281-283,413-415,582-584] |
| ENC-4 | `encode_episode` uses VIP-Seg's composition exactly: support features equal one call on the N·K support blocks, query features one call on the queries, and the queries never change the support features | 02 §2, [VIPSEG models/vipseg.py:79-89] |
| ENC-5 | Input uses 9 channels; passing 3 channels raises | 01 §2.1 |
| ENC-6 | With VIP-Seg's released weights (`load_vipseg_weights`), features equal VIP-Seg's own encoder → L2 norm → `bn` → `fc` path to 1e-6 | [VIPSEG models/vipseg.py:85-97] |
| ENC-7 | On an episode, `encode_episode` equals VIP-Seg's own support and query feature path, including its `permute`/`view` of the loader layout, to 1e-6 | [VIPSEG models/vipseg.py:77-97] |
| ENC-8 | A `state_dict` holds the 6 fixed projections; a fresh model differs until it loads them, then matches exactly | 01 §2.1 |

ENC-1 and ENC-6 need VIP-Seg's checkpoint at `$CASCADEPROTO_VIPSEG_CKPT` (default `<repo>/vipseg_S0_N2_K1.pt`); a missing file fails the test.

### 3.8b `tests/test_resume.py` (G1)

CPU, stand-in encoder and CLIP, a stand-in loader that draws from the global `np.random` and `random` like the inherited one.

| ID | Check | Source |
| :--- | :--- | :--- |
| RES-1 | Training episode i depends only on (seed, i): drawing in reverse order after reseeding the globals gives the same arrays; consecutive episodes differ in their numpy-only parts; the caller's random state is restored | 04 §5 |
| RES-2 | "+ LMA" row (dropout and noise use the torch RNG): 3 epochs uninterrupted equal 1 epoch, stop, resume with differently initialised weights, 2 more epochs — bit for bit in `last.pt`, `best.pt` and the AdamW moments; the resumed part reads episodes 8…23 in order; the learning rate decays once, after epoch 2 | 04 §5, D-12 |
| RES-3 | Resuming with a different training argument raises; `num_workers`, `save_dir`, `data_path` may differ | 04 §5 |
| RES-4 | `--resume` without `resume.pt` starts from scratch and reads every episode once; resuming a finished run trains nothing and leaves `last.pt` unchanged | 04 §5 |
| RES-5 | `resume.pt` holds weights, optimiser, scheduler, counters, all random states, configuration and arguments; no `.tmp` file remains | 04 §5 |

Mutation check (2026-09-19): fourteen wrong variants (random states, optimiser or scheduler not restored, counters reset, every epoch restarting at episode 0, no argument check, resume ignored, `resume.pt` only at validation, non-atomic save, episodes not seeded, only numpy seeded, seed ignoring the index, caller state not restored) — thirteen fail. Restoring only the torch random state survives and is equivalent: training episodes are seeded by index and validation draws nothing, so the global numpy and `random` states do not influence a resumed run. Three survivors of a first version (scheduler, episode offset, seed index) exposed tests that compared two runs of the same faulty code; RES-1, RES-2 and RES-4 now also check absolute properties.

### 3.8c `tests/test_phase14.py` (G1)

| ID | Check | Source |
| :--- | :--- | :--- |
| Q14-1 | The queue holds P1 2, P2 8, P2s 2, P3 6, P4 8 runs with 26 distinct run directories | phase-14 plan |
| Q14-2 | Every Table 4 row runs on S0 and S1 with exactly the D-17 configuration | [PAPER Tab.4], D-17 |
| Q14-3 | Table 5 T = 1…6 and Table 2 (2/3-way, 1/5-shot) are covered on both folds with the right configuration; the extra seeds are full-model runs | [PAPER Tab.2, Tab.5] |
| Q14-4 | Finished steps (`last.pt`, result JSON) are skipped; the rest run in order | 04 §5 |
| Q14-5 | Training commands resume; evaluation commands write JSON; only P1 uses `random600`; the seed is part of the run directory | D-08, D-15 |

Mutation check (2026-09-19): eleven wrong variants (cascade with ADRM, baseline with LMA, gate with two stages, seed not passed, no resume, `random600` everywhere, training or evaluation never skipped, only `best.pt` evaluated, a Table 2 setting or Table 5 depth missing) all fail.

### 3.8d `tests/test_summarize.py` (G1)

| ID | Check | Source |
| :--- | :--- | :--- |
| SUM-1 | The transcribed paper numbers are self-consistent: each Avg is the rounded mean of S0 and S1; Table 4 "full", Table 5 T = 4 and Table 2 2-way 1-shot are the same numbers | [PAPER Tab.2, Tab.4, Tab.5] |
| SUM-2 | Results are shown in %, dry runs are ignored, Avg uses the best checkpoints of both folds, the difference is ours − paper, one run fills every table cell it belongs to, protocols stay separate | D-08, D-15 |
| SUM-3 | Seed spread = sample mean and standard deviation of the finished seeds | phase-14 plan |

Mutation check (2026-09-19): eight wrong variants (no % scale, dry runs counted, Avg mixing best and last, flipped difference, seed runs mixed into the tables, two transcription errors, population standard deviation) all fail.

### 3.8e `tests/test_complexity.py` (G1)

| ID | Check | Source |
| :--- | :--- | :--- |
| CPX-1 | Parameter groups add up to the total; feature head 204,260; added modules 0 (baseline), 227,747 (T = 1), 466,444 (full) | 01 §4, D-09 |
| CPX-2 | fvcore counts FLOPs through the episode wrapper; the full model has more than the baseline; unsupported operators are reported | D-09 |
| CPX-3 | The synthetic episode has the loader shapes and foreground in every support block; the timing returns a positive median | D-09 |

### 3.9 `tests/test_episode.py` (G3, marker `clip`)

Fixture: one synthetic episode with exactly the loader contract (04 §4.3), real CLIP embeddings for S3DIS class names, the full model in float32. The VIP-Seg encoder is the per-point stand-in (it needs CUDA), so G3 also runs on a CPU-only machine.

| ID | Check | Source |
| :--- | :--- | :--- |
| EP-1 | Forward returns `L_final [N, 2048, N+1]` and a positive, finite `L_GMMN`; the model has 4 stages and `W_g` of width 4; the real class embeddings differ between ways | 02 §10 |
| EP-2 | `L_total` finite; every trainable parameter has a finite, non-`None` gradient | 02 §7 |
| EP-3 | One AdamW step (lr 1e-3, wd 0.1) changes parameters and keeps them finite | [PAPER §4.1] |
| EP-4 | `eval()` forward twice → identical logits | [DECISION D-06] |
| EP-5 | CLIP is loaded once for two episodes (load counter = 1); implemented in `tests/test_clip_text.py` (§3.3b) | 03 §2.1 |
| EP-6 | Prompt strings match 03 §2.1 for a known `sampled_classes`; implemented in `tests/test_clip_text.py` (§3.3b) | 03 §2.1 |

### 3.10 Data gate (G4, marker `data`)

| ID | Check | Source |
| :--- | :--- | :--- |
| DATA-1 | `S3DISDataset(cvfold=0, data_path)` finds `meta/s3dis_classnames.txt` in the documented layout | 04 §2.2 |
| DATA-2 | `MyDataset` with the 04 §4.1 arguments returns tensors of 04 §4.3; support masks ∈ {0, 1}; 9 channels; `XYZ` ∈ [0, 1] | 04 §4 |
| DATA-3 | Training classes of fold 0 exclude the S0 test classes and `clutter` | 04 §3 |
| DATA-4 | `MyTestDataset(mode='test')` for S0, 2-way 1-shot yields 1,500 episodes | 04 §6.1 |
| DATA-5 | `python train.py --dataset s3dis --data_path <blocks> --cvfold 0 --n_way 2 --k_shot 1 --dry_run true` loads 4 real episodes, runs one optimiser step and exits 0; the log names the data path, CLIP variant and loader arguments | 04 §4–5 |
| DATA-7 | The valid and test episode sets (1,500 each for S0 2-way 1-shot) share no episode | 04 §6.1, D-15 |
| DATA-6 | `python eval.py --dry_run true ...` evaluates 5 fixed test episodes with the 04 §6.2 metric and refuses to run without a loadable checkpoint (the refusal is also unit-tested, PIPE-6) | 04 §6 |

Before DATA-0…4, `python preprocess/verify_s3dis.py` compares the prepared blocks with `preprocess/s3dis_blocks_manifest.json` (272 rooms, 7,547 blocks, per-block point and class counts, independent of point order). DATA-0…4 are in `tests/test_data.py` (data path `$CASCADEPROTO_S3DIS`, default `datasets/S3DIS/blocks_bs1_s1`; DATA-0 checks that all 272 rooms have loadable blocks; on Windows reading a WSL path, set `HDF5_USE_FILE_LOCKING=FALSE` so the episode cache can be written); DATA-7 checks that no episode appears in both the 1,500 valid and the 1,500 test episodes. DATA-5 and DATA-6 are command-line checks that need the rewritten model (run in phase 10e).

### 3.11 `tests/test_pipeline.py` (G1)

| ID | Check | Source |
| :--- | :--- | :--- |
| PIPE-1 | A loader item becomes an `Episode` in the 02 §1 layout; non-binary masks, 3-channel input and labels above N raise | 02 §1, 04 §4.3 |
| PIPE-2 | A batch of 4 loader items yields 4 separate episodes | 04 §4.3 |
| PIPE-3 | The loss is unweighted CE on `L_final` plus λ·`L_GMMN`; wrong logit shapes raise | 02 §7 |
| PIPE-4 | One optimiser step uses the mean loss of the 4 episodes | 02 §7, [DECISION D-12] |
| PIPE-5 | Default schedule: S3DIS 50×480, ScanNet 30×800 episodes (24,000), 4 per step, AdamW 1e-3 / 0.1, StepLR 10 / 0.5 | 04 §5 |
| PIPE-6 | `eval.py` refuses a missing checkpoint; unimplemented modalities raise; switch defaults are the full model | 04 §6, 03 §2.2, 01 §3 |
| PIPE-7 | `eval.py` rebuilds the configuration stored in the checkpoint (not its own CLI) and reproduces the logits exactly | 01 §3 |
| PIPE-8 | With the same seed the valid and test sets see different random draws; the test set keeps the plain seed; the caller's random state is restored | 04 §6.1, D-15 |

---

## 4. Pipeline sanity check with VIP-Seg (before G5)

Evaluate VIP-Seg's released checkpoint `log_s3dis_VIPSeg/log_S0_N2_K1_0.722026/checkpoint.pt` with this repository's data and evaluation code: `python eval.py --model vipseg --checkpoint <file> ...` (README §5). Reference: mean IoU 0.722026 on 1,500 test episodes [VIPSEG log_s3dis_VIPSeg/log_S0_N2_K1_0.722026/log_vipseg_eval.txt:1-9]. The test episodes are sampled when the `.h5` cache is first built, so a local run uses different episodes and will not match exactly; no tolerance has been measured yet. Record the value; a gap of several points indicates that the data layout, loader arguments or metric differ from VIP-Seg, in which case CascadeProto numbers would not be comparable either.

**Result (2026-09-18, GCP L4):** 0.719687 on this repository's 1,500 S0 2-way 1-shot test episodes, against 0.722026 in VIP-Seg's log and 72.20 in [PAPER Tab.6] (−0.23 points, with independently sampled episodes).

---

## 5. Reproduction targets (G5)

Report every run with S0, S1 and Avg as in 04 §6.2, plus measured parameters/FLOPs (01 §4).

| Target | Paper value | Source |
| :--- | :--- | :--- |
| S3DIS 2-way 1-shot, text, Avg | 86.53 (S0 88.53, S1 84.53) | [PAPER Tab.2] [PAPER Tab.4] |
| S3DIS text, all four settings, Avg | 86.53 / 86.68 / 81.06 / 78.95 | [PAPER Tab.2] |
| ScanNet text, all four settings, Avg | 79.57 / 79.25 / 78.24 / 78.60 | [PAPER Tab.3] |
| Component ablation, S3DIS 2-way 1-shot | 81.28 → 82.49 → 83.91 → 85.97 → 86.53 | [PAPER Tab.4] [DECISION D-17] |
| Cascade depth T = 1..6 | 83.28, 84.55, 85.85, 86.53, 86.51, 86.34 | [PAPER Tab.5] |
| Inference time | grows ≈ 16 ms per stage | [PAPER §4.3] |

No numeric tolerance is defined: the paper reports single runs without variance, and VIP-Seg notes run-to-run randomness [VIPSEG README.md]. Report the gap to each target.

---

## 6. Acceptance matrix

| Area | Passing means | Tests | Audit findings covered |
| :--- | :--- | :--- | :--- |
| Environment | real Mamba encoder, CUDA ops, pinned inherited files | ENV-1..4, ENC-1 | C4, C5, L4 |
| Data | real loader, documented layout, 9 channels, binary masks | DATA-1..6 | C1, C5, M5 |
| Prototypes | per-way MAP on binary masks | PROTO-1..4 | C2, L8 |
| Encoder batching | blocks are separate samples; VIP-Seg's per-episode batch composition reproduced | ENC-3,4,7, XATT-6 | C3, C4, M1 |
| LMA / GMMN | squared MMD, joint fg set, deterministic eval | LMA-1..3, MMD-1..7 | H5, M3 |
| EPPM | gate, channel attention, diffusion, fusion as in 02 §5 | GATE-*, XATT-*, DIFF-*, FUSE-*, STAGE-* | H3, H4, M2 |
| Routing and loss | simplex routing, unweighted CE | ADRM-1..4, LOSS-1..3 | H2 |
| Ablations | Table 4–5 switches exist and behave | ABL-1..3 | H6 (modality errors) |
| Evaluation | accumulated IoU on fixed episodes | EVAL-1..3, DATA-4, DATA-6, §4 | H8 |
| Training schedule | batch 4, 480/800 episodes per epoch, StepLR 10 epochs | DATA-5 plus log review | H7, M7 |
| Reproduction | gaps to Tables 2–5 reported | §5 | — |

Audit finding IDs refer to [docs/research/paper_vs_repo_audit.md](../research/paper_vs_repo_audit.md).
