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
| XATT-11 | `cross_attn_support=pooled` gives one `A[b] ∈ R^{128×128}` per query, rows summing to 1, equal to an explicit loop over `φ(mean of the pooled support blocks)` (also with `sqrt_D`) | 02 §5.2, [DECISION D-23] |
| XATT-12 | With `pooled` the same `A[b]` is applied to every class row, so two class rows with the same prototype give the same `P_cross` | 02 §5.2, [DECISION D-23] |
| XATT-13 | `pooled` averages over ways and shots (order-invariant) and reads every support block; queries do not mix | 02 §5.2, [DECISION D-23] |
| XATT-14 | `pooled` has the same parameters as `class_slots`, and for N = 1, K = 1 the two readings coincide exactly | 02 §5.2, [DECISION D-23] |
| XATT-15 | With two ways the two readings differ; an unknown `cross_attn_support` raises | 02 §5.2, [DECISION D-23] |
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

### 3.4b `tests/test_eppm_s.py` (G1) — beyond the paper [DECISION D-24]

| ID | Check | Source |
| :--- | :--- | :--- |
| EPS-1 | Parameter names and budget: 37,888, no gate, no fusion MLP, no SE, no biases on φ and `W_out` | 01 §4, [DECISION D-24] |
| EPS-2 | The stage equals an explicit-loop reference (φ as a matrix, hand-written row softmax, ψ and `W_out` written out) for both `cross_attn_support` values and N, K, B_q in {1, 2, 3} | [DECISION D-24] |
| EPS-3 | Output shape and LayerNorm statistics | [DECISION D-24] |
| EPS-4 | With `W_out = 0` the stage returns `LN(P^{t-1})`, i.e. the residual survives | [DECISION D-24] |
| EPS-5 | The self term is a per-channel gate in (0, 1) applied to `ψ(P^{t-1})`; `self_branch=False` removes exactly that term | [VIPSEG models/vipseg.py:370-381] |
| EPS-6 | Queries do not mix; permuting the ways permutes the class rows | [DECISION D-24] |
| EPS-7 | The two support readings coincide exactly at N = K = 1 and differ at N = 2 | [DECISION D-23] |
| EPS-8 | Every parameter receives a gradient; `gradcheck` in the prototype | [DECISION D-24] |
| EPS-9 | Shape contract and invalid switches raise | [DECISION D-24] |

### 3.4c `tests/test_vip_stage.py` (G1; VIPS-5 in G2 with marker `cuda`) — reference [DECISION D-25]

| ID | Check | Source |
| :--- | :--- | :--- |
| VIPS-1 | The wrapper calls `module(query, supports, prototype)` in VIP-Seg's own order | [VIPSEG models/vipseg.py:155-158] |
| VIPS-2 | Odd steps add the outer residual, even steps do not | [VIPSEG models/vipseg.py:154-160] |
| VIPS-3 | Shape contract, checked before the module is called | [DECISION D-25] |
| VIPS-4 | `build_stage` alternates the modules; `stage_type=vip` rejects `cross_attn_scale` / `cross_attn_support`, and `eppm_s` rejects the EPPM-only switches | [DECISION D-24] [DECISION D-25] |
| VIPS-5 | Four wrapped stages reproduce VIP-Seg's own reasoning loop on the real PEM/PDM (GPU) | [VIPSEG models/vipseg.py:148-160] |

### 3.5 `tests/test_adrm_loss.py` (G1)

| ID | Check | Source |
| :--- | :--- | :--- |
| ADRM-1 | `w_gate` shape `[B_q, T]` for T = 2, 3, 4, 6, rows sum to 1, equal to `softmax(W_g · mean_i F^q)` written out | 02 §6 |
| ADRM-2 | `W_g` weight shape `[T, 128]`, no bias | 02 §6 |
| ADRM-3 | `L_final = Σ_t w_gate[t] · L^t` from an explicit loop, to 1e-12; identical stage logits pass through unchanged | 02 §6 |
| ADRM-4 | `∂L_final/∂L^t = w_gate[t]` per query (checked with a random upstream gradient); `W_g` receives a gradient; `gradcheck` passes (on 16 points: ADRM does not depend on the point count); queries do not mix; T = 1 and a wrong stage count raise. Mutation check (2026-09-19): twelve wrong variants (bias, softmax over queries, sigmoid, max pooling, channel pooling, pooling over all queries, uniform weights, reversed stages, last stage only, detached weights, T = 1 allowed, no count check) all fail | 02 §6 |
| LOSS-1 | `L_seg` equals Eq.27 written with Python floats, i.e. `F.cross_entropy(L_final, Y_q)` **without** `weight` (differs from the `w_cls`-weighted CE) | 02 §7, [PAPER Eq.27] |
| LOSS-2 | `L_total = L_seg + 1.0 · L_GMMN` | 02 §7 |
| LOSS-3 | No loss term uses stage logits `L^1..L^{T−1}` directly: the model contract carries `logits` and `loss_gmmn`, plus D-29's optional `loss_distill` / `distill_weight` (on `M_eff`, default weight 0) | 02 §7, [DECISION D-29] |

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

### 3.8f `tests/test_phase16.py` (G1) — the R1 queue [DECISION D-22]

| ID | Check | Source |
| :--- | :--- | :--- |
| R1-1 | Every `r1_*` variant of `experiments/diag_short.py` produces the configuration it claims, and passes the switch guards | 16e, [DECISION D-23] [DECISION D-24] [DECISION D-25] |
| R1-2 | Consecutive variants differ in exactly one switch, so R1 changes one variable at a time | 16e |
| R1-3 | `experiments/summarize_r1.py` reproduces hand-computed mean, sd and Welch's t | 16e |
| R1-4 | A variant that has not run is reported as missing, not guessed; a log without results exits non-zero | 16e |
| R1-5 | `run_r1.sh` screens on S1, three seeds, 9,600 episodes, and states rules R1.1…R1.5 | [DECISION D-22] |

### 3.8g `tests/test_transductive.py` (G1) — query-side EM refinement and its probe [DECISION D-26]

| ID | Check | Source |
| :--- | :--- | :--- |
| EM-1 | `κ = 0` or `T = 0` returns the model's logits for every weight arm | [DECISION D-26] |
| EM-2 | The refined prototype keeps the prior's norm and changes its direction | [DECISION D-26] |
| EM-3 | A uniform posterior (no confident mass) leaves every prototype unchanged at `κ = 16` | [DECISION D-26] |
| EM-4 | The oracle arm at `κ = 10⁴` gives the direction of the query class's mean unit feature; `oracle` without labels raises | [DECISION D-26] |
| EM-5 | Normalised entropy is 1 for a uniform and 0 for a one-hot posterior; the entropy weight follows | [DECISION D-26] |
| EM-6 | SSP thresholds are per class: 0.65 keeps a background point, drops a foreground one | [DECISION D-26] |
| EM-7 | The M-step equals the hand-written vMF MAP form; a class with 1 of 64 points moves < 10 % of one with 63 | [DECISION D-26] |
| EM-8 | On a synthetic query whose support prototypes are perturbed, refinement raises point accuracy by > 5 points | [DECISION D-26] |
| EM-9 | The entropy diagnostics bin points by weight and count correct argmax predictions | [DECISION D-26] |
| EM-10 | The probe's VIP-Seg reader reproduces the gated logits of a stand-in with VIP-Seg's call pattern, removes its hooks, and fails without the outer residual | [DECISION D-26] [DECISION D-25] |
| EM-11 | The per-episode count metric equals VIP-Seg's accumulated metric (`metrics_alt`) | [DECISION D-08] |
| EM-12 | The paired bootstrap gives 0 for identical arms and a CI above 0 for a uniformly better arm | [DECISION D-26] |
| EM-13 | Selection never picks an oracle arm; rules P0.1 (go), P0.2 (stop), P0.3 (S1 passes, S0 does not), P0.5, P0.6 and "incomplete" | [DECISION D-26] |
| EM-14 | Selection refuses an S0 checkpoint before loading anything | [DECISION D-22] [DECISION D-26] |

Mutation check (2026-09-23), 11 mutants: 10 killed. The go rule without its S0 condition first survived and is
killed since EM-13 has the "S1 passes, S0 does not" case; the remaining survivor (gains computed over the oracle
arms too) is equivalent, since the maximum is taken over non-oracle arms only. Not verified locally: the probe on the real
checkpoints, which needs the CUDA encoder (gate G2).

### 3.8h `tests/test_base_calibration.py` (G1) — base-class calibration and its probe [DECISION D-27]

| ID | Check | Source |
| :--- | :--- | :--- |
| BPC-1 | An occurrence prototype is the unit mean of CL2N features; an empty mask gives 0 | [DECISION D-27] |
| BPC-2 | The bank needs its centre first; the centre is the mean of every support and query point; ways map to their classes (query label way+1); empty occurrences are skipped; below `min_count` it raises | [DECISION D-27] |
| BPC-3 | Support prototypes are built like the bank's | [DECISION D-27] |
| BPC-4 | The margin is the base cosine minus the cosine to the predicted class's support prototype, `−∞` on background predictions | [DECISION D-27] |
| BPC-5 | No flip returns the logits; only flipped points move, all to the background, even under a large foreground logit; foreground logits are unchanged | [DECISION D-27] |
| BPC-5b | The top fraction q of a query's foreground predictions flips, never a point with a non-positive margin; `⌊q·n⌋ = 0` flips nothing | [DECISION D-27] |
| BPC-6 | The histogram AUC equals the pairwise AUC on the same binning | [DECISION D-27] |
| BPC-7 | Selection takes the q of the best mean gain | [DECISION D-27] |
| BPC-8 | Rules P1.1 (a large gain whose CI contains 0 is no go), P1.2, P1.3, P1.4 both ways, P1.5 and "incomplete" | [DECISION D-27] |
| BPC-9 | Selection refuses an S0 checkpoint before loading anything | [DECISION D-22] |
| BPC-10 | The bank cache is keyed by its episode count, so a smoke bank never stands in for the full one | [DECISION D-27] |

Mutation check of the revised form (2026-09-23), 16 mutants: 16 killed, after BPC-5 gained the "large foreground
logit" case that killed the one first survivor. Not verified locally: the probe on the real checkpoints (gate G2).

### 3.8i `tests/test_fused.py` (G1) — base-margin-filtered EM and its probe [DECISION D-28]

| ID | Check | Source |
| :--- | :--- | :--- |
| FUS-1 | Keeping every point equals D-26's step; keeping none leaves the foreground prototypes unchanged while the background M-step still runs | [DECISION D-28] |
| FUS-2 | The keep mask is all points at r = 0 and the complement of §12's top-fraction flips otherwise | [DECISION D-28] |
| FUS-3 | The false share counts foreground mass on points of another label and ignores the background column | [DECISION D-28] |
| FUS-4 | An r = 0 arm reproduces D-26's refinement exactly; the diagnostic is recorded once per (weight, r) | [DECISION D-28] |
| FUS-5 | Selection reads the VIP-Seg S1 checkpoint only, never ours, and raises without exactly one | [DECISION D-28] [DECISION D-25] |
| FUS-6 | Rules P2.0 (a gain without the mechanism is a stop; an unfiltered winner adds nothing), P2.1, P2.2, P2.3, P2.4, P2.5, incomplete | [DECISION D-28] |
| FUS-7 | Selection refuses an S0 checkpoint before loading anything | [DECISION D-22] |
| FUS-8 | The probes and their modules parse as Python 3.10, the VM's version | 00 §5.2 |

Mutation check (2026-09-23), 8 mutants: 7 killed, the survivor (removing the r = 0 shortcut) is equivalent.

### 3.8j `tests/test_distill.py` (G1) — oracle-direction distillation and the R2 evaluation [DECISION D-29]

| ID | Check | Source |
| :--- | :--- | :--- |
| DIS-1 | `O` is the normalised sum of the unit features of each present class, the presence mask is exact, and `O` equals the direction of 02 §11 with the oracle weight and κ → ∞ | 02 §14, [DECISION D-29] |
| DIS-2 | The teacher is `F^q Oᵀ`; `L_distill` is 0 for any positive multiple of the teacher plus a per-point shift, 2 for a negative one, averages over present pairs only, reads only the points of the two classes of a pair, matches the written-out cosine, and gives a graph-keeping zero without pairs; a common shift of the prototypes changes `cos(M_c, O_c)` but neither the loss nor the pairwise prototype cosine | 02 §14, [DECISION D-29] |
| DIS-3 | The teacher is a stopped gradient: the loss reaches the logits, never the features through the target | [DECISION D-29] |
| DIS-4 | `F^q M_effᵀ` reproduces the model's logits without stages, without ADRM, with ADRM, with `logit_scale=sqrt_D` and for the full model; the R2 rule reads the same tensors | 02 §14, 02 §6 |
| DIS-5 | The logits do not depend on the query labels in either mode; evaluation returns no `L_distill` | [DECISION D-29] |
| DIS-6 | β = 0 gives the paper's objective bit for bit with `L_distill` logged without gradient; β > 0 adds `β L_distill`, whose gradient reaches ADRM and the stages; a weight without the loss raises | 02 §7, 02 §14 |
| DIS-7 | `distill_beta` must be finite and ≥ 0; the CLI passes it, `run_r2.sh`'s run directories are the ones `train.py` writes, and older checkpoints resume at the default | [DECISION D-29], 04 §5 |
| DIS-8 | Both oracle rules leave absent classes untouched, one keeping each norm and one giving the present classes their mean norm; the logit-pair diagnostic is the model's cosine to the teacher; prototype-space sums and counts per term | [DECISION D-29] |
| DIS-9 | Rules R2.0–R2.5 and "incomplete"; the rules survive several draws (the hooks are closed once); the test refuses a checkpoint of another fold before loading it | [DECISION D-29] [DECISION D-22] |
| DIS-10 | The new and changed files parse as Python 3.10, the VM's version | 00 §5.2 |
| DIS-11 | Rules E1.1–E1.3 of D-30 at their thresholds, "incomplete" without draws or names; named pairs are computed only when both models are scored | [DECISION D-30] |

Mutation check (2026-09-23, revised form), 25 mutants over `models/oracle_distill.py`, `models/cascadeproto.py`, `pipeline/model_api.py`, `train.py` and `experiments/r2_distill_eval.py`: 25 killed, the last after DIS-8 gained the model-against-teacher case.

E1 rules (2026-09-24), 6 mutants: 6 killed, the last after DIS-11 gained the unscored-model case.

### 3.8k `tests/test_text_prior.py` (G1) — training-free text prior and P3's gap decomposition [DECISION D-31]

| ID | Check | Source |
| :--- | :--- | :--- |
| TXT-1 | The four prompt sets, the 12 frozen descriptions, unknown sets or classes raise | [DECISION D-31] |
| TXT-2 | Class embeddings average unit prompt embeddings and renormalise | [DECISION D-31] |
| TXT-3 | The ridge map interpolates the base anchors, returns unit directions and uses centred embeddings | [DECISION D-31] |
| TXT-4 | Retrieval picks the nearest base prototype at large τ and the normalised mean at τ = 0 | [DECISION D-31] |
| TXT-5 | Text logits are CL2N features times the directions; text-only query accuracy maps column c−1 to label c and skips background | [DECISION D-31] |
| TXT-6 | The support gate is 1 for correct text, 0 for swapped text and 2·acc − 1 in between | [DECISION D-31] |
| TXT-7 | κ = 0 or w = 0 is the head; background untouched; a shift common to the ways is ignored; the ways are only re-ranked | [DECISION D-31] |
| TXT-8 | The entropy weight is 1 for uniform and ≈ 0 for confident posteriors | [DECISION D-31] |
| TXT-9 | The equal-norm oracle keeps absent classes and gives present classes their mean norm | [DECISION D-31] |
| TXT-10 | Alignment sums use the two foreground classes' points only | [DECISION D-31] |
| TXT-11 | Boundary points are exactly those whose k nearest neighbours include another label | [DECISION D-31] |
| TXT-12 | The grid has 192 arms, selection takes the best valid gain, the sibling flips the weight, S0 is refused | [DECISION D-31] [DECISION D-22] |
| TXT-13 | Rules P3.0–P3.5 and Part A's bands; the Part A summary on a synthetic set | [DECISION D-31] |
| TXT-14 | The new files parse as Python 3.10 | 00 §5.2 |

Mutation check (2026-09-24), 17 mutants: 17 killed, the last two after TXT-3 and TXT-5 gained the centring and query-accuracy cases.

### 3.8l `tests/test_background_probe.py` (G1, BG-9 on G4) — background contamination probe P4 [DECISION D-32]

| ID | Check | Source |
| :--- | :--- | :--- |
| BG-1 | Foreground and background prototypes are the masked means; an empty background raises | [DECISION D-32] |
| BG-2 | With every mask-0 point kept, P^0 equals the model's L2 point prototypes | 02 §3, [DECISION D-10] |
| BG-3 | The purifier's score for a block ignores the block's own way | [DECISION D-32] |
| BG-4 | q = 0 keeps every mask-0 point; q > 0 drops the points most similar to another way, per block | [DECISION D-32] |
| BG-5 | The re-run head from P^0 reproduces the model's logits; row oracles touch only their rows | [DECISION D-32] |
| BG-6 | The histogram AUC is 1, 0 and 0.5 in the limiting cases and nan without positives | [DECISION D-32] |
| BG-7 | Recall and precision per column; the fraction with the best draw-A gain is frozen | [DECISION D-32] |
| BG-8 | Rules P4.1 (gain, CI and the floor/wall recall rise), P4.2, P4.3 | [DECISION D-32] |
| BG-9 | On the real blocks, `support=False` labels give exactly the loader's `support=True` mask for the same RNG state; drawn episodes have the loader's shapes | [VIPSEG dataloaders/loader.py:31-87] |
| BG-10 | The new files parse as Python 3.10 | 00 §5.2 |

Mutation check (2026-09-24), 9 mutants: 9 killed.

### 3.8m `tests/test_neck.py` (G1) — support → query attention neck and warm start [DECISION D-33]

| ID | Check | Source |
| :--- | :--- | :--- |
| NECK-1 | α = 0 at initialisation and the neck is the identity | [DECISION D-33] |
| NECK-2 | The update is the written attention; it reads every query of the episode; wrong shapes raise | 02 §15 |
| NECK-3 | At initialisation only α receives gradient | [DECISION D-33] |
| NECK-4 | A model with a closed neck equals the model without it; with α ≠ 0, P^0 depends on the query; parameter count | [DECISION D-33] |
| NECK-5 | Configuration: `none` builds nothing, unknown necks raise | [DECISION D-33] |
| NECK-6 | Warm start loads every tensor of the checkpoint, leaves α at 0, and raises on a different configuration, a missing non-neck key or an unexpected key | [DECISION D-33] |
| NECK-7 | CLI, run directories `_b1_ft` / `_b1_sq_attn_ft`, resume at the defaults | [DECISION D-33], 04 §5 |
| NECK-8 | Rules N1.1–N1.4 and "incomplete" | [DECISION D-33] |
| NECK-9 | The changed files parse as Python 3.10 | 00 §5.2 |
| NECK-10 | α starts at `neck_alpha_init` (0.1 from scratch) and the projections then get gradient at once; the value needs a neck and must be finite; CLI and run directory `_sq_attn_a0.1` | [DECISION D-34] |

Mutation check (2026-09-24), 10 mutants: 10 killed, the last three after NECK-4, NECK-6 and NECK-8 gained the query-dependence, missing-key and CI cases.

### 3.8n `tests/test_condition_probe.py` (G1) — condition and presence probe P5 [DECISION D-35]

| ID | Check | Source |
| :--- | :--- | :--- |
| P5-1 | Block status: own for the sampled-for class, other when present, absent otherwise; one block per way or it raises | [DECISION D-35], [VIPSEG dataloaders/loader.py:174-225] |
| P5-2 | The own/other/absent split partitions the pooled GT, TP and FP counts on random episodes | [DECISION D-35] |
| P5-3 | Split counts by hand, including FP of a class in a block that lacks it | [DECISION D-35] |
| P5-4 | Counterfactual cf-a raises other-condition TP to the own recall and never lowers it; cf-b removes absent-block FP; background untouched | [DECISION D-35] |
| P5-5 | Recall per condition, other share and FP ratios; raw counts kept separately | [DECISION D-35] |
| P5-6 | Presence-fair oracle: present rows are the query's own directions, absent rows the support's, same geometry | [DECISION D-35] |
| P5-7 | The support rule is `F^q n(P_point)ᵀ` | [DECISION D-31] |
| P5-8 | Batch statistics normalise with the batch, leave running statistics and modes unchanged, and raise without BatchNorm | [DECISION D-35] |
| P5-9 | The index copy of the sampler equals the inherited `sample_pointcloud` (both sampling modes) and over-samples by 2 − π | [VIPSEG dataloaders/loader.py:31-89] |
| P5-10 | False-positive kinds: episode, base, clutter, novel outside the episode | [DECISION D-35] |
| P5-11 | A sparse view keeps the shape, puts the foreground at the raw share (background density), re-normalises XYZ, raises without foreground | [DECISION D-35] |
| P5-12 | Dual logits take the background from the dense run and the foreground maximum; the control adds one constant and matches the foreground count | [DECISION D-35] |
| P5-13 | φ and the intervention summary (pooled recalls, own recall, bootstrap CI over episodes) | [DECISION D-35] |
| P5-14 | Rules P5.1 (a / b / in between / stop), P5.2, P5.3, P5.4, "neither" and "incomplete" | [DECISION D-35] |
| P5-15 | The new files parse as Python 3.10 | 00 §5.2 |

Mutation check (2026-09-24), 10 mutants: 10 killed. The first run of P5-5 found a real defect (the raw counts
overwrote the FP ratios under the same keys); fixed before any GPU run.

### 3.8o `tests/test_d37_trace.py` (G1, T6 on G2) — cross-term forms and the 2 × 2 [DECISION D-36] [DECISION D-37]

The inherited PEM / PDM are parsed from `models/vipseg.py` with `ast` (never edited), float64, projection weights
scaled by 20 so that a positional path is visible.

| ID | Check | Source |
| :--- | :--- | :--- |
| D37-T1 | The written-out forward with the `scrambled` cross-term reproduces the inherited PEM and PDM (N = 2, 3; K = 1, 2) to 1e-12; `CrossFormModule` in `native` calls the module unchanged | [DECISION D-36], [VIPSEG models/vipseg.py:235-405] |
| D37-T2 | `scrambled` attention equals C2's index formula; `clean` equals `softmax(Q'_bᵀ S'_w / √128)` per query and slot; unknown forms and modules raise | [DECISION D-35] point 5, [DECISION D-36] |
| D37-T3 | Four `clean` stages: permuting the queries permutes the prototypes, changing another query leaves a query's prototypes unchanged; VIP-Seg's form fails both | [DECISION D-37] |
| D37-T4 | `stage_type=vip_clean` builds `clean` stages with the state-dict keys of `vip` (an E1 checkpoint loads strictly); ignored switches and unknown forms raise | [DECISION D-36] |
| D37-T5 | The lemma of D-37 on the whole model in training mode: a permuted episode gives the same loss and every gradient (1e-10 of the largest) with `vip_clean`; `vip` does not | [DECISION D-37] |
| D37-T6 (cuda) | The real model in float32: `scrambled` reproduces `native` within C4.0a's tolerance; under a query swap the `clean` logits, training loss and gradients change by at most 3× the rounding floor (every query coordinate moved by one ULP), `native`'s loss by more than 5× | [DECISION D-36], [DECISION D-37] |
| D37-T7 | `QueryOrder` permutes blocks and labels together by `default_rng([seed, 3, i])`, leaves supports and classes, swaps about half the episodes, never draws from the global RNG, is deterministic per seed | [DECISION D-37] |
| D37-T8 | `--query_order` defaults to `fixed`; E1's run directory is unchanged, VR / CR get `_qrandom`; a checkpoint without the flag resumes at `fixed` | [DECISION D-37] |
| D37-T9 | C4's own-class bookkeeping (own, position label 2 − b, background), rule C4.1 and check C4.0b | [DECISION D-36] |
| D37-T10 | D-37's reader: the E1 reproduction check gates everything, D37.2 failure stops, D37.1 origin / stop, D37.3 clean / scrambled / tie, D37.4 reports | [DECISION D-37] |
| D37-T11 | The new files parse as Python 3.10 | 00 §5.2 |

Mutation check (2026-09-25), 17 mutants (cross-term forms, shot average, stage switch, configuration mapping, query
order, run directory, C4 and D-37 readers): 17 killed. D37-T5's first run flagged the biases in front of a BatchNorm,
whose gradient is analytically zero (rounding noise only); the tolerance is relative to the largest gradient.
D37-T6's first GPU run used a fixed 1e-2 margin for VIP-Seg's positional effect, which a freshly initialised model
does not reach in evaluation (8e-4); the test now compares every quantity with the one-ULP rounding floor, and passed
three runs in a row on the L4.

### 3.8p `tests/test_prototype_probe.py` (G1) — prototype probe P6 [DECISION D-38]

| ID | Check | Source |
| :--- | :--- | :--- |
| P6-1 | U's rows are D-35's support directions, the same unit rows for every query block | [DECISION D-38] |
| P6-2 | Row oracles replace only the present rows of their set (bg / fg / all) with the query's unit-feature direction; absent rows and other rows untouched; all rows unit | [DECISION D-38], [DECISION D-35] |
| P6-3 | Self-support by hand: the lowest-entropy fraction ρ of a class's predicted points, mixed with weight 1 − α | [DECISION D-38] |
| P6-4 | Self-support: α = 1 and too few points leave the rows; T = 2 is T = 1 twice; per query block and order-free; the row set is respected | [DECISION D-38] |
| P6-5 | Spherical k-means: deterministic, recovers separated modes, k = 1 is the unit-feature mean; the multi-background logit is the maximum over centroids, foreground untouched | [DECISION D-38] |
| P6-6 | The 54-arm grid, arm predictions, oracle arms refuse to run without labels | [DECISION D-38] |
| P6-7 | Selection (ties to the earlier arm) and the P6.1 location rule | [DECISION D-38] |
| P6-8 | Rules P6.2–P6.4, including a negative random600 draw blocking a go, and "incomplete" | [DECISION D-38] |
| P6-9 | The new files parse as Python 3.10 | 00 §5.2 |

Mutation check (2026-09-25), 9 mutants: 9 killed. The first check left a go without its CI / random600 condition
alive; P6-8 gained that case.

### 3.8q `tests/test_self_support.py` (G1) — trained self-support prototypes [DECISION D-39]

| ID | Check | Source |
| :--- | :--- | :--- |
| SS-1 | `unit_prototypes` equals D-35's support directions; a way without foreground raises | [DECISION D-39] |
| SS-2 | The trained module equals P6's rule (ρ = 1) at fixed α: background only (α_fg = 1) and all rows; every step returned; too few predicted points keep the rows | [DECISION D-38], [DECISION D-39] |
| SS-3 | Gradients reach θ_bg, θ_fg and the query features; α initialised at 0.25 / 0.5 | [DECISION D-39] |
| SS-4 | The unit rule refuses stages, LMA, L2 point prototypes and a neck; self-support needs the unit rule; the auxiliary weight needs self-support | [DECISION D-39] |
| SS-5 | Unit-rule logits (T = 0 and T = 2), effective prototype = last step, auxiliary CE on the support-only logits in training only, added by `episode_loss` | [DECISION D-39] |
| SS-6 | Training loss and every gradient are unchanged by a query permutation (order-free, as D37-T5) | [DECISION D-37], [DECISION D-39] |
| SS-7 | `train.py` flags, run directories, resuming a checkpoint without the flags | [DECISION D-39] |
| SS-8 | The D-39 reader's k-means background (max with the row) and inference arms | [DECISION D-39] |
| SS-9 | Rules D39.1–D39.3, including a negative random600 draw blocking a go, and "incomplete" | [DECISION D-39] |
| SS-10 | The new files parse as Python 3.10 | 00 §5.2 |

Mutation check (2026-09-25), 10 mutants: 10 killed (two survivors of the first check, the minimum-points rule and a go
without its CI condition, gained tests). LOSS-3 (§3.6) now lists the optional `loss_aux` / `aux_weight` fields.

### 3.8r `tests/test_propagation_probe.py` (G1) — label propagation on the query graph [DECISION D-40]

| ID | Check | Source |
| :--- | :--- | :--- |
| P7-1 | kNN graphs against brute force: neighbour sets (space or cosine, never the point itself) and affinities (1 or [cos]₊³); unknown graph and k ≥ P raise | [DECISION D-40], Iscen et al. Eq. 9 |
| P7-2 | W = A + Aᵀ, symmetric with zero diagonal; S = D^{−1/2} W D^{−1/2} entry by entry; an isolated point gets a zero row and column, no NaN | [DECISION D-40], Zhou et al. |
| P7-3 | The Cholesky solve equals the limit of F ← βSF + (1 − β)Y₀ for every β of the grid; β = 0 returns Y₀; β = 1 raises | [DECISION D-40], Zhou et al. |
| P7-4 | An isolated point keeps its seed | [DECISION D-40] |
| P7-5 | P7a quantities by hand on a path (homophily, local recall, one-hop vote with ties counted as neither, summands, fixed / broken) and against a loop form on random weighted graphs; reachability against BFS; a class without a correct seed is unreachable | [DECISION D-40] |
| P7-6 | Column and condition of every point (own / other of D-35), sums by (column, condition), one query block per way; pooled and per-class ratios | [DECISION D-35], [DECISION D-40] |
| P7-7 | The seeds are D-39's U + both; every arm and `lp_u` equal the spreading on their graph and seed | [DECISION D-39], [DECISION D-40] |
| P7-8 | No label enters a prediction; swapping the query blocks or permuting the points of a block permutes the output | [DECISION D-37], [DECISION D-40] |
| P7-9 | Rules P7.1–P7.7 (gate, adopt, trained form, between with a negative random600 draw, stop, leak-free reading, mechanism confirmed / unexplained, incomplete); the 24 arm names; a selection tie goes to the earlier arm | [DECISION D-40] |
| P7-10 | The new files parse as Python 3.10 | 00 §5.2 |

Mutation check (2026-09-25), 22 mutants: 22 killed (four survivors of the first check, the seed source, the leak-free
reading, the missed-point homophily summand and the gate threshold, gained tests).

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

### 3.12 `tests/test_protocol_guard.py` (G1)

| ID | Check | Source |
| :--- | :--- | :--- |
| PROT-1 | A checkpoint scored on the fold it was trained on is `clean` | [DECISION D-22] |
| PROT-2 | Scoring the other fold raises; with `--allow_seen_classes true` it runs and is labelled a seen-class diagnostic | [DECISION D-22] |
| PROT-3 | A checkpoint trained with `--train_classes all` raises even on its own fold | [DECISION D-21] [DECISION D-22] |
| PROT-4 | A checkpoint that records no fold needs `--checkpoint_cvfold`; a stated fold contradicting the recorded one raises; checkpoints without `train_classes` count as `split` | [DECISION D-22] |
| PROT-5 | `eval.py` runs the check before any episode is built | [DECISION D-22] |
| PROT-6 | `run_seen.sh` is the one script that passes `--allow_seen_classes true`; the clean re-scoring script states the VIP-Seg checkpoint's fold | [DECISION D-22] |

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
