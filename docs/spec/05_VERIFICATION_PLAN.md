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
| G0 Environment | GPU runtime | `pytest tests/test_environment.py -v` | Required packages import; CUDA available |
| G1 Unit (CPU) | nothing external | `pytest -m "not cuda and not clip and not data" -v` | Math and wiring of 02–03 |
| G2 Unit (GPU) | CUDA, `mamba_ssm`, `pointnet2_ops` | `pytest -m cuda -v` | Encoder batching and parameter counts |
| G3 Integration | G2 + CLIP weights | `pytest -m clip -v` | End-to-end episode with real CLIP embeddings |
| G4 Data | preprocessed S3DIS | `pytest -m data -v`, then `python train.py --dry_run ...` | Real loader, real episodes, one optimiser step |
| G5 Reproduction | full data, compute | training + evaluation runs | Tables 2–5 comparison (§5) |

Markers `cuda`, `clip`, `data` are registered in `pytest.ini`.

---

## 3. Test specifications

### 3.1 `tests/test_environment.py` (G0)

| ID | Check | Source |
| :--- | :--- | :--- |
| ENV-1 | `import mamba_ssm`, `import pointnet2_ops`, `import clip`, `import h5py`, `import transforms3d` succeed | 01 §2.1, 04 §4 |
| ENV-2 | `torch.cuda.is_available()` and a 1-point CUDA op runs on the target GPU | [PAPER §4.1 "single NVIDIA RTX 5090"] |
| ENV-3 | Inherited files are byte-identical to the pinned VIP-Seg commit: `dataloaders/{loader,s3dis,scannet}.py`, `preprocess/{collect_s3dis_data,collect_scannet_data,room2blocks}.py`, `utils/{checkpoint_util,cuda_util,logger}.py` (SHA-256 list stored in the test) | 04 header, [00 §5.2](00_SOURCES_AND_DECISIONS.md) |
| ENV-4 | `models/encoder.py` contains no fallback block and imports `mamba_ssm` unconditionally | 01 §2.1 |

### 3.2 `tests/test_prototypes.py` (G1)

Fixture: `F^s ∈ [N=3, K=2, 2048, 128]` random, binary masks with different fore-/background points per way.

| ID | Check | Source |
| :--- | :--- | :--- |
| PROTO-1 | Row k of `P_point` equals the manual mean of `F^s[k−1]` over its mask-1 points, for every k = 1..N | 02 §3 |
| PROTO-2 | Row 0 equals the mean over all mask-0 points of all ways and shots | 02 §3 |
| PROTO-3 | No row is all-zero; rows 1..N differ pairwise when the masks differ | 02 §3 |
| PROTO-4 | Changing way 2's mask leaves rows 1 and 3 unchanged | 02 §3 |

### 3.3 `tests/test_lma_gmmn.py` (G1)

| ID | Check | Source |
| :--- | :--- | :--- |
| LMA-1 | Adapter `[N+1, 512] → [N+1, 128]`; generator input width 256; parameter counts 82,432 and 65,920 | 02 §4.1–4.2, 01 §2.3 |
| LMA-2 | `model.eval()`: two forwards give identical `P_modal` (z = 0) | 02 §4.2, [DECISION D-06] |
| LMA-3 | `model.train()`: two forwards differ (fresh z) | 02 §4.2 |
| MMD-1 | `k(x, x) = 6` | 03 §4.1 |
| MMD-2 | `MMD({x}, {y}) = 2·(6 − k(x, y))` to 1e-5 | 03 §4.1 |
| MMD-3 | `MMD(X, X) = 0`; `MMD ≥ −1e-6` on random sets | 03 §4.1 |
| MMD-4 | Result equals the explicit triple-mean formula of 02 §4.3 (no square root) | 02 §4.3 |
| MMD-5 | Foreground term uses rows 1..N as one set: equals `MMD(P_modal[1:], P_point[1:])`, not the mean of per-row MMDs (for N ≥ 2 the two differ on the fixture) | 02 §4.4, [DECISION D-04] |
| MMD-6 | `L_GMMN = 0.1·bg + 1.0·fg` | 02 §4.4 |
| MMD-7 | Gradients of `L_GMMN` are non-zero for adapter, generator **and** `P_point` | [DECISION D-04] |

### 3.4 `tests/test_eppm.py` (G1)

Fixture: `B_q = 2`, N = 2, K = 2, D = 128; `F^s`, `F^q` non-negative (after ReLU) unless stated.

| ID | Check | Source |
| :--- | :--- | :--- |
| GATE-1 | For inputs in [−1000, 1000]: `H ∈ [0, ln 2]`, no NaN/Inf | 02 §5.1, 02 §9 |
| GATE-2 | At θ = 0.5: gate values lie in [0.4046, 0.7311] | 02 §5.1 |
| GATE-3 | θ is a scalar `nn.Parameter` per stage, initial value 0.5, receives gradient | 02 §5.1, 01 §2.4 |
| XATT-1 | `A` has shape `[B_q, N+1, K, 128, 128]`; every row sums to 1 | 02 §5.2 |
| XATT-2 | `A[b, c, k]` equals `softmax(Q'[b]ᵀ S'[c, k] / √72)` computed with an explicit loop | 02 §5.2, [DECISION D-01] |
| XATT-3 | Query and support use the **same** φ module (`id` equal); φ is `Conv1d(64, 72, 1, bias=False)` | 02 §5.2, [PAPER Eq.13] |
| XATT-4 | Background slot of `S'` is built from the way-mean of support features | 02 §5.2 |
| XATT-5 | `P_cross` equals the mean over K of `A · ψ(P_gated)`; with K = 1 the mean is the single term | 02 §5.2 |
| XATT-6 | Permuting queries in the batch permutes outputs identically (no cross-query mixing) | 02 §5.2, [DECISION D-01] |
| DIFF-1 | Mixed-sign features: `P_diffuse` matches Eq.15–18 computed by hand, including a non-zero `c_unique` | 02 §5.3 |
| DIFF-2 | Non-negative features with strictly positive channel means in both branches: `c_unique = 0`, `P_diffuse = (q_ch + s_ch)/4` | 02 §5.3, [DECISION D-14] |
| DIFF-3 | A channel that is zero on all support points but positive in the query gives a non-zero `c_unique` for that channel | 02 §5.3, [DECISION D-14] |
| DIFF-4 | For any input, `P_diffuse` is identical across class rows | 02 §5.3 |
| FUSE-1 | Fusion weight shape `[B_q, 2]`, rows sum to 1 | 02 §5.4, [DECISION D-11] |
| FUSE-2 | SE vector shape `[B_q, 128]` in (0, 1) | 02 §5.4 |
| FUSE-3 | Row 0 of `P_weighted` equals 0.8 × row 0 of `P_attended`; other rows × 1.0; `w_cls` is not a parameter | 02 §5.4 |
| FUSE-4 | Residual uses the **ungated** `P^{t−1}`: with the weight and bias of `W_out` set to 0, `P^t = LN(P^{t−1})` exactly, whatever the gate does | 02 §5.4, [DECISION D-02] |
| STAGE-1 | `L^t = F^q (P^t)ᵀ` exactly (no scale) | 02 §5.5 |
| STAGE-2 | One EPPM stage has 79,395 parameters; stages do not share parameter tensors | 01 §2.4, 01 §4 |

### 3.5 `tests/test_adrm_loss.py` (G1)

| ID | Check | Source |
| :--- | :--- | :--- |
| ADRM-1 | `w_gate` shape `[B_q, T]`, rows sum to 1 ± 1e-6 | 02 §6 |
| ADRM-2 | `W_g` weight shape `[T, 128]`, no bias | 02 §6 |
| ADRM-3 | `L_final = Σ_t w_gate[t] · L^t` to 1e-6 | 02 §6 |
| ADRM-4 | Gradients reach every `L^t` and `W_g` | 02 §6 |
| LOSS-1 | `L_seg` equals `F.cross_entropy(L_final, Y_q)` **without** `weight` | 02 §7, [PAPER Eq.27] |
| LOSS-2 | `L_total = L_seg + 1.0 · L_GMMN` | 02 §7 |
| LOSS-3 | No loss term uses stage logits `L^1..L^{T−1}` directly | 02 §7 |

### 3.6 `tests/test_ablation_switches.py` (G1)

| ID | Check | Source |
| :--- | :--- | :--- |
| ABL-1 | Each row of the D-17 table runs forward/backward and produces the prescribed prediction tensor | 01 §3, [DECISION D-17] |
| ABL-2 | `num_stages` ∈ {1..6} gives `w_gate` of width `num_stages` | 01 §3, [PAPER Tab.5] |
| ABL-3 | `modality=image` and `modality=audio` raise `NotImplementedError` | 03 §2.2 |

### 3.7 `tests/test_eval_metric.py` (G1)

| ID | Check | Source |
| :--- | :--- | :--- |
| EVAL-1 | The project's metric equals VIP-Seg's `evaluate_metric` on hand-made predictions over several episodes | 04 §6.2 |
| EVAL-2 | A crafted two-episode case where accumulated IoU ≠ mean of per-episode IoU; the project returns the accumulated value | 04 §6.2, [DECISION D-08] |
| EVAL-3 | Background (local label 0) is excluded from the mean | 04 §6.2 |

### 3.8 `tests/test_encoder.py` (G2, marker `cuda`)

| ID | Check | Source |
| :--- | :--- | :--- |
| ENC-1 | Encoder parameter count 2.37M ± 0.01M; feature head 204,260 | 01 §4 |
| ENC-2 | Output `[B, 2048, 128]`, all entries ≥ 0 | 02 §2 |
| ENC-3 | **Block independence:** in `eval()` mode, features of support block 0 do not change when other blocks in the batch change | 02 §2 |
| ENC-4 | **Query independence:** same property across queries | 02 §2 |
| ENC-5 | Input uses 9 channels; passing 3 channels raises | 01 §2.1 |

### 3.9 `tests/test_episode.py` (G3, marker `clip`)

Fixture: one synthetic episode with exactly the loader contract (04 §4.3), real CLIP embeddings for S3DIS class names.

| ID | Check | Source |
| :--- | :--- | :--- |
| EP-1 | Forward returns `L_final [N, 2048, N+1]`, `P_point` and `P_modal` `[N+1, 128]`, `w_gate [N, 4]` | 02 §10 |
| EP-2 | `L_total` finite; every trainable parameter has a finite, non-`None` gradient | 02 §7 |
| EP-3 | One AdamW step (lr 1e-3, wd 0.1) changes parameters and keeps them finite | [PAPER §4.1] |
| EP-4 | `eval()` forward twice → identical logits | [DECISION D-06] |
| EP-5 | CLIP is loaded once for two episodes (load counter = 1) | 03 §2.1 |
| EP-6 | Prompt strings match 03 §2.1 for a known `sampled_classes` | 03 §2.1 |

### 3.10 Data gate (G4, marker `data`)

| ID | Check | Source |
| :--- | :--- | :--- |
| DATA-1 | `S3DISDataset(cvfold=0, data_path)` finds `meta/s3dis_classnames.txt` in the documented layout | 04 §2.2 |
| DATA-2 | `MyDataset` with the 04 §4.1 arguments returns tensors of 04 §4.3; support masks ∈ {0, 1}; 9 channels; `XYZ` ∈ [0, 1] | 04 §4 |
| DATA-3 | Training classes of fold 0 exclude the S0 test classes and `clutter` | 04 §3 |
| DATA-4 | `MyTestDataset(mode='test')` for S0, 2-way 1-shot yields 1,500 episodes | 04 §6.1 |
| DATA-5 | `python train.py --dataset s3dis --cvfold 0 --n_way 2 --k_shot 1 --modality text --dry_run true` loads 4 real episodes, runs one optimiser step and exits 0; the log names the data path, CLIP variant and loader arguments | 04 §4–5 |
| DATA-6 | `python eval.py --dry_run true ...` evaluates 5 fixed test episodes with the 04 §6.2 metric and refuses to run without a loadable checkpoint | 04 §6 |

---

## 4. Pipeline sanity check with VIP-Seg (before G5)

Restore VIP-Seg's reference model and run scripts (00 §5.1), then evaluate its released checkpoint `log_s3dis_VIPSeg/log_S0_N2_K1_0.722026/checkpoint.pt` with this repository's data and evaluation code. Reference: mean IoU 0.722026 on 1,500 test episodes [VIPSEG log_s3dis_VIPSeg/log_S0_N2_K1_0.722026/log_vipseg_eval.txt:1-9]. The test episodes are sampled when the `.h5` cache is first built, so a local run uses different episodes and will not match exactly; no tolerance has been measured yet. Record the value; a gap of several points indicates that the data layout, loader arguments or metric differ from VIP-Seg, in which case CascadeProto numbers would not be comparable either.

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
| Encoder batching | blocks and queries independent | ENC-3..4, XATT-6 | C3, C4, M1 |
| LMA / GMMN | squared MMD, joint fg set, deterministic eval | LMA-1..3, MMD-1..7 | H5, M3 |
| EPPM | gate, channel attention, diffusion, fusion as in 02 §5 | GATE-*, XATT-*, DIFF-*, FUSE-*, STAGE-* | H3, H4, M2 |
| Routing and loss | simplex routing, unweighted CE | ADRM-1..4, LOSS-1..3 | H2 |
| Ablations | Table 4–5 switches exist and behave | ABL-1..3 | H6 (modality errors) |
| Evaluation | accumulated IoU on fixed episodes | EVAL-1..3, DATA-4, DATA-6, §4 | H8 |
| Training schedule | batch 4, 480/800 episodes per epoch, StepLR 10 epochs | DATA-5 plus log review | H7, M7 |
| Reproduction | gaps to Tables 2–5 reported | §5 | — |

Audit finding IDs refer to [docs/research/paper_vs_repo_audit.md](../research/paper_vs_repo_audit.md).
