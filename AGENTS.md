# AGENTS.md: Operating Rules for Coding Agents

Rules for autonomous coding agents (Claude Code, Cursor, Copilot, Devin, Aider, …) working in this repository. Humans should start with [README.md](README.md).

---

## 1. Project context

* **Mission:** an **unofficial** re-implementation of *CascadeProto: Cascaded Cross-Modal Prototype Purification via Entropy-Aware Learning for Few-Shot 3D Point Cloud Segmentation* (Wang et al.), aiming to reproduce its Tables 2–5. The authors' code is not released yet.
* **Base code:** the official VIP-Seg repository `changshuowang/VIP-Seg_NeurIPS2025`, pinned at commit `28aedc5093c0d386d526864c49505ae6921b1600`.
* **Target hardware:** one NVIDIA GPU; the paper used an RTX 5090.
* **Modality priority:** text first; image and audio are deferred and must raise until implemented.
* **Status (2026-09-18):** the docs in `docs/spec/` were rewritten against the paper. Phase 8 restored the inherited VIP-Seg files, removed the encoder fallbacks and added gate G0. Phase 9 put `train.py` and `eval.py` on real episodes through `pipeline/`. Phase 10 rewrote the feature extractor, point prototypes and `CascadeProto` (Table 4 baseline only; see docs/CHANGELOG.md). The rest of `models/`, `loss/` and `tests/` predates the rewrite and does **not** follow it yet; [docs/research/paper_vs_repo_audit.md](docs/research/paper_vs_repo_audit.md) lists the known gaps.

---

## 2. Sources of truth

Read [docs/spec/00_SOURCES_AND_DECISIONS.md](docs/spec/00_SOURCES_AND_DECISIONS.md) before any change. In short:

1. **Paper** (L1) beats **pinned VIP-Seg code** (L2) beats the **decision log** D-01…D-17 (L3).
2. Specs `01`–`05` restate L1–L3 with a source tag on every normative line: `[PAPER …]`, `[VIPSEG path:line]`, `[DECISION D-nn]`.
3. Code, tests, this file and the README are **not** sources. When code and spec disagree, the spec wins; when a spec line has no tag, treat it as unverified.
4. If the paper is ambiguous and no decision covers the case, **stop and ask the maintainer**. Record the answer as a new decision in `00` before writing code.

### 2.1 Task routing

| Task | Files | Read first |
| :--- | :--- | :--- |
| Encoder, feature head, model assembly | `models/vipseg_backbone.py` (`models/encoder.py` is read-only), `models/cascadeproto.py` | [01 §2.1](docs/spec/01_ARCHITECTURE_SPEC.md), [02 §2](docs/spec/02_TENSOR_MATH_SPEC.md) |
| Point prototypes | `models/prototypes.py` | [02 §3](docs/spec/02_TENSOR_MATH_SPEC.md) |
| Modality front-ends, LMA, generator | `models/lma.py` | [03](docs/spec/03_MULTIMODAL_SPEC.md), [02 §4](docs/spec/02_TENSOR_MATH_SPEC.md) |
| GMMN / MMD loss | `loss/gmmn_loss.py` | [03 §4](docs/spec/03_MULTIMODAL_SPEC.md), [02 §4.3–4.4](docs/spec/02_TENSOR_MATH_SPEC.md) |
| EPPM (gate, cross-attention, diffusion, fusion) | `models/eppm.py` | [02 §5](docs/spec/02_TENSOR_MATH_SPEC.md), [01 §2.4](docs/spec/01_ARCHITECTURE_SPEC.md), decisions D-01, D-02, D-11, D-14, D-16 |
| ADRM, segmentation loss | `models/adrm.py`, `loss/segmentation_loss.py` | [02 §6–7](docs/spec/02_TENSOR_MATH_SPEC.md) |
| Ablation switches | `models/cascadeproto.py`, CLI | [01 §3](docs/spec/01_ARCHITECTURE_SPEC.md), D-17 |
| Data, splits, episodes, schedule, evaluation | `pipeline/`, `train.py`, `eval.py`, `preprocess/prepare_s3dis.py`, `dataloaders/` (read-only) | [04](docs/spec/04_DATA_AND_EPISODES.md) |
| Tests | `tests/` | [05](docs/spec/05_VERIFICATION_PLAN.md) |

---

## 3. Invariants

Values are defined in the specs; this list is a reminder, not a source. If a value here disagrees with a spec, the spec is right and this file must be fixed.

| Invariant | Value | Spec |
| :--- | :--- | :--- |
| Points per block, input channels | 2048, `xyzrgbXYZ` (9) | 02 §0, 04 §4.1 |
| Feature dim D, projection d, cascade depth T | 128, 72, 4 | 02 §0 |
| Support encoding | each block independently, `[N·K, 2048, 9]` | 02 §2 |
| Support masks | binary {0, 1}, prototypes pooled per way | 02 §3 |
| Encoder | VIP-Seg with `mamba_ssm`; no fallback block | 01 §2.1 |
| Cross-attention | channel correlation `A ∈ [B_q, N+1, K, D, D]`, shared φ = `Conv1d(64→72)`, scale √72 | 02 §5.2, D-01 |
| Gate | per-channel Shannon entropy on `P^{t−1}`, ε = 10⁻⁸, θ₀ = 0.5 per stage | 02 §5.1, D-02 |
| Diffusion | τ = 0.5, α = 0.5 | 02 §5.3 |
| Class weights `w_cls = [0.8, 1, …, 1]` | inside EPPM only, **never** in the loss | 02 §5.4, 02 §7 |
| Stage logits | `F^q (P^t)ᵀ`, no temperature | 02 §5.5, D-10 |
| MMD | **squared**, σ ∈ {2, 5, 10, 20, 40, 80}, weights bg 0.1 / fg 1.0, fg rows as one set | 02 §4.3–4.4, D-04 |
| Losses | `L_total = CE(L_final, Y_q) + 1.0 · L_GMMN`, CE unweighted | 02 §7 |
| Noise at evaluation | z = 0 | D-06 |
| Optimiser | AdamW lr 1e-3, wd 0.1, StepLR ×0.5 every 10 epochs | 04 §5 |
| Schedule | batch 4 episodes; S3DIS 50 epochs × 480 episodes; ScanNet 30 × 800 | 04 §5, D-12 |
| Splits | class-based folds of the inherited loader | 04 §3, D-07 |
| Metric | TP/FP/FN accumulated per class over 100 fixed episodes per class combination, background excluded | 04 §6, D-08 |

---

## 4. Guardrails

1. **No pre-trained point-cloud weights.** Only frozen CLIP (and, later, Whisper) weights may be loaded (01 §2.1).
2. **Inherited files are read-only.** `dataloaders/{loader,s3dis,scannet}.py`, `preprocess/{collect_s3dis_data,collect_scannet_data,room2blocks}.py` and `utils/{checkpoint_util,cuda_util,logger}.py` must stay byte-identical to the pinned VIP-Seg commit. Change behaviour through arguments or wrappers. The same holds for `models/{encoder,mamba_block,model_utils,vipseg,vipseg_learner}.py`, `runs/*.py` and `main.py`; test ENV-3 checks all of them (00 §5.2).
3. **Restore, don't rewrite, VIP-Seg's evaluation.** The metric in `runs/training_free.py` and the patterns in `runs/{training,evaluate}.py` come from the pinned commit (00 §5.1). They were restored byte-identical in phase 8; call or wrap them, never copy-edit them.
4. **Shape comments.** Every tensor operation carries an inline shape comment, e.g. `# [B_q, N+1, 128]`.
5. **No silent maths drift.** Formulas must match 02 exactly. Changing an interpretation means editing the decision in `00` first, with evidence, then the spec, then the code.
6. **No shape guessing.** Never infer layouts from `shape[i] in (3, 6, 9)`, never infer N from mask values, never `reshape`/`view` across batch, class or shot axes to make a product fit; use explicit indices or `einsum`. If shapes do not fit, stop: the spec or the input is wrong. Do not invent a projection layer to force a fit.
7. **No silent fallbacks.** Missing `mamba_ssm`, `pointnet2_ops` or CLIP, an unknown modality, or an unloadable checkpoint must raise.
8. **No synthetic data outside `tests/`.** `train.py` and `eval.py` only read real episodes through the inherited loader; `--dry_run` means "real data, few steps".
9. **Tags in docs.** Any new normative line in `docs/spec/` carries a source tag (00 §2.3). Unsourced statements are deleted.
10. **Security.** Never disable TLS verification (`ssl._create_default_https_context`). Never add global monkey-patches. The inherited `utils/checkpoint_util.py` already replaces `torch.load` with `weights_only=False` on import [VIPSEG utils/checkpoint_util.py:9-16]; import it only where a VIP-Seg checkpoint must be loaded (05 §4), and only load checkpoints from trusted sources.

---

## 5. Repository layout

```text
CascadeProto/
├── AGENTS.md, README.md
├── docs/
│   ├── spec/00_SOURCES_AND_DECISIONS.md   source hierarchy + decision log (read first)
│   ├── spec/01_ARCHITECTURE_SPEC.md       modules, wiring, switches, parameter budget
│   ├── spec/02_TENSOR_MATH_SPEC.md        all formulas and shapes
│   ├── spec/03_MULTIMODAL_SPEC.md         modality front-ends, LMA, GMMN rules
│   ├── spec/04_DATA_AND_EPISODES.md       data layout, splits, episodes, schedule, metric
│   ├── spec/05_VERIFICATION_PLAN.md       tests, gates, acceptance
│   └── research/paper_vs_repo_audit.md    audit of the pre-rewrite code (2026-09-17)
├── dataloaders/            inherited, read-only
├── preprocess/             inherited scripts (read-only) + download_and_prepare_s3dis.py (local)
├── utils/                  inherited, read-only
├── models/
│   ├── encoder.py, mamba_block.py, model_utils.py   inherited VIP-Seg encoder
│   ├── vipseg_backbone.py  encoder + feature head + point prototypes
│   ├── lma.py              modality adapters and generator
│   ├── eppm.py             EPPM stage and cascade
│   ├── adrm.py             dynamic routing
│   └── cascadeproto.py     end-to-end model
├── loss/                   gmmn_loss.py, segmentation_loss.py
├── pointnet2_ops_lib/      vendored CUDA ops
├── runs/, main.py, scripts/  VIP-Seg reference code (inherited, read-only)
├── tests/                  see 05
├── pipeline/               episodes over the inherited loader, model contract, evaluation
├── train.py, eval.py
└── requirements.txt
```

---

## 6. Verification protocol

Run the gates of [05 §2](docs/spec/05_VERIFICATION_PLAN.md) in order:

```bash
pytest tests/test_environment.py -v                 # G0 environment
pytest -m "not cuda and not clip and not data" -v   # G1 unit, CPU
pytest -m cuda -v                                   # G2 encoder on GPU
pytest -m clip -v                                   # G3 episode with real CLIP
pytest -m data -v                                   # G4 real data
python train.py --dataset s3dis --data_path datasets/S3DIS/blocks_bs1_s1 --cvfold 0 --n_way 2 --k_shot 1 --dry_run true
```

Before any reproduction run, evaluate VIP-Seg's released S0 2-way 1-shot checkpoint with this repository's data and metric; the result must be close to VIP-Seg's logged 0.722; a gap of several points means the pipeline differs (05 §4).

The test files and markers named above are the target of 05; until the tests are rewritten, the existing suite passing says nothing about paper fidelity.

---

## 7. Troubleshooting

| Symptom | Likely cause | Where to look |
| :--- | :--- | :--- |
| `ModuleNotFoundError: mamba_ssm` | Mamba not installed for this PyTorch/CUDA | Install order in `requirements.txt`; pinned versions in 00 §5.2 |
| `FileNotFoundError: …/meta/s3dis_classnames.txt` | `--data_path` not inside the documented layout | 04 §2.2 |
| Class-2 prototype is all zeros | masks compared with `== k` instead of per way | 02 §3 |
| Entropy is NaN | missing probability clamp | 02 §9 |
| Different logits for the same input in `eval()` | noise z sampled at evaluation | 02 §4.2, D-06 |
| mIoU not comparable with the paper | per-episode averaging or random test episodes | 04 §6 |
| Out of memory in EPPM | point–point attention instead of channel correlation | 02 §5.2 |
