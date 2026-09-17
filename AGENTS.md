# AGENTS.md: Machine Operating Protocols & Technical Invariants

This document serves as the **"README for machines"** (autonomous AI coding agents including Cursor, Claude Code, Antigravity, Copilot, Devin, and Aider). It defines operational boundaries, non-negotiable architectural invariants, task-to-spec routing protocols, and sequential verification workflows for the **CascadeProto** repository.

---

## 1. Project Context & Mission

* **Mission:** Re-implement and reproduce the CascadeProto framework for few-shot 3D point cloud semantic segmentation without backbone pre-training.
* **Base Codebase:** Forked and modified from `changshuowang/VIP-Seg_NeurIPS2025`.
* **Hardware Target:** Single NVIDIA GPU (verified on NVIDIA RTX 5090).
* **Modality Priority:** Implement the text-guided pipeline first (which achieved 86.53% 2-way 1-shot mIoU on S3DIS), while maintaining polymorphic interfaces for image and audio adapters.

---

## 2. Task-to-Spec Citation & Routing Matrix

Before modifying, implementing, or debugging any module, agents **must** consult and cite the corresponding ground-truth specification document:

| Technical Task / Responsibility | Target Implementation File(s) | Mandatory Ground-Truth Specification | Key Invariants & Verification Focus |
| :--- | :--- | :--- | :--- |
| **Backbone & Network Assembly** | `models/vipseg_backbone.py`<br>`models/cascadeproto.py` | [01_ARCHITECTURE_SPEC.md](docs/spec/01_ARCHITECTURE_SPEC.md) | Shared VIP-Seg encoder ($D=128$), $T=4$ EPPM sequential cascade, residual prototype links $P_{t-1} \to P_t$. |
| **Mathematical Bounds & Shapes** | All modules & tensor operations | [02_TENSOR_MATH_SPEC.md](docs/spec/02_TENSOR_MATH_SPEC.md) | Subspace dimension $d=72$, Shannon entropy bounds $[0, \log 2]$, clamping $[10^{-7}, 1-10^{-7}]$, probability simplex. |
| **Multimodal Adapters & GMMN** | `models/lma.py`<br>`loss/gmmn_loss.py` | [03_MULTIMODAL_SPEC.md](docs/spec/03_MULTIMODAL_SPEC.md) | CLIP projection $512 \to 128$, Gaussian noise $z$, multi-scale RBF $\sigma \in \{2..80\}$, decoupled weights $0.1$ bg / $1.0$ fg. |
| **Purification Module (EPPM)** | `models/eppm.py` | [01_ARCHITECTURE_SPEC.md](docs/spec/01_ARCHITECTURE_SPEC.md)<br>[02_TENSOR_MATH_SPEC.md](docs/spec/02_TENSOR_MATH_SPEC.md) | Shannon gating $\theta=0.5$, query-support cross-attention affinity $A_{qs}$, diffusion threshold $\tau=0.5$, blend $\alpha=0.5$. |
| **Dynamic Routing (ADRM)** | `models/adrm.py` | [01_ARCHITECTURE_SPEC.md](docs/spec/01_ARCHITECTURE_SPEC.md)<br>[02_TENSOR_MATH_SPEC.md](docs/spec/02_TENSOR_MATH_SPEC.md) | Query global average pooling $v_q$, softmax gating simplex $\sum_{t=1}^4 w_{gate}^{(t)} = 1.0$, all 4 stages receive gradients. |
| **Dataloader & Episodic Sampling** | `dataloaders/s3dis.py`<br>`dataloaders/scannet.py` | [04_DATA_AND_EPISODES.md](docs/spec/04_DATA_AND_EPISODES.md) | $N$-way $K$-shot sampling, $N_p=2048$ points/block, minimum $\ge 50$ points/class threshold, local label remapping $Y_{q, i}$. |
| **Unit Verification & Smoke Tests** | `tests/*.py`<br>`train.py` | [05_VERIFICATION_PLAN.md](docs/spec/05_VERIFICATION_PLAN.md) | 3-phase testing pipeline, synthetic tensor fixtures, zero NaNs across all trainable parameters, 8-component acceptance matrix. |

---

## 3. Hardcoded Architectural Invariants & Constants

Every agent modifying or generating code must strictly adhere to the following numerical constants:

* **Point Cloud Backbone:** Shared VIP-Seg encoder producing feature dimension $D = 128$ for both support and query branches.
* **Input Resolution:** Block-based point clouds with exactly $N_s = 2048$ support points and $N_q = 2048$ query points.
* **Cascade Depth:** Exactly $T = 4$ sequential Entropy-aware Prototype Purification Modules (EPPM).
* **Cross-Attention Subspace:** Projected feature dimension $d = 72$ via $1 \times 1$ convolutions.
* **Information-Theoretic Gating:** Per-channel Shannon entropy calculation normalized with $\epsilon = 10^{-8}$, modulated by a learnable scalar threshold $\theta$ initialized to $0.5$.
* **Prototype Diffusion:** Channel activation threshold $\tau = 0.5$ and common-unique blending factor $\alpha = 0.5$.
* **Class Weighting Vector:** Fixed vector $w_{cls} = [0.8, 1.0, \dots, 1.0] \in \mathbb{R}^{N+1}$, where background index $0$ receives weight $0.8$ and all $N$ foreground categories receive $1.0$.
* **MMD RBF Kernel Bandwidths:** Multi-scale kernel $\sigma \in \{2, 5, 10, 20, 40, 80\}$.
* **Decoupled Distribution Alignment Weights:** Factor $0.1$ for background MMD and $1.0$ for foreground MMD.
* **Total Loss Balancing:** $\lambda = 1.0$ weighting $\mathcal{L}_{GMMN}$ against $\mathcal{L}_{seg}$.
* **Optimization Hyperparameters:** AdamW optimizer, initial learning rate $\eta = 10^{-3}$, weight decay $0.1$, StepLR scheduler halving the learning rate every 10 epochs.
* **Training Durations:** 50 epochs on S3DIS, 30 epochs on ScanNet, with batch size of 4 episodes.

---

## 4. Strict Development Guardrails

1. **Zero Pre-training Dependency:** Do not load external pre-trained checkpoints for the 3D point cloud encoder; the VIP-Seg backbone is trained from scratch within the episodic meta-learning scheme. Pre-trained weights are restricted solely to frozen CLIP encoders.
2. **Data Pipeline Immutability:** Never rewrite the episodic sampling logic, block generation, or evaluation routines inherited from the base VIP-Seg repository. All episodes must evaluate standard 2-way and 3-way settings under 1-shot and 5-shot splits.
3. **Mandatory Shape Annotations:** Every PyTorch tensor transformation must be accompanied by an inline comment explicitly tracking tensor rank and shapes (e.g., `# [B, N+1, D]`).
4. **No Silent Mathematical Drift:** Never replace explicit paper formulations (e.g., substituting Shannon entropy with simple variance, or replacing MMD with Cosine Similarity). All math must match [02_TENSOR_MATH_SPEC.md](docs/spec/02_TENSOR_MATH_SPEC.md) identically.
5. **No Blind Hallucinations:** When tensor dimensions conflict during broadcasting or multi-head attention, agents must resolve dimensions by projecting with explicit linear layers ($1 \times 1$ conv) rather than arbitrary squeezing, flattening, or truncation.

---

## 5. Repository Layout & Module Ownership

```text
CascadeProto/
├── AGENTS.md                       # Machine operating instructions & routing matrix
├── README.md                       # Human onboarding, benchmarks, and run commands
├── docs/
│   └── spec/                       # Ground-truth technical documentation
│       ├── 01_ARCHITECTURE_SPEC.md # Structural and layer-level specs
│       ├── 02_TENSOR_MATH_SPEC.md  # Exact mathematical formulas and loss functions
│       ├── 03_MULTIMODAL_SPEC.md   # CLIP text/image/audio adapter pipelines
│       ├── 04_DATA_AND_EPISODES.md # S3DIS/ScanNet splits and episode configurations
│       └── 05_VERIFICATION_PLAN.md # Test suite and pass/fail criteria
├── models/
│   ├── vipseg_backbone.py          # Inherited VIP-Seg encoder (D=128)
│   ├── lma.py                      # Learnable Modality Adapters & GMMN generator
│   ├── eppm.py                     # Entropy gating, cross-attention, diffusion
│   ├── adrm.py                     # Dynamic routing & multi-stage logit aggregation
│   └── cascadeproto.py             # End-to-end network assembly
├── loss/
│   ├── gmmn_loss.py                # Decoupled multi-scale MMD calculation
│   └── segmentation_loss.py        # Cross-entropy loss on routed logits
├── dataloaders/
│   ├── s3dis.py                    # S3DIS episodic dataloader (Areas 1-4, 6 train; Area 5 test)
│   └── scannet.py                  # ScanNet episodic dataloader
├── tests/                          # Modular unit testing suite
│   ├── test_lma.py                 # LMA projection & decoupled GMMN test
│   ├── test_eppm.py                # EPPM gating, attention, diffusion test
│   ├── test_adrm.py                # ADRM routing simplex & gradient flow test
│   └── test_smoke_episode.py       # End-to-end synthetic forward/backward smoke test
├── train.py                        # Episodic training loop
└── eval.py                         # 600-episode evaluation benchmark script
```

---

## 6. Sequential Verification Protocol

Agents must run and pass tests in this exact sequential order before initiating training or evaluation:

### Phase 1: Isolated Module Unit Tests
```bash
pytest tests/test_lma.py -v
pytest tests/test_eppm.py -v
pytest tests/test_adrm.py -v
```

### Phase 2: End-to-End Synthetic Smoke Test (1 batch, forward + backward)
```bash
pytest tests/test_smoke_episode.py -v
```

### Phase 3: Dry-Run Episode Execution on Real S3DIS Dataloader
```bash
python train.py --dataset s3dis --cvfold 0 --n_way 2 --k_shot 1 --modality text --dry_run True
```

---

## 7. Failure Modes & Troubleshooting Playbook

* **Numerical Instability in Entropy Gating:** If negative log computations cause `NaN`, enforce clamping on probability vectors: $p = \text{clamp}(\sigma(x), 10^{-7}, 1 - 10^{-7})$ before calculating $-p \log(p + \epsilon) - (1-p) \log(1-p + \epsilon)$. Consult [02_TENSOR_MATH_SPEC.md](docs/spec/02_TENSOR_MATH_SPEC.md#41-sub-module-1-entropy-aware-information-gating) for formal bounds.
* **CUDA Out of Memory (OOM) in Kernel Distance:** The multi-scale Gaussian kernel in GMMN creates pairwise distance matrices of size $|P| \times |Q|$. In episodic training, prototype sets are small ($(N+1)$ prototypes), but if intermediate point clouds are passed accidentally, assert $|P| \le 16$ before kernel evaluation to avoid memory blow-ups. Consult [03_MULTIMODAL_SPEC.md](docs/spec/03_MULTIMODAL_SPEC.md#32-decoupled-gmmn-loss-specification).
* **Routing Collapse in ADRM:** If ADRM gate weights $w_{gate}$ concentrate exclusively on stage 1 ($w_{gate}^{(1)} \to 1.0$), inspect learning rates for $W_g$; the linear projection must remain synchronized with the backbone optimizer schedule. Consult [01_ARCHITECTURE_SPEC.md](docs/spec/01_ARCHITECTURE_SPEC.md#4-attention-based-dynamic-routing-module-adrm).
