# 02_TENSOR_MATH_SPEC: Mathematical Formulations, Loss Objectives & Numerical Stability

* **Reference Paper:** *CascadeProto: Cascaded Cross-Modal Prototype Purification via Entropy-Aware Learning for Few-Shot 3D Point Cloud Segmentation* (Wang et al., ECCV 2026).
* **Reference Repository Link:** [https://github.com/changshuowang/CascadeProto](https://github.com/changshuowang/CascadeProto).
* **Base Encoder Lineage:** VIP-Seg ($D = 128$, no pre-training).

---

## 1. Episodic Formulation & Feature Extraction

Few-shot 3D point cloud semantic segmentation operates under the episodic learning paradigm. An episode comprises an $N$-way $K$-shot support set $S$ and a query set $Q$:
* **Support Set:** $S = \{(X_s^i, Y_s^i)\}_{i=1}^{N \times K}$, where $X_s^i \in \mathbb{R}^{N_s \times 3}$ and $Y_s^i \in \{0, 1\}^{N_s}$ represents the binary mask for target category $k \in \{1, \dots, N\}$.
* **Query Set:** $Q = \{X_q, Y_q\}$, where $X_q \in \mathbb{R}^{N_q \times 3}$ and $Y_q \in \{0, \dots, N\}^{N_q}$ provides point-level category labels across $N$ target classes plus background ($0$).
* **Point Budget:** Exactly $N_s = 2048$ and $N_q = 2048$ points per block.

The shared VIP-Seg encoder $f_{enc}$ maps raw point sets to dense geometric features:
$$F_s = f_{enc}(X_s) \in \mathbb{R}^{N_s \times D}, \quad F_q = f_{enc}(X_q) \in \mathbb{R}^{N_q \times D}, \quad D = 128$$

---

## 2. Geometric Prototype Extraction

Support foreground and background prototypes are extracted via masked average pooling:
$$P_{fg}^{(k)} = \frac{1}{|M_{fg}^{(k)}|} \sum_{i \in M_{fg}^{(k)}} F_{s, i}, \quad k \in \{1, \dots, N\}$$
$$P_{bg} = \frac{1}{|M_{bg}|} \sum_{i \in M_{bg}} F_{s, i}$$
where mask index sets are defined as:
$$M_{fg}^{(k)} = \{i \mid Y_{s, i} = 1 \text{ and } \text{class}(i) = k\}, \quad M_{bg} = \{i \mid Y_{s, i} = 0\}$$

The initial point prototype matrix $P_{point} \in \mathbb{R}^{(N+1) \times D}$ concatenates background and foreground prototypes:
$$P_{point} = \begin{bmatrix} P_{bg} \\ P_{fg}^{(1)} \\ \vdots \\ P_{fg}^{(N)} \end{bmatrix} \in \mathbb{R}^{(N+1) \times D}$$

---

## 3. Decoupled GMMN Alignment & Multi-Scale MMD Loss

### 3.1 Adapter Projection & Generation
For a given modality $m \in \{\text{text}, \text{image}, \text{audio}\}$, the normalized CLIP representation $E_{CLIP}^{(m)} \in \mathbb{R}^{(N+1) \times 512}$ is projected via a 2-layer MLP adapter with LayerNorm and Dropout ($p=0.1$):
$$E_{adapted}^{(m)} = W_2 \cdot \text{ReLU}(\text{LayerNorm}(W_1 \cdot E_{CLIP}^{(m)} + b_1)) + b_2 \in \mathbb{R}^{(N+1) \times D}$$
where $W_1 \in \mathbb{R}^{D \times 512}$, $b_1 \in \mathbb{R}^D$, $W_2 \in \mathbb{R}^{D \times D}$, and $b_2 \in \mathbb{R}^D$ ($D = 128$).

A 3-layer MLP generator $G$ receives the adapted embedding concatenated with Gaussian noise $z \sim \mathcal{N}(0, I_D) \in \mathbb{R}^{(N+1) \times D}$:
$$P_{modal} = G([E_{adapted}^{(m)}; z]) \in \mathbb{R}^{(N+1) \times D}$$

### 3.2 Multi-Scale Gaussian RBF Kernel
The Maximum Mean Discrepancy (MMD) in Reproducing Kernel Hilbert Space $\mathcal{H}$ between distribution samples $P \subset \mathbb{R}^D$ and $Q \subset \mathbb{R}^D$ is defined as:
$$\text{MMD}^2(P, Q) = \frac{1}{|P|^2} \sum_{x \in P} \sum_{x' \in P} k(x, x') + \frac{1}{|Q|^2} \sum_{y \in Q} \sum_{y' \in Q} k(y, y') - \frac{2}{|P||Q|} \sum_{x \in P} \sum_{y \in Q} k(x, y)$$
where the kernel function $k(x, y)$ combines six Gaussian bandwidth scales:
$$k(x, y) = \sum_{\sigma \in \{2, 5, 10, 20, 40, 80\}} \exp\left(-\frac{\|x - y\|_2^2}{2\sigma^2}\right)$$

### 3.3 Decoupled Objective
Because background features are heterogeneous across rooms and clutter, foreground and background alignments are decoupled with asymmetric weights:
$$\mathcal{L}_{GMMN} = 0.1 \cdot \text{MMD}(P_{modal}^{bg}, P_{point}^{bg}) + 1.0 \cdot \text{MMD}(P_{modal}^{fg}, P_{point}^{fg})$$
where $P_{modal}^{bg} = P_{modal}[0, :]$ and $P_{modal}^{fg} = P_{modal}[1:, :]$.

The enriched initial multi-modal prototype is computed via direct element-wise addition:
$$P_0 = P_{point} + P_{modal} \in \mathbb{R}^{(N+1) \times D}$$

---

## 4. Entropy-Aware Prototype Purification Module (EPPM)

Each stage $t \in \{1, \dots, T\}$ (with cascade depth $T = 4$) executes four sequential transformations to refine $P_{t-1} \to P_t$:

### 4.1 Step 1: Information-Theoretic Gating
For each prototype channel index $i \in \{1, \dots, D\}$ of feature vector $x \in \mathbb{R}^D$:
1. **Normalized Probability:**
   $$p_i = \sigma(x_i) = \frac{1}{1 + e^{-x_i}}$$
2. **Shannon Entropy Formulation:**
   $$H_i = -p_i \log(p_i + \epsilon) - (1 - p_i) \log(1 - p_i + \epsilon)$$
   with numerical stability constant $\epsilon = 10^{-8}$.
3. **Entropy Soft-Gating:**
   $$g_i = \sigma(2(\theta - H_i))$$
   where $\theta \in \mathbb{R}$ is a learnable scalar threshold initialized to $0.5$.
4. **Channel Modulation:**
   $$x_{gated} = x \odot g$$

### 4.2 Step 2: Cross-Attention Refinement
1. Linear projections map query and support representations into a lower-dimensional subspace of dimension $d = 72$ via $1 \times 1$ convolutions:
   $$Q' = \varphi(F_q) \in \mathbb{R}^{N_q \times 72}, \quad S' = \varphi(F_s) \in \mathbb{R}^{N_s \times 72}$$
2. Pairwise cross-correlation matrix modeling dense point-to-point geometric correspondence between query and support sets:
   $$A_{qs} = \text{softmax}\left(\frac{Q' (S')^\top}{\sqrt{72}}\right) \in \mathbb{R}^{N_q \times N_s}$$
3. Support context propagation into the query coordinate frame:
   $$F_{qs} = A_{qs} F_s \in \mathbb{R}^{N_q \times D}$$
4. Cross-attention prototype refinement over query-aligned geometric representations:
   $$A_{proto} = \text{softmax}\left(\frac{\psi(P_{t-1}) \cdot \varphi(F_{qs})^\top}{\sqrt{72}}\right) \in \mathbb{R}^{(N+1) \times N_q}$$
   $$P_{cross} = A_{proto} \cdot F_{qs} \in \mathbb{R}^{(N+1) \times D}$$
   where $\psi, \varphi: \mathbb{R}^D \to \mathbb{R}^{72}$ denote linear projection operators ($1 \times 1$ convolutions) into the cross-attention subspace $d = 72$.
   *(Note: Alternatively, under direct support cross-attention: $A_{proto} = \text{softmax}\left(\frac{\psi(P_{t-1}) (S')^\top}{\sqrt{72}}\right) \in \mathbb{R}^{(N+1) \times N_s}$ and $P_{cross} = A_{proto} \cdot F_s \in \mathbb{R}^{(N+1) \times D}$)*.

### 4.3 Step 3: Prototype Diffusion
1. Spatial channel-mean activations across query and support point sets:
   $$q_{ch} = \sigma\left(\frac{1}{N_q} \sum_{j=1}^{N_q} F_{q, j}\right) \in \mathbb{R}^D, \quad s_{ch} = \sigma\left(\frac{1}{N_s} \sum_{j=1}^{N_s} F_{s, j}\right) \in \mathbb{R}^D$$
2. Indicator masks with activation threshold $\tau = 0.5$:
   $$m_q = \mathbb{I}[q_{ch} > 0.5] \in \{0, 1\}^D, \quad m_s = \mathbb{I}[s_{ch} > 0.5] \in \{0, 1\}^D$$
   $$m_{common} = m_q \odot m_s \in \{0, 1\}^D$$
3. Channel feature components:
   $$c_{common} = \frac{q_{ch} + s_{ch}}{2} \odot m_{common} \in \mathbb{R}^D$$
   $$c_{unique} = \frac{q_{ch} \odot (m_q - m_{common}) + s_{ch} \odot (m_s - m_{common})}{2} \in \mathbb{R}^D$$
4. Blended diffusion prototype with mixing parameter $\alpha = 0.5$:
   $$P_{diffuse} = 0.5 \cdot c_{common} + 0.5 \cdot c_{unique} \in \mathbb{R}^D$$
   $P_{diffuse}$ is broadcasted along the category dimension to span $\mathbb{R}^{(N+1) \times D}$.

### 4.4 Step 4: Adaptive Fusion & Class Recalibration
1. Dynamic weighting via a 2-layer MLP fusion network $f_{fusion}: \mathbb{R}^{2D} \to \mathbb{R}^2$:
   $$w = \text{softmax}(f_{fusion}([P_{cross}; P_{diffuse}])) \in \mathbb{R}^2$$
   $$P_{combined} = w_1 P_{cross} + w_2 P_{diffuse} \in \mathbb{R}^{(N+1) \times D}$$
2. Squeeze-and-Excitation (SE) channel recalibration:
   $$a = \sigma(W_2^{se} \cdot \text{ReLU}(W_1^{se} \cdot \text{AvgPool}(P_{combined}))) \in \mathbb{R}^D$$
   $$P_{attended} = P_{combined} \odot a \in \mathbb{R}^{(N+1) \times D}$$
3. Class-specific attenuation vector $w_{cls} = [0.8, 1.0, \dots, 1.0]^\top \in \mathbb{R}^{N+1}$:
   $$P_{weighted} = P_{attended} \odot w_{cls} \in \mathbb{R}^{(N+1) \times D}$$
4. Residual connection and normalization:
   $$P_t = \text{LayerNorm}(W_{out} \cdot \text{ReLU}(P_{weighted}) + P_{t-1}) \in \mathbb{R}^{(N+1) \times D}$$

---

## 5. Dynamic Routing & Aggregation (ADRM)

Intermediate logits at stage $t \in \{1, \dots, 4\}$ are evaluated via matrix multiplication:
$$L_t = F_q (P_t)^\top \in \mathbb{R}^{N_q \times (N+1)}$$

The routing network pools query feature statistics:
$$\bar{F}_q = \frac{1}{N_q} \sum_{j=1}^{N_q} F_{q, j} \in \mathbb{R}^D$$
Gating weights across all $T = 4$ stages are computed with parameter matrix $W_g \in \mathbb{R}^{4 \times D}$:
$$w_{gate} = \text{softmax}(W_g \bar{F}_q) \in \mathbb{R}^4$$
Final aggregated logits:
$$L_{final} = \sum_{t=1}^4 w_{gate}^{(t)} \cdot L_t \in \mathbb{R}^{N_q \times (N+1)}$$

---

## 6. Optimization Objectives

The total training objective balances cross-entropy segmentation loss and decoupled GMMN distribution matching loss with balancing scalar $\lambda = 1.0$:
$$\mathcal{L}_{total} = \mathcal{L}_{seg} + 1.0 \cdot \mathcal{L}_{GMMN}$$
where the per-point query segmentation loss is formulated as:
$$\mathcal{L}_{seg} = -\frac{1}{N_q} \sum_{i=1}^{N_q} \log \left( \frac{\exp(L_{final}[i, Y_{q, i}])}{\sum_{c=0}^N \exp(L_{final}[i, c])} \right)$$

---

## 7. Numerical Stability Guardrails

When implementing these formulations in PyTorch, coding agents must enforce the following safeguards:

```python
# 1. Clamping probability for Shannon Entropy computation
p = torch.sigmoid(x)  # [B, N+1, D]
p_clamped = torch.clamp(p, min=1e-7, max=1.0 - 1e-7)
entropy = -p_clamped * torch.log(p_clamped + 1e-8) - (1.0 - p_clamped) * torch.log(1.0 - p_clamped + 1e-8)

# 2. Pairwise squared Euclidean distance for RBF Kernel
# dist_sq = ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x y^T
def pairwise_sq_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # x: [M, D], y: [N, D]
    x_norm = (x ** 2).sum(dim=-1, keepdim=True)       # [M, 1]
    y_norm = (y ** 2).sum(dim=-1, keepdim=True).t()   # [1, N]
    dist_sq = x_norm + y_norm - 2.0 * torch.matmul(x, y.t())
    return torch.clamp(dist_sq, min=0.0)

# 3. Dynamic routing softmax normalization over stage dimension
w_gate = torch.softmax(torch.matmul(query_summary, W_g.t()), dim=-1)  # [B, 4]
```
