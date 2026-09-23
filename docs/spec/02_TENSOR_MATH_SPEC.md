# 02_TENSOR_MATH_SPEC: Mathematical Formulation, Losses & Numerical Rules

* **Ground truth:** the paper (L1) and the pinned VIP-Seg code (L2), as defined in [00_SOURCES_AND_DECISIONS.md](00_SOURCES_AND_DECISIONS.md). Every normative line carries a source tag.
* **Scope:** exact math and tensor shapes for one training/evaluation episode. Module wiring lives in [01_ARCHITECTURE_SPEC.md](01_ARCHITECTURE_SPEC.md); data and protocol live in [04_DATA_AND_EPISODES.md](04_DATA_AND_EPISODES.md).
* **Rewritten:** 2026-09-17 (Phase 1). Formulas from the pre-rewrite version that have no source (two-hop attention, weighted CE, `sqrt` MMD) were removed.

---

## 0. Notation and constants

| Symbol | Value | Meaning | Source |
| :--- | :--- | :--- | :--- |
| N | 2 or 3 | Ways (foreground classes per episode) | [PAPER §4.1] |
| K | 1 or 5 | Shots per class | [PAPER §4.1] |
| N_s = N_q | 2048 | Points per block | [PAPER §4.1] |
| C_in | 9 | Input channels `xyz, rgb, XYZ` | [PAPER §4.1 "RGB point clouds"] [VIPSEG scripts/vipseg_s3dis.sh] |
| D | 128 | Feature dimension | [PAPER Eq.2] |
| d | 72 | Cross-attention projection dimension | [PAPER Eq.13] [PAPER §4.1] |
| T | 4 | Cascade depth | [PAPER §3.5] [PAPER §4.1] |
| B_q | N (one query per way) | Queries per episode | [VIPSEG scripts/vipseg_s3dis.sh `N_QUESIES=1`] [VIPSEG dataloaders/loader.py:181-223] |
| ε | 10⁻⁸ | Entropy stability constant | [PAPER Eq.10] |
| θ₀ | 0.5 | Initial gate threshold (learnable, one per stage) | [PAPER Eq.11] [PAPER §3.5] |
| τ, α | 0.5, 0.5 | Diffusion threshold and blend | [PAPER Eq.18] |
| w_cls | [0.8, 1.0, …, 1.0] ∈ R^{N+1} | Class weights inside EPPM only | [PAPER §3.4] |
| σ set | {2, 5, 10, 20, 40, 80} | RBF bandwidths | [PAPER Eq.7] |
| λ | 1.0 | GMMN loss weight | [PAPER Eq.26] |

Class index 0 is background; indices 1…N are the episode's foreground classes in sampling order [PAPER Eq.3] [VIPSEG dataloaders/loader.py:85-88].

---

## 1. Episode tensors

| Tensor | Shape | Values | Source |
| :--- | :--- | :--- | :--- |
| Support points `X^s` | `[N, K, 2048, 9]` | per-block normalised | [PAPER §3.1] [VIPSEG dataloaders/loader.py:220] |
| Support masks `Y^s` | `[N, K, 2048]` | binary {0, 1}; 1 = the way's class | [PAPER §3.1 "binary foreground mask"] [VIPSEG dataloaders/loader.py:82-83] |
| Query points `X^q` | `[B_q, 2048, 9]` | per-block normalised | [VIPSEG dataloaders/loader.py:222] |
| Query labels `Y^q` | `[B_q, 2048]` | {0, …, N} | [VIPSEG dataloaders/loader.py:84-88] |
| CLIP embeddings `E_CLIP^(m)` | `[N+1, 512]` | L2-normalised, frozen | [PAPER §4.1 "from R^512"] [DECISION D-13] |

A training batch holds 4 independent episodes; every formula below is applied per episode and the losses are averaged over the batch [PAPER §4.1] [DECISION D-12].

---

## 2. Feature extraction (Eq.2)

Each support block and each query block is its own sample of the shared VIP-Seg encoder [PAPER Eq.2] [VIPSEG models/vipseg.py:79-97]:

$$F^s = f_{enc}(X^s) \in \mathbb{R}^{N \times K \times 2048 \times D}, \qquad F^q = f_{enc}(X^q) \in \mathbb{R}^{B_q \times 2048 \times D}$$

`f_enc` = VIP-Seg encoder → channel-wise L2 normalisation → `BN(900) → ReLU → Conv1d(900→196) → BN → ReLU → Conv1d(196→128) → BN → ReLU` [VIPSEG models/vipseg.py:34-53,85-97]. Consequence: `F ≥ 0` element-wise (see §5.3 and [DECISION D-14]).

Support blocks must never be concatenated into one point set before the encoder: FPS/kNN would mix blocks whose normalised coordinates overlap [VIPSEG models/vipseg.py:79].

**Batch coupling.** Being separate samples does not make blocks independent. The encoder standardises three intermediate tensors with one mean and one standard deviation taken over the **whole batch tensor** (all blocks, points and channels), in training and evaluation alike [VIPSEG models/encoder.py:281-283,413-415,582-584]. A block's features therefore depend on the other blocks of the same encoder call. The features equal VIP-Seg's only for VIP-Seg's batch composition: one call with the N·K support blocks of one episode and one call with that episode's queries [VIPSEG models/vipseg.py:79-89]. Never encode blocks of different episodes, or supports together with queries, in one call; `encode_episode` enforces this, and `train.py` / `eval.py` forward one episode at a time.

---

## 3. Point prototypes (Eq.3)

Masked average pooling on binary masks, per way [PAPER Eq.3] [VIPSEG models/vipseg.py:108-130]:

$$P_{fg}^{(k)} = \frac{1}{|\mathcal{M}_{fg}^{(k)}|} \sum_{(j,i) \in \mathcal{M}_{fg}^{(k)}} F^s_{k,j,i}, \quad \mathcal{M}_{fg}^{(k)} = \{(j,i) \mid Y^s_{k,j,i} = 1\}, \quad k = 1..N$$

$$P_{bg} = \frac{1}{|\mathcal{M}_{bg}|} \sum_{(n,j,i) \in \mathcal{M}_{bg}} F^s_{n,j,i}, \quad \mathcal{M}_{bg} = \{(n,j,i) \mid Y^s_{n,j,i} = 0\}$$

$$P_{point} = [P_{bg}; P_{fg}^{(1)}; \dots; P_{fg}^{(N)}] \in \mathbb{R}^{(N+1) \times D}$$

* `P_fg^(k)` pools over the K shots of way k only; `P_bg` pools over all ways and shots [VIPSEG models/vipseg.py:108,127].
* No L2 normalisation of `P_point` [DECISION D-10].
* **Empty background** (every support point is foreground): `P_bg = 0.1·1` [VIPSEG models/vipseg.py:111-113].
* **Empty foreground** for a way raises an error. It cannot happen with the inherited loader: a support block holds more than `max(0.05·n, 100)` target points [VIPSEG dataloaders/s3dis.py:54-57] and keeps all of them or `int(ratio·2048) ≥ 102` [VIPSEG dataloaders/loader.py:41-47].
* Implementation: `models/prototypes.py::point_prototypes`, mask-weighted `einsum` over the K shots (per way) or over all ways and shots (background); no reshape across the way or shot axes.

---

## 4. Cross-modal prototype (Eq.4–9)

### 4.1 Adapter (Eq.4–5)

$$E_{adapted}^{(m)} = W_2 \cdot \text{Dropout}(\text{ReLU}(\text{LN}(W_1 E_{CLIP}^{(m)} + b_1))) + b_2 \in \mathbb{R}^{(N+1) \times D}$$

* One adapter per modality m ∈ {text, image, audio}; one modality per run [PAPER Eq.4] [DECISION D-05].
* `W_1 ∈ R^{D×512}`, `W_2 ∈ R^{D×D}` [PAPER Eq.5] [PAPER §4.1]; Dropout p = 0.1 after ReLU [DECISION D-16].

### 4.2 Generator (Eq.6)

$$z \sim \mathcal{N}(0, I) \in \mathbb{R}^{(N+1) \times D}, \qquad P_{modal} = G([E_{adapted}^{(m)}; z]) \in \mathbb{R}^{(N+1) \times D}$$

* `E_fused := E_adapted^(m)` [DECISION D-05]; G = `Linear(2D→D) → ReLU → Linear(D→D) → ReLU → Linear(D→D)` [PAPER Eq.6] [DECISION D-16].
* Training: sample a fresh z per episode. Evaluation: z = 0 [DECISION D-06].

### 4.3 Multi-scale MMD (Eq.7)

For sample sets `X = {x_i}_{i=1}^{M}` and `Y = {y_j}_{j=1}^{L}` in R^D, the **squared** RKHS distance of Eq.7 expands exactly to [PAPER Eq.7]:

$$\text{MMD}(X, Y) = \frac{1}{M^2}\sum_{i,i'} k(x_i, x_{i'}) + \frac{1}{L^2}\sum_{j,j'} k(y_j, y_{j'}) - \frac{2}{ML}\sum_{i,j} k(x_i, y_j)$$

$$k(x, y) = \sum_{\sigma \in \{2,5,10,20,40,80\}} \exp\left(-\frac{\|x - y\|_2^2}{2\sigma^2}\right)$$

* No square root is taken [PAPER Eq.7].
* `k(x, x) = 6` for every x (one per bandwidth).

### 4.4 Decoupled alignment loss (Eq.8)

$$\mathcal{L}_{GMMN} = 0.1 \cdot \text{MMD}(P_{modal}^{bg}, P_{point}^{bg}) + 1.0 \cdot \text{MMD}(P_{modal}^{fg}, P_{point}^{fg})$$

* `P^bg` = row 0 (a set of 1 sample); `P^fg` = rows 1…N taken jointly (a set of N samples) [PAPER Eq.8] [DECISION D-04].
* `P_point` is not detached [DECISION D-04]. It is the same `P_point` that enters Eq.9, i.e. after the `l2norm_point_proto` ablation when that flag is on [DECISION D-10].

### 4.5 Initial prototype (Eq.9)

$$P^0 = P_{point} + P_{modal} \in \mathbb{R}^{(N+1) \times D}$$
[PAPER Eq.9]

For the cascade, `P^0` is copied once per query, giving `[B_q, N+1, D]`, because every later step depends on the query [PAPER Eq.14–15] [VIPSEG models/vipseg.py:145].

---

## 5. Entropy-aware Prototype Purification Module (stage t = 1..T)

Inputs: `P^{t−1} ∈ [B_q, N+1, D]`, `F^s ∈ [N, K, 2048, D]`, `F^q ∈ [B_q, 2048, D]`. Output: `P^t ∈ [B_q, N+1, D]`. Each stage has its own parameters [DECISION D-16].

### 5.1 Information-theoretic gating (Eq.10–12)

Applied channel-wise, to the incoming prototype with `gate_target = prototype` (the default) or to `F^q` and `F^s` with `gate_target = features` [DECISION D-02]. Eq.10 names its argument a "feature vector"; the two readings differ in what `x` is:

$$p_{i} = \sigma(P^{t-1}_{i}), \quad H_{i} = -p_{i}\log(p_{i} + \epsilon) - (1 - p_{i})\log(1 - p_{i} + \epsilon)$$
$$g_{i} = \sigma(2(\theta_t - H_{i})), \qquad P^{t-1}_{gated} = P^{t-1} \odot g$$
[PAPER Eq.10–12]

* Natural logarithm; `H ∈ [0, ln 2]`; with θ = 0.5, `g ∈ [0.405, 0.731]`. `g < 1` for every finite θ, so Eq.11 can only attenuate, never "amplify" as the abstract says [DECISION D-02].
* With `gate_target = features` the gate is applied per point to `F^s [N,K,2048,D]` and `F^q [B_q,2048,D]` before Eq.13, and ψ in Eq.14 receives the **ungated** `P^{t-1}`, exactly as Eq.14 prints. With `gate_target = prototype` ψ receives `P^{t-1}_gated`, which the printed Eq.14 does not. Eq.15–18 name `F^q` and `F^s` with no mention of gating, so the diffusion branch is ungated under both readings.
* Implementation clamp of `p` to `[10⁻⁷, 1 − 10⁻⁷]` [DECISION D-16].

### 5.2 Cross-attention refinement (Eq.13–14)

Channel-correlation form [DECISION D-01]. Let `c ∈ {0..N}` index class slots and `k ∈ {1..K}` shots; the background slot uses the way-mean of support features [VIPSEG models/vipseg.py:244]:

$$F^s_{0,k} = \frac{1}{N}\sum_{n=1}^{N} F^s_{n,k} \in \mathbb{R}^{2048 \times D}$$

1. Token pooling: `F̃^q = MaxPool_32(F^q) ∈ [B_q, 64, D]`, `F̃^s_{c,k} = MaxPool_32(F^s_{c,k}) ∈ [N+1, K, 64, D]` [VIPSEG models/vipseg.py:213,248-249].
2. Shared projection φ = `Conv1d(in_channels=64, out_channels=d, kernel_size=1, bias=False)` applied to `[·, 64, D]`: the 64 pooled tokens are the conv's input channels and the D feature channels are its length axis, so each of the d outputs is a learned mix of tokens [PAPER Eq.13] [VIPSEG models/vipseg.py:219,257-258]:
$$Q' = \varphi(\tilde{F}^q) \in [B_q, d, D], \qquad S'_{c,k} = \varphi(\tilde{F}^s_{c,k}) \in [N+1, K, d, D]$$
3. Channel correlation, one matrix per (query, class, shot), row-wise softmax, scaled by √d [PAPER Eq.14] [DECISION D-01]:
$$A_{b,c,k} = \text{softmax}_{row}\left(\frac{Q'^{\top}_{b}\, S'_{c,k}}{\sqrt{d}}\right) \in \mathbb{R}^{D \times D}$$
   i.e. `A[b,c,k,i,j] = softmax_j( Σ_r Q'[b,r,i] · S'[c,k,r,j] / √72 )`.
4. Prototype refinement with ψ = `Linear(D→D)`; the average over shots is taken on `P_cross` [PAPER Eq.14] [VIPSEG models/vipseg.py:222,296] [DECISION D-01]:
$$P_{cross}[b,c] = \frac{1}{K}\sum_{k=1}^{K} A_{b,c,k}\, \psi(P^{t-1}_{gated}[b,c]) \in \mathbb{R}^{D}$$

Not to be copied: VIP-Seg's `reshape(proj_dim, -1)` before the product, which interleaves classes and filters [DECISION D-01]. Its effect is, however, to make the correlation shared by all class rows of the episode, which is what `cross_attn_support = pooled` provides in a clean per-query form [DECISION D-23]:

$$\bar{F}^s = \frac{1}{NK}\sum_{n,k} \text{MaxPool}_{32}(F^s_{n,k}) \in \mathbb{R}^{64 \times D}, \qquad \bar{S}' = \varphi(\bar{F}^s) \in \mathbb{R}^{d \times D}$$
$$A_b = \text{softmax}_{row}\left(\frac{Q'^{\top}_b \bar{S}'}{\sqrt{d}}\right) \in \mathbb{R}^{D \times D}, \qquad P_{cross}[b,c] = A_b\, \psi(P^{t-1}_{gated}[b,c])$$

The default is `class_slots`, the form D-01 chose; `pooled` is the literal reading of Eq.13's single `S′` and is measured in phase 16 [DECISION D-23].

With `cross_attn_norm = layernorm` [DECISION D-18], one shared LayerNorm standardises the columns of `Q'` and `S'_{c,k}` along the projection axis `r` before step 3:
$$Q'_{:,i} \leftarrow \mathrm{LN}(Q'_{:,i}), \qquad S'_{c,k,:,j} \leftarrow \mathrm{LN}(S'_{c,k,:,j})$$
This makes `A` invariant to the scale of the features and gives each stage 144 parameters with which to set its own sharpness. It is an ablation flag, not a fix, and scored below the default on the VM: both limits of the channel softmax leave `P_cross` constant along `D`, and `P_diffuse` has no class index [DECISION D-16], so the class-discriminative content of `P^t` comes from the residual of Eq.21 either way. The default is `none`, the literal Eq.14; see D-18 for the measurements.

### 5.3 Prototype diffusion (Eq.15–18)

$$q_{ch}[b] = \sigma\left(\text{mean}_i F^q_{b,i}\right) \in \mathbb{R}^D, \qquad s_{ch} = \sigma\left(\text{mean}_{n,j,i} F^s_{n,j,i}\right) \in \mathbb{R}^D$$
[PAPER Eq.15] [DECISION D-16]

$$m_q = \mathbb{1}[q_{ch} > \tau], \quad m_s = \mathbb{1}[s_{ch} > \tau], \quad m_{common} = m_q \odot m_s$$
$$c_{common} = \frac{q_{ch} + s_{ch}}{2} \odot m_{common}, \qquad c_{unique} = \frac{q_{ch} \odot (m_q - m_{common}) + s_{ch} \odot (m_s - m_{common})}{2}$$
$$P_{diffuse} = \alpha\, c_{common} + (1 - \alpha)\, c_{unique} \in \mathbb{R}^D$$
[PAPER Eq.15–18]

* `P_diffuse[b]` is broadcast to all N+1 class rows: `[B_q, N+1, D]` [DECISION D-16].
* `P_diffuse` carries no class information: Eq.15–18 have no class index [PAPER Eq.15–18].
* Because `F ≥ 0` (§2), a channel passes τ exactly when its mean is strictly positive. If all channel means are positive in both branches, `c_unique = 0` and `P_diffuse = (q_ch + s_ch)/4 ∈ [0.25, 0.5]^D`; a channel that is zero on every point of only one branch gives a non-zero `c_unique`. This is expected, not a bug [DECISION D-14].

### 5.4 Adaptive fusion, SE, class weights, residual (Eq.19–21)

1. Fusion weights, one pair per query [PAPER Eq.19] [DECISION D-11] [DECISION D-16]:
$$w_b = \text{softmax}\left(f_{fusion}\left(\text{mean}_c [P_{cross}[b,c]; P_{diffuse}[b,c]]\right)\right) \in \mathbb{R}^2, \qquad P_{combined} = w_{b,1} P_{cross} + w_{b,2} P_{diffuse}$$
2. Squeeze-and-excitation, pooled over classes, r = 4 [PAPER Eq.20] [DECISION D-16]:
$$a_b = \sigma\left(W_2 \,\text{ReLU}(W_1 \,\text{mean}_c P_{combined}[b,c])\right) \in \mathbb{R}^D, \qquad P_{attended} = P_{combined} \odot a_b$$
3. Class weights on rows [PAPER §3.4]:
$$P_{weighted}[b,c] = w_{cls}[c] \cdot P_{attended}[b,c]$$
4. Output with residual to the **ungated** input [PAPER Eq.21] [DECISION D-02]:
$$P^t = \text{LN}\left(W_{out}\,\text{ReLU}(P_{weighted}) + P^{t-1}\right) \in [B_q, N+1, D]$$

### 5.5 Stage logits (Eq.23)

$$L^t = F^q (P^t)^\top \in [B_q, 2048, N+1]$$
Plain dot product; no temperature, no normalisation [PAPER Eq.23] [DECISION D-10].

---

## 6. Cascade and routing (Eq.22, 24–25)

$$P^0 \xrightarrow{\text{EPPM}_1} P^1 \xrightarrow{\text{EPPM}_2} P^2 \xrightarrow{\text{EPPM}_3} P^3 \xrightarrow{\text{EPPM}_4} P^4$$
[PAPER Eq.22]

$$w_{gate}[b] = \text{softmax}\left(W_g \,\text{mean}_i F^q_{b,i}\right) \in \mathbb{R}^T, \quad W_g \in \mathbb{R}^{T \times D}, \qquad L_{final} = \sum_{t=1}^{T} w_{gate}^{(t)} L^t \in [B_q, 2048, N+1]$$
[PAPER Eq.24–25]

* `W_g` has no bias term, as printed [PAPER Eq.24]. (VIP-Seg's gating layer has one [VIPSEG models/vipseg.py:190]; L1 wins.)

---

## 7. Training objective (Eq.26–27)

$$\mathcal{L}_{total} = \mathcal{L}_{seg} + \lambda\, \mathcal{L}_{GMMN}, \quad \lambda = 1.0$$
[PAPER Eq.26]

$$\mathcal{L}_{seg} = -\frac{1}{B_q N_q}\sum_{b=1}^{B_q}\sum_{i=1}^{N_q} \log \frac{\exp(L_{final}[b,i,y_{b,i}])}{\sum_{c=0}^{N}\exp(L_{final}[b,i,c])}$$

* **Unweighted** cross-entropy on `L_final` only; no per-stage loss, no class weights [PAPER Eq.27]. The mean over all queries of the episode follows [VIPSEG models/vipseg.py:182].
* `w_cls` appears only in §5.4 step 3 [PAPER §3.4].
* Batch loss = mean of `L_total` over the 4 episodes [DECISION D-12].

---

## 8. Inference

* z = 0 in §4.2 [DECISION D-06]; dropout disabled; BatchNorm in eval mode.
* Prediction `Ŷ^q = argmax_c L_final[b, i, c]` [PAPER Eq.1].

---

## 9. Numerical rules

```python
# Entropy (§5.1): clamp is an implementation detail [DECISION D-16]
p = torch.sigmoid(x).clamp(1e-7, 1.0 - 1e-7)                      # [B_q, N+1, D]
H = -p * torch.log(p + 1e-8) - (1.0 - p) * torch.log(1.0 - p + 1e-8)  # [B_q, N+1, D], in [0, ln 2]

# Squared distances for the RBF kernel (§4.3): from the differences, never ||x||^2 + ||y||^2 - 2x.y,
# whose rounding makes k(x, x) differ from 6 at large norms; the sets hold at most N+1 rows
def pairwise_sq_dist(x, y):                                       # x [M, D], y [L, D]
    return ((x[:, None, :] - y[None, :, :]) ** 2).sum(-1)         # [M, L], >= 0, diagonal exactly 0

# MMD (§4.3): squared form, never sqrt [PAPER Eq.7]
mmd = k(X, X).mean() + k(Y, Y).mean() - 2.0 * k(X, Y).mean()      # scalar, >= 0 up to rounding

# Channel correlation (§5.2): explicit indices, no reshape across batch/class axes [DECISION D-01]
A = torch.softmax(torch.einsum('brd,ckre->bckde', Qp, Sp) / 72 ** 0.5, dim=-1)  # [B_q, N+1, K, D, D]
```

---

## 10. Shape contract (one episode)

| Tensor | Shape | Source |
| :--- | :--- | :--- |
| `X^s`, `Y^s` | `[N, K, 2048, 9]`, `[N, K, 2048]` | §1 |
| `X^q`, `Y^q` | `[B_q, 2048, 9]`, `[B_q, 2048]` | §1 |
| `F^s`, `F^q` | `[N, K, 2048, 128]`, `[B_q, 2048, 128]` | §2 |
| `E_CLIP`, `E_adapted`, `z` | `[N+1, 512]`, `[N+1, 128]`, `[N+1, 128]` | §4 |
| `P_point`, `P_modal` | `[N+1, 128]` | §3, §4 |
| `P^0` (after per-query copy) | `[B_q, N+1, 128]` | §4.5 |
| `F̃^q`, `F̃^s` | `[B_q, 64, 128]`, `[N+1, K, 64, 128]` | §5.2 |
| `Q'`, `S'` | `[B_q, 72, 128]`, `[N+1, K, 72, 128]` | §5.2 |
| `A` | `[B_q, N+1, K, 128, 128]` | §5.2 |
| `P_cross`, `P_diffuse`, `P^t` | `[B_q, N+1, 128]` | §5 |
| `w`, `a` | `[B_q, 2]`, `[B_q, 128]` | §5.4 |
| `L^t`, `L_final` | `[B_q, 2048, N+1]` | §5.5, §6 |
| `w_gate` | `[B_q, T]` | §6 |
| `L_seg`, `L_GMMN`, `L_total` | scalars | §4.4, §7 |

---

## 11. Query-side EM refinement (beyond the paper) [DECISION D-26]

Applied after a trained model's final scoring rule `L = F^q M^T`, `F^q ∈ [B_q, P, D]`, `M ∈ [B_q, N+1, D]`
(for VIP-Seg, `M = Σ_t w_t M^t` of its gated steps [VIPSEG models/vipseg.py:152-174]). For t = 1..T, the
prior `M` held fixed [DECISION D-26]:

$$r_{ic} = \mathrm{softmax}_c(L^{t-1}_{ic}), \qquad w_i = 1 - \frac{H(r_i)}{\ln(N+1)}, \qquad
u_c = \frac{1}{P}\sum_i w_i\, r_{ic}\, \frac{f_i}{\lVert f_i\rVert}$$
$$\mu_c = \lVert m_c\rVert \cdot \mathrm{normalise}\!\left(\frac{m_c}{\lVert m_c\rVert} + \kappa\, u_c\right), \qquad L^t_{ic} = \langle f_i, \mu_c\rangle$$
[DECISION D-26]

* `H` is the Shannon entropy of the class posterior of one point, natural logarithm, `0·log 0 := 0`
  (probabilities clamped at 10⁻¹²) [DECISION D-26].
* Weight arms: `entropy` as above; `none`, `w_i = 1`; `ssp`, `w_i r_ic` replaced by the one-hot argmax of
  points whose top probability exceeds 0.7 (foreground) or 0.6 (background) [DECISION D-26]; `oracle`,
  the one-hot query labels, an upper bound only [DECISION D-26].
* `‖u_c‖ ≤ Σ_i w_i r_ic / P`: a class with no confident mass is left unchanged, and `κ = 0` or `T = 0`
  reproduces `L` exactly [DECISION D-26].

| Tensor | Shape | Source |
| :--- | :--- | :--- |
| `r`, `w_i r_ic` | `[B_q, P, N+1]` | [DECISION D-26] |
| `w` | `[B_q, P]` | [DECISION D-26] |
| `u`, `μ` | `[B_q, N+1, D]` | [DECISION D-26] |
