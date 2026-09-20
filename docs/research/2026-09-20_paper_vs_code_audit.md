# Audit: paper -> spec 02 -> code, equation by equation (2026-09-20)

* **Scope.** Every equation (Eq.1-28) and every stated number of `10069.pdf` ("CascadeProto", Wang et al.),
  checked on three links: **(a)** paper -> `docs/spec/02_TENSOR_MATH_SPEC.md`, **(b)** spec 02 -> code,
  **(c)** paper -> code directly.
* **Method.** The PDF was read both as page images (pp. 1-14) and as extracted text. The extraction
  returns the LaTeX source of the display equations, so every equation below is quoted from the paper's
  own markup, and every shape annotation was re-checked against the rendered page.
* **Vocabulary.** `TRANSCRIPTION` = the spec says something the paper does not; `BUG` = the code does not
  do what the spec says; `PAPER` = a defect or ambiguity inside the paper itself; `DECISION` = a
  documented `D-nn` choice (these are not defects, and where I disagree I say so explicitly).
* **Nothing was changed** except this file. No training was run.

---

## 1. Summary of findings, ranked by severity

| # | Kind | Where | Finding |
| :--- | :--- | :--- | :--- |
| F1 | PAPER | Eq.19, Eq.10-21 | **Eq.19 fuses exactly two terms** - the repo's assumption is **confirmed verbatim**. And no equation in Eq.10-21 adds a channel-preserving function of `P^{t-1}` to the fused prototype: the only channel-identity-preserving path from `P^{t-1}` to `P^t` is the bare residual of Eq.21. VIP-Seg's `proto_self` has **no counterpart in the printed equations**. A gap in the paper, not a repo bug. |
| F2 | PAPER | Tab.4 vs Tab.2/Tab.6 | The Table 4 baseline row (81.28 Avg / 82.72 S0) is defined as "a plain VIP-Seg backbone with masked average pooling and single-step prototype matching", yet VIP-Seg's own row in Table 2 is 74.15 Avg and its Table 6 S0 is 72.20. **The paper says nothing that explains the +7.13 / +10.52 gap.** Every ablation increment is measured from this unexplained row. |
| F3 | TRANSCRIPTION | `00_SOURCES_AND_DECISIONS.md` D-01 | D-01 states that of the printed Eq.13-14 "only the annotation `in R^{Nq x Ns}` does not [match]". That is **not accurate**: the chosen reading also discards Eq.13's own annotation `Q' in R^{Nq x d}` (the code's `Q'` is `[d, D]`), and it inserts a `MaxPool1d(32)` that appears nowhere in the paper. The decision is still defensible; its evidence line is not. |
| F4 | PAPER | Eq.14 | D-01's core claim is **verified**: `A in R^{Nq x Ns}` times `psi(P^{t-1}) in R^{(N+1) x D}` is undefined for `Ns != N+1`, and even if it were defined the result would be `R^{Nq x D}`, not the `R^{(N+1) x D}` that Eq.19 and Eq.21 require. Doubly inconsistent as printed. |
| F5 | PAPER | Tab.6 / §4.4 | The parameter budget of Table 6 is **arithmetically unreachable** from the modules the paper describes. Four EPPM stages built exactly from Eq.10-21 cost 317,580 parameters on their own, already above the ~0.31M the paper implies for LMA+EPPM+ADRM; the LMA alone (148,352) exceeds the "+0.12M" net overhead §4.4 claims. |
| F6 | PAPER | Tab.4 row 3 vs Tab.5 `T=1` | Under D-17's mapping these are the **same configuration**, but the paper reports 83.91 Avg and 83.28 Avg. Either D-17's reading of "+ Entropy Gate" is wrong or the two tables disagree. The paper does not disambiguate; the repo has not recorded the conflict. |
| F7 | PAPER | abstract/§3.4 vs Eq.11-12 | The gate is claimed to be "**amplifying** low-entropy foreground information", but `g_i = sigma(2(theta - H_i))` with `H in [0, ln 2]` and `theta = 0.5` gives `g in [0.405, 0.731]`: a strict attenuator. It cannot amplify at any parameter value, not merely at initialisation. |
| F8 | PAPER | Eq.15-18 | "Prototype diffusion" contains **no prototype**. The introductory sentence promises "To propagate prototype information across both branches", yet Eq.15-18 read only `F^q` and `F^s`, carry no class index, and never touch `P^{t-1}`. |
| F9 | PAPER | Eq.24-25 vs its own prose | Eq.24 pools `F^q` over **all** query points, so `w_gate` is one `R^T` vector per query cloud. The abstract and §3.5 promise per-region behaviour ("allowing simple regions to benefit from early-stage outputs"). Not reconcilable. The code follows the equation, which is right, but spec 02 §6 does not record the tension. |
| F10 | PAPER | Eq.6, Eq.10, Eq.13, Eq.15, §3.2 | Undefined or mistyped symbols: `E_fused` (never defined), `x` in Eq.10 (feature? prototype?), `mean(F, dim=1)` in Eq.15 (yields `R^{N_points}`, not the "channel-level activations" the text names), and §3.2's "fuses both sources" contradicting the single-modality design. All covered by D-02/D-05/D-16. |
| F11 | PAPER | Tab.5 / §4.3 | §4.3 says "Table 5 reports performance **and inference time**" and quotes "approximately 16ms per stage", but Table 5 has only `S0 / S1 / Avg` rows. The 16ms figure has no source inside the paper. |
| F12 | minor | `pipeline/model_api.py:15` | `GMMN_WEIGHT = 1.0` is tagged `[PAPER §4.1]`. `lambda = 1.0` is stated in §3.6 under Eq.26, not §4.1. Spec 02 §0 tags it correctly. |

**No implementation bug (`BUG`) was found.** Every equation in spec 02 is implemented in the code as the
spec writes it; the checks are itemised per equation below.

---

## 2. Eq.1-3 - problem setup, features, masked average pooling

### What the paper says

> "`\hat{\mathbf{Y}}^q = f_{\theta}(\mathcal{S},\, \mathcal{M},\, \mathbf{X}^q)`" (Eq.1), with
> "`M = {c_1, ..., c_K}` represents optional cross-modal semantic descriptors (text, image, audio)" (§3.1).

> "`\mathbf{F} = f_{\text{enc}}(\mathbf{X}) \in \mathbb{R}^{N_s \times D}` or `\mathbb{R}^{N_q \times D}`"
> (Eq.2), "where `D = 128` is the feature dimension" (§3.3).

> "The encoder is shared between support and query branches. Initial foreground and background
> prototypes are extracted from the support features via masked average pooling:" (§3.3)
>
> `P_fg^(k) = (1/|M_fg^(k)|) sum_{i in M_fg^(k)} F_i^s,   P_bg = (1/|M_bg|) sum_{i in M_bg} F_i^s`  (Eq.3)
>
> "where `M_fg^(k) = {i | Y_i^s = 1, class(i) = k}` and `M_bg = {i | Y_i^s = 0}`, with `k = 1, ..., N`
> ranging over the `N` categories in the current episode. The stacked point prototype is
> `P_point = [P_bg; P_fg^(1); ...; P_fg^(N)] in R^{(N+1) x D}`." (§3.3)

**The background prototype is formed by a single pooled mean over every support point whose binary mask
is 0** - one row, shared by the whole episode, placed at index 0 of the stack. The paper gives `M_bg` no
way index and no shot index, so *which* support clouds contribute is left open (ambiguity below).

Note also a notation clash: `M` is the set of cross-modal descriptors in Eq.1 and `M_fg`/`M_bg` are point
index sets in Eq.3.

### (a) paper -> spec 02

Correct, and more precise than the paper. `02 §3` (lines 62-68) adds the missing way/shot indices:
`P_fg^(k)` pools over the `K` shots of way `k`; `P_bg` pools over **all** ways and shots. The extra
precision is sourced to `[VIPSEG models/vipseg.py:108-130]`, which is the right authority for something
Eq.3 does not index. The `P_bg = 0.1 * 1` fallback (`02 §3` line 72) is VIP-Seg's, correctly tagged.

*Ambiguity in the paper, correctly resolved but worth naming:* `M_bg = {i | Y_i^s = 0}` could equally be
read per support cloud (`N*K` background prototypes, then averaged) or per way. The readings coincide
only when the ways have equal background point counts. The paper does not say; the spec follows L2.

### (b) spec 02 -> code

`models/prototypes.py:200-215` implements exactly the spec:

* `models/prototypes.py:207` - `einsum("nkp,nkpd->nd", fg, support_feat) / fg_count[:, None]`, i.e. per
  way over its `K` shots.
* `models/prototypes.py:209-213` - `bg_count = bg.sum()` over all ways and shots, one `[D]` row.
* `models/prototypes.py:211` - the `0.1` fallback, `EMPTY_BACKGROUND_VALUE` at `models/prototypes.py:179`.
* `models/prototypes.py:215` - `cat([p_bg[None], p_fg])`, so index 0 is background, as Eq.3's stack order.
* Binary-mask guard at `models/prototypes.py:197-198`, matching §3.1's "binary foreground mask".

### (c) paper -> code

Matches. No L2 normalisation of `P_point` (`models/cascadeproto.py:117-119`; normalisation exists only
behind the `l2norm_point_proto` ablation), which is what Eq.3 prints. VIP-Seg **does** normalise
(`models/vipseg.py:142`), so this is a deliberate divergence from the reference implementation in favour
of the paper - D-10, and the right priority order - but it is the single largest measured effect in the
repo's own experiments (D-18: `l2norm_point_proto` alone is worth about +8 points).

---

## 3. Eq.4-9 - the LMA, the generator, the GMMN loss, and `P^0`

### What the paper says

> "we design a lightweight Learnable Modality Adapter (LMA) - one per modality - that map each CLIP
> embedding into the point cloud feature space:
> `E_adapted^(m) = Adapter^(m)(E_CLIP^(m)),  m in {text, image, audio}`" (Eq.4)
>
> "where each adapter is a **two-layer MLP with LayerNorm and Dropout**, and the adapted embedding is:
> `Adapter(E) = W_2 . ReLU(LN(W_1 . E + b_1)) + b_2`" (Eq.5)

**Adapter shape:** §4.1 fixes it - "each LMA projects CLIP embeddings from `R^512` to `R^128` via a
two-layer MLP". The printed Eq.5 shows **no Dropout**; only the surrounding prose mentions it, and it
gives neither rate nor position.

> "Given adapted cross-modal embeddings `E_fused` and random noise `z ~ N(0, I)`, a **three-layer MLP
> generator** `G` produces cross-modal prototypes:
> `P_modal = G(E_fused, z) in R^{(N+1) x D}`" (Eq.6)

**Generator input:** `G(E_fused, z)`. `E_fused` is **never defined anywhere in the paper**. Whether the
two arguments are concatenated, added, or `z` conditions the layers is not stated, and no dimension is
given for `z`.

> "The foreground and background alignment losses are defined using the Maximum Mean Discrepancy (MMD)
> with a multi-scale Gaussian kernel `k_sigma`:
> `MMD(P, Q) = || (1/|P|) sum_{x in P} phi(x) - (1/|Q|) sum_{y in Q} phi(y) ||_H^2`" (Eq.7)
>
> "where `k(x, y) = sum_{sigma in {2,5,10,20,40,80}} exp(-||x - y||^2 / 2 sigma^2)`." (§3.3)

**Is MMD squared? Yes** - the `^2` sits on the RKHS norm, visible both in the LaTeX
(`\|_{\mathcal{H}}^2`) and on the rendered page 7. **Which bandwidths: {2, 5, 10, 20, 40, 80}**, six
Gaussians **summed** (not averaged). `phi` in Eq.7 is the RKHS feature map of `k`, not the `varphi` of
Eq.13 - another symbol collision.

> "`L_GMMN = 0.1 . MMD(P_modal^bg, P_point^bg)  [underbraced L_bg]  + 1.0 . MMD(P_modal^fg, P_point^fg)  [underbraced L_fg]`" (Eq.8)
>
> "The lower weight on `L_bg` reflects background heterogeneity: forcing the background cross-modal
> prototype to match the background point prototype would impede generalization." (§3.3)

**What weights on background vs foreground: background 0.1, foreground 1.0.** Eq.8 does not define what
the sample *sets* are; with `P^bg` one row and `P^fg` `N` rows, it is asking for an MMD between sets of
size 1 and 2-3. The paper never acknowledges this.

> "The fused initial prototype used for training is: `P^0 = P_point + P_modal`." (Eq.9)

**How `P_modal` combines with `P_point`: a plain unweighted sum.** No gate, no scale, no concatenation.
The qualifier "**used for training**" is the paper's only statement about inference here, and §3 promises
an "inference strategy" that is never written.

### (a) paper -> spec 02

| Item | Verdict |
| :--- | :--- |
| Eq.5 adapter | `02 §4.1` line 82 prints `W_2 . Dropout(ReLU(LN(W_1 E + b_1))) + b_2`. Dropout's **position** is an addition to the printed Eq.5, tagged `[DECISION D-16]` on line 85. Honest. |
| `W_1 in R^{D x 512}`, `W_2 in R^{D x D}` | Correct, from §4.1's "`R^512` to `R^128`". |
| Eq.6 generator | `02 §4.2` line 89 writes `G([E_adapted^(m); z])` - concatenation - plus `E_fused := E_adapted^(m)`. Both interpretations, both tagged `[DECISION D-05]`. Correct handling of an undefined symbol. |
| Eq.7 MMD | `02 §4.3` lines 96-103 gives the standard expansion of the **squared** RKHS distance and states "No square root is taken [PAPER Eq.7]". **Correct transcription.** `k(x, x) = 6` follows from six bandwidths. |
| Eq.8 weights | `02 §4.4` line 107 reproduces 0.1 / 1.0 exactly. The set definitions (line 109) are `[DECISION D-04]`, correctly flagged as not paper content. |
| Eq.9 | `02 §4.5` line 114, verbatim. The per-query copy (line 117) is an implementation necessity, tagged. |

No transcription error in this group.

### (b) spec 02 -> code

* Adapter: `models/lma.py:23-30` - `Linear(512,128) -> LayerNorm -> ReLU -> Dropout(0.1) -> Linear(128,128)`,
  exactly the spec's line 82 ordering. `ADAPTER_DROPOUT = 0.1` at `models/lma.py:14`.
* Generator: `models/lma.py:38-39` - `Linear(256,128) -> ReLU -> Linear(128,128) -> ReLU -> Linear(128,128)`,
  three layers, input width `2D`, matching `02 §4.2` and D-16 item 2.
* `E_fused := E_adapted`, concatenated with `z`: `models/lma.py:66,74`.
* `z`: `models/lma.py:67-71` - fresh `randn_like` in training, zeros in eval (D-06); `mean_of_M` raises at
  `models/lma.py:56-57`, as D-06 requires.
* Kernel: `loss/gmmn_loss.py:83` - `RBF_BANDWIDTHS = (2.0, 5.0, 10.0, 20.0, 40.0, 80.0)`;
  `loss/gmmn_loss.py:101` sums `exp(-d2 / (2 sigma^2))` over them. Exactly Eq.7's kernel.
* Squared MMD: `loss/gmmn_loss.py:110` - `k(x,x).mean() + k(y,y).mean() - 2*k(x,y).mean()`, **no sqrt**,
  no clamp. Matches `02 §4.3` and `02 §9`.
* Distances from differences rather than `||x||^2+||y||^2-2xy`: `loss/gmmn_loss.py:95`, as `02 §9` requires.
* Weights: `loss/gmmn_loss.py:84-85,134` - `0.1 * loss_bg + 1.0 * loss_fg`.
* Sets: `loss/gmmn_loss.py:128` (row 0 vs row 0) and `:130` (rows 1..N jointly) - D-04's choice.
* Eq.9: `models/cascadeproto.py:125` - `prototypes = p_point + p_modal`. Plain sum.

No deviation.

### (c) paper -> code

Matches, with the four documented interpretations (`E_fused`, concatenation, dropout placement, MMD
sets). Two things to keep in view:

1. **The MMD estimator on 1-sample sets is degenerate by construction.** With `|P| = |Q| = 1`,
   `MMD = 12 - 2 k(x, y) in [0, 12)`, i.e. `L_bg` is a plain kernel distance, not a distribution match.
   The paper's own justification for Eq.8 ("background heterogeneity") presupposes a *distribution* that
   a single pooled row cannot represent. `03 §4.1` states the 1-vs-1 identity correctly; the paper does not.
2. `P_point` is not detached (`loss/gmmn_loss.py:126-127`, default `detach_point=False`), so `L_GMMN`
   back-propagates into the encoder. The paper is silent; D-04 chose this, which is reasonable since
   Eq.26 optimises jointly.

---

## 4. Eq.10-12 - the entropy gate

### What the paper says

> "Formally, for **a feature vector `x in R^D`**, we compute its per-channel entropy after sigmoid
> normalization:
> `p_i = sigma(x_i),  H_i = -p_i log(p_i + epsilon) - (1 - p_i) log(1 - p_i + epsilon)`" (Eq.10)
>
> "where `epsilon = 10^-8` ensures numerical stability. **A learnable threshold `theta` (initialized to
> 0.5)** governs a soft gate:
> `g_i = sigma(2(theta - H_i))`" (Eq.11)
>
> "which assigns values close to 1 for low-entropy (foreground-like) channels and close to 0 for
> high-entropy (background-like) channels. The gated feature is: `x_gated = x . g`" (Eq.12)

**The exact sigmoid argument is `2(theta - H_i)`** - a factor 2, one learnable scalar `theta`, minus the
per-channel entropy.

**Is `theta` per stage?** Eq.11 says "a learnable threshold `theta`", singular, never indexed by `t`. The
only support for per-stage is §3.5: "each EPPM applies entropy gating **independently**, leading to
compounding suppression of background noise across stages" - which speaks of the gating *operation*, not
unambiguously of the parameter. Since §3.4 defines the EPPM as one module and §3.5 stacks `T` of them,
per-stage `theta` is the natural reading, but it is an inference, not a statement.

**Is the gate applied to the prototype or to the features? The paper does not say.** Evidence pulls both
ways:

* *Toward features:* Eq.10 says "for a **feature vector** `x in R^D`"; Eq.12 says "the **gated feature**";
  the abstract says the EPPM "suppress[es] high-entropy **background features**"; §3.4's opening says the
  module "takes as input the current prototype `P^{t-1}` **along with** support features `F^s` and query
  features `F^q`".
* *Toward the prototype:* the next sentence - "**After entropy gating**, we apply cross-attention between
  the query and support to further refine the prototype" - places gating immediately before Eq.13-14, and
  Eq.13-14 read `F^q`/`F^s` **ungated** while `psi` reads `P^{t-1}`.

**`x_gated` is never referenced again.** The symbol appears in Eq.12 and nowhere else in the paper.

### (a) paper -> spec 02

`02 §5.1` lines 127-134 applies the gate to the incoming prototype, tagged `[DECISION D-02]`, with the
correct constants: `epsilon = 10^-8`, `theta_t` init 0.5, `g = sigma(2(theta - H))`, natural log,
`H in [0, ln 2]`, `g in [0.405, 0.731]`. I verified those bounds: `sigma(2(0.5 - ln 2)) = 0.4046`,
`sigma(2(0.5 - 0)) = 0.7311`. `02 §0` line 22 records `theta_0 = 0.5` as "learnable, one per stage".

**No transcription error, but two soft spots.**

* "one per stage" carries `[PAPER Eq.11] [PAPER §3.5]`; as argued above, §3.5 supports the *operation*
  being independent, not explicitly the parameter. Low risk, but the tag is stronger than the source.
* Eq.10's `x in R^D` is narrowed to `P^{t-1}`. D-02 states the problem openly, so the decision log is
  honest; spec 02 itself carries only the `[DECISION D-02]` pointer, which is the agreed convention.

**Where I disagree with D-02 (a documented decision, not a defect).** D-02's rationale says gating the
prototype "is the only reading consistent with Eq.14 and Eq.21 using `P^{t-1}`". That does not follow: if
the gate applied to `F^q`/`F^s`, Eq.13's `varphi(F^q)` would simply consume gated features and Eq.14/21
would still reference `P^{t-1}` unchanged. Both readings are consistent with Eq.14 and Eq.21. The paper's
own wording ("feature vector", "gated feature", "background features") leans toward the features reading,
which D-02 selects against. Since `gate_target = features` raises `NotImplementedError`
(`models/cascadeproto.py:82-86`), the alternative has never been measured. Given D-18's null result, this
is the cheapest unexplored reading in the repo.

### (b) spec 02 -> code

* `models/eppm.py:12-14` - `ENTROPY_EPS = 1e-8`, `PROB_CLAMP = 1e-7`, `THETA_INIT = 0.5`.
* `models/eppm.py:22-23` - `p = sigmoid(x).clamp(1e-7, 1-1e-7)`, then
  `-p*log(p+1e-8) - (1-p)*log(1-p+1e-8)`. Natural log. Exactly Eq.10 plus the D-16 clamp.
* `models/eppm.py:37` - one `nn.Parameter(0.5)` per `EntropyGate`; one `EntropyGate` per `EPPMStage`
  (`models/eppm.py:195`); `T` independent stages built at `models/cascadeproto.py:108-111`. So `theta_t`
  is per stage, as the spec says.
* `models/eppm.py:43` - `sigmoid(2.0 * (self.theta - channel_entropy(p)))`. Exact Eq.11.
* `models/eppm.py:46` - `p * self.gate(p)`. Exact Eq.12.
* Gate target: `models/eppm.py:204` - `self.cross(self.gate(p_prev), f_s, f_q)`; the gate consumes the
  prototype and its output feeds `psi`, while `f_s`/`f_q` pass ungated. Exactly D-02 / `02 §5.1`.
* `use_gate=False` sets `g = 1` (`models/eppm.py:41-42`), matching D-17's ablation row.

### (c) paper -> code

Matches Eq.10-12 literally under the D-02 reading. One paper-level consequence the code inherits (**F7**):
because `H >= 0`, `g_i = sigma(2(theta - H_i)) < sigma(2 theta)`, and with `theta = 0.5` the gate lies in
`[0.405, 0.731]`. The abstract's "**amplifying** low-entropy foreground information" and §3.4's
"enhancing low-entropy foreground features" are unachievable by Eq.11-12: the gate is a strict
attenuator. D-02 records this as "the gate is weak at initialisation", which understates it - the gate is
*incapable* of amplification by construction, not merely weak at `theta = 0.5`.

---

## 5. Eq.13-14 - the cross-attention (D-01)

### What the paper says, exactly

> "After entropy gating, we apply cross-attention between the query and support to further refine the
> prototype. Query and support features are projected into a low-dimensional space of dimension `d = 72`:
>
> `Q' = varphi(F^q) in R^{N_q x d},   S' = varphi(F^s) in R^{N_s x d}`   (Eq.13)
>
> where `varphi` is a **1 x 1 convolution**. The cross-correlation attention matrix and cross-attended
> prototype are:
>
> `A = softmax( Q' S'^T / sqrt(d) ) in R^{N_q x N_s},   P_cross = A . psi(P^{t-1})`   (Eq.14)
>
> where `psi` is a **learnable linear projection**."

* **The shape the paper prints for `A`:** `R^{N_q x N_s}` - a point-to-point matrix, `2048 x 2048` at the
  paper's own `N_s = N_q = 2048` (§4.1).
* **What `varphi` is:** "a 1 x 1 convolution", one symbol used for both branches (hence shared).
* **What `d` is:** 72 - "the cross-attention projection dimension is `d = 72`" (§4.1) - and it is both the
  output width of `varphi` and the `sqrt(d)` scale.
* **What the softmax is over:** **not stated.** No axis, no subscript. Under the printed shape the
  intended axis is presumably `N_s`, but the paper does not say so.
* **What `P_cross` is:** `A . psi(P^{t-1})` - a matrix product, nothing more. No pooling, no per-class
  construction, no shot handling (Eq.13 writes one `S'` for all `N*K` support blocks).

### Verifying D-01's claim (requested explicitly)

**D-01 is correct that Eq.14 is dimensionally inconsistent as printed, and the inconsistency is worse
than D-01 states.**

1. `A in R^{N_q x N_s}` with `N_s = 2048`; `psi(P^{t-1})` has `N+1 = 3` rows (Eq.6/Eq.9 fix
   `P in R^{(N+1) x D}`, and `psi` is a "linear projection", which cannot change the row count).
   `2048 x 2048` times `3 x 128` is **undefined**.
2. Even granting `N_s = N+1`, the product would be `R^{N_q x D}` - a per-**point** map. But Eq.19 adds
   `P_cross` to `P_diffuse` and Eq.21 adds the result to `P^{t-1} in R^{(N+1) x D}`, so `P_cross` must be
   `(N+1) x D`. The printed Eq.14 cannot produce it under any consistent typing.
3. The two halves of Eq.13-14 are *internally* consistent with **each other**: a 1x1 convolution mapping
   `D -> d` over points gives exactly `Q' in R^{2048 x 72}`, and `Q' S'^T` gives exactly `R^{2048 x 2048}`.
   **It is only the final product `A . psi(P^{t-1})` that breaks.** This matters, because the repo's
   chosen reading resolves the break by discarding the *first* half instead.

**Feasibility arithmetic, independently confirming D-01's FLOPs argument.** The literal point-point `A`
costs `2048 * 2048 * 72 * 2 = 604 MFLOP` (0.30 GMAC) per (query, support block). For 2-way 1-shot with
`T = 4` that is at least 2.4 GMAC of attention alone, against the "+0.38 GFLOPs" total overhead §4.4
claims. The channel-correlation form costs `72 * 128 * 128 * 2 = 2.36 MFLOP` per matrix. D-01's inference
from Table 6 holds.

### (a) paper -> spec 02

`02 §5.2` (lines 136-155) specifies the channel-correlation reading of D-01. The mathematics is stated
cleanly and each departure is tagged. **But the decision log's record of what is being discarded is
inaccurate (F3).** D-01 says:

> "The printed symbols `varphi` = 1x1 conv, `d` = 72, softmax, `psi` = linear and `P_cross = A.psi(P)`
> all match this construct; **only the annotation `in R^{Nq x Ns}` does not.**"

Three printed things do not match, not one:

1. **`Q' in R^{N_q x d}` (Eq.13's own annotation) is discarded.** In the chosen form `Q'` is
   `[B_q, d, D] = [B, 72, 128]` (`02 §10` line 253; `models/eppm.py:103`). The projection maps the 64
   pooled **tokens** to 72 and leaves `D` as the length axis, so `varphi` convolves a different axis than
   Eq.13's shape implies.
2. **A `MaxPool1d(32, stride=32)` is inserted that the paper never mentions.** `02 §5.2` step 1;
   `models/eppm.py:51,59-63`. Eq.13 applies `varphi` to `F^q` directly, over all 2048 points.
3. `R^{N_q x N_s}` is discarded, as D-01 says.

D-01's own step 5 already admits a fourth departure ("Class slots are a VIP-Seg construct, not paper
notation"), so the summary sentence contradicts the decision's own body. **This is a documentation defect
in the decision log, not a code defect**, but it matters: a reader of D-01 concludes the implementation is
"Eq.13-14 minus one annotation", when it is in fact VIP-Seg's PEM cross-branch with Eq.14's `sqrt(d)`
substituted for VIP-Seg's `sqrt(D)`.

**Competing readings, stated so nothing is resolved silently:**

* **R1 (chosen, D-01):** channel-channel correlation after token pooling. Keeps `P_cross = A.psi(P)`
  literal and the `(N+1) x D` output type; discards Eq.13's shape and inserts pooling.
* **R2:** point-point correlation as printed (`varphi: D -> d` over points), `A in R^{2048 x 2048}`, then
  an unstated aggregation down to `(N+1) x D`. Keeps Eq.13 and the first half of Eq.14 literal; requires
  inventing the missing step; contradicted by Table 6's FLOPs.
* **R3:** `psi(P^{t-1})` is a typo for a support-side value (`psi(F^s) in R^{N_s x D}`), making
  `P_cross = A psi(F^s) in R^{N_q x D}` a classical cross-attention - but then `P_cross` is not a
  prototype, and Eq.19/21 still need a pooling step the paper omits.

R1 is the best-supported choice (Table 6, and the paper builds on VIP-Seg). My disagreement is with the
*record*, not the choice.

**One further spec-level detail, correctly flagged by the repo:** `02 §5.2` line 151 forbids copying
VIP-Seg's `reshape(proj_dim, -1)` (`models/vipseg.py:285`), and D-18 documents that this reshape is not a
transpose. I read `models/vipseg.py:285-292`: the reshape folds the batch axis into the projection axis
before the matmul, so VIP-Seg's `crosscor` is indeed not the channel correlation its comment claims. The
repo's analysis is sound.

### (b) spec 02 -> code

Exact, index for index:

* `models/eppm.py:87` - `phi = Conv1d(64, 72, kernel_size=1, bias=False)`, one module used for both
  branches via `project()` (`models/eppm.py:103-104`). Matches `02 §5.2` step 2 and Eq.13's single symbol.
* `models/eppm.py:88` - `psi = Linear(128, 128)`.
* `models/eppm.py:89` - `scale = sqrt(72)` by default (`sqrt_d`), i.e. Eq.14's `sqrt(d)`, not VIP-Seg's
  `sqrt(128)`. D-01 step 3, honoured.
* `models/eppm.py:102` - background slot `cat([f_s.mean(dim=0, keepdim=True), f_s])`, matching
  `models/vipseg.py:244` and `02 §5.2` line 140.
* `models/eppm.py:105-106` - `einsum("bri,ckrj->bckij", q, s) / scale` then `softmax(dim=-1)`. Precisely
  `02 §5.2` line 147's index formula
  `A[b,c,k,i,j] = softmax_j( sum_r Q'[b,r,i] S'[c,k,r,j] / sqrt(72) )`.
* `models/eppm.py:112` - `einsum("bckij,bcj->bci", a, v) / K`, i.e. `A . psi(P)` averaged over shots,
  matching `02 §5.2` step 4.
* `cross_attn_norm` probe: `models/eppm.py:91,96-98`, one shared `LayerNorm(72)` on the projection axis,
  default `none`. Matches `02 §5.2` lines 153-155 and D-18; the default is the literal Eq.14.
* `cross_attn = two_hop` and `gate_target = features` raise (`models/cascadeproto.py:82-86`), as required.

### (c) paper -> code

The code implements R1. Relative to the printed equations it keeps `d = 72`, `sqrt(d)`, a shared 1x1
convolution, a softmax, a learnable linear `psi`, and the literal product `A . psi(P^{t-1})`; it does not
keep `Q' in R^{N_q x d}` or `A in R^{N_q x N_s}`, and it adds max-pooling and per-class slots. That is a
defensible reconstruction of an equation that cannot be implemented as printed - but it should be
described as such.

---

## 6. Eq.15-18 - prototype diffusion

### What the paper says

> "**To propagate prototype information across both branches**, we introduce a prototype diffusion
> mechanism. Channel-level activations are computed for the query and support:
> `q_ch = sigma(mean(F^q, dim=1)),   s_ch = sigma(mean(F^s, dim=1))`" (Eq.15)
>
> "Channels active in both branches form the common activation mask
> `m_common = 1[q_ch > tau] . 1[s_ch > tau]`, and the channel-wise diffusion prototype is:
> `c_common = ((q_ch + s_ch)/2) . m_common`" (Eq.16)
>
> "`c_unique = ( q_ch . (m_q - m_common) + s_ch . (m_s - m_common) ) / 2`" (Eq.17)
>
> "`P_diffuse = alpha . c_common + (1 - alpha) . c_unique`" (Eq.18)
>
> "**with `alpha = 0.5`, `tau = 0.5`.**"

* **`tau` and `alpha` are both 0.5**, given only after Eq.18.
* **Does the paper give `P_diffuse` a class index? No.** Eq.15-18 contain no `k`, no `c`, no `(N+1)`.
  `q_ch`, `s_ch`, `m_*`, `c_common`, `c_unique` and `P_diffuse` are all `R^D` vectors, so `P_diffuse` is
  identical for background and for every foreground class.
* **Is it computed from features or from prototypes? From features only.** `F^q` and `F^s` are the only
  inputs; `P^{t-1}` does not appear. This flatly contradicts the section's own opening clause, "to
  propagate **prototype information**" (**F8**).
* `m_q` and `m_s` are used in Eq.17 but defined only implicitly, inside the prose of Eq.16
  (`1[q_ch > tau]`, `1[s_ch > tau]`).
* `mean(F, dim=1)`: for `F in R^{N_s x D}` as Eq.2 defines it, `dim=1` is the channel axis, which returns
  `R^{N_s}` - not the "channel-level activations" the sentence names (**F10**).

### (a) paper -> spec 02

`02 §5.3` lines 157-169 transcribes Eq.15-18 correctly, defines `m_q`/`m_s` explicitly (line 162), reads
`dim=1` as the point axis with the reasoning spelled out in D-16 item 7, and states plainly:
"`P_diffuse` carries no class information: Eq.15-18 have no class index [PAPER Eq.15-18]" (line 168). The
broadcast to `N+1` rows is tagged `[DECISION D-16]` (line 167). The `F >= 0` degeneracy analysis (line
169, D-14) is correct and follows from the VIP-Seg head ending in ReLU
(`models/vipseg_backbone.py:68-69`). **No transcription error.**

### (b) spec 02 -> code

`models/eppm.py:121-136`, line by line:

* `:129` `q_ch = sigmoid(f_q.mean(dim=1))` -> `[B_q, D]` (point axis of a batched tensor).
* `:130` `s_ch = sigmoid(f_s.mean(dim=(0,1,2)))` -> `[D]`, over ways, shots and points, as `02 §5.3` line 159.
* `:131-133` strict `>` masks and `m_common = m_q * m_s`. Eq.16's indicator product.
* `:134` `c_common = (q_ch + s_ch)/2 * m_common`. Eq.16.
* `:135` `c_unique = (q_ch*(m_q - m_common) + s_ch*(m_s - m_common))/2`. Eq.17, including the sign
  convention of the mask differences.
* `:136` `0.5*c_common + 0.5*c_unique`, with `DIFFUSION_TAU = DIFFUSION_ALPHA = 0.5` (`:117-118`). Eq.18.
* Broadcast: `models/eppm.py:205` - `[:, None, :].expand_as(p_cross)`.

No deviation. The module has **no parameters** (`01 §3` row "3 Diffusion | none"), consistent with
Eq.15-18 containing no learnable symbol.

### (c) paper -> code

Literal. The two consequences are paper-level: `P_diffuse` is class-blind (F8), and with post-ReLU
features `c_unique` is usually zero so `P_diffuse in [0.25, 0.5]^D` (D-14). D-18's three-seed measurement
shows the Eq.19 weight on this branch falling to 0.008-0.082 during training, i.e. the model learns to
discard it - the expected behaviour of a class-blind summand.

---

## 7. Eq.19-21 - fusion, SE, class weights, output

### What the paper says - the two-term question, answered

> "**Adaptive Fusion and Output** A two-layer MLP fusion network `f_fusion` adaptively weights the
> **cross-attention and diffusion prototypes**:
>
> `w = softmax( f_fusion([P_cross; P_diffuse]) ) in R^2,   P_combined = w_1 P_cross + w_2 P_diffuse`" (Eq.19)

**Eq.19 fuses exactly two terms.** Three independent confirmations in the printed line: the bracket
`[P_cross; P_diffuse]` has two entries; the annotation is `in R^2`; and the expansion names exactly
`w_1 P_cross + w_2 P_diffuse`. The prose likewise names only two ("the cross-attention and diffusion
prototypes"). **The repo's assumption is correct.** There is no third summand anywhere in §3.4.

> "A channel attention module (**SE-style**) further recalibrates feature responses:
>
> `a = sigma( W_2 . ReLU(W_1 . AvgPool(P_combined)) ),   P_attended = P_combined . a`" (Eq.20)

**What the SE block is:** squeeze-and-excitation - average-pool `P_combined` to a descriptor, two
projections with a ReLU between and a sigmoid at the end, then a **channel-wise multiply** of
`P_combined` by `a`. The paper gives neither the reduction ratio nor which axis `AvgPool` reduces (it must
be the class axis for `W_1` to consume an `R^D` vector, but that is inference). No biases are printed.

> "To differentiate foreground and background refinement, we apply class-specific weights
> `w_cls = [0.8, 1.0, ..., 1.0]` (the `1.0`s underbraced `N`), yielding `P_weighted = P_attended . w_cls`." (§3.4)

**Where `w_cls` is applied:** to `P_attended`, inside the EPPM, and **only** there. The underbrace covers
the `1.0`s and is labelled `N`, so `w_cls in R^{N+1}` with 0.8 on the background row. It is **not** a loss
weight - Eq.27 is explicitly "the standard cross-entropy".

> "The final refined prototype with residual connection is:
> `P^t = LN( W_out . ReLU(P_weighted) + P^{t-1} )`" (Eq.21)

**What `W_out` is:** an unnamed linear map, applied **after** a ReLU on `P_weighted`. **What the residual
and LayerNorm are applied to:** the residual adds `P^{t-1}` - the *stage input*, not the gated version -
and the LayerNorm wraps the **sum**, not either summand.

### (a) paper -> spec 02

`02 §5.4` lines 171-180:

1. Fusion (line 174): `w_b = softmax(f_fusion(mean_c [P_cross; P_diffuse])) in R^2`. The `mean_c` pooling
   is an addition required by Eq.19's own `R^2` annotation, tagged `[DECISION D-11]`. **Correct** - Eq.19
   as printed is under-specified (an `(N+1) x 2D` input cannot yield `R^2` without pooling), and D-11 says so.
2. SE (line 176): pooled over classes, `r = 4`, tagged `[DECISION D-16]`. The ratio is not paper content
   and is tagged as such.
3. `w_cls` on rows (line 178), `[PAPER §3.4]`. Correct placement; `02 §7` line 209 states explicitly that
   "`w_cls` appears only in §5.4 step 3", i.e. not in the loss. Correct.
4. Eq.21 (line 180) verbatim, with "residual to the **ungated** input" spelled out. Correct: Eq.21 prints
   `P^{t-1}`, and `P^{t-1}_gated` is a different symbol.

**No transcription error in this group.** The `[0.8, 1.0, ..., 1.0] in R^{N+1}` reading of the underbrace
(`02 §0` line 24) is correct.

### (b) spec 02 -> code

`models/eppm.py:153-186`:

* `:161` `fusion = Linear(2D, D) -> ReLU -> Linear(D, 2)` - D-16 item 3, "two-layer MLP".
* `:169-172` concat on the channel axis, `mean(dim=1)` over classes for `per_query`, softmax on the last
  axis. Matches `02 §5.4` step 1.
* `:183` `w[...,0:1]*p_cross + w[...,1:2]*p_diffuse` - **exactly two terms**, matching Eq.19.
* `:162-163,176` SE: `Linear(128,32) -> ReLU -> Linear(32,128) -> sigmoid` on `p_combined.mean(dim=1)`;
  `:184` multiplies `p_combined` by it. Matches Eq.20 and D-16 item 4 (`r = 4`).
* `:146-150,185` `w_cls = [0.8, 1, ..., 1]`, applied to `p_attended` on the class axis, non-learnable.
* `:186` `self.norm(self.w_out(torch.relu(p_weighted)) + p_prev)` - `W_out` after the ReLU, residual on
  the **ungated** `p_prev` (passed at `models/eppm.py:206`), LayerNorm over the sum. Exactly Eq.21.

One accepted departure from the printed equations, recorded in the D-16 change log of 2026-09-19:
`W_1`, `W_2` and `W_out` are `nn.Linear` **with biases**, while Eq.20-21 print none. `02 §5.4` does not
repeat that caveat, so a reader of spec 02 alone would not know. Very minor.

### (c) paper -> code

Matches, including the two-term fusion. See §12 for what the two-term fusion implies.

---

## 8. Eq.22-25 - the cascade and ADRM routing

### What the paper says

> "We cascade `T = 4` EPPM modules to progressively purify the prototype. Starting from `P^0` defined in
> Eq.(9), each stage applies Eq.(21):
> `P^0 -[EPPM_1]-> P^1 -[EPPM_2]-> P^2 -[EPPM_3]-> P^3 -[EPPM_4]-> P^4`" (Eq.22)
>
> "At each step `t`, intermediate per-point logits are computed via **scaled dot-product matching**:
> `L^t = F^q (P^t)^T in R^{N_q x (N+1)}`" (Eq.23)
>
> "Given the query feature map `F^q`, a learnable gating network produces stage-wise weights:
> `w_gate = softmax( W_g . AvgPool(F^q) ) in R^T`" (Eq.24), "where `W_g in R^{T x D}` is learned end-to-end."
>
> "`L_final = sum_{t=1}^{T} w_gate^(t) . L^t`" (Eq.25)

* **Eq.23 is called "scaled" but prints no scale**, no temperature and no normalisation of either factor.
* **Eq.24 has no bias term** (`W_g in R^{T x D}`, a pure matrix) and pools `F^q` over **all** points, so
  `w_gate` is one vector per query cloud (**F9**: the surrounding prose promises per-region weights -
  "allows simple query regions (e.g., uniform surfaces) to be dominated by early-stage outputs" - which a
  globally pooled `R^T` cannot deliver).
* §3.5 also asserts "each EPPM applies entropy gating independently, leading to compounding suppression of
  background noise across stages" - the only textual basis for per-stage parameters.

### (a) paper -> spec 02

`02 §6` lines 189-197 is a faithful transcription, and line 197 makes the right call explicitly:
"`W_g` has no bias term, as printed [PAPER Eq.24]. (VIP-Seg's gating layer has one
[VIPSEG models/vipseg.py:190]; L1 wins.)" Confirmed: `models/vipseg.py:190` is
`nn.Linear(input_dim, num_steps)`, i.e. with bias. `02 §5.5` line 185 records Eq.23 as a plain dot product
with "no temperature, no normalisation [DECISION D-10]" - which is what Eq.23 prints; D-10 also documents
that the word "scaled" is unsupported by the formula. Correct.

**Gap:** `02 §6` does not record the Eq.24-vs-prose tension (F9). Everything else in the group is exact.

### (b) spec 02 -> code

* `models/adrm.py:159` - `Linear(dim, num_stages, bias=False)`. No bias, as specified.
* `models/adrm.py:163` - `softmax(self.w_g(f_q.mean(dim=1)), dim=-1)` -> `[B_q, T]`. Eq.24 with AvgPool
  over points.
* `models/adrm.py:170` - `einsum("bt,tbpc->bpc", w, stack(stage_logits))`. Eq.25.
* `models/adrm.py:157-158` - refuses `T < 2`, per D-17 (with one stage the softmax is identically 1).
* Cascade: `models/cascadeproto.py:132-140` - `P^0` copied per query, `T` stages in sequence, all `L^t`
  retained, then routing or `L^T`. Eq.22 plus D-17.
* Eq.23: `models/eppm.py:211` - `einsum("bpd,bcd->bpc", f_q, p)`, no scale. `logit_scale = sqrt_D` exists
  as an ablation only (`models/cascadeproto.py:141-142`).

### (c) paper -> code

Matches. The `T` stages have independent parameters (`models/cascadeproto.py:108-111`), which is D-16
item 8's reading of §3.5 - reasonable, but as under Eq.11 an inference rather than a statement.

---

## 9. Eq.26-28 - the training objective

### What the paper says

> "`L_total = L_seg + lambda . L_GMMN`" (Eq.26), "where `lambda = 1.0`." (§3.6)
>
> "The segmentation loss is the **standard cross-entropy** applied to the dynamically routed final logits
> `L_final`:
> `L_seg = -(1/N_q) sum_{i=1}^{N_q} log( exp(L_final[i, y_i]) / sum_{c=0}^{N} exp(L_final[i, c]) )`" (Eq.27)

**`lambda = 1.0`**, stated in §3.6 under Eq.26 (not in §4.1). The CE is **unweighted** ("standard"), over
all `N+1` classes including background (`sum_{c=0}^{N}`), averaged over the `N_q` query points, and
applied **only** to `L_final`. There is no per-stage or deep-supervision loss anywhere in the paper.

Eq.28 (§3.7) is the noise-contraction bound
`epsilon_t <= (1-eta)^t epsilon_0 + sum_{k=1}^{t} (1-eta)^{t-k} delta_k`. It defines `eta` only as "the
per-step contraction rate" and `delta_k` as "the residual perturbation at step k"; neither is connected to
any quantity in Eq.10-21, and no proof or condition is given ("Under mild regularity conditions"). It is
not implementable, and nothing in the repo claims otherwise.

### (a) paper -> spec 02, (b) spec -> code, (c) paper -> code

* `02 §7` lines 203-210: `lambda = 1.0`, unweighted CE on `L_final` only, "no per-stage loss, no class
  weights". Exact. The batch mean over 4 episodes is `[DECISION D-12]`, tagged.
* `pipeline/model_api.py:29-30` - `F.cross_entropy(logits.reshape(-1, n_classes), query_y.reshape(-1))`
  then `+ GMMN_WEIGHT * loss_gmmn`, `GMMN_WEIGHT = 1.0` at `pipeline/model_api.py:15`. Unweighted, all
  `N+1` classes, mean reduction. Exact.
* `train.py:134` - `torch.stack([episode_loss(...)]).mean()`, the batch mean of D-12.
* **F12 (cosmetic):** `pipeline/model_api.py:15` tags `lambda` as `[PAPER §4.1]`; it is §3.6 / Eq.26.

---

## 10. Section 4.1 - hyper-parameters

| Item | Paper, verbatim | Spec 02 / code | Verdict |
| :--- | :--- | :--- | :--- |
| `D` | "feature dimension `D = 128`" (§4.1, Eq.2) | `02 §0` line 17; `models/vipseg_backbone.py:15` | match |
| `T` | "the cascade depth is `T = 4`" | `02 §0` line 19; `models/cascadeproto.py:46` | match |
| `d` | "the cross-attention projection dimension is `d = 72`" | `02 §0` line 18; `models/eppm.py:54` | match |
| Learning rate | "AdamW at an initial learning rate of `10^-3`" | `train.py:71` default `1e-3`; `train.py:237` `AdamW` | match |
| Weight decay | "weight decay 0.1" | `train.py:72` default `0.1` | match |
| Scheduler | "a StepLR scheduler that **halves** the rate every 10 epochs" | `train.py:73-74` (`10`, `0.5`); `train.py:238` `StepLR` | match |
| Batch size | "The batch size is 4 episodes" | `pipeline/episodes.py:42` `EPISODES_PER_BATCH = 4` | match |
| Epochs | "trained for 50 epochs on S3DIS and 30 epochs on ScanNet" | `pipeline/episodes.py:39-40` | match |
| Points | "2048 randomly sampled points per block" (both datasets) | `02 §0` line 15; `models/vipseg_backbone.py:13`; enforced at `models/eppm.py:61-62` | match |
| CLIP width | "each LMA projects CLIP embeddings from `R^512` to `R^128`" | `models/lma.py:12-13` | match |
| Episodes per epoch | **not stated anywhere in the paper** | D-12: 480 (S3DIS) / 800 (ScanNet), chosen to hit VIP-Seg's 24,000 total | decision, correctly flagged |
| Evaluation episodes | "mIoU averaged over `S0` and `S1` across **600 randomly sampled episodes**" | D-08 prefers VIP-Seg's fixed 100 per class combination; `--eval_protocol random600` exists (`eval.py:23,36-37`) | documented deviation, both implemented |
| Normalising prototypes | **the paper says nothing** | D-10: none; ablation flag `l2norm_point_proto` | correct |
| Temperature / scale on Eq.23 | **the paper says nothing** - Eq.23 is called "scaled dot-product matching" yet prints no scale | D-10: none; ablation `logit_scale = sqrt_D` | correct |
| Hardware | "a single NVIDIA RTX 5090" | - | - |
| Optimiser extras | no betas, no warm-up, no gradient clipping stated | not implemented | correct (nothing invented) |

**Nothing in §4.1 mentions normalising prototypes or logits.** The only normalisations printed anywhere in
§3 are the `LN` inside the adapter (Eq.5) and the `LN` of Eq.21.

---

## 11. Tables 2, 4, 5 (and 6)

### Table 2 (S3DIS) - the rows that matter

Read from the rendered table on p.11:

| Row | 2w1s `S0` | 2w1s `S1` | 2w1s `Avg` | 2w5s `Avg` | 3w1s `Avg` | 3w5s `Avg` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| VIP-Seg [23] | 72.20 | 76.09 | **74.15** | 77.01 | 68.58 | 69.30 |
| CascadeProto (Audio) | 86.51 | 84.79 | 85.65 | 86.20 | 80.50 | **80.48** |
| CascadeProto (Image) | 87.13 | 84.68 | 85.91 | 86.36 | 79.95 | 79.31 |
| CascadeProto (Text) | **88.53** | 84.53 | **86.53** | **86.68** | **81.06** | 78.95 |

The prose checks out arithmetically: "surpassing the previous state-of-the-art VIP-Seg by **+12.38%** and
**+9.67%**" equals `86.53 - 74.15` and `86.68 - 77.01`. "CascadeProto (Image) attains the highest `S0`
score (**89.08%**) in 2-way 5-shot" and "CascadeProto (Audio) shows a notable advantage in 3-way 5-shot
(**80.48%**)" both match the table.

### What the baseline row is defined as, and the 81.28 vs 74.15 gap (F2)

§4.3 defines it:

> "Table 4 progressively ablates each component of CascadeProto, starting from **a plain VIP-Seg backbone
> with masked average pooling and single-step prototype matching as the baseline**."

That is `L = F^q P_point^T` - exactly the repo's `use_lma=false, num_stages=0` row (D-17,
`models/cascadeproto.py:130`). Table 4 gives it `S0 = 82.72`, `S1 = 79.83`, `Avg = 81.28`.

**I searched the whole paper for any statement that could explain why this exceeds VIP-Seg's own 74.15
(Table 2) and 72.20 `S0` (Tables 2 and 6). There is none.** The paper never says the baseline is
retrained, uses a different protocol, a different episode count, a different split, or an added
normalisation. The numbers are mutually inconsistent as presented:

* VIP-Seg, full model (4 reasoning steps, PEM+PDM, gating): 72.20 `S0` / 74.15 Avg.
* VIP-Seg backbone with *no* prototype module at all: 82.72 `S0` / 81.28 Avg.

That says removing VIP-Seg's entire prototype-reasoning stack improves `S0` by +10.52, while §4.2 spends a
paragraph arguing that prototype reasoning is what makes VIP-Seg the previous state of the art.

The repo's own harness shows the opposite ordering, as recorded in D-18 and the 2026-09-18 change-log
entry: VIP-Seg's released checkpoint scores 0.7197 through `eval.py --model vipseg` (paper: 72.20), while
the plain masked-average-pooling baseline scores 0.4416 (0.5218 with L2 normalisation) - about 29-37
points *below* Table 4's baseline row. Caveat: those runs use 2,400 training episodes, one tenth of
D-12's schedule, so they under-state a fully trained baseline; but they cannot be 30 points low, and they
do reproduce VIP-Seg's own published number in the same harness.

### Table 4 - component ablation (2-way 1-shot, text)

| LMA | Gate | Cascade (T=4) | ADRM | `S0` | `S1` | `Avg` |
| :---: | :---: | :---: | :---: | ---: | ---: | ---: |
| - | - | - | - | 82.72 | 79.83 | 81.28 |
| Y | - | - | - | 83.98 | 80.99 | 82.49 |
| Y | Y | - | - | 85.34 | 82.48 | 83.91 |
| Y | Y | Y | - | 87.89 | 84.05 | 85.97 |
| Y | Y | Y | Y | 88.53 | 84.53 | 86.53 |

Internally consistent: §4.3's increments +1.21, +1.42, +2.06, +0.56 are exactly the `Avg` differences, and
the last row equals Table 2's CascadeProto (Text) 2-way 1-shot and Table 5's `T = 4` column.

**A discrepancy the repo has not recorded (F6).** Table 4 row 3 ("+ Entropy Gate", no cascade) is 83.91
Avg, while Table 5's `T = 1` column is 83.28 Avg. D-17 maps row 3 to `num_stages = 1` and says Table 5
"varies `num_stages` in {1..6} with every other switch on" - so under D-17 these are the **same
configuration** and should be the same number. They differ by 0.63. Competing readings:

* (i) row 3 = a full EPPM with `T = 1` (D-17's choice) - contradicted by 83.91 != 83.28;
* (ii) row 3 = the gate only, applied to `P^0` before Eq.23, with no cross-attention and no diffusion -
  consistent with both tables, and a more literal reading of "cumulatively adds **one component**".

I lean to (ii). D-17 chose (i) without noting the conflict. The paper does not disambiguate, and the choice
changes what the "+1.42 entropy gate" increment is attributed to.

### Table 5 - cascade depth

| `T` | 1 | 2 | 3 | **4** | 5 | 6 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `S0` | 85.21 | 86.43 | 87.78 | **88.53** | 88.51 | 88.37 |
| `S1` | 81.34 | 82.67 | 83.91 | **84.53** | 84.50 | 84.31 |
| `Avg` | 83.28 | 84.55 | 85.85 | **86.53** | 86.51 | 86.34 |

Consistent with Tables 2 and 4 at `T = 4`. **F11:** §4.3 says "Table 5 reports performance **and inference
time**" and "inference time grows linearly with `T` at approximately **16ms** per stage", but the table
has no timing row and the 16ms figure has no source inside the paper.

### Table 6 - complexity, and the parameter-budget contradiction (F5)

| Method | Params (M) | FLOPs (G) | mIoU (%) |
| :--- | ---: | ---: | ---: |
| AttMPTI | 0.36 | 152.65 | 53.77 |
| QGPA | 2.79 | 16.30 | 56.30 |
| PAP3D | 2.57 | 15.05 | 59.45 |
| DPA | 5.08 | 15.67 | 66.08 |
| Seg-PN | 0.24 | 8.36 | 64.84 |
| VIP-Seg | 2.76 | 8.48 | 72.20 |
| **CascadeProto** | **2.88** | **8.86** | **88.53** |

The mIoU column agrees with Table 2's `S0` column for every row (PAP3D 59.45, Seg-PN 64.84, VIP-Seg
72.20), and §4.4's deltas check out (+16.33 over VIP-Seg, +22.45 over DPA, +29.08 over PAP3D, FLOPs
-41.1% against PAP3D). Internally consistent.

**But the budget cannot hold.** I counted the parameters of the modules exactly as Eq.4-6 and Eq.10-21
describe them, with D-16's minimal, conventional widths:

| Module | Parameters |
| :--- | ---: |
| Adapter: `Linear(512,128)` + `LN(128)` + `Linear(128,128)` | 82,432 |
| Generator `G`: three layers `256 -> 128 -> 128 -> 128` | 65,920 |
| One EPPM stage: `theta` 1, `phi` 4,608, `psi` 16,512, `f_fusion` 33,154, SE 8,352, `W_out` 16,512, `LN` 256 | 79,395 |
| Four EPPM stages | 317,580 |
| ADRM `W_g in R^{4 x 128}` | 512 |
| **Total** | **466,444** |

(This reproduces, independently, the 466,444 reported in D-09 and `01 §7`.) §4.4 says CascadeProto is "a
marginal overhead of **0.12M parameters** ... over the VIP-Seg backbone (2.76M, 8.48G, 72.20%)". Since
2.76M is the **whole** VIP-Seg model including its 0.19M prototype module, and CascadeProto replaces that
module, the paper implies about 0.31M for LMA + EPPM + ADRM. **Four EPPM stages alone are 0.32M**, before
the LMA (0.148M) and ADRM. Under the stricter reading of §4.4's own sentence (+0.12M net), **the LMA alone
already exceeds the entire stated overhead.** No choice of reduction ratio rescues this: `psi`
(`Linear(128,128)`, 16,512) and `f_fusion` (a "two-layer MLP" on a `2D`-wide input, at least 33,154) are
forced by the text, so a stage cannot cost much under ~50k, i.e. at least 0.2M for `T = 4`. D-09 rightly
declines to treat Table 6 as an acceptance criterion; this audit adds that the target is not merely
missed, it is **unreachable** from the equations as printed.

---

## 12. The `proto_self` question (VIP-Seg PEM) - what Eq.10-21 do and do not contain

### The VIP-Seg construct

`models/vipseg.py:266-278`, inside `PrototypeEnhancementModule.forward`:

```
que_G, sup_G = que.transpose(1,2) @ que, sup.transpose(1,2) @ sup   # [B,128,128], [3,128,128]
selfcor_q = self.reweight(que_G.unsqueeze(1)).squeeze(-1) / (128.**0.5)
selfcor_s = self.reweight_s(sup_G.unsqueeze(0)).squeeze()  / (128.**0.5)
proto_self_q = torch.sigmoid(selfcor_q) * new_proto                  # channel-wise gate on psi(P)
proto_self_s = torch.sigmoid(selfcor_s) * new_proto
proto_self   = self.fc_qs(proto_self_s) + self.fc_qs(proto_self_q)
proto_self   = self.layer_norm_qs(proto_self)
```

and the integration at `models/vipseg.py:304-305`:

```
output = self.fc(proto_cross + proto_self)
output = self.layer_norm(output + residual)
```

The decisive property of `proto_self` is that `sigmoid(selfcor) * new_proto` is an **element-wise**
product: output channel `i` depends only on channel `i` of `psi(P)`. It is a *channel-preserving* path.
`proto_cross` (`models/vipseg.py:296`) is the opposite: a `128x128` row-stochastic matrix times `psi(P)`,
so every output channel is a convex mixture of all input channels. VIP-Seg sums the two, so the module
always retains a channel-identity-preserving term alongside the mixing term. `PrototypeDifferenceModule`
has the same structure (`models/vipseg.py:380` `proto_self = torch.sigmoid(selfcor) * new_proto`;
`models/vipseg.py:400` `output = self.fc(proto_cross + proto_self)`), with `selfcor` computed from
`delta_G = que_G - sup_G` (`models/vipseg.py:372-379`) rather than from `que_G`/`sup_G` separately.

### Does CascadeProto's Eq.10-21 contain anything that plays this role?

**No. The text settles it three times over.**

1. **Eq.19 admits only two summands.** Verbatim: "A two-layer MLP fusion network `f_fusion` adaptively
   weights the **cross-attention and diffusion prototypes**:
   `w = softmax(f_fusion([P_cross; P_diffuse])) in R^2`, `P_combined = w_1 P_cross + w_2 P_diffuse`."
   The bracket, the `R^2` and the expansion all agree. There is no slot for a third term.

2. **The gate output `x_gated` is never added to anything.** Eq.12 reads, in full: "The gated feature is:
   `x_gated = x . g`." The symbol then **never appears again in the paper**. The only sentence that says
   what happens next is "**After entropy gating**, we apply cross-attention between the query and support
   to further refine the prototype" (§3.4, immediately before Eq.13) - which routes the gated quantity
   *into* the cross-attention, not around it. Nowhere does the paper write anything of the form
   `P_combined = ... + x_gated` or `P^t = LN(... + P_gated)`. **So: the entropy-gate output is not
   supposed to be added to the fused prototype; the paper only feeds it forward.**

   This deserves stating precisely, because `x_gated = x . g` with `x = P^{t-1}` is the *same shape of
   object* as `proto_self = sigmoid(selfcor) * new_proto`: an element-wise, channel-preserving gate on the
   prototype. The ingredient exists in CascadeProto - the paper simply never adds it back. Under D-02's
   reading, `P^{t-1}_gated` is computed and then consumed by `psi` and `A`, which destroy the channel
   identity that makes `proto_self` useful. Under the competing "gate the features" reading, the
   channel-preserving prototype term does not exist at all.

3. **The only channel-preserving path that survives is the plain residual of Eq.21**,
   `P^t = LN(W_out ReLU(P_weighted) + P^{t-1})` - an ungated, unweighted identity term. Eq.20's SE block is
   element-wise (`P_attended = P_combined . a`) and therefore channel-preserving in *form*, but its input
   is `P_combined`, which has already been mixed by `A`, and `a` is derived from `AvgPool(P_combined)`. It
   re-weights the channels of an already-mixed vector; it cannot re-inject `psi(P^{t-1})`.

**Verdict.** D-18's closing sentence - "VIP-Seg also carries a channel-preserving term
`proto_self = sigma(A_s) . psi(P)` [VIPSEG models/vipseg.py:270-274]; Eq.19 fuses only `P_cross` and
`P_diffuse`, so CascadeProto has no equivalent" - is **correct, and I could not find any text that
softens it.** The repo is right not to add `proto_self`: implementing it would mean implementing a module
the paper does not describe. The finding belongs in a report, which is where D-18 already puts it.

### Is Eq.15-18 meant to be PDM's analogue?

**Partly, by position; not at all, by construction.** The structural parallel is real: VIP-Seg alternates
PEM and PDM across its four reasoning steps (`models/vipseg.py:60-67`), and CascadeProto folds an
"enhancement" branch (Eq.13-14, PEM's cross-correlation) and a "difference/commonality" branch (Eq.15-18)
into each of its four identical stages. Both are described in the same language - VIP-Seg's PDM docstring:
"Learns common representations from differences between support and query feature distributions";
CascadeProto: "Channels active in both branches form the common activation mask ... `c_unique` ...
`P_diffuse`".

The mathematics is unrelated:

| | VIP-Seg PDM | CascadeProto Eq.15-18 |
| :--- | :--- | :--- |
| Input | projected feature **Gram matrices** `que_G`, `sup_G` (`models/vipseg.py:370`) | per-channel **means** of `F^q`, `F^s` |
| Difference | `delta_G = que_G - sup_G`, a `128x128` matrix (`models/vipseg.py:373`) | binary mask differences `m_q - m_common` |
| Prototype | multiplies `new_proto = psi(P)` (`models/vipseg.py:380`) | **never touches `P^{t-1}`** |
| Per class | yes (`new_proto` is `[B, N+1, 128]`) | **no class index at all** |
| Parameters | `reweight`, `proto_map`, `map`, `fc`, `LayerNorm` | **none** |
| Cross term | summed with `proto_cross` (`models/vipseg.py:400`) | Eq.19 softmax-weights the two instead of summing |

So Eq.15-18 occupies PDM's *slot* but keeps none of the properties that make PDM a prototype refiner: it
is parameter-free, class-blind and prototype-free. Combined with the absence of `proto_self`, the net
effect is that **every class-discriminative signal in a CascadeProto stage must pass either through
Eq.14's row-stochastic `A` or through the Eq.21 residual.** That is precisely the structural observation
D-18 reaches from measurement; reading the two implementations side by side confirms it from first
principles.

---

## 13. Things I could not verify, and why

1. **Tables 2-6 as experimental facts.** Nothing here reproduces or refutes the reported mIoU values; no
   training was run (instructed), and the repo's own runs are at one tenth of the D-12 schedule.
   Everything above about Table 4 is an *internal consistency* argument plus the repo's logged evidence.
2. **Figure 1's fine structure.** The figure's text extracts as a scatter of labels ("Audio: Whisper",
   "Text: This point cloud represents the chair.", "Image:", "Initial Multimodal Prototypes", "Cascaded
   Entropy-Aware Purification"). Its arrows are legible on the rendered page, but I cannot rule out that
   the figure encodes a wiring detail the extraction lost (for example whether `P_point` also feeds the
   LMA). D-13's reading of the Whisper -> CLIP arrow is consistent with the rendered page.
3. **Whether `theta` is per stage**, and whether **stage parameters are shared**. §3.5's "each EPPM applies
   entropy gating independently" is the only evidence, and it concerns the operation rather than the
   parameters. Unfalsifiable from the text.
4. **The softmax axis of Eq.14** and the pooling axis of Eq.20's `AvgPool`. Neither is printed; both are
   decided by D-01/D-16 on shape-consistency grounds, which is the only available argument.
5. **`eta` and `delta_k` of Eq.28.** Not connected to any other symbol in the paper, so the bound cannot be
   checked against the implementation.
6. **The 16ms-per-stage figure (§4.3)** - there is no table column to check it against.
7. **VIP-Seg's published 74.15** is taken from Table 2 of *this* paper. I did not consult the VIP-Seg paper
   itself, and the VIP-Seg logs cited in D-07/D-09
   (`log_s3dis_VIPSeg/log_S0_N2_K1_0.722026/...`) are not in the working tree, so that evidence is
   second-hand here.
8. **The Image and Audio modalities.** Not implemented (`models/cascadeproto.py:80-81` raises), so the
   Table 2/3 Audio and Image rows have no counterpart to audit.
9. **Whether the repo's parameter count would change** under different-but-equally-valid readings of the
   unspecified widths. The 466,444 figure is for D-16's choices; §11 argues the *lower bound* is still
   above Table 6's budget, but I did not search the space of readings exhaustively.
