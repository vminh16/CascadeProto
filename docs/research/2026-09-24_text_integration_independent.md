# Text in VIP-Seg's head: an independent analysis with CPU probes (phase 16, beyond the paper)

Status: 2026-09-24, written by the main session independently of the research agent's note
(`docs/research/2026-09-24_text_integration_math.md`), to be compared with it. Evidence: the repo's code,
CLIP ViT-B/16 (cached locally), P1's base-prototype banks and `experiments/t0_text_geometry.py` (CPU, run
twice with identical output). Tags: **[measured: T0-x]** from that script's section x, **[verified]** read
from code, **[inferred]** reasoning. No GPU was used.

## Summary

* **The existing text path is close to class-blind by construction, for three independent reasons.**
  1. The prompt template makes the text inputs nearly identical: the 13 prompts "This point cloud
     represents the X." have pairwise cosine 0.83–0.95 and 90 % of their energy in one common direction;
     the background prompt is at 0.92 from them [measured: T0-A].
  2. **L_GMMN carries no class identity at route B's scale.** For unit-norm prototypes the six RBF
     bandwidths (σ ≥ 2) are all in their quadratic regime, and the foreground MMD equals
     Σσ⁻² · ‖mean(P_modal^fg) − mean(P_point^fg)‖² to within 1–3 %; swapping the two text rows changes it
     by < 10⁻⁶ [measured: T0-B]. The "alignment" matches one foreground mean per episode.
  3. The adapter sees ~7 distinct inputs per fold (6 base names + background) and has no inductive bias
     that carries a novel name anywhere meaningful [measured: T0-D, see below].
  Together they predict what was measured: text +0.57 best / −0.75 last on the S0 baseline (phase 15),
  +0.38 in EDS-Net. The evidence against text so far tests a broken integration, not text.
* **What text can carry to a novel class in S3DIS is relational and weak-to-moderate.** CLIP similarity
  between class names predicts similarity between visual base prototypes only with bare names or a prompt
  ensemble (Spearman +0.18 to +0.47, p 0.05–0.23 over 15 pairs), not with the repo's template (−0.12 to
  +0.17) [measured: T0-C]. A text-weighted mixture of the other base prototypes predicts a held-out one
  better than the plain mean on every bank (S1 0.22 → 0.35, S0 0.41 → 0.72, ours 0.06 → 0.20) and stays
  well below the visual nearest neighbour (0.68 / 0.86 / 0.46) [measured: T0-C].
* **A learned text → feature map fitted on five names generalises on one fold and not the other**
  (ridge, leave-one-out: S0 0.41 → 0.66, S1 0.22 → 0.25) [measured: T0-D]. Retrieval by CLIP similarity
  does at least as well without learning a map, which is the safer inductive bias for 6 training names.
* **VIP-Seg's head processes each class row independently** (per-row linear maps, gates from that class's
  support slot and the query, LayerNorm on `output + residual`) [verified: `models/vipseg.py:255-311`].
  Text added to a row can only change that row, and the final LayerNorm makes the residual path
  scale-invariant: the gradient reaching the text branch through it is orthogonal to the row's current
  centred vector, so text can turn a prototype but not lengthen it [inferred from LN's Jacobian].
* **Recommendation, in order**: (1) fix the three flaws before any text claim (prompt ensemble, a
  class-discriminative alignment or none, no generator noise); (2) a *text-retrieved base-prototype prior*
  on the first prototype, zero-initialised gate, weighted per episode by its own support IoU; (3) a GPU probe
  on the E1 checkpoint that measures the upper bound of (2) before any 2.3-GPU-hour run. The honest risk:
  on S1 the relational signal is weak (0.22 → 0.35 in T0-C), so the gain may be small on that fold.

---

## 1. The current text path, audited

### 1.1 What enters: prompts [measured: T0-A; verified: `models/clip_text.py:22-29`]

| text | off-diagonal cosine (mean / min / max) | energy in the common direction | residual effective rank |
| :--- | :--- | ---: | ---: |
| repo template "This point cloud represents the X." | 0.896 / 0.834 / 0.945 | 0.904 | 11.2 |
| bare names | 0.776 / 0.651 / 0.857 | 0.794 | 11.4 |
| 6-template ensemble | 0.798 / 0.711 / 0.866 | 0.814 | 11.3 |

With unit-norm CLIP rows, the class-specific part of a template prompt has norm √(1 − 0.904) ≈ 0.31. The
adapter's first layer is linear, so its output is a large shared vector plus a small class term, and the
LayerNorm after it (`models/lma.py:28`) normalises across channels, not across classes: it does not remove
the shared part [verified]. The background prompt sits at 0.92 from the class prompts, i.e. the background
"class" gets nearly the same text as any foreground class [measured: T0-A].

### 1.2 What trains it: L_GMMN is a mean-matching term [measured: T0-B; verified: `loss/gmmn_loss.py`]

For ‖x − y‖² ≪ 2σ², exp(−‖x−y‖²/2σ²) ≈ 1 − ‖x−y‖²/2σ², and with that kernel the biased MMD of two sets is
exactly σ⁻² ‖x̄ − ȳ‖². Unit-norm prototypes have ‖x − y‖² ≤ 4 against 2σ² ≥ 8, so with Σσ⁻² = 0.303:

  L_GMMN ≈ 0.303 · ( 0.1 ‖P_modal^bg − P_point^bg‖² + ‖mean_c P_modal^c − mean_c P_point^c‖² ).

The approximation error is 1.2 % (median) and 2.9 % (max) over 200 random post-ReLU-like episodes, and
exchanging the two foreground text rows changes the loss by < 1e-6. **No term tells the adapter which
text row belongs to which visual prototype.** Its gradient w.r.t. P_point is the same vector for both
foreground rows, so with `gmmn_detach_point=false` it also pulls both point prototypes toward the same
point, a class-blind force on the backbone [inferred from the formula; the agent's note may quantify it].
(Without L2, as in phase 15's LMA run, prototype norms are larger and the kernel less linear; the conclusion
is only exact for route B.)

### 1.3 How it enters: P^0 = P_point + P_modal (Eq.9) [verified: `models/cascadeproto.py`]

With L2 prototypes, P_point has norm 1 and P_modal a free norm. The only class-specific training signal the
adapter ever receives is CE through P^0. Because 1.1 makes P_modal nearly the same for every class name,
the easiest use of it is a shared foreground offset: a common shift of the foreground rows that acts as a
foreground-vs-background bias and cannot move a foreground-vs-foreground decision (the gauge argument of
D-29's revision, restricted to the foreground rows) [inferred].

### 1.4 Six names are not enough to learn a map [measured: T0-D]

Ridge maps from CLIP to CL2N prototypes, fitted on five base classes, predicting the sixth (cosine; the
mean of the other five is the baseline):

| bank | template | bare | ensemble | mean of others |
| :--- | ---: | ---: | ---: | ---: |
| VIP-Seg S1 (ceiling, beam, column, chair, bookcase, board) | 0.13–0.21 | 0.22 | 0.23–0.25 | 0.216 |
| VIP-Seg S0 (floor, wall, window, door, table, sofa) | 0.45–0.67 | 0.46–0.62 | 0.49–0.66 | 0.408 |

On the S1 base classes no map beats the mean; on S0 it does. Whether text transfers depends on whether
the base classes contain semantic neighbours of the novel ones.

## 2. Where text can enter VIP-Seg's head

One PEM step for class row c (dropping the shot loop, K = 1) [verified: `models/vipseg.py:255-311`]:

  p̃ = W_p p_c + b_p;  g_q = σ(W_3 G_q / √128), g_s = σ(W_3' G_s,c / √128) (channel gates from Gram
  matrices of the max-pooled query and of class c's support slot);  self = LN_qs(W_qs(g_s ⊙ p̃) + W_qs(g_q ⊙ p̃));
  cross = softmax(A_qc) p̃ with A_qc the 128×128 channel correlation between query and slot c;
  out_c = LN(W_f(cross + self) + p_c).

Consequences for text [inferred]:

* **Rows never mix.** A text vector added to row c affects only row c. Text therefore cannot act as a
  relation between classes inside the head; any relational use (similarity to base classes) must be built
  before the head.
* **Scale.** The last LN makes out_c invariant to a common scale of its argument; the gradient that reaches
  an additive text term through the residual is orthogonal to the centred argument. Text changes the
  direction of a row. A text term whose norm is ≫ 1 would dominate p_c; ≪ 1 would be ignored — the
  norm has to be controlled explicitly (unit-normalise, then a learned scalar gate).
* **Zero-init reaches the base exactly.** P^0_c = normalise(P_point,c) + g · u_c with g a learned scalar
  initialised to 0 gives VIP-Seg's input at initialisation; ∂L/∂g = ⟨∂L/∂P^0_c, u_c⟩ is non-zero, so g
  moves only if u_c helps CE. The adapter producing u_c gets gradient g · ∂L/∂P^0_c, zero at step 0, the
  ReZero pattern. A per-episode weight computed without gradient (support IoU, 3.2) multiplies g.
* **Background row.** Its support slot is the mean of all supports and its text is nearly the class text
  (1.1); text on the background row has no causal content and should stay off.

## 3. Causal model

Class identity Y causes the support points (S), the query points (Q) and the class name (T). The model
predicts Q's labels from S; the question is what T adds given S.

* In 1-shot, S is one block: partial views, occlusion, clutter in the mask. T is independent of that
  particular block, so T's information is exactly the part of Y that one noisy block misses. Text should
  help most on episodes whose support is poor and least on those whose support is clean [inferred].
* T reaches the point-feature space only through what the training classes teach. For a novel class, the
  transferable part is its similarity to training classes (T0-C). A prior built as a text-weighted mixture
  of base prototypes, B̂_c = Σ_j softmax_j(τ cos(t_c, t_j)) B_j, uses CLIP's own geometry and the base
  prototypes' visual geometry, and learns no text → feature map from six names.
* **Pooling rule.** For two classifiers with log-likelihoods ℓ_S (support) and ℓ_T (text prior), log-linear
  pooling ℓ = ℓ_S + λ ℓ_T is Bayes-optimal when the two are conditionally independent given Y and λ
  matches their relative reliability. The reliability of ℓ_T in an episode is observable on the support
  itself: the IoU of the prior-only prediction against the support mask. That is MM-FSS's TACC weight, and
  it needs no gradient [inferred; MM-FSS to be checked against the agent's fetch].
* **Co-training condition.** D-28 failed because the base margin removed false and true mass at nearly the
  same rate (AUC 0.65–0.72). Text is a view whose errors come from name semantics, not from the query's
  geometry, so they are more likely to be uncorrelated with the support view's; that is testable (4, P-b).

## 4. What to measure before training (GPU, on the E1 checkpoint `log_r2/s3dis_S1_N2_K1_point_T4_vip_b1`)

* **P-a Upper bound of the retrieval prior.** For every S1 test episode: novel support prototype p_c, the
  text-retrieved B̂_c from a base bank of the same checkpoint (P1's machinery), cos(p_c, B̂_c) against
  cos(p_c, mean B); and the mIoU of `normalise(p_c) + g B̂_c` fed to the head for g ∈ {0.1, 0.3, 1} with the
  best g chosen on the valid draw. If the best valid gain is < +0.5, a trained version is unlikely to reach
  the 1.5–2 points one run can resolve; stop.
* **P-b Error independence.** Point-level errors of the head vs errors of the prior-only classifier
  (argmax over ⟨f, B̂_c⟩ and the background prototype) on test queries: the correlation of their error
  indicators and the fraction of the head's errors the prior gets right. Low correlation is the condition
  under which pooling helps.
* **P-c The gate.** Spearman between the prior's support IoU and the per-episode gain of adding it; TACC's
  premise, measured.

## 5. Ranked designs

1. **Fix the path first (no new idea, a precondition).** Prompt ensemble (or bare names) instead of the
   template; drop L_GMMN or replace it with a class-discriminative term (e.g. cosine between P_modal,c and
   P_point,c against the other class in the episode); unit-normalise the text term and gate it with a
   zero-initialised scalar; generator noise off (it only adds variance to 7 inputs). Measured on E1's
   schedule, S1, `best` and `last`. Evidence: 1.1–1.3.
2. **Text-retrieved base-prototype prior with a support-IoU gate.** P^0_c = normalise(p_c) + g · ω_c ·
   normalise(B̂_c) on foreground rows; B is an EMA bank of base prototypes kept during training (COSeg's
   BPC memory, P1's geometry), **excluding class c's own entry during training** so that training sees the
   same leave-one-out situation as testing; ω_c = IoU of the prior-only prediction on the support (no
   gradient); g learned, zero-init. Learns no text map. Evidence: T0-C, T0-D, section 3. Risk: weak on S1.
3. **Only if 2 transfers:** the adapter of 1 fed the retrieval weights as extra input (what text says about
   *which* base classes are relevant), trained with 2's gate. No evidence yet.

Decision rules to pre-register for design 2 (after P-a passes): `last` and `best` on S1 fixed100 and the
three random600 draws; go if the text arm beats E1 by ≥ +1.0 on fixed100 with a paired CI above 0 and on all
three draws, on both checkpoints; S0 only after that.

## Sources

* Code: `models/clip_text.py`, `models/lma.py`, `loss/gmmn_loss.py`, `models/cascadeproto.py`,
  `models/vipseg.py:255-404`, `experiments/t0_text_geometry.py`.
* Results: `results/phase15_full/SUMMARY.md` (LMA +0.57 / −0.75), `results/phase16_p1/SUMMARY.md` and its
  banks, `results/phase16_e1/SUMMARY.md`, `docs/research/2026-09-23_gap_and_upgrade_research.md` (MM-FSS
  TACC, EDS-Net +0.38, as quoted there; not re-fetched here).

## 6. Why EPPM failed: the paper's derivation, not our implementation [inferred from 02 §5 unless tagged]

* **Entropy gate (Eq.10–12)** takes the binary entropy of σ(P_i), a prototype *channel value*, not a
  probability. For post-ReLU prototypes (P ≥ 0) H falls monotonically with P_i, so g = σ(2(θ − H)) is a fixed
  monotone function of channel magnitude, bounded in [0.405, 0.731] (02 §5.1): it attenuates, never selects by
  uncertainty. The uncertainty that matters is the per-point class posterior, which P0 showed does rank
  reliable points (accuracy 88–95 % at w ≥ 0.9, 55–73 % at w < 0.5) [measured: `results/phase16_p0/SUMMARY.md`].
* **Diffusion (Eq.15–18)** has no class index and is broadcast to every row: a common shift, invisible to
  softmax (the gauge argument of D-29's revision) except through w_cls's 0.8 on the background row, i.e. a
  foreground/background bias. Its common/unique split is empty for non-negative features (D-14), and training
  switches it off by itself (fusion weight 0.008–0.082, D-18) [measured: D-18].
* **Fusion, SE (Eq.19–20)** are per query and shared by all rows; the ReLU of Eq.21 drops the sign of
  ψ-mixed prototypes. What remains class-specific is the residual plus the channel cross-attention, which is
  the same operation as PEM's cross term (`models/vipseg.py:285-296`).
* **What PEM has and EPPM lacks**: class-specific channel gates σ(W₃ G_q), σ(W₃′ G_s,c) from the Gram matrix of
  class c's own support slot, then LN (`models/vipseg.py:263-277`). That is the one path that tells row c how
  the query matches class c's support; EPPM replaced it with class-shared branches. R1's −15.5 (EPPM vs one
  PEM) and D-24's −3.8 (removing the inert parts) are what this predicts.
* **Implementation ruled out as far as it can be**: the equation audit found no code bug (2026-09-20), every
  ambiguity was run as a flag (D-01, D-02, D-14, D-18, D-19, D-23), and the flaws above are structural: they hold
  for every parameter value.
* **The idea re-derived on the right variable**: entropy-weighted pooling of the support head and a text prior,
  ℓ_i = ℓ_i^head + κ γ_e w(H_i) (T_i − mean T_i) with H_i the normalised entropy of the head's posterior at
  point i. Text is trusted where the head is unsure; w ≡ 1 is the agent's T1.

## 7. Comparison with the research agent's note (`2026-09-24_text_integration_math.md`)

| point | this note | agent | verdict |
| :--- | :--- | :--- | :--- |
| GMMN at unit norm | 0.303·‖mean diff‖², row swap < 1e-6 | same coefficient; fg-row gradient cosine 0.9997 | agree, two probes |
| prompt geometry | template cos 0.83–0.95, 90 % common energy | mean cos 0.895, 88.5 % template | agree |
| text ↔ prototype similarity (Spearman, 15 pairs) | template −0.12 / +0.17 / −0.11; ensemble +0.29 to +0.36; bare names +0.18 to +0.47 | repo prompt −0.16 to +0.17; 8-template +0.09 to +0.29; descriptions −0.49 to +0.21 | agree prompt for prompt; nothing significant after multiple comparisons |
| map from few names | ridge on 5 names: S0 0.41 → 0.66, S1 none | ridge on 4 names, held-out 2-way: 0.58 / 0.60 / 0.72 (repo / ensemble / descriptions) | agree: weak, fold-dependent, better with richer text |
| where text can enter the head | rows independent; zero-init gate at p⁰ | rows independent; p⁰, channel gates or logits only | agree |
| training through p⁰ | EMA bank without the class's own entry | shortcut: needs text dropout and leave-out maps | agent sharper; adopt its safeguards |
| first experiment | GPU probes P-a..P-c, then a trained prior at p⁰ | training-free logit prior T1 with a support-accuracy gate, one E1 evaluation | adopt T1 as the first step, with this note's entropy weight as an extra arm |
| text source | retrieval (softmax over CLIP similarity to base names) | closed-form ridge map from the banks | test both in the same probe |
| expected effect | small on S1 | 0 to +0.5 (repo prompt), ≈ +1 (descriptions) | agree |

## 8. Is 80 mIoU reachable, and would a learnable multimodal neck get there? (maintainer's question, 2026-09-24)

### 8.1 The ceiling is not the backbone [measured: `results/phase16_e1/SUMMARY.md`, `results/phase16_r2_pre/`]

| S1 fixed100 | mIoU |
| :--- | ---: |
| E1 `last` / `best` | 73.20 / 75.05 |
| VIP-Seg released (best of 12) / its own training-log test | 75.36 / 74.65 |
| **oracle rule on E1's features** (query's own class means; norms kept / equal) | **87.43 / 85.93** |
| oracle rule on VIP-Seg's features | 83.77 / 86.31 |

With the right prototypes the same frozen features score 86–87: the encoder separates the test classes
well. What limits 73–75 is estimating each class's prototype from one support block (the support → query
shift), not the features. 80 is therefore not information-theoretically out of reach on S1; it is +7 over
E1 `last`, i.e. recovering about half of the transductive gap without query labels. No measured method in
this repo moves any of that gap (D-26…D-29: ≤ +0.37), and the best published S1 number, selection
included, is 76.09. A pretrained backbone would violate guardrail 1, and where it was measured (COSeg on
MM-FSS's 2D-aligned backbone) it added +1.12 on S3DIS 2-way 1-shot [verified: repo note 2026-09-23 §6].
**Verdict: 80 is not ruled out by the features, but no evidence-backed path reaches it; the backbone is not
the thing to replace.** [inferred]

### 8.2 A learnable neck that fuses modalities before the head

What "multimodal features" exist here: per point only xyzrgbXYZ (the blocks carry no image
correspondence); per class only a name. A neck fusing the two before the head is F′_i = F_i + α·N(F_i, t_c),
which is class-conditional: the query would need one feature copy per class hypothesis (as text-query
attention does), and the neck learns from **six** (text, visual) anchor pairs per fold.

* **Information argument.** Any map, linear or not, is pinned by the data only at the six anchors; a novel
  name's CLIP embedding is 88–90 % the shared template (T0-A; agent §1.4) and reaches the point space only by
  interpolating between base anchors. T0-C/T0-D and the agent's held-out assignment (0.58 / 0.60 / 0.72)
  measure exactly that interpolation: weak on S1, better on S0 and with descriptions. A neck adds capacity,
  not information about novel names; it inherits this bound and adds the shortcut of memorising six names
  (agent §2.3). [inferred, with the cited measurements]
* **Capacity argument.** CE on base classes is already near saturation (E1 training loss ≈ 0.12–0.15,
  `results/phase16_e1/train_e1_S1.log`) while novel-class mIoU is 73: the problem is generalisation, which
  more trainable parameters between encoder and head do not address by themselves. [measured / inferred]
* **Gradient argument.** A neck changes the features of support and query at once, so the trained head's
  equilibrium moves; only a zero-initialised residual F + α·N(·) with α = 0, warm-started from E1, starts at
  the base. That is T3 of the agent's note applied to features instead of PEM's gates. [inferred]

**Verdict on the neck as proposed (text + point features, trained on the few-shot episodes): not expected
to give more than the text prior itself (0 to +1), with more ways to overfit.** It is kept as a variant of
T3, behind T1.

### 8.3 The variant that is feasible in principle: a neck distilled from dense 2-D vision-language features

MM-FSS's gain comes from per-point features aligned with a 2-D open-vocabulary model over *every* point
(dense supervision, many implicit concepts), not from class names. The same idea here would be:

* data: S3DIS's original 2D-3D-S images and poses, projected onto the blocks to give each point a
  vision-language feature from a frozen 2-D model (CLIP-based, e.g. an open-vocabulary segmenter);
* neck: a zero-init residual head on our encoder features, trained to regress those per-point features on
  **all** points of the training areas (no class labels involved), then fused with the point features
  before VIP-Seg's head; text then matches these features directly, with no six-name map;
* constraints and cost: guardrail 1 forbids pretrained point-cloud weights, not frozen 2-D models, but
  the paper and VIP-Seg state "no pretraining", so it needs a new decision and a separate comparison table;
  the 2-D projection pipeline is new work (days, not hours), and the test areas' images must not enter
  training; expected effect unknown on S3DIS (MM-FSS reports its multimodal gains on its own backbone).
* cheapest feasibility test before building it: check that 2D-3D-S images and camera poses exist for the
  areas of our blocks and that a point-to-pixel projection covers most points of a block.

Recorded as the only neck design with an information source beyond six names; not started.

## 9. Where the error sits, and a neck that opens the head's missing information path (2026-09-24)

### 9.1 Features, prototype, decoding: which step is wrong [measured: E1, R2-pre]

`L = F^q M_effᵀ`, argmax. Replacing only M_eff by the query's own class means lifts E1 from 73.20 to
85.9–87.4 with the same features and the same dot-product decoding. So (a) the features separate the novel
classes, (b) the decoding rule works when the prototype is right, and (c) **the error is the prototype the
support + head produce**, a biased estimate of the query's class centre (one support block, support → query
shift). The prototype does not "extract the novel class well": it is the step that fails.

### 9.2 Only three sources can correct a prototype [inferred]

The query's class centre μ_q,c can be informed by (i) the support block (already used), (ii) the query's
own unlabelled points (transduction), (iii) prior knowledge (base classes, class names). D-31 tests (iii).
(ii) holds the whole oracle gap by definition; fixed rules on it failed (D-26…D-28), but a *learned*
query–support interaction is what makes VIP-Seg's head 15.5 points better than EPPM (R1): learned
transduction is where the measured gains of this family come from.

### 9.3 What VIP-Seg's head cannot see [verified: `models/vipseg.py:238-311`]

* PEM/PDM max-pool 2048 points to 64 tokens (MaxPool 32) and act through 128×128 channel Gram and
  correlation matrices: **no point-level correspondence between query and support** reaches the prototype.
* Each class row is processed alone: classes never compete inside the head (only at the logits).
* The prototype entering the head is one masked mean of one support block; its composition bias (which
  parts of the object the block shows) is never corrected with point-level query evidence.

### 9.4 The neck this suggests (a design to research, not yet decided)

A zero-initialised residual **point-level query–support cross-attention neck** before the prototype and the
head: query points attend to support points (and vice versa) on the 128-d features, producing
F′ = F + α·Attn(F, F_other) with α = 0 at init and warm-start from E1, so the base is reached exactly. It
gives the head the point-level correspondence it lacks and targets source (ii), the measured bottleneck,
rather than a six-name text map. Precedents in few-shot 3-D segmentation use point-level query–support
interaction (AttMPTI's label propagation, QGE, QGPA), under the old protocol with its foreground leak, so
their sizes are not evidence here [verified: repo note 2026-09-21 external sources]. Risks: a learned
transduction can still fit base-class episodes only; memory of 2048×2048 attention per query (≈ 16 MB
float32 per head, fine on an L4). Cheapest evidence before building it, on E1 (one GPU pass): the mIoU of
the plain support-prototype rule (no head) against E1 and the oracle, which splits the gap into what the
head already recovers and what remains; and the oracle gap by point type (boundary vs interior, small vs
large support mask) to see whether it is concentrated where point-level correspondence would help.

### 9.5 Other modalities [inferred]

The paper's image and audio inputs are one item per class, like its text (image → CLIP, audio → Whisper →
CLIP; 03 §1, [PAPER Fig.1]): per-class priors learned from six base classes, with the same anchor limit as
text (§8.2) and no per-point information. The modalities that do carry new per-point information here are the
point cloud's own streams (geometry and colour, fused early in the 9-channel input) and, outside the current
data, the 2-D images of 2D-3D-S (§8.3). Neither has evidence of being under-used yet.
