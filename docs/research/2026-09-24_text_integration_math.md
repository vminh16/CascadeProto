# Text on VIP-Seg's head: audit, gradient flow, causal model, and designs that cannot start below the base

Status: 2026-09-24, phase 16, route B (D-25, D-30). Desk analysis plus CPU probes: the repository, the E1 checkpoint
(`log_r2/s3dis_S1_N2_K1_point_T4_vip_b1/last.pt`, weights and config only, no features), the P1 base-class banks
(`results/phase16_p1/bank_*_1000.pt`), the locally cached CLIP ViT-B/16 text encoder, and primary sources fetched from
arXiv HTML. No GPU run, no code change. The probe scripts lived in the session scratchpad and are not committed; §1.4
describes them well enough to rerun. Tags: **[measured: …]** a number read from a file or computed here on CPU,
**[verified: …]** read from code or a primary source, **[inferred]** a derivation or prediction.

## Summary

* **The existing text path has design flaws that explain a near-zero effect. It never had a fair test.** (a) Its
  scale is not controlled: `P^0 = normalise(P_point) + P_modal` feeds VIP-Seg's head a prototype that is no longer
  unit norm, and PEM is not scale-equivariant in it. (b) At unit-norm prototypes the GMMN kernel (σ ≥ 2) is in its
  linear regime, so `L_GMMN ≈ 0.303‖mean P_modal^fg − mean P_point^fg‖² + 0.030‖Δbg‖²`. That compares the *means*
  of the two foreground sets and ignores which name belongs to which class. Its gradient is the same for every
  foreground point prototype (cos 0.9997). (c) At initialisation the generator's noise is 7.6× the door/window text
  difference, and z = 0 at test differs from the training mean by 0.36 [measured: CPU probe, §1.3].
* **Six training names cannot carry a learned map to novel names.** CLIP prompts of the 12 classes have mean cosine
  0.895, and 88.5 % of a novel prompt's energy is the shared template direction. Only 10 % of the name-specific part
  lies in the span of the 7 training prompts, and gradient descent never updates a linear first layer outside that
  span. Kernel-ridge maps from CLIP to real point prototypes, fitted on 4 classes, assign held-out class pairs
  correctly **58 % of the time** with the repo's prompt (chance 50 %). The rate is 60 % with an 8-template ensemble
  and **72 %** with geometric descriptions. The similarity structure of CLIP and of the point prototypes does not
  correlate (Spearman −0.49 to +0.29, no permutation p < 0.15) [measured: CPU probe on the P1 banks, §1.4].
* **VIP-Seg's head has one mask-bearing input, the initial prototype.** PEM/PDM act on each prototype row
  separately, and there is no attention across rows. Their gates and cross-correlations read the *unmasked* support
  block. So an extra "text row" (design ii) does not exist in this head. Only p⁰, the channel gates or the logits can
  take text without replacing a module [verified: `models/vipseg.py:244-305,347-401`].
* **In training, the name identifies the class: it is a shortcut.** With 6 base names, a text branch trained by CE
  through p⁰ receives exactly the gradient the support prototype receives. It can memorise base-class prototypes and
  make the support redundant on training episodes, then shrink novel prototypes toward a class-agnostic vector at
  test. Any trained design needs text dropout and a text map that has not seen the episode's classes [inferred, §2.3].
* **Optimal fusion (§3.2).** For two Gaussian margins with means μ_s and μ_t, unit variances and error correlation ρ,
  the best linear fusion gains (μ_t − ρμ_s)²/(1 − ρ²) in squared SNR over the support alone. A fixed 1:1 weight
  *loses* whenever μ_t < μ_s(√2 − 1) at ρ = 0. MM-FSS shows exactly this pattern: text alone is 7.7 points below its
  no-text model, yet adding text lifts it +1.9 with an episode-adaptive weight and +0.9 at 1:1 [verified: arXiv
  2410.22489 Table 3(e)].
* **In this repo the text "view" is not a second sensor.** Its classifier ⟨f_i, t_c⟩ is linear in the same features
  as the support classifier ⟨f_i, M_c⟩. Its value is therefore bounded by how well t_c − t_c′ points along the oracle
  direction O_c − O_c′ *beyond* M_c − M_c′. That is the transductive headroom D-26…D-29 could not close. It is
  measurable on E1 with one evaluation pass and no training (§4.3).
* **Re-examine every text conclusion so far.** LMA's +0.57 best / −0.75 last came from one seed, on the headless
  baseline without L2, at 6,000 updates. That is inside the 1.3–1.9-point selection swing, and E1 showed this
  schedule under-trains the head. Text has **never** been measured on VIP-Seg's head [measured:
  `results/phase15_full/SUMMARY.md`, `results/phase16_e1/SUMMARY.md`].
* **Recommendation (ranked, §5).** T1 is a training-free, post-head, support-validated text logit prior, with a
  closed-form text map fitted on base-class prototypes that leaves out the episode's classes. It is exactly E1 at
  κ = 0. Its probe is one GPU evaluation of E1 `last.pt`. T2 is a warm-started, zero-init residual text prior at p⁰
  with text dropout, run only if T1's probe passes. T3 is a text FiLM on PEM's channel gates, conditional, with a
  mask-only control. **Prediction: 0 to +0.5 with the repo prompt, and up to about +1 only if descriptions carry
  class information** [inferred]. The honest risk: six class names give S3DIS text little that the support does not
  already carry (§5.4).

---

## 1. Audit of the current text path

### 1.1 What the code computes

`E_CLIP = normalise(CLIP_text("This point cloud represents the {name}."))`, with a fixed background sentence
[verified: `models/clip_text.py:89,95-102,155`]. `P_modal = G([A(E_CLIP); z])`, where A = Linear(512→128) → LN →
ReLU → Dropout → Linear, and G is a 3-layer MLP on 256 inputs. z ~ N(0, I) in training and z = 0 in evaluation
[verified: `models/lma.py:29-30,38-39,66-74`]. `P^0 = P_point + P_modal`, with the L2 norm applied to `P_point`
*before* the sum when `l2norm_point_proto` is set. `L_GMMN` uses the un-detached `P_point`, and λ = 1 [verified:
`models/cascadeproto.py:164-172`, `pipeline/model_api.py:386-387`]. E1, the route-B base, runs with
`l2norm_point_proto = True, use_lma = False` [measured: config stored in E1 `last.pt`].

### 1.2 (a) Scale: the head's contract is a unit-norm p⁰

VIP-Seg normalises its initial prototypes [verified: `models/vipseg.py:142`], and E1 was trained on ‖p⁰‖ = 1. In a
PEM (formulas in §2.1) p enters three places: the residual `+p` inside the output LN, the cross branch `A·(W_pm p + b)`,
and the self branch `LN_qs(W_qs(σ ⊙ (W_pm p + b)))`. The self branch is scale-invariant in p except through the bias
b, while the other two scale with ‖p‖. Scaling p therefore changes the *mix* of the head's branches, not only the size
of its input [inferred from `models/vipseg.py:259,277-278,296,304-305`]. `P_modal` has no norm constraint. At
initialisation ‖P_modal(z=0)‖ ≈ 0.71 [measured: CPU probe, 20 random inits], comparable to the unit `P_point`. After
training, GMMN only ties its mean to `P_point`'s mean (1.2(b)). **Fix used in every design below:** text enters as
`p⁰ = normalise(normalise(p) + α t̂)`, so that ‖p⁰‖ = 1 whatever α and t.

### 1.2 (b) GMMN: in the linear-kernel regime it matches means, not classes

For k(x, y) = Σ_σ exp(−‖x−y‖²/2σ²) with σ ∈ {2, 5, 10, 20, 40, 80} [verified: `loss/gmmn_loss.py:167,182-185`] and
‖x − y‖² ≪ 2σ², expand k ≈ 6 − b‖x − y‖² with b = Σ_σ 1/(2σ²) = 0.1516. For the biased estimator
[verified: `loss/gmmn_loss.py:188-194`] the constants cancel:

  MMD²(X, Y) ≈ b·(2·mean‖x−y‖² − mean‖x−x′‖² − mean‖y−y′‖²) = 2b ‖x̄ − ȳ‖².

So `L_GMMN ≈ 0.030‖t_bg − p_bg‖² + 0.303‖mean_c t_c − mean_c p_c‖²` [inferred, algebra]. Unit vectors have
‖x − y‖² ≤ 4 < 8 = 2σ²_min. For random unit prototypes the approximation's median relative error is 11 %, and the
gradients on the two foreground point rows have median cosine **0.9997** [measured: CPU probe, 200 draws, the
repository's `gmmn_loss`]. Consequences:

1. **Permutation blindness.** Swapping the text rows of the two ways leaves the loss unchanged. Nothing in
   `L_GMMN` says which name goes with which class; only CE does, through p⁰.
2. **A class-independent pull on the backbone.** With `gmmn_detach_point = false`, every foreground point prototype,
   and through Eq.3 every support foreground point feature, receives the same gradient 2b(p̄ − t̄)/N. This does not
   separate classes. It moves the foreground mean relative to the background, the quantity the fg/bg decision uses
   [inferred]. D-04 already notes the estimate is noisy for N = 2 (`00_SOURCES_AND_DECISIONS.md:131`). The point
   here is stronger: at this scale it is a first-moment penalty, not a distribution distance.
3. The logged fall of `L_GMMN` from 0.47–0.84 to 0.008–0.026 [verified: `docs/research/2026-09-22_improvement_directions.md:201-202`]
   only says that the mean text prototype approached the mean point prototype.

### 1.3 (c) Noise z: low signal-to-noise and a train/test gap

In G's first layer the text part `W_e e` and the noise part `W_z z` have per-unit standard deviations 0.17 and 0.41 at
PyTorch's default init. With real CLIP inputs (door, window) the **text difference between the two ways is
‖P_door − P_window‖ = 0.075, while the per-row noise spread is 0.57**. The mean over z differs from the z = 0 output
by 0.36, a Jensen gap: E[ReLU(a + sξ)] − ReLU(a) = sφ(a/s) − |a|Φ(−|a|/s), which is 0.4s at a = 0
[measured: CPU probe, 2,000 draws × 20 inits; inferred, algebra]. CE gains nothing from z: there is no adversary,
and in the linear regime of 1.2(b) the noise only adds E‖z̄‖² to `L_GMMN`. So training can only shrink `W_z`, and
until it does, the gradient into A is buried in noise. Whether a trained LMA had shrunk `W_z` cannot be checked
locally; no LMA checkpoint is in the repository. **Every design below drops z** (D-06's z = 0 then holds at train and
test).

### 1.4 (d) Six names: what a map fitted on 7 anchors can do on a novel name

*Argument.* Train a linear first layer W by gradient descent from W₀ on the inputs {e_1, …, e_7} (6 base names plus
the background sentence). Each update is a sum of outer products δ e_iᵀ, so W = W₀ + Σ_i a_i e_iᵀ. On a novel name,
W e_new = W₀ e_new + Σ_i a_i (e_i·e_new): a kernel regression with the CLIP cosine kernel plus the random init
[inferred, standard implicit-bias argument]. Weight decay 0.1 shrinks W₀, so the novel output tends to a
cosine-weighted mix of the base outputs. The adapter's LN, ReLU and the generator do not change this in kind: they
act on W e_new, whose name-specific part outside span{e_i} is carried only by W₀.

*Measurements* (CLIP ViT-B/16, prompts built as in `models/clip_text.py`) [measured: CPU probe]:

| quantity | repo prompt | 8-template ensemble | geometric descriptions |
| :--- | ---: | ---: | ---: |
| mean off-diagonal cosine of the 12 class prompts | 0.895 (0.834–0.945) | 0.814 | 0.787 |
| background sentence vs class prompts | 0.918 | 0.793 | — |
| share of a novel prompt's energy in the 7-anchor span (S1 / S0) | 0.897 / 0.892 | 0.814 / 0.811 | — |
| share of the *centred* (name-specific) part in the centred anchor span | 0.100 / 0.106 | 0.102 / 0.094 | — |
| leave-two-out 2-way assignment, ridge map on 4 classes (4 banks, mean) | **0.58** | 0.60 | **0.72** |
| per bank: VIP-Seg S0 / VIP-Seg S1 / ours S0 / ours S1 | 0.73 / 0.40 / 0.67 / 0.53 | 0.80 / 0.47 / 0.73 / 0.40 | 0.73 / 0.67 / 0.87 / 0.60 |
| Spearman(CLIP cos, prototype cos) over 15 pairs, per bank | +0.17 / −0.11 / −0.16 / −0.13 | +0.26 / +0.27 / +0.09 / +0.29 | +0.21 / −0.49 / +0.06 / −0.41 |

*Probe.* Each P1 bank holds 6 base-class prototypes of one checkpoint, as centred unit means over 1,000 training
episodes [verified: `models/base_calibration.py:32-46,91-96`]. For each of the 15 class pairs (h, g), a kernel ridge
map CLIP → prototype (λ = 10⁻³, linear kernel) is fitted on the other 4 classes. It predicts t_h and t_g, and the pair
counts as correct if cos(t_h, p_h) + cos(t_g, p_g) > cos(t_h, p_g) + cos(t_g, p_h). Spearman p-values come from all
720 relabellings; none is below 0.15. The 15 pairs of a bank are not independent (11/15 would give p ≈ 0.06 if they
were). The descriptions ("a large horizontal flat surface at the bottom of a room", …) were written once, before any
result with them, and never iterated. The probe has 4 anchors, not the 7 of training, and the banks are
feature-space means, not the head's internal space, so it is a proxy [inferred].

*Reading.* With the repo's prompt, a map fitted on base names places a novel name's prototype barely better than
chance in a 2-way decision. Its cosine to the true prototype is not reliably above that of the centroid of the
training prototypes: the ridge-minus-centroid gain ranges from −0.08 to +0.20 across banks [measured]. This is the
mechanism behind "text shrinks novel prototypes toward a class-agnostic vector" (§2.3). Descriptions add geometric
attributes (horizontal/vertical, planar, furniture) shared between base and novel classes. That is the classical
condition for attribute transfer, and the only variant that reaches 0.72 [measured; reading inferred].

*What helps generalisation* [inferred]. The novel output is a function of the projection onto the anchors, so what
matters is: (1) the input geometry, where centring (subtracting the template mean) removes the 88 % shared energy;
(2) the rank and Lipschitz constant of the map, since a low-rank map of small norm cannot memorise 7 points and a
frozen closed-form ridge map has no free parameters for CE to exploit; (3) anchor count, which descriptions and
prompt ensembles increase in *text* space but not in *class* space. A map fitted on class *prototypes* (the bank) in
closed form beats a map trained by CE through the head, because CE also rewards memorising base-class corrections.

### 1.5 (e) Background prompt, and (f) other findings

* The background sentence has cosine 0.918 with the class prompts, and background in a training episode contains
  other base classes, novel classes and clutter (class 12 is never a way; `dataloaders/s3dis.py:28-29`). Its
  P_modal row is therefore a constant learned bias on p_bg, not text [inferred]. The phase-15 +LMA row raised point
  accuracy by 3.5 (72.81 → 76.35) while foreground mIoU moved by 0.57, which fits a background-bias effect
  [measured: `docs/research/2026-09-21_reproduction_report.md` §3.1 table; reading inferred]. Designs below leave the
  background row to the support.
* The dtype path is sound: fp16 CLIP output is cast to float32 before the norm, then to f_q's dtype
  [verified: `models/clip_text.py:155`, `models/cascadeproto.py:168`]. The name order matches the prototype rows
  [verified: `models/clip_text.py:100-102`, `pipeline/episodes.py:105`]. No bug found.
* Parameter count: LMA alone has 148,352 parameters (report §2.3), fitted on 7 distinct inputs, with no mechanism
  preventing memorisation [verified: `docs/research/2026-09-21_reproduction_report.md:142-143`].

---

## 2. Gradient flow through VIP-Seg's head and where text can enter

### 2.1 One PEM and one PDM step

Per query b and prototype row c (c = 0 background). Q̃_b = MaxPool₃₂(F^q_b) ∈ R^{64×128}. S̃_c = MaxPool₃₂ of the
**whole** support block of way c, or of the mean of the N blocks for c = 0. Then q_b = W_map Q̃_b and
s_c = W_map S̃_c ∈ R^{72×128} [verified: `models/vipseg.py:244-258`]. With u = W_pm p_{bc} + b_pm:

* PEM: σ^q_b = sigmoid(w_qᵀ q_bᵀq_b/√128), σ^s_c = sigmoid(w_sᵀ s_cᵀs_c/√128) ∈ R^128;
  A_{bc} = softmax_row(C_{bc}/√128) ∈ R^{128×128} from q, s (with VIP-Seg's cross-query reshape);
  h = LN_qs(W_qs(σ^s_c ⊙ u) + W_qs(σ^q_b ⊙ u)); **p′ = LN(W_fc(A_{bc} u + h) + p)**
  [verified: `models/vipseg.py:266-305`].
* PDM: σ^Δ_{bc} = sigmoid(w_Δᵀ(q_bᵀq_b − s_cᵀs_c)/√128); p′ = LN(W_fc(A_{bc} u + σ^Δ ⊙ u) + p), then the outer
  residual p′ + p [verified: `models/vipseg.py:370-401,156`, `models/vip_stage.py:314-315`].
* Readout: L^t_{bic} = ⟨f_{bi}, P^t_{bc}⟩ and L_final = Σ_t w_{bt} L^t with w_b = softmax(W_g mean_i f_{bi})
  [verified: `models/cascadeproto.py:188-190`, `models/adrm.py:344`].

Structural facts [inferred from the formulas above]:

1. **Rows never interact.** Row c is transformed by operators that depend on F^q_b and support block c only.
2. **The support mask reaches the head only through p⁰.** σ^s, σ^Δ and A are built from unmasked blocks, so which
   support points are the class, and which are background, is known to the head only via the masked means
   [verified: `models/prototypes.py:255-261`]. Every stage re-reads p through W_pm.
3. **Output scale.** Each stage ends in a LayerNorm with learned γ of mean 0.23–0.29, so ‖P^t‖ ≈ ‖γ‖ = 2.9–3.5 after
   every stage, while p⁰ is unit norm [measured: E1 `last.pt`, `stages.*.module.layer_norm.weight`].

### 2.2 Gradients

Write π = softmax(L_final) and n = B_q·2048. The direct gradient on stage t's prototype is
δ^t_{bc} = (w_{bt}/n) Σ_i (π_{bic} − y_{bic}) f_{bi}: a residual-weighted sum of query features. Into p⁰ it flows
through the stage Jacobians

  J_t = J_LN · (W_fc (A_{bc} + J_LNqs W_qs diag(σ^s + σ^q)) W_pm + I)   (PEM; PDM analogous, plus I for the outer residual),

where J_LN = (γ/std(x))(I − 11ᵀ/128 − x̂x̂ᵀ/128). The identity inside every J_t, plus the outer residuals, keep the
gradient to p⁰ from vanishing across 4 stages. W_fc W_pm has spectral norm 1.3–12.7 in E1, and LN divides by the
pre-norm scale, so nothing explodes [measured: E1 weights; inferred]. Gradients reach the gate and correlation
parameters only through u, that is, multiplied by p.

### 2.3 The five entry points

| entry | exact parametrisation | identity at init | gradient into the text branch | shortcut risk |
| :--- | :--- | :--- | :--- | :--- |
| (i) additive prior on p⁰ | p⁰_c = n(n(p_c) + α t̂_c), α = 0 at init (scalar or per-row α·γ_e) | exact: n(n(p)) = n(p) | ∂ℓ/∂α = Σ_c ⟨J_nᵀ g_c, t̂_c⟩, J_n = (I − p̂⁰p̂⁰ᵀ)/‖·‖; ∂ℓ/∂t_c = α J_nᵀ g_c | **high**: t_c gets the support prototype's own gradient g_c; with 6 names it can memorise base prototypes |
| (ii) extra text row | none without a new cross-row module (fact 1) | — | — | — (would replace the head) |
| (iii) FiLM on channel gates | σ^s_c ← σ^s_c ⊙ (1 + β tanh(H ẽ_c)), β = 0 at init, H ≠ 0 | exact | ∂ℓ/∂β = Σ ⟨∂ℓ/∂(σ^s⊙u), σ^s ⊙ tanh(Hẽ_c) ⊙ u⟩; H needs β ≠ 0 first | medium: class-specific channel choice from a name |
| (iv) logit prior | L′ = L_final + κ γ_e (T − mean_{c≥1} T) on foreground columns, T_{bic} = ⟨n(f_{bi} − μ), t̂_c⟩, κ = 0 at init | exact | ∂ℓ/∂κ = (1/n) Σ (π − y) γ_e (T − T̄); zero into the head if t̂ is frozen | low if t̂ is frozen |
| (v) TACC weight γ_e | γ_e = max(0, 2·acc_e − 1), with acc_e the text-only fg-vs-fg accuracy on the support fg points | — (weight only) | none: argmax, no gradient | **γ_e is inflated in training** if t̂ was fitted on the episode's own classes |

Notes [inferred]:
* (i) and (iv) coincide when t̂ is frozen and added after the head: adding κγ_e t̂_c to M_eff is a logit prior
  (`models/cascadeproto.py:205-216`). (iv) touches no module and cannot perturb what the head learned. (i) goes
  through all four stages and so lets the head *use* the prior (e.g. amplify it where the gates say the support is
  atypical), which is also what makes it riskier.
* "Cannot end below the base" is guaranteed only for training-free forms: κ = 0 is E1 exactly, and κ is chosen on
  base classes. A trained form is guaranteed only to *start* at E1 (warm start from E1 `last.pt` with α = 0). The
  shortcut of row (i) can then pull it below E1 on novel classes, because CE on base episodes rewards memorisation.
  Two guards are needed: **text dropout** (α·m, m ~ Bernoulli(0.5) per episode, so the head must keep working
  without text), and **leave-episode-classes-out text** (t_c for a training episode computed by a ridge map fitted
  on the EMA bank *without* the episode's classes, so the text is as uninformed as it will be on a novel name).
* Zero-init pitfall: if both α and the text map start at zero, ∂ℓ/∂α = 0 and ∂ℓ/∂t = 0 forever. Only the gate may be
  zero; the map starts from the closed-form ridge solution.

---

## 3. Causal model and the optimal combination

### 3.1 Graph

```text
          ┌──────────────► name N_c ──► CLIP e_c ──(map W, fitted on 6 base classes)──► t_c
 class C ─┤
          └──► class-level 3-D appearance μ_c ──┬──► support object s = μ_c + δ_s ──(+ exact mask M)──► p_c
                                                └──► query object  q* = μ_c + δ_q ──► query features f_i
 scene / context Z (occlusion, partial scan, co-occurring clutter) ──► δ_s, δ_q
```

In 1-shot with an **exact binary mask**, the support already says which points are the class. Text cannot resolve
mask ambiguity, clutter or co-occurrence: that information is given. Text carries exactly one thing the support does
not: **a prior on μ_c**, so the information on δ_s that a single, possibly partial or atypical, support object
lacks. It reaches the point space only through W, whose error on novel names §1.4 measured as large [inferred].
Text helps when δ_s is large and W is good. It hurts through the modality gap (W's bias on novel names) and through
fixed weights (§3.2).

### 3.2 When a combination beats both views

*Prototype level (shrinkage).* s = μ + δ_s with covariance τ²I, t = μ + e with covariance σ_t²I, and the query is
scored against q* = μ + δ_q. The best estimate is p = (σ_t² s + τ² t)/(τ² + σ_t²), with
E‖p − q*‖²/D = τ²(1 + σ_t²/(τ² + σ_t²)), against 2τ² for s alone. A text prior σ_t² = rτ² acts like
K_eff = 1 + 1/r shots. With VIP-Seg's measured 1 → 5-shot gain of +1.45 (S1) to +4.28 (S0)
(`docs/research/2026-09-22_improvement_directions.md:91-92`), and the error term shrinking as 1 + 1/K, a prior 3×
worse than one shot (K_eff = 1.33) buys ≈ 31 % of that: **+0.45 (S1) to +1.3 (S0)**. A prior as good as one shot
buys 62 %: +0.9 to +2.7. **A biased prior** (t = μ + b + e) adds ‖b‖²·(τ²/(τ²+σ_t²))² to the error, and the
class-agnostic shrinkage of §1.4 is such a bias [inferred, algebra].

*Decision level (product of experts).* For a binary decision (c vs c′) at a point, let the two views' margins be
m_s, m_t with means μ_s, μ_t, unit variances and correlation ρ. The best linear rule m_s + γ m_t has

  SNR²_opt = (μ_s² + μ_t² − 2ρμ_sμ_t)/(1 − ρ²) = μ_s² + (μ_t − ρμ_s)²/(1 − ρ²),  γ* = (μ_t − ρμ_s)/(μ_s − ρμ_t)

(Fisher discriminant of two correlated features) [inferred, algebra]. Three consequences:

1. **Gain condition:** fusion helps iff μ_t ≠ ρμ_s. A weak view helps if its errors are not the support's. A strong
   view that is a noisy copy of the support (μ_t ≈ ρμ_s) adds nothing.
2. **Fixed weights can hurt:** SNR²(γ) = (μ_s + γμ_t)²/(1 + γ² + 2γρ). At γ = 1 and ρ = 0 this falls below μ_s²
   when μ_t < (√2 − 1)μ_s ≈ 0.41μ_s. Eq.9's 1:1 fusion with a near-uninformative novel text is in this regime.
3. **The optimal weight is per episode**, because μ_s (support quality) and μ_t (how well W transfers to *this* name)
   vary by episode. A gate is therefore justified. TACC's γ = IoU of the text-only prediction on the support estimates
   μ_t only [verified: arXiv 2410.22489 Eq.9–10]. The missing μ_s can be estimated honestly by **split-support
   validation**: build the prototype from one spatial half of the support mask and score the other half (in-sample
   support IoU is biased high, since the prototype is the mean of those points) [inferred].

MM-FSS, ScanNet 2-way 1-shot/5-shot [verified: arXiv 2410.22489 Table 3(e)]: no text (0:1) 42.83 / 48.04; text only
(1:0) 35.10 / 37.32; fixed 1:0.5 43.24 / 48.12; fixed 1:1 43.76 / 48.85; adaptive γ:1 **44.73 / 50.07**. A view 7.7
points worse on its own adds +0.9 at a fixed weight and +1.9 with the episode weight, as consequences 1–3 predict.
Its text view, however, comes from a separate feature head distilled from 2-D VLM features (Stratified Transformer,
100-epoch 2-D alignment, then a frozen backbone), so its ρ is structurally lower than ours [verified: same source
§3.3, §4.1].

### 3.3 Why our text view is structurally correlated with the support view

Both classifiers are linear in the same f_i: the support view is ⟨f_i, M_c − M_c′⟩ and the text view
⟨f_i, t_c − t_c′⟩. Their error correlation is set by the angle between the two directions under the query's feature
covariance. Points whose features are ambiguous are ambiguous to both. The text view can only add what lies along
**the oracle direction O_c − O_c′** (the query's own class means, worth +11 to +15 mIoU; D-29) that M_c − M_c′
misses. So "does text help?" becomes a measurable question: **is cos(t_c − t_c′, O_c − O_c′ | M_c − M_c′), the
partial alignment, positive on novel-class episodes?** [inferred]. D-26…D-29 tried to reach O from the unlabelled
query and failed. Text is a query-independent estimate of the same thing, from a different information source (the
name), and that is the one sense in which it is a second view.

---

## 4. Evidence that must be re-examined

### 4.1 What the existing text numbers can and cannot say

| claim | evidence | status |
| :--- | :--- | :--- |
| "LMA ≈ +0.57" | one seed, headless baseline (`num_stages=0`), no L2, batch 4 / 6,000 updates; best +0.57, last −0.75 [measured: `results/phase15_full/SUMMARY.md`] | inside the 1.3–1.9-point selection swing (`results/phase16_e1/SUMMARY.md`) and the 1–3-point CUDA seed spread (report §5); says nothing about VIP-Seg's head |
| "text is worth ~+0.4 at this level" | EDS-Net VSBM +0.45 S0 / +0.30 S1 on DyPolySeg (72.02 → 72.47, 73.82 → 74.12) [verified via the repo's quote: `docs/research/2026-09-22_improvement_directions.md:329-335`] | EDS-Net was not fetched here; same design family as LMA (noise generator + GMMN), so it inherits §1.2–1.3 |
| "text +3.3 / +3.7 (MM-FSS)" | 41.45 → 44.73 and 46.38 → 50.07 bundle MSF (which consumes the text guidance G_q) and TACC; TACC alone +1.90 / +2.03 [verified: arXiv 2410.22489 Table 3(a,d,e), §3.4] | corrected setting, 2-D-aligned backbone; an upper reference, not a prediction |
| "text alone +1.78 on a from-scratch net" | EPSegFZ LGPE, S3DIS S0 2-way 1-shot 68.71 → 71.49, DGCNN from scratch, additive `λ₁p̃ + λ₂p_raw + λ₃p_dyn + λ₄p_text` with λ₄ decaying as e^(−0.5t) [verified: arXiv 2511.11700 Table 5, §LGPE] | the closest precedent (no pretraining, S3DIS, prototype-level text); single run, no variance reported |
| "P0.4 text diagnostic" | proposed in `docs/research/2026-09-22_improvement_directions.md:489-491` | **never run** [verified: no result file or script] |

### 4.2 Measurements to redo on the E1 base before any conclusion about text

1. Any text arm on `stage_type=vip`, `l2norm_point_proto=true`, batch 1 / 24,000 updates, reported `last` and
   best-of-13 (D-22 amended). A trained arm needs **two seeds** (E1 itself is one seed); effects below ~1.5 points
   are unresolvable with one [measured: E1 selection gain +1.85 [+1.43, +2.30]].
2. S0 E1 has not been run, and S0 is the held-out fold (D-22 rule 2). Text claims need it.
3. A run that uses `use_lma=true` as implemented would test §1.2–1.3's flaws, not text. It should not be queued.

### 4.3 Do the failed components become useful combined with text?

* **Co-training condition.** D-28 failed because the base margin had AUC 0.65–0.72 for flagging false foreground,
  and at that AUC it removed true and false mass alike (`results/phase16_p2/SUMMARY.md`). Text would need a clearly
  higher AUC for flagging the *support view's* errors, and §3.3 says its view is structurally correlated. So combining
  text with EM is not justified by the evidence yet [inferred].
* **The cheap measurement.** One evaluation pass of E1 `last.pt` (no training) on the S1 *valid* draw, with t̂ from
  the closed-form map of §5 (T1), per episode:
  (1) the text-only fg-vs-fg accuracy on the support (acc_e) and on the query;
  (2) the error correlation ρ between the support-view margin (from L_final) and the text-view margin on query
      foreground points;
  (3) the AUC of the text margin for the support view's fg-vs-fg errors;
  (4) the partial oracle alignment of §3.3, with O from the query labels (diagnostic only, never a prediction);
  (5) the fusion gain at an *oracle* per-episode γ (an upper bound) and at γ_e.
  If (5) with an oracle γ is < +0.5, no gate can make text worth a training run.
* **Base prototypes plus text.** The P1 bank already contains what a closed-form text map needs (base-class
  prototypes from base labels only). That is the one place where a failed component (D-27's bank) is directly
  reusable [verified: `results/phase16_p1/`, `models/base_calibration.py:91-96`].

---

## 5. Ranked designs with probes and pre-registered rules

Common elements [inferred]: ẽ_c = n(e_c − ē), where ē is the mean embedding of the 12 fold-independent class prompts
(names only, no labels). The text map is the kernel ridge `W = Bᵀ(ẼẼᵀ + λI)⁻¹Ẽ` from base-class bank prototypes
B ∈ R^{6×128} (centred unit, `models/base_calibration.py`), λ = 10⁻³ as in the probe, and t̂_c = n(W ẽ_c) ∈ R^128.
Feature centring uses the bank's μ. There is no noise generator and no GMMN. κ and any threshold are chosen on
**base-class** episodes of the training fold (leave-two-base-classes-out), never on scored classes (D-22 rule 3).

### T1 (rank 1): training-free, support-validated text logit prior after the head

* **Equations.** For foreground columns c ≥ 1: T_{bic} = ⟨n(f_{bi} − μ), t̂_c⟩ and
  L′_{bic} = L_final,{bic} + κ γ_e (T_{bic} − (1/N)Σ_{c′≥1} T_{bic′}). The background column is unchanged
  [inferred]. Here γ_e = max(0, 2·acc_e − 1), with acc_e the fraction of support foreground points of both ways that
  argmax_{c≥1} T assigns to their own way. Shapes: f [B_q, 2048, 128], t̂ [N, 128], T [B_q, 2048, N], γ_e scalar per
  episode. It plugs in after `routing(...)` in `models/cascadeproto.py:190`, as a post-hoc scorer like
  `models/transductive.py`; the head is untouched.
* **Why it cannot do worse at init.** κ = 0 gives E1's logits exactly. Centring over foreground columns means text
  can only re-rank the ways, never the fg/bg decision, where §1.5 says text is uninformative.
* **Gradient path.** None; no training.
* **Causal argument.** A prior on μ_c from the name (§3.1), weighted by its measured transfer to *this* name on
  *this* support (§3.2, consequence 3).
* **Probe (P-T1, one GPU evaluation of E1 `last.pt`, ~15 min).** The five quantities of §4.3, then the fixed100
  test at the κ frozen on base classes.
* **Expected effect.** 0 to +0.5 with the repo prompt, where held-out assignment is 0.58 (§1.4); up to about +1 with
  descriptions (0.72) [inferred from §3.2 with r ≈ 3–10].
* **Pre-registered rules.** T1.0 mechanism: text-only fg-vs-fg accuracy on S1-valid queries ≥ 0.60, and the
  partial oracle alignment > 0 with its episode-bootstrap CI above 0. Otherwise stop text on this feature space
  (§5.4). T1.1 go: the oracle-γ gain ≥ +0.5 and the γ_e gain ≥ +0.3 with a CI above 0 on fixed100, then S0 test
  once. T1.2 stop: the oracle-γ gain < +0.5. T1.3 collapse watch: any class −3 IoU. Run once each with the repo
  prompt and the descriptions; the description set is frozen as written for the probe of §1.4.

### T2 (rank 2, only after T1.0 passes): trained zero-init residual text prior at p⁰

* **Equations.** p⁰_c = n(n(p_c) + a·m·γ_e·t̂_c) for c ≥ 1 (background row unchanged). Here a is a scalar
  initialised to 0, m ~ Bernoulli(0.5) per episode in training (text dropout) and m = 1 at test. In training, t̂_c
  comes from the ridge map refitted on the EMA bank **without the episode's ways** (4 of 6 base classes), so its
  quality matches a novel name. Optionally add a rank-4 correction W + UVᵀ with U = 0 at init and weight decay.
  Shapes: p [N+1, 128] → p⁰ [N+1, 128], ‖p⁰_c‖ = 1. It plugs into `models/cascadeproto.py:165-172`, replacing the
  LMA branch.
* **Why it starts at the base.** a = 0 gives E1's function exactly. Warm-start from E1 `last.pt` and fine-tune with
  the E1 schedule's last LR stage (7,200 updates, ~0.7 GPU-h), or train from scratch on the full schedule (2.3 GPU-h,
  two seeds).
* **Gradient path.** ∂ℓ/∂a = Σ_c m γ_e ⟨J_nᵀ g_c, t̂_c⟩ with g_c from §2.2. The J_n projection means a only grows if
  the text component orthogonal to p pays off through all four stages. No vanishing: t̂ ≠ 0 and the residual paths
  carry g_c.
* **Causal argument.** As T1, but the head can learn *where* to trust the prior, for example amplifying it through
  σ^s when the support block is atypical. Dropout and leave-out cut the shortcut of §2.3.
* **Probe before training.** T1's P-T1 (same t̂). Plus the check that a stays near 0 when trained with t̂ replaced
  by the centroid of the other base prototypes (a text-free control), run with the same warm start.
* **Rules.** T2.1 go: T2 − E1 (and T2 − control) ≥ +0.5 on fixed100 with a CI above 0 on `last`, same sign on all
  three random600 draws, confirmed on the second seed. Otherwise report and stop. Collapse watch as T1.3.

### T3 (rank 3, conditional): text FiLM on PEM's channel gates, with a mask-only control

σ^s_c ← σ^s_c ⊙ (1 + β tanh(H ẽ_c)), with β = 0 at init and H initialised from a ridge fit of the support Fisher
ratio J_d (directions §5.4) on base classes. It gives the head the class-specific channel selection it lacks
(fact 2). The same gate driven by the *mask* (J_d from the support) is the control, and text is credited only for
T3 − control. This requires wrapping the inherited PEM, since its gates are computed inside `forward` (AGENTS
guardrail 2: wrap, never edit), which is the most invasive of the three. Expected 0 to +1, no 3-D evidence
[inferred]. Run only if T1.0 passes and T2 is inconclusive.

### Not recommended [inferred]

The LMA as implemented (§1.2–1.3). An extra text row (does not exist, §2.1). A CE-trained MLP text map without
leave-out (the §2.3 shortcut). EM or co-training with text (§4.3), until the text view's AUC for flagging support
errors is measured well above D-28's 0.65–0.72.

### 5.4 If the honest answer is "text cannot help on S3DIS with six names"

T1.0 failing would mean the point feature space holds no name-predictable class structure beyond the support. What
would have to change, each with its cheapest test [inferred]:

1. **More text per class** (descriptions, attribute prompts; CLIP gains ~5 points on ImageNet from engineering and
   ensembling [verified: arXiv 2103.00020 §3.1.4]). Test: the CPU probe of §1.4, pass ≥ 0.70 on all four banks;
   descriptions reach 0.60–0.87, so it is borderline.
2. **A point space aligned to CLIP from a source other than six names**: 2-D–3-D distillation as in MM-FSS or
   OpenScene. It needs S3DIS images, which the pipeline does not read, and a new decision (guardrail 1 concerns
   point-cloud weights; a distilled head changes the comparison with VIP-Seg).
3. **More training names**: ScanNet has 10 per fold. Test: the same probe on ScanNet banks.
4. **Use text only where CLIP geometry is known to be meaningful**, as a relational prior (e.g. that "sofa" is
   near "chair"), by transferring the base prototype of the nearest base name. That is the `nn` arm of the probe:
   20–67 % with the repo prompt, i.e. no better.

---

## Sources

* [MM-FSS] Z. An et al., *Multimodality Helps Few-shot 3D Point Cloud Semantic Segmentation*, ICLR 2025,
  arXiv 2410.22489 (HTML fetched 2026-09-24): Eq.5 (G_q = F_i^q Tᵀ), Eq.9–10 (TACC, γ = support IoU of the text-only
  prediction), §3.3–3.4 (LSeg text encoder, IF head, MSF consumes G_q), §4.1 (Stratified Transformer, 2-D-aligned
  pretraining, COSeg's corrected protocol), Table 3(a–f), Table 1 (S3DIS 2w1s mean 44.30).
* [CLIP] A. Radford et al., arXiv 2103.00020 (HTML v1 fetched): §3.1.4, "prompt engineering and ensembling improve
  ImageNet accuracy by almost 5%", ensemble built in embedding space.
* [EPSegFZ] arXiv 2511.11700 (HTML fetched): LGPE, additive prototype fusion with decaying text weight, Table 5
  (text +1.78 on S3DIS S0 2w1s, DGCNN from scratch).
* [EDS-Net] AAAI 2026, https://ojs.aaai.org/index.php/AAAI/article/view/37929. **Not fetched here**; numbers only via
  `docs/research/2026-09-22_improvement_directions.md` §4.6.
* [CascadeProto] numbers only via the repository's specs and `docs/research/2026-09-21_reproduction_report.md`.
* [PAP3D] arXiv 2305.14335: zero-shot projected CLIP prototypes, via `docs/research/2026-09-22_improvement_directions.md:210-213`;
  not re-fetched. PointCLIP, ULIP and OpenScene were not fetched; they are cited only as the class of 2-D-aligned
  methods (§5.4 item 2).
* Repository: `models/{lma,clip_text,cascadeproto,prototypes,vip_stage,adrm,vipseg,base_calibration}.py`,
  `loss/gmmn_loss.py`, `pipeline/model_api.py`, `docs/spec/00_SOURCES_AND_DECISIONS.md` (D-04…D-06, D-13, D-17,
  D-22, D-25…D-30), `results/phase15_full/`, `results/phase16_{e1,p1,p2,r2}/SUMMARY.md`, E1 `last.pt`, P1 banks.
