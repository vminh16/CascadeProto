### D-18 — Degeneracy of the cross-attention output (Eq.14) · `LOCKED` (as an ablation flag)

* **Symptom (L0, measured).** On the VM, with the same loop, data, encoder and metric, 2,400 training
  episodes and 300 valid episodes (2026-09-20): `baseline_l2` 0.5218, `full_l2` 0.5150, `full` 0.5164,
  `baseline` 0.4416, VIP-Seg's own model 0.6948. Four EPPM stages plus ADRM plus LMA move the score by
  −0.7 points against a plain L2-normalised prototype. The +7.5 of `full` over `baseline` is the
  LayerNorm of Eq.21 equalising the prototype norms, which `l2norm_point_proto` does on its own (+8.0).
* **Structural cause, independent of any number.** `P_cross[b,c] = A[b,c] ψ(P_gated[b,c])` with the
  rows of `A` a softmax over the **channel** axis [DECISION D-01]: output channel `i` is a convex
  combination of the channels of `ψ(P)`. Both limits of that softmax destroy the channel structure —
  saturated, every row copies the same channel; uniform, every row copies the channel mean — and Eq.14
  has no learnable term that controls where between the two it sits. The other summand of Eq.19,
  `P_diffuse`, carries no class index at all [DECISION D-16]. So a stage can add class-discriminative
  structure only through the residual `P^{t-1}` of Eq.21. Measured: the update `P^1 − P^0` keeps
  87.5–99.6 % of its energy in a single singular value across every feature distribution tried, and in
  the distribution of the phase-11 fixtures the class rows leave the stage **more** similar than they
  entered (max pairwise cosine 0.9906 → 0.9964).
* **Where on that axis does the module sit?** The regime depends on the per-channel statistics of the
  encoder features, which are not reproducible on the CPU. Measured on synthetic post-ReLU features,
  `β` = the per-channel offset a trained BatchNorm supplies (one shared φ, one seed, `|mean_D|`-relative
  channel variation of `P_cross`):

  | features | `none`: softmax width (uniform = 128) | `none`: channel variation | `layernorm`: width | `layernorm`: channel variation |
  | :--- | ---: | ---: | ---: | ---: |
  | β = 0 (BatchNorm at initialisation) | 112.8 | 2.3e-01 | 127.5 | 1.3e-01 |
  | β std 0.5 | 8.1 | 6.8e-01 | 125.6 | 1.2e-01 |
  | β std 1.0 | 10.7 | 2.2e+00 | 114.0 | 1.8e-01 |
  | β std 2.0 | 27.2 | 1.9e+00 | 96.1 | 5.1e-01 |
  | positive-only offset, `relu(N(0,1))` per channel | 1.02 | 2.6e-04 | 127.4 | 1.5e-01 |

  The last row is the fully saturated case: all 128 rows of `A` select the same channel, the winner
  takes weight 0.998, `P_cross` is constant along `D`, and φ's relative gradient falls to 2.8e-04 of
  ψ's (at three times that feature scale it is exactly 0, so the stage can never leave the state).
  **Which row describes the real encoder is not established** and is the measurement of 15c.
* **Why VIP-Seg does not hit this.** Its `que.reshape(72, -1)` [VIPSEG models/vipseg.py:284] is not a
  transpose: it folds the batch axis into the filter axis, so the matrix it builds is not the channel
  correlation it is annotated as (max abs difference 513.99 against the clean form on the same inputs,
  same φ, same scale). The scrambling decorrelates the logits — row std 1.191 against 4.543 — and keeps
  the softmax usable. D-01 deliberately did not copy it. VIP-Seg also carries a channel-preserving term
  `proto_self = σ(A_s) ⊙ ψ(P)` [VIPSEG models/vipseg.py:270-274]; Eq.19 fuses only `P_cross` and
  `P_diffuse`, so CascadeProto has no equivalent.
* **Decision.** Add the switch `cross_attn_norm = {none (default), layernorm}` as a **probe, not a
  fix**. `layernorm` standardises `Q'` and `S'` along the projection axis `r` with one shared LayerNorm
  before the correlation, which makes `A` exactly invariant to the scale of the features and gives each
  stage 144 parameters (`γ`, `β`) with which to choose its own sharpness — control that the fixed
  `1/√d` does not provide. It removes the saturated regime but not the rank-1 update, and in the
  β std 0.5–2.0 rows above it has **less** channel variation than `none`, so it is not established as
  an improvement. The default stays `none`, the literal Eq.14.
* **Outcome on the VM (15c, real encoder, 2,400 train / 300 valid episodes, 2026-09-20).** The
  saturation hypothesis is **refuted** and `layernorm` is **rejected as a default**.

  | variant | valid mIoU | `attn_width` init → end | `P_cross` channel variation init → end |
  | :--- | ---: | :--- | :--- |
  | `baseline_l2` | 0.5223 | — | — |
  | `full` | 0.5346 | 127.9997 → 9.59 | 6e-04 → 0.369 |
  | `full_norm` | 0.5134 | 92.14 → 109.31 | 0.649 → 0.012 |

  Readings. (i) The real encoder puts Eq.14 at the **uniform** end at initialisation, not the
  saturated one: width 127.9997 of 128, `P_cross` channel variation 6e-04, i.e. collapsed, but by the
  other limit than the CPU probe suggested. (ii) The stage **escapes on its own**: after 600 steps the
  width is 9.59 and the channel variation 0.369, so φ does learn and the module is not dead. (iii) It
  still buys almost nothing — `full` is 1.2 points above `baseline_l2`, and the same `full`
  configuration scored 0.5164 on an earlier run of the identical command, so the run-to-run spread of
  the loop is larger than the effect. (iv) `layernorm` makes it **worse**: it holds the attention near
  the uniform end (92 → 109 instead of 128 → 9.6), the channel variation ends at 0.012 instead of
  0.369, and the score drops to 0.5134.
* **Decision.** `cross_attn_norm` stays in the code as an ablation flag with default `none`, the
  literal Eq.14. Neither value raises. It is not a fix and must not become the default.
* **Three seeds per variant (15e, 2026-09-20, `results/phase15_diag/`).** The cascade contributes
  nothing measurable, and the module is not broken.

  | variant | seeds | mean | sd | `attn_width` init → end | `P_cross` chan_var init → end | `w_diffuse` init → end |
  | :--- | :--- | ---: | ---: | :--- | :--- | :--- |
  | `baseline_l2` | .5316 / .5187 / .5111 | 0.5205 | 0.0104 | — | — | — |
  | `full` | .5029 / .5244 / .5241 | 0.5171 | 0.0123 | 128.00 → 29–54 | 0.0006–0.0019 → 0.37–0.61 | 0.47–0.51 → 0.008–0.082 |

  `full − baseline_l2 = −0.0033`, standard error 0.0093, `t = −0.36`. A 95 % interval is about
  ±2.6 points, so the +4.04 that Table 4 attributes to gate + cascade + ADRM lies outside it.
  Meanwhile every stage diagnostic is healthy on all three seeds: the attention leaves the uniform
  collapse (128.00 → 29–54), `P_cross` gains channel structure (three orders of magnitude), and the
  fusion learns to shut the class-blind branch off.
* **That closes two candidates.** (a) `P_diffuse` having no class index [DECISION D-16] is **not** the
  cause: the Eq.19 weight on it falls from ~0.49 to 0.008–0.082, i.e. training removes the branch by
  itself, and removing it earlier by hand would change nothing. (b) The cross-attention is **not**
  stuck: it is healthy after 600 steps on every seed.
* **What is left.** The module behaves as specified and adds nothing, while VIP-Seg's own PEM/PDM add
  about 17 points over the same `baseline_l2` through the same loop. The one structural difference
  left is the channel-preserving self-correlation term `proto_self = σ(A_s) ⊙ ψ(P)`
  [VIPSEG models/vipseg.py:270-274], which Eq.19 of the paper does not have. Adding it would mean
  implementing a module the paper does not describe, so the finding is recorded rather than patched:
  **Table 4's increments are not reproducible from the equations as printed.** See also D-17, where
  the paper's own baseline row (81.28 Avg) already sits above VIP-Seg's published 74.15.
