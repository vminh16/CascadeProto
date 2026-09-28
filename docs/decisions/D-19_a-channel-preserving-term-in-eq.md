### D-19 — A channel-preserving term in Eq.19 · `PROPOSED`, beyond the paper

* **Problem.** Only the part of a prototype that differs between class rows can change
  `argmax_c <f, p_c>`, because `<f, p_c> = <f, μ> + <f, d_c>` with `μ = mean_c p_c` common to every
  class. Measured on synthetic post-ReLU features: `P^0` carries 14.2–14.9 % of its energy in `d`,
  one EPPM stage leaves 3.8–4.2 %, and VIP-Seg's PEM leaves 6.9–8.9 %. Both summands Eq.19 prints are
  class-poor — `P_diffuse` has no class index at all [DECISION D-16], and `P_cross` is a row-stochastic
  mixture over **channels**, near rank 1 [DECISION D-18] — so the fused term dilutes `d` rather than
  sharpening it.
* **What the switch does.** `eq19_self = {none (default), gated}` adds `ψ(P^{t-1}_gated)` to
  `P_combined` before Eq.20, reusing the same `ψ` as Eq.14, so the parameter budget is unchanged
  (79,395 per stage). It is the CascadeProto-shaped analogue of VIP-Seg's
  `proto_self = σ(A_s) ⊙ ψ(P)` [VIPSEG models/vipseg.py:270-277], which Eq.19 has no counterpart for
  [audit F1].
* **Honest status of the evidence.** The static CPU analysis that motivated it did **not** confirm the
  fix: adding the term moves the class-varying share only from 3.9 % to 4.2 %, and reproducing
  VIP-Seg's 7–8 % additionally needs a LayerNorm on the new term and the removal of Eq.21's ReLU, the
  SE block and `W_out`'s bias — at which point a third metric (how many point predictions survive a
  stage) moves the other way. Three static metrics disagree, and 15e already showed that a stage whose
  diagnostics all look healthy after training still buys nothing. **This switch is an experiment, not
  a claim**; it is off by default and is not part of any paper table.
* **Ablation flag.** `eq19_self = {none (default), gated}`. Neither value raises. Any run using
  `gated` is outside the paper and must be reported as such.
