### D-16 — Layer details not given in the paper · `LOCKED`

* **Problem.** L1 names several layers without widths, rates or placement. The pre-rewrite specs stated values for them as if they were paper content.
* **Decision.** Use the following values, always tagged `[DECISION D-16]`:
  1. **Adapter** (Eq.5, "two-layer MLP with LayerNorm and Dropout" [PAPER §3.3]): `Linear(512→D) → LayerNorm(D) → ReLU → Dropout(p = 0.1) → Linear(D→D)`. Dropout rate and position are not in L1.
  2. **Generator G** ("three-layer MLP" [PAPER Eq.6]): `Linear(2D→D) → ReLU → Linear(D→D) → ReLU → Linear(D→D)`, input `[E_fused; z]` per D-05.
  3. **Fusion network** ("two-layer MLP" [PAPER Eq.19]): `Linear(2D→D) → ReLU → Linear(D→2)`.
  4. **SE block** (Eq.20): reduction ratio r = 4, `W_1 ∈ R^{D/r×D}`, `W_2 ∈ R^{D×D/r}`; AvgPool over the class dimension.
  5. **Output projection** (Eq.21): `W_out = Linear(D→D)`; `LN = LayerNorm(D)`.
  6. **Entropy stability:** clamp `p` to `[10⁻⁷, 1 − 10⁻⁷]` before Eq.10. L1 only specifies ε = 10⁻⁸; the clamp does not change results for |x| < 16.
  7. **Diffusion pooling** (Eq.15): `mean(F, dim=1)` is read as the mean over the point axis of a batched `[B, N_points, D]` tensor, the only reading that yields the "channel-level activations" the text describes (on an unbatched `[N_points, D]` tensor `dim=1` would average over channels). `s_ch` is the channel mean over all support points of all ways and shots; `q_ch` is computed per query. The vector `P_diffuse ∈ R^D` is broadcast to all N+1 class rows before Eq.19.
  8. **Stage parameters** are not shared across the T stages [PAPER §3.5 "each EPPM applies entropy gating independently"].
