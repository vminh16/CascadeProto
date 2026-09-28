### D-10 — Logit form (Eq.23) · `LOCKED`

* **Problem.** Eq.23 is called "scaled dot-product matching" but prints no scale: `L^t = F^q (P^t)ᵀ`.
* **Evidence (L2).** VIP-Seg uses a plain dot product with no temperature [VIPSEG models/vipseg.py:158], but L2-normalises the **initial** prototypes [VIPSEG models/vipseg.py:142]. L1 Eq.3 and Eq.9 show no normalisation.
* **Decision.** Plain dot product exactly as printed; no temperature; no L2 normalisation of `P_point`.
* **Ablation flags.** `logit_scale = {none (default), sqrt_D}`; `l2norm_point_proto = {false (default), true}`.
