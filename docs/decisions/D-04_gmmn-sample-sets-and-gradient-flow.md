### D-04 — GMMN sample sets and gradient flow · `LOCKED`

* **Problem.** Eq.7 defines MMD between sets, Eq.8 applies it to `P^bg` and `P^fg` without defining the sets. The paper does not say whether `P_point` is detached.
* **Decision.**
  1. Use the squared MMD of Eq.7 (paper-explicit, Section 3).
  2. `P^fg` = the N foreground rows taken **jointly** as one set of N samples, not N separate 1-vs-1 comparisons. `P^bg` = the single background row (1-vs-1).
  3. `P_point` is **not** detached; gradients from `L_GMMN` reach the backbone, since Eq.26 optimises all parameters jointly and the paper does not mention stopping gradients.
* **Known limitation.** For N = 2 or 3 the MMD estimate is very noisy; a 1-vs-1 MMD reduces to `2·(6 − k(x, y))`.
* **Ablation flags.** `gmmn_fg_mode = {joint (default), per_class}`; `gmmn_detach_point = {false (default), true}`.
