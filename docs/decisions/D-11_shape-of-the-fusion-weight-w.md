### D-11 — Shape of the fusion weight w (Eq.19) · `LOCKED`

* **Problem.** Eq.19 states `w ∈ R²` while `P_cross`, `P_diffuse ∈ R^{(N+1)×D}`.
* **Decision.** One weight pair per query: `w = softmax(f_fusion(mean_c [P_cross; P_diffuse]))`, where `f_fusion` is a two-layer MLP `2D → D → 2`, pooled over the class dimension, as printed `w ∈ R²`.
* **Ablation flag.** `fusion_weight = {per_query (default), per_class}`.
