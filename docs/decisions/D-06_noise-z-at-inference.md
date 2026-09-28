### D-06 — Noise z at inference · `LOCKED`

* **Problem.** Eq.9 says "The fused initial prototype **used for training** is P^0 = P_point + P_modal"; §3 promises an "inference strategy" that is never described.
* **Decision.** Training samples `z ~ N(0, I)` per forward pass. Evaluation uses `z = 0`, making predictions deterministic.
* **Ablation flag.** `eval_noise = {zero (default), sample, mean_of_M}`. `mean_of_M` needs a value of M that L1 does not give; it raises until one is chosen.
