### D-17 — Meaning of the ablation switches (Table 4–5) · `LOCKED`

* **Problem.** Table 4 "cumulatively adds one component": LMA, Entropy Gate, Cascade (T = 4), ADRM, "starting from a plain VIP-Seg backbone with masked average pooling and single-step prototype matching" [PAPER §4.3]. L1 does not say what the model looks like between rows, e.g. what "Entropy Gate" without "Cascade" is.
* **Observation.** The row increments in Table 4 match the text: +1.21 (LMA), +1.42 (gate), +2.06 (cascade), +0.56 (ADRM) [PAPER Tab.4] [PAPER §4.3]. The baseline row (81.28 Avg) is far above VIP-Seg's own 74.15 Avg in Table 2; L1 does not explain the gap.
* **Decision.** Four switches, mapped to the rows of Table 4:

  | Row | `use_lma` | `num_stages` | `use_gate` | `use_adrm` | Prediction |
  | :--- | :---: | :---: | :---: | :---: | :--- |
  | Baseline | false | 0 | – | – | `F^q P_pointᵀ` |
  | + LMA | true | 0 | – | – | `F^q (P^0)ᵀ` |
  | + Entropy Gate | true | 1 | true | – | `L^1` |
  | + Cascade (T = 4) | true | 4 | true | false | `L^4` |
  | + ADRM (full) | true | 4 | true | true | `L_final` |

  `use_gate = false` sets `g ≡ 1`. `use_lma = false` sets `P_modal ≡ 0` and drops `L_GMMN`. Without ADRM the prediction is the last stage's `L^T`. With `num_stages = 1` ADRM is a softmax over one stage, i.e. weight 1, so `L_final = L^1` whether `use_adrm` is on or off; `W_g` is then not built, because it could never receive a gradient (maintainer decision 2026-09-19). Table 5 varies `num_stages` ∈ {1..6} with every other switch on [PAPER Tab.5].
