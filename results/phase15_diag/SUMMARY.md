# Phase 15 diagnostics: every variant, three seeds each

S3DIS S0, 2-way 1-shot. 2,400 training episodes, batch 4 (600 steps), AdamW lr 1e-3 wd 0.1, no
decay; 300 valid episodes. The loop is not bit-reproducible on CUDA, so the seed spread is the
resolution of the whole table.

| variant | what it tests | seeds | mean | sd |
| :--- | :--- | :--- | ---: | ---: |
| `baseline_l2` | prototype matching with L2-normalised prototypes, no added module | 0.5316 / 0.5187 / 0.5111 | 0.5205 | 0.0104 |
| `full` | the paper as specified (D-01, D-02, D-16) | 0.5029 / 0.5244 / 0.5241 | 0.5171 | 0.0123 |
| `full_gatefeat` | D-02 read as gating the features, the only reading that consumes Eq.12 | 0.4951 / 0.5219 / 0.5484 | 0.5218 | 0.0267 |
| `full_scaled` | D-10: Eq.23's prose says "scaled dot-product", the equation prints no scale | 0.4343 / 0.4961 / 0.4673 | 0.4659 | 0.0309 |

Against `baseline_l2`, Welch's t on 3 + 3 runs:

| variant | difference | standard error | t |
| :--- | ---: | ---: | ---: |
| `full` | -0.0033 | 0.0093 | -0.36 |
| `full_gatefeat` | +0.0013 | 0.0165 | +0.08 |
| `full_scaled` | -0.0546 | 0.0188 | -2.90 |

| variant | attn_width init -> end | P_cross chan_var init -> end | w_diffuse init -> end |
| :--- | :--- | :--- | :--- |
| `full` | 128.00 -> 53.85/29.42/40.09 | 0.00/0.00/0.00 -> 0.37/0.53/0.61 | 0.51/0.49/0.47 -> 0.01/0.03/0.08 |
| `full_gatefeat` | 128.00 -> 16.36/47.42/37.50 | 0.00/0.00/0.00 -> 0.31/1.91/0.67 | 0.51/0.49/0.47 -> 0.15/0.11/0.17 |
| `full_scaled` | 128.00 -> 49.24/15.73/47.89 | 0.00/0.00/0.00 -> 0.81/8.28/0.53 | 0.51/0.49/0.47 -> 0.17/0.56/0.65 |

## Readings

* **Neither reading of the paper's ambiguities helps.** `full_gatefeat` lands on `baseline_l2`
  (t = +0.08) with more than twice the seed spread, and `full_scaled` is clearly **worse**
  (t = -2.90, and its training loss plateaus at 0.446-0.463 against 0.33-0.39 everywhere else,
  because dividing the logits by sqrt(D) flattens the softmax). The default `logit_scale=none`
  of D-10 is therefore the right reading of Eq.23, and D-02's `gate_target` does not matter.
* **The variants that keep `w_diffuse` high score worst.** `full` drives it to 0.008-0.082 and
  scores 0.5171; `full_gatefeat` leaves it at 0.11-0.17 and scores 0.5218; `full_scaled` leaves
  it at 0.17-0.65 and scores 0.4659. Consistent with `P_diffuse` carrying no class information
  (D-16), but not evidence that removing it by hand would help, since the model already does.
* **Nothing in this table beats prototype matching with normalised prototypes.**
