# Phase 15 diagnostic: three seeds per variant (S0, 2-way 1-shot)

Budget: 2,400 training episodes, batch 4 (600 steps), AdamW lr 1e-3 wd 0.1, no decay;
300 valid episodes (every 5th of the 1,500 cached ones). Commit db72cc7, NVIDIA L4.

| variant | seed | valid mIoU | loss_last100 | attn_width init -> end | P_cross chan_var init -> end | w_diffuse init -> end |
| :--- | ---: | ---: | ---: | :--- | :--- | :--- |
| baseline_l2 | 0 | 0.5316 | 0.3330 | - -> - | - -> - | - -> - |
| baseline_l2 | 1 | 0.5187 | 0.3636 | - -> - | - -> - | - -> - |
| baseline_l2 | 2 | 0.5111 | 0.3753 | - -> - | - -> - | - -> - |
| full | 0 | 0.5029 | 0.3763 | 127.9997 -> 53.8467 | 0.0006 -> 0.3674 | 0.5069 -> 0.0077 |
| full | 1 | 0.5244 | 0.3723 | 127.9995 -> 29.4160 | 0.0019 -> 0.5337 | 0.4902 -> 0.0254 |
| full | 2 | 0.5241 | 0.3863 | 127.9998 -> 40.0859 | 0.0013 -> 0.6142 | 0.4681 -> 0.0819 |

| variant | mean | sd | min | max |
| :--- | ---: | ---: | ---: | ---: |
| baseline_l2 | 0.5205 | 0.0104 | 0.5111 | 0.5316 |
| full | 0.5171 | 0.0123 | 0.5029 | 0.5244 |

full - baseline_l2 = -0.0033 (standard error 0.0093, t = -0.36 on ~4 degrees of
freedom). The four EPPM stages, ADRM and LMA together are worth nothing measurable at this budget;
the point estimate is slightly negative. A 95 % interval is about +-2.6 points, so the +4.04 that
Table 4 attributes to gate + cascade + ADRM lies outside it.

Earlier runs of the identical command, for the run-to-run spread at fixed seed 0:
baseline_l2 0.5218, 0.5223, 0.5316; full 0.5164, 0.5346, 0.5029. The loop is not bit-reproducible
on CUDA, so single-seed differences below about 2 points carry no information.
