# Phase 15: the full schedule, S3DIS S0, 2-way 1-shot, fixed100 on 1,500 episodes

50 epochs of 480 episodes at batch 4 (6,000 steps), AdamW lr 1e-3 wd 0.1, StepLR halving every 10
epochs. `best` is the checkpoint with the best validation mIoU, `last` is epoch 50 [DECISION D-15].

| configuration | switches | best | last | source |
| :--- | :--- | ---: | ---: | :--- |
| Baseline | `use_lma=false num_stages=0` | 0.4908 | 0.4907 | phase 14, P1 |
| Baseline + L2 | `… l2norm_point_proto=true` | **0.5244** | 0.5019 | phase 15j |
| Full model | defaults | 0.5715 | 0.5670 | phase 14, P1 |

## What the +8.07 of P1 was made of

| step | gain |
| :--- | ---: |
| L2-normalising the point prototypes (D-10, VIP-Seg does it, the paper does not print it) | **+3.36** |
| LMA + 4 EPPM stages + ADRM | **+4.71** |

The paper claims +5.81 for the same step (82.72 -> 88.53, Table 4). Since VIP-Seg normalises its
prototypes [VIPSEG models/vipseg.py:142], the paper's baseline row corresponds to `Baseline + L2`
here, so the increment to compare with +5.81 is **+4.71**. The relative claim of Table 4 is therefore
approximately reproduced; the absolute level is not (52.44 against 82.72, 57.15 against 88.53).

## This overturns the phase-15e headline

15e concluded from the short diag harness that "the cascade adds nothing measurable" (t = -0.36 on
three seeds). That harness trained for 600 steps. The P1 logs show the full model sitting at the
baseline's level at epoch 10 (0.4942 against 0.4642) and separating only between epoch 10 and epoch
20, i.e. between 1,200 and 2,400 steps. Every diag conclusion was drawn before the effect existed.
See 15k for the fix to the harness and for which conclusions survive.
