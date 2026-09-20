# CascadeProto: what an unofficial re-implementation reproduces, and what it does not

Status: draft of 2026-09-21. Numbers marked `PENDING` are runs still on the VM at the time of writing.

* **Paper.** "CascadeProto", Wang et al., `10069.pdf`. Method in §3 (Eq.1–27), hyper-parameters in
  §4.1, results in Tables 2–6.
* **Reference implementation.** VIP-Seg at the pinned commit, vendored in `models/vipseg.py`; the
  paper names it as the shared encoder (§4.1).
* **This work.** Every equation implemented from the paper, with each ambiguity recorded as a
  numbered decision in `docs/spec/00_SOURCES_AND_DECISIONS.md` (D-01…D-19) and each equation
  transcribed in `docs/spec/02_TENSOR_MATH_SPEC.md`. 272 CPU tests, mutation-checked per sub-phase.
* **Setting for every number below.** S3DIS, fold S0, 2-way 1-shot, 2,048 points per block, one NVIDIA
  L4. Training: 50 epochs × 480 episodes, batch 4, AdamW lr 1e-3 wd 0.1, StepLR halving every 10
  epochs, as §4.1 states. Evaluation: the `fixed100` protocol of D-08, 1,500 cached episodes.

---

## 1. The pipeline is sound

Three independent checks, because a reproduction that fails is only interesting if the harness is
not the reason.

| check | result |
| :--- | :--- |
| VIP-Seg's released S0 checkpoint, scored through **our** `eval.py`, our data and our metric | **0.7197** against its published **0.7220** [PAPER Tab.6] |
| VIP-Seg's own model trained through **our** loop, our data, our loss, 2,400 episodes | **0.6948**, against **0.689** from its own training script at the same point |
| CPU gate | 272 tests, every equation of spec 02 checked against an independent reference implementation |

So the data pipeline, the episode sampler, the loss, the optimiser loop and the mIoU metric all
reproduce VIP-Seg's published behaviour. Whatever the gap below is, it is not the harness.

---

## 2. What the full schedule gives

`fixed100`, 1,500 episodes, `best` checkpoint (D-15). One seed per row.

| configuration | switches | ours | paper [Tab.4] |
| :--- | :--- | ---: | ---: |
| Baseline | `use_lma=false num_stages=0` | 0.4908 | 0.8272 |
| Baseline, prototypes L2-normalised | `… l2norm_point_proto=true` | 0.5244 | — |
| + LMA | `use_lma=true num_stages=0` | `PENDING` | 0.8393 |
| + Entropy Gate | `num_stages=1` | `PENDING` | 0.8535 |
| + Cascade (T = 4) | `num_stages=4 use_adrm=false` | `PENDING` | 0.8741 |
| + ADRM, the full model | defaults | 0.5715 | 0.8853 |

### The increments

The paper's baseline is "a plain VIP-Seg backbone with masked average pooling and single-step
prototype matching" (§4.3). VIP-Seg L2-normalises its prototypes (`models/vipseg.py:142`), so the row
to compare the paper's baseline with is **Baseline + L2**, not Baseline.

| step | ours | paper |
| :--- | ---: | ---: |
| L2-normalising the point prototypes (not printed in the paper; D-10) | +3.36 | — |
| Baseline + L2 → full model | **+4.71** | **+5.81** |

**The relative claim of Table 4 is approximately reproduced. The absolute level is not**: 52.44
against 82.72 for the baseline, 57.15 against 88.53 for the full model.

---

## 3. Why the absolute level is out of reach

The paper's own numbers are internally inconsistent, which is enough to explain part of the gap and
is documented in full in `2026-09-20_paper_vs_code_audit.md`.

* **The baseline row is above the method it is built on.** Table 4's baseline is 81.28 Avg / 82.72 S0,
  while VIP-Seg's own row in Table 2 is 74.15 Avg and its Table 6 S0 is 72.20. Deleting VIP-Seg's
  entire prototype stack would therefore improve S0 by 10.52. The paper states no protocol,
  retraining or normalisation difference that would account for it, and every §4.3 increment is
  measured from this row.
* **Two tables disagree about the same configuration.** Table 4 row 3 (83.91 Avg) and Table 5 `T=1`
  (83.28 Avg) are the same model under D-17's mapping.
* **Eq.14 is dimensionally invalid as printed.** `A ∈ R^{Nq×Ns}` times `ψ(P^{t-1}) ∈ R^{(N+1)×D}` is
  undefined unless `Ns = N+1`, and even then it yields `R^{Nq×D}`, not the `(N+1)×D` that Eq.19 and
  Eq.21 need. Any implementation must discard part of what is printed; D-01 records the choice.
* **Eq.12's output is never used.** `x_gated` appears in Eq.12 and nowhere else; Eq.14 multiplies the
  ungated `ψ(P^{t-1})` and Eq.21's residual is the ungated `P^{t-1}`. Under the prototype reading the
  entropy gate is dead code; under the feature reading it is consumed by Eq.13. Both were measured
  (§4) and neither changes the score.
* **Table 6's parameter budget is unreachable.** Four EPPM stages built exactly from Eq.10–21 cost
  317,580 parameters, already more than the ~0.31M the paper implies for LMA + EPPM + ADRM together,
  and the LMA alone (148,352) exceeds the "+0.12M" of §4.4.
* **Eq.11 cannot do what the abstract claims.** `g = σ(2(θ − H))` with `H ∈ [0, ln 2]` gives `g < 1`
  for every finite `θ`, so the gate can only attenuate, never "amplify low-entropy foreground
  information".

---

## 4. Ambiguities we resolved by measurement

Each was implemented behind a switch, with the default unchanged, and measured on three seeds.

| ambiguity | readings | outcome |
| :--- | :--- | :--- |
| D-02: what Eq.10–12 gates | the prototype, or the features | No measurable difference (t = +0.08). The feature reading is the only one that consumes Eq.12, so it is the better reading of the text, but it changes nothing. |
| D-10: Eq.23's "scaled dot-product matching" | no scale, as the equation prints, or `1/√D` | **No scale.** `1/√D` costs 5.5 points (t = −2.90) and its training loss plateaus at 0.446–0.463 against 0.33–0.39, because it flattens the softmax. |
| D-18: the scale inside Eq.14's softmax | as printed, or standardised projections | As printed. The LayerNorm probe costs 0.9 points and pins the attention near the uniform end. |

Caveat on all three: they were measured with the short diagnostic harness, whose budget was too small
(§5). The D-10 result has a budget-independent mechanism and is likely to survive; the other two
should be treated as untested at the real schedule.

---

## 5. A methodological error of our own

The diagnostic harness trained for 600 optimiser steps, five epochs of the real schedule. The P1
training logs show the full model sitting at the baseline's level at epoch 10 (valid 0.4942 against
0.4642) and separating only between epoch 10 and epoch 20 — between 1,200 and 2,400 steps. Every
conclusion drawn from that harness was therefore drawn before the effect existed, including the claim
we held for several hours that "the cascade adds nothing measurable". The full-schedule run of §2
overturns it. The harness default is now 2,400 steps.

The loop is also not bit-reproducible on CUDA: the identical command at a fixed seed gave 0.5218,
0.5223 and 0.5316 for one configuration and 0.5164, 0.5346 and 0.5029 for another, so differences
below about two points carry no information on one seed.

---

## 6. What is not established

* Every full-schedule number is **one seed**. At the short budget the seed spread was 0.010–0.031, so
  +4.71 is comfortably outside it but +3.36 is not by much.
* Only fold S0 and only 2-way 1-shot. Table 2's other columns and ScanNet were never run.
* Table 6 (parameters, FLOPs, time) is not an acceptance criterion here (D-09), and its FLOPs are a
  lower bound because fvcore does not count the custom CUDA kernels.
* Whether the remaining absolute gap is a difference in the encoder's training, in the episode
  sampler's class balance, or simply unreachable from the paper as written, is not settled.
