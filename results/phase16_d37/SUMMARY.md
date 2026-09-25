# Phase 16 C4 + D-37: where the query-position shortcut lives and what the head is worth without it

VM run of 2026-09-24 (NVIDIA L4), commit `0fe4723`, `AUTOSTOP=1 bash experiments/run_d37.sh full`: C4 17:34–18:00,
VR training 18:00–19:42, CR training 19:42–21:24, evaluation 21:24–23:48 UTC. S1, seed 0, one training run per arm,
E1's schedule (D-30). Every checkpoint `protocol: clean`. [DECISION D-36] [DECISION D-37]

## Checks

* C4.0a: the written-out PEM/PDM with VIP-Seg's cross-term reproduces the inherited modules on all 1,500 episodes,
  max relative logit difference 0.00e+00 (E1 and VIP-Seg released).
* C4.0b: the native passes reproduce C3 (73.20 / 0.92, 75.36 / 0.87).
* E1 re-scored in the same evaluation reproduces E1's own numbers on all four draws, |diff| 0.000.
* D37-T1…T11 (CPU and GPU) passed before the run.

## C4 (D-36): the reshape is the only carrier

| checkpoint | native stored | native swapped | relabel shift | clean stored | clean swapped | relabel shift |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| E1 `last` | 73.20 | 0.92 | +0.912 | 37.91 | 37.91 | +0.000 |
| VIP-Seg released | 75.36 | 0.87 | +0.898 | 35.45 | 35.45 | −0.000 |

C4.1 holds on both. Same weights, only the cross-term changed: order dependence disappears exactly, and the level
falls to 36–38 (below the support rule's 49–52), so the trained heads route most of their prediction through the
positional path.

## D-37: the 2 × 2 (fixed100 mIoU, stored / swapped order)

| | fixed order | random order |
| :--- | :--- | :--- |
| scrambled head | VF = E1: 73.20 / 0.92 (`best` 75.05 / 0.61) | VR: 53.26 / 52.75 (`best` 54.38 / 54.48) |
| clean head | CF = CR (lemma, D37-T5/T6) | CR: 54.84 / 54.84 (`best` 55.11 / 55.11) |

| draw | E1 last | E1 best | VR last | VR best | CR last | CR best |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed100 | 73.20 | 75.05 | 53.26 | 54.38 | **54.84** | **55.11** |
| random600 seed 0 | 72.51 | 74.34 | 50.44 | 52.01 | 51.83 | 53.01 |
| random600 seed 1 | 74.23 | 75.50 | 56.08 | 55.90 | 56.73 | 57.36 |
| random600 seed 2 | 75.50 | 77.23 | 53.42 | 55.18 | 54.76 | 56.15 |
| leak-free (seed 4) | 17.89 | 15.84 | 24.04 | 25.24 | 28.10 | 27.41 |
| support rule, fixed100 (same model's features) | 49.27 | 49.25 | 52.90 | 53.95 | 55.02 | 56.73 |
| oracle (unit), fixed100 | 85.93 | 85.08 | 82.92 | 84.41 | 83.38 | 85.17 |

Per-class IoU on fixed100 (door, floor, sofa, table, wall, window): E1 73.9 / 62.1 / 75.4 / 69.8 / 69.5 / 88.6;
VR 50.5 / 50.5 / 61.3 / 46.5 / 54.7 / 55.9; CR 52.2 / 56.6 / 61.4 / 49.0 / 52.9 / 56.9.

Validation (S1 valid draw, stored order): VR 45.7 → best 54.45 (epoch 48) → 53.07; CR 48.6 → best 55.82 (epoch 48)
→ 55.17.

## Rules (`d37_eval.py decide`)

`last.pt` (headline, D-22):
* **D37.2 equivariance**: CR 54.84 / 54.84, relabel shift −0.0000. Holds.
* **D37.1 origin**: VR gap 0.50, relabel shift +0.004 → the loader's fixed order is the origin of the shortcut.
* **D37.3 clean head**: CR − VR +1.58 [+0.50, +2.66] on fixed100, +1.39 / +0.65 / +1.35 on random600 → the clean
  head is the base.
* **D37.4**: shortcut worth VF − VR +19.94 [+18.40, +21.52] (random600 +22.07 / +18.15 / +22.09); the base's oracle
  gap +28.54 [+27.04, +30.11] (unit), +28.94 (norms kept).

`best.pt`: D37.2 and D37.1 hold; D37.3 tie (+0.73 [−0.32, +1.85], random600 +1.00 / +1.46 / +0.97) → clean head by the
tie rule; VF − VR +20.67; oracle gap +30.06.

## What it establishes

1. **The shortcut is fully traced.** Carrier: the `reshape(72, −1)` cross-term (C4, exact). Origin: the loader's
   fixed query order in training (VR has no positional dependence left). Neither alone produces it.
2. **About 20 of VIP-Seg's ~73–75 points on S1 are the position shortcut** (one training seed per arm; the effect is
   ten times any seed noise measured in phase 16).
3. **Without the shortcut the head adds nothing measurable over plain prototype matching**: CR 54.84 against its own
   support rule 55.02, VR 53.26 against 52.90. The clean head is +1.58 over the scrambled head on `last`, a tie on
   `best`; one seed each, so the size is uncertain but the sign holds on every draw of `last`.
4. **The density cue is the second leak, and large**: from the standard protocol to the leak-free draw CR falls
   54.84 → 28.10, the support rule 55.02 → 32.15. The oracle stays at 61 there.
5. **The real prototype gap is +28.5 points (CR, fixed100), not +12–14**: the earlier gap was measured on heads whose
   prediction came mostly from the query position. This is the quantity the next decision is built on.
