# D-48 fix register (2026-09-28)

Every problem found in D-48 before code, with its fix and where the fix is recorded. The rules live in
`docs/decisions/D-48_correlation-architecture-with-base-exclusion.md`; this file keeps the list in one place.

**Sources.**
- R = the agent review, `docs/research/2026-09-28_d48_math_debate.md`.
- M = the maintainer's own critique, `debate.md` (repository root, not tracked).
- A1 / A2 = the amendment of D-48 that carries the fix.

**Status.** Every row is planned. None is implemented or tested yet.

| # | Src | Problem | Fix | Where |
| :--- | :--- | :--- | :--- | :--- |
| F1 | R | The base CE reaches the encoder. "None" covers every novel class, which pulls them toward one direction (M5 precedent: novel oracles 80.84 → 78.03 / 73.72) | Stop-gradient into the encoder; the base MLP trains alone | A1.1 |
| F2 | R | ψ · logit(g) lowers the background logit wherever g < 0.5, which covers 57 % of the test background | Arm A: g as an input channel of the background row. Arm B and ablation: one-sided ψ · (−log(1 − g)). Clamp g to [1e-4, 1 − 1e-4] | A1.2 |
| F3 | R | Excluding the targets without renormalising means training never sees (high g, foreground) | Leave-target-out softmax over {non-target base, none} | A1.2 |
| F4 | R | Density trap: the dense class is always base in training and always novel at test | P11.1b reports g on dense novel foreground; gate at ≤ 10 % with g > 0.5, with ≤ 5 % also reported (M's Gate 3) | A1.3, A2 |
| F5 | R | P11.1, the oracle exclusion, cannot fail (estimated +3.35 against a bar of 1.0) | P11.1b: a base probe on frozen CR features, net gain ≥ +1.0 | A1.3 |
| F6 | R | Stored episodes carry no raw labels | Seeded draws with P5's `draw_episodes` (`query_raw`); budgeted | A1.3 |
| F7 | R | A2 (correlations transfer from base to novel classes) is untested, and the head-level evidence is negative | P11.4: an MLP trained on base descriptors, scored on novel classes, leak-free and own / other; gates arm A | A1.4 |
| F8 | R | The text prior's history was misstated: P3 already weighted per episode, giving +0.41 | Corrected in A1.5 | A1.5 |
| F9 | R | Every modality goes through the same 6-prototype ridge bank, so extra modalities add no direction; audio double-counts; the background text row is undefined | The rules are scored with [6] off. [6] is an add-on with κ selected on valid, refitted per arm, and one coefficient vector per class. Audio is an identity check only | A1.5 |
| F10 | R | LP and "both" selected on CR do not transfer (−16.6 on D-45's arm B) | Re-selected on each arm's valid; "none" allowed | A1.6 |
| F11 | R, M | Fixed M = 16 cells: 9.6 % of support blocks have < 20 points per cell (minimum 11.7); 32 background cells bias the max | Adaptive M_c = clamp(⌊n_c / 32⌋, 2, 16), so every cell has at least 32 points except at the floor of 2. The descriptor is reduced to (max, mean of the top 2, mean), which is defined for M ≥ 2 | A2 |
| F12 | R, M | Non-negative features (‖mean unit feature‖² 0.56) compress every cosine into a narrow band | Descriptor spaces compared in P11.3/P11.4: raw, centred on μ_base, centred and projected on the top r ∈ {6, 8} of Σ_base, centred and truncated-whitened (M's module 1) | A2 |
| F13 | M | Full whitening diverges with PR 5.56 | Full inversion is never used. Truncated whitening is only a P11.3 variant. Prior evidence against it: the P10.4 base metric adds nothing (λ 0.9 − 1.0 = +0.36 [−0.01, +0.72]); query whitening 31–35 < U 55.96; nuisance and discriminant share directions (κ < ρ at every r) | A2 |
| F14 | M | Balanced OT forces support mass into the query background | Semi-relaxed OT as proposed still forces it: Σ_j T_mj = 1/M with column cap κ/P means mass must reach ≥ P/κ = 1,024 points. P11.5 therefore uses unbalanced OT (KL-relaxed marginals, ρ) between prototypes (background included) and query points, as a label-free assignment rule | A2 |
| F15 | M | Taking the max of the OT background logit and the base logit mixes scales: ln-mass ≤ ln(2/2048) ≈ −6.9 against ψ·logit(g) ≈ −0.85ψ at g = 0.3, so the max fires below g = 0.5 | Not adopted; F2's additive one-sided term is kept | A2 |
| F16 | R | One seed cannot resolve +1.0 | Second seed for any arm passing by less than +2.0 | A1.9 |
| F17 | R | ψ = 0 and M = 1 ablations are co-adaptation readings; the confusion numbers are in-sample (95 % of test blocks were training blocks) | Labelled as such in D48.5 | A1.8 |
| F18 | R | The label wrapper must call the sampler once per block | One-call form (20/20 bit-identical) written into the spec and a test | A1, tests |
| F19 | M | Additive expectations of 66–70 are not measurements, and "leak-free" was read as the standard draw | Not adopted as expectations; leak-free ≈ 33 is reported separately | — |
| F21 | design | [2] used as an input channel in A is entangled with the neck's training, so its effect is not attributable | [2] is decoupled and post hoc in every head (one-sided additive term) | A3 |
| F22 | design | One seed per arm; B is redundant once [2] is decoupled | B dropped; A gets seeds 0 and 1 | A3 |
| F20 | M | Gate 1 (λ_max/λ_min ≤ 1.5 after whitening on base) holds by construction | Replaced by the energy share of the top r and the share of d* inside the top-r span, both reported | A2 |
