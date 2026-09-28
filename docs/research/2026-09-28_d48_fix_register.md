# D-48 fix register (2026-09-28)

Every problem found in D-48 before code, with its fix and where the fix is recorded. The rules live in
`docs/decisions/D-48_correlation-architecture-with-base-exclusion.md`; this file keeps the list in one place.

**Sources.**
- R = the agent review, `docs/research/2026-09-28_d48_math_debate.md`.
- M = the maintainer's own critique, `debate.md` (repository root, not tracked).
- design = the attribution design of amendment 3.
- A1…A4 = the amendment of D-48 that carries the fix.

**Status legend.**
- **done** = implemented and covered by a test of `tests/test_p11.py` (05 §3.8x).
- **wired** = implemented in `experiments/p11_precheck.py` and exercised only by the GPU run.
- **rule** = a reading or a rule that applies when the results or the arms exist.

| # | Src | Problem | Fix | Where | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| F1 | R | The base CE reaches the encoder. "None" covers every novel class, which pulls them toward one direction (M5 precedent: novel oracles 80.84 → 78.03 / 73.72) | Stop-gradient into the encoder; the base MLP trains alone | A1.1 | done (P11-4) |
| F2 | R | ψ · logit(g) lowers the background logit wherever g < 0.5, which covers 57 % of the test background | One-sided ψ · (−log(1 − g)) in every head; g clamped to [1e-4, 1 − 1e-4]. The input-channel form was dropped in A3 | A1.2, A3 | done (P11-2, P11-3) |
| F3 | R | Excluding the targets without renormalising means training never sees (high g, foreground) | Leave-target-out softmax over {non-target base, none}. It is moot at test, since ψ is never trained inside episodes (A4) | A1.2, A4 | done (P11-1) |
| F4 | R | Density trap: the dense class is always base in training and always novel at test | P11.1b reads g on dense (own-condition) novel foreground on valid_raw; gate at ≤ 10 % with g > 0.5, with 5 % also reported (M's Gate 3) | A1.3, A2 | wired |
| F5 | R | P11.1, the oracle exclusion, cannot fail (estimated +3.35 against a bar of 1.0) | P11.1b: a base learner trained on frozen CR features; holds at +1.0 | A1.3, A4 | wired |
| F6 | R | Stored episodes carry no raw labels | `raw_episodes`: seeded draws through P5's checked sampler copy (valid_raw seed 11, training seed 12) | A1.3, A4 | wired (CPU smoke on 8 real episodes) |
| F7 | R | A2 (correlations transfer from base to novel classes) is untested, and the head-level evidence is negative | P11.4: a way-equivariant descriptor probe trained on base episodes, scored against U on novel valid, leak-free, own / other and held-out base; gates arm A | A1.4 | done (P11-15), wired |
| F8 | R | The text prior's history was misstated: P3 already weighted per episode, giving +0.41 | Corrected | A1.5 | done (docs) |
| F9 | R | Every modality goes through the same 6-prototype ridge bank, so extra modalities add no direction; audio double-counts; the background text row is undefined | The rules are scored with [6] off, and [6] only re-ranks the ways (no background row). Text only until D-47. κ is selected on valid, refitted per arm | A1.5, A4 | wired |
| F10 | R | LP and "both" selected on CR do not transfer (−16.6 on D-45's arm B) | Re-selected on each arm's valid; "none" allowed | A1.6 | rule (arms) |
| F11 | R, M | Fixed M = 16 cells: 9.6 % of support blocks have < 20 points per cell (minimum 11.7); 32 background cells bias the max | Adaptive M_c = clamp(⌊n_c / 32⌋, 2, cap). The descriptor is (max, top-2 mean, mean). Background caps {8, 16, 32} are compared in P11.3 | A2 | done (P11-10…12) |
| F12 | R, M | Non-negative features (‖mean unit feature‖² 0.56) compress every cosine into a narrow band | Spaces raw / centred / proj6 / proj8 / white6 / white8 from base-labelled moments, compared in P11.3/P11.4 | A2, A4 | done (P11-14) |
| F13 | M | Full whitening diverges with PR 5.56 | Only truncated whitening with δ = 10⁻³ λ₁; the full inverse is never formed | A2, A4 | done (P11-14) |
| F14 | M | Balanced OT forces support mass into the query background; the semi-relaxed form still forces ≥ P/κ points | Unbalanced OT (KL on both marginals) over the classes, with support-share masses and a leak-free guard | A2, A4 | done (P11-6…9) |
| F15 | M | max(L_OT_bg, ψ·logit g) mixes scales and fires below g = 0.5 | Not adopted; F2's additive one-sided term | A2 | done (P11-3, P11-18) |
| F16 | R | One seed cannot resolve +1.0 | Second seed for any arm passing by less than +2.0; A has seeds 0 and 1 | A1.9, A3 | rule (arms) |
| F17 | R | ψ = 0 and M = 1 ablations are co-adaptation readings; the confusion numbers are in-sample (95 % of test blocks were training blocks) | Labelled as such | A1.8 | rule (reading) |
| F18 | R | The label wrapper must call the sampler once per block | One sampler call per block, checked against P5's index copy on every block | A1, A4 | wired (checked on every call) |
| F19 | M | Additive expectations of 66–70 are not measurements, and "leak-free" was read as the standard draw | Not adopted as expectations; leak-free ≈ 33 is reported separately | — | done (docs) |
| F20 | M | Gate 1 (λ_max/λ_min ≤ 1.5 after whitening on base) holds by construction | Replaced by the energy share of the top r (fit) and the oracle contrast share inside the top-r span (valid) | A2 | done (P11-23), wired |
| F21 | design | [2] used as an input channel in A is entangled with the neck's training, so its effect is not attributable | [2] is decoupled and post hoc in every head | A3 | done (P11-4, P11-18) |
| F22 | design | One seed per arm; B is redundant once [2] is decoupled | B dropped; A gets seeds 0 and 1 | A3 | rule (arms) |
