### D-31 — P3: where the prototype error sits, and whether a text prior carries novel-class information · `PROPOSED`, beyond the paper

* **Problem.** On E1 (route B's base, D-30) replacing only the prototypes by the query's own class means
  lifts S1 fixed100 from 73.20 to 85.93–87.43 with the same features and the same dot-product decoding
  (`results/phase16_e1/SUMMARY.md`): the error is the prototype that the support and the head produce, not the
  features or the decoding (`docs/research/2026-09-24_text_integration_independent.md` §9.1). A prototype
  can be corrected only from the support, the query's unlabelled points, or prior knowledge (§9.2). Before
  designing a trained module (a point-level query–support neck, §9.4, or a text branch), two things must be
  measured on E1 with no training: which kind of error the oracle fixes, and whether class names carry
  usable information about novel classes.
* **Part A — gap decomposition** (descriptive; it chooses the mechanism of the next decision):
  * rules scored on the same episodes: the support-prototype rule without the head (`F^q n(P_point)ᵀ`), E1,
    and the oracle rule with one common norm (R2-pre); head recovery `r = (E1 − support) / (oracle − support)`;
  * **fixable points** = wrong under E1 and right under the oracle; their share among boundary points (a
    query point whose 16 nearest neighbours in xyz include another label) against interior points; their
    share by support-mask size tercile; Spearman between the per-episode oracle gain (foreground point
    accuracy) and the support foreground fraction.
  * Interpretation bands, stated before the run (not go/stop rules): fixable points enriched ≥ 1.5× at
    boundaries → a local refinement problem; < 1.2× → region-level confusions, i.e. a global prototype
    shift, the target of a point-level query–support neck; Spearman ≤ −0.3 with the support foreground
    fraction → support quality, the target of a prior (text, base classes) or of several prototypes per class.
* **Part B — training-free text prior after the head** (the research agent's T1 plus the entropy weight
  of the independent note §6, `docs/research/2026-09-24_text_integration_math.md` §5):
  * text direction per class `t̂_c` from the base-class bank of the checkpoint (P1's machinery, 1,000 training
    episodes of S1's base classes): **ridge** `t̂_c = n(Bᵀ(ẼẼᵀ + λI)⁻¹ ẽ_c)` with λ = 10⁻³, or **retrieval**
    `t̂_c = n(Σ_j softmax_j(τ cos(e_c, e_j)) B_j)` with τ ∈ {30, 100} (the best values of T0-C: 100 for the ensemble on all three banks, 30 for bare names
    on two);
    `ẽ = n(e − ē)`, ē the mean of the 12 class embeddings (names only);
  * prompts: the repo template, bare names, the 6-template ensemble of T0 and the 12 geometric descriptions
    frozen in the agent's probe (written before any result with them);
  * `T_ic = ⟨n(f_i − μ), t̂_c⟩`; for foreground columns `L′_ic = L_ic + κ γ_e w_i (T_ic − mean_{c′≥1} T_ic′)`;
    background unchanged; `γ_e = max(0, 2 acc_e − 1)` with acc_e the text-only foreground-vs-foreground
    accuracy on the support's foreground points; `w_i = 1` or the normalised entropy of E1's posterior at
    point i (text trusted where the head is unsure);
  * κ ∈ {0.25, 0.5, 1, 2, 4, 8, 16, 32}: the scale of E1's logits is not known in advance, so the grid is
    geometric over two decades;
  * selection on the S1 **valid** draw, test once on fixed100 (P0's protocol, D-22 rules 2 and 3); the
    frozen arm, the same arm with `w = 1`, and an oracle-γ arm (κ per episode chosen with the query labels,
    an upper bound, never a result).
* **Rules, fixed before the run (Part B).**
  * P3.0 mechanism: for the frozen (source, prompt), text-only foreground-vs-foreground accuracy on S1-valid
    query points ≥ 0.60 (the agent's bar; 0.5 is chance), and the alignment of the text direction with the
    oracle's correction (cosine over the points of the two foreground classes between `T_1 − T_2` and
    `(O_1 − O_2) − (L_1 − L_2)`) with an episode-bootstrap CI above 0. Otherwise stop text on this feature
    space.
  * P3.1 go: P3.0 holds, the frozen arm gains ≥ +0.5 on fixed100 with a paired CI above 0, and the oracle-γ
    arm gains ≥ +0.5 → a trained text prior (T2) gets its own decision.
  * P3.2 stop: the oracle-γ arm gains < +0.5 on fixed100 (even a per-episode weight chosen with the labels
    cannot reach the smallest effect worth a training run, P0–P2's +0.5).
  * P3.3 in between: otherwise; report.
  * P3.4 entropy weight: claimed only if the frozen arm with `w = H` beats the same arm with `w = 1` with a
    paired CI above 0 on fixed100.
  * P3.5 collapse watch: a gain with any class below E1 by more than 3 IoU points is reported.
* **Checkpoint.** E1 `last.pt` (`log_r2/s3dis_S1_N2_K1_point_T4_vip_b1/last.pt`); S1 only, S0 held out.
* **Affects.** `models/text_prior.py` (new), `experiments/p3_probe.py` and `experiments/run_p3.sh` (new),
  05 §3.8k (TXT-…).
* **Outcome (2026-09-24, `results/phase16_p3/SUMMARY.md`).** Part B: **P3.0 fails, text stops on this feature
  space**: 0 of 192 arms gain on the valid draw, the frozen arm is −0.00 on fixed100, and a per-episode weight
  chosen with the labels reaches only +0.70 [+0.52, +0.88]; the best text-only fg-vs-fg accuracy is 0.62. Part A:
  support rule 49.27 → E1 73.20 → oracle 85.93 (head recovery 0.65); fixable points are 1.54× enriched at
  boundaries (the pre-registered "boundary-enriched" band) but 83.5 % interior, flat across support sizes
  (Spearman +0.02), and mostly floor and wall predicted as background (recall 0.72 / 0.73 → 0.99 / 0.95 under
  the oracle; background precision 0.861 → 0.977). The head lowers floor below the plain support prototype
  (IoU 66.0 → 62.1).
