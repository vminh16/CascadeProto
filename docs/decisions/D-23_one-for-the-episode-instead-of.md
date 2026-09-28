### D-23 — One `S′` for the episode instead of one per class slot (Eq.13) · `PROPOSED`, to be measured

* **Problem.** D-01 item 5 already recorded that Eq.13 writes a single `S′ = φ(F^s)` with no class
  index, and that one `A` per class slot follows VIP-Seg rather than the paper. Phase 16 found that
  this is the one place where our implementation differs from **every** published member of the
  family it belongs to.
* **Evidence (L2 and literature, 2026-09-22).** VIP-Seg's PDM is token-identical to Seg-PN's QUEST
  (`models/quest.py` of github.com/yangyangyang127/Seg-NN) and its PEM to TaylorSeg's APP
  (`models/app.py` of github.com/changshuowang/TaylorSeg); DyPolySeg's PCM prints the same two
  branches. In all of them the correlation is computed after `que.reshape(72, -1)` and
  `sup.reshape(72, -1)` [VIPSEG models/vipseg.py:285-291,387-391]; differentiating each output block
  with respect to each input shows that **every** `A[b′, c′]` depends on both queries and on all
  three class slots, so the published operator is an episode-level correlation, six row-subsampled
  views of one statistic, not a per-class one (research note §4.4, Appendix A). DPA states the shared
  form cleanly — one correlation against the mean of all support features — and measures +13.7 mean
  IoU for it alone (arXiv 2401.16051, Table 3). The ablations that isolate this branch credit it with
  +15.4 (Seg-PN Table 6) and +15.2 (TaylorSeg Table 4) on S0, which is the size of our gap to VIP-Seg's
  head.
* **Why the difference can matter.** With one `A_b` shared by the class rows, the contrast of two
  classes is `fᵀA_bW(p_c − p_c′)`: an episode-adaptive bilinear metric on the support-derived contrast.
  With one `A_{b,c}` per class slot the contrast also contains `(A_{b,c} − A_{b,c′})ψ(·)`, where
  `A_{b,c}` is the channel co-activation of the query with the **whole block** sampled for way c,
  background included, so a class is pulled toward the query whenever its support *scene* resembles it
  — a class bias no label supports.
* **Decision.** Add the switch `cross_attn_support = {class_slots (default), pooled}`. `pooled` is
  Eq.13's literal single `S′`: the pooled tokens of all N·K support blocks are averaged into one
  `F̄^s`, `S̄′ = φ(F̄^s)`, and `A_b = softmax_row(Q′_bᵀ S̄′ / √d) ∈ R^{D×D}` is applied to every class
  row, `P_cross[b,c] = A_b ψ(P_gated[b,c])`. Unlike the published reshape it keeps the queries
  separate, so one query's prediction never depends on another's. No parameter changes.
* **Status: measured and negative (2026-09-23, R1, `results/phase16_r1/SUMMARY.md`).** On S1, T = 1,
  no LMA, 2,400 steps, two seeds, everything else fixed: `pooled` 0.5339 ± 0.0015 against
  `class_slots` 0.5305 ± 0.0070, i.e. **+0.33 points, t = +0.65**, against the rule's +3 with t > 3.
  The same run resolves a 12.50-point difference between a VIP-Seg module and no stage at all
  (t = +7.86), so the budget is not what hides it. The support reading is **not** the cause of the gap
  to VIP-Seg's head. D-01 item 5 stands; the flag stays as an ablation with default `class_slots`.
* **Affects.** `models/eppm.py` (`CrossAttention`), `models/cascadeproto.py`, `train.py`
  (run directories get `_pooled`), 01 §3, 02 §5.2, 05 §3.4 (XATT-11…15).
