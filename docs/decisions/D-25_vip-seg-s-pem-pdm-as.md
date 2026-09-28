### D-25 — VIP-Seg's PEM/PDM as cascade stages · `PROPOSED`, reference only

* **Problem.** "VIP-Seg's head is worth about 17 points more than ours through the same loop"
  (CHANGELOG 15e) was measured by training VIP-Seg's **whole model**, which also differs in its
  prototype normalisation, its logits and its gating. To attribute the gap to the stage itself, the
  same pipeline must run VIP-Seg's stage and ours with everything else held fixed.
* **What the switch does.** `stage_type = vip` wraps the inherited
  `models/vipseg.py::PrototypeEnhancementModule` / `PrototypeDifferenceModule` in
  `models/vip_stage.py::VIPStage`, which reproduces VIP-Seg's alternation (PEM on even steps, PDM on
  odd ones) and the outer residual it adds to the PDM output [VIPSEG models/vipseg.py:154-160]. The
  modules are imported, never edited (AGENTS guardrail 2), and imported lazily, because
  `models.vipseg` needs `pointnet2_ops` (gate G2). `cross_attn_scale` and `cross_attn_support` are
  fixed inside VIP-Seg's code, so a non-default value raises.
* **Known property, kept on purpose.** VIP-Seg's `reshape(72, -1)` makes one query's prediction depend
  on the other queries of the episode (research note §4.4). This is a reference configuration, not a
  design to copy; our own stages keep the queries separate.
* **Reporting.** A number produced with `stage_type=vip` is VIP-Seg's module inside our pipeline, not
  CascadeProto, and must be labelled that way.
* **Measured (2026-09-23, R1, `results/phase16_r1/SUMMARY.md`).** One PEM on our prototypes:
  **0.6852 ± 0.0036** on S1 after 2,400 steps, **+15.47 over the printed stage (t = +27.78)** and
  +12.50 over no stage at all. The 15 points that CHANGELOG 15e measured between two training scripts
  are reproduced inside one pipeline with only the stage changed, with the caveat that the printed
  stage runs without the L2 input that VIP-Seg's head gets, worth up to 3.4 of those points
  (`results/phase16_r1/SUMMARY.md`). Rule R1.4 of `run_r1.sh` therefore
  fires: phase 16 continues on **route B**, adding to VIP-Seg's head rather than repairing the printed
  stage. VIP-Seg's released S1 checkpoint is cited at 0.7609, so this is a real starting point and the
  head remains VIP-Seg's contribution.
* **Affects.** `models/vip_stage.py` (new), `models/cascadeproto.py`, `train.py` (run directories get
  `_vip`), 01 §3, 05 §3.4c (VIPS-1…5).
