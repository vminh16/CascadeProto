### D-24 — EPPM-S, a stripped purification stage · `PROPOSED`, beyond the paper

* **Problem.** The printed stage is 15 points behind VIP-Seg's on the same features (CHANGELOG 15e),
  and the research note shows why each of its parts cannot help: `P_diffuse` is common to every class
  and provably cannot change a prediction, so training drives its fusion weight to 0.008–0.082
  (CHANGELOG 15e); the entropy gate is an even pointwise function of the prototype value with one
  scalar [DECISION D-02]; Eq.20's SE gate is pooled over the classes; Eq.21's ReLU truncates the
  update. The family this module belongs to instead carries a **channel-preserving** term beside the
  correlation [VIPSEG models/vipseg.py:262-277], which [DECISION D-19] measured as missing.
* **What the switch does.** `stage_type = {eppm (default), eppm_s, vip}`; `eppm_s` builds
  `models/eppm_s.py::EPPMSharedStage`:
  `P^t = LN(W(P_cross + σ(W_3(Q′ᵀQ′ − S′ᵀS′)/√D) ⊙ ψ(P^{t-1})) + P^{t-1})`, with `P_cross` as in
  Eq.13–14 under either `cross_attn_support` [DECISION D-23]. Dropped: `P_diffuse`, the entropy gate,
  Eq.19's fusion MLP, Eq.20's SE, `w_cls` and the ReLU. 37,888 parameters per stage against 79,395.
  Like VIP-Seg's head it expects L2-normalised prototypes at the top of the cascade
  [VIPSEG models/vipseg.py:142]; supply them with `--l2norm_point_proto true` [DECISION D-10].
* **Honest status.** This is **not** a reading of the paper: it is the CascadeProto-shaped member of
  the QUEST / APP / PEM / PDM family, built to locate the 15 points. Any run using it is outside the
  paper and must be reported as such. It is measured against the printed stage and against
  [DECISION D-25] in R1 (16e).
* **Outcome: negative (2026-09-23, R1, `results/phase16_r1/SUMMARY.md`).** EPPM-S scores 0.4927 ± 0.0115
  against the printed stage's 0.5305 ± 0.0070 and plain L2 prototype matching's 0.5602 ± 0.0222, i.e.
  **−3.79 points against the stage it replaces and −6.75 against using no stage at all**, and 19.26
  below one VIP-Seg PEM on the same run. Removing the parts that provably cannot change a prediction
  does not produce a stage that helps: the static argument of this decision is not sufficient. The
  switch stays as an ablation, off by default. Three candidates for the difference to PEM remain
  untested: the LayerNorm on the self term (named in CHANGELOG 15i, omitted here), PEM's separate
  query and support gates against the difference gate used here, and the query mixing of VIP-Seg's
  reshape, which this stage deliberately avoids.
* **Guard.** With `stage_type ≠ eppm` the EPPM-only switches (`use_gate`, `gate_target`, `eq19_self`,
  `cross_attn_norm`, `fusion_weight`, `diffusion_input`) must stay at their defaults; a non-default
  value raises rather than being ignored, so a run's configuration always describes what it ran.
* **Affects.** `models/eppm_s.py` (new), `models/cascadeproto.py` (`build_stage`), `train.py`
  (run directories get `_eppm_s`), 01 §3–§4, 05 §3.4b (EPS-1…9).
