### D-36 — C4: is the cross-term reshape the only carrier of the query position? · `PROPOSED`, diagnostic

* **Problem.** C3 shows that E1 and VIP-Seg depend on the query position (D-35 outcome). By the index algebra of
  C2 (`results/c1/c2_crosscorr.txt`), `reshape(72, −1)` gives query b′ the projected rows of parity b′ and slot w′ the
  rows ≡ w′ (mod 3), so the cross-term can learn a different map for every (position, slot) pair: a *necessary*
  condition for the shortcut. Every other operation of the model is symmetric in the query index [verified: the
  encoder and feature head treat each block alike and couple them only through the batch-wide `torch.std` and, in
  training, BatchNorm, both invariant to the order of the batch (`models/encoder.py:182-189`); PEM/PDM's self-gates
  are per query and per slot (`models/vipseg.py:266-277,370-380`); the gating network and the logits are per query
  (`models/vipseg.py:158-176`)]. Whether the reshape is also *sufficient*, i.e. the only carrier, is not measured.
* **What it does** (`experiments/c4_crossterm_ablation.py`, inference only). A re-implementation of VIP-Seg's
  PEM/PDM forward that reads the inherited modules' weights (never edits them) and computes the cross-term in one of
  two forms: `scrambled` (VIP-Seg's, checked to reproduce the inherited module's output on every episode) and
  `clean`, `A[b, w] = softmax(Q′_bᵀ S′_w / √128)` per query b and slot w, D-01's form with VIP-Seg's scale. E1 and
  VIP-Seg's released head re-run with each form on fixed100, stored and swapped order (C3's scoring).
  * **Implementation** (amended 2026-09-24, before any code). `models/vip_stage.py` gains
    `vip_module_forward(module, query, supports, prototype, form)`: PEM (Eq.9–14) and PDM (Eq.15–18) written out
    from the inherited source, reading the module's own submodules (`maxpool`, `map`, `proto_map`, `reweight`,
    `reweight_s`, `fc`, `fc_qs`, `layer_norm_qs`, `layer_norm`); only the cross-term differs between the forms
    [VIPSEG models/vipseg.py:285-296,387-394]. `VIPStage` gains `cross_form ∈ {native, scrambled, clean}`; `native`
    calls the inherited module exactly as before, so E1's configuration and every state-dict key are unchanged.
    On VIP-Seg's released model the entries of `vip_module` are wrapped in the loaded instance only; no class is
    edited (AGENTS guardrail 2).
  * **Passes per checkpoint** (fixed100): native stored and swapped (a repeat of C3), `scrambled` stored,
    `clean` stored and swapped.
  * **Checks, on every episode; a failure stops the run.** C4.0a: `scrambled` reproduces the native logits,
    max |diff| ≤ 1e-4 · max(1, max |native logit|). C4.0b: the native passes reproduce C3's mIoU (73.20 / 0.92,
    75.36 / 0.87) within 0.01.
* **Rules, fixed before the run.**
  * **Relabel shift** (defined 2026-09-24, before any run). After the swap, the label of the position a block moved
    to is the other episode class's label, which a model can also predict by honest confusion. The raw relabel
    share is therefore not zero for an order-free model; the quantity is its shift, the share of own-class points
    predicted as the other episode class in the swapped order minus the same share in the stored order (0 for an
    order-free model; for E1, 0.913 minus a stored-order share that C3 did not record and C4 does).
  * C4.1 sole carrier: with `clean`, |stored − swapped| ≤ 0.1 mIoU and |relabel shift| ≤ 0.01 on both
    checkpoints. The equivariance argument above predicts exact invariance up to floating point, so a larger
    difference means a second carrier exists and must be found before D-37 is read.
  * C4.2 reported: `clean` stored-order mIoU against `scrambled` (73.20 / 75.36) and the support rule (49.27 /
    51.57). The weights were trained with the scrambled form, so the level is not a model's quality; it measures how
    much of the trained head's prediction runs through the positional path.
* **Cost.** Five passes over fixed100 per checkpoint, about 30 minutes on an L4 (not measured).
* **Order.** C4 runs before D-37's training; a C4.1 failure stops D-37's queue, because a second carrier would
  also be present in the clean arm.
* **Affects.** `experiments/c4_crossterm_ablation.py`, `models/vip_stage.py` (the `clean` form is shared with
  D-37), tests (05 §3.8o).
