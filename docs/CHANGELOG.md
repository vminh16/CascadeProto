# Change log

One entry per commit-sized step from phase 10 onwards, newest first. Each entry says what changed,
why, which spec lines it implements, and how it was verified, so any line of code can be traced
back to its source. Earlier phases are summarised in [00 §7](spec/00_SOURCES_AND_DECISIONS.md).
Find the commit of an entry with `git log --oneline --grep "<step id>"`.

---

## Phase 10 — backbone, point prototypes, interim model

### 10a — point prototypes (Eq.3)

* **What.** `models/prototypes.py::point_prototypes(F_s [N,K,P,D], Y_s [N,K,P]) -> [N+1, D]`;
  `tests/test_prototypes.py` (PROTO-1…10). Spec 02 §3 gains the empty-mask rules; 05 §3.2 lists the
  stricter tests.
* **Why.** Audit C2: the old `extract_point_prototypes` flattened all ways into one point set, so with
  the loader's binary masks the first foreground prototype became the union of all ways and the others
  were zero. The new function pools per way with explicit mask weights and never reshapes across the
  way or shot axes (AGENTS guardrail 6).
* **Sources.** Eq.3 [PAPER §3.2]; per-way foreground over the K shots and background over all ways and
  shots [VIPSEG models/vipseg.py:108-130]; empty background → `0.1·1` [VIPSEG models/vipseg.py:111-113];
  no L2 normalisation [DECISION D-10]. An empty foreground raises: a support block is a candidate only if it holds more
  than `max(0.05·n, 100)` target points [VIPSEG dataloaders/s3dis.py:54-57], and the loader then keeps
  all of them (n < 2048) or `int(ratio·2048) ≥ 102` of them [VIPSEG dataloaders/loader.py:41-47], so
  every support block carries at least 101 foreground points; an empty mask can only come from a bug.
* **Verification.** PROTO-1…10 (float64, exact up to 1e-12): manual masked means per row, equality with a
  line-by-line port of VIP-Seg's loop, locality across ways, invariance to point and shot order,
  equivariance to way order, affine equivariance, analytic gradient `mask / count`, empty-mask rules,
  input validation. Mutation check: six wrong variants (per-shot means, per-way background means,
  L2 normalisation, background from way 1 only, audit C2's flattened ways, zero empty background)
  each fail 1–6 tests; none survives.

### 10b — shared point feature extractor (Eq.2)

* **What.** `models/vipseg_backbone.py` rewritten as `PointFeatureExtractor` (`forward`,
  `encode_episode`, `load_vipseg_weights`); `tests/test_feature_extractor.py` (FEAT-1…8, CPU) and
  `tests/test_encoder.py` (ENC-1…8, GPU). `tests/test_backbone.py` removed (tested the deleted API).
* **Why.** Audit C3/M1 and guardrail 6: the old backbone guessed the input layout from
  `shape[i] in (3, 6, 9)`, padded 3-channel input with invented colours, and clamped the norm. The new
  one accepts only `[B, 2048, 9]`, copies VIP-Seg's head layer for layer, and encodes every block as its
  own sample.
* **New finding.** The VIP-Seg encoder draws two fixed random projections (`vv`, `ww`) at construction
  and keeps them as plain attributes [VIPSEG models/encoder.py:619-620]; `state_dict` does not contain
  them. VIP-Seg is unaffected because it pickles the whole model, but our `train.py` saves a
  `state_dict`, so `eval.py` would have rebuilt the model with new projections and scored a different
  network. They are now buffers (same values), saved and restored with the weights; ENC-8 checks the
  round trip on the real encoder. Recorded in 01 §2.1.
* **Sources.** Encoder configuration and head [VIPSEG models/vipseg.py:34-53]; per-point channel L2
  norm without clamp and the separate support/query batches [VIPSEG models/vipseg.py:79-97]; 9-channel
  input read as rgb + XYZ [VIPSEG models/encoder.py:645].
* **Verification.** CPU: 14 FEAT tests (float64, 1e-12). Mutation check: twelve wrong variants all fail
  (see 05 §3.2b). GPU (ENC-1…8) runs on the VM with VIP-Seg's checkpoint; ENC-6 compares our features
  with VIP-Seg's own forward path.
