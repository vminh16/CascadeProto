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
