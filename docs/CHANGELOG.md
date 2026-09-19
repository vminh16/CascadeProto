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

### 10c — CascadeProto as the Table 4 baseline; checkpoint carries its configuration

* **What.** `models/cascadeproto.py` rewritten: `CascadeProtoConfig` holds the switches of 01 §3 (full
  model by default) and `CascadeProto.forward(episode) -> EpisodeOutput`. `train.py` exposes the
  switches, logs the configuration and saves it in `best.pt` / `last.pt`; `eval.py` rebuilds the model
  from the checkpoint's configuration. `tests/test_cascadeproto.py` (CP-1…8) and PIPE-6/7 added;
  `tests/test_smoke_episode.py` and `test_lma.py::test_invariant_6` removed (old API).
* **Why.** Phase 11–13 add LMA, EPPM and ADRM one block at a time; with the baseline row implemented
  first, every phase ends with a model that trains end to end on real data. Only
  `use_lma=false, num_stages=0` is implemented; any other combination raises `NotImplementedError`
  naming the phase, so no half-built configuration can be trained by mistake. Storing the configuration
  in the checkpoint prevents evaluating weights with a different architecture.
* **Sources.** Table 4 baseline "plain VIP-Seg backbone with masked average pooling and single-step
  prototype matching" [PAPER §4.3] = `F^q P_pointᵀ` [DECISION D-17]; Eq.23 without temperature or L2
  normalisation, with the ablation flags `logit_scale`, `l2norm_point_proto` [DECISION D-10];
  `use_lma=false` drops `L_GMMN` [DECISION D-17].
* **Verification.** CP-1…8 in float64 (element-wise logits, way equivariance, query independence, flags,
  gradients, learning on a structured episode, exact state_dict round trip); PIPE-7 checks that
  eval.py restores the stored configuration. Mutation check: eight wrong variants all fail. A first
  version of CP-8 used random labels, which no model can learn; it was replaced by an episode whose
  colours depend on the class, as real data does.

### 10b-fix — the VIP-Seg encoder couples the blocks of one call

* **What.** ENC-3, ENC-4 and ENC-7 rewritten; spec 02 §2, 01 §2.1 and 05 §3.6b/§3.8 corrected; CP-4
  renamed. No model code changed.
* **Why.** On the GCP L4, ENC-3/4/7 failed with differences up to 0.064: a block's features changed
  when other blocks of the same encoder call changed. The encoder standardises three intermediate
  tensors with a single mean and standard deviation over the whole batch tensor (all blocks, points and
  channels), in training and evaluation [VIPSEG models/encoder.py:281-283,413-415,582-584]. The spec's
  claim that blocks are encoded "independently" was therefore wrong; they are separate samples for
  FPS/kNN, but coupled through these statistics. This is VIP-Seg's own behaviour and produced its
  72.20, so the inherited encoder stays unchanged.
* **Consequence.** Features equal VIP-Seg's only with VIP-Seg's batch composition: one encoder call
  for the N·K support blocks of an episode and one for its queries [VIPSEG models/vipseg.py:79-89].
  `encode_episode` does exactly this and `train.py` / `eval.py` forward one episode at a time, so no
  code change is needed; the rule is now written in 02 §2 so later phases never batch several
  episodes into one encoder call.
* **Verification.** ENC-3 documents the coupling; ENC-4 checks the composition exactly
  (`torch.equal`); ENC-7 compares an episode's features with VIP-Seg's own path, including its
  `permute`/`view` of the loader layout, to 1e-6. ENC-6 (same batch, VIP-Seg weights) had already passed.

### 10d-fix — validation and test episodes were identical (defect introduced in phase 9a)

* **What.** `pipeline/episodes.py::build_eval_dataset` seeds the valid set with a separate numpy seed
  stream `[seed, 1]`; the test set keeps the integer seed. PIPE-8 (CPU) and DATA-7 (real data) added;
  spec 04 §6.1 and 05 updated.
* **Why.** The phase-10e run on the VM gave valid mIoU 0.4601 and test mIoU 0.460089 for the same
  checkpoint. Both sets were built by `MyTestDataset` with `np.random.seed(0)` and the same code path,
  so the 1,500 valid episodes were the 1,500 test episodes, and D-15's "best on valid" would have been
  "best on test". VIP-Seg draws the two sets independently [VIPSEG runs/training.py:52-66].
* **Compatibility.** The test set's seeding is unchanged, so the cached test episodes and the sanity
  result 0.719687 stay valid. The valid cache built before this fix
  (`<data>/vipseg_S_0_N_2_K_1_episodes_100_pts_2048`) holds test episodes and must be deleted.
* **Verification.** PIPE-8 with a recording stand-in for `MyTestDataset`; mutation check: reusing the
  test seed for the valid set, or changing the test seed, both fail PIPE-8. DATA-7 builds both real sets
  and finds no shared episode.

### 10e — end-to-end run of the baseline on the GCP L4

* **Gates.** `pytest -m "not clip"`: 117 passed, including ENC-1…8 with VIP-Seg's released weights.
* **DATA-5 (dry run).** One step on 4 real episodes, loss 1.5253; 5 valid episodes scored.
* **Short training.** Table 4 baseline (`--use_lma false --num_stages 0`), S3DIS S0 2-way 1-shot,
  2 epochs × 480 episodes, 85 s per epoch: loss 0.8304 → 0.4814; valid mIoU 0.4601 (valid = test
  episodes at that time, see 10d-fix).
* **DATA-6.** `eval.py` on `last.pt`: 5-episode dry run 0.5267; all 1,500 test episodes 0.460089.
  The checkpoint reloads through its stored configuration and fixed projections (ENC-8, PIPE-7).
* **Reading.** Loss falls and the test mIoU is well above chance after 2 of 50 epochs; the full
  schedule would take about 71 min of training plus validations. Paper numbers are not expected from
  this row: Table 4's baseline reaches 81.28 (average of S0 and S1) after full training.

---

## Phase 11 — LMA and GMMN (Table 4 "+ LMA" row)

D-05 locked by the maintainer before this phase: one modality per run, `E_fused := E_adapted^(m)`,
generator input `[E_fused; z]` of width 2D. `eval_noise=mean_of_M` raises until M is chosen (D-06).

### 11a — decoupled GMMN loss (Eq.7–8)

* **What.** `loss/gmmn_loss.py` rewritten: `pairwise_sq_dist`, `rbf_kernel`, `mmd`,
  `gmmn_loss(p_modal [N+1,D], p_point [N+1,D], fg_mode, detach_point)`. `tests/test_lma_gmmn.py`
  (MMD-1…7) replaces `tests/test_lma.py`. `loss/segmentation_loss.py` deleted: its class-weighted CE
  (audit H-series) was dead code since phase 9, when `pipeline/model_api.py::episode_loss` became the
  objective of Eq.26–27.
* **Why.** Audit H5/M3: the old loss took `sqrt(mmd² + eps) − sqrt(eps)` by default, averaged N
  per-class 1-vs-1 MMDs for the foreground, and clamped the result at 0 (no gradient near a perfect
  match). Eq.7 is the squared MMD; D-04 compares the N foreground rows as one set.
* **Sources.** Squared MMD with the six bandwidths [PAPER Eq.7]; weights 0.1 / 1.0 [PAPER Eq.8];
  joint foreground set, no detach, flags `gmmn_fg_mode`, `gmmn_detach_point` [DECISION D-04].
* **Deviation from the old spec text.** 02 §9 wrote the squared distance as
  `‖x‖² + ‖y‖² − 2x·y` with a clamp. At norm 50 this gives `k(x, x) = 6 − 2.6e-11` in float64 (worse
  in float32). The sets hold at most N+1 = 4 rows, so the distance is now computed from the
  differences: never negative, exactly 0 on the diagonal. 02 §9 and 03 §4 updated.
* **Verification.** 26 CPU tests, float64, 1e-12, against Python-scalar references (kernel, distances,
  triple-sum MMD, `2(6 − k)`, joint vs per-class, 0.1/1.0 split, analytic background gradient,
  `gradcheck`, detach flag, input validation). Mutation check: 14/14 wrong variants killed
  (list in 05 §3.3).

### 11b — modality adapter and prototype generator (Eq.4–6)

* **What.** `models/lma.py` rewritten: `ModalityAdapter` (Eq.5), `PrototypeGenerator` (G of Eq.6),
  `LearnableModalityAdapter(eval_noise)` with `forward(E_CLIP [N+1,512]) -> P_modal [N+1,128]`. The CLIP
  helpers of the old file move to phase 11c. LMA-1…3 and three further tests added to
  `tests/test_lma_gmmn.py`.
* **Why.** Audit M-series: the old module always drew `z`, so two evaluation passes differed by up to
  0.249 in the logits (D-06 requires `z = 0`); it carried a batch axis the episode contract does not
  have, and used in-place ReLUs.
* **Sources.** Adapter `Linear(512→128) → LN → ReLU → Dropout(0.1) → Linear(128→128)` [PAPER Eq.5]
  [PAPER §4.1] [DECISION D-16]; G = three-layer MLP on `[E_fused; z]`, `E_fused := E_adapted^(m)`
  [PAPER Eq.6] [DECISION D-05] [DECISION D-16]; `z ~ N(0, I)` per forward in training, `z = 0` in
  evaluation, `eval_noise=sample` as ablation, `mean_of_M` raises [DECISION D-06].
* **Verification.** 11 CPU tests, float64, 1e-12: parameter counts 82,432 / 65,920 (sum 148,352 = 01 §4),
  forward equal to Eq.5–6 written out with explicit operations, deterministic evaluation with `z = 0`,
  exact replay of the training random stream (dropout mask, then z), z statistics, gradients to every
  parameter through `L_GMMN`. Mutation check: 14/14 wrong variants killed; one equivalent mutant noted
  in 05 §3.3.

### 11c — CLIP text front-end (03 §2.1)

* **What.** New `models/clip_text.py`: `episode_prompts`, `load_clip(variant, device)` (once per
  process, frozen), `ClipTextEmbedding(variant, encode=None)` mapping class names to
  `E_CLIP [N+1, 512]` with a per-prompt cache. `tests/test_clip_text.py` (TXT-1…6, EP-5, EP-6).
* **Why.** Audit M-series and L3: the old helper used the background prompt "background clutter"
  (D-13 says "background"), fell back to `ViT-B/32` without a message when the `ViT-B/16` file was
  missing, and reloaded CLIP on every call (3.07 s and 2.37 s per call in the audit).
* **Sources.** Foreground prompt [PAPER Fig.1]; background prompt, class names verbatim, default
  `ViT-B/16`, float32 + L2 norm, frozen and cached CLIP [DECISION D-13]; row order of P_point
  [PAPER Eq.3].
* **Design.** CLIP is a plain attribute, not an `nn.Module` child: checkpoints stay free of its 150M
  parameters and the model's `.double()` in tests cannot change it. On a GPU `clip.load` keeps the
  released float16 weights; outputs are cast to float32 before the norm, so CPU and GPU embeddings can
  differ in the last float16 bits (recorded in 03 §2.1).
* **Verification.** 5 CPU tests with a recording float16 stand-in; 3 `clip` tests with the real
  `ViT-B/16`, run locally on the CPU (8 passed): load count, frozen parameters, bit-exact equality with a
  direct CLIP call, no fallback variant. Mutation check: 12/12 wrong variants killed.

### 11d — "+ LMA" row wired into CascadeProto, train.py and eval.py

* **What.** `CascadeProtoConfig` gains `clip_variant`, `eval_noise`, `gmmn_fg_mode`,
  `gmmn_detach_point` (defaults = D-13, D-06, D-04). `CascadeProto(config, feature_extractor,
  text_embedding)` builds the LMA when `use_lma` and returns `F^q (P_point + P_modal)ᵀ` and `L_GMMN`.
  The modality check moved from `train.build_model` into the configuration. `train.py` exposes the four
  switches and logs `L_GMMN` per step (dry run) and per epoch; `eval.py` logs the configuration read
  from the checkpoint. CP-9…13 added, CP-7/8 run for both rows, PIPE-7 now round-trips an LMA
  checkpoint.
* **Why.** Table 4 "+ LMA" row = `use_lma=true, num_stages=0` with prediction `F^q (P^0)ᵀ` [DECISION
  D-17]; `L_total = L_seg + 1.0·L_GMMN` [PAPER Eq.26].
* **Sources.** `P^0 = P_point + P_modal` [PAPER Eq.9]; GMMN between `P_modal` and `P_point`
  [PAPER Eq.8] [DECISION D-04]; one modality, text only [DECISION D-05] [DECISION D-13]; z = 0 at
  evaluation [DECISION D-06].
* **Compatibility.** Checkpoints of phase 10 lack the new fields; their defaults rebuild the same
  baseline (CP-13), so `last.pt` of 10e still evaluates.
* **Verification.** CPU gate 146 passed. Mutation check: 12/12 wrong variants killed after CP-11 was
  strengthened (its fixture classes were already sorted, so a sorting bug survived the first run).
* **Incident.** During the mutation check a `git checkout -- models/cascadeproto.py` restored the
  committed phase-10 file over the uncommitted 11d version; the file was rewritten from the session
  and the full CPU gate re-run before this commit.
