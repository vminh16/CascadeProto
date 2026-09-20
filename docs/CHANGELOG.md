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

### 11e — VM run of the "+ LMA" row on the GCP L4

* **Gates.** `pytest -m "not clip"`: 172 passed; `pytest -m clip`: 3 passed (real `ViT-B/16` on the GPU).
* **Dry run.** One step on 4 real episodes: loss 4.8252, of which `L_GMMN` 2.4429 (baseline dry run
  1.5253). The higher start is expected: the untrained generator adds a random `P_modal` to
  `P_point`, whose rows have norm ≤ 1 because point features are L2-normalised.
* **Short training.** S3DIS S0 2-way 1-shot, 2 epochs × 480 episodes, 93 s per epoch (baseline 85 s):
  loss 1.4242 → 0.6941, `L_GMMN` 0.5022 → 0.1536, so `L_seg` ≈ 0.922 → 0.541 (baseline 0.830 → 0.481).
  Valid mIoU 0.4345 on the independent valid set of 10d-fix.
* **Test.** `eval.py` on `last.pt`, 1,500 episodes: 0.429550 (baseline row after 2 epochs: 0.460089).
* **Reading.** The integration works: CLIP loads once on the GPU, `L_GMMN` falls, the checkpoint
  rebuilds its configuration. The −3.1 points against the baseline come from one seed after 2 of 50
  epochs and are not a comparison with Table 4 (+1.21 for LMA after full training). The adapter has
  seen the text of the 6 training classes only; at test time `P_modal` comes from 6 unseen class names,
  so an early, undertrained adapter can hurt. Whether LMA helps is decided by the full-schedule runs of
  phase 14, not here.

---

## Phase 12 — EPPM and the cascade (Table 4 "+ Entropy Gate" and "+ Cascade" rows)

Before this phase the maintainer closed D-01 step 5 (one attention matrix per query, class slot and
shot, as in VIP-Seg) and scoped the flags: `cross_attn=two_hop`, `gate_target=features` and
`diffusion_input=pre_relu` raise `NotImplementedError`; `use_gate`, `cross_attn_scale` and
`fusion_weight` are implemented. Cross-checked against the paper (§3.4, Eq.10–23) and VIP-Seg's
`PrototypeEnhancementModule` [VIPSEG models/vipseg.py:196-310].

### 12a — entropy gate (Eq.10–12)

* **What.** `models/eppm.py` rewritten from scratch, starting with `channel_entropy` and
  `EntropyGate(enabled)`. `tests/test_eppm.py` rewritten (GATE-1…3 plus the disabled gate).
* **Why.** The old file carried the rejected two-hop attention and an API whose batch axes broadcast
  support against query (audit); it is replaced stage by stage in 12a–12d.
* **Sources.** Eq.10 with natural log and ε = 1e-8, Eq.11 `g = σ(2(θ − H))`, Eq.12 [PAPER §3.4];
  θ learnable, initialised at 0.5, one per stage [PAPER Eq.11] [PAPER §3.5]; gate applied to the
  prototype channel-wise [DECISION D-02]; clamp of p [DECISION D-16]; `use_gate=false` means g ≡ 1
  [DECISION D-17].
* **Verification.** 8 CPU tests, float64: entropy against Python floats to 1e-12 on [−1000, 1000],
  bounds `[0, ln 2]`, gate range [0.4046, 0.7311] at θ = 0.5, entry-wise formula, locality,
  analytic θ gradient `Σ 2·P·g(1 − g)`, `gradcheck`. `H(x) = H(−x)` holds to 2e-15, not bit for bit
  (`σ(−x)` and `1 − σ(x)` round differently). Mutation check: 13/13 killed.

### 12b — cross-attention refinement (Eq.13–14, D-01)

* **What.** `pool_tokens` and `CrossAttention(scale)` in `models/eppm.py`: `attention(F^s, F^q) ->
  A [B_q, N+1, K, D, D]`, `forward(P_gated, F^s, F^q) -> P_cross [B_q, N+1, D]`. XATT-1…6 plus pooling,
  `gradcheck` and flag tests.
* **Why.** The old module implemented the two-hop point attention that D-01 rejected.
* **Sources.** φ = 1×1 convolution shared by query and support, d = 72 [PAPER Eq.13] [VIPSEG
  models/vipseg.py:219]; softmax scaled by √d [PAPER Eq.14]; `P_cross = A·ψ(P)` with ψ linear
  [PAPER Eq.14] [VIPSEG models/vipseg.py:222,296]; MaxPool 32 → 64 tokens and way-mean background slot
  [VIPSEG models/vipseg.py:213,244,248-249]; one A per (query, class slot, shot), average over shots
  taken on `P_cross` [DECISION D-01].
* **Cross-check with VIP-Seg.** VIP-Seg's `crosscor` has the same `[128, 128]` channel correlation,
  softmax over the last axis and `crosscor @ ψ(P)`; the two deliberate differences are the scale
  (√72 as printed, VIP-Seg √128, available as `cross_attn_scale=sqrt_D`) and the per-(query, class)
  product instead of VIP-Seg's `reshape(proj_dim, -1)`, which interleaves classes (D-01).
* **Verification.** 12 new CPU tests, float64, 1e-12, against an explicit loop that uses VIP-Seg's
  own pooling call. Mutation check: 18/18 killed.

### 12c — prototype diffusion (Eq.15–18)

* **What.** `prototype_diffusion(F^s, F^q) -> P_diffuse [B_q, D]` in `models/eppm.py`, no
  parameters. DIFF-1…5 added.
* **Sources.** Eq.15–18 with τ = α = 0.5 [PAPER Eq.15–18]; `mean(F, dim=1)` read as the mean over
  points, support mean over all ways and shots, one vector broadcast to all class rows
  [DECISION D-16]; degeneracy with ReLU features documented, not changed [DECISION D-14].
* **Verification.** 6 CPU tests, float64, 1e-12: every (query, channel) against Python floats on
  mixed-sign features with all four mask cases, the degenerate `(q_ch + s_ch)/4` case, the
  non-degenerate `q/4` and `s/4` channels of D-14, strict threshold, invariance to way order and
  grouping, query independence, gradient flow. Mutation check: 13/14 killed; the survivor (union
  instead of intersection for `m_common`) is an equivalent mutant at α = 0.5, shown in 05 §3.4.

### 12d — adaptive fusion, SE, class weights, output; one EPPM stage (Eq.19–21, 23)

* **What.** `FusionOutput(fusion_weight)`, `EPPMStage(use_gate, cross_attn_scale, fusion_weight)`,
  `class_weights`, `stage_logits` in `models/eppm.py`. STAGE-0…3 and FUSE-1…4 (plus FUSE-3b) added.
* **Sources.** `w = softmax(f_fusion([P_cross; P_diffuse]))`, one pair per query pooled over classes
  [PAPER Eq.19] [DECISION D-11]; SE `σ(W_2 ReLU(W_1 AvgPool_c))` with r = 4 [PAPER Eq.20]
  [DECISION D-16]; `w_cls = [0.8, 1, …, 1]` fixed [PAPER §3.4]; `P^t = LN(W_out ReLU(P_weighted) +
  P^{t−1})` with the ungated `P^{t−1}` [PAPER Eq.21] [DECISION D-02]; `L^t = F^q (P^t)ᵀ` per query
  without scale [PAPER Eq.23] [DECISION D-10]; `P_diffuse` broadcast to every row [DECISION D-16].
* **Open point, not changed.** Eq.20 and Eq.21 print `W_1`, `W_2`, `W_out` without bias terms,
  while Eq.5 prints its biases explicitly; D-16 (locked) uses `nn.Linear` with bias, which the
  79,395-parameter count of 01 §4 includes. Table 6 cannot settle it (D-09: the described layers
  already exceed the paper's +0.12M by far). The biases add 288 parameters per stage.
* **Verification.** 42 EPPM tests in total, float64; the whole stage matches the written-out
  reference to 1e-11 (the composition of attention, LayerNorm and softmax accumulates a few ulps
  beyond 1e-12). Mutation check: 23/23 killed after FUSE-3b was added (the ReLU before `W_out` was
  untested while `P_weighted` stayed non-negative on the fixture).

### 12e — cascade wired into CascadeProto and train.py (Table 4 "+ Entropy Gate", "+ Cascade")

* **What.** `CascadeProtoConfig` gains `cross_attn`, `cross_attn_scale`, `gate_target`,
  `fusion_weight`, `diffusion_input`; `CascadeProto` builds `num_stages` independent `EPPMStage`s,
  copies `P^0` per query and returns `L^T`. `use_adrm=true` with T ≥ 2 raises "phase 13"; the three
  unimplemented flag values raise. `train.py` exposes the five flags and tags run folders with
  `_nogate`. CP-14…16 added; CP-6/7/8 extended to the new rows.
* **Sources.** Cascade `P^0 → EPPM_1 → … → P^T` with independent stages [PAPER Eq.22] [PAPER §3.5];
  `L^t = F^q (P^t)ᵀ` [PAPER Eq.23]; rows of Table 4 and "prediction `L^T` without ADRM"
  [DECISION D-17]; `P^0` copied once per query [PAPER Eq.14–15] [VIPSEG models/vipseg.py:145].
* **Note.** With T = 1 ADRM is a softmax over one stage, i.e. weight 1, so the "+ Entropy Gate" row is
  exact with either value of `use_adrm` (CP-15). The default configuration (full model) still raises
  until phase 13; the "+ Cascade" row needs `--use_adrm false`.
* **Verification.** CPU gate 206 passed. Mutation check: 14/14 killed after CP-14 gained the
  `logit_scale` case.

### 12f — VM check · PENDING

* **Status.** Not run: the VM is not reachable at the moment (2026-09-19). Phase 12 is verified on the
  CPU only (206 passed, mutation checks of 12a–12e) and is **not yet closed**.
* **To run when the VM is back.**
  `git pull && pytest -m "not clip" -q && pytest -m clip -q`, then
  `python train.py --dataset s3dis --data_path datasets/S3DIS/blocks_bs1_s1 --cvfold 0 --n_way 2 --k_shot 1 --use_lma true --num_stages 4 --use_adrm false --dry_run true`.
* **What only the VM can show.** The EPPM cascade on the real VIP-Seg encoder (CUDA-only
  `mamba_ssm`, `pointnet2_ops`), float32 on the GPU, memory and time per step.

---

## Phase 13 — ADRM and the full model

Maintainer decisions before this phase: no `W_g` for T = 1 (the prediction is `L^1` either way and
`W_g` could never be trained); the D-16 biases of `W_1`, `W_2`, `W_out` stay although Eq.20–21 print
none.

### 13a — dynamic routing (Eq.24–25)

* **What.** `models/adrm.py` rewritten as `DynamicRouting(num_stages)` with `weights(F^q)` and
  `forward(stage logits, F^q)`; `tests/test_adrm_loss.py` (ADRM-1…4 and extras) replaces
  `tests/test_adrm.py`.
* **Why.** The old module had the right formula (audit) but the pre-rewrite batch API.
* **Sources.** `w_gate = softmax(W_g AvgPool(F^q))`, `W_g ∈ R^{T×D}` without bias, `L_final = Σ_t
  w_gate^(t) L^t` [PAPER Eq.24–25]; VIP-Seg's gating layer has a bias [VIPSEG models/vipseg.py:190],
  the paper does not print one and wins (02 §6); T = 1 handled without `W_g` [DECISION D-17].
* **Verification.** 12 CPU tests, float64, 1e-12. The first `gradcheck` on a full 2048-point query
  crashed the process (numerical Jacobian over 262,144 inputs); it now runs on 16 points, which is
  equivalent because ADRM only averages over points. Mutation check: 12/12 killed.

### 13b — full model: ADRM on the cascade; loss and ablation tests

* **What.** `CascadeProto` keeps every stage's logits and routes them with `DynamicRouting` when
  `use_adrm` and T ≥ 2; the "phase 13" error is gone, so the default configuration (full model)
  trains. CP-17/18 added, CP-7/8 cover the full model; LOSS-1…3 in `tests/test_adrm_loss.py`;
  new `tests/test_ablation_switches.py` (ABL-1…3).
* **Sources.** `L_final = Σ_t w_gate^(t) L^t` [PAPER Eq.24–25]; `L_total = CE(L_final) + 1.0·L_GMMN`,
  unweighted, no per-stage loss [PAPER Eq.26–27]; rows of Table 4 and depths of Table 5
  [PAPER Tab.4–5] [DECISION D-17].
* **Parameter budget.** Added modules = 148,352 (LMA) + 4 × 79,395 (EPPM) + 512 (`W_g`) = 466,444,
  as 01 §4 predicts (CP-17).
* **Verification.** 109 tests in the model, loss, ablation and pipeline files pass on the CPU.
  Mutation check of the wiring: 6/6 killed.

### 13c — full model with the real CLIP (EP-1…4)

* **What.** New `tests/test_episode.py` (marker `clip`): the default configuration on one episode
  with the real frozen `ViT-B/16`, float32, stand-in encoder. EP-1 follows the model contract of
  `pipeline/model_api.py` (logits and `L_GMMN`) instead of the pre-rewrite list of outputs.
* **Verification.** 4 passed locally on the CPU: output contract, finite loss with a gradient on
  every parameter, one AdamW step (lr 1e-3, wd 0.1) changes every parameter and keeps it finite,
  deterministic evaluation.

### 13d — status and layout docs brought up to date

* **What.** README status block, AGENTS status line and repository tree (new files `prototypes.py`,
  `clip_text.py`, `prepare_s3dis.py`, `verify_s3dis.py`; the removed `download_and_prepare_s3dis.py`),
  01 "Code status", and a history note on the audit. No code change: the old `models/adrm.py`,
  `tests/test_adrm.py`, `tests/test_lma.py`, `loss/segmentation_loss.py` were already replaced or
  removed in 11a–13a, and no tracked file still uses the pre-rewrite APIs.
* **Remaining open items.** GPU checks of phases 12–13 (12f, 13e); full training runs and the
  comparison with Tables 2–5 (phase 14); image and audio modalities; `eval_noise=mean_of_M`.

### 13e — VM check of phases 12 and 13 · PENDING

* **Status.** Not run; the VM is unreachable (2026-09-19). Phases 12 and 13 are verified on the CPU
  only (240 G1 tests, 11 `clip` tests locally, mutation checks) and are **not yet closed**. This
  check replaces 12f.
* **To run when the VM is back.**
  1. `git pull && pytest -m "not clip" -q && pytest -m clip -q`
  2. Dry run of the "+ Cascade" row: `python train.py --dataset s3dis --data_path datasets/S3DIS/blocks_bs1_s1 --cvfold 0 --n_way 2 --k_shot 1 --use_lma true --num_stages 4 --use_adrm false --dry_run true`
  3. Dry run of the full model (defaults): same command without `--use_lma/--num_stages/--use_adrm`.
  4. Optional: 2 epochs of the full model (`--epochs 2 --valid_every 2`) and `eval.py` on `last.pt`.

---

## Phase 14 — full training runs against the paper

Maintainer decisions: S3DIS first (ScanNet later, it needs access to the dataset); one seed per
configuration plus two extra seeds for the full model on S0 2-way 1-shot; `fixed100` (D-08) as the
reported protocol, `random600` also for the P1 runs; group A (infrastructure) runs locally now, group
B (training) after the pending VM check 13e.

### 14a — seeded training episodes and exact resumption

* **What.** `pipeline/episodes.py::SeededEpisodes` draws training episode i with the seed
  `[seed, 2, i]`. `train.py` is restructured around `train_loop` (one DataLoader per epoch, atomic
  `resume.pt` after every epoch with weights, optimiser, scheduler, counters and all random states,
  `best.pt`/`last.pt` as before) and gains `--resume`. `seed_worker` removed (no longer needed).
  `tests/test_resume.py` (RES-1…5).
* **Why.** A 50-epoch run of the full model takes hours on the VM; before this step an interruption
  lost the whole run (only `last.pt`, written at the end). With a global loader stream a resumed run
  would also see different episodes than an uninterrupted one.
* **Change of behaviour.** The training episode sequence differs from phases 9–13 (per-episode seeds
  instead of the worker streams); results of those short runs are not comparable bit for bit with new
  runs. Validation and test episodes are unchanged.
* **Verification.** CPU gate 247 passed. RES-2 shows bit-identical weights and AdamW moments after
  stop/resume. Mutation check: 13/14 killed; the survivor is equivalent (05 §3.8b). The first run had
  3 more survivors, which led to the absolute checks now in RES-1/2/4.

### 14b — run queue for the phase-14 experiments

* **What.** New `experiments/phase14.py`: the 26 S3DIS runs of the plan (P1 full + baseline S0;
  P2 the rest of Table 4; P2s two extra seeds of the full model; P3 the rest of Table 2; P4 Table 5
  depths 2, 3, 5, 6), each with the cells of Tables 2/4/5 it fills. It trains with `--resume true`,
  evaluates `best.pt` and `last.pt` (D-15) with `fixed100` (plus `random600` for P1, D-08) and skips
  every step whose output exists, so the same command can be rerun after an interruption.
  `eval.py --result_json` writes the result as JSON; `train.run_dir` appends `_seed<n>` for seeds
  other than 0. `tests/test_phase14.py` (Q14-1…5).
* **Note.** Table 4's "+ Entropy Gate" row and Table 5's T = 1 are the same configuration under
  D-17, and the full model is Table 5's T = 4, so one run fills two cells. The paper reports
  different numbers for them (85.34/82.48 vs 85.21/81.34 for the gate row and T = 1), i.e. they were
  separate runs there; phase 14 compares our single run with both.
* **Verification.** 6 CPU tests; mutation check 11/11. `eval.py --result_json` needs CUDA and is
  checked by the first P1 evaluation on the VM.

### 14c — summary tables against the paper

* **What.** New `experiments/summarize.py`: reads the evaluation JSON files and prints Tables 4, 5
  and 2 (Text) with S0/S1 for `best` and `last`, the average of the best checkpoints, the paper's
  numbers and the difference, plus the seed spread of the full model; ASCII output (the Windows
  console cannot print Δ). `tests/test_summarize.py` (SUM-1…3).
* **Sources.** Paper numbers transcribed from [PAPER Tab.2] (CascadeProto (Text) row), [PAPER Tab.4]
  and [PAPER Tab.5]; SUM-1 checks each printed Avg against the mean of its S0 and S1.
* **Verification.** 3 CPU tests; mutation check 8/8.

### 14d — complexity report (Table 6, D-09)

* **What.** New `experiments/complexity.py`: parameters per part (encoder, feature head, LMA, EPPM
  stages, ADRM), FLOPs with fvcore on one 2-way 1-shot episode, and the median time of an evaluation
  forward, for any configuration (train.py switches). `fvcore==0.1.5.post20221221` added to
  `requirements.txt` (the version already installed locally, published on PyPI).
  `tests/test_complexity.py` (CPX-1…3).
* **Caveat.** fvcore does not count custom CUDA kernels (pointnet2 sampling and grouping, the Mamba
  selective scan) or element-wise operators; they are listed under `unsupported_ops`, so the GFLOPs
  are a lower bound and not directly comparable with the paper's 8.86 G, whose tool is not named.
  D-09 keeps Table 6 out of the acceptance criteria.
* **Verification.** 5 CPU tests on the stand-in encoder. The real numbers need the CUDA encoder and
  are measured on the VM with the P1 runs.

### 14e — training runs on the VM · PENDING

* **Status.** Group A (14a–14d) done on the CPU (261 G1 tests). Group B needs the VM, which is
  unreachable (2026-09-19); it starts after the pending 13e check passes.
* **Commands** (inside `tmux`, from the repository root, after `git pull` and
  `pip install -r requirements.txt` for fvcore):
  1. 13e first (see above).
  2. `python experiments/complexity.py --dataset s3dis --data_path datasets/S3DIS/blocks_bs1_s1 --cvfold 0 --n_way 2 --k_shot 1`
  3. `python experiments/phase14.py --data_path datasets/S3DIS/blocks_bs1_s1 --priority P1` — rerun the
     same command after any interruption; then P2, P2s, P3, P4.
  4. `python experiments/summarize.py` (and `--protocol random600` for P1).

### 13e — VM check of phases 12 and 13 · PASSED (2026-09-19)

* **Result.** On the GCP L4 the maintainer ran the isolation sequence (steps 1–7): environment,
  data and metric, encoder and baseline, LMA/CLIP, EPPM cascade, ADRM and full model, phase-14
  infrastructure. Every pytest gate and every dry run (baseline, + LMA, + Entropy Gate, + Cascade,
  full model) passed. Phases 12 and 13 are closed; 12f is covered by this check.
* **Not yet shown.** Reproduction of the paper's numbers: that needs the full-schedule runs of 14e.

### 14e-P1 — first full-schedule runs (S3DIS S0 2-way 1-shot) · far below the paper

| Run | valid @10 / 20 / 30 / 40 / 50 | best valid | test best (fixed100) | paper S0 |
| :--- | :--- | ---: | ---: | ---: |
| baseline (`point_T0`) | 46.42 / 49.49 / 46.07 / 49.42 / 49.31 | 49.49 | pending | 82.72 |
| full model (`text_T4`) | 49.42 / 58.03 / 57.45 / 57.23 / 57.41 | 58.03 | 57.15 | 88.53 |

* **Reading.**
  1. The added modules help: full − baseline ≈ +8.5 valid points (paper S0: +5.81). The direction of
     Table 4 holds; the gap is common to both runs.
  2. Both runs are far below VIP-Seg's own 72.20, which uses the same encoder and which our pipeline
     scores at 71.97 with VIP-Seg's checkpoint. The problem lies in what the two runs share: training
     schedule or prototype/logit scale, not in LMA/EPPM/ADRM.
  3. Validation plateaus after epoch 10–20 while the training loss keeps falling (0.33 → 0.15 for the
     baseline): the model fits the training classes but does not transfer to the test classes.
* **Differences from VIP-Seg's training that both runs share.**
  * Schedule: VIP-Seg makes 24,000 optimiser steps with one episode each and halves the rate every
    7,000 steps (3 times); D-12 gives 6,000 steps of 4 episodes and halves every 1,200 steps (5 times).
  * Prototype scale: VIP-Seg L2-normalises the initial prototypes [VIPSEG models/vipseg.py:142]; D-10
    follows the paper (no normalisation), so the logits `F^q P_pointᵀ` are unbounded, which fits the
    very low training loss.
* **Decision.** P2–P4 are on hold. Control run in progress: VIP-Seg trained with its own code
  (`main.py`, `scripts/vipseg_s3dis.sh` settings) on our data.

### 14e-diag — short differential harness for the P1 gap (debugging aid)

* **What.** `experiments/diag_short.py` (marked DEBUG): trains a variant from scratch for 2,400
  seeded episodes (VIP-Seg's own run reached 68.9 valid after 2,000), AdamW lr 1e-3 wd 0.1, no LR
  decay, then scores 300 valid episodes spread over all class combinations. Variants: VIP-Seg's own
  model through our loop, baseline, baseline with L2-normalised point prototypes, full model, full
  model with L2 prototypes; batch 1 or 4.
* **Why.** The full-schedule loop takes 1.5 h per run; this gives a comparable signal in about 8 min,
  on the same valid cache that VIP-Seg's control run used (`vipseg_S_0_N_2_K_1_episodes_100_pts_2048`).

### 15a - cross_attn_norm probe and the D-18 evidence

* **Why.** Phase 14e left a gap that the diag runs localised (VM, 2,400 train / 300 valid episodes,
  2026-09-20): `baseline` 0.4416, `baseline_l2` 0.5218, `full` 0.5164, `full_l2` 0.5150, VIP-Seg's own
  model through our loop 0.6948. Three readings follow. The training loop, the data, the loss and the
  metric are sound, since VIP-Seg reaches its own published level through them. `l2norm_point_proto`
  is worth +8.0 points on the baseline (D-10 deviates from VIPSEG models/vipseg.py:142). And the four
  EPPM stages, ADRM and LMA together are worth -0.7 points against `baseline_l2`: the +7.5 of `full`
  over `baseline` is the LayerNorm of Eq.21 equalising the prototype norms, nothing else.
* **Cause, as far as it is established.** The rows of `A` are a softmax over the channel axis
  [DECISION D-01], so each channel of `P_cross` is a convex combination of the channels of `psi(P)`;
  saturated and uniform both collapse it to a constant along D, and Eq.14 has no learnable term that
  controls which. `P_diffuse`, the other summand of Eq.19, has no class index [DECISION D-16]. A stage
  can therefore add class-discriminative structure only through the residual of Eq.21. Measured on the
  CPU: 87.5-99.6 % of the energy of `P^1 - P^0` in one singular value across every feature
  distribution tried, and in the phase-11 fixture distribution the class rows leave a stage more
  similar than they entered (max cosine 0.9906 -> 0.9964). In the fully saturated regime phi's
  relative gradient is 2.8e-04 of psi's and reaches exactly 0 at three times that feature scale.
  **Which regime the real encoder produces is not established**; the CPU measurements span widths
  from 1.02 to 112.8 depending on the per-channel offset. That measurement belongs to 15c.
* **What.** New switch `cross_attn_norm = {none (default), layernorm}` on `CrossAttention`,
  `EPPMStage`, `CascadeProtoConfig` and `train.py`, plus a `full_norm` variant in
  `experiments/diag_short.py`. `layernorm` standardises `Q'` and `S'` along the projection axis with
  one shared LayerNorm (144 parameters per stage) before the correlation, making `A` exactly invariant
  to the feature scale. Specs 00 (D-18), 01 3, 02 5.2 updated.
* **Not a fix, and not the default.** `layernorm` removes the saturated regime but not the rank-1
  update, and on synthetic features with a per-channel offset it has *less* channel variation in
  `P_cross` than `none`. `none` stays the default, i.e. the literal Eq.14, and the parameter count of
  the default configuration is unchanged (79,395 per stage, 466,444 added modules).
* **Verification.** G1 CPU gate; the new tests are 15b, the VM measurement is 15c.
