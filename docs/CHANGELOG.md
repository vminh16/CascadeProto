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

### 15b - regression tests for the D-18 collapse

* **What.** `tests/test_eppm.py` gains XATT-7…10 and a `saturating_features` fixture: post-ReLU
  features with a positive per-channel offset, i.e. what a trained BatchNorm's beta supplies. The
  phase-11 fixture `features()` has no such offset and therefore never reached the collapsed regime,
  which is why 261 passing tests said nothing about it. `tests/test_cascadeproto.py` gains CP-19 for
  the switch reaching all four stages. Spec 05 3.4, 3.6b updated.
* **What XATT-10 pins.** At feature scale 10 the literal Eq.14 gives a softmax width of 1.00 out of
  128, a `P_cross` that is exactly constant along D, and `|grad phi|/|phi|` of 4.2e-17 - the state the
  stage cannot leave. The same model at scale 1.0 has width 1.92 and gradient 0.31, so the collapse is
  a property of the feature magnitude, not of the initialisation. With `layernorm` the three numbers
  are identical at scale 1 and 10 (width 126.93, channel variation 2.05e-01, gradient 2.66e-04);
  XATT-9 pins that invariance, which LayerNorm's eps = 1e-5 makes exact only to 4e-07.
* **Not asserted.** That `layernorm` is better. It is not, on every distribution measured: with a
  signed per-channel offset the literal form has more channel variation in `P_cross` (D-18). The tests
  pin the collapse and the invariance, nothing else.
* **Mutation check.** 5 mutants on `models/eppm.py`, all killed: the norm ignored, normalising along D
  instead of the projection axis, the norm always built, the norm never built, and the support branch left
  unnormalised while the query is normalised (the last one is what a separate norm per branch would be).
* **Verification.** G1 CPU gate: 269 passed, 33 deselected.

### 15c - measure the Eq.14 regime on real features (harness, DEBUG)

* **Why.** D-18 leaves one question open that the CPU cannot answer: where on the saturated-to-uniform
  axis the real encoder puts Eq.14. Both ends leave `P_cross` constant along D, and the CPU
  measurements span softmax widths from 1.02 to 127 depending on the per-channel offset of the
  features, so only a run with the CUDA encoder settles it.
* **What.** `experiments/diag_short.py` gains `attention_regime(model, episode, device)`: the effective
  softmax width of stage 1's `A` (`exp(H(row))` averaged over the rows, out of D = 128) and the channel
  variation of `P_cross` on the real `P^0` of one training episode, measured before the first optimiser
  step and again after the last. Reported in the `[diag]` line as `attn_width_init`, `attn_width_end`,
  `p_cross_chan_var_init`, `p_cross_chan_var_end`; `nan` for a model without EPPM stages. The probe
  switches the model to eval for its forward pass, so it does not move the BatchNorm running
  statistics, and restores the previous mode.
* **Reading the numbers.** A width near 1 means every row of `A` copies one channel; a width near 128
  means every row copies the channel mean. Both make `P_cross` constant along D, so a channel variation
  below about 1e-2 says the stage passes only its residual through, whatever the width. On the CPU
  stand-in encoder the default configuration already sits at the uniform end (width 128.00, channel
  variation 1.1e-03) while `cross_attn_norm=layernorm` gives 94.79 and 4.8e-01.
* **Verification.** Smoke-tested on CPU with the stand-in encoder for the full, full_norm and baseline
  configurations; the real measurement is the VM run.

### 15d - the saturation hypothesis is refuted; fusion weight and seeds in the harness

* **VM result (15c run, real encoder, 2,400 train / 300 valid episodes).** `baseline_l2` 0.5223,
  `full` 0.5346, `full_norm` 0.5134. Stage 1 of `full`: `attn_width` 127.9997 -> 9.59,
  `P_cross` channel variation 6e-04 -> 0.369. Stage 1 of `full_norm`: 92.14 -> 109.31 and
  0.649 -> 0.012.
* **What that overturns.** The cross-attention is **not** stuck: the real encoder collapses Eq.14 at
  the *uniform* end at initialisation (every row of A copies the channel mean, `P_cross` channel
  variation 6e-04), and training moves it out on its own. phi learns. The saturated regime measured on
  synthetic CPU features is reachable but is not the regime of the real model, and the
  `cross_attn_norm=layernorm` probe of 15a is rejected as a default: it pins the attention near the
  uniform end and costs 0.9 points. D-18 keeps it as an ablation flag with default `none`.
* **What survives.** A stage with a healthy `P_cross` is still worth about a point against
  `baseline_l2`, where VIP-Seg's own modules are worth about seventeen through the same loop. The
  collapse-at-both-ends argument and the class-blind `P_diffuse` (D-16) stand; the measured Eq.19
  fusion weight on `P_diffuse` is 0.504 at initialisation, so about half of `P_combined` is the same
  vector for every class.
* **Also measured.** The loop is not bit-reproducible on CUDA: the identical `full` command gave
  0.5164 and 0.5346 on two runs (loss_last100 0.3648 and 0.3556). Differences of about a point on one
  seed carry no information, which the earlier readings of 14e did not account for.
* **What.** `experiments/diag_short.py`: `attention_regime` now also returns the Eq.19 fusion weight on
  `P_diffuse`, the run record carries the seed, and `--seed` becomes `--seeds` so one command measures
  the noise floor. Spec 00 (D-18 outcome), 02 5.2 updated.
* **Verification.** CPU smoke test of the probe on the stand-in encoder for full, full_norm and
  baseline; G1 unchanged (no model code touched).

### 15e - three seeds: the cascade adds nothing, and the module is not broken

* **Result** (`results/phase15_diag/`, commit db72cc7, 2,400 train / 300 valid episodes, batch 4).

  | variant | seeds | mean | sd |
  | :--- | :--- | ---: | ---: |
  | baseline_l2 | 0.5316 / 0.5187 / 0.5111 | 0.5205 | 0.0104 |
  | full | 0.5029 / 0.5244 / 0.5241 | 0.5171 | 0.0123 |

  `full - baseline_l2 = -0.0033`, standard error 0.0093, t = -0.36. The point estimate is slightly
  negative and a 95 % interval is about +-2.6 points, so the +4.04 that Table 4 attributes to
  gate + cascade + ADRM lies outside it.
* **And yet every stage diagnostic is healthy**, on all three seeds: `attn_width` 128.00 -> 29-54,
  `P_cross` channel variation 0.0006-0.0019 -> 0.37-0.61, `w_diffuse` 0.47-0.51 -> 0.008-0.082. The
  module escapes the uniform collapse of Eq.14, gains channel structure, and learns to switch the
  class-blind diffusion branch off. It behaves as specified and buys nothing.
* **Two candidates closed.** `P_diffuse` having no class index (D-16) is not the cause, because the
  fusion removes it by itself. The cross-attention being stuck is not the cause, because it is not
  stuck. The `cross_attn_norm=layernorm` probe of 15a was already rejected in 15d.
* **What the gap is, then.** VIP-Seg's own model reaches 0.6948 through the same loop, the same data,
  the same encoder and the same metric, i.e. its PEM/PDM are worth about 17 points over the same
  `baseline_l2` where ours are worth zero. The one structural difference left is the
  channel-preserving `proto_self = sigma(A_s) . psi(P)` [VIPSEG models/vipseg.py:270-274], which
  Eq.19 of the paper does not have. Implementing it would mean inventing a module the paper does not
  describe, so this is recorded as a finding: **Table 4's increments are not reproducible from the
  equations as printed.** D-17 already noted that the paper's own baseline row (81.28 Avg) sits above
  VIP-Seg's published 74.15.
* **Also settled.** The loop is not bit-reproducible on CUDA: the identical command at seed 0 gave
  0.5218 / 0.5223 / 0.5316 for baseline_l2 and 0.5164 / 0.5346 / 0.5029 for full. Every earlier
  single-seed reading in phases 14e and 15a-15d that rested on a difference below about 2 points is
  void, including "the cascade is worth about a point".
* **Verification.** 6 VM runs, raw log and summary committed under `results/phase15_diag/`.

### 15f - paper audit, and gate_target=features implemented (D-02)

* **Audit.** `docs/research/2026-09-20_paper_vs_code_audit.md` checks every equation and every stated
  number of the paper on three links: paper -> spec 02, spec 02 -> code, paper -> code. **No
  implementation bug was found**; of the 12 findings, 10 are defects or ambiguities of the paper
  itself. The equations were read from the PDF's own LaTeX markup and cross-checked against the
  rendered pages.
* **The one thing that changes the model.** Eq.10 says "for a **feature vector** x in R^D", Eq.12 says
  "the **gated feature** is x_gated = x . g", and the symbol `x_gated` **never appears again in the
  paper**. Eq.14 multiplies `psi(P^(t-1))`, the ungated prototype, and Eq.21's residual is `P^(t-1)`
  as well. So under D-02's `gate_target=prototype` the gate output is consumed nowhere, and our code
  only makes Eq.10-12 matter by feeding psi the gated prototype - a silent departure from what Eq.14
  prints. Under `gate_target=features` the gate output is consumed by Eq.13 exactly where the text
  says ("After entropy gating, we apply cross-attention"), and Eq.14 stays literal.
* **What.** `gate_target=features` is implemented and no longer raises. The gate is applied per point
  to `F^s [N,K,2048,D]` and `F^q [B_q,2048,D]` before Eq.13; `psi` receives the ungated `P^(t-1)`; the
  diffusion branch of Eq.15-18 is ungated under both readings, since those equations name `F^q` and
  `F^s` with no mention of gating. `EntropyGate` is elementwise on the last axis, so the same module
  and the same single `theta` serve both readings and the parameter count is identical. New
  `full_gatefeat` variant in `experiments/diag_short.py`; `tests/test_cascadeproto.py` gains CP-20.
  Specs 00 (D-02 revised), 01 3, 02 5.1 updated.
* **Default unchanged.** `prototype` stays the default until a VM run separates the two readings,
  because changing it changes the meaning of every trained checkpoint.
* **Also recorded from the audit, not acted on.** Table 4 row 3 (83.91 Avg) and Table 5 `T=1` (83.28
  Avg) are the same configuration under D-17 yet differ by 0.63. The Table 4 baseline row (81.28 Avg /
  82.72 S0) exceeds VIP-Seg's own Table 2 row (74.15 Avg) and Table 6 S0 (72.20) with no stated
  reason. Eq.11 can only attenuate (`g < 1` for every finite theta), so the abstract's "amplifying
  low-entropy foreground information" is unachievable. Eq.23's prose calls the matching "scaled
  dot-product" while the equation prints no scale, which is the untested `logit_scale=sqrt_D` flag of
  D-10.
* **Verification.** G1 CPU gate. One file was damaged by a bad line-split during this step and
  restored from the committed blob of 5794de5; no work was lost, since the only uncommitted change to
  it was the damage.

### 15g - two more diag variants for the readings the paper leaves open

* **full_gatefeat** (`--gate_target features`, D-02): the reading in which Eq.12's `x_gated` is
  consumed, by Eq.13, and Eq.14 keeps the ungated `psi(P^(t-1))` it prints.
* **full_scaled** (`--logit_scale sqrt_D`, D-10): Eq.23's prose calls the matching "scaled
  dot-product" while the equation prints no scale. Never measured.
* Both are three-seed runs, because the loop's run-to-run spread is about a point (15e). The
  comparison set already exists: baseline_l2 0.5205 +- 0.0104 and full 0.5171 +- 0.0123.

### 15i - a channel-preserving term in Eq.19, off by default (D-19)

* **The measurement behind it.** Only the class-varying part of a prototype can change
  `argmax_c <f, p_c>`. On synthetic post-ReLU features `P^0` carries 14.2-14.9 % of its energy there,
  one EPPM stage leaves 3.8-4.2 %, VIP-Seg's PEM leaves 6.9-8.9 %. Both summands of Eq.19 are
  class-poor by construction, so the stage dilutes the only signal that matters.
* **What.** `eq19_self = {none (default), gated}` on `FusionOutput`, `EPPMStage`, `CascadeProtoConfig`
  and `train.py`; `gated` adds `psi(P^(t-1)_gated)` to `P_combined` before Eq.20, reusing the same
  `psi` as Eq.14, so the parameter budget is unchanged at 79,395 per stage. New `full_self` and
  `full_self_gatefeat` variants in `experiments/diag_short.py`. Specs 00 (D-19), 01 3 updated.
* **The analysis did not confirm it.** Adding the term moves the class-varying share only from 3.9 %
  to 4.2 %. Reaching VIP-Seg's 7-8 % additionally needs a LayerNorm on the new term and the removal of
  Eq.21's ReLU, the SE block and W_out's bias - and then a third metric (how many point predictions
  survive the stage) moves the wrong way, from 0.37 to 0.00. Three static metrics disagree, and 15e
  showed a stage whose diagnostics are all healthy after training still buys nothing. The switch is an
  experiment, not a claim; it is off by default and any run using it is outside the paper.
* **Verification.** G1 CPU gate; two new tests in `tests/test_eppm.py` for the term and for the
  switch/argument mismatch.

### 15h - both open readings of the paper measured, three seeds each

* **Result** (`results/phase15_diag/SUMMARY.md`, 2,400 train / 300 valid episodes, batch 4).

  | variant | seeds | mean | sd | t vs baseline_l2 |
  | :--- | :--- | ---: | ---: | ---: |
  | baseline_l2 | 0.5316 / 0.5187 / 0.5111 | 0.5205 | 0.0104 | - |
  | full | 0.5029 / 0.5244 / 0.5241 | 0.5171 | 0.0123 | -0.36 |
  | full_gatefeat | 0.4951 / 0.5219 / 0.5484 | 0.5218 | 0.0267 | +0.08 |
  | full_scaled | 0.4343 / 0.4961 / 0.4673 | 0.4659 | 0.0309 | -2.90 |

* **D-02 settled, negatively.** `gate_target=features`, the only reading in which Eq.12's x_gated is
  consumed, lands exactly on baseline_l2 with more than twice the seed spread. Which of the two
  readings is right changes nothing measurable, so the audit finding stands as a correctness point
  about the paper, not as an explanation of the gap. The default stays `prototype`.
* **D-10 settled, positively.** `logit_scale=sqrt_D`, the reading suggested by Eq.23's prose
  ("scaled dot-product matching"), is clearly worse: -5.5 points, t = -2.90, and its training loss
  plateaus at 0.446-0.463 against 0.33-0.39 everywhere else, because dividing the logits by sqrt(D)
  flattens the softmax. The printed Eq.23 without a scale is the right reading; the default `none`
  stays and the ambiguity is closed.
* **A pattern across variants.** The ones that leave the class-blind `P_diffuse` weighted highest
  score lowest: full drives w_diffuse to 0.008-0.082 (0.5171), full_gatefeat leaves 0.11-0.17
  (0.5218), full_scaled leaves 0.17-0.65 (0.4659). Consistent with D-16, but not evidence that
  removing the branch by hand would help, since training already removes it.
* **Nothing measured so far beats prototype matching with normalised prototypes.**

### 15k - the diag budget was below the divergence point; the short-budget conclusions are void

* **The error.** `experiments/diag_short.py` defaulted to 2,400 episodes = 600 steps at batch 4 = 5
  epochs of the real schedule. The P1 training logs, already in the repo, show the full model sitting
  at the baseline's level at epoch 10 (valid 0.4942 against 0.4642) and separating only between epoch
  10 and epoch 20, where it reaches 0.5803 - that is, between 1,200 and 2,400 steps, two to four times
  the diag budget. Every diag run therefore compared variants in a regime where none had diverged.
* **What that voids.** The headline of 15e ("the cascade adds nothing measurable", t = -0.36) and the
  D-02 result of 15h (full_gatefeat, t = +0.08) say nothing about the real schedule; they say the
  variants are indistinguishable after 5 epochs, which the P1 logs already implied. The D-10 result
  (full_scaled, -5.5 points, t = -2.90) is more likely to survive, because its mechanism does not
  depend on the budget: dividing the logits by sqrt(D) flattens the softmax, and its training loss
  plateaus at 0.446-0.463 against 0.33-0.39 for every other variant.
* **What.** The default budget is now 9,600 episodes = 2,400 steps = 20 epochs, and the module
  docstring states the divergence point and why the budget must clear it. A run cost 8 minutes at the
  old budget and about 32 at the new one.
* **How it was found.** By reading the phase-14 P1 valid curves against the full-schedule
  `baseline_l2` run of 15j, not by any new measurement. The evidence had been in the repo since 14e.

### 15j - baseline_l2 on the full schedule: the cascade is worth +4.71, and 15e is retracted

* **Result** (`results/phase15_full/`, 50 epochs, fixed100 on 1,500 episodes, same protocol as P1).

  | configuration | best | last |
  | :--- | ---: | ---: |
  | baseline | 0.4908 | 0.4907 |
  | baseline + L2 normalised prototypes | 0.5244 | 0.5019 |
  | full model | 0.5715 | 0.5670 |

* **Decomposition of P1's +8.07.** L2-normalising the point prototypes is worth **+3.36**; the added
  modules (LMA, four EPPM stages, ADRM) are worth **+4.71** on top of that.
* **Retraction.** 15e concluded "the cascade adds nothing measurable" from the diag harness. That is
  wrong, and the cause is 15k: the harness trained for 600 steps, while the full model only separates
  from the baseline between 1,200 and 2,400 steps. The added modules do work; they were measured
  before they had done anything.
* **Against the paper.** Table 4 claims +5.81 from its baseline to the full model. VIP-Seg normalises
  its prototypes (models/vipseg.py:142), so the paper's baseline row corresponds to our baseline + L2,
  and the increment to compare is +4.71 against +5.81 - approximately reproduced. What is not
  reproduced is the absolute level: 52.44 against 82.72 and 57.15 against 88.53. The audit already
  recorded that the paper's own baseline row exceeds VIP-Seg's published 72.20 with no stated reason.
* **Next.** The intermediate rows of Table 4 on the full schedule (15n), so each claimed increment can
  be checked on its own.

### 15n - every row of Table 4 on the full schedule

* **Result** (`results/phase15_full/`, 50 epochs, fixed100 on 1,500 episodes, one seed per row).

  | row | ours best | paper | increment ours | increment paper |
  | :--- | ---: | ---: | ---: | ---: |
  | Baseline | 0.4908 | 0.8272 | - | - |
  | + LMA | 0.4965 | 0.8393 | +0.57 | +1.21 |
  | + Entropy Gate (T = 1) | 0.5672 | 0.8535 | +7.07 | +1.42 |
  | + Cascade (T = 4) | 0.5655 | 0.8741 | -0.17 | +2.06 |
  | + ADRM (full) | 0.5715 | 0.8853 | +0.60 | +0.56 |

* **The total is reproduced, the attribution is not.** Baseline to full is +8.07 against the paper's
  +5.81, but almost all of ours comes from the first EPPM stage; going from one stage to four costs
  0.17 points where the paper claims +2.06, its largest single increment. ADRM's +0.56 is reproduced
  to within 0.04.
* **Of the first stage's +7.07, about 3.4 is normalisation.** The LayerNorm of Eq.21 equalises the
  prototype row norms, which `l2norm_point_proto` does on its own for +3.36 (15j). The stage's own
  refinement is therefore worth about +3.7.
* **What this says about the cascade.** T = 4 is not better than T = 1 at the real budget, which is
  the opposite of Table 5's ordering and consistent with the audit's F6: Table 4 row 3 and Table 5
  T = 1 are the same configuration and the paper reports two different numbers for it.
* **Caveat.** One seed per row. The seed spread at the short budget was 0.010-0.031, so the +0.57,
  the -0.17 and the +0.60 are each within one standard deviation of it.
* **Housekeeping.** The VM was stopped after the last run; artefacts (eval JSONs, training logs, the
  batch logs) are under `results/phase15_full/`.

### 15o - docs brought up to the phase-15 results

* **README** status block: no longer "not usable for results"; states what reproduces (pipeline,
  total gain of the modules, ADRM), what does not (absolute level, cascade depth), and links the
  report. Decision log range D-01…D-19.
* **AGENTS.md**: status of 2026-09-21, links to both research documents, decision range D-19, the gate
  invariant names the `gate_target` switch, `experiments/` and `results/` in the layout, and the
  verification section now states the two rules phase 15 paid for: train past the divergence point
  (1,200-2,400 steps at batch 4) before comparing variants, and use several seeds below two points.
* **Reproduction report**: a verdict section up front answering why the paper does not fully
  reproduce - no implementation bug; the paper's absolute numbers are inconsistent with its own base
  method (baseline 82.72 above VIP-Seg's own 72.20); the cascade depth does nothing under the printed
  Eq.19; and an unprinted L2 normalisation is worth +3.36.

### 15p - alternative mIoU definitions, to test whether the paper scored its own rows differently

* **Hypothesis.** The absolute gap (baseline 49.08 against 82.72, full 57.15 against 88.53) is about
  30 points, far more than any unmeasured switch can move, while VIP-Seg's released checkpoint scores
  71.97 here against the 72.20 the paper reports for it. One explanation that fits both: the VIP-Seg
  row was taken from the VIP-Seg paper, and the paper's own rows were scored with a different mIoU.
  If so, re-scoring our checkpoints with that definition lands near the paper's level.
* **What.** `pipeline/metrics_alt.py` (pure numpy) computes, from the same predictions: VIP-Seg's
  accumulated foreground mIoU re-implemented (must equal the primary number), the same counts with
  background included, per-episode mIoU over foreground labels, per-episode mIoU with background, and
  point accuracy. `pipeline/evaluation.py` gains `collect_predictions`, so predictions are computed
  once; `eval.py --extra_metrics true` logs the alternatives and writes them under `extra` in the
  result JSON. The primary metric (D-08) is untouched. `experiments/rescore_alt_metrics.sh` re-scores
  VIP-Seg's released checkpoint and the six full-schedule best checkpoints on the VM, then powers the
  VM off after 20 minutes.
* **Tests.** `tests/test_metrics_alt.py` (ALT-1..4): equality with a copy of VIP-Seg's loop on random
  episodes, a hand-computed episode for every metric, episode labels mapping to one global class, and
  perfect prediction.

### 15q - the metric hypothesis is rejected

* **Result** (`results/rescore/`, fixed100 on 1,500 episodes, best checkpoints).

  | run | primary | acc. with bg | episode fg | episode with bg | point acc. | paper |
  | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
  | VIP-Seg released | 0.7197 | 0.7272 | 0.7427 | 0.7535 | 0.8546 | 0.7220 |
  | baseline | 0.4908 | 0.5188 | 0.5112 | 0.5715 | 0.7281 | 0.8272 |
  | baseline + L2 | 0.5244 | 0.5509 | 0.5464 | 0.6038 | 0.7556 | - |
  | + LMA | 0.4965 | 0.5323 | 0.5074 | 0.5892 | 0.7635 | 0.8393 |
  | + gate (T=1) | 0.5672 | 0.5921 | 0.5845 | 0.6393 | 0.7832 | 0.8535 |
  | + cascade (T=4) | 0.5655 | 0.5952 | 0.5804 | 0.6463 | 0.7984 | 0.8741 |
  | full | 0.5715 | 0.5997 | 0.5855 | 0.6485 | 0.7974 | 0.8853 |

* **Sanity.** `accumulated_fg` equals the primary number on all seven rows.
* **Verdict.** No mIoU definition brings the full model past 0.65; the most generous adds 7.7 points,
  a quarter of the 31-point gap. Even point accuracy leaves baseline and full 9.9 and 8.8 points
  short. The ordering is identical under every definition, and VIP-Seg leads every CascadeProto row
  by 13-15 points, so the paper's central claim (CascadeProto above VIP-Seg) fails under all of them.
* **Operational.** The script's timed in-VM `shutdown -h +20` locked every non-root login for its
  last five minutes and kept the results out until the user pushed them; it is removed, and the VM
  is stopped from outside after the copy.

### 15r - D-20: the baseline on VIP-Seg's trained encoder (diagnostic, breaks guardrail #1)

* **Why.** With the metric hypothesis rejected (15q), the one remaining explanation large enough for
  the 31-point gap is that the paper's rows sit on VIP-Seg's trained encoder. Its baseline (82.72)
  beating VIP-Seg itself (72.20) would then be possible.
* **What.** `pipeline/vipseg_baseline.py` gains `load_vipseg_model` and `init_features_from_vipseg`;
  `train.py --init_from_vipseg <checkpoint>` initialises `model.features` from it before any resume and
  tags the run directory `_vipinit`; `experiments/vipseg_init_probe.py` scores the baseline and
  baseline + L2 on VIP-Seg's trained features with no training, the cheapest test of the hypothesis.
  New test: the `_vipinit` run directory. Spec 00 gains D-20.
* **Status.** Approved by the maintainer as a diagnostic on 2026-09-21. Never a result configuration.

### 15s - citation and metric check (no GPU)

* **Metric.** AttMPTI's `evaluate_metric` ([34], the protocol the paper says it follows) accumulates
  per-class counts over all episodes and averages IoU without background - the computation we use
  through VIP-Seg's unchanged copy.
* **Citations.** Table 2's VIP-Seg row equals VIP-Seg's released log folders to two decimals
  (S0 2w1s 0.722026, S1 0.760875, and so on for all eight cells), i.e. it was produced by that metric
  on the loader we use, which scores the S0 checkpoint at 71.97 here.
* **Consequence.** The cited rows are on our scale; only the paper's own rows are not. Its S0 2w1s
  claim is +16.3 over VIP-Seg where every earlier step in the table is 0.2-5.6 points.
* **Untested.** The "Area 5 for testing" sentence of §4.1 (D-07's unimplemented `area5` flag); low
  prior, since an unseen test area would make the task harder.

### 15t - D-20 rejected at zero training

* **Result** (`results/vipinit/`, fixed100 on 1,500 episodes, VIP-Seg's released S0 encoder and
  feature head, no training). Baseline 0.4737 (per episode with bg 0.5696, point accuracy 0.7419);
  baseline + L2 0.5497 (0.6308, 0.7794). The `[probe]` lines were pasted from the VM; the JSON files
  carry them to four decimals.
* **Reading.** Prototype matching on VIP-Seg's own trained features scores the same as on ours
  (0.4908 and 0.5244 from scratch), so VIP-Seg's 0.7197 comes from its PEM/PDM head, not its encoder,
  and a "plain VIP-Seg backbone with masked average pooling" cannot reach the paper's 0.8272 with
  either encoder. Rejected by the rule fixed before the run (below 0.60), so the fine-tuning step was
  skipped. It also validates our from-scratch encoder training independently.
* **What remains.** Every explanation large enough for the ~30-point absolute gap that the paper's
  text supports has now been tested and rejected: pipeline, implementation, protocol, mIoU
  definition, citations, and a VIP-Seg-trained encoder. Only the low-prior `area5` split is untested.

### 15u - D-21: training on the test classes, to test the protocol hypothesis

* **Why.** The absolute gap is split in two. About 15-22 points are reproduced and explained: removing
  VIP-Seg's PEM/PDM costs about 22 points even on VIP-Seg's own trained encoder (15t), and the printed
  CascadeProto head is about 15 points weaker than PEM/PDM. The other 10-16 points - the paper's rows
  lying *above* VIP-Seg - are explained by nothing in the paper. The gap is nearly uniform across
  Table 4, which points at data or protocol, and §4.1's "Areas 1-4, 6 for training and Area 5 for
  testing" admits a reading under which test classes were seen in training.
* **What.** `pipeline/episodes.build_train_dataset(train_classes="all")` and `train.py --train_classes
  all` widen the training sampler to the test classes; run directories get `_leak`. Two tests.
  Spec 00 gains D-21.

### 15v - trace back through the method; the one untested decision on the baseline's path

* **Argument.** The absolute gap is 33.6 points on the baseline, which contains no added module, so
  only a decision on the baseline's path can explain it. Of the 21 decisions, those are D-07, D-08,
  D-10, D-12, D-15 and D-17; all but D-12 are verified or measured (report 3.3). Code on the path is
  checked against VIP-Seg's own (encoder, head, pooling, optimiser).
* **D-12.** VIP-Seg trains 24,000 steps at batch 1 and halves the LR every 7,000 steps; our D-12 keeps
  its 24,000 episodes at the paper's batch 4, i.e. 6,000 steps halving every 1,200. New
  `train.py --batch_size` (default 4, the paper's) makes the VIP-Seg schedule runnable; run
  directories get `_b<n>`. One test.

### 15w - resume accepts flags added after the checkpoint

* **Symptom.** Resuming `full_leak` (D-21), whose `resume.pt` was written before `--batch_size`
  existed, failed with `cannot resume ...: arguments differ in []`. The check compared the two argument
  dicts but listed only the checkpoint's keys, so the one extra key was reported as nothing.
* **Fix.** `train.resume_mismatch` compares the union of keys and treats a key missing from the
  checkpoint as the parser default; every flag added so far defaults to the behaviour that existed
  before it. `build_parser` split out of `parse_args` to read the defaults. One test; G1 281 passed.

### 15x - D-12 and D-21 measured: the schedule is not it, leakage explains part

* **D-12, VIP-Seg's schedule** (`results/b1/`). The baseline at batch 1, 24,000 steps, learning rate
  halved every 7,200 steps: fixed100 **0.4901** (best, epoch 40) and 0.4788 (last), against 0.4908
  and 0.4907 on D-12's schedule. Validation tracked the old run within noise at every checkpoint
  (0.469 / 0.469 / 0.422 / 0.498 / 0.485 against 0.464 / 0.495 / 0.461 / 0.494 / 0.493). Four times
  the updates and a slower decay change nothing, as predicted. D-12 is cleared, which closes the trace
  back: every decision on the baseline's path is now verified or measured (report 3.3).
* **D-21, test classes seen in training** (`results/leak/`, 20 epochs each, fixed100):

  | row | split (50 epochs) | all classes (20 epochs) | change | paper |
  | :-- | --: | --: | --: | --: |
  | baseline | 0.4908 | **0.6354** | +14.46 | 82.72 |
  | full | 0.5715 | **0.6924** | +12.09 | 88.53 |
  | full - baseline | +8.07 | +5.70 | | +5.81 |

  Seeing the test classes lifts both rows by 12-15 points and leaves the module gain about where the
  paper puts it, which fits a protocol difference better than a model difference. It still leaves the
  leaked full model 13.5 points **below the paper's baseline**, and this diagnostic is an upper bound
  of the area-split reading (it leaks classes and possibly blocks). Leakage therefore explains part of
  the level, not the level. Twenty epochs is less than the full schedule; the leaked runs had not
  plateaued (validation 0.586 to 0.637 and 0.638 to 0.704 between epochs 10 and 20), so the part it
  explains may be somewhat larger. One seed each.
* **Incident.** The first queue for these runs waited on `pgrep -f run_b1.sh`, which matched its own
  command line and never returned; about 90 minutes of VM time were idle before it was replaced.


### 15y - scored on classes seen in training: the paper's level within 5-8 points; project closed

* **Why.** A second review of the report (`docs/research/2026-09-21_gap_diagnosis.md`) found that the
  D-21 run of 15x is a partial leak, not the upper bound 15x called it: each test class fills
  9,600 x 2 / 12 = 1,600 way slots against 24,000 x 2 / 6 = 8,000 for a class trained on normally, and
  the runs had not converged. The full form needs no new model: S0's training classes are fold 1's test
  classes, so an S0 checkpoint scored with `--cvfold 1` is scored on classes it was trained on, and
  symmetrically for S1. Each reading rule below was written down before its run.
* **Result** (`results/seen/`, `log_s1/`, fixed100, 1,500 episodes, one seed; VM, 2026-09-22).

  | checkpoint | scored on | seen in training? | mIoU |
  | :-- | :-- | :-- | --: |
  | S0 baseline best (epoch 20) | fold 1 | yes | 0.7140 |
  | S0 baseline last (epoch 50) | fold 1 | yes | **0.7732** |
  | S0 full best / last | fold 1 | yes | 0.7599 / **0.8128** |
  | VIP-Seg released S0 | fold 1 | yes | 0.7913 |
  | S1 baseline last (epoch 50, new run) | fold 0 | yes | **0.7158** |
  | S1 baseline best = last | fold 1 | no | **0.5191** |

  Baseline against the paper (S0 / S1 / Avg): paper 82.72 / 79.83 / 81.28; standard protocol
  49.08 / 51.91 / 50.50 (gap 30.8); seen classes 77.32 / 71.58 / 74.45 (gap 6.8).
* **Reading.** Having seen the classes is worth 22-25 points on either fold; the easier fold-1
  classes are worth 2.8 unseen. Seen-class scoring reproduces the paper's S0 > S1 order, which the
  standard protocol reverses for us (51.91 > 49.08) and for every method in Table 2 with both folds. It
  also explains a baseline above VIP-Seg: a seen-class 77.32 next to VIP-Seg's cited unseen 72.20. The
  remaining 5-8 points are the size of one seed's spread, random600 against fixed100 and checkpoint
  choice combined; not decomposed. Consistent with the paper's numbers, not proof of how they were
  made. Predictions: `best.pt` S0 baseline >= 0.75 on fold 1 (0.714, inconclusive), `last.pt` >= 0.75
  (0.773, met), S1 on fold 0 72-77 and below 0.773 (0.716, below by 0.4, order met), S1 unseen 52-55
  (0.519, met within 0.1).
* **Report corrections.** (1) The paper column of Table 4 rows 2-4 had been reconstructed as 82.72 plus
  the cumulative Avg increments (83.93 / 85.35 / 87.41); the printed S0 values are 83.98 / 85.34 /
  87.89, and the increments to compare with ours are the S0 ones (+1.26, +1.36, +2.55, +0.64). No
  verdict changes; the unreproduced cascade claim grows from +2.06 to +2.55. Stale copies of the old
  values remain in 15n and 15q above. (2) 15x's "upper bound" is withdrawn. (3) The "3.4 of +7.07 is
  normalisation" split is marked as an analogy, not a measurement. (4) Table 6 (2.88M vs 2.76M params,
  8.86G vs 8.48G FLOPs) is cited against an unstated stronger backbone.
* **What.** `experiments/run_seen.sh` (seen-class probe); report sections Verdict, 2, 3, 3.5, new 3.6, 6;
  spec 00 D-21; README and AGENTS status. New research notes: `2026-09-21_external_sources_on_gap.md`
  (primary sources: the first author's own ablations put this baseline at 49-52; no method > 80 in this
  protocol) and `2026-09-21_gap_diagnosis.md` (the review and the prediction log).
* **Status.** The project is closed at this point by the maintainer. Not done: extra seeds, random600
  rescoring, the full model on S1, other Table 2 settings, and contacting the authors.


## Phase 16 — improvement research, beyond the paper

Opened by the maintainer on 2026-09-22 after the reproduction was closed (15y). Goal: test, without
leakage, whether CascadeProto's ideas can beat VIP-Seg (72.20 / 76.09) and EDS-Net (73.32 / 74.67)
under the standard protocol. Every change is behind a flag whose default keeps the reproduction.

### 16.0 - research note: diagnosis and ranked directions

* **What.** `docs/research/2026-09-22_improvement_directions.md`: why the printed EPPM cannot close the
  gap to VIP-Seg (class-common summands cannot change a prediction, the entropy gate is a fixed even
  pointwise function, no stage reads the previous prediction, the text prior is fused 1:1 after being
  aligned to the point prototype), what the modules this paper descends from have in common (Seg-PN's
  QUEST = VIP-Seg's PDM, TaylorSeg's APP = VIP-Seg's PEM; their ablations credit the channel
  cross-correlation with +15 points, computed after a reshape that shares it across the episode's
  classes), EDS-Net as an independent calibration by the same first author (+1.30 S0 for the same
  ideas under the standard protocol), and a ranked plan with zero-training probes first.
* **Leading hypothesis for the 15-point gap to VIP-Seg's head:** D-01 computes one correlation per
  class slot from whole support blocks, where every published member of the family effectively uses one
  correlation for the whole episode. Phase 16a-16e implement the test.

### 16a - protocol guard: eval.py refuses to score seen classes (D-22)

* **Why.** 15y showed that scoring a checkpoint on its own training classes adds 22-25 points, and
  nothing stopped it: `eval.py` took `--cvfold` from the command line, independently of the fold the
  checkpoint was trained on. Phase 16 looks for gains of a few points, so the guard comes first.
* **What.** `eval.protocol_check` / `training_fold`: the training fold is read from the checkpoint
  (`args.cvfold`) or stated with `--checkpoint_cvfold` (VIP-Seg's release records none); a different
  scoring fold or `--train_classes all` raises before any episode is built, unless
  `--allow_seen_classes true`, which labels the log and result JSON as a seen-class diagnostic. The
  JSON now carries `protocol_check`, `cvfold` and `checkpoint_cvfold`. `run_seen.sh` (the one
  diagnostic) passes the flag; `rescore_alt_metrics.sh` and README state the VIP-Seg checkpoint's fold.
  D-22 also fixes the phase-16 reporting rules: `last.pt` headline, design screening on S1, S0 held out
  until the design is frozen, no tuning on scored classes. Specs 00 (D-22), 04 6.1, 05 3.12.
* **Verification.** `tests/test_protocol_guard.py` (PROT-1...6). Mutation check, 7 mutants, all killed:
  fold comparison inverted, `train_classes=all` unchecked, the flag inverted, `main` skipping the check,
  a contradicting `--checkpoint_cvfold` accepted, an unknown fold assumed equal to the scoring fold, a
  diagnostic labelled clean.

### 16b - Eq.13's single S' as a switch: `cross_attn_support = pooled` (D-23)

* **Why.** D-01 item 5 recorded that Eq.13 writes one `S′ = φ(F^s)` without class slots and that our
  per-slot form follows VIP-Seg. Phase 16 found that this is the one place where we differ from every
  published member of the family: in QUEST (Seg-PN), APP (TaylorSeg), PEM and PDM (VIP-Seg) the
  `reshape(72, -1)` makes each correlation block depend on **both** queries and **all** class slots,
  i.e. an episode-level statistic, and DPA writes that shared form explicitly and measures +13.7 for
  it alone. The ablations that isolate this branch credit it with +15.4 (Seg-PN Table 6) and +15.2
  (TaylorSeg Table 4) on S0 - the size of our gap to VIP-Seg's head.
* **What.** `CrossAttention(support=...)`: `pooled` averages the pooled tokens of all N·K support
  blocks into one `F̄^s`, projects it once and applies `A_b = softmax_row(Q′_bᵀ S̄′/√d) ∈ R^{D×D}` to
  every class row. Unlike the published reshape it keeps the queries separate. No new parameters.
  Wired through `EPPMStage`, `CascadeProtoConfig` and `train.py`; run directories get `_pooled`.
  Specs 00 (D-23), 01 §3, 02 §5.2, 05 §3.4.
* **Default unchanged** (`class_slots`) until the R1 comparison of 16e decides it.
* **Verification.** XATT-11...15 and CP-21 (explicit-loop reference for the pooled attention, the same
  `A` on every class row, order invariance over ways and shots, every block read, no query mixing,
  identical parameters, exact coincidence of the two readings at N = K = 1, and a difference at N = 2).
  On the stand-in encoder the two readings differ by 1.3e-4 of `P_cross`'s scale and 1.8e-8 in the
  logits, because Eq.14 sits at the uniform end at initialisation (attention width 127.9998 of 128,
  D-18); the readings can only separate once φ has trained, which is what R1 measures.
  Mutation check, 8 mutants on `models/eppm.py`, all killed: `S′` from the first block only, `S′`
  averaged over the shots of way 0, the softmax over the wrong axis, the product transposed, the switch
  inverted in `attention` and in `forward`, the switch ignored, and the queries pooled together the way
  VIP-Seg's reshape does.

### 16c - EPPM-S: the stripped stage, `stage_type=eppm_s` (D-24)

* **Why.** To locate the 15 points between the printed stage and VIP-Seg's, the parts the research
  note proved inert have to come off: `P_diffuse` (common to every class, so it cannot change a
  prediction, and training drives its weight to 0.008-0.082 anyway), the entropy gate (an even
  pointwise function of the prototype value with one scalar), Eq.19's fusion MLP, Eq.20's class-pooled
  SE, `w_cls` and Eq.21's ReLU. What goes in instead is the channel-preserving self term that every
  published member of this family carries [VIPSEG models/vipseg.py:262-277], which D-19 had measured
  as the missing piece.
* **What.** `models/eppm_s.py::EPPMSharedStage`:
  `P^t = LN(W(P_cross + σ(W_3(Q'ᵀQ' − S'ᵀS')/√D) ⊙ ψ(P^{t-1})) + P^{t-1})`, either support reading
  (D-23), 37,888 parameters against 79,395. `stage_type = {eppm (default), eppm_s, vip}` on
  `CascadeProtoConfig`, `build_stage` and `train.py`; run directories get `_eppm_s`. Non-default
  EPPM-only switches now raise with another stage type instead of being ignored. Specs 00 (D-24),
  01 §3-§4, 05 §3.4b.
* **Verification.** EPS-1...9 and CP-22 (explicit-loop reference for both support readings at N, K,
  B_q in {1, 2, 3}; the residual; the self gate; no query mixing; class-row equivariance; the two
  readings coinciding at N = K = 1; gradients and gradcheck; the model-level cascade written out).
  Mutation check, 10 mutants on `models/eppm_s.py`, all killed: residual dropped, self term ungated,
  the gate applied to `P` instead of `ψ(P)`, the softmax axis, the Gram scale inverted, the pooled
  support taken from way 0, `ψ` skipped, the shot mean dropped, the self gate from shot 0 only, and
  the support Gram ignored.

### 16d - VIP-Seg's PEM/PDM as a reference stage, `stage_type=vip` (D-25)

* **Why.** "VIP-Seg's head is worth 17 points more" was measured on its whole model, which also
  differs in prototype normalisation, logits and gating. `stage_type=vip` puts its module in our
  cascade with everything else held fixed, so R1 can compare stages rather than pipelines.
* **What.** `models/vip_stage.py::VIPStage` wraps the inherited modules (imported, never edited,
  and lazily, since `models.vipseg` needs `pointnet2_ops`), reproducing VIP-Seg's alternation and the
  outer residual on odd steps [VIPSEG models/vipseg.py:154-160]. `cross_attn_scale` and
  `cross_attn_support` raise with this stage type, because VIP-Seg's code fixes both. Run directories
  get `_vip`. Specs 00 (D-25), 01 §3, 05 §3.4c.
* **Verification.** VIPS-1...4 on the CPU with a recording stand-in module (call order, outer residual
  per step, shape guard before the call, `build_stage` alternation and the switch guards); VIPS-5 on
  the GPU checks four wrapped stages against VIP-Seg's own loop. Mutation check, 5 mutants: 4 killed
  on the CPU (residual on the wrong steps, query and support swapped, residual never applied, shape
  guard removed); the fifth - PEM and PDM exchanged in `build_vip_module` - **survives the CPU gate by
  construction** and is killed only by VIPS-5, which needs the GPU. Recorded here so the VM run is not
  skipped.
* **Incident.** The first run of the mutation check included VIPS-5, which always fails locally
  (`pointnet2_ops` missing), so every mutant was reported killed. The runner now requires the tests to
  pass on the unmutated file first.

### 16e - the R1 queue: which head difference costs the 15 points

* **What.** `experiments/diag_short.py` gains `--cvfold` (default 0, phase 16 runs S1) and the seven
  `r1_*` variants, all at T = 1 without LMA so that only the head differs: `r1_baseline_l2` (no stage),
  `r1_eppm` (a), `r1_pooled` (b, D-23 only), `r1_eppms` (c, D-24 with the shared `S′`),
  `r1_eppms_slots` (c', D-24 with D-01's class slots), `r1_vippem` (d, one VIP-Seg PEM) and `r1_vip4`
  (VIP-Seg's four modules, the 72-level reference). `attention_regime` now handles a stage without a
  gate or a diffusion branch, and returns `nan` for `stage_type=vip`. `experiments/run_r1.sh` runs the
  queue on the VM; `experiments/summarize_r1.py` prints seeds, mean, sd and Welch's t per variant and
  fills the decision rules in.
* **Protocol.** Screening trains and validates on **S1**, so the S0 test classes stay untouched until
  the design is frozen [DECISION D-22]; validation uses the fold's `valid` draw, as every run here does
  [DECISION D-15]. Three seeds per variant, 9,600 episodes = 2,400 steps (past the divergence point of
  15k); differences below 2 points are noise.
* **Decision rules, fixed before the run.** R1.1 `pooled − eppm ≥ +3` with `t > 3` makes the per-slot
  correlation of D-01 a main cause and revises D-01; R1.2 `|eppms − vippem| ≤ 2` keeps route A;
  R1.3 `eppms − eppm ≥ +3` while `pooled ≈ eppm` puts the cause in Eq.19-21's additions; R1.4
  `vippem − eppm ≥ +3` with EPPM-S behind sends the work to route B; R1.5 everything within 2 points of
  `baseline_l2` closes R1 and re-opens the oracle probe.
* **Verification.** R1-1...5 (`tests/test_phase16.py`): every variant is the configuration it claims and
  passes the switch guards, consecutive variants differ in exactly one switch, the summary reproduces
  hand-computed statistics, a missing variant is reported rather than guessed, and the queue script
  states its fold, seeds, budget and rules. Spec 05 §3.8f.
* **Not verified locally.** The runs themselves: `diag_short.py` needs the CUDA encoder (gate G2).

### 16f - R1 measured: the support reading is not the cause, VIP-Seg's stage is +15.5, route B

* **Result** (`results/phase16_r1/SUMMARY.md`, VM 2026-09-23; S1, T = 1, no LMA, 2,400 steps, two
  seeds, 300 valid episodes; **not** comparable with the S0 fixed100 numbers of the report).

  | variant | mean | sd | vs `r1_baseline_l2` |
  | :--- | ---: | ---: | ---: |
  | `r1_baseline_l2` | 0.5602 | 0.0222 | - |
  | `r1_eppm` (the printed stage) | 0.5305 | 0.0070 | -2.97 |
  | `r1_pooled` (D-23) | 0.5339 | 0.0015 | -2.63 |
  | `r1_eppms` (D-24) | 0.4927 | 0.0115 | -6.75 |
  | `r1_vippem` (D-25) | **0.6852** | 0.0036 | **+12.50** (t = +7.86) |

* **The rules, as fixed before the run.** R1.1 `pooled - eppm` = **+0.33**, t = +0.65: not met, the
  leading hypothesis is refuted. R1.2 `eppms - vippem` = -19.26: not met. R1.3 `eppms - eppm` =
  **-3.79**, t = -3.97: not met and negative, stripping the inert parts made the stage worse. R1.4
  `vippem - eppm` = **+15.47**, t = +27.78: **met**, so phase 16 continues on route B.
* **What this settles.** (i) D-23 is measured and negative: reading Eq.13 with one `S′` for the episode,
  the form every published member of this family effectively computes, is worth a third of a point
  here. (ii) D-24 is negative: removing `P_diffuse`, the entropy gate, the SE block, `w_cls` and
  Eq.21's ReLU does not produce a stage that helps; the static argument that those parts cannot change
  a prediction does not imply that a stage without them can. (iii) D-25 reproduces the 15-point gap
  **inside one pipeline with only the stage changed**, where 15e had compared two training scripts.
* **The budget is not the explanation.** `r1_vippem` resolves +12.50 with t = +7.86 at this very
  budget, and the full-schedule validation curves already in the repository move by at most 1.1 points
  between epoch 20 and epoch 50 on six configurations (baseline .4949 -> .4931, baseline_l2 .5057 ->
  .5048, +LMA .4809 -> .4807, T = 1 .5670 -> .5767, T = 4 .5710 -> .5599, full .5803 -> .5741). A
  15-point verdict cannot be a budget artefact, and a 0.33-point one cannot become a 3-point one.
* **The L2 confound, raised by `docs/research/2026-09-23_gap_and_upgrade_research.md` §4.5 and verified
  here.** (a) and (b) run without `l2norm_point_proto`, (c), (d) and the baseline with it, and L2 alone
  measured +3.36 on the S0 baseline. R1.1 is unaffected (both arms without L2). R1.3 is confounded in
  EPPM-S's favour, so like for like it is worse by up to about 7 points. R1.4 is confounded in PEM's
  favour by up to 3.4, so the stage-only gap is about 12-15 and route B still fires. `eppms` against
  `baseline_l2` is like for like; `eppm` and `pooled` against `baseline_l2` are not, and need the
  missing `r1_eppm_l2` run (1.1-1.6 GPU-hours) before anything is concluded from them.
* **Not settled.** Two seeds, not three (at this seed spread three seeds resolve about 5 points, not
  the 3 the rules assume, and R1.2 cannot establish equivalence); no curve for EPPM-S or PEM; `r1_eppms_slots` and `r1_vip4`
  not run; S1 valid draw only, no fixed100 and no S0. Why PEM wins is untested: the LayerNorm on the
  self term that 15i named and 16c omitted, PEM's two separate gates against the difference gate, and
  the query mixing that VIP-Seg's reshape performs and our stages avoid - the last one would be
  evidence for the transductive direction of the research note.
* **Prediction log.** The research note predicted route A's stage 0 at 64-71 and D-23 as the leading
  cause; both are wrong. It predicted route B's stage 0 at 71-72; one PEM reaches 68.5 at 40 % of the
  schedule, on track.
