# 04_DATA_AND_EPISODES: Datasets, Preprocessing, Episodes, Schedule & Evaluation

* **Ground truth:** [00_SOURCES_AND_DECISIONS.md](00_SOURCES_AND_DECISIONS.md). Every normative line carries a source tag.
* **Scope:** everything between raw datasets and the numbers reported in Tables 2–3: preprocessing, directory layout, class splits, episode sampling, training schedule, model selection and the evaluation metric.
* **Inherited code:** `dataloaders/*.py` and `preprocess/{collect_s3dis_data,collect_scannet_data,room2blocks}.py` are byte-identical to the pinned VIP-Seg commit (checked 2026-09-17) and must stay that way. Change behaviour through arguments, never by editing these files [VIPSEG dataloaders/loader.py] [VIPSEG README.md].
* **Rewritten:** 2026-09-17 (Phase 3).

---

## 1. Datasets

| Item | S3DIS | ScanNet | Source |
| :--- | :--- | :--- | :--- |
| Content | RGB point clouds, 272 rooms, 6 areas | 1513 RGB-D scans | [PAPER §4.1] |
| Annotated classes | 13 | 20 | [PAPER §4.1] |
| Class ids in files | 0–12; `clutter` = 12 | 0–20; `unannotated` = 0 | [VIPSEG dataloaders/s3dis.py:12-13] [VIPSEG dataloaders/scannet.py:12-13] |
| Classes eligible as episode targets | 0–11 (`clutter` never sampled) | 1–20 (`unannotated` never sampled) | [VIPSEG dataloaders/s3dis.py:30] [VIPSEG dataloaders/scannet.py:31] |
| Scene split stated in the paper | Areas 1, 2, 3, 4, 6 train; Area 5 test | 1201 train / 312 validation scenes | [PAPER §4.1] |
| Blocks | 2048 points sampled per block | 36,350 blocks of 2048 points in total | [PAPER §4.1 "split into 1201 training and 312 validation scenes partitioned into 36350 blocks of 2048 points each"] |

How the paper's scene split is used is decided in §3 [DECISION D-07].

---

## 2. Preprocessing and directory layout

### 2.1 Pipeline

Data preparation follows AttMPTI, as VIP-Seg does [VIPSEG README.md]:

1. **Rooms → `.npy`.** `preprocess/collect_s3dis_data.py` writes one `N×7` array (`X Y Z R G B L`) per room to `<root>/scenes/data/Area_<a>_<room>.npy` [VIPSEG preprocess/collect_s3dis_data.py:16-20]. ScanNet uses `preprocess/collect_scannet_data.py`, which additionally needs `<root>/meta/scannet_classnames.txt` and `<root>/meta/scannetv2-labels.combined.tsv` [VIPSEG preprocess/collect_scannet_data.py:123-125].
2. **Rooms → blocks.** `preprocess/room2blocks.py --data_path <root>/scenes --dataset {s3dis|scannet}` cuts 1 m × 1 m columns with stride 1 m and discards blocks with fewer than 1000 points (defaults) [VIPSEG preprocess/room2blocks.py:73-78,49]. Each block is saved as `<room>_block_<i>.npy` [VIPSEG preprocess/room2blocks.py:102].
3. **Directory name.** `room2blocks.py` names the output `blocks_bs{block_size}_s{stride}` [VIPSEG preprocess/room2blocks.py:87]. With the default arguments (argparse does not apply `type=float` to non-string defaults) this is `blocks_bs1_s1`, which the VIP-Seg run scripts expect [VIPSEG preprocess/room2blocks.py:75-76] [VIPSEG scripts/vipseg_s3dis.sh]. Passing `--block_size 1 --stride 1` explicitly produces `blocks_bs1.0_s1.0` instead; then rename the directory or pass that name as `--data_path`.

### 2.2 Required layout

```text
<root>/                          e.g. datasets/S3DIS
├── meta/
│   └── s3dis_classnames.txt     13 lines, id order: ceiling floor wall beam column window door table chair sofa bookcase board clutter
├── scenes/data/*.npy            step 1 output
└── blocks_bs1_s1/               --data_path points here
    ├── data/*.npy               step 2 output (Area_<a>_<room>_block_<i>.npy)
    ├── class2scans_100.pkl      created by the loader on first use
    └── <model>_S_<fold>_N_<N>_K_<K>_[test_]episodes_100_pts_2048/*.h5   fixed evaluation episodes (§6)
```

* The loader reads class names from `dirname(data_path)/meta/` [VIPSEG dataloaders/s3dis.py:15] [VIPSEG dataloaders/scannet.py:16].
* Class-name order follows the commented id map in the loader [VIPSEG dataloaders/s3dis.py:13-14] [VIPSEG dataloaders/scannet.py:13-15].
* `class2scans_<n>.pkl` and the `.h5` episode folders are caches; delete them after changing the data or the split [VIPSEG dataloaders/s3dis.py:36-62] [VIPSEG dataloaders/loader.py:239-253].

---

## 3. Class splits

Primary protocol: split by **class**, over all scenes [DECISION D-07].

| Dataset | Fold | Test (unseen) classes | Source |
| :--- | :--- | :--- | :--- |
| S3DIS | S0 | beam, board, bookcase, ceiling, chair, column | [VIPSEG dataloaders/s3dis.py:20] |
| S3DIS | S1 | door, floor, sofa, table, wall, window | [VIPSEG dataloaders/s3dis.py:21] |
| ScanNet | S0 | bathtub, bed, bookshelf, cabinet, chair, counter, curtain, desk, door, floor | [VIPSEG dataloaders/scannet.py:21] |
| ScanNet | S1 | otherfurniture, picture, refridgerator, shower curtain, sink, sofa, table, toilet, wall, window | [VIPSEG dataloaders/scannet.py:22] |

* Training classes = eligible classes (§1) minus the fold's test classes [VIPSEG dataloaders/s3dis.py:30-31] [VIPSEG dataloaders/scannet.py:31-32].
* The paper does not list the classes of S0/S1; it only names the splits [PAPER §4.1].
* Ablation `split_protocol=area5` (S3DIS only): additionally restrict training blocks to names starting with `Area_1`–`Area_4`, `Area_6` and test/valid blocks to `Area_5` [PAPER §4.1] [DECISION D-07]. Implement it as a filter on scan names passed around the inherited loader, not as an edit to it.

---

## 4. Episode sampling

All sampling is done by the inherited `MyDataset` [VIPSEG dataloaders/loader.py:116-225].

### 4.1 Required loader arguments

| Argument | Value | Source |
| :--- | :--- | :--- |
| `num_point` | 2048 | [PAPER §4.1] [VIPSEG scripts/vipseg_s3dis.sh] |
| `pc_attribs` | `'xyzrgbXYZ'` | [VIPSEG scripts/vipseg_s3dis.sh] [VIPSEG main.py:48] |
| `way_ratio`, `way_num` | `[0.05, 0.05]`, `[100, 100]` | [VIPSEG scripts/vipseg_s3dis.sh] |
| `n_queries` | 1 | [VIPSEG scripts/vipseg_s3dis.sh] |
| `pc_augm` | true in training, false in evaluation | [DECISION D-12] [VIPSEG dataloaders/loader.py:236] |
| `pc_augm_config` | `scale 0, rot 1, mirror_prob 0, jitter 1, shift 0.1, random_color 0` | [VIPSEG main.py:56-66] [VIPSEG scripts/vipseg_s3dis.sh] [DECISION D-12] |
| `random_sample` | false | [VIPSEG main.py:69] |

The loader's own defaults (`num_point=4096`, `pc_attribs='xyz'`) are wrong for this project; always pass the values above [VIPSEG dataloaders/loader.py:118,232].

### 4.2 Algorithm (one episode)

1. **Candidate blocks per class.** A block is a candidate for class c if it holds **more than** `max(int(0.05 · n_points_in_block), 100)` points of c [VIPSEG dataloaders/s3dis.py:54-57].
2. **Classes.** Sample N distinct classes from the active pool (training or test classes) [VIPSEG dataloaders/loader.py:163]. The order of sampling defines local labels 1…N.
3. **Blocks.** For each sampled class, draw `n_queries` query blocks, then K support blocks, never reusing a block within the episode [VIPSEG dataloaders/loader.py:181-194].
4. **Points.** From a block with fraction r of target-class points, draw `int(r · 2048)` target points without replacement plus `2048 − int(r · 2048)` points from the whole block (with replacement only if the block has fewer than 2048 points), then shuffle the point order [VIPSEG dataloaders/loader.py:37-58]. The expected target fraction is therefore ≈ r + (1 − r)·r: the target class is over-represented (measured 0.36 → 0.59).
5. **Features.** `xyz ← xyz − min(xyz)`; training augmentation on `xyz` (rotation about z, jitter σ = 0.01 clipped at 0.05, shift); `XYZ = (xyz − min(xyz)) / max(xyz − min(xyz))` per axis ∈ [0, 1]; `rgb ← rgb / 255`; concatenate `xyz, rgb, XYZ` [VIPSEG dataloaders/loader.py:65-80,92-113]. The encoder ignores columns 0–2 [VIPSEG models/encoder.py:645] and the minimum is subtracted again before `XYZ`, so the shift augmentation has no effect on the model input [DECISION D-12].
6. **Labels.** Support: binary mask `label == sampled_class`. Query: `k` if the point's class is the k-th sampled class, else 0 [VIPSEG dataloaders/loader.py:82-88].

### 4.3 Loader output (one episode)

| Field | Shape | dtype | Source |
| :--- | :--- | :--- | :--- |
| `support_x` | `[N, K, 9, 2048]` | float32 | [VIPSEG dataloaders/loader.py:220,280] |
| `support_y` | `[N, K, 2048]` ∈ {0, 1} | int32 | [VIPSEG dataloaders/loader.py:168,221] |
| `query_x` | `[N·n_queries, 9, 2048]` | float32 | [VIPSEG dataloaders/loader.py:222,281] |
| `query_y` | `[N·n_queries, 2048]` ∈ {0..N} | int64 | [VIPSEG dataloaders/loader.py:223,281] |
| `sampled_classes` | `[N]` global class ids | int32 | [VIPSEG dataloaders/loader.py:171,283] |

* `batch_task_collate` returns exactly **one** episode per loader item (`batch[0]`) and moves the channel axis before the points [VIPSEG dataloaders/loader.py:277-283]. The model converts to the shapes of 02 §1.
* A batch of 4 episodes = 4 loader items, each forwarded separately, with the mean loss back-propagated once [PAPER §4.1] [DECISION D-12].
* CLIP prompts are built from `sampled_classes` through the class-name file (§2.2) [DECISION D-13].

---

## 5. Training schedule

| Parameter | S3DIS | ScanNet | Source |
| :--- | :--- | :--- | :--- |
| Optimiser | AdamW, lr 1e-3, weight decay 0.1 | same | [PAPER §4.1] |
| Scheduler | StepLR, ×0.5 every 10 epochs | same | [PAPER §4.1] |
| Batch | 4 episodes | 4 episodes | [PAPER §4.1] |
| Epochs | 50 | 30 | [PAPER §4.1] |
| Episodes per epoch | 480 | 800 | [DECISION D-12] |
| Optimiser steps per epoch | 120 | 200 | [DECISION D-12] |
| Total training episodes | 24,000 | 24,000 | [DECISION D-12] [VIPSEG scripts/vipseg_s3dis.sh] [VIPSEG scripts/vipseg_scannet.sh] |
| Validation | every 10 epochs | every 10 epochs | [DECISION D-15] |
| Checkpoint kept | best validation mIoU, plus the last epoch | same | [DECISION D-15] [VIPSEG runs/training.py:82-98] |

* Validation uses `MyTestDataset(mode='valid')`, test-time evaluation `MyTestDataset(mode='test')`; both are built from the **test classes** with different random episodes [VIPSEG dataloaders/loader.py:235-244] [VIPSEG runs/training.py:58-73]. Selecting the checkpoint on them leaks test-class information; this matches VIP-Seg and is logged next to the last-epoch result [DECISION D-15].
* One run per (dataset, fold, N, K, modality).

---

## 6. Evaluation

### 6.1 Protocol

Primary protocol [DECISION D-08]:

* Episodes: `MyTestDataset(mode='test', num_episode_per_comb=100, n_queries=1)`, i.e. 100 fixed episodes for every combination of N test classes, cached as `.h5` [VIPSEG dataloaders/loader.py:230-267] [VIPSEG scripts/vipseg_eval_s3dis.sh].
* Episode counts per fold:

  | Dataset (test classes) | 2-way | 3-way |
  | :--- | ---: | ---: |
  | S3DIS (6) | C(6,2)·100 = 1,500 | C(6,3)·100 = 2,000 |
  | ScanNet (10) | C(10,2)·100 = 4,500 | C(10,3)·100 = 12,000 |

* Model in eval mode, z = 0, no augmentation [DECISION D-06] [VIPSEG dataloaders/loader.py:236].
* Ablation `eval_protocol=random600`: 600 episodes with random class draws [PAPER §4.1] [DECISION D-08].

### 6.2 Metric

Accumulated over **all** episodes of the fold, per global test class, background excluded [VIPSEG runs/training_free.py:14-61]:

$$\text{IoU}_c = \frac{TP_c}{GT_c + PRED_c - TP_c + 0.001}, \qquad \text{mIoU} = \frac{1}{|\mathcal{C}_{test}|}\sum_{c \in \mathcal{C}_{test}} \text{IoU}_c$$

* Local labels are mapped back to global classes through `sampled_classes`; local 0 counts as background [VIPSEG runs/training_free.py:36-52].
* Per-episode averaging is not used [DECISION D-08].
* Report S0, S1 and `Avg = (S0 + S1) / 2` for every (N, K) setting, as in Tables 2–3 [PAPER Tab.2] [PAPER Tab.3].
* L2 reproduces its paper number with this protocol: the released S0 2-way 1-shot checkpoint evaluates 1,500 test episodes to `Mean IoU: 0.722026`, the 72.20 of Table 6 [PAPER Tab.6] [VIPSEG log_s3dis_VIPSeg/log_S0_N2_K1_0.722026/log_vipseg_eval.txt:3-9] [DECISION D-08].
