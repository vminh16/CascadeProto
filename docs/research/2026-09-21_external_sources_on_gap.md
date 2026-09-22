# External sources on the 30-point gap: what published work says a "plain prototype baseline" scores

Status: 2026-09-21. Web research against primary sources only (arXiv, conference proceedings, official
repositories, the vendored VIP-Seg code). It complements `2026-09-21_reproduction_report.md`, which
this note does not repeat. Every number carries its source; "not found" means searched and not found.

## Verdict

Every primary source consulted puts a **backbone + masked-average-pooling + prototype-matching baseline
at 47–55 mIoU** on S3DIS S0, 2-way 1-shot, in the same 2,048-point, class-split protocol we use.
That holds for the ProtoNet row of AttMPTI (48.39), the training-free Seg-NN (49.45), and, most
importantly, for **three ablations by CascadeProto's own first author** on his earlier networks:
TaylorSeg without its prototype module (49.42), DyPolySeg without its prototype module (49.17 / 52.21).
Our baseline (49.08) sits exactly there. **No published method we could find reports > 80 on S3DIS S0
2-way 1-shot in this protocol**; the highest are VIP-Seg (72.20 in its released logs, 73.50 on its
NeurIPS poster) and DyPolySeg (72.02). The paper's 82.72 baseline is therefore not only inconsistent with
VIP-Seg, it is 30+ points above every comparable baseline in the literature, including the author's own.

Known benchmark pitfalls (foreground leakage, model selection on test classes, text of the novel class
names) are real, but they are either **shared by VIP-Seg and by us** (so they cannot open a gap between
the two) or worth a few points, not thirty.

---

## 1. VIP-Seg (NeurIPS 2025): baseline, pretraining, schedule

**Paper access.** The paper PDF (https://openreview.net/pdf?id=r8zHRmM4uE, linked from the README of
https://github.com/changshuowang/VIP-Seg_NeurIPS2025 and from https://mlanthology.org/neurips/2025/wang2025neurips-reasoning/)
is behind an OpenReview browser challenge and could not be fetched; it is not on arXiv (arXiv API title
search "visual introspective": no hit). **VIP-Seg's own ablation table, and so its plain-baseline number,
could not be read.** What could be read:

* **NeurIPS poster** (https://neurips.cc/media/PosterPDFs/NeurIPS%202025/115811.png, linked from
  https://neurips.cc/virtual/2025/poster/115811), "Table 1: Few-shot Results (%) on S3DIS": VIP-Seg
  2-way 1-shot **S0 73.50, S1 74.92**, 2-way 5-shot 73.84 / 76.88. These differ from the released logs
  (`log_S0_N2_K1_0.722026`, `log_S1_N2_K1_0.760875`, `log_S0_N2_K5_0.764836`), which CascadeProto's
  Table 2 copies (72.20 / 76.09 / 76.48). Which version the camera-ready paper prints is unverified.
  The poster contains no ablation table.
* **Pretraining: none.** The poster's Fig. 1 contrasts "Pre-training DGCNN / Fine-tuning" (previous
  methods) with "VIP-Seg — No Pre-training"; the README says results are obtained "without requiring
  pre-training". The code agrees: `runs/training.py:25-26` builds the `Learner` from scratch and
  loads a checkpoint only after training, for the test (`runs/training.py:105`).
* **Schedule** (`scripts/vipseg_s3dis.sh`): 24,000 iterations, batch 1 episode, AdamW lr 1e-3,
  weight decay 0.1 on all parameters (`models/vipseg_learner.py:16-21`), StepLR ×0.5 every 7,000
  iterations, validation every 2,000.
* **Augmentation** (`--pc_augm --pc_augm_shift 0.1` plus `main.py:55-67` defaults; implementation
  `dataloaders/loader.py:90-111`): random rotation about z, Gaussian jitter σ 0.01 clipped at 0.05 on
  all 9 channels, random shift ±0.1 m; no scaling, no mirroring, no colour. Our `pipeline/episodes.py:27`
  uses the same dictionary.
* **Model selection on the test classes.** The "valid" set is `MyTestDataset` in mode `valid`, built from
  the **test classes** (`dataloaders/loader.py:231-244`, `runs/training.py:52-62`); the best of 12
  validation scores is kept (`runs/training.py:88-97`) and then scored on a second draw of test-class
  episodes. This is inherited from AttMPTI and is shared by us (D-15).

**Indirect evidence for VIP-Seg's baseline: its predecessor DyPolySeg** (ICML 2025, same first author;
same design of polynomial local convolution + Mamba blocks, prototype module on top;
https://raw.githubusercontent.com/mlresearch/v267/main/assets/wang25aa/wang25aa.pdf):

| DyPolySeg configuration, S3DIS 2-way 1-shot | S0 | S1 | Avg | source |
| :--- | ---: | ---: | ---: | :--- |
| LoConv only (plain prototype matching) | 47.32 | 50.05 | 48.68 | Table 3, row 1 |
| LoConv + DyHoConv + Mamba, **no PCM** (backbone + prototypes) | 52.21 | 54.35 | 53.28 | Table 3, row 4 |
| full encoder, 0 PCM | 49.17 | 52.32 | 50.75 | Table 6 / Table 5 row 1 |
| + 1 PCM | 68.42 | 70.18 | 69.30 | Table 6 |
| full DyPolySeg (2 PCM) | 72.02 | 73.82 | 72.92 | Tables 1, 3, 6 |

DyPolySeg §4.2: pretraining-free, AdamW lr 0.001 halved every 7,000 iterations, rotation / translation /
jitter augmentation, 100 test episodes — the same recipe as VIP-Seg's script. On this recipe the
prototype-module-free network scores **49–52 on S0**; the prototype module is worth ~20 points.

**TaylorSeg** (AAAI 2025, same first author, https://arxiv.org/pdf/2504.02454), Table 4, "Ablation for
the APP Module in TaylorSeg-PN", S3DIS 2-way 1-shot: without APP **49.42 / 52.67 / 51.05** (S0 / S1 /
Avg); + ACP 51.02; + CAP 64.64; full 67.12. Again: the plain prototype baseline is ~49 on S0.

## 2. The CascadeProto paper: public status

| question | finding |
| :--- | :--- |
| arXiv | Not found (arXiv API, title search "prototype purification": 0 hits; author search `au:Wang_Changshuo`, 2023–2026: no such title). |
| OpenReview / venue | Not found by web search; the only hits for the title are this repository. The local copy is `10069.pdf`, presumably a submission ID; the venue is not identifiable from primary sources. |
| official code | https://github.com/changshuowang/CascadeProto contains only a README, "We will release it soon." (repository listing: updated Jul 1, 2026). |
| errata, reviews questioning numbers | None found (no public forum to hold them). |
| protocol stated in the paper | From our copy (spec 00 D-07/D-08, audit §"Schedule"): "standard N-way K-shot episodic protocol [34]" (AttMPTI); "Areas 1, 2, 3, 4, 6 for training and Area 5 for testing under two category splits S0 and S1"; mIoU "averaged over S0 and S1 across 600 randomly sampled episodes"; AdamW 1e-3, wd 0.1, StepLR ×0.5 per 10 epochs, batch 4, 50 epochs. **Pretraining is not mentioned**; the metric formula is not given. The Area-5 statement contradicts the class-only split of [34] (AttMPTI §4.1, below). |

## 3. Benchmark pitfalls that inflate numbers

### 3.1 Foreground leakage (COSeg, An et al., CVPR 2024, https://arxiv.org/abs/2403.00592)

§3.2 "Issues in the Current Setting": the AttMPTI sampler first draws target-class points in proportion
to their share and then draws the rest from the whole block, so foreground points are sampled twice and
are denser than background; models "segment foreground classes by identifying denser regions". The
same code is in our inherited loader, `dataloaders/loader.py:39-51` (`random_sample=False` is VIP-Seg's
default and ours). Table 1 (1-way, DGCNN backbone, 2,048 points, models retrained in each setting):

| method, S3DIS 1-way | 1-shot S0 w/ FG | 1-shot S0 w/o FG | drop | 5-shot mean w/ → w/o |
| :--- | ---: | ---: | ---: | :--- |
| AttMPTI | 64.89 | 41.56 | −23.3 | 79.82 → 48.34 |
| QGE | 74.05 | 46.27 | −27.8 | 78.93 → 53.76 |
| QGPA | 62.72 | 35.62 | −27.1 | 81.80 → 45.52 (−36.28, the largest drop quoted in §5.2) |

ScanNet drops are of the same size (AttMPTI 1-shot mean 60.39 → 32.58). **Relevance here:** leakage
inflates every number in CascadeProto's Table 2, VIP-Seg's 72.20, and our 49.08 alike, because all use
the same sampler. It would explain a 30-point gap only if CascadeProto's rows were run with *more*
leakage than VIP-Seg's, which nothing in the paper indicates.

### 3.2 Sparse point distribution (COSeg §3.2)

2,048 points per 1 m block makes objects hard to recognise. COSeg's corrected setting: 0.02 m voxel grid,
up to 20,480 points, uniform sampling, backbone pretrained 100 epochs per fold (§5.1). In that setting
S3DIS 2-way 1-shot S0 is AttMPTI 31.09, QGE 33.45, QGPA 25.52, COSeg 37.44 (COSeg Table 2; repeated in
MM-FSS Table 1). This lowers numbers; it cannot raise ours.

### 3.3 Other protocol facts

* **Class split over all areas; the same block can appear in training and test.** AttMPTI §4.1
  (https://arxiv.org/abs/2006.12052): classes are split alphabetically into two folds; "the same point
  cloud can appear in both Ttrain and Ttest, but the annotations ... are different". No Area-5 split.
* **Checkpoint selection on test classes** (§1 above): present in VIP-Seg and in our pipeline. No source
  quantifies it for this benchmark; our own `best` vs `last` differ by < 2.3 points (report §3.3).
* **Query-label leakage / evaluation on training classes:** no primary source reporting either for this
  benchmark was found.

## 4. Reference numbers, S3DIS S0 2-way 1-shot

Old setting (2,048 points, AttMPTI sampler, class split, 100 episodes per class combination) unless noted.

| method | pretrained backbone | S0 | source |
| :--- | :---: | ---: | :--- |
| FT (fine-tuning) — labelled "DGCNN" in later tables | yes | 36.34 | AttMPTI Table 1 |
| ProtoNet (DGCNN + linear mapper, mean prototype, squared Euclidean) | yes (100 ep.) | **48.39** | AttMPTI Table 1, §4.2–4.3 |
| AttProtoNet / MPTI / AttMPTI | yes | 50.98 / 52.27 / 53.77 | AttMPTI Table 1 |
| Point-NN / **Seg-NN (training-free)** | no training | 42.12 / **49.45** | Seg-PN Table 1 (https://arxiv.org/abs/2404.04050) |
| TaylorSeg-NN (training-free) | no training | 52.30 | TaylorSeg Table 1 |
| TaylorSeg-PN without APP (plain prototypes) | no | **49.42** | TaylorSeg Table 4 |
| DyPolySeg without PCM | no | **49.17–52.21** | DyPolySeg Tables 3, 6 |
| PENet-base | yes | 54.63 | PENet Table V (https://arxiv.org/abs/2509.12878) |
| BFG / 2CBR | yes | 55.60 / 55.89 | Seg-PN Table 1 |
| QGPA (listed as PAP3D) | yes (100 ep.) | 59.45 | QGPA Table V (https://arxiv.org/abs/2305.14335) |
| Seg-PN | no | 64.84 | Seg-PN Table 1 |
| PENet | yes | 65.13 | PENet Table I |
| TaylorSeg-PN | no | 67.12 | TaylorSeg Table 1 |
| SDSimPoint (TNNLS 25, as cited) | — | 68.73 | PENet Table I |
| DyPolySeg | no | 72.02 | DyPolySeg Table 1 |
| VIP-Seg | no | 72.20 (logs) / 73.50 (poster) | repo logs; NeurIPS poster Table 1 |
| **CascadeProto baseline / full model** | no (not stated) | **82.72 / 88.53** | paper Table 4 / Table 2 |
| *this repository*: baseline / full / VIP-Seg checkpoint | no | 49.08 / 57.15 / 71.97 | reproduction report §2 |

Corrected setting (20,480 points, uniform sampling; not comparable with the table above): COSeg 37.44,
MM-FSS 41.98 (MM-FSS Table 1), WARM ≤ ~51 on 2-way 1-shot (https://arxiv.org/abs/2509.13907, Table 3).

**Methods with > 80 on S3DIS S0 2-way 1-shot, 2024–2025: none found.** The papers above (the most recent
being PENet, 2025, and WARM, ECCV 2026) all report ≤ 73.5 in the old setting.

## 5. Text / CLIP prompts with the names of the test classes

| source | what the text is | gain from text |
| :--- | :--- | :--- |
| MM-FSS (An et al., ICLR 2025, https://proceedings.iclr.cc/paper_files/paper/2025/file/8b21a7ea42cbcd1c29a7a88c444cce45-Paper-Conference.pdf) | LSeg text embeddings of `background` and the **N target (novel) class names**, used at training and at test time (§3.2 "Text Embeddings"); argued to be "cost-free" because the name is known when the support is annotated (§1). Corrected 20,480-point setting. | Table 3d, ScanNet 1-way, mean of splits: 3D only 40.69 / 45.51 (1-/5-shot) → + image 41.45 / 46.38 → **+ text 44.73 / 50.07**, i.e. **+3.3 / +3.7** from the class names |
| QGPA (He et al., TIP 2023, https://arxiv.org/abs/2305.14335), Table VII | Zero-shot: support annotations replaced by prototypes projected from **word2vec or CLIP embeddings of the class names**; no support points at test time. Old 2,048-point setting. | S3DIS 2-way: word2vec 59.98 / 63.54, CLIP 61.09 / 64.91 (columns = model jointly trained in the 1-shot / 5-shot setting), i.e. **names alone ≈ one visual shot** (its few-shot S0 is 59.45); 3DGenZ with word2vec: 34.93 / 36.12 |

Reading: using the test-class name is an accepted, disclosed design choice (MM-FSS, QGPA), not a hidden
leak; the published gain is **~3–4 points** on top of a visual model, and names alone get to roughly the
level of a 1-shot visual model. CascadeProto's LMA uses CLIP text of the episode's class names; its
Table 4 credits it +1.21 and we measure +0.57. No source suggests text can add ~30 points to a
prototype baseline.

## 6. What could not be established

* VIP-Seg's ablation table and plain-baseline number (paper PDF not retrievable; DyPolySeg and TaylorSeg
  ablations are the closest proxies).
* Why VIP-Seg's poster (73.50) and released logs (72.20) differ, and which the camera-ready prints.
* CascadeProto's venue, reviews, and any unpublished protocol detail (pretraining, sampler, metric code).
