### D-07 — Train/test split · `LOCKED`

* **Conflict between L1 and the protocol it cites.** L1: "using Areas 1, 2, 3, 4, 6 for training and Area 5 for testing under two category splits S0 and S1" [PAPER §4.1], but also "We follow the standard N-way K-shot episodic protocol [34]" [PAPER §4.1]. The loader VIP-Seg releases (inherited from AttMPTI [34]) splits by **class only**, over all areas [VIPSEG dataloaders/s3dis.py:20-31], and VIP-Seg's released S0 result (72.20) was produced with it [VIPSEG log_s3dis_VIPSeg/log_S0_N2_K1_0.722026/log_vipseg_eval.txt:7-9]. The other Table 2 baselines are cited, not re-run, so their protocol is assumed to be the same.
* **Decision.** Primary protocol = class-only split with the fold lists of the inherited loader:
  * S3DIS S0 = `beam, board, bookcase, ceiling, chair, column`; S1 = `door, floor, sofa, table, wall, window` [VIPSEG dataloaders/s3dis.py:20-21].
  * ScanNet S0 = `bathtub, bed, bookshelf, cabinet, chair, counter, curtain, desk, door, floor`; S1 = `otherfurniture, picture, refridgerator, shower curtain, sink, sofa, table, toilet, wall, window` [VIPSEG dataloaders/scannet.py:21-22].
  * The paper itself never lists the classes of S0/S1; the old spec's lists were invented.
* **Ablation flag.** `split_protocol = {class_only (default), area5}`. `area5` restricts training scans to Areas 1–4, 6 and test scans to Area 5 (S3DIS only).
