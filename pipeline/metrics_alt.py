"""Alternative mIoU definitions, for diagnosing the absolute gap to the paper (not a result metric).

The primary metric is VIP-Seg's, unchanged [DECISION D-08]: TP/FP/FN accumulated over every episode
per global test class, background excluded [VIPSEG runs/training_free.py:14-61]. The paper's own rows
may have been scored differently from the VIP-Seg row it copies, so this module computes the common
variants from the same predictions:

* `accumulated_fg`: VIP-Seg's metric, re-implemented (must equal the primary number).
* `accumulated_with_bg`: the same counts, background included in the mean.
* `episode_fg`: IoU per episode over its labels 1..N, averaged over episodes.
* `episode_with_bg`: IoU per episode over labels 0..N, averaged over episodes.
* `point_accuracy`: fraction of query points labelled correctly.

Pure numpy, so it runs on the CPU without the VIP-Seg encoder.
"""

from typing import Dict, List, Sequence

import numpy as np

VIPSEG_EPS = 0.001  # added to the IoU denominator [VIPSEG runs/training_free.py:56]


def accumulated_counts(preds: Sequence[np.ndarray], gts: Sequence[np.ndarray],
                       sampled_classes: Sequence[Sequence[int]], test_classes: Sequence[int]):
    """Per global class (index 0 = background, i = test_classes[i-1]): GT count, predicted count, TP.

    Mirrors the loop of [VIPSEG runs/training_free.py:26-53], vectorised.
    """
    test_classes = list(test_classes)
    n = len(test_classes) + 1
    gt_count, pred_count, tp = np.zeros(n), np.zeros(n), np.zeros(n)
    for pred, gt, label2class in zip(preds, gts, sampled_classes):
        pred, gt = np.asarray(pred).ravel(), np.asarray(gt).ravel()  # [B_q * 2048]
        # episode label k (1..N) -> global index test_classes.index(label2class[k-1]) + 1; 0 stays 0
        lut = np.array([0] + [test_classes.index(int(c)) + 1 for c in label2class])  # [N+1]
        g, p = lut[gt], lut[pred]  # [B_q * 2048]
        gt_count += np.bincount(g, minlength=n)
        pred_count += np.bincount(p, minlength=n)
        tp += np.bincount(g[gt == pred], minlength=n)
    return gt_count, pred_count, tp


def episode_ious(pred: np.ndarray, gt: np.ndarray, n_labels: int, include_bg: bool) -> List[float]:
    """IoU of every episode label with a non-empty union; labels 1..N, plus 0 if `include_bg`."""
    pred, gt = np.asarray(pred).ravel(), np.asarray(gt).ravel()
    ious = []
    for k in range(0 if include_bg else 1, n_labels):
        inter = np.sum((pred == k) & (gt == k))
        union = np.sum((pred == k) | (gt == k))
        if union > 0:
            ious.append(inter / union)
    return ious


def alternative_metrics(preds: Sequence[np.ndarray], gts: Sequence[np.ndarray],
                        sampled_classes: Sequence[Sequence[int]], test_classes: Sequence[int]) -> Dict[str, float]:
    gt_count, pred_count, tp = accumulated_counts(preds, gts, sampled_classes, test_classes)
    iou = tp / (gt_count + pred_count - tp + VIPSEG_EPS)  # [C+1]
    per_episode_fg, per_episode_bg = [], []
    for pred, gt, label2class in zip(preds, gts, sampled_classes):
        n_labels = len(label2class) + 1
        fg = episode_ious(pred, gt, n_labels, include_bg=False)
        per_episode_fg.append(np.mean(fg) if fg else 0.0)
        per_episode_bg.append(np.mean(episode_ious(pred, gt, n_labels, include_bg=True)))
    correct = sum(int(np.sum(np.asarray(p) == np.asarray(g))) for p, g in zip(preds, gts))
    total = sum(np.asarray(g).size for g in gts)
    return {"accumulated_fg": float(iou[1:].mean()),
            "accumulated_with_bg": float(iou.mean()),
            "episode_fg": float(np.mean(per_episode_fg)),
            "episode_with_bg": float(np.mean(per_episode_bg)),
            "point_accuracy": correct / total}
