"""ALT-1..4 (gate G1): alternative mIoU definitions of pipeline/metrics_alt.py, diagnostic only."""

import numpy as np
import pytest

from pipeline.metrics_alt import accumulated_counts, alternative_metrics, episode_ious


def vipseg_reference(preds, gts, label2class_list, test_classes):
    """The loop of VIP-Seg's evaluate_metric, copied (runs/training_free.py imports the CUDA encoder)."""
    n = len(test_classes) + 1
    gt_c, pos_c, tp_c = [0] * n, [0] * n, [0] * n
    for batch_gt, batch_pred, label2class in zip(gts, preds, label2class_list):
        for j in range(batch_pred.shape[0]):
            for k in range(batch_pred.shape[1]):
                gt, pred = int(batch_gt[j, k]), int(batch_pred[j, k])
                gi = 0 if gt == 0 else test_classes.index(label2class[gt - 1]) + 1
                pi = 0 if pred == 0 else test_classes.index(label2class[pred - 1]) + 1
                gt_c[gi] += 1
                pos_c[pi] += 1
                tp_c[gi] += int(gt == pred)
    iou = [tp_c[c] / float(gt_c[c] + pos_c[c] - tp_c[c] + 0.001) for c in range(n)]
    return float(np.array(iou[1:]).mean()), float(np.array(iou).mean())


def random_episodes(seed=0, episodes=12, n_way=2, points=64):
    g = np.random.default_rng(seed)
    test_classes = [3, 11, 10, 0, 8, 4]
    preds, gts, l2c = [], [], []
    for _ in range(episodes):
        l2c.append(list(g.choice(test_classes, n_way, replace=False)))
        gts.append(g.integers(0, n_way + 1, (n_way, points)))
        preds.append(np.where(g.random((n_way, points)) < 0.6, gts[-1], g.integers(0, n_way + 1, (n_way, points))))
    return preds, gts, l2c, test_classes


def test_alt1_accumulated_metrics_equal_vipseg_loop():
    preds, gts, l2c, tc = random_episodes()
    fg, with_bg = vipseg_reference(preds, gts, l2c, tc)
    m = alternative_metrics(preds, gts, l2c, tc)
    assert m["accumulated_fg"] == pytest.approx(fg, abs=1e-12)
    assert m["accumulated_with_bg"] == pytest.approx(with_bg, abs=1e-12)


def test_alt2_hand_computed_episode():
    """One episode, 2-way, 8 points; every metric worked out by hand."""
    gt = np.array([[0, 0, 0, 1, 1, 2, 2, 2]])
    pred = np.array([[0, 0, 1, 1, 1, 2, 2, 0]])
    # label 0: inter 2, union 4 -> 1/2; label 1: inter 2, union 3 -> 2/3; label 2: inter 2, union 3 -> 2/3
    assert episode_ious(pred, gt, 3, include_bg=False) == pytest.approx([2 / 3, 2 / 3])
    assert episode_ious(pred, gt, 3, include_bg=True) == pytest.approx([1 / 2, 2 / 3, 2 / 3])
    m = alternative_metrics([pred], [gt], [[5, 7]], [5, 7])
    assert m["episode_fg"] == pytest.approx(2 / 3)
    assert m["episode_with_bg"] == pytest.approx((1 / 2 + 2 / 3 + 2 / 3) / 3)
    assert m["point_accuracy"] == pytest.approx(6 / 8)
    assert m["accumulated_fg"] == pytest.approx((2 / 3.001 + 2 / 3.001) / 2)


def test_alt3_labels_map_to_global_classes_across_episodes():
    """The same global class under different episode labels accumulates into one row."""
    gt_a, gt_b = np.array([[1, 1, 0]]), np.array([[2, 2, 0]])
    # label 1 of episode a and label 2 of episode b are both global class 9
    gt_count, pred_count, tp = accumulated_counts([gt_a, gt_b], [gt_a, gt_b], [[9, 4], [4, 9]], [4, 9])
    assert list(gt_count) == [2, 0, 4]  # background, class 4, class 9
    assert list(tp) == list(gt_count) == list(pred_count)


def test_alt4_perfect_prediction():
    preds, gts, l2c, _ = random_episodes(seed=3)
    present = sorted({int(c) for classes in l2c for c in classes})  # an absent test class scores IoU 0
    m = alternative_metrics(gts, gts, l2c, present)
    assert m["episode_fg"] == m["episode_with_bg"] == m["point_accuracy"] == 1.0
    assert m["accumulated_fg"] > 0.999
