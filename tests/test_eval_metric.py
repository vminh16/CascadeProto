"""EVAL-1..3 (05 §3.7): the evaluation metric is VIP-Seg's accumulated IoU (04 §6.2, D-08).

pipeline.evaluation calls VIP-Seg's `evaluate_metric`, whose module imports the VIP-Seg encoder,
so these tests carry the `cuda` marker and run in the GPU environment.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.cuda

TEST_CLASSES = [3, 4, 10, 11]  # global ids


class SilentLogger:
    def cprint(self, text):
        pass


def reference_miou(preds, gts, label2class, test_classes):
    """04 §6.2 written independently: IoU_c = TP / (GT + PRED - TP + 0.001), mean over test classes."""
    tp, gt_n, pred_n = (dict.fromkeys(test_classes, 0) for _ in range(3))
    for pred, gt, classes in zip(preds, gts, label2class):
        glob_gt = np.where(gt > 0, np.asarray(classes)[np.maximum(gt - 1, 0)], -1)  # [B_q, 2048]
        glob_pred = np.where(pred > 0, np.asarray(classes)[np.maximum(pred - 1, 0)], -1)
        for c in test_classes:
            tp[c] += int(((glob_gt == c) & (glob_pred == c)).sum())
            gt_n[c] += int((glob_gt == c).sum())
            pred_n[c] += int((glob_pred == c).sum())
    return float(np.mean([tp[c] / (gt_n[c] + pred_n[c] - tp[c] + 0.001) for c in test_classes]))


def random_episodes(n_episodes, seed):
    rng = np.random.default_rng(seed)
    preds, gts, classes = [], [], []
    for _ in range(n_episodes):
        cls = rng.choice(TEST_CLASSES, 2, replace=False)
        gts.append(rng.integers(0, 3, (1, 64)))
        preds.append(rng.integers(0, 3, (1, 64)))
        classes.append(cls)
    return preds, gts, classes


def test_eval1_matches_the_spec_formula():
    from pipeline.evaluation import accumulated_miou

    preds, gts, classes = random_episodes(20, seed=0)
    got = accumulated_miou(SilentLogger(), preds, gts, classes, TEST_CLASSES)
    assert got == pytest.approx(reference_miou(preds, gts, classes, TEST_CLASSES), abs=1e-9)


def test_eval2_accumulates_instead_of_averaging_episodes():
    from pipeline.evaluation import accumulated_miou

    classes = [np.array([3, 4]), np.array([3, 4])]
    # episode 1: class 3 perfect on 1000 points; episode 2: class 3 completely missed on 10 points
    gts = [np.array([[1] * 1000 + [2] * 10]), np.array([[1] * 10 + [2] * 1000])]
    preds = [np.array([[1] * 1000 + [2] * 10]), np.array([[0] * 10 + [2] * 1000])]
    got = accumulated_miou(SilentLogger(), preds, gts, classes, [3, 4])
    per_episode = np.mean([reference_miou([p], [g], [c], [3, 4]) for p, g, c in zip(preds, gts, classes)])
    assert got == pytest.approx(reference_miou(preds, gts, classes, [3, 4]), abs=1e-9)
    assert abs(got - per_episode) > 0.1


def test_eval3_background_is_not_averaged():
    from pipeline.evaluation import accumulated_miou

    classes = [np.array([3, 4])]
    gts = [np.array([[0] * 500 + [1] * 10 + [2] * 10])]
    preds = [np.array([[0] * 500 + [1] * 10 + [2] * 10])]
    # perfect on classes 3 and 4 -> mean over test classes {3, 4} only, background ignored
    assert accumulated_miou(SilentLogger(), preds, gts, classes, [3, 4]) == pytest.approx(1.0, abs=1e-3)
    # classes never present in the episodes count with IoU 0, as in VIP-Seg
    assert accumulated_miou(SilentLogger(), preds, gts, classes, [3, 4, 10]) == pytest.approx(2 / 3, abs=1e-3)
