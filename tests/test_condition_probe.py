"""P5-1..14 (05 §3.8n): the condition and presence probe P5 [DECISION D-35]. Beyond the paper.

CPU tests on synthetic tensors and, for the sampler copy, a synthetic scan written to a temporary directory
(synthetic data stays inside tests/, AGENTS guardrail 8). The GPU passes are covered by the smoke run.
"""

import ast
import pathlib

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments import p0_em_probe as p0
from experiments import p5_condition_probe as p5

REPO = pathlib.Path(__file__).resolve().parents[1]
TEST = [6, 1, 9, 7, 2, 5]  # S1's test classes in the loader's order


def test_p5_1_block_status():
    gt = np.array([[1, 1, 2, 0], [2, 2, 0, 0]])  # block 0 holds its own class 1 and class 2; block 1 lacks class 1
    s = p5.block_status(gt, 2)
    assert s.tolist() == [[p5.OWN, p5.OTHER], [p5.ABSENT, p5.OWN]]
    with pytest.raises(ValueError):
        p5.block_status(gt[:1], 2)


def test_p5_2_condition_counts_partition_the_pooled_counts():
    rng = np.random.default_rng(0)
    for _ in range(20):
        gt = rng.integers(0, 3, size=(2, 50))
        gt[1][gt[1] == 1] = 0  # class 1 absent from block 1 half the time
        if rng.random() < 0.5:
            gt[1, :3] = 1
        pred = rng.integers(0, 3, size=(2, 50))
        l2c = np.array([1, 7])  # floor, table
        split = p5.condition_counts(pred, gt, l2c, TEST)
        pooled = p0.episode_counts(pred, gt, l2c, TEST)
        assert np.allclose((split[p5.GT_OWN] + split[p5.GT_OTHER])[1:], pooled[0][1:])
        assert np.allclose((split[p5.TP_OWN] + split[p5.TP_OTHER])[1:], pooled[2][1:])
        assert np.allclose((split[p5.FP_OWN] + split[p5.FP_OTHER] + split[p5.FP_ABSENT])[1:], (pooled[1] - pooled[2])[1:])


def test_p5_3_condition_counts_by_hand():
    gt = np.array([[1, 1, 2, 0], [2, 2, 0, 0]])
    pred = np.array([[1, 0, 0, 1], [2, 1, 1, 0]])  # block 1: class 1 predicted twice where it is absent
    split = p5.condition_counts(pred, gt, np.array([1, 7]), TEST)
    floor, table = TEST.index(1) + 1, TEST.index(7) + 1
    assert split[p5.GT_OWN, floor] == 2 and split[p5.TP_OWN, floor] == 1 and split[p5.FP_OWN, floor] == 1
    assert split[p5.FP_ABSENT, floor] == 2
    assert split[p5.GT_OTHER, table] == 1 and split[p5.TP_OTHER, table] == 0
    assert split[p5.GT_OWN, table] == 2 and split[p5.TP_OWN, table] == 1
    assert (split[:, 0] == 0).all()


def test_p5_4_counterfactuals():
    pooled = np.array([[100, 50, 50], [90, 60, 40], [70, 40, 30]], dtype=float)  # gt, pred, tp; columns bg, c1, c2
    split = np.zeros((7, 3))
    split[p5.GT_OWN, 1:], split[p5.TP_OWN, 1:] = [40, 40], [36, 30]
    split[p5.GT_OTHER, 1:], split[p5.TP_OTHER, 1:] = [10, 10], [4, 9]  # c2's other recall beats its own
    split[p5.FP_ABSENT, 1:] = [5, 0]
    a = p5.counterfactual(pooled, split, "a")
    assert a[2, 1] == pytest.approx(40 + 0.9 * 10 - 4) and a[1, 1] == pytest.approx(60 + 5)
    assert a[2, 2] == 30 and a[1, 2] == 40  # never lowers
    assert (a[:, 0] == pooled[:, 0]).all() and (a[0] == pooled[0]).all()
    b = p5.counterfactual(pooled, split, "b")
    assert b[1, 1] == 55 and b[1, 2] == 40 and (b[2] == pooled[2]).all()
    assert p0.miou_from_counts(a) >= p0.miou_from_counts(pooled) and p0.miou_from_counts(b) >= p0.miou_from_counts(pooled)
    with pytest.raises(ValueError):
        p5.counterfactual(pooled, split, "c")


def test_p5_5_split_summary():
    split = np.zeros((7, 2))
    split[p5.GT_OWN, 1], split[p5.TP_OWN, 1], split[p5.GT_OTHER, 1], split[p5.TP_OTHER, 1] = 80, 72, 20, 5
    split[p5.FP_ABSENT, 1] = 10
    s = p5.split_summary(split)
    assert s["recall_own"][1] == pytest.approx(0.9) and s["recall_other"][1] == pytest.approx(0.25)
    assert s["other_share"][1] == pytest.approx(0.2) and s["fp_absent"][1] == pytest.approx(0.1)
    assert np.isnan(s["recall_own"][0])
    assert s["counts"]["fp_absent"][1] == 10 and s["counts"]["gt_other"][1] == 20


def test_p5_6_presence_fair_oracle():
    torch.manual_seed(0)
    f_q = torch.rand(2, 30, 8, dtype=torch.float64)
    f_s = torch.rand(2, 1, 30, 8, dtype=torch.float64)
    sy = torch.zeros(2, 1, 30, dtype=torch.long)
    sy[0, 0, :10], sy[1, 0, 20:] = 1, 1
    labels = torch.zeros(2, 30, dtype=torch.long)
    labels[0, :10], labels[0, 10:15], labels[1, :12] = 1, 2, 2  # class 1 absent from block 1
    logits = p5.presence_fair_logits(f_q, f_s, sy, labels)
    s = p5.support_directions(f_s, sy)
    assert torch.allclose(s[1], F.normalize(F.normalize(f_s[0, 0, :10], dim=-1).sum(0), dim=0))
    assert torch.allclose(s[0], F.normalize(torch.cat([F.normalize(f_s[0, 0, 10:], dim=-1),
                                                       F.normalize(f_s[1, 0, :20], dim=-1)]).sum(0), dim=0))
    assert torch.allclose(logits[1, :, 1], f_q[1] @ s[1])  # absent class: the support direction competes
    o1 = F.normalize(F.normalize(f_q[0, :10], dim=-1).sum(0), dim=0)
    assert torch.allclose(logits[0, :, 1], f_q[0] @ o1)  # present class: the query's own direction


def test_p5_7_support_rule():
    from models.prototypes import point_prototypes

    f_q, f_s = torch.rand(2, 5, 4), torch.rand(2, 1, 6, 4)
    sy = torch.tensor([[[1, 1, 0, 0, 0, 0]], [[0, 0, 0, 1, 1, 1]]])
    want = torch.einsum("bpd,cd->bpc", f_q, F.normalize(point_prototypes(f_s, sy), dim=-1))
    assert torch.allclose(p5.support_rule_logits(f_q, f_s, sy), want)


def test_p5_8_batch_statistics():
    torch.manual_seed(0)
    net = nn.Sequential(nn.Conv1d(3, 4, 1), nn.BatchNorm1d(4), nn.ReLU(), nn.Conv1d(4, 2, 1), nn.BatchNorm1d(2))
    net.train()
    for _ in range(5):
        net(torch.randn(3, 3, 20) * 2 + 1)  # non-trivial running statistics
    net.eval()
    x = torch.randn(2, 3, 20)
    before = [(m.running_mean.clone(), m.running_var.clone()) for m in net if isinstance(m, nn.BatchNorm1d)]
    y_eval = net(x)
    with p5.batch_statistics(net) as n:
        assert n == 2
        y_b = net(x)
        h = net[0](x)
        want = (h - h.mean(dim=(0, 2), keepdim=True)) / torch.sqrt(h.var(dim=(0, 2), unbiased=False, keepdim=True) + 1e-5)
        assert torch.allclose(net[1](h), want * net[1].weight[None, :, None] + net[1].bias[None, :, None], atol=1e-5)
    assert not torch.allclose(y_b, y_eval)
    assert all(not m.training for m in net)
    for m, (mean, var) in zip([m for m in net if isinstance(m, nn.BatchNorm1d)], before):
        assert torch.equal(m.running_mean, mean) and torch.equal(m.running_var, var) and m.momentum == 0.1
    with pytest.raises(ValueError):
        with p5.batch_statistics(nn.Linear(2, 2)):
            pass


def _write_scan(tmp_path, n=6000, seed=0):
    rng = np.random.default_rng(seed)
    data = np.zeros((n, 7))
    data[:, :3] = rng.random((n, 3)) * [1.0, 1.0, 3.0]
    data[:, 3:6] = rng.integers(0, 256, (n, 3))
    data[:, 6] = rng.choice([1, 2, 7, 12], size=n, p=[0.3, 0.3, 0.2, 0.2])
    (tmp_path / "data").mkdir(exist_ok=True)
    np.save(tmp_path / "data" / "scan_a.npy", data)
    return data


@pytest.mark.parametrize("random_sample", [False, True])
def test_p5_9_sampler_copy_matches_the_inherited_sampler(tmp_path, random_sample):
    from dataloaders.loader import sample_pointcloud

    data = _write_scan(tmp_path)
    classes = np.array([1, 7])
    pc, gt, raw = p5.sample_block(str(tmp_path), "scan_a", classes, 1, random_sample, [0, 1, 2])
    np.random.seed([0, 1, 2])
    pc2, gt2 = sample_pointcloud(str(tmp_path), 2048, "xyzrgbXYZ", False, None, "scan_a", classes, 1,
                                 support=False, random_sample=random_sample)
    assert np.array_equal(pc, pc2) and np.array_equal(gt, gt2)
    assert np.array_equal(p5.episode_labels(raw, classes), gt)
    share = (raw == 1).mean()
    raw_share = (data[:, 6] == 1).mean()
    if random_sample:
        assert abs(share - raw_share) < 0.05
    else:  # the sampled-for class is over-sampled by 2 - pi [DECISION D-35]
        assert share == pytest.approx(raw_share * (2 - raw_share), abs=0.03)


def test_p5_10_fp_kind():
    train = [0, 3, 4, 8, 10, 11]
    assert p5.fp_kind(7, np.array([1, 7]), train, TEST) == "episode"
    assert p5.fp_kind(8, np.array([1, 7]), train, TEST) == "base"
    assert p5.fp_kind(12, np.array([1, 7]), train, TEST) == "clutter"
    assert p5.fp_kind(2, np.array([1, 7]), train, TEST) == "novel_out"


def test_p5_11_sparse_view():
    rng = np.random.default_rng(0)
    p, r = 20000, 0.3
    f = r * (2 - r)
    block = np.concatenate([rng.random((p, 3)), rng.random((p, 3)), np.zeros((p, 3))], axis=1).astype(np.float32)
    mask = (rng.random(p) < f).astype(np.int64)
    view, m = p5.sparse_view(block, mask, np.random.default_rng(1))
    assert view.shape == block.shape and m.shape == mask.shape and view.dtype == block.dtype
    assert set(np.unique(m)) <= {0, 1}
    assert m.mean() == pytest.approx(r, abs=0.01)  # foreground at the raw share: background density
    assert view[:, 6:].min() >= 0 and view[:, 6:].max() == pytest.approx(1.0) and view[:, :3].min() == pytest.approx(0.0)
    with pytest.raises(ValueError):
        p5.sparse_view(block, np.zeros(p, dtype=np.int64), rng)


def test_p5_12_dual_and_control():
    torch.manual_seed(0)
    dense, sparse = torch.randn(2, 100, 3), torch.randn(2, 100, 3)
    dual = p5.dual_logits(dense, sparse)
    assert torch.equal(dual[..., 0], dense[..., 0]) and torch.equal(dual[..., 1:], torch.maximum(dense[..., 1:], sparse[..., 1:]))
    target = int((dual.argmax(-1) != 0).sum())
    ctl = p5.match_fg_count(dense, target)
    assert abs(int((ctl.argmax(-1) != 0).sum()) - target) <= 2
    assert torch.allclose(ctl[..., 1:] - dense[..., 1:], (ctl[..., 1:] - dense[..., 1:]).flatten()[0].expand_as(dense[..., 1:]))
    assert torch.equal(ctl[..., 0], dense[..., 0])


def test_p5_13_phi_and_intervention_summary():
    assert p5.phi(0.4, 0.7, 0.8) == pytest.approx(0.75) and np.isnan(p5.phi(0.8, 0.9, 0.8))
    events = [{"episode": e, "c": 1, "a": 2, "gt_c": [100, 100, 100], "tp_c": [40, 70, 50],
               "gt_a": [200, 200], "tp_a": [180, 150]} for e in range(30)]
    s = p5.intervention_summary(events, {1: 0.8}, boot=200)
    assert s["r_v0"] == pytest.approx(0.4) and s["r_v1"] == pytest.approx(0.7) and s["r_own"] == pytest.approx(0.8)
    assert s["phi"] == pytest.approx(0.75) and s["v1_minus_v0_ci"][0] == pytest.approx(0.3)
    assert s["r_a_v0"] == pytest.approx(0.9) and s["r_a_v2"] == pytest.approx(0.75) and s["episodes"] == 30
    assert p5.intervention_summary([], {})["events"] == 0


def _fixed(cf_a, cf_b, tbn, tbn_lo, tbn600, dual, dual_lo):
    pair = lambda g, lo: {"gain": g, "ci_low": lo, "ci_high": g + 1}  # noqa: E731
    return {"checkpoints": {"e1": {
        "cf_a_gain": cf_a, "cf_b_gain": cf_b, "tbn_gain_extra": tbn600,
        "miou": {"model": 0.732, "support_rule": 0.49},
        "paired": {"tbn_vs_model": pair(tbn, tbn_lo), "dual_vs_control": pair(dual, dual_lo),
                   "oracle_unit_vs_model": pair(12.7, 12), "presence_fair_vs_model": pair(9.0, 8)}}}}


def test_p5_14_rules():
    def rules(fixed, inter=None, leak=None):
        return dict(p5.decide(fixed, inter, leak))

    v = rules(_fixed(2.0, 0.3, 1.5, 0.2, 0.4, 0.6, 0.1), {"events": 10, "phi": 0.6, "v1_minus_v0_ci": [0.05, 0.2]})
    assert "P5.1a causal" in v and "M3" not in v["P5.2 E-b"] and "later arms" in v["P5.3 batch statistics"]
    assert "M2 is trained" in v["P5.4 dual prototypes"] and "neither" not in v
    v = rules(_fixed(2.0, 1.2, 1.5, -0.1, 0.4, 0.6, -0.1), {"events": 10, "phi": 0.1, "v1_minus_v0_ci": [-0.1, 0.2]})
    assert "P5.1b not density" in v and "M3" in v["P5.2 E-b"]
    assert "not adopted" in v["P5.3 batch statistics"] and "not enough" in v["P5.4 dual prototypes"]
    v = rules(_fixed(2.0, 0.0, 1.5, 0.2, -0.1, 0.1, -0.3), {"events": 10, "phi": 0.6, "v1_minus_v0_ci": [-0.01, 0.2]})
    assert "P5.1 in between" in v and "not adopted" in v["P5.3 batch statistics"]
    v = rules(_fixed(0.4, 0.2, 0.0, -1, 0.0, 0.0, -1))
    assert "stops" in v["P5.1 E-a"] and "neither" in v and "P5.1a/b" not in v
    assert "incomplete" in rules(_fixed(2.0, 0.2, 0, -1, 0, 0, -1))["P5.1a/b"]


def test_p5_15_parses_as_python_310():
    for f in ("experiments/p5_condition_probe.py", "tests/test_condition_probe.py"):
        ast.parse((REPO / f).read_text(encoding="utf-8"), feature_version=(3, 10))
