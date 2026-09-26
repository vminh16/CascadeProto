"""P8-1..8 (05 §3.8s): the condition split of [DECISION D-41], beyond the paper. CPU, float64, synthetic."""

import ast
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from experiments import p6_prototype_probe as p6
from experiments import p7_propagation_probe as p7
from experiments import p8_condition_probe as p8

REPO = Path(__file__).resolve().parents[1]
ATOL = 1e-12


def episode(bq=2, p=64, d=8, n=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    f_q = torch.rand(bq, p, d, generator=g, dtype=torch.float64)
    f_s = torch.rand(n, 1, p, d, generator=g, dtype=torch.float64)
    support_y = (torch.rand(n, 1, p, generator=g) < 0.3).long()
    labels = torch.randint(0, n + 1, (bq, p), generator=g)
    return f_q, f_s, support_y, labels


def test_p8_1_own_mask():
    m = p8.own_mask(2, 3)
    assert m.tolist() == [[False, True, False], [False, False, True]]
    with pytest.raises(ValueError, match="one query block per way"):
        p8.own_mask(3, 3)


def test_p8_2_condition_oracles_partition_the_foreground_oracle():
    f_q, f_s, sy, labels = episode()
    labels[0][labels[0] == 2] = 1  # class 2 absent from block 0
    rows = p6.base_rows(f_q, f_s, sy)
    out = {w: p8.condition_oracle_rows(f_q, labels, rows, w) for w in p8.WHICH}
    assert torch.equal(out["fg"], p6.oracle_rows(f_q, labels, rows, "fg"))
    for b in range(2):
        for c in range(3):
            present = bool((labels[b] == c).any())
            ref = F.normalize(F.normalize(f_q[b][labels[b] == c], dim=-1).sum(0), dim=-1) if present else None
            own = c == b + 1
            for w in ("own", "other"):
                if c >= 1 and present and own == (w == "own"):
                    assert torch.allclose(out[w][b, c], ref, atol=ATOL)
                else:
                    assert torch.equal(out[w][b, c], rows[b, c])
    assert torch.equal(out["own"][:, 0], rows[:, 0]) and torch.equal(out["other"][:, 0], rows[:, 0])
    with pytest.raises(ValueError, match="which"):
        p8.condition_oracle_rows(f_q, labels, rows, "bg")


def test_p8_3_alignment_records_by_hand():
    f_q, f_s, sy, labels = episode(seed=1)
    labels[1][labels[1] == 1] = 0  # class 1 absent from block 1
    rows = p6.base_rows(f_q, f_s, sy)
    pred = torch.randint(0, 3, labels.shape, generator=torch.Generator().manual_seed(2))
    rec = p8.alignment(f_q, labels, rows, pred, [9, 5])
    assert rec.shape == (3, 5)  # block 0: classes 1, 2; block 1: class 2
    expect = [(0, 1, 9, 0), (0, 2, 5, 1), (1, 2, 5, 0)]  # (block, local class, global class, condition)
    for r, (b, k, cls, cond) in zip(rec, expect):
        o = F.normalize(F.normalize(f_q[b][labels[b] == k], dim=-1).sum(0), dim=-1)
        assert r[0] == cls and r[1] == cond and abs(r[2] - float(o @ rows[b, k])) < ATOL
        assert r[3] == float((labels[b] == k).sum()) and r[4] == float(((labels[b] == k) & (pred[b] == k)).sum())


def test_p8_4_split_logits_read_labels_only_in_the_oracles():
    f_q, f_s, sy, labels = episode(seed=3)
    out = p8.split_logits(f_q, f_s, sy, labels)
    rows = p6.base_rows(f_q, f_s, sy)
    assert torch.equal(out["U"], p6.rule_logits(f_q, rows))
    assert torch.equal(out["both"], p7.both_logits(f_q, f_s, sy))
    other = p8.split_logits(f_q, f_s, sy, torch.zeros_like(labels) + (labels + 1) % 3)
    assert torch.equal(out["U"], other["U"]) and torch.equal(out["both"], other["both"])
    assert not torch.equal(out["oracle_fg"], other["oracle_fg"])
    for w in p8.WHICH:
        assert torch.equal(out[f"oracle_{w}"], p6.rule_logits(f_q, p8.condition_oracle_rows(f_q, labels, rows, w)))


def test_p8_5_counts_and_rank_correlation():
    y, pred = np.array([0, 1, 1, 2, 1]), np.array([1, 1, 0, 2, 1])
    assert p8.class_counts(y, pred, 1) == (3.0, 2.0)
    x = np.array([0.1, 0.5, 0.3, 0.9])
    assert p8.spearman(x, 2 * x + 1) == pytest.approx(1.0) and p8.spearman(x, -x) == pytest.approx(-1.0)
    rx, ry = np.array([0.5, 0.5, 2.0]), np.array([0.0, 1.0, 2.0])  # average ranks of [1, 1, 2] and [1, 2, 3]
    assert p8.spearman(np.array([1.0, 1.0, 2.0]), np.array([1.0, 2.0, 3.0])) == pytest.approx(np.corrcoef(rx, ry)[0, 1])
    assert np.isnan(p8.spearman(np.ones(4), x)) and np.isnan(p8.spearman(x[:2], x[:2]))


def test_p8_6_alignment_summary():
    rec = np.array([[9, 0, 0.8, 10, 8], [9, 0, 0.6, 10, 4], [9, 1, 0.2, 10, 1], [5, 0, 0.7, 20, 10],
                    [5, 1, 0.4, 5, 0]], dtype=np.float64)
    s = p8.summarize_alignment(rec, [9, 5])
    assert s["9_own"]["blocks"] == 2 and s["9_own"]["cos_mean"] == pytest.approx(0.7)
    assert s["9_own"]["recall_pooled"] == pytest.approx(12 / 20)
    assert s["all_own"]["blocks"] == 3 and s["all_other"]["recall_pooled"] == pytest.approx(1 / 15)
    assert s["all_own"]["spearman_cos_recall"] == pytest.approx(1.0)  # recalls 0.8, 0.4, 0.5 against cos 0.8, 0.6, 0.7


def fake(g_own=10.0, g_other=3.0, phi=0.6, ci=(0.1, 0.3), events=100):
    u = 0.55
    split = {"valid": {"miou": {"U": u, "oracle_own": u + g_own / 100, "oracle_other": u + g_other / 100,
                                "oracle_fg": u + (g_own + g_other) / 100},
                       "alignment": p8.summarize_alignment(np.array([[9, 0, 0.8, 10, 8], [9, 1, 0.2, 10, 1],
                                                                     [9, 0, 0.5, 10, 3]], dtype=np.float64), [9])}}
    s = {"events": events, "phi": phi, "v1_minus_v0_ci": list(ci), "r_v0": 0.2, "r_v1": 0.5, "r_v2": 0.3,
         "r_own": 0.7, "r_a_v0": 0.7, "r_a_v2": 0.6}
    return split, {"U": s, "both": dict(s, phi=0.5)}


def test_p8_7_rules():
    v = dict(p8.decide(*fake()))
    assert "P8.1 other-condition bound holds" in v and "P8.3 own-condition bound holds" in v
    assert "P8.2 density causal" in v and "P8.4 next run: instance-alignment training" in v
    assert "condition-balanced training" in v["P8.4 next run: instance-alignment training"]
    v = dict(p8.decide(*fake(g_own=2.0, g_other=4.0)))
    assert "P8.4 next run: condition-balanced training" in v
    v = dict(p8.decide(*fake(phi=0.1)))
    assert "P8.2 density not density" in v and "condition-balanced" not in v["P8.4 next run: instance-alignment training"]
    assert "P8.2 density partial" in dict(p8.decide(*fake(ci=(-0.01, 0.3))))  # phi >= 0.5 without a positive CI
    assert "P8.2 density partial" in dict(p8.decide(*fake(phi=0.3)))
    assert "P8.2 density undefined" in dict(p8.decide(*fake(phi=float("nan"))))
    v = dict(p8.decide(*fake(g_own=0.5, g_other=0.5)))
    assert "P8.1 other-condition bound below" in v and "P8.4 next run: none" in v
    v = dict(p8.decide(*fake(g_own=0.5, g_other=4.0, phi=0.05)))
    assert "P8.4 next run: none" in v  # the other-condition bound needs a density cause
    split, interv = fake()
    assert p8.decide({}, interv)[0][0] == "incomplete"
    assert p8.decide(split, None)[-1][0] == "incomplete"
    assert p8.decide(split, {"U": {"events": 0}})[-1][0] == "incomplete"


def test_p8_8_files_parse_as_python_310():
    for f in ("experiments/p8_condition_probe.py", "tests/test_condition_split.py"):
        ast.parse((REPO / f).read_text(encoding="utf-8"), feature_version=(3, 10))
