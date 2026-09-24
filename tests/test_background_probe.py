"""BG-1..9 (05 §3.8l): the background-contamination probe P4 [DECISION D-32]. Beyond the paper.

BG-1..8 are CPU tests on synthetic tensors (G1); BG-9 (marker `data`) checks, on the real blocks, that the
support masks derived from `support=False` labels equal the loader's own `support=True` masks.
"""

import ast
import os
import pathlib

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from experiments import p4_background_probe as p4

REPO = pathlib.Path(__file__).resolve().parents[1]
D = 8


def rand(*shape, seed=0):
    return torch.rand(*shape, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)


def test_bg1_prototypes():
    f_s = rand(2, 1, 6, D)
    y = torch.tensor([[[1, 1, 0, 0, 0, 0]], [[0, 0, 0, 1, 1, 1]]])
    fg = p4.fg_prototypes(f_s, y)
    assert torch.allclose(fg[0], f_s[0, 0, :2].mean(0)) and torch.allclose(fg[1], f_s[1, 0, 3:].mean(0))
    keep = y == 0
    assert torch.allclose(p4.bg_prototype(f_s, keep), torch.cat([f_s[0, 0, 2:], f_s[1, 0, :3]]).mean(0))
    with pytest.raises(ValueError):
        p4.bg_prototype(f_s, torch.zeros_like(keep))


def test_bg2_first_prototype_is_route_bs_p0():
    """With every mask-0 point kept, P^0 equals the model's own L2 point prototypes (Eq.3 + D-10)."""
    from models.prototypes import point_prototypes

    f_s = rand(2, 1, 6, D, seed=1)
    y = torch.tensor([[[1, 1, 0, 0, 0, 0]], [[0, 0, 0, 1, 1, 1]]])
    want = F.normalize(point_prototypes(f_s, y), dim=-1)
    assert torch.allclose(p4.first_prototype(f_s, y, y == 0), want, atol=1e-12)


def test_bg3_other_way_score_excludes_the_own_way():
    fg = torch.eye(D, dtype=torch.float64)[:2]  # way 0 -> channel 0, way 1 -> channel 1
    f_s = torch.zeros(2, 1, 3, D, dtype=torch.float64)
    f_s[0, 0, 0, 1] = 1.0  # in way 0's block, a point that looks like way 1
    f_s[0, 0, 1, 0] = 1.0  # in way 0's block, a point that looks like way 0 itself
    f_s[0, 0, 2, 2] = 1.0
    s = p4.other_way_score(f_s, fg)
    assert s[0, 0, 0] == pytest.approx(1.0) and s[0, 0, 1] == pytest.approx(0.0) and s[0, 0, 2] == pytest.approx(0.0)


def test_bg4_purify_keep():
    f_s = torch.zeros(2, 1, 10, D, dtype=torch.float64)
    y = torch.zeros(2, 1, 10, dtype=torch.long)
    y[0, 0, :2], y[1, 0, :2] = 1, 1
    f_s[0, 0, :2, 0], f_s[1, 0, :2, 1] = 1.0, 1.0  # foreground of way 0 on channel 0, way 1 on channel 1
    f_s[0, 0, 2:, 2] = 1.0
    f_s[0, 0, 9, 1], f_s[0, 0, 9, 2] = 5.0, 0.0  # way 0's block holds one point of way 1's class
    f_s[1, 0, 2:, 3] = 1.0
    assert torch.equal(p4.purify_keep(f_s, y, 0.0), y == 0)
    keep = p4.purify_keep(f_s, y, 0.125)  # one of the eight mask-0 points per block
    assert not keep[0, 0, 9] and keep[0, 0, 2:9].all() and not keep[:, :, :2].any()
    assert int(keep[1].sum()) == 7


def test_bg5_run_head_and_row_oracle():
    from models.cascadeproto import CascadeProtoConfig
    from tests.test_cascadeproto import episode, model

    m, ep = model(CascadeProtoConfig(use_lma=False, num_stages=2, l2norm_point_proto=True)).eval(), episode(n=2, k=1)
    f_s, f_q = m.features.encode_episode(ep.support_x, ep.query_x)
    p0 = p4.first_prototype(f_s, ep.support_y, ep.support_y == 0)
    m_eff, logits = p4.run_head(m, f_s, f_q, p0)
    assert torch.allclose(logits, m(ep).logits, atol=1e-10)  # the re-run head is the model
    same = p4.row_oracle_logits(f_q, m_eff, ep.query_y, [])
    assert torch.allclose(same, logits)
    bg_only = p4.row_oracle_logits(f_q, m_eff, ep.query_y, [0])
    assert torch.allclose(bg_only[..., 1:], logits[..., 1:]) and not torch.allclose(bg_only[..., 0], logits[..., 0])


def test_bg6_auc():
    pos, neg = np.array([0, 0, 5]), np.array([5, 0, 0])
    assert p4.auc(pos, neg) == 1.0 and p4.auc(neg, pos) == 0.0
    assert p4.auc(np.array([2, 2]), np.array([2, 2])) == pytest.approx(0.5)
    assert np.isnan(p4.auc(np.zeros(3), neg))
    score = torch.tensor([-0.9, 0.9, 0.0])
    hp, hn = p4.auc_terms(score, torch.tensor([False, True, False]), torch.tensor([True, False, True]))
    assert hp.sum() == 1 and hn.sum() == 2 and p4.auc(hp, hn) == 1.0


def test_bg7_class_rates_and_selection():
    r = p4.class_rates(np.array([[10, 4], [8, 5], [6, 2]]))
    assert r["recall"] == [0.6, 0.5] and r["precision"] == [0.75, 0.4]
    res = {"miou": {"model": 0.70, **{f"purify_q{q:g}": 0.70 for q in p4.FRACTIONS}}}
    res["miou"]["purify_q0.2"] = 0.705
    sel = p4.select_fraction(res)
    assert sel["q"] == 0.2 and sel["valid_gain"] == pytest.approx(0.5)


def _test(clean=1.5, clo=0.5, fix=0.8, flo=0.2, gap=0.7, floor=(0.72, 0.80), wall=(0.73, 0.81)):
    rates = {"model": {"recall": [0.9, 0.9, floor[0], 0.9, 0.9, wall[0], 0.9]},
             "clean_bg": {"recall": [0.9, 0.9, floor[1], 0.9, 0.9, wall[1], 0.9]}}
    paired = {"clean_bg": {"gain": clean, "ci_low": clo, "ci_high": clean + 0.5},
              "purify_q0.1": {"gain": fix, "ci_low": flo, "ci_high": fix + 0.4},
              "clean_bg_vs_frozen": {"gain": gap, "ci_low": gap - 0.3, "ci_high": gap + 0.3},
              **{k: {"gain": 5.0, "ci_low": 4.0, "ci_high": 6.0} for k in ("oracle_bg_only", "oracle_fg_only", "oracle_all")}}
    return {"test_classes": [6, 1, 9, 7, 2, 5], "rates": rates, "paired": paired, "frozen": "purify_q0.1",
            "contamination": 0.1, "purifier_auc": 0.7}


def test_bg8_rules():
    v = dict(p4.decide(_test()))
    assert "P4.1 causal" in v and v["P4.2 label-free fix"].endswith("a training-free candidate")
    assert v["P4.3 neck"].endswith("the rule suffices, no neck")
    assert dict(p4.decide(_test(gap=1.2)))["P4.3 neck"].endswith("a learned purification is warranted")
    assert "P4.1 not causal" in dict(p4.decide(_test(clean=0.3, clo=-0.1)))
    assert "P4.1 in between" in dict(p4.decide(_test(floor=(0.72, 0.70))))  # a gain without the recall rise
    assert "P4.1 in between" in dict(p4.decide(_test(clean=0.8, clo=0.2)))
    assert dict(p4.decide(_test(fix=0.4)))["P4.2 label-free fix"].endswith("not enough")
    assert dict(p4.decide(_test(clean=0.3, clo=-0.1)))["P4.3 neck"].startswith("not reached")


@pytest.mark.parametrize("path", ["experiments/p4_background_probe.py", "experiments/c0_background_contamination.py"])
def test_bg10_parses_as_python_3_10(path):
    ast.parse((REPO / path).read_text(encoding="utf-8"), feature_version=(3, 10))


@pytest.mark.data
def test_bg9_derived_masks_equal_the_loaders():
    """support=False labels, same RNG state: (label == way index) is exactly the loader's support mask."""
    from dataloaders.loader import MyDataset, sample_K_pointclouds
    from pipeline.episodes import NUM_POINT, PC_ATTRIBS, WAY_NUM, WAY_RATIO

    data = str(REPO / "datasets" / "S3DIS" / "blocks_bs1_s1")
    if not os.path.isdir(os.path.join(data, "data")):
        pytest.skip("S3DIS blocks not present")
    ds = MyDataset(data, "s3dis", cvfold=1, num_episode=1, n_way=2, k_shot=1, n_queries=1, mode="test",
                   num_point=NUM_POINT, pc_attribs=PC_ATTRIBS, pc_augm=False, way_ratio=WAY_RATIO, way_num=WAY_NUM)
    classes = np.array([1, 2])  # floor, wall
    scan = ds.support_class2scans[1][0]
    np.random.seed(7)
    x_s, mask = sample_K_pointclouds(data, NUM_POINT, PC_ATTRIBS, False, None, [scan], 1, classes, is_support=True)
    np.random.seed(7)
    x_m, multi = sample_K_pointclouds(data, NUM_POINT, PC_ATTRIBS, False, None, [scan], 1, classes, is_support=False)
    assert np.array_equal(x_s, x_m) and np.array_equal(mask.astype(bool), multi == 1)
    item, multi = next(p4.draw_episodes(data, 1, seed=1, per_pair=1))
    assert np.array_equal(item[1], np.stack([(multi[k] == k + 1) for k in range(2)]).astype(np.int32))
    assert item[0].shape == (2, 1, NUM_POINT, 9) and item[2].shape == (2, NUM_POINT, 9)
