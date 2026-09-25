"""P6-1..9 (05 §3.8p): the prototype probe of [DECISION D-38], beyond the paper. CPU, float64, synthetic features."""

import ast
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from experiments import p6_prototype_probe as p6
from experiments.p5_condition_probe import support_directions

REPO = Path(__file__).resolve().parents[1]
ATOL = 1e-12


def features(bq=2, p=256, d=8, n=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    f_q = torch.rand(bq, p, d, generator=g, dtype=torch.float64)
    f_s = torch.rand(n, 1, p, d, generator=g, dtype=torch.float64)
    support_y = (torch.rand(n, 1, p, generator=g) < 0.3).long()
    labels = torch.randint(0, n + 1, (bq, p), generator=g)
    return f_q, f_s, support_y, labels


def test_p6_1_base_rule_is_the_support_directions_for_every_query():
    f_q, f_s, y, _ = features()
    rows = p6.base_rows(f_q, f_s, y)
    assert rows.shape == (2, 3, 8)
    assert torch.allclose(rows[0], support_directions(f_s, y), atol=ATOL) and torch.equal(rows[0], rows[1])
    assert torch.allclose(rows.norm(dim=-1), torch.ones(2, 3, dtype=torch.float64), atol=ATOL)
    assert torch.allclose(p6.rule_logits(f_q, rows)[1, 5], f_q[1, 5] @ rows[1].T, atol=ATOL)


@pytest.mark.parametrize("which,changed", [("bg", [0]), ("fg", [1, 2]), ("all", [0, 1, 2])])
def test_p6_2_oracle_rows_replace_present_rows_of_the_set_only(which, changed):
    f_q, f_s, y, labels = features()
    labels[1][labels[1] == 2] = 1  # class 2 absent from query block 1
    rows = p6.base_rows(f_q, f_s, y)
    out = p6.oracle_rows(f_q, labels, rows, which)
    for b in range(2):
        for c in range(3):
            present = bool((labels[b] == c).any())
            if c in changed and present:
                ref = F.normalize(F.normalize(f_q[b][labels[b] == c], dim=-1).sum(0), dim=-1)
                assert torch.allclose(out[b, c], ref, atol=ATOL)
            else:
                assert torch.equal(out[b, c], rows[b, c])
    assert torch.allclose(out.norm(dim=-1), torch.ones(2, 3, dtype=torch.float64), atol=ATOL)  # one gauge
    with pytest.raises(ValueError, match="row set"):
        p6.row_set("rows", 3, "cpu")


def test_p6_3_self_support_by_hand():
    """Two classes, four points; rho = 0.5 keeps the two most confident points predicted as class 1."""
    rows = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float64)  # [1, 2, 2]
    f_q = torch.tensor([[[3.0, 0.1], [0.1, 3.0], [0.2, 2.0], [0.9, 1.0]]], dtype=torch.float64)
    out = p6.self_support(f_q, rows, rho=0.5, alpha=0.5, steps=1, which="fg", min_points=1)
    h = p6.entropy(p6.rule_logits(f_q, rows))[0]
    assert h[1] < h[2] < h[3]  # points 1-3 are predicted 1; the rho = 0.5 quantile of three is the middle one
    sel = [1, 2]
    s = F.normalize(F.normalize(f_q[0, sel], dim=-1).sum(0), dim=-1)
    assert torch.allclose(out[0, 1], F.normalize(0.5 * rows[0, 1] + 0.5 * s, dim=-1), atol=ATOL)
    assert torch.equal(out[0, 0], rows[0, 0])  # background not in the set


def test_p6_4_self_support_properties():
    f_q, f_s, y, _ = features(bq=3)
    rows = p6.base_rows(f_q, f_s, y)
    assert torch.allclose(p6.self_support(f_q, rows, 0.5, 1.0, 2, "all"), rows, atol=ATOL)  # alpha = 1: unchanged
    assert torch.equal(p6.self_support(f_q, rows, 0.5, 0.5, 1, "all", min_points=10 ** 6), rows)  # too few points
    one = p6.self_support(f_q, rows, 0.5, 0.5, 1, "all")
    assert torch.allclose(p6.self_support(f_q, rows, 0.5, 0.5, 2, "all"),
                          p6.self_support(f_q, one, 0.5, 0.5, 1, "all"), atol=ATOL)  # T = 2 is T = 1 twice
    perm = torch.tensor([2, 0, 1])
    two = p6.self_support(f_q, rows, 0.5, 0.5, 2, "all")
    assert torch.allclose(p6.self_support(f_q[perm], rows[perm], 0.5, 0.5, 2, "all"), two[perm], atol=ATOL)  # order-free
    changed = p6.self_support(f_q, rows, 0.5, 0.5, 1, "bg")
    assert torch.equal(changed[:, 1:], rows[:, 1:]) and not torch.equal(changed[:, 0], rows[:, 0])
    assert torch.allclose(one.norm(dim=-1), torch.ones(3, 3, dtype=torch.float64), atol=ATOL)


def test_p6_5_spherical_kmeans_and_multi_background():
    g = torch.Generator().manual_seed(1)
    centres = F.normalize(torch.randn(3, 8, generator=g, dtype=torch.float64), dim=-1)
    x = F.normalize(centres.repeat_interleave(50, 0) + 0.01 * torch.randn(150, 8, generator=g, dtype=torch.float64), dim=-1)
    c = p6.spherical_kmeans(x, 3)
    assert torch.equal(c, p6.spherical_kmeans(x, 3))  # deterministic
    assert (c @ centres.T).max(dim=0).values.min() > 0.999  # recovers the three modes
    one = p6.spherical_kmeans(x, 1)
    assert torch.allclose(one[0], F.normalize(x.sum(0), dim=-1), atol=ATOL)  # k = 1 is the unit-feature mean
    with pytest.raises(ValueError):
        p6.spherical_kmeans(x[:2], 3)
    f_q, f_s, y, _ = features()
    rows = p6.base_rows(f_q, f_s, y)
    bgu = p6.background_units(f_s, y)
    assert bgu.shape[0] == int((y == 0).sum())
    assert torch.allclose(p6.multi_bg_logits(f_q, rows, p6.spherical_kmeans(bgu, 1)),
                          p6.rule_logits(f_q, rows), atol=1e-10)  # one centroid = U's background row
    lk = p6.multi_bg_logits(f_q, rows, c[:, :8])
    assert torch.equal(lk[..., 1:], p6.rule_logits(f_q, rows)[..., 1:])
    assert torch.allclose(lk[..., 0], (f_q @ c.T).max(dim=-1).values, atol=ATOL)


def test_p6_6_arm_predictions_and_grid():
    arms = p6.ssp_arms()
    assert len(arms) == 54 and len({p6.ssp_name(a) for a in arms}) == 54
    f_q, f_s, y, labels = features()
    names = ["U", "oracle_all", p6.ssp_name(arms[0]), "km3"]
    out = p6.arm_predictions(f_q, f_s, y, names, labels)
    assert set(out) == set(names) and all(v.shape == (2, 256) for v in out.values())
    with pytest.raises(ValueError, match="labels"):
        p6.arm_predictions(f_q, f_s, y, ["oracle_bg"])
    with pytest.raises(ValueError, match="unknown"):
        p6.arm_predictions(f_q, f_s, y, ["nope"])


def test_p6_7_selection_and_location():
    miou = {"U": 0.50, "a": 0.52, "b": 0.53, "c": 0.53}
    assert p6.select(miou, ["a", "b", "c"]) == ("b", pytest.approx(3.0))  # tie goes to the earlier arm
    assert p6.location({"bg": 1.0, "fg": 20.0, "all": 25.0}) == "foreground"
    assert p6.location({"bg": 20.0, "fg": 1.0, "all": 25.0}) == "background"
    assert p6.location({"bg": 10.0, "fg": 10.0, "all": 25.0}) == "joint"
    assert p6.location({"bg": 20.0, "fg": 20.0, "all": 25.0}) == "either row suffices"


def fake(miou_by_arm, episodes=60):
    def counts(tp, seed):
        rng = np.random.default_rng(seed)
        c = np.zeros((episodes, 3, 3))
        c[:, 0, :] = c[:, 1, :] = 100.0
        c[:, 2, 1:] = np.clip(tp + rng.normal(0, 2, (episodes, 2)), 0, 100)
        c[:, 2, 0] = 90.0
        return c

    draws = {}
    for i, d in enumerate(p6.TEST_DRAWS):
        cnt = {n: counts(tp[i] if isinstance(tp, list) else tp, i) for n, tp in miou_by_arm.items()}
        draws[d] = ({"miou": {n: float(p6.p0.miou_from_counts(v.sum(0))) for n, v in cnt.items()}}, cnt)
    return draws


def test_p6_8_rules():
    base = {"model": 70, "support_rule": 70, "U": 70, "oracle_bg": 72, "oracle_fg": 90, "oracle_all": 92}
    frozen = {"ssp": "s", "km": "k"}
    v = dict(p6.decide(fake(dict(base, s=75, k=70.2)), frozen))
    assert v["P6.1 location fixed100"].startswith("foreground")
    assert "P6.2 self-support go" in v and "P6.3 multi-background stop" in v and "P6.4 stop" not in v
    v = dict(p6.decide(fake(dict(base, s=70.1, k=70.1)), frozen))
    assert "P6.4 stop" in v
    v = dict(p6.decide(fake(dict(base, s=70.8, k=70.1)), frozen))
    assert "P6.2 self-support between" in v
    v = dict(p6.decide(fake(dict(base, s=[75, 75, 69, 75, 75], k=70.1)), frozen))
    assert "P6.2 self-support between" in v  # a negative random600 draw blocks the go
    assert p6.decide({}, frozen)[0][0] == "incomplete"


def test_p6_9_files_parse_as_python_310():
    for f in ("experiments/p6_prototype_probe.py", "tests/test_prototype_probe.py"):
        ast.parse((REPO / f).read_text(encoding="utf-8"), feature_version=(3, 10))
