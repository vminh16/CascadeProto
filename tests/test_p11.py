"""P11-1..20 (05 §3.8x): the blocks and the attribution of [DECISION D-48] (amendments 1-3), beyond the paper.
CPU, float64 unless noted.
"""

import itertools
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from experiments import attribution as att
from experiments import p11_precheck as p11
from models import base_learner as bl
from models import correlation as corr
from models import ot_assign as ota


def rand(*shape, seed=0):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)


def unit(*shape, seed=0):
    return F.normalize(rand(*shape, seed=seed), dim=-1)


# ------------------------------------------------------------------ [2] base learner

def test_p11_1_base_score_renormalises_over_kept_columns():
    z = rand(5, 7, seed=1)  # 6 base + none
    g = bl.base_score(z, exclude=[0, 2], eps=0.0)
    keep = [1, 3, 4, 5, 6]
    ref = torch.softmax(z[:, keep], dim=-1)[:, :4].max(dim=-1).values  # base columns 1, 3, 4, 5
    assert torch.allclose(g, ref)
    assert torch.allclose(bl.base_score(z, eps=0.0), torch.softmax(z, -1)[:, :6].max(-1).values)


def test_p11_2_base_score_clamp_and_errors():
    z = torch.tensor([[50.0, -50.0, -50.0]], dtype=torch.float64)  # base 0 certain
    assert float(bl.base_score(z)) == pytest.approx(1 - bl.G_EPS)
    z = torch.tensor([[-50.0, -50.0, 50.0]], dtype=torch.float64)  # none certain
    assert float(bl.base_score(z)) == pytest.approx(bl.G_EPS)
    with pytest.raises(ValueError):
        bl.base_score(z, exclude=[0, 1])  # every base column excluded
    with pytest.raises(ValueError):
        bl.base_score(z, exclude=[2])  # "none" is not a base column


def test_p11_3_exclusion_is_one_sided_and_monotone():
    g = torch.linspace(bl.G_EPS, 1 - bl.G_EPS, 50, dtype=torch.float64)
    term = bl.exclusion_term(g, 0.3)
    assert bool((term >= 0).all()) and bool((term[1:] > term[:-1]).all())
    logits = rand(2, 50, 3, seed=2)
    out = bl.apply_exclusion(logits, g.expand(2, -1), 0.3)
    assert torch.equal(out[..., 1:], logits[..., 1:])  # the ways are untouched
    assert bool((out[..., 0] >= logits[..., 0]).all())  # the background never goes down (fix F2)
    with pytest.raises(ValueError):
        bl.exclusion_term(g, -0.1)


def test_p11_4_base_learner_sends_no_gradient_to_the_features():
    model = bl.BaseLearner(8, 3).double()
    f = rand(10, 8, seed=3).requires_grad_(True)
    F.cross_entropy(model(f), torch.zeros(10, dtype=torch.long)).backward()
    assert f.grad is None  # fix F1
    assert all(p.grad is not None for p in model.parameters())


def test_p11_5_base_targets():
    raw = torch.tensor([0, 3, 12, 6, 8, 3])
    assert bl.base_targets(raw, [0, 3, 8]).tolist() == [0, 1, 3, 3, 2, 1]


# ------------------------------------------------------------------ OT

def test_p11_6_support_masses():
    y = torch.zeros(2, 1, 100, dtype=torch.int32)
    y[0, 0, :40], y[1, 0, :20] = 1, 1
    a = ota.support_masses(y)
    assert torch.allclose(a, torch.tensor([1 - 0.2 - 0.1, 0.2, 0.1]))


def test_p11_7_ot_vanishes_as_rho_goes_to_zero():
    logits = rand(2, 64, 3, seed=4) * 0.2
    a = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float64)
    out = ota.ot_logits(logits, a, eps=0.05, rho=1e-9)
    assert torch.allclose(out, logits, atol=1e-6)


def test_p11_8_ot_matches_row_masses_in_the_balanced_limit():
    logits = rand(2, 64, 3, seed=5) * 0.2
    a = torch.tensor([0.6, 0.3, 0.1], dtype=torch.float64)
    log_t, _ = ota.uot_log_plan(logits, a, eps=0.05, rho=1e6, iters=3000)
    rows = log_t.exp().sum(dim=(0, 1))
    assert torch.allclose(rows / rows.sum(), a, atol=2e-3)
    cols = log_t.exp().sum(dim=-1)
    assert torch.allclose(cols, torch.full_like(cols, 1.0 / 128), rtol=5e-3)


def test_p11_9_ot_decision_is_a_per_class_bias():
    logits = rand(2, 32, 3, seed=6) * 0.2
    a = torch.tensor([0.5, 0.25, 0.25], dtype=torch.float64)
    log_t, log_u = ota.uot_log_plan(logits, a, eps=0.1, rho=1.0)
    assert torch.equal(log_t.argmax(-1), (logits + 0.1 * log_u).argmax(-1))
    assert torch.equal(ota.ot_logits(logits, a, eps=0.1, rho=1.0).argmax(-1), log_t.argmax(-1))
    with pytest.raises(ValueError):
        ota.uot_log_plan(logits, a, eps=0.0, rho=1.0)
    with pytest.raises(ValueError):
        ota.uot_log_plan(logits, torch.tensor([1.0, 0.0, 0.0]), eps=0.1, rho=1.0)


# ------------------------------------------------------------------ [3]-[4] cells and descriptors

def test_p11_10_cell_count():
    assert corr.cell_count(187) == 5 and corr.cell_count(764) == 16 and corr.cell_count(40) == 2
    assert corr.cell_count(2048, cap=8) == 8
    with pytest.raises(ValueError):
        corr.cell_count(1)


def test_p11_11_cells_partition_and_are_deterministic():
    u = unit(200, 16, seed=7)
    c1, s1 = corr.cells(u, 5)
    c2, s2 = corr.cells(u, 5)
    assert torch.equal(c1, c2) and float(s1.sum()) == 200
    assert torch.allclose(c1.norm(dim=-1), torch.ones(5, dtype=torch.float64))
    with pytest.raises(ValueError):
        corr.cells(u[:3], 5)


def test_p11_12_descriptor_values_and_order_invariance():
    cos = torch.tensor([[0.1, 0.9, 0.5, 0.3]], dtype=torch.float64)
    assert torch.allclose(corr.descriptor(cos), torch.tensor([[0.9, 0.7, 0.45]], dtype=torch.float64))
    perm = cos[:, torch.tensor([2, 0, 3, 1])]
    assert torch.allclose(corr.descriptor(perm), corr.descriptor(cos))
    with pytest.raises(ValueError):
        corr.descriptor(cos[:, :1])


def test_p11_13_raw_descriptors_are_rotation_invariant():
    u_s = unit(2, 1, 120, 12, seed=8)
    y = torch.zeros(2, 1, 120, dtype=torch.long)
    y[0, 0, :50], y[1, 0, 60:100] = 1, 1
    u_q = unit(2, 30, 12, seed=9)
    q, _ = torch.linalg.qr(rand(12, 12, seed=10))
    d1 = corr.row_descriptors(u_q, corr.row_cells(u_s, y))
    d2 = corr.row_descriptors(u_q @ q, corr.row_cells(u_s @ q, y))
    assert d1.shape == (2, 30, 3, 3)
    assert torch.allclose(d1, d2, atol=1e-10)


def test_p11_14_spaces():
    x = rand(4000, 10, seed=11) * torch.tensor([3.0, 2.0, 1.5, 1.0, 0.5, 0.2, 0.1, 0.1, 0.1, 0.1], dtype=torch.float64)
    u = F.normalize(x + 0.5, dim=-1)
    mu, cov = corr.base_moments(u)
    assert torch.equal(corr.Space("raw")(u), u)
    cen = corr.Space("centred", mu)
    assert torch.allclose(cen(u), F.normalize(u - mu, dim=-1))
    white = corr.Space("white6", mu, cov)
    z = torch.einsum("md,dr->mr", u - mu, white.proj)  # before the normalisation
    c = z.T @ z / (z.shape[0] - 1)
    lam = torch.tensor(white.info["eigenvalues"], dtype=torch.float64)
    expected = torch.diag(lam / (lam + corr.WHITEN_DELTA * lam[0]))  # identity up to the regulariser delta
    assert torch.allclose(c, expected, atol=1e-8)
    proj = corr.Space("proj6", mu, cov)
    assert proj(u).shape == (4000, 6) and 0 < proj.info["energy_share"] <= 1
    with pytest.raises(ValueError):
        corr.Space("white6")
    with pytest.raises(ValueError):
        corr.Space("nope")


def test_p11_23_contrast_share():
    x = rand(3000, 10, seed=17) * torch.tensor([3.0, 2.0, 1.5, 1.0, 0.5, 0.2, 0.1, 0.1, 0.1, 0.1], dtype=torch.float64)
    mu, cov = corr.base_moments(x)
    proj = corr.Space("proj6", mu, cov)
    inside = proj.proj[:, :2].T  # [2, D], vectors in the kept span
    outside = torch.linalg.eigh(cov)[1][:, :2].T  # the two smallest directions
    zero = torch.zeros_like(inside)
    assert corr.contrast_share(proj, inside, zero) == pytest.approx(1.0)
    assert corr.contrast_share(proj, outside, zero) == pytest.approx(0.0, abs=1e-10)
    assert corr.contrast_share(corr.Space("raw"), outside, zero) == 1.0


def test_p11_15_probe_is_equivariant_to_the_ways():
    probe = corr.DescriptorProbe().double()
    d = rand(7, 3, 3, seed=12)
    out = probe(d)
    swapped = probe(d[:, [0, 2, 1]])
    assert torch.allclose(swapped, out[:, [0, 2, 1]])
    # training standardises the inputs from the data and learns a separable toy problem
    y = torch.randint(0, 3, (600,), generator=torch.Generator().manual_seed(3))
    desc = 0.5 + 0.01 * rand(600, 3, 3, seed=18)  # a narrow band, as raw cosines
    desc[torch.arange(600), y, 0] += 0.05  # the true row has the higher max
    trained = corr.train_probe(desc, y, epochs=60, batch=128)
    flat = desc.reshape(-1, 3)
    assert torch.allclose(trained.shift, flat.mean(0)) and torch.allclose(trained.scale, flat.std(0))
    assert float((trained(desc).argmax(-1) == y).double().mean()) > 0.9
    # standardised inputs: an affine change of the descriptors gives the same trained probe's outputs
    moved = corr.train_probe(desc * 10.0 + 3.0, y, epochs=5, batch=128)
    again = corr.train_probe(desc, y, epochs=5, batch=128)
    assert torch.allclose(moved(desc * 10.0 + 3.0), again(desc), atol=1e-8)


# ------------------------------------------------------------------ attribution

def brute_shapley(v, blocks):
    phi = {b: 0.0 for b in blocks}
    perms = list(itertools.permutations(blocks))
    for order in perms:
        s = frozenset()
        for b in order:
            phi[b] += v[s | {b}] - v[s]
            s = s | {b}
    return {b: x / len(perms) for b, x in phi.items()}


def test_p11_16_shapley_coefficients_match_permutations():
    blocks = ("a", "b", "c", "d")
    rng = np.random.default_rng(0)
    v = {c: float(rng.normal()) for c in att.combos(blocks)}
    lin = att.contrasts(blocks, [("a", "b")])
    ref = brute_shapley(v, blocks)
    for b in blocks:
        val = sum(coef * v[s] for s, coef in lin[f"shapley:{b}"].items())
        assert val == pytest.approx(ref[b])
    total = sum(sum(coef * v[s] for s, coef in lin[f"shapley:{b}"].items()) for b in blocks)
    assert total == pytest.approx(v[frozenset(blocks)] - v[frozenset()])  # efficiency
    inter = sum(coef * v[s] for s, coef in lin["inter:axb"].items())
    assert inter == pytest.approx(v[frozenset("ab")] - v[frozenset("a")] - v[frozenset("b")] + v[frozenset()])


def test_p11_17_attribute_on_counts():
    blocks = ("x", "y")
    rng = np.random.default_rng(1)
    base = rng.integers(50, 100, size=(40, 3, 3)).astype(np.float64)
    base[:, 2] = np.minimum(base[:, 2], np.minimum(base[:, 0], base[:, 1]))  # tp <= gt, pred
    counts = {c: base.copy() for c in att.combos(blocks)}
    counts[frozenset({"x"})][:, 2] = np.minimum(base[:, 2] + 5, np.minimum(base[:, 0], base[:, 1]))
    counts[frozenset({"x", "y"})] = counts[frozenset({"x"})].copy()
    out = att.attribute(counts, blocks, [("x", "y")], boot=200)
    assert out["shapley:y"]["value"] == pytest.approx(0.0)  # a dummy block
    assert out["shapley:x"]["value"] == pytest.approx(out["total"]["value"])
    assert out["shapley:x"]["ci_low"] <= out["shapley:x"]["value"] <= out["shapley:x"]["ci_high"]
    rands = [att.point_values(counts, blocks, [("x", "y")])] * 3
    assert att.kept(out, rands, "x", 0.0) == (out["shapley:x"]["ci_low"] > 0 and out["shapley:x"]["value"] > 0)
    assert not att.kept(out, rands, "y", 0.0)


# ------------------------------------------------------------------ P11 plumbing

def test_p11_18_combine_identity_and_order():
    l0 = rand(2, 20, 3, seed=13) * 0.2
    g = torch.full((2, 20), 0.5, dtype=torch.float64)
    t = rand(2, 20, 2, seed=14)
    gamma = torch.tensor(0.5, dtype=torch.float64)
    masses = torch.tensor([0.5, 0.25, 0.25], dtype=torch.float64)
    zero = {"psi": 0.0, "kappa": 0.0, "eps": 0.1, "rho": 1e-9}
    out = p11.combine(l0, {"excl", "text", "ot"}, g, t, gamma, masses, zero)
    assert torch.allclose(out, l0, atol=1e-6)
    only = p11.combine(l0, {"excl"}, g, t, gamma, masses, {"psi": 1.0, "kappa": 9.0, "eps": 0.1, "rho": 1.0})
    assert torch.allclose(only[..., 0], l0[..., 0] + math.log(2.0))
    assert torch.equal(only[..., 1:], l0[..., 1:])


def test_p11_19_coordinate_ascent():
    score = lambda p: -(p["a"] - 2) ** 2 - (p["b"] - 1) ** 2  # noqa: E731
    params, trace = p11.coordinate_ascent(score, {"a": [0, 1, 2, 3], "b": [0, 1, 2]}, {"a": 0, "b": 0})
    assert params == {"a": 2, "b": 1} and len(trace) == 2 * 7
    tie = lambda p: 0.0  # noqa: E731
    assert p11.coordinate_ascent(tie, {"a": [5, 6]}, {"a": 6})[0] == {"a": 5}  # ties keep the earlier value


def test_p11_20_spread_many_equals_single_spreads():
    from experiments import p7_propagation_probe as p7

    xyz = rand(2, 40, 3, seed=15)
    u = unit(2, 40, 8, seed=16)
    seeds = [torch.randint(0, 3, (2, 40), generator=torch.Generator().manual_seed(s)) for s in (1, 2)]
    arm = {"graph": "xyzf", "k": 8, "beta": 0.9}
    many = p11.spread_many(xyz, u, seeds, 3, arm)
    idx, a = p7.knn_graph(xyz, u, "xyzf", 8)
    s = p7.normalized(p7.affinity(idx, a, 40))
    for sd, got in zip(seeds, many):
        ref = p7.spread(s, F.one_hot(sd, 3).double(), 0.9).argmax(-1)
        assert torch.equal(got, ref)


def test_p11_22_kept_rule():
    fixed = {"shapley:x": {"value": 0.8, "ci_low": 0.1, "ci_high": 1.5}}
    assert att.kept(fixed, [{"shapley:x": 0.3}] * 3, "x", 0.5)
    assert not att.kept(fixed, [{"shapley:x": 0.3}] * 3, "x", 1.0)  # below the bar
    assert not att.kept({"shapley:x": {"value": 0.8, "ci_low": -0.1, "ci_high": 1.5}}, [{"shapley:x": 0.3}] * 3, "x", 0.5)
    assert not att.kept(fixed, [{"shapley:x": 0.3}, {"shapley:x": -0.1}, {"shapley:x": 0.3}], "x", 0.5)


def test_p11_21_raw_readings_kinds():
    r = p11.RawReadings(base_classes=[0, 3], test_classes=[1, 2, 5, 6])
    gt = np.array([[1, 2, 0, 0, 0], [2, 1, 0, 0, 0]])
    raw = np.array([[1, 2, 0, 12, 6], [2, 1, 3, 5, 12]])
    k = r.kinds(gt, raw)
    assert k.tolist() == [[0, 1, 2, 3, 4], [0, 1, 2, 4, 3]]
