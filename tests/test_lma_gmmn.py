"""LMA-1..3 and MMD-1..7 (05 §3.3, gate G1): adapter and generator of Eq.4-6, GMMN loss of Eq.7-8.

float64 throughout; references are written with explicit tensor operations or Python scalars and
explicit loops, independent of the modules under test, and compared to 1e-12.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from loss.gmmn_loss import RBF_BANDWIDTHS, gmmn_loss, mmd, pairwise_sq_dist, rbf_kernel
from models.lma import LearnableModalityAdapter

ATOL = 1e-12
D = 128


def rows(m, seed, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return scale * torch.randn(m, D, generator=g, dtype=torch.float64)  # [m, D]


def k_ref(x, y):
    """Eq.7 kernel for two vectors, from Python floats."""
    d2 = sum((a - b) ** 2 for a, b in zip(x.tolist(), y.tolist()))
    return sum(math.exp(-d2 / (2.0 * s * s)) for s in (2, 5, 10, 20, 40, 80))


def mmd_ref(x, y):
    """Expansion of Eq.7 with explicit sums over all pairs, diagonal included (02 §4.3)."""
    m, l = len(x), len(y)
    xx = sum(k_ref(x[i], x[j]) for i in range(m) for j in range(m)) / m ** 2
    yy = sum(k_ref(y[i], y[j]) for i in range(l) for j in range(l)) / l ** 2
    xy = sum(k_ref(x[i], y[j]) for i in range(m) for j in range(l)) / (m * l)
    return xx + yy - 2.0 * xy


def test_bandwidths_are_the_papers():
    assert RBF_BANDWIDTHS == (2.0, 5.0, 10.0, 20.0, 40.0, 80.0)


def test_pairwise_sq_dist_matches_scalar_reference():
    x, y = rows(4, 0, 3.0), rows(3, 1, 3.0)
    d2 = pairwise_sq_dist(x, y)
    for i in range(4):
        for j in range(3):
            ref = sum((a - b) ** 2 for a, b in zip(x[i].tolist(), y[j].tolist()))
            assert math.isclose(d2[i, j].item(), ref, rel_tol=1e-14, abs_tol=0)
    big = rows(5, 2, 50.0)
    assert (pairwise_sq_dist(big, big).diagonal() == 0).all()  # exact at any norm (02 §9)


@pytest.mark.parametrize("scale", [0.1, 1.0, 50.0])
def test_mmd1_kernel_diagonal_is_six(scale):
    x = rows(5, 2, scale)
    assert torch.allclose(rbf_kernel(x, x).diagonal(), torch.full((5,), 6.0, dtype=torch.float64),
                          atol=ATOL, rtol=0)


def test_mmd1_kernel_matches_scalar_reference():
    x, y = rows(3, 3, 4.0), rows(2, 4, 4.0)
    k = rbf_kernel(x, y)
    for i in range(3):
        for j in range(2):
            assert math.isclose(k[i, j].item(), k_ref(x[i], y[j]), rel_tol=0, abs_tol=ATOL)
    assert (k > 0).all() and (k <= 6.0).all()


@pytest.mark.parametrize("scale", [0.5, 5.0, 30.0])
def test_mmd2_single_samples(scale):
    x, y = rows(1, 5, scale), rows(1, 6, scale)
    expected = 2.0 * (6.0 - k_ref(x[0], y[0]))
    assert math.isclose(mmd(x, y).item(), expected, rel_tol=0, abs_tol=ATOL)
    assert 0.0 <= mmd(x, y).item() < 12.0


def test_mmd3_identity_symmetry_non_negativity():
    x = rows(3, 7, 3.0)
    assert abs(mmd(x, x).item()) < ATOL
    for seed in range(20):
        a, b = rows(3, 100 + seed, 3.0), rows(2, 200 + seed, 3.0)
        assert mmd(a, b).item() >= -ATOL
        assert math.isclose(mmd(a, b).item(), mmd(b, a).item(), rel_tol=0, abs_tol=ATOL)


@pytest.mark.parametrize("m,l", [(1, 1), (2, 2), (3, 3), (2, 3)])
def test_mmd4_equals_the_explicit_formula(m, l):
    """Squared, biased (diagonal included), no square root, no clamp."""
    x, y = rows(m, 10 + m, 3.0), rows(l, 20 + l, 3.0)
    assert math.isclose(mmd(x, y).item(), mmd_ref(x, y), rel_tol=0, abs_tol=ATOL)


def prototypes(n, seed, scale=3.0):
    return rows(n + 1, seed, scale), rows(n + 1, seed + 1, scale)  # p_modal, p_point [N+1, D]


@pytest.mark.parametrize("n", [1, 2, 3])
def test_mmd5_foreground_rows_form_one_set(n):
    p_modal, p_point = prototypes(n, 30)
    fg = gmmn_loss(p_modal, p_point) - 0.1 * mmd_ref(p_modal[:1], p_point[:1])
    assert math.isclose(fg.item(), mmd_ref(p_modal[1:], p_point[1:]), rel_tol=0, abs_tol=ATOL)


def test_mmd5_joint_differs_from_per_class_and_ignores_row_pairing():
    p_modal, p_point = prototypes(3, 40)
    joint, per_class = gmmn_loss(p_modal, p_point), gmmn_loss(p_modal, p_point, fg_mode="per_class")
    per_class_ref = 0.1 * mmd_ref(p_modal[:1], p_point[:1]) + sum(
        mmd_ref(p_modal[c:c + 1], p_point[c:c + 1]) for c in (1, 2, 3)) / 3
    assert math.isclose(per_class.item(), per_class_ref, rel_tol=0, abs_tol=ATOL)
    assert abs(joint.item() - per_class.item()) > 1e-3
    shuffled = p_modal[[0, 3, 1, 2]]  # foreground rows reordered, background kept
    assert math.isclose(gmmn_loss(shuffled, p_point).item(), joint.item(), rel_tol=0, abs_tol=ATOL)
    assert abs(gmmn_loss(shuffled, p_point, fg_mode="per_class").item() - per_class.item()) > 1e-3


def test_mmd6_weights_point_one_and_one():
    p_modal, p_point = prototypes(2, 50)
    only_bg = torch.cat([p_modal[:1], p_point[1:]])  # foreground identical: fg term exactly 0
    assert math.isclose(gmmn_loss(only_bg, p_point).item(), 0.1 * mmd_ref(p_modal[:1], p_point[:1]),
                        rel_tol=0, abs_tol=ATOL)
    only_fg = torch.cat([p_point[:1], p_modal[1:]])  # background identical: bg term exactly 0
    assert math.isclose(gmmn_loss(only_fg, p_point).item(), 1.0 * mmd_ref(p_modal[1:], p_point[1:]),
                        rel_tol=0, abs_tol=ATOL)
    assert abs(gmmn_loss(p_point, p_point).item()) < ATOL


def test_mmd7_gradients_reach_both_inputs_and_match_analytic_background():
    p_modal, p_point = prototypes(2, 60)
    p_modal.requires_grad_(True)
    p_point.requires_grad_(True)
    gmmn_loss(p_modal, p_point).backward()
    assert p_modal.grad.abs().sum() > 0 and p_point.grad.abs().sum() > 0
    # d/dx of 0.1 * 2 (6 - k(x, y)) = 0.2 * sum_s exp(-d2 / 2s^2) (x - y) / s^2
    x, y = p_modal[0].detach(), p_point[0].detach()
    d2 = ((x - y) ** 2).sum()
    grad_bg = 0.2 * sum(torch.exp(-d2 / (2 * s * s)) / (s * s) for s in RBF_BANDWIDTHS) * (x - y)
    assert torch.allclose(p_modal.grad[0], grad_bg, atol=ATOL, rtol=0)
    assert torch.allclose(p_point.grad[0], -grad_bg, atol=ATOL, rtol=0)


def test_mmd7_gradcheck():
    p_modal, p_point = prototypes(2, 70, scale=2.0)
    p_modal = p_modal[:, :8].clone().requires_grad_(True)
    p_point = p_point[:, :8].clone().requires_grad_(True)
    assert torch.autograd.gradcheck(gmmn_loss, (p_modal, p_point), eps=1e-6, atol=1e-8)


def test_mmd7_detach_point_flag():
    p_modal, p_point = prototypes(2, 80)
    p_modal.requires_grad_(True)
    p_point.requires_grad_(True)
    loss = gmmn_loss(p_modal, p_point, detach_point=True)
    assert math.isclose(loss.item(), gmmn_loss(p_modal, p_point).item(), rel_tol=0, abs_tol=ATOL)
    loss.backward()
    assert p_point.grad is None and p_modal.grad.abs().sum() > 0


@pytest.mark.parametrize("a,b,kwargs", [((3, D), (4, D), {}), ((1, D), (1, D), {}), ((2, 3, D), (2, 3, D), {}),
                                        ((3, D), (3, D), {"fg_mode": "mean"})])
def test_invalid_inputs_raise(a, b, kwargs):
    with pytest.raises(ValueError):
        gmmn_loss(torch.zeros(a, dtype=torch.float64), torch.zeros(b, dtype=torch.float64), **kwargs)


# ---------------------------------------------------------------- LMA-1..3 (02 §4.1-4.2, 03 §3)

def lma(seed=0, **kwargs):
    torch.manual_seed(seed)
    return LearnableModalityAdapter(**kwargs).double()


def clip_rows(n, seed=90):
    """Stand-in for E_CLIP: fixed random L2-normalised rows (05 §1 principle 3)."""
    g = torch.Generator().manual_seed(seed)
    return F.normalize(torch.randn(n + 1, 512, generator=g, dtype=torch.float64), dim=-1)  # [N+1, 512]


def reference(m, e, z, dropout_mask=None):
    """Eq.5-6 written out with the module's weights: fc1, LayerNorm (biased variance, eps 1e-5),
    ReLU, Dropout (identity in eval), fc2; then [E; z] through Linear-ReLU-Linear-ReLU-Linear."""
    a = m.adapter
    h = e @ a.fc1.weight.T + a.fc1.bias  # [N+1, 128]
    mu = h.mean(-1, keepdim=True)
    var = ((h - mu) ** 2).mean(-1, keepdim=True)
    h = (h - mu) / torch.sqrt(var + 1e-5) * a.norm.weight + a.norm.bias
    h = torch.clamp(h, min=0.0)
    if dropout_mask is not None:
        h = h * dropout_mask
    e_adapted = h @ a.fc2.weight.T + a.fc2.bias  # [N+1, 128]
    g = m.generator.net
    x = torch.cat([e_adapted, z], dim=-1)  # [N+1, 256]
    x = torch.clamp(x @ g[0].weight.T + g[0].bias, min=0.0)
    x = torch.clamp(x @ g[2].weight.T + g[2].bias, min=0.0)
    return x @ g[4].weight.T + g[4].bias  # [N+1, 128]


def test_lma1_shapes_and_parameter_counts():
    m = lma()
    count = lambda mod: sum(p.numel() for p in mod.parameters())
    assert count(m.adapter) == 82_432 and count(m.generator) == 65_920 and count(m) == 148_352
    assert m.generator.net[0].in_features == 256 and m.adapter.norm.eps == 1e-5
    for n in (1, 2, 3):
        assert m.eval()(clip_rows(n)).shape == (n + 1, 128)
    assert m.adapter.dropout.p == 0.1


@pytest.mark.parametrize("n", [2, 3])
def test_lma1_forward_matches_written_out_eq5_eq6(n):
    m, e = lma().eval(), clip_rows(n)
    z = torch.randn(n + 1, 128, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    assert torch.allclose(m(e, z=z), reference(m, e, z), atol=ATOL, rtol=0)


def test_lma2_eval_uses_zero_noise():
    m, e = lma().eval(), clip_rows(2)
    first, second = m(e), m(e)
    assert torch.equal(first, second)
    assert torch.allclose(first, reference(m, e, torch.zeros(3, 128, dtype=torch.float64)), atol=ATOL, rtol=0)


def test_lma3_training_draws_dropout_then_fresh_standard_normal_noise():
    """Exact replay of the RNG stream: dropout mask on the adapter's hidden layer (p = 0.1, after the
    ReLU), then z ~ N(0, I) of shape [N+1, 128]."""
    m, e = lma().train(), clip_rows(2)
    torch.manual_seed(5)
    out = m(e)
    torch.manual_seed(5)
    mask = F.dropout(torch.ones(3, 128, dtype=torch.float64), p=0.1, training=True)  # values 0 or 1/0.9
    z = torch.randn(3, 128, dtype=torch.float64)
    assert torch.allclose(out, reference(m, e, z, dropout_mask=mask), atol=ATOL, rtol=0)
    assert not torch.equal(m(e), m(e))


def test_lma3_noise_is_standard_normal():
    m = lma().train()
    m.adapter.dropout.p = 0.0
    seen = []
    m.generator.register_forward_pre_hook(lambda mod, args: seen.append(args[0][:, 128:].clone()))
    for _ in range(200):
        m(clip_rows(3))
    z = torch.stack(seen)  # [200, 4, 128]
    assert abs(z.mean().item()) < 0.02 and abs(z.std().item() - 1.0) < 0.02
    assert not torch.equal(z[0], z[1])


def test_lma_eval_noise_flag():
    e = clip_rows(2)
    sample = lma(eval_noise="sample").eval()
    assert not torch.equal(sample(e), sample(e))
    with pytest.raises(NotImplementedError, match="mean_of_M"):
        lma(eval_noise="mean_of_M")
    with pytest.raises(ValueError):
        lma(eval_noise="mean")


def test_lma_gradients_reach_every_parameter_through_gmmn():
    m, e = lma().train(), clip_rows(2)
    p_point = rows(3, 95).requires_grad_(True)
    gmmn_loss(m(e), p_point).backward()
    for name, p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, name
    assert p_point.grad.abs().sum() > 0


@pytest.mark.parametrize("e,z", [((3, 256), None), ((2, 3, 512), None), ((3, 512), (3, 256))])
def test_lma_invalid_inputs_raise(e, z):
    m = lma().eval()
    with pytest.raises(ValueError):
        m(torch.zeros(e, dtype=torch.float64), z=None if z is None else torch.zeros(z, dtype=torch.float64))
