"""MMD-1..7 (05 §3.3, gate G1): the decoupled GMMN loss of Eq.7-8 / spec 02 §4.3-4.4.

float64 throughout; references are written with Python scalars and explicit loops, independent of
the vectorised code, and compared to 1e-12.
"""

import math

import pytest
import torch

from loss.gmmn_loss import RBF_BANDWIDTHS, gmmn_loss, mmd, pairwise_sq_dist, rbf_kernel

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
