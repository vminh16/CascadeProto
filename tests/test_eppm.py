"""GATE, XATT, DIFF, FUSE, STAGE (05 §3.4, gate G1): one EPPM stage of §3.4 / spec 02 §5.

float64 throughout; references are written with Python scalars or explicit loops, independent of the
vectorised code, and compared to 1e-12 unless stated.
"""

import math

import pytest
import torch

from models.eppm import EntropyGate, channel_entropy

ATOL = 1e-12
LN2 = math.log(2.0)


def rand(*shape, seed=0, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return scale * torch.randn(*shape, generator=g, dtype=torch.float64)


def entropy_ref(x):
    """Eq.10 for one Python float, with the D-16 clamp."""
    p = min(max(1.0 / (1.0 + math.exp(-x)), 1e-7), 1.0 - 1e-7)
    return -p * math.log(p + 1e-8) - (1.0 - p) * math.log(1.0 - p + 1e-8)


def gate(enabled=True):
    return EntropyGate(enabled).double()


# ------------------------------------------------------------------ GATE (02 §5.1, Eq.10-12)

def test_gate1_entropy_is_eq10_and_bounded():
    x = torch.linspace(-1000.0, 1000.0, 20001, dtype=torch.float64)
    h = channel_entropy(x)
    assert torch.isfinite(h).all() and h.min() >= 0.0 and h.max() <= LN2
    for v in (-30.0, -16.5, -3.0, -0.25, 0.0, 0.7, 5.0, 16.5, 40.0):
        assert math.isclose(channel_entropy(torch.tensor(v, dtype=torch.float64)).item(), entropy_ref(v),
                            rel_tol=0, abs_tol=ATOL)
    assert torch.allclose(channel_entropy(x), channel_entropy(-x), atol=ATOL, rtol=0)  # H(p) = H(1 - p)


def test_gate1_entropy_is_maximal_at_zero_and_decreasing_in_abs_x():
    x = torch.linspace(0.0, 15.0, 301, dtype=torch.float64)
    h = channel_entropy(x)
    assert (h[1:] < h[:-1]).all()
    assert math.isclose(h[0].item(), -math.log(0.5 + 1e-8), rel_tol=0, abs_tol=ATOL)  # ~ ln 2


def test_gate2_gate_values_at_theta_half():
    m = gate()
    p = torch.cat([torch.zeros(1), torch.linspace(-60, 60, 1001)]).double().reshape(1, 1, -1)
    g = m.gate(p)
    lo, hi = 1 / (1 + math.exp(-2 * (0.5 + math.log(0.5 + 1e-8)))), 1 / (1 + math.exp(-1.0 * 2 * 0.5))
    assert math.isclose(g[0, 0, 0].item(), lo, rel_tol=0, abs_tol=ATOL)  # x = 0: H = ln 2
    assert abs(lo - 0.4046) < 1e-4 and abs(hi - 0.7311) < 1e-4
    assert g.min().item() >= lo - ATOL and g.max().item() <= hi + 1e-6


def test_gate2_forward_is_p_times_sigmoid_two_theta_minus_h():
    m = gate()
    with torch.no_grad():
        m.theta.fill_(0.3)
    p = rand(2, 3, 128, seed=1, scale=3.0)
    out = m(p)
    for b, c, i in [(0, 0, 0), (1, 2, 127), (0, 1, 64), (1, 0, 5)]:
        x = p[b, c, i].item()
        g = 1.0 / (1.0 + math.exp(-2.0 * (0.3 - entropy_ref(x))))
        assert math.isclose(out[b, c, i].item(), x * g, rel_tol=0, abs_tol=ATOL)


def test_gate2_gate_is_channel_wise():
    """Changing one entry of P changes only that entry's gate (Eq.10-12 have no mixing)."""
    m = gate()
    p = rand(2, 3, 128, seed=2)
    q = p.clone()
    q[1, 2, 40] += 1.5
    diff = (m.gate(q) - m.gate(p)).abs()
    assert diff[1, 2, 40] > 1e-3
    diff[1, 2, 40] = 0
    assert diff.max() == 0


def test_gate3_theta_is_a_learnable_scalar_initialised_at_half():
    m = gate()
    assert isinstance(m.theta, torch.nn.Parameter) and m.theta.shape == () and m.theta.item() == 0.5
    assert [n for n, _ in m.named_parameters()] == ["theta"]
    p = rand(2, 3, 128, seed=3)
    m(p).sum().backward()
    g = m.gate(p).detach()
    expected = (p * g * (1 - g) * 2.0).sum()  # d/dθ of Σ p·σ(2(θ - H))
    assert math.isclose(m.theta.grad.item(), expected.item(), rel_tol=1e-12, abs_tol=ATOL)


def test_gate3_gradcheck():
    m = gate()
    p = rand(1, 3, 8, seed=4, scale=2.0).requires_grad_(True)
    assert torch.autograd.gradcheck(lambda x: m(x), (p,), eps=1e-6, atol=1e-8)


def test_gate_disabled_is_identity_without_parameters():
    m = gate(enabled=False)
    p = rand(2, 3, 128, seed=5)
    assert torch.equal(m(p), p) and list(m.parameters()) == []
