"""GATE, XATT, DIFF, FUSE, STAGE (05 §3.4, gate G1): one EPPM stage of §3.4 / spec 02 §5.

float64 throughout; references are written with Python scalars or explicit loops, independent of the
vectorised code, and compared to 1e-12 unless stated.
"""

import math

import pytest
import torch

from models.eppm import CrossAttention, EntropyGate, channel_entropy, pool_tokens

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


# ------------------------------------------------------------------ XATT (02 §5.2, Eq.13-14, D-01)

N, K, BQ, P, D = 2, 2, 2, 2048, 128


def features(n=N, k=K, bq=BQ, seed=10):
    """Non-negative features like the VIP-Seg head's ReLU output (02 §2)."""
    f_s = rand(n, k, P, D, seed=seed).abs()  # [N, K, 2048, D]
    f_q = rand(bq, P, D, seed=seed + 1).abs()  # [B_q, 2048, D]
    return f_s, f_q


def xatt(scale="sqrt_d", seed=0):
    torch.manual_seed(seed)
    return CrossAttention(scale=scale).double()


def vipseg_pool(x):
    """VIP-Seg's pooling: MaxPool1d(32, stride=32) on the transposed [.., D, 2048] tensor [VIPSEG models/vipseg.py:247-249]."""
    return torch.nn.MaxPool1d(32, stride=32)(x.transpose(-1, -2)).transpose(-1, -2)


def attention_ref(m, f_s, f_q, scale=72):
    """A[b, c, k] from an explicit loop: slot 0 = way-mean, then pool, φ as a matrix, row softmax by hand."""
    w = m.phi.weight[:, :, 0]  # [72, 64]
    n, k = f_s.shape[:2]
    slots = [sum(f_s[i] for i in range(n)) / n] + [f_s[i] for i in range(n)]  # N+1 tensors [K, 2048, D]
    out = torch.empty(f_q.shape[0], n + 1, k, D, D, dtype=torch.float64)
    for b in range(f_q.shape[0]):
        q = w @ vipseg_pool(f_q[b])  # [72, D]
        for c in range(n + 1):
            for j in range(k):
                s = w @ vipseg_pool(slots[c][j])  # [72, D]
                logits = (q.T @ s) / math.sqrt(scale)  # [D, D]
                e = torch.exp(logits - logits.max(dim=1, keepdim=True).values)
                out[b, c, j] = e / e.sum(dim=1, keepdim=True)
    return out


def p_cross_ref(m, a, p_gated):
    """P_cross[b, c] = (1/K) Σ_k A[b, c, k] ψ(P_gated[b, c]), ψ written out."""
    v = p_gated @ m.psi.weight.T + m.psi.bias  # [B_q, N+1, D]
    bq, c1, k = a.shape[:3]
    out = torch.zeros(bq, c1, D, dtype=torch.float64)
    for b in range(bq):
        for c in range(c1):
            for j in range(k):
                out[b, c] += a[b, c, j] @ v[b, c] / k
    return out


def test_pool_tokens_matches_vipseg_maxpool():
    f = rand(3, P, D, seed=20)
    assert torch.equal(pool_tokens(f), vipseg_pool(f))
    assert pool_tokens(f).shape == (3, 64, D)
    with pytest.raises(ValueError):
        pool_tokens(rand(3, 1024, D))


def test_xatt1_shape_and_row_sums():
    m, (f_s, f_q) = xatt(), features()
    a = m.attention(f_s, f_q)
    assert a.shape == (BQ, N + 1, K, D, D)
    assert torch.allclose(a.sum(-1), torch.ones(BQ, N + 1, K, D, dtype=torch.float64), atol=ATOL, rtol=0)
    assert (a > 0).all()


@pytest.mark.parametrize("scale,value", [("sqrt_d", 72), ("sqrt_D", 128)])
def test_xatt2_attention_equals_explicit_loop(scale, value):
    m, (f_s, f_q) = xatt(scale), features()
    assert torch.allclose(m.attention(f_s, f_q), attention_ref(m, f_s, f_q, value), atol=ATOL, rtol=0)


def test_xatt3_one_shared_phi():
    m = xatt()
    convs = [mod for mod in m.modules() if isinstance(mod, torch.nn.Conv1d)]
    assert len(convs) == 1 and convs[0] is m.phi
    assert (m.phi.in_channels, m.phi.out_channels, m.phi.kernel_size, m.phi.bias) == (64, 72, (1,), None)
    assert (m.psi.in_features, m.psi.out_features) == (128, 128) and m.psi.bias is not None
    assert sum(p.numel() for p in m.parameters()) == 64 * 72 + 128 * 128 + 128


def test_xatt4_background_slot_is_the_way_mean_before_pooling():
    m, (f_s, f_q) = xatt(), features(n=3)
    a = m.attention(f_s, f_q)
    only_mean = m.attention(f_s.mean(0, keepdim=True), f_q)  # a 1-way episode whose way is the mean
    assert torch.allclose(a[:, 0], only_mean[:, 1], atol=ATOL, rtol=0)
    ways = torch.tensor([2, 0, 1])
    b = m.attention(f_s[ways], f_q)  # permuting ways: slot 0 unchanged, slots 1..N permuted
    assert torch.allclose(b[:, 0], a[:, 0], atol=ATOL, rtol=0)
    assert torch.allclose(b[:, 1:], a[:, 1:][:, ways], atol=ATOL, rtol=0)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_xatt5_p_cross_is_the_shot_mean_of_a_times_psi(k):
    m, (f_s, f_q) = xatt(), features(k=k)
    p_gated = rand(BQ, N + 1, D, seed=30)
    a = m.attention(f_s, f_q)
    assert torch.allclose(m(p_gated, f_s, f_q), p_cross_ref(m, a, p_gated), atol=ATOL, rtol=0)
    shots = torch.arange(k).flip(0)
    assert torch.allclose(m(p_gated, f_s[:, shots], f_q), m(p_gated, f_s, f_q), atol=ATOL, rtol=0)


def test_xatt6_queries_do_not_mix():
    m, (f_s, f_q) = xatt(), features(bq=3)
    p_gated = rand(3, N + 1, D, seed=31)
    out = m(p_gated, f_s, f_q)
    perm = torch.tensor([2, 0, 1])
    assert torch.allclose(m(p_gated[perm], f_s, f_q[perm]), out[perm], atol=ATOL, rtol=0)
    other_q, other_p = f_q.clone(), p_gated.clone()
    other_q[1:], other_p[1:] = rand(2, P, D, seed=32).abs(), rand(2, N + 1, D, seed=33)
    assert torch.allclose(m(other_p, f_s, other_q)[0], out[0], atol=ATOL, rtol=0)


def test_xatt_gradcheck_in_the_prototype():
    m, (f_s, f_q) = xatt(), features(k=1, bq=1)
    p = rand(1, N + 1, D, seed=34).requires_grad_(True)
    assert torch.autograd.gradcheck(lambda x: m(x, f_s, f_q), (p,), eps=1e-6, atol=1e-8)


def test_xatt_invalid_scale_raises():
    with pytest.raises(ValueError):
        CrossAttention(scale="sqrt_72")
