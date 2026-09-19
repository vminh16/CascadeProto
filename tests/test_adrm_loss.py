"""ADRM-1..4 (05 §3.5, gate G1): the dynamic routing of Eq.24-25 / spec 02 §6.

float64; references use explicit loops and a hand-written softmax, compared to 1e-12.
"""

import math

import pytest
import torch

from models.adrm import DynamicRouting

ATOL = 1e-12
BQ, P, C, D = 3, 2048, 3, 128


def rand(*shape, seed=0, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return scale * torch.randn(*shape, generator=g, dtype=torch.float64)


def routing(t=4, seed=0):
    torch.manual_seed(seed)
    return DynamicRouting(t).double()


def inputs(t=4, seed=1):
    f_q = rand(BQ, P, D, seed=seed).abs()  # ReLU features (02 §2)
    logits = [rand(BQ, P, C, seed=seed + 1 + i) for i in range(t)]
    return logits, f_q


def weights_ref(m, f_q):
    """softmax(W_g · mean_i F^q[b, i]) per query, with the softmax written out."""
    out = []
    for b in range(f_q.shape[0]):
        z = [float(m.w_g.weight[t] @ f_q[b].mean(0)) for t in range(m.w_g.out_features)]
        e = [math.exp(v - max(z)) for v in z]
        out.append([v / sum(e) for v in e])
    return torch.tensor(out, dtype=torch.float64)


@pytest.mark.parametrize("t", [2, 3, 4, 6])
def test_adrm1_weights_shape_simplex_and_formula(t):
    m, (_, f_q) = routing(t), inputs(t)
    w = m.weights(f_q)
    assert w.shape == (BQ, t) and (w > 0).all()
    assert torch.allclose(w.sum(-1), torch.ones(BQ, dtype=torch.float64), atol=ATOL, rtol=0)
    assert torch.allclose(w, weights_ref(m, f_q), atol=ATOL, rtol=0)


def test_adrm2_w_g_has_no_bias():
    m = routing(4)
    assert m.w_g.weight.shape == (4, 128) and m.w_g.bias is None
    assert sum(p.numel() for p in m.parameters()) == 512  # 01 §4


@pytest.mark.parametrize("t", [2, 4])
def test_adrm3_final_logits_are_the_weighted_sum(t):
    m, (logits, f_q) = routing(t), inputs(t)
    w = weights_ref(m, f_q)
    expected = torch.zeros(BQ, P, C, dtype=torch.float64)
    for b in range(BQ):
        for s in range(t):
            expected[b] += w[b, s] * logits[s][b]
    assert torch.allclose(m(logits, f_q), expected, atol=ATOL, rtol=0)


def test_adrm3_equal_stage_logits_pass_through():
    """A convex combination of identical logits is those logits."""
    m, (logits, f_q) = routing(4), inputs(4)
    same = [logits[0]] * 4
    assert torch.allclose(m(same, f_q), logits[0], atol=ATOL, rtol=0)


def test_adrm4_gradients_to_every_stage_and_w_g():
    m, (logits, f_q) = routing(4), inputs(4)
    logits = [x.clone().requires_grad_(True) for x in logits]
    g = rand(BQ, P, C, seed=20)
    (m(logits, f_q) * g).sum().backward()
    w = m.weights(f_q).detach()
    for s in range(4):  # dL/dL^t = w_t[b] * G
        assert torch.allclose(logits[s].grad, w[:, s, None, None] * g, atol=ATOL, rtol=0)
    assert m.w_g.weight.grad is not None and m.w_g.weight.grad.abs().sum() > 0


def test_adrm4_gradcheck():
    """On 16 points: ADRM does not depend on the point count, and a full-size numerical Jacobian is too big."""
    m = routing(3)
    f_q = rand(2, 16, D, seed=30).abs().requires_grad_(True)
    logits = [rand(2, 16, 2, seed=31 + i).requires_grad_(True) for i in range(3)]
    assert torch.autograd.gradcheck(lambda q, a, b, c: m([a, b, c], q), (f_q, *logits), eps=1e-6, atol=1e-8)


def test_adrm_queries_do_not_mix():
    m, (logits, f_q) = routing(4), inputs(4)
    out = m(logits, f_q)
    other_q = f_q.clone()
    other_q[1:] = rand(BQ - 1, P, D, seed=40).abs()
    other_l = [x.clone() for x in logits]
    for x in other_l:
        x[1:] = torch.randn_like(x[1:])
    assert torch.allclose(m(other_l, other_q)[0], out[0], atol=ATOL, rtol=0)


def test_adrm_rejects_single_stage_and_wrong_count():
    with pytest.raises(ValueError, match="at least 2"):
        DynamicRouting(1)
    m, (logits, f_q) = routing(4), inputs(4)
    with pytest.raises(ValueError):
        m(logits[:3], f_q)
