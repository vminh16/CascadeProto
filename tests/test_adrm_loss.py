"""ADRM-1..4 and LOSS-1..3 (05 §3.5, gate G1): dynamic routing of Eq.24-25 and the objective of Eq.26-27.

float64; references use explicit loops and a hand-written softmax, compared to 1e-12.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from models.adrm import DynamicRouting
from pipeline.model_api import GMMN_WEIGHT, EpisodeOutput, episode_loss

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


# ------------------------------------------------------------------ LOSS (02 §7, Eq.26-27)

class _Episode:
    def __init__(self, query_y, n_way):
        self.query_y, self.n_way = query_y, n_way


def test_loss1_segmentation_loss_is_unweighted_ce_on_final_logits():
    logits = rand(2, P, 3, seed=50)
    y = torch.randint(0, 3, (2, P), generator=torch.Generator().manual_seed(51))
    out = EpisodeOutput(logits=logits, loss_gmmn=torch.zeros((), dtype=torch.float64))
    ref = -sum(math.log(math.exp(logits[b, i, y[b, i]].item()) / sum(math.exp(v) for v in logits[b, i].tolist()))
               for b in range(2) for i in range(0, P, 97)) / len(range(0, P, 97)) / 2
    sub = EpisodeOutput(logits=logits[:, ::97], loss_gmmn=torch.zeros((), dtype=torch.float64))
    assert math.isclose(episode_loss(sub, _Episode(y[:, ::97], 2)).item(), ref, rel_tol=0, abs_tol=ATOL)
    weighted = F.cross_entropy(logits.reshape(-1, 3), y.reshape(-1),
                               weight=torch.tensor([0.8, 1.0, 1.0], dtype=torch.float64))
    assert abs(episode_loss(out, _Episode(y, 2)).item() - weighted.item()) > 1e-4  # no w_cls in the loss


def test_loss2_total_is_seg_plus_one_times_gmmn():
    assert GMMN_WEIGHT == 1.0
    logits = rand(2, P, 3, seed=52)
    y = torch.randint(0, 3, (2, P), generator=torch.Generator().manual_seed(53))
    gmmn = torch.tensor(0.731, dtype=torch.float64)
    ce = F.cross_entropy(logits.reshape(-1, 3), y.reshape(-1))
    total = episode_loss(EpisodeOutput(logits, gmmn), _Episode(y, 2))
    assert math.isclose(total.item(), ce.item() + 0.731, rel_tol=0, abs_tol=ATOL)


def test_loss3_the_objective_sees_only_final_logits_and_gmmn():
    """No per-stage loss: the model contract carries L_final and L_GMMN only (Eq.27)."""
    assert EpisodeOutput._fields == ("logits", "loss_gmmn")
