"""EPS-1..9 (05 §3.4b, gate G1): the stripped EPPM-S stage of [DECISION D-24], beyond the paper.

References are written with explicit loops, independent of the vectorised code, in float64 to 1e-12.
"""

import math

import pytest
import torch

from models.eppm_s import EPPMSharedStage
from tests.test_eppm import P, rand, vipseg_pool

ATOL = 1e-12
D = 128


def stage(support="pooled", self_branch=True, scale="sqrt_d", seed=0):
    torch.manual_seed(seed)
    return EPPMSharedStage(support=support, self_branch=self_branch, cross_attn_scale=scale).double()


def features(n=2, k=2, bq=2, seed=50):
    f_s = rand(n, k, P, D, seed=seed).abs()  # [N, K, 2048, D], non-negative like the head's ReLU output
    f_q = rand(bq, P, D, seed=seed + 1).abs()  # [B_q, 2048, D]
    return f_s, f_q


def row_softmax(logits):
    e = torch.exp(logits - logits.max(dim=-1, keepdim=True).values)
    return e / e.sum(dim=-1, keepdim=True)


def layer_norm_ref(m, x):
    """LayerNorm(D) written out, eps as PyTorch's default."""
    mean = x.mean(-1, keepdim=True)
    var = ((x - mean) ** 2).mean(-1, keepdim=True)
    return (x - mean) / torch.sqrt(var + m.norm.eps) * m.norm.weight + m.norm.bias


def forward_ref(m, p, f_s, f_q):
    """One EPPM-S stage from loops: φ as a matrix, row softmax by hand, ψ and W_out written out."""
    w = m.phi.weight[:, :, 0]  # [72, 64]
    n, k, bq = f_s.shape[0], f_s.shape[1], f_q.shape[0]
    v = p @ m.psi.weight.T + m.psi.bias  # ψ(P^{t-1}) [B_q, N+1, D]
    r = m.reweight.weight[0]  # [D]
    update = torch.zeros_like(v)
    if m.support == "pooled":
        s = w @ (sum(vipseg_pool(f_s[i, j]) for i in range(n) for j in range(k)) / (n * k))  # [72, D]
        g_s = s.T @ s  # [D, D]
    else:
        slots = [sum(f_s[i] for i in range(n)) / n] + [f_s[i] for i in range(n)]  # N+1 tensors [K, 2048, D]
    for b in range(bq):
        q = w @ vipseg_pool(f_q[b])  # [72, D]
        g_q = q.T @ q  # [D, D]
        if m.support == "pooled":
            a = row_softmax(q.T @ s / math.sqrt(72))  # [D, D]
            gate = torch.sigmoid(((g_q - g_s) / math.sqrt(D)) @ r)  # [D]
            for c in range(n + 1):
                update[b, c] = a @ v[b, c] + (gate * v[b, c] if m.self_branch else 0.0)
        else:
            for c in range(n + 1):
                cross, gate = torch.zeros(D, dtype=torch.float64), torch.zeros(D, dtype=torch.float64)
                for j in range(k):
                    sj = w @ vipseg_pool(slots[c][j])  # [72, D]
                    cross += row_softmax(q.T @ sj / math.sqrt(72)) @ v[b, c] / k
                    gate += torch.sigmoid(((g_q - sj.T @ sj) / math.sqrt(D)) @ r) / k
                update[b, c] = cross + (gate * v[b, c] if m.self_branch else 0.0)
    return layer_norm_ref(m, update @ m.w_out.weight.T + p)


def test_eps1_parameter_budget_and_dropped_components():
    m = stage()
    assert sorted(n for n, _ in m.named_parameters()) == ["norm.bias", "norm.weight", "phi.weight",
                                                          "psi.bias", "psi.weight", "reweight.weight",
                                                          "w_out.weight"]
    assert sum(p.numel() for p in m.parameters()) == 64 * 72 + 128 * 128 + 128 + 128 + 128 * 128 + 2 * 128
    assert sum(p.numel() for p in m.parameters()) == 37888  # against 79,395 for an EPPM stage (01 §4)
    for dropped in ("gate", "out", "fusion", "se_down", "se_up"):  # entropy gate, Eq.19-20, w_cls
        assert not hasattr(m, dropped)
    assert m.w_out.bias is None and m.phi.bias is None


@pytest.mark.parametrize("support", ["pooled", "class_slots"])
@pytest.mark.parametrize("n,k,bq", [(2, 1, 2), (2, 3, 2), (3, 2, 3), (1, 1, 1)])
def test_eps2_forward_equals_the_written_out_stage(support, n, k, bq):
    m = stage(support)
    f_s, f_q = features(n, k, bq)
    p = rand(bq, n + 1, D, seed=60)
    assert torch.allclose(m(p, f_s, f_q), forward_ref(m, p, f_s, f_q), atol=ATOL, rtol=0)


def test_eps3_output_shape_and_layernorm():
    m = stage()
    f_s, f_q = features()
    out = m(rand(2, 3, D, seed=61), f_s, f_q)
    assert out.shape == (2, 3, D)
    assert torch.allclose(out.mean(-1), torch.zeros(2, 3, dtype=torch.float64), atol=1e-10, rtol=0)
    # var = 1 - eps/(var + eps) from LayerNorm's epsilon, so the tolerance is loose on purpose
    assert torch.allclose(out.var(-1, unbiased=False), torch.ones(2, 3, dtype=torch.float64), atol=1e-3, rtol=0)


def test_eps4_zero_update_leaves_the_normalised_residual():
    m = stage()
    with torch.no_grad():
        m.w_out.weight.zero_()
    f_s, f_q = features()
    p = rand(2, 3, D, seed=62)
    assert torch.allclose(m(p, f_s, f_q), layer_norm_ref(m, p), atol=ATOL, rtol=0)


def test_eps5_self_branch_is_a_per_channel_gate_on_psi_of_the_prototype():
    m, off = stage(self_branch=True, seed=3), stage(self_branch=False, seed=3)
    f_s, f_q = features()
    p = rand(2, 3, D, seed=63)
    q, s = m.projections(f_s, f_q)
    gate = m.self_gate(q, s)
    assert gate.shape == (2, 1, D) and (gate > 0).all() and (gate < 1).all()
    v = m.psi(p)
    assert torch.allclose(m(p, f_s, f_q),
                          layer_norm_ref(m, (m.cross(q, s, v) + gate * v) @ m.w_out.weight.T + p),
                          atol=ATOL, rtol=0)
    assert torch.allclose(off(p, f_s, f_q), layer_norm_ref(off, off.cross(q, s, off.psi(p))
                                                           @ off.w_out.weight.T + p), atol=ATOL, rtol=0)
    assert not torch.allclose(m(p, f_s, f_q), off(p, f_s, f_q))


@pytest.mark.parametrize("support", ["pooled", "class_slots"])
def test_eps6_queries_do_not_mix_and_class_rows_are_equivariant(support):
    m = stage(support)
    f_s, f_q = features(bq=3)
    p = rand(3, 3, D, seed=64)
    out = m(p, f_s, f_q)
    other_q, other_p = f_q.clone(), p.clone()
    other_q[1:], other_p[1:] = rand(2, P, D, seed=65).abs(), rand(2, 3, D, seed=66)
    assert torch.allclose(m(other_p, f_s, other_q)[0], out[0], atol=ATOL, rtol=0)  # no batch coupling
    ways = torch.tensor([1, 0])
    rows = torch.tensor([0, 2, 1])  # background stays, the two ways swap
    assert torch.allclose(m(p[:, rows], f_s[ways], f_q), out[:, rows], atol=ATOL, rtol=0)


def test_eps7_the_two_support_readings_coincide_at_one_way_one_shot_and_differ_otherwise():
    pooled, slots = stage("pooled", seed=5), stage("class_slots", seed=5)
    f_s, f_q = features(n=1, k=1)
    p = rand(2, 2, D, seed=67)
    assert torch.allclose(pooled(p, f_s, f_q), slots(p, f_s, f_q), atol=ATOL, rtol=0)
    f_s, f_q = features(n=2, k=2)
    p = rand(2, 3, D, seed=68)
    assert (pooled(p, f_s, f_q) - slots(p, f_s, f_q)).abs().max() > 1e-6


def test_eps8_every_parameter_receives_a_gradient_and_the_prototype_gradcheck_passes():
    m = stage()
    f_s, f_q = features(k=1)
    # A plain .sum() gives LayerNorm's gamma the gradient Σ_D x̂ = 0, so weight the sum randomly
    (m(rand(2, 3, D, seed=69), f_s, f_q) * rand(2, 3, D, seed=73)).sum().backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())
    small = stage(seed=6)
    f_s, f_q = features(n=1, k=1, bq=1)
    p = rand(1, 2, D, seed=70).requires_grad_(True)
    assert torch.autograd.gradcheck(lambda x: small(x, f_s, f_q), (p,), eps=1e-6, atol=1e-8)


def test_eps9_shape_contract_and_invalid_switches_raise():
    m = stage()
    f_s, f_q = features()
    with pytest.raises(ValueError, match="does not match"):
        m(rand(2, 4, D, seed=71), f_s, f_q)  # N+2 class rows
    with pytest.raises(ValueError, match="does not match"):
        m(rand(3, 3, D, seed=72), f_s, f_q)  # more prototypes than queries
    with pytest.raises(ValueError, match="cross_attn_support"):
        EPPMSharedStage(support="episode")
    with pytest.raises(ValueError, match="cross_attn_scale"):
        EPPMSharedStage(cross_attn_scale="sqrt_72")
