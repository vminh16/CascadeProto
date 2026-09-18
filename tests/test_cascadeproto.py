"""CP-1..8 (05 §3.6b, gate G1): the phase-10 CascadeProto (Table 4 "Baseline" row of D-17).

The VIP-Seg encoder is replaced by the per-point stand-in of test_feature_extractor.py; everything
after the encoder (feature head, prototypes, logits, loss) is the real code. float64, 1e-12.
"""

import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from models.cascadeproto import CascadeProto, CascadeProtoConfig
from models.prototypes import point_prototypes
from models.vipseg_backbone import PointFeatureExtractor
from pipeline.episodes import make_episode
from pipeline.model_api import episode_loss, predict
from tests.test_feature_extractor import StandInEncoder

ATOL = 1e-12
BASELINE = CascadeProtoConfig(use_lma=False, num_stages=0)
CLASS_NAMES = ["ceiling", "floor", "wall", "beam", "column", "window", "door",
               "table", "chair", "sofa", "bookcase", "board", "clutter"]


def model(config=BASELINE, seed=0):
    torch.manual_seed(seed)
    return CascadeProto(config, PointFeatureExtractor(encoder=StandInEncoder())).double()


def episode(n=2, k=2, bq=2, seed=0):
    rng = np.random.default_rng(seed)
    item = (rng.random((n, k, 2048, 9)), rng.integers(0, 2, (n, k, 2048)).astype(np.int32),
            rng.random((bq, 2048, 9)), rng.integers(0, n + 1, (bq, 2048)), np.arange(3, 3 + n))
    ep = make_episode(item, CLASS_NAMES)  # the pipeline's own validation of the 02 §1 layout
    ep.support_x, ep.query_x = ep.support_x.double(), ep.query_x.double()
    return ep


def test_cp1_output_contract():
    out = model().eval()(episode(n=3, bq=3))
    assert out.logits.shape == (3, 2048, 4) and out.logits.dtype == torch.float64
    assert out.loss_gmmn.shape == () and out.loss_gmmn.item() == 0.0  # use_lma=false drops L_GMMN (D-17)


def test_cp2_logits_are_query_features_times_point_prototypes():
    m, ep = model().eval(), episode(n=3, k=2, bq=3)
    f_s, f_q = m.features.encode_episode(ep.support_x, ep.query_x)
    p = point_prototypes(f_s, ep.support_y)  # [N+1, D]
    expected = torch.stack([f_q[b] @ p.T for b in range(3)])  # [B_q, 2048, N+1]
    assert torch.allclose(m(ep).logits, expected, atol=ATOL, rtol=0)
    b, i, c = 1, 777, 2
    assert math.isclose(m(ep).logits[b, i, c].item(), float(torch.dot(f_q[b, i], p[c])), abs_tol=ATOL)


def test_cp3_way_permutation_permutes_foreground_logits():
    m, ep = model().eval(), episode(n=3)
    ways = torch.tensor([2, 0, 1])
    permuted = episode(n=3)
    permuted.support_x, permuted.support_y = ep.support_x[ways], ep.support_y[ways]
    a, b = m(ep).logits, m(permuted).logits
    assert torch.allclose(b[..., 0], a[..., 0], atol=ATOL, rtol=0)
    assert torch.allclose(b[..., 1:], a[..., 1:][..., ways], atol=ATOL, rtol=0)


def test_cp4_queries_do_not_influence_each_other_in_eval():
    m, ep = model().eval(), episode(bq=3)
    other = episode(bq=3)
    other.query_x = ep.query_x.clone()
    other.query_x[1:] = torch.rand_like(other.query_x[1:])
    assert torch.allclose(m(ep).logits[0], m(other).logits[0], atol=ATOL, rtol=0)


def test_cp5_d10_ablation_flags():
    ep = episode()
    base = model().eval()
    scaled = model(CascadeProtoConfig(use_lma=False, num_stages=0, logit_scale="sqrt_D")).eval()
    assert torch.allclose(scaled(ep).logits, base(ep).logits / math.sqrt(128), atol=ATOL, rtol=0)
    normed = model(CascadeProtoConfig(use_lma=False, num_stages=0, l2norm_point_proto=True)).eval()
    f_s, f_q = base.features.encode_episode(ep.support_x, ep.query_x)
    p = point_prototypes(f_s, ep.support_y)
    expected = torch.einsum("bpd,cd->bpc", f_q, p / p.norm(dim=-1, keepdim=True))
    assert torch.allclose(normed(ep).logits, expected, atol=ATOL, rtol=0)


@pytest.mark.parametrize("config,phase", [(CascadeProtoConfig(), "phase 11"),
                                          (CascadeProtoConfig(use_lma=False, num_stages=1), "phase 12"),
                                          (CascadeProtoConfig(use_lma=True, num_stages=0), "phase 11")])
def test_cp6_unimplemented_configurations_raise(config, phase):
    with pytest.raises(NotImplementedError, match=phase):
        model(config)


@pytest.mark.parametrize("kwargs", [dict(num_stages=7), dict(num_stages=-1), dict(modality="video"),
                                    dict(logit_scale="sqrt_d")])
def test_cp6_invalid_configurations_raise(kwargs):
    with pytest.raises(ValueError):
        CascadeProtoConfig(use_lma=False, **{"num_stages": 0, **kwargs})


def test_cp7_loss_backward_reaches_every_parameter():
    m, ep = model().train(), episode()
    episode_loss(m(ep), ep).backward()
    for name, p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, name


def learnable_episode(n=2, k=2, bq=2, seed=3):
    """Every point has a class 0..n (0 = background) whose colour (channels 3-5) is centred on a
    class-specific value; support masks and query labels follow the same classes, as the loader's do."""
    rng = np.random.default_rng(seed)
    centres = rng.random((n + 1, 3))

    def cloud(labels):
        x = rng.random(labels.shape + (9,))
        x[..., 3:6] = centres[labels] + 0.05 * rng.standard_normal(labels.shape + (3,))
        return x

    s_lab = np.stack([rng.choice([0, w + 1], size=(k, 2048), p=[0.6, 0.4]) for w in range(n)])  # [N, K, P]
    q_lab = rng.integers(0, n + 1, (bq, 2048))
    item = (cloud(s_lab), (s_lab > 0).astype(np.int32), cloud(q_lab), q_lab, np.arange(3, 3 + n))
    ep = make_episode(item, CLASS_NAMES)
    ep.support_x, ep.query_x = ep.support_x.double(), ep.query_x.double()
    return ep


def test_cp8_baseline_learns_a_fixed_episode():
    """A few AdamW steps (the paper's optimiser settings) lower the loss and raise query accuracy."""
    m, ep = model().train(), learnable_episode()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.1)
    losses = []
    for _ in range(30):
        loss = episode_loss(m(ep), ep)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.8 * losses[0]
    m.eval()
    assert (predict(m(ep)) == ep.query_y).double().mean() > 1.0 / 3.0  # better than chance for N=2


def test_cp8_state_dict_round_trip():
    trained, ep = model(seed=0).eval(), episode()
    fresh = model(seed=1).eval()
    assert not torch.allclose(fresh(ep).logits, trained(ep).logits)
    fresh.load_state_dict(trained.state_dict(), strict=True)
    assert torch.equal(fresh(ep).logits, trained(ep).logits)
