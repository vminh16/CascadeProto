"""SS-1..10 (05 §3.8q): trained self-support prototypes and the D-39 reader [DECISION D-39], beyond the paper.

CPU, float64; the encoder is the per-point stand-in of test_feature_extractor.py.
"""

import ast
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from experiments import d39_eval as d39
from experiments import p6_prototype_probe as p6
from experiments.p5_condition_probe import support_directions
from models.cascadeproto import CascadeProto, CascadeProtoConfig
from models.prototypes import unit_prototypes
from models.self_support import SelfSupport
from models.vipseg_backbone import PointFeatureExtractor
from pipeline.model_api import episode_loss
from tests.test_cascadeproto import episode
from tests.test_feature_extractor import StandInEncoder

REPO = Path(__file__).resolve().parents[1]
ATOL = 1e-12
A0 = dict(use_lma=False, num_stages=0, prototype_rule="unit")
A1 = dict(A0, self_support_steps=2, support_aux=1.0)


def feats(bq=2, p=512, d=16, n=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    f_q = torch.rand(bq, p, d, generator=g, dtype=torch.float64)
    f_s = torch.rand(n, 1, p, d, generator=g, dtype=torch.float64)
    y = (torch.rand(n, 1, p, generator=g) < 0.3).long()
    return f_q, f_s, y


def model(cfg, seed=0):
    torch.manual_seed(seed)
    return CascadeProto(CascadeProtoConfig(**cfg), PointFeatureExtractor(encoder=StandInEncoder())).double()


def test_ss1_unit_prototypes_are_the_support_directions():
    f_q, f_s, y = feats()
    assert torch.allclose(unit_prototypes(f_s, y), support_directions(f_s, y), atol=ATOL)
    with pytest.raises(ValueError, match="foreground"):
        unit_prototypes(f_s, torch.zeros_like(y))


@pytest.mark.parametrize("a_bg,a_fg,which,alpha", [(0.25, 1.0, "bg", 0.25), (0.5, 0.5, "all", 0.5)])
def test_ss2_self_support_equals_p6_rule_with_rho_1(a_bg, a_fg, which, alpha):
    f_q, f_s, y = feats()
    rows = p6.base_rows(f_q, f_s, y)
    ss = SelfSupport(2).double()
    with torch.no_grad():
        ss.theta_bg.fill_(np.log(a_bg / (1 - a_bg)))
        ss.theta_fg.fill_(1e4 if a_fg == 1.0 else np.log(a_fg / (1 - a_fg)))
    out = ss(f_q, rows)
    assert len(out) == 2
    assert torch.allclose(out[-1], p6.self_support(f_q, rows, 1.0, alpha, 2, which), atol=1e-10)
    assert torch.allclose(out[0], p6.self_support(f_q, rows, 1.0, alpha, 1, which), atol=1e-10)
    with pytest.raises(ValueError):
        SelfSupport(0)
    few = SelfSupport(1, min_points=10 ** 6).double()  # no class has enough predicted points: rows unchanged
    assert torch.equal(few(f_q, rows)[0], rows)


def test_ss3_gradients_reach_alphas_and_features():
    f_q, f_s, y = feats()
    f_q.requires_grad_(True)
    ss = SelfSupport(2).double()
    rows = p6.base_rows(f_q, f_s, y)
    loss = torch.einsum("bpd,bcd->bpc", f_q, ss(f_q, rows)[-1]).logsumexp(-1).mean()
    loss.backward()
    assert ss.theta_bg.grad is not None and ss.theta_bg.grad.abs() > 0 and ss.theta_fg.grad.abs() > 0
    assert f_q.grad.abs().sum() > 0
    assert torch.allclose(ss.alphas(3), torch.tensor([0.25, 0.5, 0.5], dtype=torch.float64), atol=1e-12)


@pytest.mark.parametrize("kwargs", [dict(A0, num_stages=2), dict(A0, use_lma=True), dict(A0, l2norm_point_proto=True),
                                    dict(A0, neck="sq_attn"), dict(use_lma=False, num_stages=0, self_support_steps=2),
                                    dict(A0, support_aux=1.0), dict(A0, prototype_rule="other"),
                                    dict(A0, self_support_steps=-1)])
def test_ss4_invalid_configurations_raise(kwargs):
    with pytest.raises(ValueError):
        CascadeProtoConfig(**kwargs)


def test_ss5_unit_rule_logits_rows_and_aux_loss():
    ep = episode(n=2, k=1, bq=2, seed=1)
    m0 = model(A0).eval()
    f_s, f_q = m0.features.encode_episode(ep.support_x, ep.query_x)
    r0 = unit_prototypes(f_s, ep.support_y)
    assert torch.allclose(m0(ep).logits, torch.einsum("bpd,cd->bpc", f_q, r0), atol=ATOL)
    m1 = model(A1).eval()
    f_s, f_q = m1.features.encode_episode(ep.support_x, ep.query_x)
    r0 = unit_prototypes(f_s, ep.support_y)
    steps = m1.self_support(f_q, r0.unsqueeze(0).expand(2, -1, -1))
    assert torch.allclose(m1(ep).logits, torch.einsum("bpd,bcd->bpc", f_q, steps[-1]), atol=ATOL)
    q, p0, st, _ = m1.cascade(ep)
    assert torch.allclose(m1.effective_prototype(q, p0, st), steps[-1], atol=ATOL)
    assert m1(ep).loss_aux is None  # evaluation never builds the auxiliary loss
    m1.train()
    out = m1(ep)
    f_s, f_q = m1.features.encode_episode(ep.support_x, ep.query_x)
    ref = F.cross_entropy(torch.einsum("bpd,cd->bpc", f_q, unit_prototypes(f_s, ep.support_y)).reshape(-1, 3),
                          ep.query_y.reshape(-1))
    assert torch.allclose(out.loss_aux, ref, atol=1e-10) and out.aux_weight == 1.0
    total = episode_loss(out, ep)
    seg = F.cross_entropy(out.logits.reshape(-1, 3), ep.query_y.reshape(-1))
    assert torch.allclose(total, seg + out.loss_aux, atol=1e-10)
    assert model(A0).train()(ep).loss_aux is None


def test_ss6_trained_self_support_is_order_free():
    ep = episode(n=2, k=1, bq=2, seed=5)
    sw = episode(n=2, k=1, bq=2, seed=5)
    sw.query_x, sw.query_y = ep.query_x[[1, 0]], ep.query_y[[1, 0]]
    m = model(A1).train()

    def lg(e):
        m.zero_grad()
        loss = episode_loss(m(e), e)
        loss.backward()
        return loss.item(), {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}

    l0, g0 = lg(ep)
    l1, g1 = lg(sw)
    scale = max(g.abs().max().item() for g in g0.values())
    assert abs(l0 - l1) < 1e-12 and max((g0[n] - g1[n]).abs().max().item() for n in g0) / scale < 1e-10
    assert "self_support.theta_fg" in g0


def test_ss7_train_flags():
    import train

    base = ["--dataset", "s3dis", "--data_path", "x", "--cvfold", "1", "--n_way", "2", "--k_shot", "1",
            "--use_lma", "false", "--num_stages", "0", "--prototype_rule", "unit", "--batch_size", "1",
            "--lr_step_epochs", "15", "--valid_every", "4", "--query_order", "random", "--save_dir", "log_d39"]
    a0 = train.parse_args(base)
    a1 = train.parse_args(base + ["--self_support_steps", "2", "--support_aux", "1"])
    assert train.run_dir(a0).replace("\\", "/") == "log_d39/s3dis_S1_N2_K1_point_T0_b1_qrandom_unit"
    assert train.run_dir(a1).replace("\\", "/") == "log_d39/s3dis_S1_N2_K1_point_T0_b1_qrandom_unit_ssp2_aux1"
    assert train.model_config(a1).self_support_steps == 2 and train.model_config(a1).support_aux == 1.0
    plain = train.parse_args(base[:12] + ["--num_stages", "0"])  # a run without the D-39 flags
    old = {k: v for k, v in train.comparable_args(plain).items()
           if k not in ("prototype_rule", "self_support_steps", "support_aux")}  # a checkpoint before D-39
    assert train.resume_mismatch(old, train.comparable_args(plain)) == []
    assert "prototype_rule" in train.resume_mismatch(old, train.comparable_args(a0))


def test_ss8_km_logits_and_arms():
    f_q, f_s, y = feats()
    rows = p6.base_rows(f_q, f_s, y)
    cent = p6.spherical_kmeans(p6.background_units(f_s, y), 3)
    lk = d39.km_logits(f_q, rows, cent)
    plain = p6.rule_logits(f_q, rows)
    assert torch.equal(lk[..., 1:], plain[..., 1:])
    assert torch.allclose(lk[..., 0], torch.maximum(plain[..., 0], (f_q @ cent.T).max(-1).values), atol=ATOL)
    labels = torch.randint(0, 3, (2, 512))
    out = d39.arm_logits(f_q, f_s, y, rows, labels)
    assert set(out) == {"base", "ssp_bg", "km3", "both", "oracle"}
    assert torch.allclose(out["base"], plain, atol=ATOL)


def fake(tp):
    def counts(t, seed):
        rng = np.random.default_rng(seed)
        c = np.zeros((60, 3, 3))
        c[:, 0, :] = c[:, 1, :] = 100.0
        c[:, 2, 1:] = np.clip(t + rng.normal(0, 2, (60, 2)), 0, 100)
        c[:, 2, 0] = 90.0
        return c

    draws = {}
    for i, d in enumerate(d39.DRAWS):
        cnt = {k: counts(v, i) for k, v in tp.items()}
        draws[d] = ({"miou": {k: float(d39.p0.miou_from_counts(v.sum(0))) for k, v in cnt.items()}}, cnt)
    return draws


def levels(cr_both=70.0, a0=70.0, a1=70.0, a1_both=70.0):
    t = {}
    for n, model_tp in (("cr", 68.0), ("a0", a0), ("a1", a1)):
        for a in d39.ARMS:
            t[f"{n}:{a}"] = model_tp
        t[f"{n}:ssp_bg"], t[f"{n}:km3"], t[f"{n}:oracle"] = model_tp + 0.5, model_tp + 0.8, 90.0
        t[f"{n}:both"] = model_tp + 0.8
    t["cr:both"] = cr_both
    t["a1:both"] = a1_both
    return t


def test_ss9_rules():
    v = dict(d39.decide(fake(levels(cr_both=70.5, a0=69.0, a1=72.0, a1_both=73.0)), {"a1": {"bg": 0.3, "fg": 0.6}}))
    assert "D39.1 background -> both" in v and "D39.2 self-support go" in v
    assert any(k.startswith("D39.3 new base a1:both") for k in v)
    v = dict(d39.decide(fake(levels(cr_both=68.9, a0=69.0, a1=69.1)), {}))
    assert "D39.1 background -> km3" in v and "D39.2 self-support stop" in v
    assert any(k.startswith("D39.3 CR stays") or k.startswith("D39.3 new base a0:km3") for k in v)
    t = levels(cr_both=70.5, a0=69.0, a1=72.0, a1_both=73.0)
    draws = fake(t)
    r = draws["random600:1"][0]["miou"]
    r["a1:model"] = r["a0:model"] - 0.01  # one random600 draw below A0 blocks the go
    assert "D39.2 self-support between" in dict(d39.decide(draws, {}))
    assert d39.decide({}, {})[0][0] == "incomplete"


def test_ss10_files_parse_as_python_310():
    for f in ("models/self_support.py", "models/prototypes.py", "models/cascadeproto.py", "pipeline/model_api.py",
              "experiments/d39_eval.py", "train.py", "tests/test_self_support.py"):
        ast.parse((REPO / f).read_text(encoding="utf-8"), feature_version=(3, 10))
