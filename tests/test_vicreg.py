"""VR-1..10 (05 §3.8v): VICReg's variance and covariance terms on the query features and the D-45 monitor
[DECISION D-45], beyond the paper. CPU, float64; the encoder is the per-point stand-in of test_feature_extractor.py.
"""

import ast
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from experiments import d45_monitor as d45
from experiments import p9_placement_probe as p9
from models.cascadeproto import CascadeProto, CascadeProtoConfig
from models.vicreg import EPS, GAMMA, covariance_term, variance_term, vicreg_regulariser
from models.vipseg_backbone import PointFeatureExtractor
from pipeline.model_api import episode_loss
from tests.test_cascadeproto import episode
from tests.test_feature_extractor import StandInEncoder

REPO = Path(__file__).resolve().parents[1]
CR = dict(use_lma=False, num_stages=0, l2norm_point_proto=True)  # no stage: VIP-Seg stages need the GPU ops


def test_vr1_variance_term_by_hand():
    z = torch.zeros(10, 3, dtype=torch.float64)
    assert variance_term(z).item() == pytest.approx(GAMMA - np.sqrt(EPS))  # every column constant
    z[:, 0] = torch.arange(10, dtype=torch.float64)  # std 3.03 > 1: no penalty on that column
    z[:, 1] = 0.5 * torch.tensor([1.0, -1.0] * 5, dtype=torch.float64)
    std1 = np.sqrt(np.var([0.5, -0.5] * 5, ddof=1) + EPS)
    assert variance_term(z).item() == pytest.approx((0 + (1 - std1) + (1 - np.sqrt(EPS))) / 3)
    with pytest.raises(ValueError, match="n >= 2"):
        variance_term(torch.zeros(1, 3))
    with pytest.raises(ValueError, match="n >= 2"):
        variance_term(torch.zeros(4))


def test_vr2_covariance_term_by_hand():
    g = torch.Generator().manual_seed(0)
    z = torch.randn(50, 4, generator=g, dtype=torch.float64)
    c = np.cov(z.numpy(), rowvar=False, ddof=1)
    ref = (c ** 2).sum() - (np.diag(c) ** 2).sum()
    assert covariance_term(z).item() == pytest.approx(ref / 4)
    a = torch.randn(50, 1, generator=g, dtype=torch.float64)
    same = torch.cat([a, a, a, a], dim=1)  # perfectly correlated: every off-diagonal = var(a)
    v = float(a.var())
    assert covariance_term(same).item() == pytest.approx(12 * v ** 2 / 4)
    with pytest.raises(ValueError, match="n >= 2"):
        covariance_term(torch.zeros(1, 2))


def test_vr3_regulariser_flattens_blocks_and_points_and_weights():
    g = torch.Generator().manual_seed(1)
    f_q = torch.randn(2, 30, 5, generator=g, dtype=torch.float64)
    z = torch.cat([f_q[0], f_q[1]])  # [60, 5]
    got = vicreg_regulariser(f_q, 2.0, 0.5)
    assert got.item() == pytest.approx(2.0 * variance_term(z).item() + 0.5 * covariance_term(z).item())
    assert vicreg_regulariser(f_q, 0.0, 0.0).item() == 0.0


def test_vr4_training_against_the_terms_widens_a_collapsed_spectrum():
    """A learned linear map whose output lies on two directions: Adam on μ v + ν c raises the participation ratio.
    A single step on the features themselves barely moves it (the gradient of v is proportional to a point's
    deviation from the mean, near zero on a dead direction); through learned weights it accumulates over points."""
    g = torch.Generator().manual_seed(2)
    x = torch.randn(400, 16, generator=g, dtype=torch.float64)
    w0 = (torch.randn(16, 2, generator=g, dtype=torch.float64) @ torch.randn(2, 16, generator=g, dtype=torch.float64)
          + 1e-3 * torch.randn(16, 16, generator=g, dtype=torch.float64))
    w = w0.clone().requires_grad_(True)
    opt = torch.optim.Adam([w], lr=1e-2)
    for _ in range(300):
        opt.zero_grad()
        vicreg_regulariser((x @ w).unsqueeze(0), 1.0, 0.04).backward()
        opt.step()
    before = p9.participation_ratio(torch.cov((x @ w0).T))
    after = p9.participation_ratio(torch.cov((x @ w).detach().T))
    assert before < 2.0 and after > before + 1.5


def test_vr5_config_fields():
    c = CascadeProtoConfig()
    assert c.vicreg_var == 0.0 and c.vicreg_cov == 0.0
    for bad in (dict(vicreg_var=-1.0), dict(vicreg_cov=float("nan")), dict(vicreg_var=float("inf"))):
        with pytest.raises(ValueError, match="vicreg"):
            CascadeProtoConfig(**bad)
    old = CascadeProtoConfig().to_dict()
    del old["vicreg_var"], old["vicreg_cov"]  # a checkpoint written before D-45
    assert CascadeProtoConfig(**old).vicreg_var == 0.0


def model(cfg, seed=0):
    torch.manual_seed(seed)
    return CascadeProto(CascadeProtoConfig(**cfg), PointFeatureExtractor(encoder=StandInEncoder())).double()


def test_vr6_training_loss_adds_the_regulariser_only_when_asked():
    ep = episode(n=2, k=1, bq=2, seed=3)
    plain = model(CR).train()
    out0 = plain(ep)
    assert out0.loss_reg is None
    m = model(dict(CR, vicreg_var=1.0, vicreg_cov=0.04)).train()
    out = m(ep)
    f_s, f_q = m.features.encode_episode(ep.support_x, ep.query_x)
    ref = vicreg_regulariser(f_q, 1.0, 0.04)
    assert torch.allclose(out.loss_reg, ref, atol=1e-10)
    seg = F.cross_entropy(out.logits.reshape(-1, 3), ep.query_y.reshape(-1))
    assert torch.allclose(episode_loss(out, ep), seg + out.loss_reg, atol=1e-10)
    assert torch.allclose(episode_loss(out0, ep), F.cross_entropy(out0.logits.reshape(-1, 3), ep.query_y.reshape(-1)),
                          atol=1e-10)
    m.eval()
    assert m(ep).loss_reg is None  # evaluation never builds it
    out = m.train()(ep)
    episode_loss(out, ep).backward()
    grads = [p.grad for p in m.features.parameters() if p.requires_grad]
    assert any(g is not None and g.abs().sum() > 0 for g in grads)


def test_vr7_train_flags_and_run_tag():
    import train

    base = ["--dataset", "s3dis", "--data_path", "x", "--cvfold", "1", "--n_way", "2", "--k_shot", "1",
            "--use_lma", "false", "--num_stages", "4", "--l2norm_point_proto", "true", "--stage_type", "vip_clean",
            "--batch_size", "1", "--lr_step_epochs", "15", "--valid_every", "4", "--seed", "0",
            "--query_order", "random", "--save_dir", "log_d45"]
    cr = train.parse_args(base)
    a = train.parse_args(base + ["--vicreg_var", "1", "--vicreg_cov", "0.04"])
    b = train.parse_args(base + ["--vicreg_var", "4", "--vicreg_cov", "0.16"])
    assert train.run_dir(cr).replace("\\", "/") == "log_d45/s3dis_S1_N2_K1_point_T4_vip_clean_b1_qrandom"
    assert train.run_dir(a).replace("\\", "/").endswith("_vip_clean_b1_qrandom_vic1_0.04")
    assert train.run_dir(b).replace("\\", "/").endswith("_vip_clean_b1_qrandom_vic4_0.16")
    assert train.model_config(a).vicreg_var == 1.0 and train.model_config(a).vicreg_cov == 0.04
    old = {k: v for k, v in train.comparable_args(cr).items() if not k.startswith("vicreg")}  # before D-45
    assert train.resume_mismatch(old, train.comparable_args(cr)) == []
    assert "vicreg_var" in train.resume_mismatch(old, train.comparable_args(a))


def test_vr8_monitor_moments_and_early_stop():
    g = torch.Generator().manual_seed(4)
    x = torch.randn(300, 6, generator=g, dtype=torch.float64)
    m = d45.Moments()
    m.add(x[:100]), m.add(x[100:])
    assert torch.allclose(m.cov(), torch.cov(x.T, correction=0), atol=1e-12)
    assert m.ratio() == pytest.approx(p9.participation_ratio(torch.cov(x.T, correction=0)))
    assert d45.early_stop(24, 7.9) and not d45.early_stop(24, 8.0) and not d45.early_stop(20, 3.0)
    assert d45.early_stop(28, 7.0)  # the first measurement at or after epoch 24


def part(pr, u=0.56):
    return {"collapse": {"participation_ratio": pr, "span_share_median": 0.5}, "purity": {"miss": 0.2},
            "miou": {"U": u}}


def draws(gain_a, gain_b, rand_a=(1.0, 1.0, 1.0)):
    rng = np.random.default_rng(0)
    base = rng.integers(50, 100, size=(300, 3, 7)).astype(float)

    def shifted(g):
        c = base.copy()
        c[:, 2, 1:] += g * 0.02 * c[:, 0, 1:]  # more true positives -> higher IoU
        return c

    fixed = {"cr:U": base, "m2a:U": shifted(gain_a), "m2b:U": shifted(gain_b)}
    out = {"fixed100": ({"miou": {}}, fixed)}
    for i, d in enumerate(d45.d39.DRAWS[1:4]):
        out[d] = ({"miou": {"cr:U": 0.5, "m2a:U": 0.5 + rand_a[i] / 100, "m2b:U": 0.5}}, {})
    out["leakfree"] = ({"miou": {}}, {})
    return out


def test_vr9_rules():
    names = lambda *a: [n for n, _ in d45.decide(*a)]  # noqa: E731
    stopped = [{"arm": "m2a", "stopped": True, "ratio": 6.0, "epoch": 24},
               {"arm": "m2b", "stopped": True, "ratio": 7.0, "epoch": 24}]
    assert any("D45.4" in n for n in names({"m2a": None, "m2b": None}, {}, stopped))
    assert names({"m2a": None, "m2b": None}, {}, []) == ["incomplete"]  # a missing result is never a verdict
    assert "incomplete" in names({"m2a": part(15.0), "m2b": None}, draws(3, 3), [])
    assert any("D45.4" in n for n in names({"m2a": part(11.9), "m2b": part(6.0)}, draws(3, 3), []))
    got = names({"m2a": part(12.0), "m2b": part(6.0)}, draws(3, 3), [])
    assert "D45.1 m2a mechanism holds" in got and "D45.1 m2b mechanism fails" in got
    assert "D45.2 m2a U holds at +1" in got and "D45.2 base" in got and not any("m2b U" in n for n in got)
    weak = names({"m2a": part(15.0), "m2b": part(6.0)}, draws(3, 3, rand_a=(1.0, -0.1, 1.0)), [])
    assert "D45.2 m2a U fails at +1" in weak and any("D45.3" in n for n in weak)
    assert any(n == "incomplete" for n in names({"m2a": part(15.0), "m2b": None}, {}, stopped[1:]))  # no draws


def test_vr10_files_parse_as_python_310():
    for f in ("models/vicreg.py", "models/cascadeproto.py", "pipeline/model_api.py", "train.py",
              "experiments/d45_monitor.py", "experiments/p9_placement_probe.py", "tests/test_vicreg.py"):
        ast.parse((REPO / f).read_text(encoding="utf-8"), feature_version=(3, 10))
