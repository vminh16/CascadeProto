"""NECK-1..9 (05 §3.8m): the support → query attention neck and its warm start [DECISION D-33]. Beyond the paper, CPU."""

import ast
import math
import pathlib

import pytest
import torch
import torch.nn.functional as F

import train
from experiments import r2_distill_eval as r2
from models.cascadeproto import CascadeProtoConfig
from models.neck import SupportQueryAttention, build_neck
from tests.test_cascadeproto import episode, model

REPO = pathlib.Path(__file__).resolve().parents[1]
BASE = CascadeProtoConfig(use_lma=False, num_stages=2, l2norm_point_proto=True)
WITH_NECK = CascadeProtoConfig(use_lma=False, num_stages=2, l2norm_point_proto=True, neck="sq_attn")


def feats(seed=0, d=128):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(2, 1, 32, d, generator=g, dtype=torch.float64), torch.rand(2, 32, d, generator=g, dtype=torch.float64)


def test_neck1_identity_at_initialisation():
    n = SupportQueryAttention().double()
    f_s, f_q = feats()
    assert n.alpha.item() == 0.0 and torch.equal(n(f_s, f_q), f_s)


def test_neck2_update_is_the_written_attention():
    n = SupportQueryAttention().double()
    with torch.no_grad():
        n.alpha.fill_(0.7)
    f_s, f_q = feats(1)
    s, q = f_s.reshape(-1, 128), f_q.reshape(-1, 128)
    att = torch.softmax(n.w_q(n.norm_s(s)) @ n.w_k(n.norm_q(q)).T / math.sqrt(64), dim=-1)
    want = f_s + 0.7 * n.w_o(att @ n.w_v(n.norm_q(q))).view_as(f_s)
    assert torch.allclose(n(f_s, f_q), want, atol=1e-12)
    other = f_q.clone()
    other[1] = torch.rand_like(other[1])  # the support update reads every query of the episode
    assert not torch.allclose(n(f_s, other), n(f_s, f_q))
    with pytest.raises(ValueError):
        n(f_s[0], f_q)


def test_neck3_rezero_gradients_at_initialisation():
    n = SupportQueryAttention().double()
    f_s, f_q = feats(2)
    n(f_s, f_q).pow(2).sum().backward()
    assert n.alpha.grad.abs() > 0
    assert all(p.grad is None or p.grad.abs().sum() == 0 for name, p in n.named_parameters() if name != "alpha")


def test_neck4_model_with_a_closed_neck_is_the_model_without_it():
    base, ep = model(BASE).eval(), episode(n=2, k=1)
    necked = model(WITH_NECK).eval()
    result = necked.load_state_dict(base.state_dict(), strict=False)
    assert all(k.startswith("neck.") for k in result.missing_keys) and not result.unexpected_keys
    assert torch.allclose(necked(ep).logits, base(ep).logits, atol=1e-12)
    with torch.no_grad():
        necked.neck.alpha.fill_(0.5)
    assert not torch.allclose(necked(ep).logits, base(ep).logits)
    moved = episode(n=2, k=1)
    moved.query_x = ep.query_x + 0.5  # same support, another query: F_s' and so P^0 must change
    assert not torch.allclose(necked.cascade(moved)[1], necked.cascade(ep)[1])
    n_params = sum(p.numel() for p in necked.neck.parameters())
    assert n_params == 2 * 2 * 128 + 3 * 128 * 64 + 64 * 128 + 1  # two LayerNorms, W_Q/K/V, W_O, alpha


def test_neck5_configuration():
    assert build_neck("none") is None and isinstance(build_neck("sq_attn"), SupportQueryAttention)
    with pytest.raises(ValueError):
        CascadeProtoConfig(neck="bogus")
    assert model(BASE).neck is None


def test_neck6_warm_start(tmp_path):
    base = model(BASE)
    path = tmp_path / "last.pt"
    torch.save({"model": base.state_dict(), "config": BASE.to_dict()}, path)
    necked = model(WITH_NECK)
    train.init_from_checkpoint(necked, WITH_NECK, str(path))
    theirs, ours = base.state_dict(), necked.state_dict()
    assert all(torch.equal(ours[k], v) for k, v in theirs.items())  # every loaded tensor is the checkpoint's
    assert necked.neck.alpha.item() == 0.0
    other = CascadeProtoConfig(use_lma=False, num_stages=3, l2norm_point_proto=True, neck="sq_attn")
    with pytest.raises(ValueError, match="configurations differ"):
        train.init_from_checkpoint(model(other), other, str(path))
    sd = base.state_dict()
    del sd[next(k for k in sd if k.startswith("stages."))]
    torch.save({"model": sd, "config": BASE.to_dict()}, path)
    with pytest.raises(ValueError, match="missing"):
        train.init_from_checkpoint(model(WITH_NECK), WITH_NECK, str(path))
    sd = base.state_dict()
    sd["extra.weight"] = torch.zeros(1)
    torch.save({"model": sd, "config": BASE.to_dict()}, path)
    with pytest.raises(ValueError, match="unexpected"):
        train.init_from_checkpoint(model(WITH_NECK), WITH_NECK, str(path))


def test_neck7_cli_and_run_directories():
    base = ["--dataset", "s3dis", "--data_path", "x", "--cvfold", "1", "--n_way", "2", "--k_shot", "1",
            "--use_lma", "false", "--num_stages", "4", "--stage_type", "vip", "--l2norm_point_proto", "true",
            "--batch_size", "1", "--init_checkpoint", "e1.pt"]
    ctl, neck = train.parse_args(base), train.parse_args(base + ["--neck", "sq_attn"])
    assert train.model_config(ctl).neck == "none" and train.model_config(neck).neck == "sq_attn"
    assert train.run_dir(ctl).endswith("s3dis_S1_N2_K1_point_T4_vip_b1_ft")
    assert train.run_dir(neck).endswith("s3dis_S1_N2_K1_point_T4_vip_b1_sq_attn_ft")
    saved = {k: v for k, v in train.comparable_args(ctl).items() if k not in ("neck", "init_checkpoint")}
    assert train.resume_mismatch(saved, train.comparable_args(train.parse_args(base[:-2]))) == []


def _n1(gain=1.4, lo=0.5, rand=(1.1, 0.9, 1.3), heads=(12.0, 10.0)):
    out = []
    for draw, g in zip(r2.DRAWS, (gain,) + tuple(rand)):
        paired = {"neck_vs_ctl": {"gain": g, "ci_low": lo, "ci_high": g + 0.5},
                  "ctl_oracle_unit_vs_ctl": {"gain": heads[0]}, "neck_oracle_unit_vs_neck": {"gain": heads[1]},
                  "ctl_vs_e1": {"gain": 0.2, "ci_low": -0.1, "ci_high": 0.5}}
        out.append({"draw": draw, "cvfold": 1, "miou": {"ctl": 0.73, "neck": 0.73 + g / 100}, "paired": paired})
    return out


def test_neck8_rules():
    v = dict(r2.decide_n1(_n1(), 1))
    assert "N1.1 go" in v and "flagged" not in v["N1.4 oracle gap"] and "reported ctl_vs_e1" in v
    assert "flagged" in dict(r2.decide_n1(_n1(heads=(10.0, 10.5)), 1))["N1.4 oracle gap"]
    assert "N1.2 stop" in dict(r2.decide_n1(_n1(gain=0.3, lo=-0.2), 1))
    assert "N1.2 stop" in dict(r2.decide_n1(_n1(rand=(0.2, 0.3, 0.4)), 1))
    assert "N1.3 in between" in dict(r2.decide_n1(_n1(gain=0.8, lo=0.2), 1))
    assert "N1.3 in between" in dict(r2.decide_n1(_n1(lo=-0.1), 1))  # CI containing 0
    assert "N1.3 in between" in dict(r2.decide_n1(_n1(rand=(1.1, -0.1, 2.0)), 1))
    assert r2.decide_n1(_n1()[:2], 1)[0][0] == "incomplete"


def test_neck10_alpha_initialisation_from_scratch():
    """D-34: the gate starts at 0.1 when asked, so the neck's projections get gradient at once."""
    n = SupportQueryAttention(alpha_init=0.1).double()
    f_s, f_q = feats(3)
    assert n.alpha.item() == pytest.approx(0.1) and not torch.allclose(n(f_s, f_q), f_s)
    n(f_s, f_q).pow(2).sum().backward()
    assert n.w_q.weight.grad.abs().sum() > 0
    cfg = CascadeProtoConfig(use_lma=False, num_stages=2, l2norm_point_proto=True, neck="sq_attn", neck_alpha_init=0.1)
    assert model(cfg).neck.alpha.item() == pytest.approx(0.1)
    for bad in (dict(neck_alpha_init=0.1), dict(neck="sq_attn", neck_alpha_init=float("nan"))):
        with pytest.raises(ValueError, match="neck_alpha_init"):
            CascadeProtoConfig(**bad)
    base = ["--dataset", "s3dis", "--data_path", "x", "--cvfold", "1", "--n_way", "2", "--k_shot", "1",
            "--use_lma", "false", "--num_stages", "4", "--stage_type", "vip", "--l2norm_point_proto", "true",
            "--batch_size", "1", "--neck", "sq_attn", "--neck_alpha_init", "0.1"]
    args = train.parse_args(base)
    assert train.model_config(args).neck_alpha_init == 0.1
    assert train.run_dir(args).endswith("s3dis_S1_N2_K1_point_T4_vip_b1_sq_attn_a0.1")


@pytest.mark.parametrize("path", ["models/neck.py", "models/cascadeproto.py", "train.py", "experiments/r2_distill_eval.py"])
def test_neck9_parses_as_python_3_10(path):
    ast.parse((REPO / path).read_text(encoding="utf-8"), feature_version=(3, 10))
