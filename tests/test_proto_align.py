"""PA-1..9 (05 §3.8w): the prototype-alignment loss of [DECISION D-46] and P10.1's rule, beyond the paper.
CPU, float64; the encoder is the per-point stand-in of test_feature_extractor.py.
"""

import ast
import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from experiments import p10_align_probe as p10
from models.cascadeproto import CascadeProto, CascadeProtoConfig
from models.proto_align import TAU, alignment_loss
from models.vicreg import vicreg_regulariser
from models.vipseg_backbone import PointFeatureExtractor
from pipeline.model_api import episode_loss
from tests.test_cascadeproto import episode
from tests.test_feature_extractor import StandInEncoder

REPO = Path(__file__).resolve().parents[1]
CR = dict(use_lma=False, num_stages=0, l2norm_point_proto=True)  # no stage: VIP-Seg stages need the GPU ops


def rand(*shape, seed=0):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)


def reference(f_s, s_y, f_q, q_y, tau):
    """Explicit loops: support rows as normalised sums of unit features, query means likewise, CE per present class."""
    n = f_s.shape[0]
    u_s = F.normalize(f_s, dim=-1)
    rows = [F.normalize(sum(u_s[w, k, p] for w in range(n) for k in range(f_s.shape[1]) for p in range(f_s.shape[2])
                            if s_y[w, k, p] == 0), dim=0)]
    for w in range(n):
        rows.append(F.normalize(sum(u_s[w, k, p] for k in range(f_s.shape[1]) for p in range(f_s.shape[2])
                                    if s_y[w, k, p] == 1), dim=0))
    terms = []
    for b in range(f_q.shape[0]):
        for c in range(n + 1):
            pts = [F.normalize(f_q[b, i], dim=0) for i in range(f_q.shape[1]) if q_y[b, i] == c]
            if not pts:
                continue
            mu = F.normalize(sum(pts), dim=0)
            z = [float(mu @ r) / tau for r in rows]
            terms.append(-(z[c] - math.log(sum(math.exp(v) for v in z))))
    return sum(terms) / len(terms)


def small(seed=0, absent=None):
    f_s, f_q = rand(2, 1, 12, 5, seed=seed), rand(2, 10, 5, seed=seed + 1)
    s_y = torch.zeros(2, 1, 12, dtype=torch.long)
    s_y[0, 0, :4] = 1
    s_y[1, 0, 4:9] = 1
    q_y = torch.tensor([[0, 1, 2, 0, 1, 2, 0, 1, 2, 0], [0, 0, 1, 1, 0, 0, 1, 1, 0, 0]])
    if absent is not None:
        q_y[q_y == absent] = 0
    return f_s, s_y, f_q, q_y


def test_pa1_loss_by_hand():
    f_s, s_y, f_q, q_y = small()
    for tau in (TAU, 1.0):
        assert alignment_loss(f_s, s_y, f_q, q_y, tau).item() == pytest.approx(reference(f_s, s_y, f_q, q_y, tau))
    with pytest.raises(ValueError, match="tau"):
        alignment_loss(f_s, s_y, f_q, q_y, 0.0)


def test_pa2_absent_classes_are_not_scored():
    f_s, s_y, f_q, q_y = small(absent=2)  # class 2 in neither query block
    assert alignment_loss(f_s, s_y, f_q, q_y).item() == pytest.approx(reference(f_s, s_y, f_q, q_y, TAU))


def test_pa3_collapse_does_not_lower_the_loss():
    """Every feature on one direction gives the uniform softmax, log(N+1); aligned, separated means give less."""
    f_s, s_y, f_q, q_y = small()
    one = torch.ones(5, dtype=torch.float64)
    assert alignment_loss(one.expand_as(f_s).clone(), s_y, one.expand_as(f_q).clone(), q_y).item() == \
        pytest.approx(math.log(3))
    basis = torch.eye(5, dtype=torch.float64)[:3]  # class c -> e_c
    fs = torch.stack([basis[s_y[w, 0] * (w + 1)] for w in range(2)]).unsqueeze(1)  # [2, 1, 12, 5]
    fq = basis[q_y]  # [2, 10, 5]
    assert alignment_loss(fs, s_y, fq, q_y).item() < 1e-3 < math.log(3)


def test_pa4_gradients_reach_support_and_query():
    f_s, s_y, f_q, q_y = small()
    f_s.requires_grad_(True), f_q.requires_grad_(True)
    alignment_loss(f_s, s_y, f_q, q_y).backward()
    assert f_s.grad.abs().sum() > 0 and f_q.grad.abs().sum() > 0
    assert torch.autograd.gradcheck(lambda a, b: alignment_loss(a, s_y, b, q_y, 1.0),
                                    (f_s.detach().requires_grad_(True), f_q.detach().requires_grad_(True)))


def test_pa5_config_fields():
    c = CascadeProtoConfig()
    assert c.align_weight == 0.0 and c.align_tau == TAU
    for bad in (dict(align_weight=-1.0), dict(align_weight=float("nan")), dict(align_tau=0.0),
                dict(align_tau=float("inf"))):
        with pytest.raises(ValueError, match="align"):
            CascadeProtoConfig(**bad)
    old = CascadeProtoConfig().to_dict()
    del old["align_weight"], old["align_tau"]  # a checkpoint written before D-46
    assert CascadeProtoConfig(**old).align_weight == 0.0


def model(cfg, seed=0):
    torch.manual_seed(seed)
    return CascadeProto(CascadeProtoConfig(**cfg), PointFeatureExtractor(encoder=StandInEncoder())).double()


def test_pa6_training_loss_adds_the_alignment_only_when_asked():
    ep = episode(n=2, k=1, bq=2, seed=3)
    assert model(CR).train()(ep).loss_reg is None
    m = model(dict(CR, align_weight=0.5)).train()
    out = m(ep)
    f_s, f_q = m.features.encode_episode(ep.support_x, ep.query_x)
    ref = 0.5 * alignment_loss(f_s, ep.support_y, f_q, ep.query_y, TAU)
    assert torch.allclose(out.loss_reg, ref, atol=1e-10)
    seg = F.cross_entropy(out.logits.reshape(-1, 3), ep.query_y.reshape(-1))
    assert torch.allclose(episode_loss(out, ep), seg + out.loss_reg, atol=1e-10)
    both = model(dict(CR, align_weight=0.5, vicreg_var=1.0, vicreg_cov=0.04)).train()
    assert torch.allclose(both(ep).loss_reg, ref + vicreg_regulariser(f_q, 1.0, 0.04), atol=1e-10)
    assert m.eval()(ep).loss_reg is None  # evaluation never reads the query labels
    out = m.train()(ep)
    episode_loss(out, ep).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.features.parameters() if p.requires_grad)


def test_pa7_cascade_keeps_its_four_outputs():
    m, ep = model(CR).eval(), episode(n=2, k=1, bq=2, seed=4)
    four, five = m.cascade(ep), m.cascade(ep, return_support=True)
    assert len(four) == 4 and len(five) == 5
    assert all(torch.equal(a, b) for a, b in zip(four[:2], five[:2]))
    assert torch.equal(five[4], m.features.encode_episode(ep.support_x, ep.query_x)[0])


def test_pa8_train_flags_run_tag_and_rule():
    import train

    base = ["--dataset", "s3dis", "--data_path", "x", "--cvfold", "1", "--n_way", "2", "--k_shot", "1",
            "--use_lma", "false", "--num_stages", "4", "--l2norm_point_proto", "true", "--stage_type", "vip_clean",
            "--batch_size", "1", "--lr_step_epochs", "15", "--valid_every", "4", "--seed", "0",
            "--query_order", "random", "--save_dir", "log_d46"]
    cr = train.parse_args(base)
    a = train.parse_args(base + ["--align_weight", "0.25"])
    b = train.parse_args(base + ["--align_weight", "1"])
    assert train.run_dir(cr).replace("\\", "/") == "log_d46/s3dis_S1_N2_K1_point_T4_vip_clean_b1_qrandom"
    assert train.run_dir(a).replace("\\", "/").endswith("_vip_clean_b1_qrandom_align0.25")
    assert train.run_dir(b).replace("\\", "/").endswith("_vip_clean_b1_qrandom_align1")
    assert train.model_config(b).align_weight == 1.0 and train.model_config(b).align_tau == TAU
    old = {k: v for k, v in train.comparable_args(cr).items() if not k.startswith("align")}  # before D-46
    assert train.resume_mismatch(old, train.comparable_args(cr)) == []
    assert "align_weight" in train.resume_mismatch(old, train.comparable_args(a))
    assert p10.gap_holds(5.0) and not p10.gap_holds(4.99)


def test_pa9_files_parse_as_python_310():
    for f in ("models/proto_align.py", "models/cascadeproto.py", "train.py", "experiments/p10_align_probe.py",
              "tests/test_proto_align.py"):
        ast.parse((REPO / f).read_text(encoding="utf-8"), feature_version=(3, 10))


def part(u, oracle=0.80):
    return {"miou": {"U": u, "cos_oracle": oracle}}


def draws(gain_a, gain_b, rand_a=(1.0, 1.0, 1.0)):
    import numpy as np

    rng = np.random.default_rng(0)
    base = rng.integers(50, 100, size=(300, 3, 7)).astype(float)

    def shifted(g):
        c = base.copy()
        c[:, 2, 1:] += g * 0.02 * c[:, 0, 1:]  # more true positives -> higher IoU
        return c

    out = {"fixed100": ({"miou": {}}, {"cr:U": base, "m5a:U": shifted(gain_a), "m5b:U": shifted(gain_b)})}
    for i, d in enumerate(p10.d39.DRAWS[1:4]):
        out[d] = ({"miou": {"cr:U": 0.5, "m5a:U": 0.5 + rand_a[i] / 100, "m5b:U": 0.5}}, {})
    out["leakfree"] = ({"miou": {}}, {})
    return out


def test_pa10_rules():
    names = lambda *a: [n for n, _ in p10.decide(*a)]  # noqa: E731
    assert names({"m5a": None, "m5b": None}, {}) == ["incomplete"]  # a missing result is never a verdict
    assert names({"m5a": part(0.60), "m5b": None}, draws(3, 3))[-1] == "incomplete"
    wide = 0.80 - (p10.CR_VALID_GAP - p10.MECH_DROP) / 100  # U at exactly the bar
    assert "D46.1 m5a mechanism holds" in names({"m5a": part(wide + 1e-9), "m5b": part(0.50)}, draws(3, 3))
    assert any("D46.4" in n for n in names({"m5a": part(0.55), "m5b": part(0.50)}, draws(3, 3)))
    assert any("D46.4" in n for n in names({"m5a": part(0.57), "m5b": part(0.50)}, draws(3, 3)))  # gap 23 > 22.88
    got = names({"m5a": part(0.62), "m5b": part(0.50)}, draws(3, 3))
    assert "D46.2 m5a U holds at +1" in got and "D46.2 base" in got and not any("m5b U" in n for n in got)
    weak = names({"m5a": part(0.62), "m5b": part(0.50)}, draws(3, 3, rand_a=(1.0, -0.1, 1.0)))
    assert "D46.2 m5a U fails at +1" in weak and any("D46.3" in n for n in weak)
    assert names({"m5a": part(0.62), "m5b": part(0.50)}, {})[-1] == "incomplete"  # no draws
