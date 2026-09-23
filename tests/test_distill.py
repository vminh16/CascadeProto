"""DIS-1..10 (05 §3.8j): oracle-direction distillation of the pairwise decisions [DECISION D-29]. Beyond the paper, CPU.

The model tests use CascadeProto with the per-point stand-in encoder of test_cascadeproto.py and the
printed EPPM stages, which run on the CPU; the distillation does not depend on the stage type.
"""

import ast
import math
import pathlib

import pytest
import torch
import torch.nn.functional as F

import train
from experiments import r2_distill_eval as r2
from models.cascadeproto import CascadeProtoConfig
from models.oracle_distill import (cosine_to_oracle, oracle_directions, oracle_distill_loss, oracle_logits,
                                   pair_logit_cosine, pairwise_cosine)
from pipeline.model_api import EpisodeOutput, episode_loss
from tests.test_cascadeproto import episode, model

REPO = pathlib.Path(__file__).resolve().parents[1]
D = 8


def rand(*shape, seed=0):
    return torch.rand(*shape, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)


def test_dis1_oracle_directions_are_the_normalised_sum_of_unit_features():
    f_q = rand(2, 6, D)
    labels = torch.tensor([[0, 1, 1, 0, 0, 1], [0, 0, 0, 0, 0, 0]])
    o, present = oracle_directions(f_q, labels, 3)
    unit = f_q / f_q.norm(dim=-1, keepdim=True)
    want = unit[0, [1, 2, 5]].sum(0)
    assert torch.allclose(o[0, 1], want / want.norm(), atol=1e-12)
    assert torch.allclose(o.norm(dim=-1)[present], torch.ones(int(present.sum()), dtype=torch.float64))
    assert present.tolist() == [[True, True, False], [True, False, False]]
    assert torch.equal(o[1, 1:], torch.zeros(2, D, dtype=torch.float64))  # absent rows never read


def test_dis1_oracle_is_p0s_oracle_replace_direction():
    """κ → ∞ in 02 §11 with the one-hot labels gives O (P0's ORACLE_REPLACE arm)."""
    from models.transductive import em_step

    f_q, prior = rand(2, 40, D, seed=1), torch.randn(2, 3, D, generator=torch.Generator().manual_seed(2),
                                                     dtype=torch.float64)
    labels = torch.randint(0, 3, (2, 40), generator=torch.Generator().manual_seed(3))
    mu, _ = em_step(f_q, prior, torch.einsum("bpd,bcd->bpc", f_q, prior), 1e8, "oracle", labels)
    o, present = oracle_directions(f_q, labels, 3)
    assert torch.allclose(F.normalize(mu, dim=-1)[present], o[present], atol=1e-6)


LABELS = torch.tensor([[0, 1, 2, 0, 1, 2, 0, 0, 1, 2], [0, 0, 1, 1, 0, 0, 1, 0, 0, 1]])  # pairs: 3 + 1


def teacher(seed=10):
    f_q = rand(2, 10, D, seed=seed)
    return f_q, oracle_logits(f_q, LABELS, 3)


def test_dis2_teacher_is_the_oracle_direction_rule():
    f_q, t = teacher()
    o, _ = oracle_directions(f_q, LABELS, 3)
    assert torch.allclose(t, torch.einsum("bpd,bcd->bpc", f_q, o), atol=1e-12)


def test_dis2_loss_ignores_shift_and_scale_and_sees_orientation():
    f_q, t = teacher()
    shift = rand(2, 10, 1, seed=11) * 5  # one value per point, added to every class logit
    assert oracle_distill_loss(3.0 * t + shift, f_q, LABELS).item() == pytest.approx(0.0, abs=1e-12)
    assert oracle_distill_loss(-2.0 * t + shift, f_q, LABELS).item() == pytest.approx(2.0, abs=1e-12)
    flipped = t.clone()
    flipped[1] = -flipped[1]  # query 1 holds one pair of the four: 2 / 4
    assert oracle_distill_loss(flipped, f_q, LABELS).item() == pytest.approx(0.5, abs=1e-12)


def test_dis2_pair_cosine_uses_the_points_of_the_two_classes_only():
    f_q, t = teacher()
    logits = t + 0.3 * rand(2, 10, 3, seed=12)
    cos, mask = pair_logit_cosine(logits, t, LABELS)
    assert mask.tolist() == [[[False, True, True], [False, False, True], [False, False, False]],
                             [[False, True, False], [False, False, False], [False, False, False]]]
    idx = (LABELS[0] == 0) | (LABELS[0] == 1)  # the pair (0, 1) of query 0, written out
    a, b = logits[0, idx, 0] - logits[0, idx, 1], t[0, idx, 0] - t[0, idx, 1]
    assert cos[0, 0, 1].item() == pytest.approx(float(a @ b / (a.norm() * b.norm())), abs=1e-12)
    moved = logits.clone()
    moved[0, LABELS[0] == 2, :2] += 7.0  # points of class 2 do not enter the pair (0, 1)
    assert pair_logit_cosine(moved, t, LABELS)[0][0, 0, 1].item() == pytest.approx(cos[0, 0, 1].item(), abs=1e-12)


def test_dis2_prototype_shift_moves_the_prototype_cosine_not_the_loss():
    """The refuted first form (D-29 revision, point 1): a common shift of M changes cos(M_c, O_c)."""
    f_q = rand(1, 10, D, seed=13)
    m = torch.randn(1, 3, D, generator=torch.Generator().manual_seed(14), dtype=torch.float64)
    labels = LABELS[:1]
    shifted = m + 4.0 * rand(1, 1, D, seed=15)
    la, lb = (torch.einsum("bpd,bcd->bpc", f_q, x) for x in (m, shifted))
    assert oracle_distill_loss(la, f_q, labels).item() == pytest.approx(
        oracle_distill_loss(lb, f_q, labels).item(), abs=1e-12)
    assert not torch.allclose(cosine_to_oracle(m, f_q, labels)[0], cosine_to_oracle(shifted, f_q, labels)[0])
    o, present = oracle_directions(f_q, labels, 3)
    assert torch.allclose(pairwise_cosine(m, o, present)[0], pairwise_cosine(shifted, o, present)[0], atol=1e-12)


def test_dis2_no_pair_gives_a_zero_that_keeps_the_graph():
    logits = rand(1, 10, 3, seed=16).requires_grad_(True)
    loss = oracle_distill_loss(logits, rand(1, 10, D, seed=17), torch.zeros(1, 10, dtype=torch.long))
    assert loss.item() == 0.0 and loss.requires_grad


def test_dis3_the_target_is_a_stopped_gradient():
    f_q = rand(2, 10, D, seed=6).requires_grad_(True)
    logits = rand(2, 10, 3, seed=7).requires_grad_(True)
    oracle_distill_loss(logits, f_q, LABELS).backward()
    assert f_q.grad is None  # the teacher carries no gradient to the features
    assert logits.grad is not None and logits.grad.abs().sum() > 0


CONFIGS = [CascadeProtoConfig(use_lma=False, num_stages=0),
           CascadeProtoConfig(use_lma=False, num_stages=2, use_adrm=False),
           CascadeProtoConfig(use_lma=False, num_stages=3),
           CascadeProtoConfig(use_lma=False, num_stages=2, logit_scale="sqrt_D"),
           CascadeProtoConfig()]


@pytest.mark.parametrize("config", CONFIGS)
def test_dis4_effective_prototype_reproduces_the_logits(config):
    m, ep = model(config).eval(), episode(n=2, bq=2)
    f_q, p0, steps, _ = m.cascade(ep)
    m_eff = m.effective_prototype(f_q, p0, steps)
    scale = math.sqrt(128) if config.logit_scale == "sqrt_D" else 1.0
    assert m_eff.shape == (2, 3, 128) and len(steps) == config.num_stages
    assert torch.allclose(torch.einsum("bpd,bcd->bpc", f_q, m_eff) / scale, m(ep).logits, atol=1e-10, rtol=0)
    rule = r2.OursRule(m)
    f2, m2, steps2, logits = rule(ep)
    assert torch.allclose(torch.einsum("bpd,bcd->bpc", f2, m2), logits, atol=1e-10, rtol=0)
    assert len(steps2) == config.num_stages + 1


def test_dis5_logits_never_read_the_query_labels():
    config = CascadeProtoConfig(use_lma=False, num_stages=2, distill_beta=1.0)
    m, ep = model(config), episode(n=2, bq=2)
    shuffled = episode(n=2, bq=2)
    shuffled.query_y = torch.roll(ep.query_y, 7, dims=1)
    for mode in (m.train, m.eval):
        mode()
        a, b = m(ep), m(shuffled)
        assert torch.equal(a.logits, b.logits)
    m.eval()
    out = m(ep)
    assert out.loss_distill is None and out.distill_weight == 0.0  # nothing label-dependent at evaluation
    m.train()
    assert not torch.equal(m(ep).loss_distill, m(shuffled).loss_distill)


def test_dis6_objective_with_and_without_the_weight():
    ep = episode(n=2, bq=2)
    off = model(CascadeProtoConfig(use_lma=False, num_stages=3)).train()
    out = off(ep)
    assert out.distill_weight == 0.0 and not out.loss_distill.requires_grad  # logged, no gradient
    ce = F.cross_entropy(out.logits.reshape(-1, 3), ep.query_y.reshape(-1))
    assert torch.equal(episode_loss(out, ep), ce + out.loss_gmmn)  # the paper's objective, bit for bit
    f_q, _, _, _ = off.cascade(ep)
    want = oracle_distill_loss(out.logits, f_q, ep.query_y)
    assert out.loss_distill.item() == pytest.approx(want.item(), abs=1e-12)
    on = model(CascadeProtoConfig(use_lma=False, num_stages=3, distill_beta=0.5)).train()
    out = on(ep)
    ce = F.cross_entropy(out.logits.reshape(-1, 3), ep.query_y.reshape(-1))
    assert torch.allclose(episode_loss(out, ep), ce + out.loss_gmmn + 0.5 * out.loss_distill, atol=1e-12)
    on.zero_grad()
    out.loss_distill.backward()
    assert on.routing.w_g.weight.grad.abs().sum() > 0  # the loss reaches ADRM through M_eff
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in on.stages[0].parameters())
    with pytest.raises(ValueError, match="training mode"):
        episode_loss(EpisodeOutput(out.logits, out.loss_gmmn, None, 0.5), ep)


def test_dis7_configuration_and_cli():
    for bad in (-0.1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="distill_beta"):
            CascadeProtoConfig(distill_beta=bad)
    base = ["--dataset", "s3dis", "--data_path", "x", "--cvfold", "1", "--n_way", "2", "--k_shot", "1",
            "--use_lma", "false", "--num_stages", "4", "--stage_type", "vip", "--l2norm_point_proto", "true"]
    a0, a1 = train.parse_args(base), train.parse_args(base + ["--distill_beta", "1"])
    assert train.model_config(a0).distill_beta == 0.0 and train.model_config(a1).distill_beta == 1.0
    assert train.run_dir(a0).endswith("s3dis_S1_N2_K1_point_T4_vip")  # run_r2.sh's R0 directory
    assert train.run_dir(a1).endswith("s3dis_S1_N2_K1_point_T4_vip_distill1")  # and D29's
    saved = {k: v for k, v in train.comparable_args(a0).items() if k != "distill_beta"}  # older checkpoint
    assert train.resume_mismatch(saved, train.comparable_args(a0)) == []
    assert train.resume_mismatch(saved, train.comparable_args(a1)) == ["distill_beta"]


def test_dis8_oracle_rules_keep_absent_classes_and_set_the_norms():
    f_q, m = rand(1, 6, D, seed=8), rand(1, 3, D, seed=9)
    labels = torch.tensor([[0, 0, 1, 1, 1, 0]])  # class 2 absent: its prototype is untouched
    o, _ = oracle_directions(f_q, labels, 3)
    kept = torch.stack([m[0, 0].norm() * o[0, 0], m[0, 1].norm() * o[0, 1], m[0, 2]])
    assert torch.equal(r2.oracle_replaced(f_q, m, labels), torch.einsum("pd,cd->pc", f_q[0], kept).argmax(-1)[None])
    s = (m[0, 0].norm() + m[0, 1].norm()) / 2  # one common norm for the present classes
    unit = torch.stack([s * o[0, 0], s * o[0, 1], m[0, 2]])
    assert torch.equal(r2.oracle_replaced(f_q, m, labels, unit=True),
                       torch.einsum("pd,cd->pc", f_q[0], unit).argmax(-1)[None])
    logits = torch.einsum("bpd,bcd->bpc", f_q, m)
    terms = r2.cosine_terms(f_q, m, [m, o], labels, logits)
    assert terms["bg"][1] == 1 and terms["fg"][1] == 1 and terms["step1"][0] == pytest.approx(2.0)
    assert terms["logit_pair"][1] == 1 and terms["pair"][1] == 1 and terms["pair_step1"][0] == pytest.approx(1.0)
    want = pair_logit_cosine(logits, oracle_logits(f_q, labels, 3), labels)[0][0, 0, 1].item()
    assert terms["logit_pair"][0] == pytest.approx(want, abs=1e-12) and abs(want) < 0.999  # model vs teacher
    own = r2.cosine_terms(f_q, m, [m], labels, oracle_logits(f_q, labels, 3))
    assert own["logit_pair"][0] == pytest.approx(1.0)  # the teacher agrees with itself


def _draws(gain_fixed, ci_low, rand_gains, cos_r0=0.80, cos_d29=0.85, r0=0.70, vip=0.7536, fold=1, drop=0.0):
    out = []
    for draw, g in zip(r2.DRAWS, [gain_fixed] + list(rand_gains)):
        d29 = r0 + g / 100
        out.append({"draw": draw, "cvfold": fold, "test_classes": [6, 1, 9, 7, 2, 5],
                    "miou": {"r0": r0, "d29": d29, "vipseg": vip},
                    "cos": {"r0": {"logit_pair": cos_r0}, "d29": {"logit_pair": cos_d29},
                            "vipseg": {"logit_pair": 0.8}},
                    "class_iou": {"r0": [0.9] + [0.7] * 6, "d29": [0.9, 0.7 + drop / 100] + [0.7 + g / 100] * 5},
                    "paired": {"d29_vs_r0": {"gain": g, "ci_low": ci_low, "ci_high": g + 0.5},
                               **{f"{n}{o}_vs_{n}": {"gain": 8.0} for n in ("r0", "d29", "vipseg")
                                  for o in ("_oracle", "_oracle_unit")}}})
    return out


def test_dis9_rules():
    v = dict(r2.decide(_draws(1.4, 0.6, [1.1, 0.9, 1.3]), 1))
    assert "R2.1 go" in v and "S0" in v["R2.1 go"] and v["R2.0 reference"].endswith("worse; a gain over R0 is not a gain over VIP-Seg")
    assert dict(r2.decide(_draws(1.4, 0.6, [1.1, 0.9, 1.3], r0=0.74), 1))["R2.0 reference"].endswith("comparable")
    assert "R2.2 stop" in dict(r2.decide(_draws(0.4, -0.1, [1.1, 0.9, 1.3]), 1))
    assert "R2.2 stop" in dict(r2.decide(_draws(1.4, 0.6, [0.2, 0.3, 0.4]), 1))  # draws disagree with fixed100
    assert "R2.3 in between" in dict(r2.decide(_draws(0.8, 0.2, [1.1, 0.9, 1.3]), 1))
    assert "R2.3 in between" in dict(r2.decide(_draws(1.4, 0.6, [1.1, -0.1, 2.0]), 1))  # one draw negative
    assert "R2.3 in between" in dict(r2.decide(_draws(1.4, -0.1, [1.1, 0.9, 1.3]), 1))  # CI contains 0
    assert "R2.4 mechanism" in dict(r2.decide(_draws(1.4, 0.6, [1.1, 0.9, 1.3], cos_d29=0.79), 1))
    assert "R2.5 collapse watch" in dict(r2.decide(_draws(1.4, 0.6, [1.1, 0.9, 1.3], drop=-3.5), 1))
    s0 = dict(r2.decide(_draws(1.4, 0.6, [1.1, 0.9, 1.3], r0=0.72, fold=0), 0))
    assert s0["R2.1 go"].endswith("beats VIP-Seg's 72.20: yes")
    assert r2.decide(_draws(1.4, 0.6, [1.1, 0.9, 1.3])[:3], 1)[0][0] == "incomplete"
    assert r2.decide(_draws(1.4, 0.6, [1.1, 0.9, 1.3]), 0)[0][0] == "incomplete"  # wrong fold


def test_dis9_test_refuses_a_checkpoint_of_the_other_fold():
    from types import SimpleNamespace

    args = SimpleNamespace(checkpoint=["r0:ours:0:missing.pt"], cvfold=1, data_path="x", draws=["fixed100"],
                           max_episodes=None)
    with pytest.raises(ValueError, match="D-22"):
        r2.cmd_test(args, torch.device("cpu"))


class _Closing:
    """An OursRule that records `close()`; after it, a real hook-based rule would read nothing."""

    def __init__(self, m):
        self.rule, self.closed = r2.OursRule(m), 0

    def __call__(self, ep):
        assert not self.closed, "rule used after close()"
        return self.rule(ep)

    def close(self):
        self.closed += 1


def test_dis9_rules_survive_every_draw(monkeypatch):
    """The smoke run of 2026-09-23 caught score_draw closing VIP-Seg's hooks after the first draw."""
    import numpy as np

    import pipeline.episodes as episodes
    from tests.test_cascadeproto import CLASS_NAMES

    def items(seed):
        rng = np.random.default_rng(seed)
        return (rng.random((2, 1, 2048, 9)), rng.integers(0, 2, (2, 1, 2048)).astype(np.int32),
                rng.random((2, 2048, 9)), rng.integers(0, 3, (2, 2048)), np.array([3, 4]))

    class Draw(list):
        classes = np.array([3, 4, 5])

    monkeypatch.setattr(r2, "episodes_of", lambda draw, path, fold: (Draw([items(0), items(1)]), [3, 4, 5]))
    monkeypatch.setattr(episodes, "read_class_names", lambda path, name: CLASS_NAMES)

    rules = {"r0": _Closing(model(CascadeProtoConfig(use_lma=False, num_stages=2)).eval().float()),
             "d29": _Closing(model(CascadeProtoConfig(use_lma=False, num_stages=3)).eval().float())}
    for draw in r2.DRAWS[1:]:
        result, stacked = r2.score_draw(rules, draw, "x", 1, torch.device("cpu"))
        assert result["episodes"] == 2 and set(result["paired"]) == {
            "d29_vs_r0", "r0_oracle_vs_r0", "d29_oracle_vs_d29", "r0_oracle_unit_vs_r0", "d29_oracle_unit_vs_d29"}
    assert all(rule.closed == 0 for rule in rules.values())


@pytest.mark.parametrize("path", ["experiments/r2_distill_eval.py", "models/oracle_distill.py",
                                  "models/cascadeproto.py", "pipeline/model_api.py", "train.py"])
def test_dis10_parses_as_python_3_10(path):
    """The VM runs Python 3.10 (00 §5.2)."""
    ast.parse((REPO / path).read_text(encoding="utf-8"), feature_version=(3, 10))
