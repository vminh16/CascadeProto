"""TXT-1..13 (05 §3.8k): training-free text prior and P3's gap decomposition [DECISION D-31]. Beyond the paper, CPU."""

import ast
import math
import pathlib

import pytest
import torch
import torch.nn.functional as F

from experiments import p3_probe as p3
from models import text_prior as tp
from models.base_calibration import centred_unit

REPO = pathlib.Path(__file__).resolve().parents[1]
D, E = 8, 16


def rand(*shape, seed=0):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)


def test_txt1_prompt_sets():
    assert tp.prompts_for("door", "bare") == ["door"]
    assert tp.prompts_for("door", "template") == ["This point cloud represents the door."]
    assert len(tp.prompts_for("door", "ensemble")) == len(tp.ENSEMBLE) == 6
    assert tp.prompts_for("door", "descriptions")[1] == "a flat vertical panel in a wall that opens."
    assert len(tp.DESCRIPTIONS) == 12 and "clutter" not in tp.DESCRIPTIONS
    for bad in (("clutter", "descriptions"), ("door", "poem")):
        with pytest.raises(ValueError):
            tp.prompts_for(*bad)


def test_txt2_class_embeddings_average_unit_prompt_embeddings():
    table = {}

    def encode(prompts):
        return torch.stack([table.setdefault(p, rand(E, seed=len(table) + 1)) for p in prompts])

    e = tp.class_embeddings(["door", "wall"], "ensemble", encode)
    want = F.normalize(F.normalize(encode(tp.prompts_for("door", "ensemble")), dim=-1).mean(0), dim=0)
    assert e.shape == (2, E) and torch.allclose(e[0], want.float(), atol=1e-6)


def test_txt3_ridge_interpolates_the_base_anchors():
    e_all = F.normalize(rand(6, E, seed=1), dim=-1)
    base = F.normalize(rand(4, D, seed=2), dim=-1)
    t = tp.ridge_directions(e_all, [0, 1], [0, 1, 2, 3], base, lam=1e-9)  # the "novel" are base anchors
    assert torch.allclose(t, base[:2].float(), atol=1e-4)
    assert torch.allclose(tp.ridge_directions(e_all, [4, 5], [0, 1, 2, 3], base).norm(dim=-1),
                          torch.ones(2), atol=1e-6)


def test_txt4_retrieval_limits():
    e_all = F.normalize(rand(6, E, seed=3), dim=-1).float()
    base = F.normalize(rand(4, D, seed=4), dim=-1).float()
    nearest = int((e_all[5] @ e_all[:4].T).argmax())
    assert torch.allclose(tp.retrieval_directions(e_all, [5], [0, 1, 2, 3], base, 1e4)[0], base[nearest], atol=1e-5)
    assert torch.allclose(tp.retrieval_directions(e_all, [5], [0, 1, 2, 3], base, 0.0)[0],
                          F.normalize(base.mean(0), dim=0), atol=1e-6)


def test_txt5_text_logits():
    f, mu, t_hat = rand(2, 5, D, seed=5), rand(D, seed=6), F.normalize(rand(2, D, seed=7), dim=-1)
    assert torch.allclose(tp.text_logits(f, mu, t_hat), centred_unit(f, mu) @ t_hat.T)


def test_txt6_support_gamma():
    mu = torch.zeros(D, dtype=torch.float64)
    t_hat = torch.eye(D, dtype=torch.float64)[:2]  # way 0 -> channel 0, way 1 -> channel 1
    f_s = torch.zeros(2, 1, 6, D, dtype=torch.float64)
    f_s[0, 0, :, 0], f_s[1, 0, :, 1] = 1.0, 1.0
    y = torch.tensor([[[1, 1, 1, 0, 0, 0]], [[1, 1, 0, 0, 0, 0]]])
    gamma, acc = tp.support_gamma(f_s, y, mu, t_hat)
    assert acc.item() == 1.0 and gamma.item() == 1.0
    gamma, acc = tp.support_gamma(f_s, y, mu, t_hat.flip(0))  # the text swaps the ways
    assert acc.item() == 0.0 and gamma.item() == 0.0
    f_s[0, 0, 0, 0], f_s[0, 0, 0, 1] = 0.0, 1.0  # one of five points wrong: acc 0.8, gamma 0.6
    gamma, acc = tp.support_gamma(f_s, y, mu, t_hat)
    assert acc.item() == pytest.approx(0.8) and gamma.item() == pytest.approx(0.6)


def test_txt7_apply_prior():
    logits, t = rand(2, 5, 3, seed=8), rand(2, 5, 2, seed=9)
    w, g = torch.rand(2, 5, dtype=torch.float64), torch.tensor(0.7, dtype=torch.float64)
    assert torch.equal(tp.apply_prior(logits, t, 0.0, g, w), logits)
    out = tp.apply_prior(logits, t, 2.0, g, w)
    assert torch.equal(out[..., 0], logits[..., 0])  # background untouched
    assert torch.allclose(tp.apply_prior(logits, t + 5.0, 2.0, g, w), out)  # a shift common to the ways is ignored
    assert torch.allclose(out[..., 1:].sum(-1), logits[..., 1:].sum(-1))  # re-ranks the ways only
    assert torch.equal(tp.apply_prior(logits, t, 2.0, g, torch.zeros_like(w)), logits)
    up = (t[..., 0] > t[..., 1])
    assert ((out[..., 1] - logits[..., 1])[up] > 0).all()


def test_txt8_entropy_weight():
    assert torch.allclose(tp.entropy_weight(torch.zeros(1, 4, 3)), torch.ones(1, 4))
    w = tp.entropy_weight(torch.tensor([[[50.0, 0.0, 0.0]]]))
    assert w.item() < 1e-6


def test_txt9_unit_oracle_logits():
    f_q, m = rand(1, 6, D, seed=10).abs(), rand(1, 3, D, seed=11)
    labels = torch.tensor([[0, 0, 1, 1, 1, 0]])  # class 2 absent
    from models.oracle_distill import oracle_directions

    o, _ = oracle_directions(f_q, labels, 3)
    s = (m[0, 0].norm() + m[0, 1].norm()) / 2
    want = f_q[0] @ torch.stack([s * o[0, 0], s * o[0, 1], m[0, 2]]).T
    assert torch.allclose(tp.unit_oracle_logits(f_q, m, labels)[0], want)


def test_txt10_alignment_terms():
    t = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [5.0, 5.0]]])  # [1, 3, 2]
    logits = torch.zeros(1, 3, 3)
    oracle = torch.tensor([[[0.0, 2.0, 0.0], [0.0, 0.0, 2.0], [9.0, 9.0, -9.0]]])
    labels = torch.tensor([[1, 2, 0]])  # the background point never enters
    sab, saa, sbb = tp.alignment_terms(t, logits, oracle, labels)
    assert (sab, saa, sbb) == (4.0, 2.0, 8.0)  # a = [1, -1], b = [2, -2]: cos = 1


def test_txt11_boundary_mask():
    xyz = torch.cat([torch.rand(1, 20, 3), torch.rand(1, 20, 3) + 10.0], dim=1)  # two far clusters
    labels = torch.cat([torch.zeros(1, 20), torch.ones(1, 20)], dim=1).long()
    assert not tp.boundary_mask(xyz, labels, k=4).any()
    labels[0, 0] = 1  # one point of cluster A carries B's label
    b = tp.boundary_mask(xyz, labels, k=4)
    knn = torch.cdist(xyz, xyz)[0].topk(5, largest=False).indices[:, 1:]  # [40, 4], kNN is not symmetric
    want = torch.tensor([i == 0 or 0 in knn[i].tolist() for i in range(40)])
    want[0] = True  # point 0's own neighbours all carry the other label
    assert torch.equal(b[0], want) and not b[0, 20:].any()


def test_txt12_probe_arms_and_selection():
    assert len(p3.grid()) == 3 * 4 * 2 * 8
    names = [p3.name_of(a) for a in p3.grid()]
    res = {"miou": {"model": 0.70, **{n: 0.70 for n in names}}}
    res["miou"]["retrieval_t100_descriptions_entropy_k4"] = 0.712
    choice = p3.select_arm(res)
    assert choice["name"] == "retrieval_t100_descriptions_entropy_k4" and choice["valid_gain"] == pytest.approx(1.2)
    assert p3.sibling(choice["frozen"])["weight"] == "one"
    assert p3.name_of(p3.sibling(p3.sibling(choice["frozen"]))) == choice["name"]
    with pytest.raises(ValueError, match="S0 stays held out"):
        p3.cmd_select(type("A", (), dict(checkpoint="e1:ours:0:x.pt", data_path="x", bank_episodes=5,
                                         max_episodes=None))(), torch.device("cpu"))
    assert tp.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert math.isnan(tp.spearman([1, 2], [1, 2]))


def _test(frozen="ridge_bare_entropy_k2", acc=0.65, lo=0.02, gain=0.8, glo=0.3, og=1.2, sib_lo=0.1, drop=0.0,
          enrich=1.0, rho=0.0):
    return {"frozen": frozen, "sibling": frozen.replace("entropy", "one"),
            "valid_mechanism": {"text_query_acc": acc, "align": 0.1, "align_ci_low": lo, "align_ci_high": 0.2},
            "paired": {"frozen_vs_model": {"gain": gain, "ci_low": glo, "ci_high": gain + 0.3},
                       "oracle_gamma_vs_model": {"gain": og, "ci_low": og - 0.2, "ci_high": og + 0.2},
                       "frozen_vs_sibling": {"gain": 0.3, "ci_low": sib_lo, "ci_high": 0.5}},
            "class_iou": {"model": [0.9, 0.6, 0.6], frozen: [0.9, 0.6 + drop / 100, 0.62]},
            "part_a": {"support_rule": 60.0, "model": 73.2, "oracle_unit": 85.9, "head_recovery": 0.51,
                       "fixable_share_of_points": 0.12, "boundary_share_of_points": 0.2,
                       "boundary_share_of_fixable": 0.2 * enrich, "boundary_enrichment": enrich,
                       "fixable_share_by_support_tercile": [0.4, 0.3, 0.3], "spearman_oracle_gain_vs_support_fg": rho}}


def test_txt13_rules():
    v = dict(p3.decide(_test()))
    assert "P3.1 go" in v and v["P3.0 mechanism"].endswith("holds") and v["P3.4 entropy weight"].endswith(": claimable")
    assert "region-level" in v["A fixable points"]
    assert "P3.2 stop" in dict(p3.decide(_test(og=0.4)))
    v = dict(p3.decide(_test(acc=0.55)))
    assert "P3.3 in between" in v and "fails" in v["P3.0 mechanism"]  # a gain without the mechanism is no go
    assert "P3.3 in between" in dict(p3.decide(_test(lo=-0.01)))
    assert "P3.3 in between" in dict(p3.decide(_test(glo=-0.1)))
    assert dict(p3.decide(_test(sib_lo=-0.1)))["P3.4 entropy weight"].endswith("not claimable")
    assert "not claimable: selection froze" in dict(p3.decide(_test(frozen="ridge_bare_one_k2")))["P3.4 entropy weight"]
    assert "P3.5 collapse watch" in dict(p3.decide(_test(drop=-3.5)))
    assert "boundary-enriched" in dict(p3.decide(_test(enrich=1.6)))["A fixable points"]
    assert "between the bands" in dict(p3.decide(_test(enrich=1.3)))["A fixable points"]
    assert dict(p3.decide(_test(rho=-0.35)))["A support quality"].endswith("support quality matters")


def test_txt13_part_a_summary():
    eps = [{"support_fg": f, "fixable": 10, "points": 100, "fixable_boundary": 5, "boundary": 20, "errors_model": 20,
            "errors_model_boundary": 8, "fg_acc_model": (40, 50), "fg_acc_oracle": (40 + i, 50)}
           for i, f in enumerate([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])]
    a = p3.summarise_part_a(eps, {"support_rule": 0.6, "model": 0.7, "oracle_unit": 0.8})
    assert a["head_recovery"] == pytest.approx(0.5) and a["boundary_enrichment"] == pytest.approx(2.5)
    assert a["spearman_oracle_gain_vs_support_fg"] == pytest.approx(1.0)
    assert sum(a["fixable_share_by_support_tercile"]) == pytest.approx(1.0)


@pytest.mark.parametrize("path", ["models/text_prior.py", "experiments/p3_probe.py"])
def test_txt14_parses_as_python_3_10(path):
    ast.parse((REPO / path).read_text(encoding="utf-8"), feature_version=(3, 10))


def test_txt3b_ridge_uses_centred_embeddings():
    e_all = F.normalize(rand(6, E, seed=12) + 3.0, dim=-1)  # a strong common direction, as in CLIP prompts
    base = F.normalize(rand(4, D, seed=13), dim=-1)
    e = F.normalize(e_all - e_all.mean(0, keepdim=True), dim=-1)
    coef = torch.linalg.solve(e[:4] @ e[:4].T + 1e-3 * torch.eye(4, dtype=torch.float64), e[:4] @ e[4:].T)
    want = F.normalize(coef.T @ base, dim=-1).float()
    assert torch.allclose(tp.ridge_directions(e_all, [4, 5], [0, 1, 2, 3], base), want, atol=1e-6)


def test_txt5b_text_query_accuracy():
    t = torch.tensor([[[2.0, 0.0], [0.0, 2.0], [2.0, 0.0], [9.0, 0.0]]])  # [1, 4, 2]
    labels = torch.tensor([[1, 2, 2, 0]])  # column c-1 predicts label c; the background point is skipped
    assert tp.text_query_accuracy(t, labels) == (2, 3)
