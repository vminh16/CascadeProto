"""BPC-1..10 (05 §3.8h): base-class calibration and its P1 probe [DECISION D-27]. Beyond the paper, all CPU."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from experiments import p1_bpc_probe as probe
from models.base_calibration import (BasePrototypeBank, auc_from_histogram, base_margin, calibrate_background,
                                     centred_unit, occurrence_prototypes, separability_histogram,
                                     support_prototypes)

D, P = 8, 32


def rand(*shape, seed=0):
    return torch.rand(*shape, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)


def test_bpc1_occurrence_prototype_is_the_unit_mean_of_cl2n_features():
    f, mu = rand(2, P, D) * 5, rand(D, seed=1)
    masks = torch.zeros(2, P, dtype=torch.float64)
    masks[0, :4] = 1
    out = occurrence_prototypes(f, masks, mu)
    want = F.normalize(F.normalize(f[0, :4] - mu, dim=-1).sum(0), dim=0)
    assert torch.allclose(out[0], want, atol=1e-12)
    assert torch.equal(out[1], torch.zeros(D, dtype=torch.float64))  # empty mask
    assert torch.allclose(centred_unit(f, mu).norm(dim=-1), torch.ones(2, P, dtype=torch.float64))


def test_bpc2_bank_centre_then_prototypes():
    f_s, f_q = rand(2, 1, P, D, seed=2), rand(2, P, D, seed=3)
    support_y = torch.zeros(2, 1, P)
    support_y[0, 0, :5] = 1
    support_y[1, 0, 5:9] = 1
    query_y = torch.zeros(2, P, dtype=torch.long)
    query_y[0, :3] = 1  # way 0 present in query 0 only; way 1 absent from the queries
    bank = BasePrototypeBank(D)
    with pytest.raises(RuntimeError, match="freeze_centre"):
        bank.add_episode(f_s, support_y, f_q, query_y, np.array([7, 3]))
    bank.add_centre(f_s, f_q)
    mu = bank.freeze_centre()
    assert torch.allclose(mu, torch.cat([f_s.reshape(-1, D), f_q.reshape(-1, D)]).mean(0), atol=1e-12)
    bank.add_episode(f_s, support_y, f_q, query_y, np.array([7, 3]))
    assert bank.counts == {7: 2, 3: 1}
    sup7 = occurrence_prototypes(f_s[0], support_y[0], mu)[0]
    qry7 = occurrence_prototypes(f_q[:1], (query_y[:1] == 1).double(), mu)[0]
    assert torch.allclose(bank.prototypes()[7], F.normalize(sup7 + qry7, dim=0), atol=1e-12)
    with pytest.raises(RuntimeError, match="fewer than 2"):
        bank.prototypes(min_count=2)
    assert 0 < bank.common_cos_sum / bank.common_cos_count <= 1  # positive features share a component


def test_bpc3_support_prototypes_are_built_like_the_bank():
    f_s, mu = rand(2, 3, P, D, seed=4), rand(D, seed=5)
    support_y = (rand(2, 3, P, seed=6) > 0.5).double()
    u = support_prototypes(f_s, support_y, mu)
    want = F.normalize(occurrence_prototypes(f_s[1], support_y[1], mu).sum(0), dim=0)
    assert u.shape == (2, D) and torch.allclose(u[1], want, atol=1e-12)


def test_bpc4_margin_is_base_minus_own_class_and_undefined_on_background():
    mu = torch.zeros(D, dtype=torch.float64)
    e = torch.eye(D, dtype=torch.float64)
    base, support = e[:1], e[1:3]  # one base direction, two support classes
    f_q = torch.stack([e[0] + 0.1 * e[1], e[1], e[2] + 0.5 * e[0]]).unsqueeze(0)  # [1, 3, D]
    logits = torch.tensor([[[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.5]]], dtype=torch.float64)
    m = base_margin(f_q, logits, support, base, mu)
    n0 = f_q[0, 0] / f_q[0, 0].norm()
    assert m[0, 0].item() == pytest.approx((n0[0] - n0[1]).item())  # predicted class 1 -> u_1 = e1
    assert m[0, 1].item() == pytest.approx(-1.0)
    assert m[0, 2].item() == float("-inf")  # predicted background


def test_bpc5_calibration_only_moves_foreground_to_background():
    logits = rand(2, P, 3, seed=7)
    margin = rand(2, P, seed=8) - 0.5
    margin[logits.argmax(-1) == 0] = float("-inf")
    assert torch.equal(calibrate_background(logits, margin, float("inf")), logits)
    out = calibrate_background(logits, margin, 0.1)
    flip = margin > 0.1
    assert (out.argmax(-1)[flip] == 0).all()
    assert torch.equal(out.argmax(-1)[~flip], logits.argmax(-1)[~flip])
    assert torch.equal(out[..., 1:], logits[..., 1:])
    far = torch.tensor([[[0.0, 50.0, 3.0]]], dtype=torch.float64)  # foreground far above the background
    assert calibrate_background(far, torch.tensor([[1.0]], dtype=torch.float64), 0.0).argmax(-1).item() == 0


def test_bpc6_histogram_auc_equals_pairwise_auc():
    rng = np.random.default_rng(0)
    pos, neg = torch.tensor(rng.normal(0.3, 0.3, 400)), torch.tensor(rng.normal(0.0, 0.3, 500))
    g = torch.cat([pos, neg])
    is_pos = torch.arange(900) < 400
    hist = separability_histogram(g, is_pos, ~is_pos)
    q = ((g.clamp(-2, 2) + 2) / 4 * 199).round()  # the histogram's own binning
    brute = ((q[:400, None] > q[None, 400:]).double() + 0.5 * (q[:400, None] == q[None, 400:]).double()).mean()
    assert auc_from_histogram(hist) == pytest.approx(brute.item(), abs=1e-12)
    assert auc_from_histogram(torch.zeros(2, 200)) is None


def test_bpc7_selection_picks_the_best_mean_delta():
    results = [{"miou": {"model": 0.5} | {probe.arm(d): 0.5 for d in probe.DELTAS}} for _ in range(2)]
    results[0]["miou"][probe.arm(0.05)] = 0.52
    results[1]["miou"][probe.arm(0.2)] = 0.515
    assert probe.select_delta(results)["delta"] == 0.05


def _t(name, kind, fold, gain, low, auc=0.75, frozen="bpc_d0"):
    return {"checkpoint": {"name": name, "kind": kind, "fold": fold}, "cvfold": fold, "frozen": frozen,
            "paired": {"frozen_vs_model": {"gain": gain, "ci_low": low, "ci_high": gain + 0.3}},
            "separability": {"auc_false_fg_vs_true_fg": auc},
            "class_iou": {"model": [0.9, 0.5, 0.5], frozen: [0.9, 0.5 + gain / 100, 0.46 if gain > 1 else 0.5]}}


def test_bpc8_rules():
    go = [_t("v1", "vipseg", 1, 0.8, 0.4), _t("o1", "ours", 1, 0.6, 0.2),
          _t("v0", "vipseg", 0, 0.7, 0.3), _t("o0", "ours", 0, 0.5, 0.1)]
    v = dict(probe.decide(go))
    assert "P1.1 go" in v and v["P1.1 gain v0"].endswith("above 0") and "justified" in v["P1.4 base signal"]
    stop = [_t("v1", "vipseg", 1, 0.1, -0.1, auc=0.55), _t("o1", "ours", 1, -0.2, -0.4),
            _t("v0", "vipseg", 0, -0.5, -0.8, auc=0.52), _t("o0", "ours", 0, 0.0, -0.1)]
    v = dict(probe.decide(stop))
    assert "P1.2 stop (training-free form)" in v and "drop D-27" in v["P1.4 base signal"]
    assert v["P1.1 gain v0"].endswith("below 0") and v["P1.1 gain o0"].endswith("contains 0")
    assert "P1.3 in between" in dict(probe.decide(go[:1] + [_t("o1", "ours", 1, 0.1, -0.1, auc=0.65)] + go[2:]))
    wide = go[:3] + [_t("o0", "ours", 0, 0.6, -0.1)]  # large enough but its CI contains 0: no go
    assert "P1.1 go" not in dict(probe.decide(wide))
    assert "P1.5 collapse watch v1" in dict(probe.decide([_t("v1", "vipseg", 1, 1.5, 1.0)] + go[1:]))
    assert probe.decide(go[:3])[0][0] == "incomplete"


def test_bpc9_selection_refuses_s0_checkpoints():
    args = SimpleNamespace(checkpoint=["v0:vipseg:0:missing.pt"], data_path="x", max_episodes=None,
                           bank_episodes=10)
    with pytest.raises(ValueError, match="S0 stays held out"):
        probe.cmd_select(args, torch.device("cpu"))


def test_bpc10_bank_cache_is_keyed_by_episode_count(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "OUT_DIR", str(tmp_path))
    calls = []

    def fake_build(rule, data_path, fold, device, n):
        calls.append(n)
        return torch.zeros(D), torch.eye(2, D), {"classes": [0, 1], "occurrences": {}, "episodes": n}

    monkeypatch.setattr(probe, "build_bank", fake_build)
    ck = SimpleNamespace(name="v1", fold=1)
    probe.load_or_build_bank(ck, None, "x", "cpu", 5)
    probe.load_or_build_bank(ck, None, "x", "cpu", 5)  # cached
    mu, base, info = probe.load_or_build_bank(ck, None, "x", "cpu", 1000)  # a smoke bank never stands in
    assert calls == [5, 1000] and info["episodes"] == 1000 and mu.shape == (D,)
