"""BPC-1..9 (05 §3.8h): base-class calibration and its P1 probe [DECISION D-27]. Beyond the paper, all CPU."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from experiments import p1_bpc_probe as probe
from models.base_calibration import (BasePrototypeBank, auc_from_histogram, base_similarity, calibrate_background,
                                     occurrence_prototypes, separability_histogram)

D, P = 8, 32


def test_bpc1_occurrence_prototype_is_the_unit_mean_of_unit_features():
    g = torch.Generator().manual_seed(0)
    f = torch.rand(2, P, D, generator=g, dtype=torch.float64) * 5
    masks = torch.zeros(2, P, dtype=torch.float64)
    masks[0, :4] = 1
    out = occurrence_prototypes(f, masks)
    want = F.normalize(F.normalize(f[0, :4], dim=-1).sum(0), dim=0)
    assert torch.allclose(out[0], want, atol=1e-12)
    assert torch.equal(out[1], torch.zeros(D, dtype=torch.float64))  # empty mask


def test_bpc2_bank_maps_ways_to_classes_and_skips_empty_occurrences():
    g = torch.Generator().manual_seed(1)
    f_s = torch.rand(2, 1, P, D, generator=g, dtype=torch.float64)
    f_q = torch.rand(2, P, D, generator=g, dtype=torch.float64)
    support_y = torch.zeros(2, 1, P)
    support_y[0, 0, :5] = 1
    support_y[1, 0, 5:9] = 1
    query_y = torch.zeros(2, P, dtype=torch.long)
    query_y[0, :3] = 1  # way 0 present in query 0 only; way 1 absent from the queries
    bank = BasePrototypeBank(D)
    bank.add_episode(f_s, support_y, f_q, query_y, np.array([7, 3]))
    assert bank.counts == {7: 2, 3: 1}
    sup7 = occurrence_prototypes(f_s[0], support_y[0])[0]
    qry7 = occurrence_prototypes(f_q[:1], (query_y[:1] == 1).double())[0]
    protos = bank.prototypes()
    assert torch.allclose(protos[7], F.normalize(sup7 + qry7, dim=0), atol=1e-12)
    assert protos[3].norm().item() == pytest.approx(1.0)
    with pytest.raises(RuntimeError, match="fewer than 2"):
        bank.prototypes(min_count=2)


def test_bpc3_omega_zero_is_the_model_and_only_the_background_can_rise():
    g = torch.Generator().manual_seed(2)
    f_q = torch.rand(2, P, D, generator=g, dtype=torch.float64)
    prior = torch.randn(2, 3, D, generator=g, dtype=torch.float64)
    base = F.normalize(torch.randn(4, D, generator=g, dtype=torch.float64), dim=-1)
    logits = torch.einsum("bpd,bcd->bpc", f_q, prior)
    assert calibrate_background(f_q, prior, logits, base, 0.0) is logits
    out = calibrate_background(f_q, prior, logits, base, 1.0)
    assert torch.equal(out[..., 1:], logits[..., 1:])
    assert (out[..., 0] >= logits[..., 0]).all()


def test_bpc4_base_logit_uses_the_foreground_norm():
    """A point on a base direction scores s <f, b> for the background, s = mean foreground prototype norm."""
    b = F.normalize(torch.ones(1, D, dtype=torch.float64), dim=-1)
    f_q = b.expand(1, 1, D).clone()  # [1, 1, D], one point exactly on the base direction
    fg = torch.zeros(D, dtype=torch.float64)
    fg[0] = 1.0
    prior = torch.stack([100.0 * -b[0], 3.0 * fg, 5.0 * fg]).unsqueeze(0)  # huge background norm, ignored by s
    logits = torch.einsum("bpd,bcd->bpc", f_q, prior)
    out = calibrate_background(f_q, prior, logits, b, 1.0)
    assert out[0, 0, 0].item() == pytest.approx(4.0 * (f_q[0, 0] @ b[0]).item())  # s = (3 + 5) / 2
    assert out[0, 0].argmax().item() == 0 and logits[0, 0].argmax().item() == 2  # the point turns background


def test_bpc5_similarity_and_auc():
    base = torch.eye(D, dtype=torch.float64)[:2]
    f_q = torch.eye(D, dtype=torch.float64)[:3].unsqueeze(0) * 7  # cos 1, 1, 0 to the base
    assert base_similarity(f_q, base).tolist() == [[1.0, 1.0, 0.0]]
    rng = np.random.default_rng(0)
    pos, neg = torch.tensor(rng.normal(0.3, 0.2, 400)), torch.tensor(rng.normal(0.0, 0.2, 500))
    g = torch.cat([pos, neg]).clamp(-1, 1)
    is_pos = torch.arange(900) < 400
    hist = separability_histogram(g, is_pos, ~is_pos)
    q = ((g + 1) / 2 * 199).round()  # the histogram's own binning
    brute = ((q[:400, None] > q[None, 400:]).double() + 0.5 * (q[:400, None] == q[None, 400:]).double()).mean()
    assert auc_from_histogram(hist) == pytest.approx(brute.item(), abs=1e-12)
    assert auc_from_histogram(torch.zeros(2, 200)) is None


def test_bpc6_selection_picks_the_best_mean_omega():
    results = [{"miou": {"model": 0.5} | {probe.arm(w): 0.5 for w in probe.OMEGAS}} for _ in range(2)]
    results[0]["miou"][probe.arm(0.9)] = 0.52
    results[1]["miou"][probe.arm(1.2)] = 0.515
    assert probe.select_omega(results)["omega"] == 0.9


def _t(name, kind, fold, gain, low, auc=0.75, frozen="bpc_w1"):
    return {"checkpoint": {"name": name, "kind": kind, "fold": fold}, "cvfold": fold, "frozen": frozen,
            "paired": {"frozen_vs_model": {"gain": gain, "ci_low": low, "ci_high": gain + 0.3}},
            "separability": {"auc_false_fg_vs_true_fg": auc},
            "class_iou": {"model": [0.9, 0.5, 0.5], frozen: [0.9, 0.5 + gain / 100, 0.46 if gain > 1 else 0.5]}}


def test_bpc7_rules():
    go = [_t("v1", "vipseg", 1, 0.8, 0.4), _t("o1", "ours", 1, 0.6, 0.2),
          _t("v0", "vipseg", 0, 0.7, 0.3), _t("o0", "ours", 0, 0.5, 0.1)]
    v = dict(probe.decide(go))
    assert "P1.1 go" in v and v["P1.1 gain v0"].endswith("above 0") and v["P1.4 base signal"].startswith("AUC")
    assert "justified" in v["P1.4 base signal"]
    stop = [_t("v1", "vipseg", 1, 0.1, -0.1, auc=0.55), _t("o1", "ours", 1, -0.2, -0.4),
            _t("v0", "vipseg", 0, -0.5, -0.8, auc=0.52), _t("o0", "ours", 0, 0.0, -0.1)]
    v = dict(probe.decide(stop))
    assert "P1.2 stop (training-free form)" in v and "drop D-27" in v["P1.4 base signal"]
    assert v["P1.1 gain v0"].endswith("below 0") and v["P1.1 gain o0"].endswith("contains 0")
    mixed = go[:1] + [_t("o1", "ours", 1, 0.1, -0.1, auc=0.65)] + go[2:]
    v = dict(probe.decide(mixed))
    assert "P1.3 in between" in v
    wide = go[:3] + [_t("o0", "ours", 0, 0.6, -0.1)]  # large enough but its CI contains 0: no go
    assert "P1.1 go" not in dict(probe.decide(wide))
    collapse = [_t("v1", "vipseg", 1, 1.5, 1.0)] + go[1:]
    assert "P1.5 collapse watch v1" in dict(probe.decide(collapse))
    assert probe.decide(go[:3])[0][0] == "incomplete"


def test_bpc8_selection_refuses_s0_checkpoints():
    args = SimpleNamespace(checkpoint=["v0:vipseg:0:missing.pt"], data_path="x", max_episodes=None,
                           bank_episodes=10)
    with pytest.raises(ValueError, match="S0 stays held out"):
        probe.cmd_select(args, torch.device("cpu"))


def test_bpc9_bank_cache_is_keyed_by_episode_count(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "OUT_DIR", str(tmp_path))
    calls = []

    def fake_build(rule, data_path, fold, device, n):
        calls.append(n)
        return torch.eye(2, D), {"classes": [0, 1], "occurrences": {}, "episodes": n}

    monkeypatch.setattr(probe, "build_bank", fake_build)
    ck = SimpleNamespace(name="v1", fold=1)
    probe.load_or_build_bank(ck, None, "x", "cpu", 5)
    probe.load_or_build_bank(ck, None, "x", "cpu", 5)  # cached
    _, info = probe.load_or_build_bank(ck, None, "x", "cpu", 1000)  # a smoke bank never stands in
    assert calls == [5, 1000] and info["episodes"] == 1000
