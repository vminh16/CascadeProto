"""FUS-1..8 (05 §3.8i): base-margin-filtered EM refinement and its P2 probe [DECISION D-28]. Beyond the paper, CPU."""

import ast
import pathlib
from types import SimpleNamespace

import pytest
import torch

from experiments import p2_fused_probe as probe
from models.base_calibration import base_margin, top_fraction_flips
from models.transductive import em_refine, em_step

D, P = 8, 64
REPO = pathlib.Path(__file__).resolve().parents[1]


def rand(*shape, seed=0):
    return torch.rand(*shape, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)


def rule(seed=0):
    f_q, prior = rand(2, P, D, seed=seed), torch.randn(2, 3, D, generator=torch.Generator().manual_seed(seed),
                                                       dtype=torch.float64)
    return f_q, prior, torch.einsum("bpd,bcd->bpc", f_q, prior)


def test_fus1_fg_keep_filters_the_foreground_m_step_only():
    f_q, prior, logits = rule()
    keep_all = torch.ones(2, P, dtype=torch.bool)
    assert torch.equal(em_step(f_q, prior, logits, 2.0, "none", fg_keep=keep_all)[0],
                       em_step(f_q, prior, logits, 2.0, "none")[0])
    mu, _ = em_step(f_q, prior, logits, 2.0, "none", fg_keep=torch.zeros(2, P, dtype=torch.bool))
    assert torch.allclose(mu[:, 1:], prior[:, 1:], atol=1e-12)  # no foreground point left: unchanged
    assert not torch.allclose(mu[:, 0], prior[:, 0])  # the background M-step still runs


def test_fus2_keep_mask():
    f_q, _, logits = rule(1)
    support, base, mu = rand(2, D, seed=2), torch.nn.functional.normalize(rand(3, D, seed=3), dim=-1), rand(D, seed=4)
    assert probe.fg_keep_mask(f_q, logits, support, base, mu, 0.0).all()
    want = ~top_fraction_flips(base_margin(f_q, logits, support, base, mu), 0.2)
    assert torch.equal(probe.fg_keep_mask(f_q, logits, support, base, mu, 0.2), want)


def test_fus3_false_share_terms():
    wr = torch.zeros(1, 4, 3, dtype=torch.float64)
    wr[0, 0, 1], wr[0, 1, 1], wr[0, 2, 2], wr[0, 3, 0] = 1.0, 0.5, 2.0, 7.0  # background mass is ignored
    labels = torch.tensor([[1, 0, 2, 0]])
    wrong, total = probe.false_share_terms(wr, labels)
    assert (wrong, total) == (0.5, 3.5)


def test_fus4_unfiltered_arm_is_p0():
    f_q, prior, logits = rule(5)
    support, base, mu = rand(2, D, seed=6), torch.nn.functional.normalize(rand(3, D, seed=7), dim=-1), rand(D, seed=8)
    labels = torch.randint(0, 3, (2, P), generator=torch.Generator().manual_seed(9))
    arms = [dict(weight="ssp", kappa=2.0, steps=2, r=0.0), dict(weight="entropy", kappa=1.0, steps=1, r=0.2)]
    preds, eps = probe.run_arms(f_q, prior, logits, support, base, mu, arms, labels)
    assert torch.equal(preds["ssp_k2_T2_r0"], em_refine(f_q, prior, 2, 2.0, "ssp").argmax(-1))
    assert set(eps) == {"ssp_r0", "entropy_r0.2"} and all(len(v) == 2 for v in eps.values())


def test_fus5_selection_uses_the_vipseg_checkpoint_only():
    names = [probe.arm_name(a) for a in probe.grid_arms()]
    vip = {"checkpoint": {"kind": "vipseg"}, "miou": {"model": 0.7} | {n: 0.7 for n in names}}
    ours = {"checkpoint": {"kind": "ours"}, "miou": {"model": 0.5} | {n: 0.5 for n in names}}
    vip["miou"]["ssp_k1_T1_r0.2"] = 0.71
    ours["miou"]["entropy_k8_T2_r0.3"] = 0.60  # a larger gain on ours must not decide
    assert probe.select_arm([vip, ours])["name"] == "ssp_k1_T1_r0.2"
    with pytest.raises(ValueError):
        probe.select_arm([ours])
    assert len(probe.grid_arms()) == 2 * 5 * 4 * 2


def _t(name, kind, fold, gain, low, ratio=0.6, frozen="ssp_k1_T1_r0.2", u_low=0.1):
    return {"checkpoint": {"name": name, "kind": kind}, "cvfold": fold, "frozen": frozen,
            "false_share": {"ssp_r0": 0.10, "ssp_r0.2": 0.10 * ratio},
            "paired": {"frozen_vs_model": {"gain": gain, "ci_low": low, "ci_high": gain + 0.3},
                       "frozen_vs_unfiltered": {"gain": 0.4, "ci_low": u_low, "ci_high": 0.7}},
            "class_iou": {"model": [0.9, 0.5, 0.5], frozen: [0.9, 0.5 + gain / 100, 0.46 if gain > 1 else 0.5]}}


def test_fus6_rules():
    go = [_t("v1", "vipseg", 1, 0.8, 0.4), _t("v0", "vipseg", 0, 0.6, 0.2), _t("o1", "ours", 1, -0.1, -0.2)]
    v = dict(probe.decide(go))
    assert "P2.1 go" in v and "cleans" in v["P2.0 mechanism"] and v["P2.4 filter on held-out S0"].endswith("claimable")
    dirty = [_t("v1", "vipseg", 1, 0.8, 0.4, ratio=0.9), _t("v0", "vipseg", 0, 0.6, 0.2, ratio=0.9)]
    v = dict(probe.decide(dirty))
    assert "P2.2 stop" in v and "does not" in v["P2.0 mechanism"]  # gains without the mechanism: stop
    mid = [_t("v1", "vipseg", 1, 0.8, 0.4), _t("v0", "vipseg", 0, 0.3, -0.1, u_low=-0.1)]
    v = dict(probe.decide(mid))
    assert "P2.3 in between" in v and v["P2.4 filter on held-out S0"].endswith("not claimable")
    unfiltered = [_t("v1", "vipseg", 1, 0.8, 0.4, frozen="ssp_k1_T1_r0"),
                  _t("v0", "vipseg", 0, 0.6, 0.2, frozen="ssp_k1_T1_r0")]
    assert "adds nothing" in dict(probe.decide(unfiltered))["P2.0 mechanism"]
    assert "P2.5 collapse watch v1" in dict(probe.decide([_t("v1", "vipseg", 1, 1.5, 1.0)] + go[1:]))
    assert probe.decide(go[2:])[0][0] == "incomplete"


def test_fus7_selection_refuses_s0_checkpoints():
    args = SimpleNamespace(checkpoint=["v0:vipseg:0:missing.pt"], data_path="x", max_episodes=None, bank_episodes=10)
    with pytest.raises(ValueError, match="S0 stays held out"):
        probe.cmd_select(args, torch.device("cpu"))


@pytest.mark.parametrize("path", ["experiments/p0_em_probe.py", "experiments/p1_bpc_probe.py",
                                  "experiments/p2_fused_probe.py", "models/transductive.py",
                                  "models/base_calibration.py"])
def test_fus8_probes_parse_as_python_3_10(path):
    """The VM runs Python 3.10 (00 §5.2): no syntax that needs 3.12, such as reused quotes in f-strings."""
    ast.parse((REPO / path).read_text(encoding="utf-8"), feature_version=(3, 10))
