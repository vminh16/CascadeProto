"""EM-1..14 (05 §3.8g): query-side entropy-weighted EM refinement and its P0 probe [DECISION D-26].

Beyond the paper. All CPU: the probe's scoring-rule reader is checked on a stand-in that follows
VIP-Seg's call pattern [VIPSEG models/vipseg.py:137-174], because `models.vipseg` needs pointnet2_ops.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments import p0_em_probe as probe
from models.transductive import (SSP_BG_THRESHOLD, SSP_FG_THRESHOLD, em_refine, em_step, entropy_diagnostics,
                                 normalized_entropy, responsibilities)
from pipeline.metrics_alt import alternative_metrics

D, P = 16, 64


def rule(bq=2, n=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    f_q = torch.randn(bq, P, D, generator=g, dtype=torch.float64).relu()  # post-ReLU features
    prior = torch.randn(bq, n + 1, D, generator=g, dtype=torch.float64)
    return f_q, prior


def test_em1_zero_kappa_and_zero_steps_are_the_model():
    f_q, prior = rule()
    base = torch.einsum("bpd,bcd->bpc", f_q, prior)
    for w in ("entropy", "none", "ssp"):
        assert torch.allclose(em_refine(f_q, prior, 2, 0.0, w), base, atol=1e-12)
    assert torch.equal(em_refine(f_q, prior, 0, 4.0, "entropy"), base)


def test_em2_prototype_norm_is_kept():
    f_q, prior = rule()
    logits = torch.einsum("bpd,bcd->bpc", f_q, prior)
    mu, _ = em_step(f_q, prior, logits, 2.0, "none")
    assert torch.allclose(mu.norm(dim=-1), prior.norm(dim=-1), atol=1e-12)
    assert not torch.allclose(mu, prior)


def test_em3_class_without_confident_mass_is_unchanged():
    f_q, prior = rule()
    logits = torch.zeros(2, P, 3, dtype=torch.float64)  # uniform posterior: entropy weight 0, no SSP point
    for w in ("entropy", "ssp"):
        mu, _ = em_step(f_q, prior, logits, 16.0, w)
        assert torch.allclose(mu, prior, atol=1e-12)


def test_em4_oracle_replace_gives_the_query_class_mean_direction():
    f_q, prior = rule()
    labels = torch.randint(0, 3, (2, P), generator=torch.Generator().manual_seed(1))
    logits = torch.einsum("bpd,bcd->bpc", f_q, prior)
    mu, _ = em_step(f_q, prior, logits, probe.ORACLE_REPLACE, "oracle", labels)
    for b in range(2):
        for c in range(3):
            mean = F.normalize(f_q[b][labels[b] == c], dim=-1).sum(0)
            assert torch.allclose(F.normalize(mu[b, c], dim=0), F.normalize(mean, dim=0), atol=1e-3)
    with pytest.raises(ValueError):
        em_step(f_q, prior, logits, 1.0, "oracle")


def test_em5_entropy_weight_range():
    confident = torch.tensor([[[30.0, 0.0, 0.0]]])
    uniform = torch.zeros(1, 1, 3)
    assert normalized_entropy(torch.softmax(uniform, -1)).item() == pytest.approx(1.0)
    assert normalized_entropy(torch.softmax(confident, -1)).item() == pytest.approx(0.0, abs=1e-9)
    assert responsibilities(uniform, "entropy").abs().max().item() == pytest.approx(0.0, abs=1e-7)
    assert responsibilities(confident, "entropy")[0, 0, 0].item() == pytest.approx(1.0, abs=1e-6)


def test_em6_ssp_thresholds_are_per_class():
    p = 0.65  # between SSP's background (0.6) and foreground (0.7) thresholds [SSP §3.2]
    assert SSP_BG_THRESHOLD < p < SSP_FG_THRESHOLD
    rest = (1 - p) / 2
    bg = torch.log(torch.tensor([[[p, rest, rest]]]))
    fg = torch.log(torch.tensor([[[rest, p, rest]]]))
    assert responsibilities(bg, "ssp")[0, 0].tolist() == pytest.approx([1.0, 0.0, 0.0])
    assert responsibilities(fg, "ssp")[0, 0].tolist() == pytest.approx([0.0, 0.0, 0.0])


def test_em7_update_is_the_vmf_map_direction_and_scales_with_mass():
    f_q, prior = rule()
    logits = torch.einsum("bpd,bcd->bpc", f_q, prior)
    wr = responsibilities(logits, "none")
    mu, _ = em_step(f_q, prior, logits, 2.0, "none")
    u = torch.einsum("bpc,bpd->bcd", wr, F.normalize(f_q, dim=-1)) / P  # hand-written M-step
    n = prior.norm(dim=-1, keepdim=True)
    assert torch.allclose(mu, n * F.normalize(prior / n + 2.0 * u, dim=-1), atol=1e-12)
    # one class holds 1 point of 64, the other 63: the small one must move much less
    g = torch.Generator().manual_seed(2)
    f1 = torch.rand(1, P, D, generator=g, dtype=torch.float64)
    p1 = F.normalize(torch.randn(1, 2, D, generator=g, dtype=torch.float64), dim=-1)
    lab = torch.zeros(1, P, dtype=torch.long)
    lab[0, 0] = 1
    mu1, _ = em_step(f1, p1, torch.zeros(1, P, 2, dtype=torch.float64), 4.0, "oracle", lab)
    moved = 1 - (F.normalize(mu1, dim=-1) * p1).sum(-1)  # [1, 2], 1 - cosine to the prior
    assert moved[0, 1] < 0.1 * moved[0, 0]


def test_em8_refinement_recovers_a_shifted_query_class():
    """Mechanism: support prototypes off by a rotation; the query's own confident mean pulls them back."""
    g = torch.Generator().manual_seed(3)
    centres = F.normalize(torch.randn(3, D, generator=g, dtype=torch.float64), dim=-1) * 3
    labels = torch.randint(0, 3, (1, 512), generator=g)
    f_q = (centres[labels] + 0.8 * torch.randn(1, 512, D, generator=g, dtype=torch.float64))
    prior = (centres + 1.6 * torch.randn(3, D, generator=g, dtype=torch.float64)).unsqueeze(0)
    base = torch.einsum("bpd,bcd->bpc", f_q, prior).argmax(-1)
    refined = em_refine(f_q, prior, 2, 8.0, "entropy").argmax(-1)
    assert (refined == labels).double().mean() > (base == labels).double().mean() + 0.05


def test_em9_entropy_diagnostics_bins():
    logits = torch.tensor([[[0.0, 0.0, 0.0], [30.0, 0.0, 0.0], [0.0, 30.0, 0.0]]])
    labels = torch.tensor([[0, 0, 2]])
    d = entropy_diagnostics(logits, labels)
    assert d["count"].tolist() == [1, 0, 2]
    assert d["correct"].tolist() == [1.0, 0.0, 1.0]  # uniform argmax is 0 (correct), last point wrong


class _Step(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, query, supports, prototype):
        return self.scale * prototype + query.mean(dim=1)[:, None, :] * 0.01


class _FakeVIPSeg(nn.Module):
    """VIP-Seg's call pattern: fc on support then query, alternating steps with the outer residual on
    odd steps, logits per step, gated sum [VIPSEG models/vipseg.py:92-174]. Test stand-in only."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Conv1d(4, D, 1)
        self.vip_module = nn.ModuleList([_Step(0.5), _Step(2.0), _Step(-1.0), _Step(1.5)])
        self.gating_network = nn.Sequential(nn.Linear(D, 4), nn.Softmax(dim=-1))

    def forward(self, support_x, query_x, prototype):
        support_feat = self.fc(support_x).permute(0, 2, 1)
        query_feat = self.fc(query_x).permute(0, 2, 1)  # [B, P, D]
        memory, all_logits = prototype, []
        for step, module in enumerate(self.vip_module):
            out = module(query_feat, support_feat, memory)
            memory = out + memory if step % 2 == 1 else out
            all_logits.append(query_feat @ memory.transpose(1, 2))
        w = self.gating_network(query_feat.mean(dim=1))  # [B, T]
        return (torch.stack(all_logits, 1) * w[:, :, None, None]).sum(1)


def fake_baseline():
    g = torch.Generator().manual_seed(5)
    model = _FakeVIPSeg().double()
    inputs = (torch.randn(2, 4, P, generator=g, dtype=torch.float64),
              torch.randn(2, 4, P, generator=g, dtype=torch.float64),
              torch.randn(2, 3, D, generator=g, dtype=torch.float64))
    return model, inputs, (lambda episode: SimpleNamespace(logits=model(*inputs)))


def test_em10_vipseg_rule_reproduces_the_gated_logits():
    model, inputs, call = fake_baseline()

    class Baseline:
        def __init__(self):
            self.model = model

        def __call__(self, episode):
            return call(episode)

    reader = probe.VIPSegScoringRule(Baseline())
    f_q, m_eff, logits = reader(None)
    probe.check_identity(f_q, m_eff, logits)
    reader.close()
    assert not model.fc._forward_hooks  # hooks removed, the model is left as it was
    with pytest.raises(RuntimeError):  # dropping the outer residual must break the identity
        stages = [o for o in reader.step_out]
        wrong = torch.einsum("bt,tbcd->bcd", reader.gate[0], torch.stack(stages))
        probe.check_identity(f_q, wrong, logits)


def test_em11_count_metric_is_vipsegs():
    rng = np.random.default_rng(0)
    test_classes = [3, 5, 7, 9]
    preds, gts, l2c, counts = [], [], [], []
    for _ in range(20):
        cls = rng.choice(test_classes, 2, replace=False)
        gt, pred = rng.integers(0, 3, (2, P)), rng.integers(0, 3, (2, P))
        preds.append(pred), gts.append(gt), l2c.append(cls)
        counts.append(probe.episode_counts(pred, gt, cls, test_classes))
    want = alternative_metrics(preds, gts, l2c, test_classes)["accumulated_fg"]
    assert probe.miou_from_counts(np.stack(counts).sum(0)) == pytest.approx(want, abs=1e-12)


def test_em12_paired_bootstrap():
    rng = np.random.default_rng(1)
    a = rng.integers(1, 50, (40, 3, 4)).astype(float)
    a[:, 2] = np.minimum(a[:, 2], np.minimum(a[:, 0], a[:, 1]))
    same = probe.paired_bootstrap(a, a.copy())
    assert same["gain"] == 0 and same["ci_low"] == 0 and same["ci_high"] == 0
    better = a.copy()
    better[:, 2, 1:] = np.minimum(better[:, 0, 1:], better[:, 1, 1:])  # every foreground TP maximal
    up = probe.paired_bootstrap(a, better)
    assert up["gain"] > 0 and up["ci_low"] > 0 and up["p_le_zero"] == 0


def _test_json(name, kind, fold, gain, unweighted_low=0.5, oracle=20.0, frozen="entropy_k8_T1"):
    ci = {"gain": gain, "ci_low": gain - 0.5, "ci_high": gain + 0.5, "p_le_zero": 0.0}
    return {"checkpoint": {"name": name, "kind": kind, "fold": fold}, "cvfold": fold, "frozen": frozen,
            "paired": {"frozen_vs_model": ci,
                       "frozen_vs_unweighted": {"gain": 1.0, "ci_low": unweighted_low, "ci_high": 1.5},
                       "oracle_replace_vs_model": {"gain": oracle}},
            "class_iou": {"model": [0.9, 0.5, 0.5], frozen: [0.9, 0.52, 0.51]}}


def test_em13_selection_and_rules():
    arms = [a for a in probe.grid_arms() if a["weight"] != "oracle"]
    assert len(arms) == 3 * 6 * 3 and len(probe.grid_arms()) == len(arms) + 7
    results = [{"miou": {probe.arm_name(a): 0.5 for a in probe.grid_arms()} | {"model": 0.5}} for _ in range(2)]
    results[0]["miou"]["entropy_k4_T2"] = 0.53
    for r in results:
        r["miou"][probe.arm_name(dict(weight="oracle", kappa=probe.ORACLE_REPLACE, steps=1))] = 0.99
    assert probe.select_setting(results)["name"] == "entropy_k4_T2"  # an oracle arm is never selected
    go = [_test_json("v1", "vipseg", 1, 2.0), _test_json("o1", "ours", 1, 1.6),
          _test_json("v0", "vipseg", 0, 1.2), _test_json("o0", "ours", 0, 1.1)]
    verdicts = dict(probe.decide(go))
    assert "P0.1 go" in verdicts and verdicts["P0.5 entropy weight"] == "claimable"
    s0_short = go[:2] + [_test_json("v0", "vipseg", 0, 0.8), _test_json("o0", "ours", 0, 1.1)]
    assert "P0.3 in between" in dict(probe.decide(s0_short))  # S1 passes, the held-out fold does not
    stop = [_test_json("v1", "vipseg", 1, 0.2, oracle=5.0), _test_json("o1", "ours", 1, 0.1),
            _test_json("v0", "vipseg", 0, 0.1, unweighted_low=-0.2), _test_json("o0", "ours", 0, 0.0)]
    verdicts = dict(probe.decide(stop))
    assert "P0.2 stop" in verdicts and verdicts["P0.5 entropy weight"].startswith("not claimable")
    assert "features, not the prototypes" in verdicts["P0.6 oracle v1"]
    assert probe.decide(go[:2])[0][0] == "incomplete"


def test_em14_selection_refuses_s0_checkpoints():
    args = SimpleNamespace(checkpoint=["v0:vipseg:0:missing.pt"], data_path="x", max_episodes=None)
    with pytest.raises(ValueError, match="S0 stays held out"):
        probe.cmd_select(args, torch.device("cpu"))
