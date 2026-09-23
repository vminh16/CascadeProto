"""VIPS-1..5 (05 §3.4c): VIP-Seg's PEM/PDM behind the stage contract [DECISION D-25], beyond the paper.

VIPS-1..4 run on the CPU with a recording stand-in module, because `models.vipseg` imports the encoder
and `pointnet2_ops`. VIPS-5 (marker `cuda`) checks the real modules against VIP-Seg's own loop.
"""

import pytest
import torch
import torch.nn as nn

from models.cascadeproto import CascadeProtoConfig, build_stage
from models.vip_stage import VIPStage

D, P = 128, 2048


class Recorder(nn.Module):
    """Stands in for PEM/PDM: records the call and returns a deterministic function of its arguments."""

    def __init__(self):
        super().__init__()
        self.calls = []
        self.scale = nn.Parameter(torch.tensor(2.0, dtype=torch.float64))

    def forward(self, query, supports, prototype):
        self.calls.append((query.shape, supports.shape, prototype.shape))
        return self.scale * prototype + query.mean(dim=1)[:, None, :] + supports.mean(dim=(0, 1, 2))


def episode_tensors(n=2, k=1, bq=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    f_s = torch.randn(n, k, P, D, generator=g, dtype=torch.float64).abs()
    f_q = torch.randn(bq, P, D, generator=g, dtype=torch.float64).abs()
    p = torch.randn(bq, n + 1, D, generator=g, dtype=torch.float64)
    return p, f_s, f_q


def test_vips1_call_order_matches_vipsegs_own():
    """VIP-Seg calls module(query_feat, support_feat, feature_memory) [VIPSEG models/vipseg.py:155-158]."""
    module = Recorder()
    p, f_s, f_q = episode_tensors()
    out = VIPStage(step=0, module=module)(p, f_s, f_q)
    assert module.calls == [(f_q.shape, f_s.shape, p.shape)]
    assert torch.equal(out, module(f_q, f_s, p))


@pytest.mark.parametrize("step,residual", [(0, False), (1, True), (2, False), (3, True)])
def test_vips2_odd_steps_add_the_outer_residual(step, residual):
    """PDM's output is added to the running prototype [VIPSEG models/vipseg.py:157]."""
    module = Recorder()
    p, f_s, f_q = episode_tensors()
    stage = VIPStage(step=step, module=module)
    assert stage.outer_residual is residual
    expected = module(f_q, f_s, p) + (p if residual else 0.0)
    assert torch.equal(stage(p, f_s, f_q), expected)


def test_vips3_shape_contract():
    module = Recorder()
    p, f_s, f_q = episode_tensors()
    with pytest.raises(ValueError, match="does not match"):
        VIPStage(module=module)(p[:, :-1], f_s, f_q)  # one class row missing
    with pytest.raises(ValueError, match="does not match"):
        VIPStage(module=module)(p[:1], f_s, f_q)  # fewer prototypes than queries
    assert module.calls == []  # the guard runs before the module


def test_vips4_build_stage_alternates_and_the_config_rejects_ignored_switches(monkeypatch):
    import models.vip_stage as vip_stage

    monkeypatch.setattr(vip_stage, "build_vip_module", lambda step: Recorder())
    config = CascadeProtoConfig(stage_type="vip", num_stages=4)
    stages = [build_stage(config, step) for step in range(4)]
    assert [s.outer_residual for s in stages] == [False, True, False, True]
    assert [s.step for s in stages] == [0, 1, 2, 3]
    with pytest.raises(ValueError, match="ignores"):
        CascadeProtoConfig(stage_type="vip", cross_attn_scale="sqrt_D").check_implemented()
    with pytest.raises(ValueError, match="ignores"):
        CascadeProtoConfig(stage_type="vip", cross_attn_support="pooled").check_implemented()
    with pytest.raises(ValueError, match="ignores"):
        CascadeProtoConfig(stage_type="eppm_s", use_gate=False).check_implemented()
    CascadeProtoConfig(stage_type="eppm_s", cross_attn_support="pooled").check_implemented()  # allowed


@pytest.mark.cuda
def test_vips5_four_vip_stages_reproduce_vipsegs_own_loop():
    """The stage wrapper must equal VIP-Seg's reasoning loop [VIPSEG models/vipseg.py:148-160]."""
    from models.vip_stage import build_vip_module

    torch.manual_seed(0)
    modules = [build_vip_module(step).cuda() for step in range(4)]
    p, f_s, f_q = episode_tensors(seed=1)
    p, f_s, f_q = p.float().cuda(), f_s.float().cuda(), f_q.float().cuda()
    memory = p
    for step, module in enumerate(modules):  # VIP-Seg's own loop, written out
        memory = module(f_q, f_s, memory) + (memory if step % 2 == 1 else 0.0)
    ours = p
    for step, module in enumerate(modules):
        ours = VIPStage(step=step, module=module)(ours, f_s, f_q)
    assert torch.allclose(ours, memory, atol=1e-6, rtol=0)
