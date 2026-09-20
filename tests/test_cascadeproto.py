"""CP-1..18 (05 §3.6b, gate G1): CascadeProto as every row of Table 4 (D-17) and Table 5.

The VIP-Seg encoder is replaced by the per-point stand-in of test_feature_extractor.py and CLIP by the
recording stand-in of test_clip_text.py; everything else (feature head, prototypes, adapter,
generator, GMMN, logits, loss) is the real code. float64, 1e-12.
"""

import math

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from loss.gmmn_loss import gmmn_loss
from models.cascadeproto import CascadeProto, CascadeProtoConfig
from models.clip_text import ClipTextEmbedding, episode_prompts
from models.eppm import prototype_diffusion
from models.prototypes import point_prototypes
from models.vipseg_backbone import PointFeatureExtractor
from pipeline.episodes import make_episode
from pipeline.model_api import episode_loss, predict
from tests.test_clip_text import RecordingEncoder
from tests.test_feature_extractor import StandInEncoder
from tests.test_lma_gmmn import mmd_ref, reference

ATOL = 1e-12
BASELINE = CascadeProtoConfig(use_lma=False, num_stages=0)
LMA = CascadeProtoConfig(use_lma=True, num_stages=0)
GATE = CascadeProtoConfig(use_lma=True, num_stages=1)  # "+ Entropy Gate" [DECISION D-17]
CASCADE = CascadeProtoConfig(use_lma=True, num_stages=4, use_adrm=False)  # "+ Cascade (T = 4)"
FULL = CascadeProtoConfig()  # "+ ADRM", the full model: LMA, T = 4, gate, ADRM
ROWS = [BASELINE, LMA, GATE, CASCADE, FULL]
CLASS_NAMES = ["ceiling", "floor", "wall", "beam", "column", "window", "door",
               "table", "chair", "sofa", "bookcase", "board", "clutter"]


def model(config=BASELINE, seed=0, encoder=None):
    torch.manual_seed(seed)
    text = ClipTextEmbedding(encode=encoder or RecordingEncoder())
    return CascadeProto(config, PointFeatureExtractor(encoder=StandInEncoder()), text_embedding=text).double()


def episode(n=2, k=2, bq=2, seed=0, classes=None):
    rng = np.random.default_rng(seed)
    item = (rng.random((n, k, 2048, 9)), rng.integers(0, 2, (n, k, 2048)).astype(np.int32),
            rng.random((bq, 2048, 9)), rng.integers(0, n + 1, (bq, 2048)),
            np.arange(3, 3 + n) if classes is None else np.array(classes))
    ep = make_episode(item, CLASS_NAMES)  # the pipeline's own validation of the 02 §1 layout
    ep.support_x, ep.query_x = ep.support_x.double(), ep.query_x.double()
    return ep


def test_cp1_output_contract():
    out = model().eval()(episode(n=3, bq=3))
    assert out.logits.shape == (3, 2048, 4) and out.logits.dtype == torch.float64
    assert out.loss_gmmn.shape == () and out.loss_gmmn.item() == 0.0  # use_lma=false drops L_GMMN (D-17)


def test_cp2_logits_are_query_features_times_point_prototypes():
    m, ep = model().eval(), episode(n=3, k=2, bq=3)
    f_s, f_q = m.features.encode_episode(ep.support_x, ep.query_x)
    p = point_prototypes(f_s, ep.support_y)  # [N+1, D]
    expected = torch.stack([f_q[b] @ p.T for b in range(3)])  # [B_q, 2048, N+1]
    assert torch.allclose(m(ep).logits, expected, atol=ATOL, rtol=0)
    b, i, c = 1, 777, 2
    assert math.isclose(m(ep).logits[b, i, c].item(), float(torch.dot(f_q[b, i], p[c])), abs_tol=ATOL)


def test_cp3_way_permutation_permutes_foreground_logits():
    m, ep = model().eval(), episode(n=3)
    ways = torch.tensor([2, 0, 1])
    permuted = episode(n=3)
    permuted.support_x, permuted.support_y = ep.support_x[ways], ep.support_y[ways]
    a, b = m(ep).logits, m(permuted).logits
    assert torch.allclose(b[..., 0], a[..., 0], atol=ATOL, rtol=0)
    assert torch.allclose(b[..., 1:], a[..., 1:][..., ways], atol=ATOL, rtol=0)


def test_cp4_post_encoder_computation_is_per_query():
    """With a per-point stand-in encoder nothing after the encoder mixes queries. The real VIP-Seg encoder
    does couple the blocks of one call through batch-wide statistics (02 §2, test ENC-3)."""
    m, ep = model().eval(), episode(bq=3)
    other = episode(bq=3)
    other.query_x = ep.query_x.clone()
    other.query_x[1:] = torch.rand_like(other.query_x[1:])
    assert torch.allclose(m(ep).logits[0], m(other).logits[0], atol=ATOL, rtol=0)


def test_cp5_d10_ablation_flags():
    ep = episode()
    base = model().eval()
    scaled = model(CascadeProtoConfig(use_lma=False, num_stages=0, logit_scale="sqrt_D")).eval()
    assert torch.allclose(scaled(ep).logits, base(ep).logits / math.sqrt(128), atol=ATOL, rtol=0)
    normed = model(CascadeProtoConfig(use_lma=False, num_stages=0, l2norm_point_proto=True)).eval()
    f_s, f_q = base.features.encode_episode(ep.support_x, ep.query_x)
    p = point_prototypes(f_s, ep.support_y)
    expected = torch.einsum("bpd,cd->bpc", f_q, p / p.norm(dim=-1, keepdim=True))
    assert torch.allclose(normed(ep).logits, expected, atol=ATOL, rtol=0)


@pytest.mark.parametrize("config,phase", [(CascadeProtoConfig(num_stages=2, use_adrm=False, cross_attn="two_hop"), "two_hop"),
                                          (CascadeProtoConfig(num_stages=1, diffusion_input="pre_relu"), "pre_relu"),
                                          (CascadeProtoConfig(num_stages=0, modality="audio"), "modality 'audio'"),
                                          (CascadeProtoConfig(num_stages=0, modality="image"), "modality 'image'"),
                                          (CascadeProtoConfig(num_stages=0, eval_noise="mean_of_M"), "mean_of_M")])
def test_cp6_unimplemented_configurations_raise(config, phase):
    with pytest.raises(NotImplementedError, match=phase):
        model(config)


@pytest.mark.parametrize("kwargs", [dict(num_stages=7), dict(num_stages=-1), dict(modality="video"),
                                    dict(logit_scale="sqrt_d"), dict(eval_noise="mean"),
                                    dict(gmmn_fg_mode="per_way"), dict(cross_attn="point"),
                                    dict(cross_attn_scale="sqrt_72"), dict(gate_target="both"),
                                    dict(fusion_weight="per_way"), dict(diffusion_input="raw")])
def test_cp6_invalid_configurations_raise(kwargs):
    with pytest.raises(ValueError):
        CascadeProtoConfig(use_lma=False, **{"num_stages": 0, **kwargs})


@pytest.mark.parametrize("config", ROWS)
def test_cp7_loss_backward_reaches_every_parameter(config):
    m, ep = model(config).train(), episode()
    episode_loss(m(ep), ep).backward()
    for name, p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, name


def learnable_episode(n=2, k=2, bq=2, seed=3):
    """Every point has a class 0..n (0 = background) whose colour (channels 3-5) is centred on a
    class-specific value; support masks and query labels follow the same classes, as the loader's do."""
    rng = np.random.default_rng(seed)
    centres = rng.random((n + 1, 3))

    def cloud(labels):
        x = rng.random(labels.shape + (9,))
        x[..., 3:6] = centres[labels] + 0.05 * rng.standard_normal(labels.shape + (3,))
        return x

    s_lab = np.stack([rng.choice([0, w + 1], size=(k, 2048), p=[0.6, 0.4]) for w in range(n)])  # [N, K, P]
    q_lab = rng.integers(0, n + 1, (bq, 2048))
    item = (cloud(s_lab), (s_lab > 0).astype(np.int32), cloud(q_lab), q_lab, np.arange(3, 3 + n))
    ep = make_episode(item, CLASS_NAMES)
    ep.support_x, ep.query_x = ep.support_x.double(), ep.query_x.double()
    return ep


@pytest.mark.parametrize("config", ROWS)
def test_cp8_model_learns_a_fixed_episode(config):
    """A few AdamW steps (the paper's optimiser settings) lower the loss and raise query accuracy."""
    m, ep = model(config).train(), learnable_episode()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.1)
    losses = []
    for _ in range(30):
        loss = episode_loss(m(ep), ep)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.8 * losses[0]
    m.eval()
    assert (predict(m(ep)) == ep.query_y).double().mean() > 1.0 / 3.0  # better than chance for N=2


@pytest.mark.parametrize("config", ROWS)
def test_cp8_state_dict_round_trip(config):
    trained, ep = model(config, seed=0).eval(), episode()
    fresh = model(config, seed=1).eval()
    assert not torch.allclose(fresh(ep).logits, trained(ep).logits)
    fresh.load_state_dict(trained.state_dict(), strict=True)
    assert torch.equal(fresh(ep).logits, trained(ep).logits)


# ------------------------------------------------------------ "+ LMA" row (02 §4, D-17)

def lma_terms(m, ep):
    """F^q, P_point and P_modal (z = 0), the last written out from the module's weights."""
    f_s, f_q = m.features.encode_episode(ep.support_x, ep.query_x)
    p_point = point_prototypes(f_s, ep.support_y)  # [N+1, D]
    e_clip = m.text(ep.class_names, torch.device("cpu")).double()  # [N+1, 512]
    p_modal = reference(m.lma, e_clip, torch.zeros_like(p_point))  # Eq.5-6 with z = 0
    return f_q, p_point, p_modal


@pytest.mark.parametrize("n", [2, 3])
def test_cp9_lma_logits_are_query_features_times_p0(n):
    m, ep = model(LMA).eval(), episode(n=n, bq=n)
    f_q, p_point, p_modal = lma_terms(m, ep)
    expected = torch.stack([f_q[b] @ (p_point + p_modal).T for b in range(n)])  # Eq.9, Eq.23
    out = m(ep)
    assert torch.allclose(out.logits, expected, atol=ATOL, rtol=0)
    assert torch.abs(p_modal).max() > 1e-3  # P_modal really contributes
    assert not torch.allclose(out.logits, model(BASELINE).eval()(ep).logits)


def test_cp10_lma_gmmn_term_is_eq8_on_point_and_modal_prototypes():
    m, ep = model(LMA).eval(), episode(n=3, bq=3)
    _, p_point, p_modal = lma_terms(m, ep)
    expected = 0.1 * mmd_ref(p_modal[:1], p_point[:1]) + 1.0 * mmd_ref(p_modal[1:], p_point[1:])
    out = m(ep)
    assert math.isclose(out.loss_gmmn.item(), expected, rel_tol=0, abs_tol=ATOL)
    ce = F.cross_entropy(out.logits.reshape(-1, 4), ep.query_y.reshape(-1)).item()
    assert math.isclose(episode_loss(out, ep).item(), ce + expected, rel_tol=0, abs_tol=ATOL)
    per_class = model(CascadeProtoConfig(use_lma=True, num_stages=0, gmmn_fg_mode="per_class")).eval()
    assert math.isclose(per_class(ep).loss_gmmn.item(),
                        gmmn_loss(p_modal, p_point, fg_mode="per_class").item(), rel_tol=0, abs_tol=ATOL)


def test_cp11_prompts_follow_the_episode_ways():
    enc = RecordingEncoder()
    m, ep = model(LMA, encoder=enc).eval(), episode(n=3, classes=[9, 3, 6])  # not in sorted order
    m(ep)
    assert ep.class_names == ["sofa", "beam", "door"]
    assert enc.calls == [["This point cloud represents the background.", "This point cloud represents the sofa.",
                          "This point cloud represents the beam.", "This point cloud represents the door."]]


def test_cp12_lma_evaluation_is_deterministic_and_training_draws_noise():
    m, ep = model(LMA), episode()
    m.eval()
    assert torch.equal(m(ep).logits, m(ep).logits)
    sample = model(CascadeProtoConfig(use_lma=True, num_stages=0, eval_noise="sample")).eval()
    assert not torch.equal(sample(ep).logits, sample(ep).logits)
    m.train()
    assert not torch.equal(m(ep).logits, m(ep).logits)


@pytest.mark.parametrize("detach", [False, True])
def test_cp12_gmmn_gradient_reaches_the_backbone_unless_detached(detach):
    m, ep = model(CascadeProtoConfig(use_lma=True, num_stages=0, gmmn_detach_point=detach)).train(), episode()
    m(ep).loss_gmmn.backward()
    head = [p.grad for name, p in m.named_parameters() if name.startswith("features.")]
    reached = any(g is not None and g.abs().sum() > 0 for g in head)
    assert reached is not detach
    assert all(p.grad.abs().sum() > 0 for p in m.lma.parameters())


def test_cp13_clip_is_outside_the_state_dict():
    m = model(LMA)
    keys = list(m.state_dict())
    assert all(k.startswith(("features.", "lma.")) for k in keys) and any(k.startswith("lma.") for k in keys)
    assert sum(p.numel() for p in m.lma.parameters()) == 148_352  # 01 §4
    base = sum(p.numel() for p in model(BASELINE).parameters())
    assert sum(p.numel() for p in m.parameters()) == base + 148_352
    m(episode())
    assert all(v.dtype == torch.float32 for v in m.text.cache.values())  # .double() left CLIP alone


def test_cp13_phase10_checkpoint_configuration_still_loads():
    phase10 = {"use_lma": False, "num_stages": 0, "use_gate": True, "use_adrm": True, "modality": "text",
               "logit_scale": "none", "l2norm_point_proto": False}
    config = CascadeProtoConfig(**phase10)
    assert config == BASELINE and config.clip_variant == "ViT-B/16" and config.eval_noise == "zero"


# ------------------------------------------------- "+ Entropy Gate" and "+ Cascade" rows (02 §6, D-17)

def cascade_ref(m, ep):
    """P^0 of Eq.9 (written out), copied per query, through the model's own stages in order; L^T of Eq.23."""
    f_q, p_point, p_modal = lma_terms(m, ep)
    f_s, _ = m.features.encode_episode(ep.support_x, ep.query_x)
    p = (p_point + p_modal).unsqueeze(0).repeat(f_q.shape[0], 1, 1)  # [B_q, N+1, D]
    for t in range(len(m.stages)):
        p = m.stages[t](p, f_s, f_q)
    return torch.einsum("bpd,bcd->bpc", f_q, p)


@pytest.mark.parametrize("config", [GATE, CascadeProtoConfig(use_lma=True, num_stages=2, use_adrm=False), CASCADE])
def test_cp14_logits_are_the_last_stage_of_the_cascade(config):
    m, ep = model(config).eval(), episode(n=2, bq=2)
    assert len(m.stages) == config.num_stages
    assert torch.allclose(m(ep).logits, cascade_ref(m, ep), atol=1e-11, rtol=0)
    stage_ids = [{id(p) for p in s.parameters()} for s in m.stages]
    assert all(not a & b for i, a in enumerate(stage_ids) for b in stage_ids[i + 1:])  # no sharing
    base = sum(p.numel() for p in model(LMA).parameters())
    assert sum(p.numel() for p in m.parameters()) == base + 79_395 * config.num_stages  # 01 §4


def test_cp14_stages_run_in_order():
    config = CascadeProtoConfig(use_lma=True, num_stages=2, use_adrm=False)
    m, ep = model(config).eval(), episode()
    swapped = model(config).eval()
    swapped.load_state_dict(m.state_dict())
    swapped.stages = torch.nn.ModuleList([m.stages[1], m.stages[0]])
    assert not torch.allclose(swapped(ep).logits, m(ep).logits)


def test_cp14_logit_scale_flag_applies_to_the_cascade_output():
    config = CascadeProtoConfig(use_lma=True, num_stages=2, use_adrm=False, logit_scale="sqrt_D")
    m, ep = model(config).eval(), episode()
    assert torch.allclose(m(ep).logits, cascade_ref(m, ep) / math.sqrt(128), atol=1e-11, rtol=0)


def test_cp15_single_stage_prediction_does_not_depend_on_use_adrm():
    ep = episode()
    on = model(CascadeProtoConfig(use_lma=True, num_stages=1, use_adrm=True)).eval()
    off = model(CascadeProtoConfig(use_lma=True, num_stages=1, use_adrm=False)).eval()
    assert torch.equal(on(ep).logits, off(ep).logits)


def test_cp15_gate_switch_and_flags_reach_every_stage():
    m = model(CascadeProtoConfig(use_lma=True, num_stages=3, use_adrm=False, use_gate=False,
                                 cross_attn_scale="sqrt_D", fusion_weight="per_class"))
    for s in m.stages:
        assert not s.gate.enabled and s.cross.scale == math.sqrt(128) and s.out.fusion_weight == "per_class"
    assert m.eval()(episode()).logits.shape == (2, 2048, 3)


@pytest.mark.parametrize("config", [GATE, CASCADE, CascadeProtoConfig(use_lma=False, num_stages=2, use_adrm=False)])
def test_cp16_cascade_keeps_way_equivariance_and_query_independence(config):
    m, ep = model(config).eval(), episode(n=3, bq=3)
    ways = torch.tensor([2, 0, 1])
    permuted = episode(n=3, bq=3, classes=[3 + int(w) for w in ways])
    permuted.support_x, permuted.support_y = ep.support_x[ways], ep.support_y[ways]
    a, b = m(ep).logits, m(permuted).logits
    assert torch.allclose(b[..., 0], a[..., 0], atol=1e-11, rtol=0)
    assert torch.allclose(b[..., 1:], a[..., 1:][..., ways], atol=1e-11, rtol=0)
    other = episode(n=3, bq=3)
    other.query_x = ep.query_x.clone()
    other.query_x[1:] = torch.rand_like(other.query_x[1:])
    assert torch.allclose(m(other).logits[0], a[0], atol=1e-11, rtol=0)


# ------------------------------------------------------------ "+ ADRM" row and Table 5 (02 §6)

def full_ref(m, ep):
    """Stage logits from the model's own stages in order, then Σ_t w_gate^(t) L^t with w_gate written out."""
    from tests.test_adrm_loss import weights_ref

    f_q, p_point, p_modal = lma_terms(m, ep)
    f_s, _ = m.features.encode_episode(ep.support_x, ep.query_x)
    p = (p_point + p_modal).unsqueeze(0).repeat(f_q.shape[0], 1, 1)
    per_stage = []
    for t in range(len(m.stages)):
        p = m.stages[t](p, f_s, f_q)
        per_stage.append(torch.einsum("bpd,bcd->bpc", f_q, p))
    w = weights_ref(m.routing, f_q)  # [B_q, T]
    return sum(w[:, t, None, None] * per_stage[t] for t in range(len(per_stage))), per_stage


@pytest.mark.parametrize("t", [2, 4])
def test_cp17_full_model_logits_are_the_routed_stage_logits(t):
    m, ep = model(CascadeProtoConfig(num_stages=t)).eval(), episode(n=2, bq=2)
    expected, per_stage = full_ref(m, ep)
    out = m(ep).logits
    assert torch.allclose(out, expected, atol=1e-11, rtol=0)
    assert not torch.allclose(out, per_stage[-1])  # ADRM really mixes the stages
    assert m.routing.w_g.weight.shape == (t, 128) and m.routing.w_g.bias is None


def test_cp17_full_model_parameter_budget():
    """Added modules = adapter + generator + 4 stages + W_g = 466,444 (01 §4)."""
    full = sum(p.numel() for p in model(FULL).parameters())
    base = sum(p.numel() for p in model(BASELINE).parameters())
    assert full - base == 148_352 + 4 * 79_395 + 512 == 466_444


@pytest.mark.parametrize("t", [1, 2, 3, 4, 5, 6])
def test_cp18_table5_depths_with_adrm(t):
    """Table 5 varies T with every other switch on; T = 1 has no W_g (D-17)."""
    m, ep = model(CascadeProtoConfig(num_stages=t)).train(), episode()
    assert len(m.stages) == t
    assert (m.routing is None) if t == 1 else (m.routing.w_g.out_features == t)
    episode_loss(m(ep), ep).backward()
    for name, p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, name


def test_cp19_cross_attn_norm_reaches_every_stage():
    """D-18: the switch is off by default and, when on, adds one LayerNorm per stage and nothing else."""
    default, normed = model(FULL), model(CascadeProtoConfig(cross_attn_norm="layernorm"))
    assert all(s.cross.proj_norm is None for s in default.stages)
    assert [s.cross.proj_norm.normalized_shape for s in normed.stages] == [(72,)] * 4
    count = lambda m: sum(p.numel() for p in m.parameters())
    assert count(normed) - count(default) == 4 * 144
    with pytest.raises(ValueError):
        CascadeProtoConfig(cross_attn_norm="rmsnorm")
    ep = episode()
    assert torch.isfinite(normed(ep).logits).all()


def test_cp20_gate_target_features_is_the_literal_eq13_14():
    """D-02: with `features` the gate feeds phi (Eq.13) and psi keeps the ungated P^(t-1) of Eq.14."""
    proto = model(CascadeProtoConfig(num_stages=1))
    feats = model(CascadeProtoConfig(num_stages=1, gate_target="features"))
    count = lambda m: sum(p.numel() for p in m.parameters())
    assert count(proto) == count(feats)  # same parameters, different wiring
    assert [s.gate_target for s in feats.stages] == ["features"]
    ep = episode()
    assert not torch.allclose(proto(ep).logits, feats(ep).logits)

    stage = feats.stages[0]
    f_s, f_q = feats.features.encode_episode(ep.support_x, ep.query_x)
    p = point_prototypes(f_s, ep.support_y).unsqueeze(0).expand(f_q.shape[0], -1, -1)
    expected = stage.out(stage.cross(p, stage.gate(f_s), stage.gate(f_q)),
                         prototype_diffusion(f_s, f_q)[:, None, :].expand_as(p), p)
    assert torch.allclose(stage(p, f_s, f_q), expected, atol=ATOL, rtol=0)


def test_cp20_the_gate_is_elementwise_so_one_module_serves_both_targets():
    """Eq.12 is elementwise on the last axis, so one EntropyGate gates [N,K,P,D] and [B,N+1,D] alike."""
    stage = model(CascadeProtoConfig(num_stages=1, gate_target="features")).stages[0]
    with torch.no_grad():
        stage.gate.theta.fill_(0.42)
    f = torch.rand(2, 3, 5, 128, dtype=torch.float64)
    assert torch.allclose(stage.gate(f)[1, 2, 3], stage.gate(f[1, 2, 3]), atol=ATOL, rtol=0)
    lo, hi = torch.sigmoid(torch.tensor([2 * (0.42 - math.log(2.0)), 2 * 0.42], dtype=torch.float64))
    assert (stage.gate(f) <= hi * f).all() and (stage.gate(f) >= lo * f).all()  # H in [0, ln 2]
