"""FEAT-1..8 (05 §3.2b, gate G1): feature head and episode batching of PointFeatureExtractor.

The VIP-Seg encoder needs CUDA, so here it is replaced by a per-point linear stand-in with the same
contract ([B, 2048, 9] -> [B, 900, 2048]); the real encoder is tested in tests/test_encoder.py.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vipseg_backbone import ENCODER_DIM, FEATURE_DIM, IN_CHANNELS, NUM_POINT, PointFeatureExtractor

ATOL = 1e-12


class StandInEncoder(nn.Module):
    """Per-point map 9 -> 900, so each block's output depends only on that block."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(IN_CHANNELS, ENCODER_DIM)

    def forward(self, x):  # [B, 2048, 9]
        return self.proj(x).transpose(1, 2)  # [B, 900, 2048]


def extractor(seed=0):
    torch.manual_seed(seed)
    model = PointFeatureExtractor(encoder=StandInEncoder()).double()
    for bn in (m for m in model.modules() if isinstance(m, nn.BatchNorm1d)):  # non-trivial running stats
        bn.running_mean.uniform_(-0.5, 0.5)
        bn.running_var.uniform_(0.5, 2.0)
        bn.weight.data.uniform_(0.5, 1.5)
        bn.bias.data.uniform_(-0.2, 0.2)
    return model


def points(*batch, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(*batch, NUM_POINT, IN_CHANNELS, generator=g, dtype=torch.float64)


def test_feat1_head_matches_vipseg_layers_and_count():
    head = extractor()
    head_params = sum(p.numel() for n, p in head.named_parameters() if not n.startswith("encoder."))
    assert head_params == 204_260  # 01 §4: BN(900) 1,800 + conv 176,596 + BN(196) 392 + conv 25,216 + BN(128) 256
    layers = [type(m).__name__ for m in head.bn] + [type(m).__name__ for m in head.fc]
    assert layers == ["BatchNorm1d", "ReLU", "Conv1d", "BatchNorm1d", "ReLU", "Conv1d", "BatchNorm1d", "ReLU"]
    assert (head.fc[0].in_channels, head.fc[0].out_channels, head.fc[3].out_channels) == (900, 196, 128)
    assert not any(getattr(m, "inplace", False) for m in head.modules())  # nn.ReLU() as in VIP-Seg


def test_feat2_forward_equals_manual_formula_in_eval():
    model = extractor().eval()
    x = points(3)
    e = model.encoder(x)  # [3, 900, 2048]
    h = e / e.norm(dim=1, keepdim=True)

    def bn(t, m):
        return F.batch_norm(t, m.running_mean, m.running_var, m.weight, m.bias, False, 0.0, m.eps)

    h = F.relu(bn(h, model.bn[0]))
    h = F.relu(bn(F.conv1d(h, model.fc[0].weight, model.fc[0].bias), model.fc[1]))
    h = F.relu(bn(F.conv1d(h, model.fc[3].weight, model.fc[3].bias), model.fc[4]))
    out = model(x)
    assert out.shape == (3, NUM_POINT, FEATURE_DIM)
    assert torch.allclose(out, h.transpose(1, 2), atol=ATOL, rtol=0)
    assert (out >= 0).all()


def test_feat3_invariant_to_per_point_encoder_scale():
    """The per-point L2 normalisation removes any positive per-point scale of the encoder output."""
    model = extractor().eval()
    x = points(2)
    scale = torch.rand(2, 1, NUM_POINT, dtype=torch.float64) + 0.1  # [B, 1, 2048]
    base = model(x)
    original = model.encoder.forward
    model.encoder.forward = lambda t: original(t) * scale
    assert torch.allclose(model(x), base, atol=1e-10, rtol=0)


def test_feat4_encode_episode_keeps_way_and_shot_order():
    model = extractor().eval()
    n, k, bq = 3, 2, 3
    sx, qx = points(n, k, seed=2), points(bq, seed=3)
    f_s, f_q = model.encode_episode(sx, qx)
    assert f_s.shape == (n, k, NUM_POINT, FEATURE_DIM) and f_q.shape == (bq, NUM_POINT, FEATURE_DIM)
    for i in range(n):
        for j in range(k):
            assert torch.allclose(f_s[i, j], model(sx[i, j][None])[0], atol=ATOL, rtol=0), (i, j)
    for b in range(bq):
        assert torch.allclose(f_q[b], model(qx[b][None])[0], atol=ATOL, rtol=0)


def test_feat5_train_mode_support_and_query_are_separate_batches():
    """BatchNorm batch statistics come from the support blocks alone and the queries alone."""
    model = extractor().train()
    sx, qx = points(2, 2, seed=4), points(2, seed=5)
    f_s, f_q = model.encode_episode(sx, qx)
    assert torch.allclose(f_s, model(sx.flatten(0, 1)).unflatten(0, (2, 2)), atol=ATOL, rtol=0)
    f_s_other_query, _ = model.encode_episode(sx, points(2, seed=6))
    assert torch.allclose(f_s, f_s_other_query, atol=ATOL, rtol=0)


@pytest.mark.parametrize("shape", [(2, NUM_POINT, 3), (2, 4096, IN_CHANNELS), (2, IN_CHANNELS, NUM_POINT),
                                   (NUM_POINT, IN_CHANNELS), (1, 2, NUM_POINT, IN_CHANNELS)])
def test_feat6_rejects_other_layouts(shape):
    with pytest.raises(ValueError):
        extractor()(torch.rand(*shape, dtype=torch.float64))


def test_feat6_rejects_wrong_encoder_output():
    model = PointFeatureExtractor(encoder=nn.Identity()).double()
    with pytest.raises(ValueError, match="encoder output"):
        model(points(1))


def test_feat7_gradients_reach_encoder_and_head():
    model = extractor().train()
    f_s, f_q = model.encode_episode(points(2, 1, seed=7), points(2, seed=8))
    (f_s.sum() + f_q.sum()).backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    assert model.encoder.proj.weight.grad.abs().sum() > 0


class Projection(nn.Module):
    """Mimics the encoder's low-/high-order convolutions: a plain-tensor attribute `vv`."""

    def __init__(self, vv):
        super().__init__()
        self.vv = vv
        self.lin = nn.Linear(IN_CHANNELS, ENCODER_DIM)


class ProjectionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        vv = torch.randn(1, ENCODER_DIM)  # shared by both stages, like the real encoder
        self.stage1, self.stage2 = Projection(vv), Projection(vv)

    def forward(self, x):  # [B, 2048, 9] -> [B, 900, 2048]
        return (self.stage1.lin(x) * self.stage1.vv + self.stage2.lin(x) * self.stage2.vv).transpose(1, 2)


def test_feat8_fixed_projections_are_saved_and_restored():
    torch.manual_seed(0)
    trained = PointFeatureExtractor(encoder=ProjectionEncoder()).double().eval()
    state = trained.state_dict()
    assert {"encoder.stage1.vv", "encoder.stage2.vv"} <= set(state)
    torch.manual_seed(1)  # a fresh model draws different projections
    fresh = PointFeatureExtractor(encoder=ProjectionEncoder()).double().eval()
    assert not torch.equal(fresh.encoder.stage1.vv, trained.encoder.stage1.vv)
    x = points(2)
    assert not torch.allclose(fresh(x), trained(x))
    fresh.load_state_dict(state, strict=True)
    assert torch.equal(fresh(x), trained(x))


def test_feat8_load_vipseg_weights_copies_plain_projections():
    """A pickled VIP-Seg model keeps vv / ww as plain attributes; load_vipseg_weights must copy them."""
    torch.manual_seed(2)
    donor = PointFeatureExtractor(encoder=ProjectionEncoder()).double().eval()
    vipseg_like = nn.Module()
    vipseg_like.encoder = ProjectionEncoder().double()  # plain attributes, not buffers
    vipseg_like.encoder.load_state_dict({k: v for k, v in donor.encoder.state_dict().items()
                                         if not k.endswith(".vv")})
    vipseg_like.bn, vipseg_like.fc = donor.bn, donor.fc
    vipseg_like.vip_module = nn.Linear(2, 2)  # parts CascadeProto does not take
    assert "encoder.stage1.vv" not in vipseg_like.state_dict()
    torch.manual_seed(3)
    ours = PointFeatureExtractor(encoder=ProjectionEncoder()).double().eval()
    ours.load_vipseg_weights(vipseg_like)
    x = points(2)
    ref = vipseg_like.encoder(x)
    ref = vipseg_like.fc(vipseg_like.bn(ref / ref.norm(dim=1, keepdim=True))).transpose(1, 2)
    assert torch.allclose(ours(x), ref, atol=ATOL, rtol=0)


def test_feat8_load_vipseg_weights_rejects_a_different_encoder():
    """Strict loading: an encoder with extra or missing weights must not load silently."""
    donor = PointFeatureExtractor(encoder=ProjectionEncoder()).double()
    vipseg_like = nn.Module()
    vipseg_like.encoder = ProjectionEncoder().double()
    vipseg_like.encoder.extra = nn.Linear(3, 3).double()  # not part of our encoder
    vipseg_like.bn, vipseg_like.fc = donor.bn, donor.fc
    with pytest.raises(RuntimeError, match="Unexpected key"):
        PointFeatureExtractor(encoder=ProjectionEncoder()).double().load_vipseg_weights(vipseg_like)
