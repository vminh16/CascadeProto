"""ENC-1..7 (05 §3.8, gate G2): PointFeatureExtractor with the real VIP-Seg encoder on the GPU.

ENC-1 and ENC-6 load VIP-Seg's released checkpoint ($CASCADEPROTO_VIPSEG_CKPT, default
<repo>/vipseg_S0_N2_K1.pt, see README §5); only load checkpoints from the official repository.
"""

import os

import pytest
import torch

from models.vipseg_backbone import FEATURE_DIM, IN_CHANNELS, NUM_POINT, PointFeatureExtractor

pytestmark = pytest.mark.cuda

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.environ.get("CASCADEPROTO_VIPSEG_CKPT", os.path.join(REPO, "vipseg_S0_N2_K1.pt"))
BLOCK_TOL = 1e-5  # different batch compositions may pick different CUDA kernels


@pytest.fixture(scope="module")
def vipseg_model():
    if not os.path.isfile(CKPT):
        pytest.fail(f"VIP-Seg checkpoint not found at {CKPT} (README §5 has the download command)")
    import models.vipseg  # noqa: F401  (class definitions for unpickling)

    return torch.load(CKPT, map_location="cuda", weights_only=False)["model"].eval()


@pytest.fixture(scope="module")
def extractor(vipseg_model):
    model = PointFeatureExtractor().cuda()
    model.load_vipseg_weights(vipseg_model)  # strict: identical names and shapes, vv / ww included
    return model.eval()


def blocks(b, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(b, NUM_POINT, IN_CHANNELS, generator=g)  # xyz, rgb, XYZ all in [0, 1)
    return x.cuda()


def test_enc1_parameter_counts(extractor, vipseg_model):
    enc = sum(p.numel() for p in extractor.encoder.parameters())
    head = sum(p.numel() for n, p in extractor.named_parameters() if not n.startswith("encoder."))
    assert enc == sum(p.numel() for p in vipseg_model.encoder.parameters())
    assert 2_360_000 <= enc <= 2_380_000  # 2.37M [VIPSEG log_s3dis_VIPSeg/log_S0_N2_K1_0.722026/log_vipseg.txt:5]
    assert head == 204_260


def test_enc2_output_shape_and_range(extractor):
    with torch.no_grad():
        out = extractor(blocks(3, seed=0))
    assert out.shape == (3, NUM_POINT, FEATURE_DIM)
    assert torch.isfinite(out).all() and (out >= 0).all()


@pytest.mark.parametrize("first", [0, 2])
def test_enc3_support_blocks_are_independent(extractor, first):
    x = blocks(4, seed=1)
    y = x.clone()
    y[[i for i in range(4) if i != first]] = blocks(3, seed=2)
    with torch.no_grad():
        diff = (extractor(x)[first] - extractor(y)[first]).abs().max().item()
    assert diff <= BLOCK_TOL, diff


def test_enc4_queries_are_independent_and_match_single_block(extractor):
    q = blocks(3, seed=3)
    with torch.no_grad():
        batched = extractor(q)
        single = torch.stack([extractor(q[i:i + 1])[0] for i in range(3)])
    assert (batched - single).abs().max().item() <= BLOCK_TOL


def test_enc5_rejects_non_9_channel_input(extractor):
    with pytest.raises(ValueError):
        extractor(blocks(1, seed=4)[..., :3])
    with pytest.raises(ValueError):
        extractor(blocks(1, seed=4).transpose(1, 2))


def test_enc6_features_equal_vipseg_model(extractor, vipseg_model):
    """Same weights, same input -> the features VIP-Seg feeds to its prototype module
    [VIPSEG models/vipseg.py:85-97]."""
    x = blocks(4, seed=5)
    with torch.no_grad():
        ref = vipseg_model.encoder(x)
        ref = ref / ref.norm(dim=1, keepdim=True)
        ref = vipseg_model.fc(vipseg_model.bn(ref)).permute(0, 2, 1)  # [B, 2048, 128]
        ours = extractor(x)
    assert (ours - ref).abs().max().item() <= 1e-6


def test_enc7_encode_episode_matches_per_block_calls(extractor):
    sx, qx = blocks(6, seed=6).unflatten(0, (3, 2)), blocks(3, seed=7)  # [3, 2, 2048, 9], [3, 2048, 9]
    with torch.no_grad():
        f_s, f_q = extractor.encode_episode(sx, qx)
        for i in range(3):
            for j in range(2):
                assert (f_s[i, j] - extractor(sx[i, j][None])[0]).abs().max().item() <= BLOCK_TOL
        assert (f_q - extractor(qx)).abs().max().item() == 0.0


def test_enc8_checkpoint_round_trip_restores_fixed_projections(extractor):
    """A state_dict must carry vv / ww: a fresh model draws new ones [VIPSEG models/encoder.py:619-620]."""
    state = extractor.state_dict()
    assert sum(k.endswith((".vv", ".ww")) for k in state) == 6  # 3 stages x (low-order vv, high-order ww)
    torch.manual_seed(12345)
    fresh = PointFeatureExtractor().cuda().eval()
    x = blocks(2, seed=8)
    with torch.no_grad():
        assert (fresh(x) - extractor(x)).abs().max().item() > 1e-3  # different projections, different model
        fresh.load_state_dict(state, strict=True)
        assert (fresh(x) - extractor(x)).abs().max().item() == 0.0
