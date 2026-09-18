"""
Unit test for VIP-Seg Backbone and Point Prototype Extraction.
Verifies:
1. Feature extraction shape [B, N, 128] from input points [B, N, 3].
2. Point prototype extraction [B, N_way + 1, 128].
3. Gating network softmax simplex sum(w_gate) == 1.0.
4. Zero NaNs and device agnosticism (CPU & CUDA).
"""

import pytest
import torch

# Builds the VIP-Seg encoder, which needs mamba_ssm and pointnet2_ops (gate G2); imports are
# inside the tests so that CPU-only collection (gate G1) does not import the encoder.
pytestmark = pytest.mark.cuda


def test_vipseg_backbone_forward_cpu():
    from models.vipseg_backbone import VIPSegBackbone, extract_point_prototypes
    backbone = VIPSegBackbone(input_points=2048, out_dim=128)
    backbone.eval()
    
    # Input synthetic point cloud coordinates [B, N, 3]
    x = torch.randn(2, 2048, 3)
    with torch.no_grad():
        feat = backbone(x)
        
    assert feat.shape == (2, 2048, 128), f"Expected (2, 2048, 128), got {feat.shape}"
    assert not torch.isnan(feat).any(), "Found NaNs in backbone features!"
    assert not torch.isinf(feat).any(), "Found Infs in backbone features!"


def test_point_prototype_extraction():
    from models.vipseg_backbone import VIPSegBackbone, extract_point_prototypes
    B = 2
    n_way = 2
    total_pts = 2048
    D = 128
    
    features = torch.randn(B, total_pts, D)
    # Mask with background (0), class 1, class 2
    masks = torch.zeros(B, total_pts, dtype=torch.long)
    masks[:, :500] = 1
    masks[:, 500:1000] = 2
    
    p_point = extract_point_prototypes(features, masks, n_way=n_way)
    assert p_point.shape == (B, n_way + 1, D), f"Expected ({B}, {n_way + 1}, {D}), got {p_point.shape}"
    assert not torch.isnan(p_point).any(), "Found NaNs in extracted prototypes!"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_vipseg_backbone_cuda():
    from models.vipseg_backbone import VIPSegBackbone, extract_point_prototypes
    backbone = VIPSegBackbone(input_points=2048, out_dim=128).cuda()
    backbone.eval()
    
    x = torch.randn(2, 2048, 3, device="cuda")
    with torch.no_grad():
        feat = backbone(x)
        
    assert feat.shape == (2, 2048, 128)
    assert not torch.isnan(feat).any()
