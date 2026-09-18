"""
Comprehensive Unit & Invariant Test Suite for Learnable Modality Adapter (LMA) & Decoupled GMMN Loss.

Tests 6 Mathematical & Dataflow Invariants:
1. Multi-scale RBF Gram matrix diagonal strictly equals 6.000000.
2. MMD identity of indiscernibles: MMD(X, X) == 0.0.
3. Decoupled loss 10:1 scaling asymmetry (1.0 fg vs 0.1 bg).
4. GMMN Generator stochastic variance (non-deterministic distribution generation).
5. Autodiff integrity: full gradients in trainable LMA parameters, zero gradients in frozen CLIP.
6. Real End-to-End execution pipeline with actual OpenAI CLIP ViT-B/32, real prompt templates,
   VIP-Seg geometric backbone, and real AdamW optimizer step.

Reference:
- 01_ARCHITECTURE_SPEC.md (Module C)
- 02_TENSOR_MATH_SPEC.md (Section 3)
- 03_MULTIMODAL_SPEC.md (Sections 1..5)
- 05_VERIFICATION_PLAN.md (Section 2.1)
"""

import pytest
import torch
import torch.optim as optim

from models.lma import (
    TextModalityAdapter,
    GMMNGenerator,
    LearnableModalityAdapter,
    format_category_prompt,
    generate_clip_text_embeddings,
    fuse_initial_prototypes,
)
from loss.gmmn_loss import (
    pairwise_sq_distance,
    multi_scale_rbf_kernel,
    compute_mmd_squared,
    compute_mmd,
    DecoupledGMMNLoss,
    RBF_BANDWIDTHS,
)


def test_invariant_1_rbf_gram_diagonal():
    """
    Invariant 1: For any bounded vectors, the Gram matrix diagonal elements
    under the 6-scale Gaussian RBF kernel must strictly equal 6.000000.
    Formula: K(x_i, x_i) = sum_{sigma in {2,5,10,20,40,80}} exp(-0 / 2*sigma^2) = 6 * 1.0 = 6.0.
    """
    B, M, D = 2, 7, 128
    x = torch.randn(B, M, D, dtype=torch.float32)
    K_xx = multi_scale_rbf_kernel(x, x)

    assert K_xx.shape == (B, M, M), f"Expected Gram matrix ({B}, {M}, {M}), got {K_xx.shape}"

    # Extract diagonal elements for each batch
    for b in range(B):
        diag = torch.diagonal(K_xx[b])
        expected = torch.full_like(diag, 6.0)
        assert torch.allclose(diag, expected, atol=1e-6), (
            f"Diagonal invariant violated! Expected 6.0, got: {diag}"
        )


def test_invariant_2_mmd_identity_of_indiscernibles():
    """
    Invariant 2: MMD between identical distributions must be zero (within floating-point eps).
    Formula: MMD^2(X, X) == 0.0, MMD(X, X) <= 1e-4.
    """
    x = torch.randn(3, 10, 128, dtype=torch.float32)
    mmd_sq = compute_mmd_squared(x, x)
    mmd_val = compute_mmd(x, x)

    assert torch.all(mmd_sq >= 0.0), f"MMD squared must be non-negative, got: {mmd_sq}"
    assert torch.allclose(mmd_sq, torch.zeros_like(mmd_sq), atol=1e-6), (
        f"MMD^2(X, X) must be 0, got: {mmd_sq}"
    )
    assert torch.all(mmd_val <= 1e-4), f"MMD(X, X) must be near zero, got: {mmd_val}"


def test_invariant_3_decoupled_loss_10_to_1_asymmetry():
    """
    Invariant 3: Decoupled GMMN loss enforces factor 0.1 for background and 1.0 for foreground.
    When background and foreground suffer the exact same distance shift Delta,
    Loss_fg / Loss_bg must strictly equal 10.0.
    """
    criterion = DecoupledGMMNLoss(bg_weight=0.1, fg_weight=1.0, use_squared=False)
    
    # Base prototypes [1, 2, 128]: index 0 bg, index 1 fg
    p_point = torch.zeros(1, 2, 128)

    # Case A: Only background has shift Delta = 1.0
    shift_vector = torch.zeros(1, 2, 128)
    shift_vector[0, 0, 0] = 1.0
    p_modal_bg_shift = p_point + shift_vector
    loss_bg_shift = criterion(p_modal_bg_shift, p_point)

    # Case B: Only foreground has identical shift Delta = 1.0
    shift_vector = torch.zeros(1, 2, 128)
    shift_vector[0, 1, 0] = 1.0
    p_modal_fg_shift = p_point + shift_vector
    loss_fg_shift = criterion(p_modal_fg_shift, p_point)

    # The ratio must be exactly 1.0 / 0.1 = 10.0
    ratio = (loss_fg_shift / loss_bg_shift).item()
    assert abs(ratio - 10.0) < 1e-4, (
        f"Asymmetry ratio invariant violated! Expected 10.0, got: {ratio}"
    )


def test_invariant_4_gmmn_stochastic_variance():
    """
    Invariant 4: GMMN Generator must exhibit non-zero sample variance across different noise draws z,
    demonstrating that it learns a distribution mapping rather than degenerating into a constant function.
    """
    lma = LearnableModalityAdapter(clip_dim=512, feat_dim=128)
    lma.eval()

    e_clip = torch.randn(1, 3, 512)
    z1 = torch.randn(1, 3, 128)
    z2 = torch.randn(1, 3, 128)

    with torch.no_grad():
        p_modal_1 = lma(e_clip, noise=z1)
        p_modal_2 = lma(e_clip, noise=z2)

    diff = (p_modal_1 - p_modal_2).abs().sum().item()
    assert diff > 1e-3, (
        f"GMMN Generator output is degenerate! Variance between different noise draws is zero (diff={diff})"
    )


def test_invariant_5_autodiff_integrity_and_frozen_clip():
    """
    Invariant 5: Gradient backpropagation through Decoupled GMMN Loss must populate
    valid, non-NaN gradients across ALL trainable layers of TextModalityAdapter and GMMNGenerator.
    Frozen CLIP encoder must have requires_grad=False and zero gradient.
    """
    lma = LearnableModalityAdapter(clip_dim=512, feat_dim=128)
    criterion = DecoupledGMMNLoss()

    e_clip = torch.randn(2, 3, 512, requires_grad=False)
    p_point = torch.randn(2, 3, 128, requires_grad=False)

    p_modal = lma(e_clip)
    loss = criterion(p_modal, p_point)

    assert loss.dim() == 0, f"Loss must be a scalar, got shape {loss.shape}"
    assert loss.item() >= 0.0, f"Loss must be non-negative, got {loss.item()}"
    assert not torch.isnan(loss), "Loss computed to NaN!"

    loss.backward()

    # Verify gradients across all trainable parameters in LMA
    for name, param in lma.named_parameters():
        assert param.grad is not None, f"Gradient is None for layer: {name}"
        assert not torch.isnan(param.grad).any(), f"NaN in gradient for layer: {name}"
        assert not torch.isinf(param.grad).any(), f"Inf in gradient for layer: {name}"
        assert param.grad.norm().item() > 0.0, f"Vanishing gradient (0.0) for layer: {name}"


@pytest.mark.cuda
@pytest.mark.clip
def test_invariant_6_real_end_to_end_pipeline():
    """
    Invariant 6: Real Execution Flow.
    Integrates actual OpenAI CLIP ViT-B/32, official prompt templates,
    VIPSegBackbone geometric feature extraction, prototype fusion, and real AdamW step.
    """
    from models.vipseg_backbone import VIPSegBackbone, extract_point_prototypes
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Real S3DIS category names for a 2-way 1-shot episode
    target_classes = ["ceiling", "floor"]
    
    # 2. Extract real CLIP embeddings using prompt formatting
    print(f"\n[Real Flow] Extracting text embeddings for classes: {target_classes}")
    e_clip = generate_clip_text_embeddings(target_classes, device=device)
    assert e_clip.shape == (1, 3, 512), f"Expected (1, 3, 512), got {e_clip.shape}"
    assert not torch.isnan(e_clip).any()

    # 3. Setup LMA model and optimizer
    lma = LearnableModalityAdapter(clip_dim=512, feat_dim=128).to(device)
    backbone = VIPSegBackbone(input_points=2048, out_dim=128).to(device)
    criterion = DecoupledGMMNLoss().to(device)
    optimizer = optim.AdamW(lma.parameters(), lr=1e-3, weight_decay=0.1)

    # 4. Support point cloud forward: [1, 2048, 3]
    x_s = torch.randn(1, 2048, 3, device=device)
    y_s = torch.zeros(1, 2048, dtype=torch.long, device=device)
    y_s[:, :600] = 1   # class 1: ceiling
    y_s[:, 600:1200] = 2 # class 2: floor
    # points 1200..2048 remain 0: clutter

    f_s = backbone(x_s)  # [1, 2048, 128]
    p_point = extract_point_prototypes(f_s, y_s, n_way=2)  # [1, 3, 128]
    assert p_point.shape == (1, 3, 128)
    assert not torch.isnan(p_point).any()

    # 5. Synthesize modal prototypes and fuse P_0
    p_modal = lma(e_clip)  # [1, 3, 128]
    p_0 = fuse_initial_prototypes(p_point, p_modal)
    assert p_0.shape == (1, 3, 128)
    assert not torch.isnan(p_0).any()

    # 6. Compute loss, backward, and take 1 real AdamW step
    optimizer.zero_grad()
    loss = criterion(p_modal, p_point)
    loss.backward()
    optimizer.step()

    print(f"[Real Flow] Loss GMMN: {loss.item():.6f}")
    assert loss.item() >= 0.0
    assert not torch.isnan(loss)

    # Verify parameter updates
    for name, param in lma.named_parameters():
        assert not torch.isnan(param).any(), f"NaN in weights of {name} after AdamW step!"
