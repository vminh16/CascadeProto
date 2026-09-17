"""End-to-End Episode Smoke Pass for CascadeProto.

Verification Plan Phase 2 (tests/test_smoke_episode.py):
1. End-to-end forward pass on synthetic episodic batch (2-way 1-shot).
2. Joint loss calculation: L_total = L_seg + 1.0 * L_GMMN.
3. Full backward autodiff: zero NaNs across all trainable parameters.
4. One optimizer step of AdamW (lr=1e-3, weight_decay=0.1) parameter update.
5. Real CLIP text embedding integration.
6. Execution on both CPU and CUDA (when available).
"""

import pytest
import torch
import torch.optim as optim

from models.cascadeproto import CascadeProto
from models.lma import generate_clip_text_embeddings
from loss.segmentation_loss import CascadeProtoLoss


def build_synthetic_episode(
    B: int = 1,
    N_way: int = 2,
    K_shot: int = 1,
    N_p: int = 2048,
    device: torch.device = torch.device("cpu")
):
    """Generates synthetic episode conforming to Phase 2 contract."""
    num_classes = N_way + 1  # 3

    # Support point clouds: [B, N_way * K_shot, N_p, 3]
    X_s = torch.randn(B, N_way * K_shot, N_p, 3, device=device, dtype=torch.float32)

    # Support binary masks: [B, N_way * K_shot, N_p]
    # Assign foreground points to class index 1 and 2
    Y_s = torch.zeros(B, N_way * K_shot, N_p, device=device, dtype=torch.float32)
    # Give first 500 points in shot 0 to class 1, first 500 points in shot 1 to class 2
    Y_s[:, 0, :500] = 1.0
    Y_s[:, 1, :500] = 2.0

    # Query point clouds: [B, N_p, 3]
    X_q = torch.randn(B, N_p, 3, device=device, dtype=torch.float32)

    # Query labels: [B, N_p] in {0, 1, 2}
    Y_q = torch.randint(0, num_classes, (B, N_p), device=device, dtype=torch.int64)

    # Synthetic CLIP text embeddings: [B, num_classes, 512]
    E_text = torch.randn(B, num_classes, 512, device=device, dtype=torch.float32)
    E_text = E_text / E_text.norm(dim=-1, keepdim=True)

    return X_s, Y_s, X_q, Y_q, E_text


def test_smoke_episode_forward_shapes():
    """Verify full forward pass output shapes and contracts."""
    device = torch.device("cpu")
    model = CascadeProto(input_points=2048, d_feature=128, d_subspace=72, num_stages=4).to(device)

    X_s, Y_s, X_q, Y_q, E_text = build_synthetic_episode(device=device)

    # Forward execution
    output = model(X_s, Y_s, X_q, text_embeddings=E_text, n_way=2)

    # 1. Tuple unpacking contract
    logits, p_modal, p_point = output.logits, output.p_modal, output.p_point

    # 2. Shape assertions
    assert logits.shape == (1, 2048, 3), f"Expected logits [1, 2048, 3], got {logits.shape}"
    assert p_modal.shape == (1, 3, 128), f"Expected p_modal [1, 3, 128], got {p_modal.shape}"
    assert p_point.shape == (1, 3, 128), f"Expected p_point [1, 3, 128], got {p_point.shape}"
    assert output.w_gate.shape == (1, 4), f"Expected w_gate [1, 4], got {output.w_gate.shape}"
    assert len(output.stage_logits) == 4
    assert len(output.stage_prototypes) == 4

    # 3. No NaNs or Infs
    assert not torch.isnan(logits).any(), "NaN in output logits"
    assert not torch.isnan(p_modal).any(), "NaN in p_modal"
    assert not torch.isnan(p_point).any(), "NaN in p_point"
    assert not torch.isnan(output.w_gate).any(), "NaN in w_gate"


def test_smoke_episode_joint_loss_and_backward():
    """Verify joint loss calculation, full autodiff, and zero NaNs across all gradients."""
    device = torch.device("cpu")
    model = CascadeProto(input_points=2048, d_feature=128, d_subspace=72, num_stages=4).to(device)
    criterion = CascadeProtoLoss(lambda_gmmn=1.0, w_bg=0.8, w_fg=1.0).to(device)

    X_s, Y_s, X_q, Y_q, E_text = build_synthetic_episode(device=device)

    output = model(X_s, Y_s, X_q, text_embeddings=E_text, n_way=2)

    total_loss, seg_loss, gmmn_loss = criterion(
        output.logits, Y_q, output.p_modal, output.p_point
    )

    # 1. Non-negative loss assertions
    assert total_loss.item() > 0.0, f"Total loss must be positive, got {total_loss.item()}"
    assert seg_loss.item() > 0.0, f"Seg loss must be positive, got {seg_loss.item()}"
    assert gmmn_loss.item() >= 0.0, f"GMMN loss must be non-negative, got {gmmn_loss.item()}"

    # 2. Joint loss identity: total_loss = seg_loss + 1.0 * gmmn_loss
    expected_total = seg_loss + 1.0 * gmmn_loss
    assert torch.allclose(total_loss, expected_total, atol=1e-5), (
        f"L_total mismatch: {total_loss.item()} vs {expected_total.item()}"
    )

    # 3. Autodiff backward pass
    total_loss.backward()

    # 4. Check all trainable parameters for valid, finite gradients
    trainable_params_count = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params_count += 1
            assert param.grad is not None, f"Parameter {name} has None gradient"
            assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradient"
            assert not torch.isinf(param.grad).any(), f"Parameter {name} has Inf gradient"

    assert trainable_params_count > 0, "No trainable parameters found"


def test_smoke_episode_optimizer_step():
    """Verify one AdamW optimizer step updates parameters without divergence."""
    device = torch.device("cpu")
    model = CascadeProto(input_points=2048, d_feature=128, d_subspace=72, num_stages=4).to(device)
    criterion = CascadeProtoLoss(lambda_gmmn=1.0).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)

    X_s, Y_s, X_q, Y_q, E_text = build_synthetic_episode(device=device)

    # Record parameter clone before step
    params_before = [p.clone() for p in model.parameters() if p.requires_grad]

    optimizer.zero_grad()
    output = model(X_s, Y_s, X_q, text_embeddings=E_text, n_way=2)
    total_loss, _, _ = criterion(output.logits, Y_q, output.p_modal, output.p_point)
    total_loss.backward()
    optimizer.step()

    # Verify parameters updated and remain finite
    params_after = [p for p in model.parameters() if p.requires_grad]
    any_changed = False
    for p_before, p_after in zip(params_before, params_after):
        assert not torch.isnan(p_after).any(), "NaN parameter detected after optimizer step"
        assert not torch.isinf(p_after).any(), "Inf parameter detected after optimizer step"
        if not torch.equal(p_before, p_after):
            any_changed = True

    assert any_changed, "Optimizer step did not update any parameters"


def test_smoke_episode_with_real_clip():
    """Verify integration with real CLIP text embeddings extracted for category names."""
    device = torch.device("cpu")
    class_names = ["chair", "table"]

    # Extract real CLIP embeddings for 2-way setting
    E_text = generate_clip_text_embeddings(class_names, device=device)
    assert E_text.shape == (1, 3, 512)

    model = CascadeProto(input_points=2048, d_feature=128, d_subspace=72, num_stages=4).to(device)
    criterion = CascadeProtoLoss().to(device)

    X_s, Y_s, X_q, Y_q, _ = build_synthetic_episode(device=device)

    output = model(X_s, Y_s, X_q, text_embeddings=E_text, n_way=2)
    total_loss, seg_loss, gmmn_loss = criterion(
        output.logits, Y_q, output.p_modal, output.p_point
    )

    total_loss.backward()
    assert output.logits.shape == (1, 2048, 3)
    assert not torch.isnan(total_loss).any()


def test_smoke_episode_cuda():
    """Verify end-to-end forward/backward on CUDA GPU if available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available on this host")

    device = torch.device("cuda")
    model = CascadeProto(input_points=2048, d_feature=128, d_subspace=72, num_stages=4).to(device)
    criterion = CascadeProtoLoss().to(device)

    X_s, Y_s, X_q, Y_q, E_text = build_synthetic_episode(device=device)

    output = model(X_s, Y_s, X_q, text_embeddings=E_text, n_way=2)
    total_loss, seg_loss, gmmn_loss = criterion(
        output.logits, Y_q, output.p_modal, output.p_point
    )

    total_loss.backward()

    assert output.logits.is_cuda
    assert output.logits.shape == (1, 2048, 3)
    assert not torch.isnan(total_loss).any()
