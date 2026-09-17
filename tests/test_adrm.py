"""Unit tests for Attention-Based Dynamic Routing Module (ADRM).

Verifies mathematical invariants:
1. Routing Simplex Constraint: sum_{t=1}^T w_gate^(t) = 1.0 +- 1e-6.
2. Convex Logit Aggregation: convex combination properties.
3. Multi-Stage Gradient Flow: all T stages receive non-zero, finite gradients.
4. Tensor shape contracts and CPU/CUDA device execution.
"""

import pytest
import torch

from models.adrm import AttentionDynamicRouting


def test_invariant_1_routing_simplex_constraint():
    """Invariant 1: ADRM gating weights lie on the probability simplex sum(w_gate) = 1.0."""
    B, N_q, D, T = 4, 1024, 128, 4
    num_classes = 3

    adrm = AttentionDynamicRouting(dim=D, num_stages=T)

    F_q = torch.randn(B, N_q, D)
    stage_logits = [torch.randn(B, N_q, num_classes) for _ in range(T)]

    L_final, w_gate = adrm(F_q, stage_logits)

    # 1. Check shape of w_gate: [B, T]
    assert w_gate.shape == (B, T), f"Expected w_gate shape {(B, T)}, got {w_gate.shape}"

    # 2. Strict bounds: w_gate in (0, 1)
    assert (w_gate > 0.0).all(), "Gate probabilities must be strictly positive"
    assert (w_gate < 1.0).all(), "Gate probabilities must be strictly less than 1.0"

    # 3. Simplex sum: sum_{t=1}^T w_gate^(t) == 1.0
    gate_sums = w_gate.sum(dim=-1)
    assert torch.allclose(gate_sums, torch.ones_like(gate_sums), atol=1e-6), (
        f"Gating weights must sum to 1.0, max error: {(gate_sums - 1.0).abs().max().item()}"
    )


def test_invariant_2_convex_logit_aggregation():
    """Invariant 2: Convex combination properties of aggregated logits."""
    B, N_q, D, T = 2, 512, 128, 4
    num_classes = 3

    adrm = AttentionDynamicRouting(dim=D, num_stages=T)

    F_q = torch.randn(B, N_q, D)

    # Case A: If all stages output the exact same logits S, L_final must equal S identically
    identical_logit = torch.randn(B, N_q, num_classes)
    stage_logits = [identical_logit.clone() for _ in range(T)]

    L_final, w_gate = adrm(F_q, stage_logits)

    assert torch.allclose(L_final, identical_logit, atol=1e-5), (
        "When all stages have identical logits, convex combination must equal the same logits"
    )

    # Case B: Check bounded interpolation
    # If S_1 <= S_2 <= S_3 <= S_4, then S_1 <= L_final <= S_4
    ordered_logits = [
        torch.full((B, N_q, num_classes), fill_value=float(t))
        for t in range(1, T + 1)
    ]
    L_ordered, _ = adrm(F_q, ordered_logits)
    assert (L_ordered >= 1.0 - 1e-5).all(), "Convex combination violated lower bound"
    assert (L_ordered <= float(T) + 1e-5).all(), "Convex combination violated upper bound"


def test_invariant_3_multi_stage_gradient_flow():
    """Invariant 3: Gradients flow to ALL T intermediate stage logits and query features."""
    B, N_q, D, T = 2, 256, 128, 4
    num_classes = 3

    adrm = AttentionDynamicRouting(dim=D, num_stages=T)

    F_q = torch.randn(B, N_q, D, requires_grad=True)
    stage_logits = [
        torch.randn(B, N_q, num_classes, requires_grad=True)
        for _ in range(T)
    ]

    L_final, w_gate = adrm(F_q, stage_logits)

    # Loss touches aggregated logits
    loss = L_final.sum()
    loss.backward()

    # 1. Every single intermediate stage logit must receive non-zero, non-NaN gradients
    for t in range(T):
        grad_t = stage_logits[t].grad
        assert grad_t is not None, f"Stage {t+1} logits received None gradient"
        assert not torch.isnan(grad_t).any(), f"Stage {t+1} logits received NaN gradient"
        norm = grad_t.norm().item()
        assert norm > 0.0, f"Stage {t+1} logits received zero gradient (norm={norm})"

    # 2. Query features must receive non-zero, non-NaN gradients via gating network
    assert F_q.grad is not None and not torch.isnan(F_q.grad).any()
    assert F_q.grad.norm().item() > 0.0

    # 3. Projection matrix W_g must receive gradients
    assert adrm.W_g.weight.grad is not None and not torch.isnan(adrm.W_g.weight.grad).any()
    assert adrm.W_g.weight.grad.norm().item() > 0.0


def test_invariant_4_dimension_contract_cpu_and_cuda():
    """Invariant 4: Dimension contracts on CPU and CUDA at full query resolution (N_q = 2048)."""
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")

    for dev_name in devices:
        device = torch.device(dev_name)
        B, N_q, D, T = 4, 2048, 128, 4
        num_classes = 3

        adrm = AttentionDynamicRouting(dim=D, num_stages=T).to(device)
        F_q = torch.randn(B, N_q, D, device=device)
        stage_logits = [
            torch.randn(B, N_q, num_classes, device=device)
            for _ in range(T)
        ]

        L_final, w_gate = adrm(F_q, stage_logits)

        assert L_final.shape == (B, N_q, num_classes), f"{dev_name} L_final shape mismatch"
        assert w_gate.shape == (B, T), f"{dev_name} w_gate shape mismatch"
        assert not torch.isnan(L_final).any(), f"{dev_name} NaNs in L_final"
        assert not torch.isnan(w_gate).any(), f"{dev_name} NaNs in w_gate"
