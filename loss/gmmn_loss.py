"""
Decoupled Generative Moment Matching Network (GMMN) Loss.
Implements:
1. Pairwise squared Euclidean distance with non-negative clamping.
2. Multi-Scale Gaussian RBF Kernel across bandwidths sigma in {2, 5, 10, 20, 40, 80}.
3. Maximum Mean Discrepancy (MMD) with numerical stability clamping.
4. Decoupled foreground-background distribution matching objective:
   L_GMMN = 0.1 * MMD(P_modal_bg, P_point_bg) + 1.0 * (1/N) * sum_k MMD(P_modal_fg_k, P_point_fg_k)

Reference:
- 02_TENSOR_MATH_SPEC.md (Section 3 & Section 7)
- 03_MULTIMODAL_SPEC.md (Section 3.2)
- 05_VERIFICATION_PLAN.md (Section 2.1)
"""

from typing import List, Sequence, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

# Multi-scale Gaussian bandwidths as hardcoded invariant
RBF_BANDWIDTHS = (2.0, 5.0, 10.0, 20.0, 40.0, 80.0)


def pairwise_sq_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Computes pairwise squared Euclidean distance:
        ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * x * y^T
    Args:
        x: [..., M, D]
        y: [..., K, D]
    Returns:
        dist_sq: [..., M, K] clamped to [0.0, inf).
    """
    # x_norm: [..., M, 1]
    x_norm = (x ** 2).sum(dim=-1, keepdim=True)
    # y_norm: [..., 1, K]
    y_norm = (y ** 2).sum(dim=-1, keepdim=True).transpose(-1, -2)
    # Cross product: [..., M, K]
    cross = torch.matmul(x, y.transpose(-1, -2))
    dist_sq = x_norm + y_norm - 2.0 * cross
    # Strict numerical clamp to eliminate floating point negatives
    return torch.clamp(dist_sq, min=0.0)


def multi_scale_rbf_kernel(
    x: torch.Tensor,
    y: torch.Tensor,
    bandwidths: Sequence[float] = RBF_BANDWIDTHS,
) -> torch.Tensor:
    """
    Computes multi-scale Gaussian RBF kernel:
        K(x, y) = sum_{sigma} exp(- ||x - y||^2 / (2 * sigma^2))
    Args:
        x: [..., M, D]
        y: [..., K, D]
        bandwidths: Tuple of scale parameters.
    Returns:
        Gram matrix K: [..., M, K].
        Mathematical invariant: when x == y, diagonal elements K(x_i, x_i) == len(bandwidths) == 6.0.
    """
    dist_sq = pairwise_sq_distance(x, y)  # [..., M, K]
    kernel_val = torch.zeros_like(dist_sq)

    for sigma in bandwidths:
        gamma = 1.0 / (2.0 * (sigma ** 2))
        kernel_val = kernel_val + torch.exp(-gamma * dist_sq)

    return kernel_val


def compute_mmd_squared(
    x: torch.Tensor,
    y: torch.Tensor,
    bandwidths: Sequence[float] = RBF_BANDWIDTHS,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Computes Maximum Mean Discrepancy squared (MMD^2) between two sample sets:
        MMD^2(X, Y) = (1/M^2) sum K(X, X) + (1/K^2) sum K(Y, Y) - (2/MK) sum K(X, Y)
    Args:
        x: [..., M, D]
        y: [..., K, D]
        bandwidths: Gaussian kernel scales.
        eps: Numerical stability threshold.
    Returns:
        mmd_sq: [...] scalar or batch of scalars clamped to [0.0, inf).
    """
    M = x.shape[-2]
    K = y.shape[-2]

    k_xx = multi_scale_rbf_kernel(x, x, bandwidths)  # [..., M, M]
    k_yy = multi_scale_rbf_kernel(y, y, bandwidths)  # [..., K, K]
    k_xy = multi_scale_rbf_kernel(x, y, bandwidths)  # [..., M, K]

    mean_xx = k_xx.sum(dim=(-2, -1)) / float(M * M)
    mean_yy = k_yy.sum(dim=(-2, -1)) / float(K * K)
    mean_xy = k_xy.sum(dim=(-2, -1)) / float(M * K)

    mmd_sq = mean_xx + mean_yy - 2.0 * mean_xy
    return torch.clamp(mmd_sq, min=0.0)


def compute_mmd(
    x: torch.Tensor,
    y: torch.Tensor,
    bandwidths: Sequence[float] = RBF_BANDWIDTHS,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Computes standard MMD metric:
        MMD(X, Y) = sqrt(MMD^2(X, Y))
    Uses smooth numerical formulation sqrt(mmd_sq + eps) - sqrt(eps) ensuring:
    1. Identity of indiscernibles: MMD(X, X) == 0.0 exactly.
    2. Finite, non-exploding gradient at 0: bounded by 1/(2*sqrt(eps)).
    """
    import math
    mmd_sq = compute_mmd_squared(x, y, bandwidths, eps)
    return torch.sqrt(torch.clamp(mmd_sq, min=0.0) + eps) - math.sqrt(eps)



class DecoupledGMMNLoss(nn.Module):
    """
    Decoupled GMMN Alignment Loss for CascadeProto.
    Weights foreground distribution alignment with factor 1.0 and background alignment with factor 0.1:
        L_GMMN = 0.1 * MMD(P_modal_bg, P_point_bg) + 1.0 * (1/N) * sum_k MMD(P_modal_fg_k, P_point_fg_k)
    """
    def __init__(
        self,
        bg_weight: float = 0.1,
        fg_weight: float = 1.0,
        bandwidths: Sequence[float] = RBF_BANDWIDTHS,
        use_squared: bool = False,
    ):
        super().__init__()
        self.bg_weight = bg_weight
        self.fg_weight = fg_weight
        self.bandwidths = bandwidths
        self.use_squared = use_squared

    def forward(
        self,
        p_modal: torch.Tensor,
        p_point: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            p_modal: [B, N+1, D] synthesized modal prototypes from LMA.
            p_point: [B, N+1, D] support geometric prototypes from VIP-Seg backbone.
        Returns:
            loss: Scalar non-negative loss tensor.
        """
        assert p_modal.shape == p_point.shape, (
            f"Shape mismatch: p_modal {p_modal.shape} vs p_point {p_point.shape}"
        )
        assert p_modal.shape[-2] >= 2, (
            f"Expected at least 1 background + 1 foreground class, got num_classes={p_modal.shape[-2]}"
        )

        num_classes = p_modal.shape[-2]
        N_way = num_classes - 1

        mmd_fn = compute_mmd_squared if self.use_squared else compute_mmd

        # 1. Background alignment (class index 0): [B, 1, D]
        p_modal_bg = p_modal[..., 0:1, :]
        p_point_bg = p_point[..., 0:1, :]
        loss_bg = mmd_fn(p_modal_bg, p_point_bg, self.bandwidths).mean()

        # 2. Foreground alignment (classes 1..N): per-class alignment averaged over classes
        fg_losses = []
        for k in range(1, num_classes):
            p_modal_k = p_modal[..., k:k+1, :]
            p_point_k = p_point[..., k:k+1, :]
            loss_k = mmd_fn(p_modal_k, p_point_k, self.bandwidths).mean()
            fg_losses.append(loss_k)

        loss_fg = torch.stack(fg_losses).mean()

        # 3. Decoupled combination
        total_loss = self.bg_weight * loss_bg + self.fg_weight * loss_fg
        return total_loss
