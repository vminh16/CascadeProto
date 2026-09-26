"""Pure operations of the density-invariant encoder [DECISION D-43]. **Beyond the paper.**

Kept apart from `models/density_encoder.py` because that file imports the VIP-Seg encoder, which needs
`mamba_ssm` and `pointnet2_ops` (GPU environment only); these functions are tested on the CPU.
"""

from typing import Tuple

import torch

BALL_RADII: Tuple[float, ...] = (0.1, 0.2, 0.4)  # metres, one per encoder stage [DECISION D-43]
BALL_K = 16  # neighbours per centre, VIP-Seg's k [models/vipseg_backbone.py ENCODER_CONFIG]
METRIC_CHANNELS = slice(0, 3)  # xyz in metres from the block minimum [VIPSEG dataloaders/loader.py:65-66]


def ball_group(xyz: torch.Tensor, centers: torch.Tensor, radius: float, k: int) -> torch.Tensor:
    """Indices [B, G, k] of the points within `radius` of each centre, the first k in point order, padded with the
    first one when fewer (PointNet++'s ball query) [DECISION D-43].

    xyz [B, N, 3], centers [B, G, 3]. A centre that is one of the points always finds itself. A centre with no point
    within `radius` raises (it would have no neighbourhood).
    """
    if radius <= 0 or k <= 0:
        raise ValueError(f"radius and k must be positive, got {radius}, {k}")
    n = xyz.shape[1]
    dist = torch.cdist(centers, xyz, compute_mode="donot_use_mm_for_euclid_dist")  # [B, G, N]
    order = torch.arange(n, device=xyz.device).expand_as(dist)  # [B, G, N]
    key = torch.where(dist <= radius, order, torch.full_like(order, n))  # outside the ball -> n
    idx = key.sort(dim=-1).values[..., :k]  # [B, G, k], the first k inside in point order
    first = idx[..., :1]  # [B, G, 1]
    if bool((first == n).any()):
        raise ValueError(f"a centre has no point within {radius} m")
    return torch.where(idx == n, first.expand_as(idx), idx)  # [B, G, k]


def per_block_scale(t: torch.Tensor, eps: float) -> torch.Tensor:
    """t / (std of t over every axis but the first + eps): VIP-Seg's `t / (torch.std(t) + eps)` with the statistic of
    each block instead of the whole batch [VIPSEG models/encoder.py:182-184] [DECISION D-43]."""
    dims = tuple(range(1, t.dim()))
    return t / (t.std(dim=dims, keepdim=True) + eps)


def per_block_standardize(t: torch.Tensor, eps: float) -> torch.Tensor:
    """(t - mean) / (std(t - mean) + eps) with the mean and std of each block (every axis but the first), in place of
    VIP-Seg's batch-global `t.mean()` / `torch.std` [VIPSEG models/encoder.py:282-284, 413-415, 582-584]
    [DECISION D-43]."""
    dims = tuple(range(1, t.dim()))
    centred = t - t.mean(dim=dims, keepdim=True)
    return centred / (centred.std(dim=dims, keepdim=True) + eps)
