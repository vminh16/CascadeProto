"""Decoupled GMMN alignment loss of Eq.7-8 (spec 02 §4.3-4.4, spec 03 §4).

All functions work on one episode: prototype matrices `[N+1, D]` with row 0 = background and rows
1..N = the foreground ways (02 §0). No batch axis; train.py forwards one episode at a time.
"""

import torch

RBF_BANDWIDTHS = (2.0, 5.0, 10.0, 20.0, 40.0, 80.0)  # sigma set of Eq.7 [PAPER Eq.7]
BACKGROUND_WEIGHT = 0.1  # [PAPER Eq.8]
FOREGROUND_WEIGHT = 1.0  # [PAPER Eq.8]
FG_MODES = ("joint", "per_class")  # [DECISION D-04]


def pairwise_sq_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """||x_i - y_j||^2 for x [M, D], y [L, D] -> [M, L] (02 §9).

    Computed from the differences, not as ||x||^2 + ||y||^2 - 2 x.y: the sets hold at most N+1 rows,
    and the difference form is never negative and gives exactly 0 on the diagonal at any norm.
    """
    return ((x[:, None, :] - y[None, :, :]) ** 2).sum(-1)  # [M, L]


def rbf_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """k(x, y) = sum_sigma exp(-||x - y||^2 / (2 sigma^2)) for x [M, D], y [L, D] -> [M, L] (Eq.7)."""
    d2 = pairwise_sq_dist(x, y)  # [M, L]
    return sum(torch.exp(-d2 / (2.0 * sigma ** 2)) for sigma in RBF_BANDWIDTHS)  # [M, L]


def mmd(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Squared MMD of Eq.7 between the sample sets x [M, D] and y [L, D] -> scalar.

    The biased estimate: the three means include the diagonal terms k(x_i, x_i) = 6 (02 §4.3).
    No square root and no clamp of the result [PAPER Eq.7]; it is >= 0 up to rounding.
    """
    return rbf_kernel(x, x).mean() + rbf_kernel(y, y).mean() - 2.0 * rbf_kernel(x, y).mean()


def gmmn_loss(p_modal: torch.Tensor, p_point: torch.Tensor, fg_mode: str = "joint",
              detach_point: bool = False) -> torch.Tensor:
    """L_GMMN = 0.1 MMD(P_modal^bg, P_point^bg) + 1.0 MMD(P_modal^fg, P_point^fg) (Eq.8) -> scalar.

    p_modal, p_point: [N+1, D]. Background = row 0 (sets of one sample); foreground = rows 1..N taken
    as one set of N samples (fg_mode='joint'), or the mean of N row-by-row MMDs (fg_mode='per_class',
    ablation only) [DECISION D-04]. P_point keeps its gradient unless detach_point [DECISION D-04].
    """
    if fg_mode not in FG_MODES:
        raise ValueError(f"fg_mode must be one of {FG_MODES}, got {fg_mode!r}")
    if p_modal.dim() != 2 or p_modal.shape != p_point.shape or p_modal.shape[0] < 2:
        raise ValueError(f"expected two [N+1, D] prototype matrices with N >= 1, got "
                         f"{tuple(p_modal.shape)} and {tuple(p_point.shape)}")
    if detach_point:
        p_point = p_point.detach()
    loss_bg = mmd(p_modal[:1], p_point[:1])  # 1 vs 1 sample
    if fg_mode == "joint":
        loss_fg = mmd(p_modal[1:], p_point[1:])  # N vs N samples
    else:
        n_way = p_modal.shape[0] - 1
        loss_fg = sum(mmd(p_modal[c:c + 1], p_point[c:c + 1]) for c in range(1, n_way + 1)) / n_way
    return BACKGROUND_WEIGHT * loss_bg + FOREGROUND_WEIGHT * loss_fg
