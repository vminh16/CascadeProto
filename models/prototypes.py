"""Point prototypes by masked average pooling (Eq.3, spec 02 §3).

Pure PyTorch: importing this module does not need mamba_ssm or pointnet2_ops.
"""

import torch

# Background prototype when no support point is background [VIPSEG models/vipseg.py:111-113]
EMPTY_BACKGROUND_VALUE = 0.1


def point_prototypes(support_feat: torch.Tensor, support_mask: torch.Tensor) -> torch.Tensor:
    """P_point = [P_bg; P_fg^(1); ...; P_fg^(N)] (02 §3).

    Args:
        support_feat: F^s [N, K, P, D], features of every support point.
        support_mask: Y^s [N, K, P], binary; 1 = the point belongs to its way's class.
    Returns:
        [N+1, D]. Row k (1..N) is the mean of way k's mask-1 points over its K shots; row 0 is the
        mean of all mask-0 points of all ways and shots [VIPSEG models/vipseg.py:108-130]. No L2
        normalisation [DECISION D-10].
    """
    if support_feat.dim() != 4:
        raise ValueError(f"support_feat must be [N, K, P, D], got {tuple(support_feat.shape)}")
    if support_mask.shape != support_feat.shape[:3]:
        raise ValueError(f"support_mask {tuple(support_mask.shape)} != {tuple(support_feat.shape[:3])}")
    if not torch.isin(support_mask, torch.tensor([0, 1], device=support_mask.device)).all():
        raise ValueError("support_mask must be binary {0, 1} (02 §1)")

    fg = support_mask.to(support_feat.dtype)  # [N, K, P]
    bg = 1.0 - fg  # [N, K, P]

    fg_count = fg.sum(dim=(1, 2))  # [N]
    if (fg_count == 0).any():
        empty = (fg_count == 0).nonzero().flatten().tolist()
        raise ValueError(f"ways {empty} have no foreground support point; the loader guarantees >= 101")
    p_fg = torch.einsum("nkp,nkpd->nd", fg, support_feat) / fg_count[:, None]  # [N, D]

    bg_count = bg.sum()  # scalar
    if bg_count == 0:
        p_bg = support_feat.new_full((support_feat.shape[-1],), EMPTY_BACKGROUND_VALUE)  # [D]
    else:
        p_bg = torch.einsum("nkp,nkpd->d", bg, support_feat) / bg_count  # [D]

    return torch.cat([p_bg[None], p_fg], dim=0)  # [N+1, D]
