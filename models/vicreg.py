"""Variance and covariance terms of VICReg on the point features, against dimensional collapse [DECISION D-45].
**Beyond the paper.**

P9 (D-44 outcome) measured the clean base's point features collapsed to about C - 1 = 6 directions (participation
ratio 5.56 of 128) with seven training classes. VICReg's two regularisers act on exactly the two symptoms of that
collapse [Bardes, Ponce, LeCun, ICLR 2022, Eqs. 1-4, re-checked in the PDF]:

    v(Z) = (1/d) Σ_j max(0, γ - sqrt(Var(z^j) + ε))        every dimension keeps a standard deviation of γ
    c(Z) = (1/d) Σ_{i≠j} C(Z)_ij²                           the dimensions are decorrelated

with Z [n, d] the point features of one episode's query blocks, Var and C over the n points (C with 1/(n - 1),
Eq. 3), γ = 1 and ε = 1e-4 (§4.2). The invariance term is not used: the episode's cross-entropy plays its role.
Unlike the paper, which applies the loss to an 8192-d expander output (§4), the terms act on the 128-d features the
prototypes are built from, the space P9 found collapsed.
"""

import torch

GAMMA = 1.0  # [Bardes et al. 2022, Eq. 1-2, §4.1]
EPS = 1e-4  # [Bardes et al. 2022, §4.2]


def variance_term(z: torch.Tensor, gamma: float = GAMMA, eps: float = EPS) -> torch.Tensor:
    """v(Z) for z [n, d] -> scalar."""
    if z.dim() != 2 or z.shape[0] < 2:
        raise ValueError(f"z must be [n >= 2, d], got {tuple(z.shape)}")
    std = torch.sqrt(z.var(dim=0) + eps)  # [d], S(z^j, ε) of Eq. 2 (1/(n-1), as C in Eq. 3)
    return torch.relu(gamma - std).mean()  # scalar


def covariance_term(z: torch.Tensor) -> torch.Tensor:
    """c(Z) for z [n, d] -> scalar: squared off-diagonal covariances summed, divided by d."""
    if z.dim() != 2 or z.shape[0] < 2:
        raise ValueError(f"z must be [n >= 2, d], got {tuple(z.shape)}")
    n, d = z.shape
    zc = z - z.mean(dim=0)  # [n, d]
    cov = zc.T @ zc / (n - 1)  # [d, d]
    off = cov - torch.diag(torch.diagonal(cov))  # [d, d], zero diagonal
    return off.pow(2).sum() / d  # scalar


def vicreg_regulariser(f_q: torch.Tensor, var_weight: float, cov_weight: float) -> torch.Tensor:
    """μ v(Z) + ν c(Z) with Z the query point features f_q [B_q, P, D] flattened over blocks and points."""
    z = f_q.reshape(-1, f_q.shape[-1])  # [B_q·P, D]: the block and point axes only, the channel axis kept
    return var_weight * variance_term(z) + cov_weight * covariance_term(z)
