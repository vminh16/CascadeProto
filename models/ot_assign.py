"""P11.5 of [DECISION D-48 amendment 2]: unbalanced entropic optimal transport as a label-free assignment rule.
**Beyond the paper.**

Rows are the episode's classes (background first), columns are the query points of all the episode's query blocks
taken as one set (each foreground class lives mostly in its own block). With cost C = 1 - L (L the current logits
in cosine units), row masses a (from the support, `support_masses`) and uniform column masses b, the plan solves

    min_T <T, C> + eps KL(T | a b^T) + rho KL(T 1 | a) + rho KL(T^T 1 | b)

by log-domain Sinkhorn scaling [Chizat et al. 2018]: log u = lam (log a - LSE(log K + log v)), the same for v, with
lam = rho / (rho + eps). The per-point decision is argmax_c (L_c + eps log u_c): the plan shifts each class by one
bias fitted to the masses. rho -> 0 returns L unchanged; rho -> inf with small eps matches the row masses.

Why not the semi-relaxed form of the maintainer's critique (fix F14): with row sums fixed to a and a column cap
kappa / P, the mass must reach at least P / kappa points, which forces support mass into the background of a small
object. Both marginals are relaxed here instead.
"""

from typing import Tuple

import torch

ITERS = 200


def support_masses(support_y: torch.Tensor) -> torch.Tensor:
    """Row masses a [N+1]: way w gets its support foreground share / N, the background the rest.

    support_y [N, K, P] binary. With one query block per way, class w fills about its own block's share of one of
    the N blocks. The share is density-biased (the sampler), hence P11.5's leak-free guard.
    """
    n_way = support_y.shape[0]
    share = support_y.float().mean(dim=(1, 2))  # [N]
    fg = share / n_way  # [N]
    return torch.cat([(1.0 - fg.sum()).reshape(1), fg])  # [N+1]


def uot_log_plan(logits: torch.Tensor, row_mass: torch.Tensor, eps: float, rho: float,
                 iters: int = ITERS) -> Tuple[torch.Tensor, torch.Tensor]:
    """(log T [B_q, P, R], log u [R]) for logits [B_q, P, R] over all points of the episode's query blocks.

    The points of the B_q blocks form one column set of B_q * P equal masses; sums over it run over the block and
    point axes together (no reshape).
    """
    if eps <= 0 or rho <= 0:
        raise ValueError(f"eps and rho must be positive, got {eps}, {rho}")
    if row_mass.shape[0] != logits.shape[-1] or bool((row_mass <= 0).any()):
        raise ValueError("row masses must be positive, one per logit column")
    x = logits.double()
    log_k = (x - 1.0) / eps  # [B_q, P, R], -C / eps with C = 1 - L
    log_a = torch.log(row_mass.double() / row_mass.double().sum())  # [R]
    n_cols = x.shape[0] * x.shape[1]
    log_b = torch.full(x.shape[:2], -torch.log(torch.tensor(float(n_cols), dtype=torch.float64)).item(),
                       dtype=torch.float64, device=x.device)  # [B_q, P]
    lam = rho / (rho + eps)
    log_u = torch.zeros(x.shape[-1], dtype=torch.float64, device=x.device)  # [R]
    log_v = torch.zeros_like(log_b)  # [B_q, P]
    for _ in range(iters):
        kv = torch.logsumexp(log_k + log_v.unsqueeze(-1), dim=(0, 1))  # [R]
        log_u = lam * (log_a - kv)
        ktu = torch.logsumexp(log_k + log_u, dim=-1)  # [B_q, P]
        log_v = lam * (log_b - ktu)
    return log_u + log_k + log_v.unsqueeze(-1), log_u  # [B_q, P, R], [R]


def ot_logits(logits: torch.Tensor, row_mass: torch.Tensor, eps: float, rho: float,
              iters: int = ITERS) -> torch.Tensor:
    """L + eps log u [B_q, P, R]: the per-point decision of the plan, in the logits' units."""
    _, log_u = uot_log_plan(logits, row_mass, eps, rho, iters)
    return logits + (eps * log_u).to(logits.dtype)
