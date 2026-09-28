"""Prototype alignment of the query's class means to the support's prototypes, training only [DECISION D-46].
**Beyond the paper.**

On the clean base CR the cosine oracle (the query's own unit class directions) scores 80.84 on valid against U's
55.96 with the same features (P9): the error is δ = p̂_support − μ_query. D-29 aligned the model's pairwise
decisions point by point, which CE already constrains on training episodes, and moved no score. This loss states
δ itself: each class mean of a query block must sit on its own support prototype rather than on the others.

    μ_bc  = n(Σ_{i: y_bi = c} n(f_bi))                  query class direction    [B_q, N+1, D]  (oracle_directions)
    p̂_c'  = n(Σ n(f_s)) over the support points of c'   U's rows                 [N+1, D]       (unit_prototypes)
    L     = mean_{(b, c) present} −log softmax_{c'}(⟨μ_bc, p̂_c'⟩ / τ)[c]

The softmax over the other prototypes keeps the trivial solution (every feature on one direction) from lowering
the loss. Gradients reach both the query and the support features.
"""

import torch
import torch.nn.functional as F

from models.oracle_distill import oracle_directions
from models.prototypes import unit_prototypes

TAU = 0.1  # [DECISION D-46], a common supervised-contrastive temperature (a convention, not measured)


def alignment_loss(f_s: torch.Tensor, support_y: torch.Tensor, f_q: torch.Tensor, query_y: torch.Tensor,
                   tau: float = TAU) -> torch.Tensor:
    """f_s [N, K, P, D], support_y [N, K, P] in {0, 1}, f_q [B_q, P, D], query_y [B_q, P] in {0..N} -> scalar."""
    if not (tau > 0):
        raise ValueError(f"tau must be > 0, got {tau}")
    n_cls = f_s.shape[0] + 1
    rows = unit_prototypes(f_s, support_y)  # [N+1, D]
    mu, present = oracle_directions(f_q, query_y, n_cls)  # [B_q, N+1, D], [B_q, N+1]
    logits = torch.einsum("bcd,kd->bck", mu, rows) / tau  # [B_q, N+1 (query class), N+1 (support row)]
    target = torch.arange(n_cls, device=f_q.device).expand(f_q.shape[0], -1)  # [B_q, N+1], class c -> row c
    nll = F.cross_entropy(logits.reshape(-1, n_cls), target.reshape(-1), reduction="none")  # [B_q·(N+1)]
    keep = present.reshape(-1)  # [B_q·(N+1)], absent classes have no query mean
    return nll[keep].mean()  # scalar
