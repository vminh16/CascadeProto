"""Oracle-prototype distillation of the effective prototype, training only [DECISION D-29]. **Beyond the paper.**

P0 of D-26 measured that the one-shot error of a trained head sits in the *direction* of its effective
prototype `M_eff` (the prototype with `L_final = F^q M_effᵀ`): replacing that direction by the query's
own class mean of unit features, the norm kept, gains +8 to +22 mIoU points
(`results/phase16_p0/SUMMARY.md`), while no fixed test-time rule recovers it (D-26, D-27, D-28).
On a training episode that direction is computable from the query's base-class labels, so the head can
be taught to produce it (spec 02 §14):

    O_bc       = normalise(sum_{i: y_bi = c} f_bi / ||f_bi||)       stop-gradient    [B_q, N+1, D]
    L_distill  = mean_{(b, c) present} 1 - cos(M_eff_bc, O_bc)                        scalar

`O` is the direction of P0's `ORACLE_REPLACE` arm (02 §11 with κ → ∞ and the oracle weight).
"""

from typing import Tuple

import torch
import torch.nn.functional as F


def oracle_directions(f_q: torch.Tensor, labels: torch.Tensor, n_cls: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Unit oracle directions O [B_q, N+1, D] and the presence mask [B_q, N+1] of each class in each query.

    f_q [B_q, P, D] query features, labels [B_q, P] in {0..N}. Rows of absent classes are zero and
    masked out; the caller never reads them.
    """
    onehot = F.one_hot(labels, n_cls).to(f_q.dtype)  # [B_q, P, N+1]
    s = torch.einsum("bpc,bpd->bcd", onehot, F.normalize(f_q, dim=-1))  # [B_q, N+1, D], sum of unit features
    present = onehot.sum(dim=1) > 0  # [B_q, N+1]
    return F.normalize(s, dim=-1), present  # [B_q, N+1, D], [B_q, N+1]


def cosine_to_oracle(m: torch.Tensor, f_q: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """cos(m_bc, O_bc) [B_q, N+1] and the presence mask [B_q, N+1]; O is built from detached features."""
    if m.dim() != 3 or m.shape[0] != f_q.shape[0] or m.shape[2] != f_q.shape[2]:
        raise ValueError(f"prototypes {tuple(m.shape)} do not match F^q {tuple(f_q.shape)}")
    o, present = oracle_directions(f_q.detach(), labels, m.shape[1])  # [B_q, N+1, D], [B_q, N+1]
    return F.cosine_similarity(m, o, dim=-1), present  # [B_q, N+1], [B_q, N+1]


def oracle_distill_loss(m_eff: torch.Tensor, f_q: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """L_distill of 02 §14: mean of 1 - cos(M_eff, O) over the (query, class) pairs present; scalar."""
    cos, present = cosine_to_oracle(m_eff, f_q, labels)  # [B_q, N+1], [B_q, N+1]
    return (1.0 - cos)[present].mean()  # scalar; every query has at least one class present
