"""Oracle-direction distillation of the pairwise decision functions, training only [DECISION D-29]. **Beyond the paper.**

P0 of D-26 measured that the one-shot error of a trained head sits in its prototypes' directions:
scoring the query with its own class means of unit features gains +8 to +22 mIoU points, and the same
rule with one common norm for the present classes, which depends on the directions alone, gains
+10.95 (S1) / +14.12 (S0) on VIP-Seg's released checkpoints (`results/phase16_r2_pre/`). No fixed
test-time rule recovers it (D-26, D-27, D-28); on a training episode the target is computable from the
query's base-class labels, so the head can be taught it (spec 02 §14).

The comparison is made where the prediction is decided, in logit space. With `L = F^q Mᵀ`, adding one
vector to every prototype adds the same term to every class logit of a point, and components of `M`
orthogonal to the features change no logit, so neither `cos(M_c, O_c)` nor `cos(M_c - M_c', O_c - O_c')`
is determined by the predictions. For each pair of classes c < c' present in query b, over the points
labelled c or c':

    O_bc          = normalise(sum_{i: y_bi = c} f_bi / ||f_bi||)       stop-gradient   [B_q, N+1, D]
    T_bic         = <f_bi, O_bc>                                         stop-gradient   [B_q, P, N+1]
    cos_bcc'      = cos_i(L_bic - L_bic', T_bic - T_bic')                                [B_q, N+1, N+1]
    L_distill     = mean_{(b, c < c') present} 1 - cos_bcc'                               scalar

`L_distill = 0` exactly when, on those points, each pairwise decision function of the model is a
positive multiple of the oracle rule's, so both predict the same class between c and c'. It does not
depend on a common shift or a scale of the logits.
"""

from typing import Tuple

import torch
import torch.nn.functional as F

EPS = 1e-12  # guards the norm of a pairwise difference that is zero on every point


def oracle_directions(f_q: torch.Tensor, labels: torch.Tensor, n_cls: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Unit oracle directions O [B_q, N+1, D] and the presence mask [B_q, N+1] of each class in each query.

    f_q [B_q, P, D] query features, labels [B_q, P] in {0..N}. Rows of absent classes are zero and
    masked out; the caller never reads them.
    """
    onehot = F.one_hot(labels, n_cls).to(f_q.dtype)  # [B_q, P, N+1]
    s = torch.einsum("bpc,bpd->bcd", onehot, F.normalize(f_q, dim=-1))  # [B_q, N+1, D], sum of unit features
    present = onehot.sum(dim=1) > 0  # [B_q, N+1]
    return F.normalize(s, dim=-1), present  # [B_q, N+1, D], [B_q, N+1]


def oracle_logits(f_q: torch.Tensor, labels: torch.Tensor, n_cls: int) -> torch.Tensor:
    """Teacher logits T = F^q Oᵀ [B_q, P, N+1] of the oracle-direction rule, from detached features."""
    f = f_q.detach()  # [B_q, P, D]
    o, _ = oracle_directions(f, labels, n_cls)  # [B_q, N+1, D]
    return torch.einsum("bpd,bcd->bpc", f, o)  # [B_q, P, N+1]


def pair_logit_cosine(logits: torch.Tensor, teacher: torch.Tensor,
                      labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """cos_i(L_c - L_c', T_c - T_c') over the points labelled c or c' [B_q, N+1, N+1], and the mask of pairs
    c < c' both present in the query [B_q, N+1, N+1]. logits, teacher [B_q, P, N+1]; labels [B_q, P]."""
    n = logits.shape[-1]
    onehot = F.one_hot(labels, n).to(logits.dtype)  # [B_q, P, N+1]
    points = onehot.unsqueeze(-1) + onehot.unsqueeze(-2)  # [B_q, P, N+1, N+1], 1 on points of c or c' (c != c')
    da = logits.unsqueeze(-1) - logits.unsqueeze(-2)  # [B_q, P, N+1, N+1], L_c - L_c'
    db = teacher.unsqueeze(-1) - teacher.unsqueeze(-2)  # [B_q, P, N+1, N+1]
    num = (points * da * db).sum(dim=1)  # [B_q, N+1, N+1]
    den = ((points * da * da).sum(dim=1).clamp_min(EPS) * (points * db * db).sum(dim=1).clamp_min(EPS)).sqrt()
    present = onehot.sum(dim=1) > 0  # [B_q, N+1]
    upper = torch.ones(n, n, dtype=torch.bool, device=logits.device).triu(diagonal=1)  # [N+1, N+1], c < c'
    return num / den, present.unsqueeze(2) & present.unsqueeze(1) & upper  # [B_q, N+1, N+1] x 2


def oracle_distill_loss(logits: torch.Tensor, f_q: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """L_distill of 02 §14 from the model's final logits [B_q, P, N+1] and query features [B_q, P, D]; scalar.

    A query whose points all carry one label has no pair; if no query of the episode has one, the loss
    is a zero that keeps the graph.
    """
    cos, mask = pair_logit_cosine(logits, oracle_logits(f_q, labels, logits.shape[-1]), labels)  # [B_q, N+1, N+1]
    if not mask.any():
        return logits.sum() * 0.0  # scalar, no pair to compare
    return (1.0 - cos)[mask].mean()  # scalar


# ------------------------------------------------------------------ diagnostics in prototype space

def cosine_to_oracle(m: torch.Tensor, f_q: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """cos(m_bc, O_bc) [B_q, N+1] and the presence mask [B_q, N+1]; depends on the common shift of m."""
    if m.dim() != 3 or m.shape[0] != f_q.shape[0] or m.shape[2] != f_q.shape[2]:
        raise ValueError(f"prototypes {tuple(m.shape)} do not match F^q {tuple(f_q.shape)}")
    o, present = oracle_directions(f_q.detach(), labels, m.shape[1])  # [B_q, N+1, D], [B_q, N+1]
    return F.cosine_similarity(m, o, dim=-1), present  # [B_q, N+1], [B_q, N+1]


def pairwise_cosine(m: torch.Tensor, o: torch.Tensor, present: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """cos(m_c - m_c', o_c - o_c') [B_q, N+1, N+1] and the mask of pairs c < c' both present [B_q, N+1, N+1].

    Free of the common shift, not of components orthogonal to the features; `pair_logit_cosine` is free
    of both.
    """
    dm = m.unsqueeze(2) - m.unsqueeze(1)  # [B_q, N+1, N+1, D], m_c - m_c'
    do = o.unsqueeze(2) - o.unsqueeze(1)  # [B_q, N+1, N+1, D]
    n = m.shape[1]
    upper = torch.ones(n, n, dtype=torch.bool, device=m.device).triu(diagonal=1)  # [N+1, N+1], c < c'
    mask = present.unsqueeze(2) & present.unsqueeze(1) & upper  # [B_q, N+1, N+1]
    return F.cosine_similarity(dm, do, dim=-1), mask  # [B_q, N+1, N+1], [B_q, N+1, N+1]
