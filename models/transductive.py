"""Query-side, entropy-weighted EM refinement of the final prototypes [DECISION D-26]. **Beyond the paper.**

The principled form of "entropy-aware prototype purification" (research note 2026-09-22 §5.2): the
entropy is that of each query point's class posterior, the quantity that is high exactly where the
model cannot tell the classes apart, and the purified prototype is estimated from the query's own
confident points, the term that carries most of the one-shot error (research note §2.1, QGE T1).

It is applied on top of a trained model's final scoring rule `L = F^q M^T`, with no parameters, so
it runs on any checkpoint (probe P0 of D-26). For t = 1..T, with the prior `M` held fixed:

    E-step   r_ic   = softmax_c(L^{t-1}_ic)                                 [B_q, P, N+1]
    weights  w_i    = 1 - H(r_i) / ln(N+1)       (weight = entropy)          [B_q, P]
    M-step   u_c    = (1/P) sum_i w_i r_ic f_i / ||f_i||                     [B_q, N+1, D]
    update   mu_c   = ||m_c|| * normalise(m_c/||m_c|| + kappa u_c)           [B_q, N+1, D]
    logits   L^t_ic = <f_i, mu_c>                                            [B_q, P, N+1]

The update is the MAP mean direction of a von Mises-Fisher mixture with a vMF prior centred on the
model's prototype (research note 2026-09-22 §5.2): ||u_c|| is at most the confident mass fraction of
class c, so a class the query barely contains barely moves, and unit features make kappa comparable
across models whose feature norms differ. (A first version normalised u_c to unit length; test EM-3
showed that it hands a class with a vanishing confident mass a full-weight update [DECISION D-26].)

`||m_c||` is kept so that only the prototype's direction changes: the trained rule is an unnormalised
dot product [VIPSEG models/vipseg.py:162-164] [PAPER Eq.23], whose prototype norms are part of the
decision. With `kappa = 0`, or a class without any confident mass, the logits are returned unchanged.
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

WEIGHTS = ("entropy", "none", "ssp", "oracle")
# SSP's self-support thresholds: foreground points above 0.7, background points above 0.6 [SSP §3.2]
SSP_FG_THRESHOLD, SSP_BG_THRESHOLD = 0.7, 0.6


def normalized_entropy(r: torch.Tensor) -> torch.Tensor:
    """H(r_i) / ln(N+1) in [0, 1] for posteriors r [B_q, P, N+1] -> [B_q, P]."""
    h = -(r * torch.log(r.clamp_min(1e-12))).sum(dim=-1)  # [B_q, P], 0 * log 0 := 0
    return h / math.log(r.shape[-1])  # [B_q, P]


def responsibilities(logits: torch.Tensor, weight: str, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Weighted responsibilities `w_i r_ic` [B_q, P, N+1] from logits [B_q, P, N+1].

    entropy: soft posterior times 1 - H/ln(N+1); none: soft posterior; ssp: hard pseudo-labels of
    points above SSP's thresholds; oracle: the query labels (upper bound, never a result).
    """
    n_cls = logits.shape[-1]
    if weight == "oracle":
        if labels is None:
            raise ValueError("weight='oracle' needs the query labels")
        return F.one_hot(labels, n_cls).to(logits.dtype)  # [B_q, P, N+1]
    r = torch.softmax(logits, dim=-1)  # [B_q, P, N+1]
    if weight == "none":
        return r  # [B_q, P, N+1]
    if weight == "entropy":
        return r * (1.0 - normalized_entropy(r)).unsqueeze(-1)  # [B_q, P, N+1]
    if weight == "ssp":
        conf, arg = r.max(dim=-1)  # [B_q, P], [B_q, P]
        thr = torch.where(arg == 0, torch.full_like(conf, SSP_BG_THRESHOLD),
                          torch.full_like(conf, SSP_FG_THRESHOLD))  # [B_q, P]
        keep = (conf > thr).to(logits.dtype)  # [B_q, P]
        return F.one_hot(arg, n_cls).to(logits.dtype) * keep.unsqueeze(-1)  # [B_q, P, N+1]
    raise ValueError(f"weight must be one of {WEIGHTS}, got {weight!r}")


def em_step(f_q: torch.Tensor, prior: torch.Tensor, logits: torch.Tensor, kappa: float, weight: str,
            labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """One E/M step anchored on `prior` [B_q, N+1, D]; returns (prototypes, logits) [B_q,N+1,D], [B_q,P,N+1]."""
    wr = responsibilities(logits, weight, labels)  # [B_q, P, N+1]
    u = torch.einsum("bpc,bpd->bcd", wr, F.normalize(f_q, dim=-1)) / f_q.shape[1]  # [B_q, N+1, D], |u_c| <= mass
    norm = prior.norm(dim=-1, keepdim=True)  # [B_q, N+1, 1]
    direction = F.normalize(prior / norm.clamp_min(1e-12) + kappa * u, dim=-1)  # [B_q, N+1, D]
    mu = norm * direction  # [B_q, N+1, D], the prior's norm, a purified direction
    return mu, torch.einsum("bpd,bcd->bpc", f_q, mu)  # [B_q, N+1, D], [B_q, P, N+1]


def em_refine(f_q: torch.Tensor, prior: torch.Tensor, steps: int, kappa: float, weight: str,
              labels: Optional[torch.Tensor] = None) -> torch.Tensor:
    """T EM steps from the model's own logits `F^q prior^T`; returns the final logits [B_q, P, N+1].

    f_q [B_q, P, D] and prior [B_q, N+1, D] are the features and the prototypes of the model's own
    scoring rule. `steps = 0` or `kappa = 0` returns that rule's logits unchanged.
    """
    if f_q.dim() != 3 or prior.dim() != 3 or prior.shape[0] != f_q.shape[0] or prior.shape[2] != f_q.shape[2]:
        raise ValueError(f"F^q {tuple(f_q.shape)} and prototypes {tuple(prior.shape)} do not match")
    logits = torch.einsum("bpd,bcd->bpc", f_q, prior)  # [B_q, P, N+1], the model's own L
    for _ in range(steps):
        _, logits = em_step(f_q, prior, logits, kappa, weight, labels)  # [B_q, P, N+1]
    return logits


def entropy_diagnostics(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Does 1 - H/ln(N+1) single out correct points? Counts per weight bin, for the probe's report.

    Returns per bin [0, .5), [.5, .9), [.9, 1]: point count and count of correct argmax predictions,
    plus the sum of normalised entropies. All tensors are on the logits' device.
    """
    r = torch.softmax(logits, dim=-1)  # [B_q, P, N+1]
    w = 1.0 - normalized_entropy(r)  # [B_q, P]
    correct = (r.argmax(dim=-1) == labels)  # [B_q, P]
    edges = torch.tensor([0.5, 0.9], device=w.device, dtype=w.dtype)  # bin edges
    bins = torch.bucketize(w, edges, right=True)  # [B_q, P] in {0, 1, 2}; edges[i-1] <= w < edges[i]
    count = torch.bincount(bins.flatten(), minlength=3)  # [3]
    hit = torch.bincount(bins.flatten(), weights=correct.flatten().to(w.dtype), minlength=3)  # [3]
    return {"count": count, "correct": hit, "entropy_sum": (1.0 - w).sum()}
