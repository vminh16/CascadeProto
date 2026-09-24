"""Training-free text prior after the head, and the gap-decomposition helpers of P3 [DECISION D-31]. **Beyond the paper.**

Text reaches a novel class only through what the base classes teach (research notes 2026-09-24): a
direction `t̂_c` in the checkpoint's CL2N feature geometry, from CLIP embeddings of the class names and the
base-class prototype bank of P1 (`models/base_calibration.py`), either by a closed-form ridge map or by a
CLIP-similarity-weighted mixture of base prototypes. No parameter is trained. The prior re-ranks the
foreground ways only:

    T_ic  = < n(f_i - mu), t̂_c >                                              [B_q, P, N]
    L'_ic = L_ic + kappa * gamma_e * w_i * (T_ic - mean_c' T_ic')    (c >= 1)     [B_q, P, N+1]

with gamma_e = max(0, 2 acc_e - 1) from the text-only accuracy on the support's foreground points, and
w_i = 1 or the normalised entropy of the head's posterior at point i. kappa = 0 returns the head's logits.
"""

import math
from typing import Callable, List, Sequence, Tuple

import torch
import torch.nn.functional as F

from models.base_calibration import centred_unit
from models.transductive import normalized_entropy

PROMPT_SETS = ("template", "bare", "ensemble", "descriptions")
# T0's ensemble (`experiments/t0_text_geometry.py`)
ENSEMBLE = ("a photo of a {}.", "a point cloud of a {}.", "a 3D scan of a {} in a room.", "the {} in an indoor scene.",
            "a {} in an office.", "a rendering of a {}.")
# Frozen in the research agent's probe before any result with them (2026-09-24_text_integration_math.md §1.4)
DESCRIPTIONS = {
    "ceiling": "a large horizontal flat surface at the top of a room",
    "floor": "a large horizontal flat surface at the bottom of a room",
    "wall": "a large vertical flat surface enclosing a room",
    "beam": "a long horizontal structural bar along the ceiling",
    "column": "a tall vertical structural pillar",
    "window": "a flat glass pane set in a wall",
    "door": "a flat vertical panel in a wall that opens",
    "table": "a flat horizontal top on legs, a piece of furniture",
    "chair": "a seat with a backrest and legs, a piece of furniture",
    "sofa": "a large upholstered seat for several people, a piece of furniture",
    "bookcase": "a tall shelf unit filled with books against a wall",
    "board": "a flat whiteboard or blackboard mounted on a wall",
}
RIDGE_LAMBDA = 1e-3  # the agent's probe
BOUNDARY_K = 16  # neighbours in xyz that define a boundary point [DECISION D-31]


def prompts_for(name: str, kind: str) -> List[str]:
    """The prompts whose unit CLIP embeddings are averaged into one class embedding."""
    if kind == "template":
        from models.clip_text import class_prompt

        return [class_prompt(name)]
    if kind == "bare":
        return [name]
    if kind == "ensemble":
        return [t.format(name) for t in ENSEMBLE]
    if kind == "descriptions":
        if name not in DESCRIPTIONS:
            raise ValueError(f"no frozen description for class {name!r}")
        return [f"This point cloud represents the {name}, {DESCRIPTIONS[name]}.", f"{DESCRIPTIONS[name]}."]
    raise ValueError(f"prompt set must be one of {PROMPT_SETS}, got {kind!r}")


def class_embeddings(names: Sequence[str], kind: str,
                     encode: Callable[[List[str]], torch.Tensor]) -> torch.Tensor:
    """Unit class embeddings [C, E]: `encode(prompts) -> [len, E]` rows are L2-normalised, then averaged."""
    rows = []
    for name in names:
        e = F.normalize(encode(prompts_for(name, kind)).float(), dim=-1)  # [M, E]
        rows.append(F.normalize(e.mean(dim=0), dim=-1))  # [E]
    return torch.stack(rows)  # [C, E]


def ridge_directions(e_all: torch.Tensor, novel: Sequence[int], base_idx: Sequence[int],
                     base: torch.Tensor, lam: float = RIDGE_LAMBDA) -> torch.Tensor:
    """t̂ [N, D] = n(Bᵀ(ẼẼᵀ + λI)⁻¹ ẽ_c): kernel ridge from centred class embeddings to base prototypes.

    e_all [C, E] unit embeddings of every class (their mean is the centre, names only); base [J, D] the
    bank's prototypes of the classes `base_idx` (same order).
    """
    e = F.normalize(e_all - e_all.mean(dim=0, keepdim=True), dim=-1).double()  # [C, E]
    eb, en = e[list(base_idx)], e[list(novel)]  # [J, E], [N, E]
    k = eb @ eb.T + lam * torch.eye(eb.shape[0], dtype=e.dtype, device=e.device)  # [J, J]
    coef = torch.linalg.solve(k, eb @ en.T)  # [J, N]
    return F.normalize((coef.T @ base.double()).float(), dim=-1)  # [N, D]


def retrieval_directions(e_all: torch.Tensor, novel: Sequence[int], base_idx: Sequence[int],
                         base: torch.Tensor, tau: float) -> torch.Tensor:
    """t̂ [N, D] = n(Σ_j softmax_j(τ cos(e_c, e_j)) B_j): base prototypes weighted by CLIP similarity."""
    sim = e_all[list(novel)] @ e_all[list(base_idx)].T  # [N, J]
    w = torch.softmax(tau * sim, dim=-1)  # [N, J]
    return F.normalize(w @ base, dim=-1)  # [N, D]


def text_logits(f: torch.Tensor, mu: torch.Tensor, t_hat: torch.Tensor) -> torch.Tensor:
    """T [..., P, N] = <n(f - mu), t̂_c> for features f [..., P, D]."""
    return torch.einsum("...pd,nd->...pn", centred_unit(f, mu), t_hat)  # [..., P, N]


def support_gamma(f_s: torch.Tensor, support_y: torch.Tensor, mu: torch.Tensor,
                  t_hat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """(gamma_e, acc_e): text-only foreground-vs-foreground accuracy on the support's foreground points.

    f_s [N, K, P, D], support_y [N, K, P] binary; way n's foreground points should score highest on t̂_n.
    gamma_e = max(0, 2 acc_e - 1) maps chance (0.5 in 2-way) to 0.
    """
    n_way = f_s.shape[0]
    pred = text_logits(f_s, mu, t_hat).argmax(dim=-1)  # [N, K, P]
    target = torch.arange(n_way, device=f_s.device).view(n_way, 1, 1).expand_as(pred)  # [N, K, P]
    fg = support_y.bool()  # [N, K, P]
    acc = (pred[fg] == target[fg]).float().mean()  # scalar
    chance = 1.0 / n_way
    return ((acc - chance) / (1.0 - chance)).clamp_min(0.0), acc  # scalar, scalar


def entropy_weight(logits: torch.Tensor) -> torch.Tensor:
    """w_i = H(softmax(L_i)) / ln(N+1) in [0, 1], [B_q, P]: large where the head is unsure."""
    return normalized_entropy(torch.softmax(logits, dim=-1))  # [B_q, P]


def apply_prior(logits: torch.Tensor, t: torch.Tensor, kappa: float, gamma: torch.Tensor,
                w: torch.Tensor) -> torch.Tensor:
    """L' [B_q, P, N+1]: foreground columns plus kappa * gamma * w * (T - mean over ways); background as is."""
    centred = t - t.mean(dim=-1, keepdim=True)  # [B_q, P, N], only re-ranks the ways
    fg = logits[..., 1:] + kappa * gamma * w.unsqueeze(-1) * centred  # [B_q, P, N]
    return torch.cat([logits[..., :1], fg], dim=-1)  # [B_q, P, N+1]


def unit_oracle_logits(f_q: torch.Tensor, m_eff: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Logits [B_q, P, N+1] of the oracle rule with one common norm for the present classes (R2-pre)."""
    from models.oracle_distill import oracle_directions

    o, present = oracle_directions(f_q, labels, m_eff.shape[1])  # [B_q, N+1, D], [B_q, N+1]
    norm = m_eff.norm(dim=-1, keepdim=True)  # [B_q, N+1, 1]
    pres = present.unsqueeze(-1).to(norm.dtype)  # [B_q, N+1, 1]
    common = (norm * pres).sum(dim=1, keepdim=True) / pres.sum(dim=1, keepdim=True)  # [B_q, 1, 1]
    m = torch.where(present.unsqueeze(-1), common * o, m_eff)  # [B_q, N+1, D]
    return torch.einsum("bpd,bcd->bpc", f_q, m)  # [B_q, P, N+1]


def alignment_terms(t: torch.Tensor, logits: torch.Tensor, oracle: torch.Tensor,
                    labels: torch.Tensor) -> Tuple[float, float, float]:
    """Sums (Σab, Σa², Σb²) over foreground pairs c < c' and the points labelled c or c', with
    a = T_c - T_c' (text) and b = (O_c - O_c') - (L_c - L_c') (the oracle's correction of the head).
    cos = Σab / sqrt(Σa² Σb²) is formed after pooling over episodes."""
    n = t.shape[-1]
    sab = saa = sbb = 0.0
    for c in range(n):
        for d in range(c + 1, n):
            pts = (labels == c + 1) | (labels == d + 1)  # [B_q, P], foreground ways are columns 1..N
            a = (t[..., c] - t[..., d])[pts]  # [M]
            b = ((oracle[..., c + 1] - oracle[..., d + 1]) - (logits[..., c + 1] - logits[..., d + 1]))[pts]  # [M]
            sab, saa, sbb = sab + float(a @ b), saa + float(a @ a), sbb + float(b @ b)
    return sab, saa, sbb


def text_query_accuracy(t: torch.Tensor, labels: torch.Tensor) -> Tuple[int, int]:
    """(correct, total) of argmax_c T over the query's foreground points (label c -> column c-1)."""
    fg = labels > 0  # [B_q, P]
    pred = t.argmax(dim=-1) + 1  # [B_q, P]
    return int((pred[fg] == labels[fg]).sum()), int(fg.sum())


def boundary_mask(xyz: torch.Tensor, labels: torch.Tensor, k: int = BOUNDARY_K) -> torch.Tensor:
    """[B_q, P] bool: a point whose k nearest neighbours in xyz [B_q, P, 3] include another label."""
    d = torch.cdist(xyz, xyz)  # [B_q, P, P]
    idx = d.topk(k + 1, dim=-1, largest=False).indices[..., 1:]  # [B_q, P, k], the point itself dropped
    neigh = torch.gather(labels.unsqueeze(1).expand(-1, labels.shape[1], -1), 2, idx)  # [B_q, P, k]
    return (neigh != labels.unsqueeze(-1)).any(dim=-1)  # [B_q, P]


def point_accuracy(pred: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> Tuple[int, int]:
    """(correct, total) over the points of `mask` [B_q, P]."""
    return int((pred[mask] == labels[mask]).sum()), int(mask.sum())


def arm_name(source: str, prompt: str, weight: str, kappa: float) -> str:
    return f"{source}_{prompt}_{weight}_k{kappa:g}"


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    """Rank correlation without ties handling beyond average ranks; nan for fewer than 3 pairs."""
    if len(a) < 3:
        return math.nan
    ra = torch.tensor(a).argsort().argsort().double()
    rb = torch.tensor(b).argsort().argsort().double()
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra @ rb) / (ra.norm() * rb.norm()))

