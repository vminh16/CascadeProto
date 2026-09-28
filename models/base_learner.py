"""Block [2] of [DECISION D-48]: a base learner that marks the base classes as background. **Beyond the paper.**

An MLP on unit point features predicts the fold's base classes plus "none" (test classes and clutter). It is trained
apart, on frozen features, with no gradient into the encoder (amendment 1, fix F1). It enters any head post hoc
through a one-sided background term (amendments 1 and 3, fixes F2 and F21):

    g(x)      = max_{j in kept base} softmax over {kept base, none} of z(x)          clamped to [G_EPS, 1 - G_EPS]
    L[x, bg] += psi * (-log(1 - g(x)))                                               >= 0, increasing in g

A symmetric psi * logit(g) would lower the background logit wherever g < 1/2, which covers most of the test
background (fix F2). Excluding classes renormalises the softmax over the kept columns (leave-target-out, fix F3);
at test nothing is excluded, since every episode class is novel.
"""

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

G_EPS = 1e-4  # clamp of g (fix F2): -log(1 - g) <= -log(G_EPS) ~ 9.2
HIDDEN = 128


class BaseLearner(nn.Module):
    """Unit features [..., D] -> logits [..., J + 1] over the J base classes (fold order) and "none" (last)."""

    def __init__(self, dim: int, n_base: int, hidden: int = HIDDEN):
        super().__init__()
        if n_base < 1:
            raise ValueError(f"need at least one base class, got {n_base}")
        self.n_base = n_base
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, n_base + 1))

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return self.net(F.normalize(f.detach(), dim=-1))  # [..., J + 1]; detach: no gradient into the encoder (F1)


def base_score(logits: torch.Tensor, exclude: Optional[Sequence[int]] = None, eps: float = G_EPS) -> torch.Tensor:
    """g [...]: the largest base-class probability after renormalising over the kept columns.

    logits [..., J + 1] with "none" last; `exclude` lists base columns (0..J-1) removed from the softmax
    (leave-target-out, fix F3). "none" is never excluded.
    """
    n_base = logits.shape[-1] - 1
    keep = torch.ones(n_base + 1, dtype=torch.bool, device=logits.device)  # [J + 1]
    for j in exclude or ():
        if not 0 <= int(j) < n_base:
            raise ValueError(f"excluded column {j} is not a base column (0..{n_base - 1})")
        keep[int(j)] = False
    if not bool(keep[:n_base].any()):
        raise ValueError("every base column is excluded; g is undefined")
    p = torch.softmax(logits.masked_fill(~keep, float("-inf")), dim=-1)  # [..., J + 1]
    return p[..., :n_base].max(dim=-1).values.clamp(eps, 1.0 - eps)  # [...]


def exclusion_term(g: torch.Tensor, psi: float) -> torch.Tensor:
    """psi * (-log(1 - g)) [...]: zero at g = 0, one-sided (never lowers the background)."""
    if psi < 0:
        raise ValueError(f"psi must be non-negative, got {psi}")
    return psi * (-torch.log1p(-g))


def apply_exclusion(logits: torch.Tensor, g: torch.Tensor, psi: float) -> torch.Tensor:
    """L' [B_q, P, N+1]: the background column raised by the exclusion term, the ways untouched."""
    bg = logits[..., :1] + exclusion_term(g, psi).unsqueeze(-1)  # [B_q, P, 1]
    return torch.cat([bg, logits[..., 1:]], dim=-1)  # [B_q, P, N+1]


def base_targets(raw: torch.Tensor, base_classes: Sequence[int]) -> torch.Tensor:
    """Training targets [...] of the base learner: column j for raw class base_classes[j], J ("none") otherwise."""
    out = torch.full_like(raw, len(base_classes))
    for j, c in enumerate(base_classes):
        out[raw == int(c)] = j
    return out


def train_base_learner(x: torch.Tensor, y: torch.Tensor, n_base: int, epochs: int, lr: float = 1e-3,
                       batch: int = 8192, seed: int = 0) -> BaseLearner:
    """Plain CE on (unit feature [M, D], target [M]) pairs, Adam, shuffled mini-batches; deterministic for a seed."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    torch.manual_seed(seed)
    model = BaseLearner(x.shape[-1], n_base).to(x.device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        order = torch.randperm(x.shape[0], generator=g).to(x.device)  # [M]
        for i in range(0, x.shape[0], batch):
            idx = order[i:i + batch]
            loss = F.cross_entropy(model(x[idx]), y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return model.eval()
