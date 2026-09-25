"""Trained self-support prototypes [DECISION D-39]. **Beyond the paper.**

P6 (D-38) found that adapting a prototype row to the query's own predicted region helps the background without
training and hurts the foreground. This module makes the same rule trainable: per query block, the points predicted
as class c (argmax of `F^q Rᵀ`, no gradient through the assignment) give `S_c = n(sum of their unit features)`, and
`R_c <- n(alpha_r R_c + (1 - alpha_r) S_c)` when at least `min_points` points are predicted c; `alpha_bg` and
`alpha_fg` are learned through a sigmoid. Every operation is per query block, so the module is order-free.
"""

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

MIN_POINTS = 16  # as P6 [DECISION D-38]
ALPHA_BG_INIT, ALPHA_FG_INIT = 0.25, 0.5  # P6's frozen background value; neutral foreground [DECISION D-39]


def logit(p: float) -> float:
    return math.log(p / (1.0 - p))


class SelfSupport(nn.Module):
    def __init__(self, steps: int, min_points: int = MIN_POINTS, alpha_bg: float = ALPHA_BG_INIT,
                 alpha_fg: float = ALPHA_FG_INIT):
        super().__init__()
        if steps < 1:
            raise ValueError(f"self-support needs at least one step, got {steps}")
        self.steps, self.min_points = steps, min_points
        self.theta_bg = nn.Parameter(torch.tensor(logit(alpha_bg)))
        self.theta_fg = nn.Parameter(torch.tensor(logit(alpha_fg)))

    def alphas(self, n_cls: int) -> torch.Tensor:
        """[N+1] mixing weights: alpha_bg for row 0, alpha_fg for the foreground rows."""
        return torch.sigmoid(torch.cat([self.theta_bg.reshape(1), self.theta_fg.reshape(1).expand(n_cls - 1)]))

    def forward(self, f_q: torch.Tensor, rows: torch.Tensor) -> List[torch.Tensor]:
        """f_q [B_q, P, D], rows R_0 [B_q, N+1, D] (unit) -> [R_1 .. R_T], each [B_q, N+1, D]."""
        u = F.normalize(f_q, dim=-1)  # [B_q, P, D]
        a = self.alphas(rows.shape[1]).to(rows.dtype)[None, :, None]  # [1, N+1, 1]
        out = []
        for _ in range(self.steps):
            with torch.no_grad():
                pred = torch.einsum("bpd,bcd->bpc", f_q, rows).argmax(dim=-1)  # [B_q, P]
                onehot = F.one_hot(pred, rows.shape[1]).to(f_q.dtype)  # [B_q, P, N+1]
            counts = onehot.sum(dim=1)  # [B_q, N+1]
            s = F.normalize(torch.einsum("bpc,bpd->bcd", onehot, u), dim=-1)  # [B_q, N+1, D]
            mixed = F.normalize(a * rows + (1.0 - a) * s, dim=-1)  # [B_q, N+1, D]
            rows = torch.where((counts >= self.min_points).unsqueeze(-1), mixed, rows)  # [B_q, N+1, D]
            out.append(rows)
        return out
