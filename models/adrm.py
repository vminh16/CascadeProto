"""Attention-based Dynamic Routing Mechanism of §3.5, Eq.24-25 (spec 02 §6, spec 01 §2.6).

One episode at a time: stage logits `L^t [B_q, 2048, N+1]` for t = 1..T and query features
`F^q [B_q, 2048, D]` give `L_final [B_q, 2048, N+1]`.
"""

from typing import Sequence

import torch
import torch.nn as nn


class DynamicRouting(nn.Module):
    """w_gate = softmax(W_g AvgPool(F^q)) ∈ R^T per query, L_final = Σ_t w_gate^(t) L^t (Eq.24-25).

    W_g ∈ R^{T×D} without bias, as printed [PAPER Eq.24]. Only built for T >= 2: with one stage the
    softmax is identically 1 and W_g would never receive a gradient, so CascadeProto returns L^1
    directly, which is the same prediction [DECISION D-17].
    """

    def __init__(self, num_stages: int, dim: int = 128):
        super().__init__()
        if num_stages < 2:
            raise ValueError(f"ADRM needs at least 2 stages, got {num_stages} (with T = 1, L_final = L^1)")
        self.w_g = nn.Linear(dim, num_stages, bias=False)

    def weights(self, f_q: torch.Tensor) -> torch.Tensor:
        """f_q [B_q, 2048, D] -> w_gate [B_q, T]; AvgPool over the query's points (Eq.24)."""
        return torch.softmax(self.w_g(f_q.mean(dim=1)), dim=-1)

    def forward(self, stage_logits: Sequence[torch.Tensor], f_q: torch.Tensor) -> torch.Tensor:
        """T tensors L^t [B_q, 2048, N+1] -> L_final [B_q, 2048, N+1] (Eq.25)."""
        if len(stage_logits) != self.w_g.out_features:
            raise ValueError(f"expected {self.w_g.out_features} stage logits, got {len(stage_logits)}")
        w = self.weights(f_q)  # [B_q, T]
        return torch.einsum("bt,tbpc->bpc", w, torch.stack(list(stage_logits)))
