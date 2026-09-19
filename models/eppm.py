"""Entropy-aware Prototype Purification Module of §3.4 (spec 02 §5, spec 01 §2.4).

One episode at a time: prototypes `P [B_q, N+1, D]` (one copy per query, 02 §4.5), support features
`F^s [N, K, 2048, D]`, query features `F^q [B_q, 2048, D]`. Class row 0 = background.
"""

import torch
import torch.nn as nn

ENTROPY_EPS = 1e-8  # epsilon of Eq.10 [PAPER Eq.10]
PROB_CLAMP = 1e-7  # p in [1e-7, 1 - 1e-7] before Eq.10, implementation detail [DECISION D-16]
THETA_INIT = 0.5  # initial gate threshold [PAPER Eq.11]


def channel_entropy(x: torch.Tensor) -> torch.Tensor:
    """Eq.10 per channel: H = -p log(p + eps) - (1 - p) log(1 - p + eps), p = sigmoid(x); same shape as x.

    Natural logarithm, so H lies in [0, ln 2] (02 §5.1).
    """
    p = torch.sigmoid(x).clamp(PROB_CLAMP, 1.0 - PROB_CLAMP)
    return -p * torch.log(p + ENTROPY_EPS) - (1.0 - p) * torch.log(1.0 - p + ENTROPY_EPS)


class EntropyGate(nn.Module):
    """Eq.11-12 on the incoming prototype, channel-wise: P_gated = P ⊙ sigmoid(2(θ - H(P))) [DECISION D-02].

    One learnable scalar θ per stage [PAPER §3.5]. `enabled=False` is the `use_gate=false` switch:
    g ≡ 1 and no θ [DECISION D-17].
    """

    def __init__(self, enabled: bool = True):
        super().__init__()
        self.enabled = enabled
        if enabled:
            self.theta = nn.Parameter(torch.tensor(THETA_INIT))

    def gate(self, p: torch.Tensor) -> torch.Tensor:
        """g [B_q, N+1, D] for p [B_q, N+1, D]."""
        if not self.enabled:
            return torch.ones_like(p)
        return torch.sigmoid(2.0 * (self.theta - channel_entropy(p)))

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        return p * self.gate(p)  # [B_q, N+1, D] (Eq.12)
