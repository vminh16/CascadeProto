"""Point-level support → query attention neck before the prototypes [DECISION D-33]. **Beyond the paper.**

P4 showed that E1's gap to the oracle is a joint, query-conditioned shift of every prototype, and VIP-Seg's
head sees the query only through max-pooled channel statistics. The neck gives the support points a
point-level view of the query before the masked means are taken (spec 02 §15):

    F_s' = F_s + alpha * softmax((LN(F_s) W_Q)(LN(F_q) W_K)ᵀ / sqrt(d)) (LN(F_q) W_V) W_O

one head, d = 64, every support point of the episode against every query point, alpha a scalar initialised
to 0 so that the model starts as the checkpoint it is warm-started from. The query features are unchanged.
"""

import math

import torch
import torch.nn as nn

NECKS = ("none", "sq_attn")
NECK_DIM = 64  # d [DECISION D-33]


class SupportQueryAttention(nn.Module):
    """F_s [N, K, P, D], F_q [B_q, P, D] -> F_s' [N, K, P, D]; identity while alpha = 0."""

    def __init__(self, dim: int = 128, d: int = NECK_DIM, alpha_init: float = 0.0):
        super().__init__()
        self.norm_s = nn.LayerNorm(dim)
        self.norm_q = nn.LayerNorm(dim)
        self.w_q = nn.Linear(dim, d, bias=False)
        self.w_k = nn.Linear(dim, d, bias=False)
        self.w_v = nn.Linear(dim, d, bias=False)
        self.w_o = nn.Linear(d, dim, bias=False)
        # ReZero-style gate, 0 for a warm start [DECISION D-33]; 0.1 from scratch [DECISION D-34]
        self.alpha = nn.Parameter(torch.full((), float(alpha_init)))
        self.scale = math.sqrt(d)

    def forward(self, f_s: torch.Tensor, f_q: torch.Tensor) -> torch.Tensor:
        if f_s.dim() != 4 or f_q.dim() != 3 or f_s.shape[-1] != f_q.shape[-1]:
            raise ValueError(f"F_s {tuple(f_s.shape)} and F_q {tuple(f_q.shape)} do not match 02 §2")
        n, k, p, dim = f_s.shape
        s = f_s.reshape(n * k * p, dim)  # [N*K*P, D]; way, shot and point axes of the support only
        q = f_q.reshape(-1, dim)  # [B_q*P, D]; every query point of the episode
        att = torch.softmax(self.w_q(self.norm_s(s)) @ self.w_k(self.norm_q(q)).T / self.scale, dim=-1)  # [NKP, BqP]
        update = self.w_o(att @ self.w_v(self.norm_q(q)))  # [N*K*P, D]
        return f_s + self.alpha * update.view(n, k, p, dim)  # [N, K, P, D]


def build_neck(kind: str, dim: int = 128, alpha_init: float = 0.0):
    if kind == "none":
        return None
    if kind == "sq_attn":
        return SupportQueryAttention(dim, alpha_init=alpha_init)
    raise ValueError(f"neck must be one of {NECKS}, got {kind!r}")
