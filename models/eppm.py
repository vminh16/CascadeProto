"""Entropy-aware Prototype Purification Module of §3.4 (spec 02 §5, spec 01 §2.4).

One episode at a time: prototypes `P [B_q, N+1, D]` (one copy per query, 02 §4.5), support features
`F^s [N, K, 2048, D]`, query features `F^q [B_q, 2048, D]`. Class row 0 = background.
"""

import math

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


# ---------------------------------------------------------------- cross-attention (02 §5.2, D-01)

POOL_WINDOW = 32  # MaxPool1d(32, stride=32): 2048 points -> 64 tokens [VIPSEG models/vipseg.py:213]
NUM_POINTS = 2048  # [PAPER §4.1]
NUM_TOKENS = NUM_POINTS // POOL_WINDOW  # 64, the input channels of phi
PROJ_DIM = 72  # d [PAPER Eq.13] [PAPER §4.1]
CROSS_ATTN_SCALES = ("sqrt_d", "sqrt_D")  # [DECISION D-01]


def pool_tokens(f: torch.Tensor) -> torch.Tensor:
    """Max over consecutive windows of 32 points, per channel: [..., 2048, D] -> [..., 64, D]."""
    if f.shape[-2] != NUM_POINTS:
        raise ValueError(f"expected {NUM_POINTS} points, got {f.shape[-2]}")
    return f.unflatten(-2, (NUM_TOKENS, POOL_WINDOW)).amax(dim=-2)


class CrossAttention(nn.Module):
    """Eq.13-14 in the channel-correlation form of [DECISION D-01].

    A[b,c,k] = softmax_row(Q'[b]ᵀ S'[c,k] / √d) ∈ R^{D×D} with Q' = φ(MaxPool(F^q)), S' = φ(MaxPool(F^s_{c,k})),
    one φ shared by query and support; class slot 0 is the way-mean of the support features
    [VIPSEG models/vipseg.py:244]. P_cross[b,c] = (1/K) Σ_k A[b,c,k] ψ(P_gated[b,c]).
    """

    def __init__(self, dim: int = 128, scale: str = "sqrt_d"):
        super().__init__()
        if scale not in CROSS_ATTN_SCALES:
            raise ValueError(f"cross_attn_scale must be one of {CROSS_ATTN_SCALES}, got {scale!r}")
        self.phi = nn.Conv1d(NUM_TOKENS, PROJ_DIM, kernel_size=1, bias=False)  # shared [PAPER Eq.13]
        self.psi = nn.Linear(dim, dim)  # [PAPER Eq.14] [VIPSEG models/vipseg.py:222]
        self.scale = math.sqrt(PROJ_DIM if scale == "sqrt_d" else dim)

    def attention(self, f_s: torch.Tensor, f_q: torch.Tensor) -> torch.Tensor:
        """f_s [N, K, 2048, D], f_q [B_q, 2048, D] -> A [B_q, N+1, K, D, D], rows summing to 1."""
        slots = torch.cat([f_s.mean(dim=0, keepdim=True), f_s], dim=0)  # [N+1, K, 2048, D]
        q = self.phi(pool_tokens(f_q))  # [B_q, 64, D] -> [B_q, d, D]
        s = self.phi(pool_tokens(slots).flatten(0, 1)).unflatten(0, slots.shape[:2])  # [N+1, K, d, D]
        logits = torch.einsum("bri,ckrj->bckij", q, s) / self.scale  # [B_q, N+1, K, D, D]
        return torch.softmax(logits, dim=-1)

    def forward(self, p_gated: torch.Tensor, f_s: torch.Tensor, f_q: torch.Tensor) -> torch.Tensor:
        """p_gated [B_q, N+1, D] -> P_cross [B_q, N+1, D]."""
        a = self.attention(f_s, f_q)  # [B_q, N+1, K, D, D]
        v = self.psi(p_gated)  # [B_q, N+1, D]
        return torch.einsum("bckij,bcj->bci", a, v) / a.shape[2]  # mean over the K shots
