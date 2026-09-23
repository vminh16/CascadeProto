"""EPPM-S: the stripped purification stage of phase 16 [DECISION D-24]. **Beyond the paper.**

Not a reading of any equation: it is the CascadeProto-shaped member of the QUEST / APP / PEM / PDM
family, built to test where the 15 points between the printed EPPM and VIP-Seg's head sit. Against
`models/eppm.py::EPPMStage` it keeps only the two summands that can carry class information and drops
everything the research note showed to be inert:

* kept: the channel correlation of Eq.13-14 (`cross_attn_support` of [DECISION D-23]) and a
  channel-preserving self term, as in QUEST [VIPSEG models/vipseg.py:262-277,370-381];
* dropped: `P_diffuse` (no class index, its fusion weight is driven to 0.008-0.082 by training,
  CHANGELOG 15e), the entropy gate (a fixed even pointwise function of the prototype value, research
  note §3.1), Eq.19's fusion MLP, Eq.20's class-pooled SE, the class weights `w_cls` and the ReLU
  before `W_out`.

One stage: `P^t = LN(W(P_cross + P_self) + P^{t-1})`, 37,888 parameters against 79,395 for an EPPM
stage. Like VIP-Seg's head it expects L2-normalised prototypes at the top of the cascade
[VIPSEG models/vipseg.py:142]; `train.py --l2norm_point_proto true` supplies them (02 §3, D-10).
"""

import math

import torch
import torch.nn as nn

from models.eppm import CROSS_ATTN_SCALES, CROSS_ATTN_SUPPORTS, NUM_TOKENS, PROJ_DIM, pool_tokens


class EPPMSharedStage(nn.Module):
    """One EPPM-S stage: prototypes `[B_q, N+1, D]`, `F^s [N, K, 2048, D]`, `F^q [B_q, 2048, D]`.

    `support="pooled"` (the default here) builds one `A_b ∈ R^{D×D}` per query from the mean of the
    pooled support blocks [DECISION D-23]; `"class_slots"` keeps D-01's one matrix per (query, class
    slot, shot) and averages the two branches over the shots, so that the two readings can be compared
    with everything else held fixed.

    The self term is QUEST's, in the same form: a per-channel gate from the difference of the query's
    and the support's channel covariance, applied to `ψ(P^{t-1})`, which preserves the class structure
    the correlation branch mixes away [VIPSEG models/vipseg.py:370-381].
    """

    def __init__(self, dim: int = 128, cross_attn_scale: str = "sqrt_d", support: str = "pooled",
                 self_branch: bool = True):
        super().__init__()
        if cross_attn_scale not in CROSS_ATTN_SCALES:
            raise ValueError(f"cross_attn_scale must be one of {CROSS_ATTN_SCALES}, got {cross_attn_scale!r}")
        if support not in CROSS_ATTN_SUPPORTS:
            raise ValueError(f"cross_attn_support must be one of {CROSS_ATTN_SUPPORTS}, got {support!r}")
        self.support, self.self_branch, self.dim = support, self_branch, dim
        self.phi = nn.Conv1d(NUM_TOKENS, PROJ_DIM, kernel_size=1, bias=False)  # shared φ [PAPER Eq.13]
        self.psi = nn.Linear(dim, dim)  # ψ [PAPER Eq.14] [VIPSEG models/vipseg.py:222]
        self.reweight = nn.Linear(dim, 1, bias=False)  # W_3 of the self term [VIPSEG models/vipseg.py:377]
        self.w_out = nn.Linear(dim, dim, bias=False)  # [VIPSEG models/vipseg.py:280]
        self.norm = nn.LayerNorm(dim)
        self.scale = math.sqrt(PROJ_DIM if cross_attn_scale == "sqrt_d" else dim)
        self.gram_scale = math.sqrt(dim)  # [VIPSEG models/vipseg.py:373]

    def projections(self, f_s: torch.Tensor, f_q: torch.Tensor):
        """φ of the pooled tokens: `Q' [B_q, d, D]` and `S'` — `[d, D]` pooled, `[N+1, K, d, D]` per slot."""
        q = self.phi(pool_tokens(f_q))  # [B_q, 64, D] -> [B_q, d, D]
        if self.support == "pooled":
            return q, self.phi(pool_tokens(f_s).mean(dim=(0, 1)))  # [64, D] -> [d, D], one S' per episode
        slots = torch.cat([f_s.mean(dim=0, keepdim=True), f_s], dim=0)  # [N+1, K, 2048, D] [D-01 item 5]
        s = self.phi(pool_tokens(slots).flatten(0, 1)).unflatten(0, slots.shape[:2])  # [N+1, K, d, D]
        return q, s

    def cross(self, q: torch.Tensor, s: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Row-softmax channel correlation applied to `v = ψ(P^{t-1})` [B_q, N+1, D] -> [B_q, N+1, D]."""
        if self.support == "pooled":
            a = torch.softmax(torch.einsum("bri,rj->bij", q, s) / self.scale, dim=-1)  # [B_q, D, D]
            return torch.einsum("bij,bcj->bci", a, v)  # [B_q, N+1, D]
        a = torch.softmax(torch.einsum("bri,ckrj->bckij", q, s) / self.scale, dim=-1)  # [B_q,N+1,K,D,D]
        return torch.einsum("bckij,bcj->bci", a, v) / a.shape[2]  # mean over the K shots

    def self_gate(self, q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """σ(W_3(Q'ᵀQ' − S'ᵀS')/√D): `[B_q, 1, D]` pooled, `[B_q, N+1, D]` per class slot (shot mean)."""
        g_q = torch.einsum("bri,brj->bij", q, q)  # [B_q, D, D], channel covariance of the query
        if self.support == "pooled":
            g_s = torch.einsum("ri,rj->ij", s, s)  # [D, D]
            delta = (g_q - g_s) / self.gram_scale  # [B_q, D, D]
            return torch.sigmoid(self.reweight(delta).squeeze(-1)).unsqueeze(1)  # [B_q, 1, D]
        g_s = torch.einsum("ckri,ckrj->ckij", s, s)  # [N+1, K, D, D]
        delta = (g_q[:, None, None] - g_s) / self.gram_scale  # [B_q, N+1, K, D, D]
        return torch.sigmoid(self.reweight(delta).squeeze(-1)).mean(dim=2)  # [B_q, N+1, D]

    def forward(self, p_prev: torch.Tensor, f_s: torch.Tensor, f_q: torch.Tensor) -> torch.Tensor:
        """p_prev [B_q, N+1, D], f_s [N, K, 2048, D], f_q [B_q, 2048, D] -> P^t [B_q, N+1, D]."""
        if p_prev.dim() != 3 or p_prev.shape[0] != f_q.shape[0] or p_prev.shape[1] != f_s.shape[0] + 1:
            raise ValueError(f"P^(t-1) {tuple(p_prev.shape)} does not match F^s {tuple(f_s.shape)} and "
                             f"F^q {tuple(f_q.shape)}")
        q, s = self.projections(f_s, f_q)
        v = self.psi(p_prev)  # [B_q, N+1, D]
        update = self.cross(q, s, v)  # [B_q, N+1, D]
        if self.self_branch:
            update = update + self.self_gate(q, s) * v  # channel-preserving term [VIPSEG models/vipseg.py:274]
        return self.norm(self.w_out(update) + p_prev)  # [B_q, N+1, D]
