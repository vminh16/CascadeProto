"""Entropy-aware Prototype Purification Module of §3.4 (spec 02 §5, spec 01 §2.4).

One episode at a time: prototypes `P [B_q, N+1, D]` (one copy per query, 02 §4.5), support features
`F^s [N, K, 2048, D]`, query features `F^q [B_q, 2048, D]`. Class row 0 = background.
"""

import math
from typing import Optional

import torch
import torch.nn as nn

ENTROPY_EPS = 1e-8  # epsilon of Eq.10 [PAPER Eq.10]
PROB_CLAMP = 1e-7  # p in [1e-7, 1 - 1e-7] before Eq.10, implementation detail [DECISION D-16]
THETA_INIT = 0.5  # initial gate threshold [PAPER Eq.11]
GATE_TARGETS = ("prototype", "features")  # [DECISION D-02]
EQ19_SELF = ("none", "gated")  # [DECISION D-19], beyond the paper


def channel_entropy(x: torch.Tensor) -> torch.Tensor:
    """Eq.10 per channel: H = -p log(p + eps) - (1 - p) log(1 - p + eps), p = sigmoid(x); same shape as x.

    Natural logarithm, so H lies in [0, ln 2] (02 §5.1).
    """
    p = torch.sigmoid(x).clamp(PROB_CLAMP, 1.0 - PROB_CLAMP)
    return -p * torch.log(p + ENTROPY_EPS) - (1.0 - p) * torch.log(1.0 - p + ENTROPY_EPS)


class EntropyGate(nn.Module):
    """Eq.11-12 channel-wise on its argument: x_gated = x ⊙ sigmoid(2(θ - H(x))) [PAPER Eq.10-12].

    Elementwise on the last axis, so the same module gates a prototype `[B_q, N+1, D]` or features
    `[N, K, 2048, D]` / `[B_q, 2048, D]`; which one is the `gate_target` switch of [DECISION D-02].
    One learnable scalar θ per stage [PAPER §3.5]. `enabled=False` is the `use_gate=false` switch:
    g ≡ 1 and no θ [DECISION D-17].
    """

    def __init__(self, enabled: bool = True):
        super().__init__()
        self.enabled = enabled
        if enabled:
            self.theta = nn.Parameter(torch.tensor(THETA_INIT))

    def gate(self, p: torch.Tensor) -> torch.Tensor:
        """g, elementwise, same shape as `p`."""
        if not self.enabled:
            return torch.ones_like(p)
        return torch.sigmoid(2.0 * (self.theta - channel_entropy(p)))

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        return p * self.gate(p)  # same shape as p (Eq.12)


# ---------------------------------------------------------------- cross-attention (02 §5.2, D-01)

POOL_WINDOW = 32  # MaxPool1d(32, stride=32): 2048 points -> 64 tokens [VIPSEG models/vipseg.py:213]
NUM_POINTS = 2048  # [PAPER §4.1]
NUM_TOKENS = NUM_POINTS // POOL_WINDOW  # 64, the input channels of phi
PROJ_DIM = 72  # d [PAPER Eq.13] [PAPER §4.1]
CROSS_ATTN_SCALES = ("sqrt_d", "sqrt_D")  # [DECISION D-01]
CROSS_ATTN_NORMS = ("none", "layernorm")  # [DECISION D-18]


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

    Rows of A are a softmax over the **channel** axis, so every channel of P_cross is a convex
    combination of the channels of ψ(P): both a saturated and a uniform A leave P_cross constant along
    D, and the printed 1/√d gives no control over where between the two the module sits.
    `norm="layernorm"` standardises Q' and S' along the projection axis r before the correlation, which
    makes A invariant to the scale of the features and gives 144 learnable parameters for the sharpness.
    It is a probe, not a fix [DECISION D-18].
    """

    def __init__(self, dim: int = 128, scale: str = "sqrt_d", norm: str = "none"):
        super().__init__()
        if scale not in CROSS_ATTN_SCALES:
            raise ValueError(f"cross_attn_scale must be one of {CROSS_ATTN_SCALES}, got {scale!r}")
        if norm not in CROSS_ATTN_NORMS:
            raise ValueError(f"cross_attn_norm must be one of {CROSS_ATTN_NORMS}, got {norm!r}")
        self.phi = nn.Conv1d(NUM_TOKENS, PROJ_DIM, kernel_size=1, bias=False)  # shared [PAPER Eq.13]
        self.psi = nn.Linear(dim, dim)  # [PAPER Eq.14] [VIPSEG models/vipseg.py:222]
        self.scale = math.sqrt(PROJ_DIM if scale == "sqrt_d" else dim)
        # One LayerNorm for both branches, because φ is shared [PAPER Eq.13] [DECISION D-18]
        self.proj_norm = nn.LayerNorm(PROJ_DIM) if norm == "layernorm" else None

    def project(self, pooled: torch.Tensor) -> torch.Tensor:
        """φ on [*, 64, D] -> [*, d, D], standardised along r when `norm="layernorm"` [DECISION D-18]."""
        out = self.phi(pooled)
        if self.proj_norm is None:
            return out
        return self.proj_norm(out.transpose(-1, -2)).transpose(-1, -2)

    def attention(self, f_s: torch.Tensor, f_q: torch.Tensor) -> torch.Tensor:
        """f_s [N, K, 2048, D], f_q [B_q, 2048, D] -> A [B_q, N+1, K, D, D], rows summing to 1."""
        slots = torch.cat([f_s.mean(dim=0, keepdim=True), f_s], dim=0)  # [N+1, K, 2048, D]
        q = self.project(pool_tokens(f_q))  # [B_q, 64, D] -> [B_q, d, D]
        s = self.project(pool_tokens(slots).flatten(0, 1)).unflatten(0, slots.shape[:2])  # [N+1, K, d, D]
        logits = torch.einsum("bri,ckrj->bckij", q, s) / self.scale  # [B_q, N+1, K, D, D]
        return torch.softmax(logits, dim=-1)

    def forward(self, p_gated: torch.Tensor, f_s: torch.Tensor, f_q: torch.Tensor) -> torch.Tensor:
        """p_gated [B_q, N+1, D] -> P_cross [B_q, N+1, D]."""
        a = self.attention(f_s, f_q)  # [B_q, N+1, K, D, D]
        v = self.psi(p_gated)  # [B_q, N+1, D]
        return torch.einsum("bckij,bcj->bci", a, v) / a.shape[2]  # mean over the K shots


# ------------------------------------------------------------------- diffusion (02 §5.3, D-14)

DIFFUSION_TAU = 0.5  # τ [PAPER Eq.18]
DIFFUSION_ALPHA = 0.5  # α [PAPER Eq.18]


def prototype_diffusion(f_s: torch.Tensor, f_q: torch.Tensor) -> torch.Tensor:
    """Eq.15-18 -> P_diffuse [B_q, D]; no parameters and no class index.

    q_ch = σ(mean over the query's points), s_ch = σ(mean over all support points of all ways and
    shots) [DECISION D-16]. Masks use a strict `> τ`. With ReLU features a channel is active exactly
    when its mean is positive, which usually makes c_unique = 0 [DECISION D-14]. The caller broadcasts
    the result to the N+1 class rows [DECISION D-16].
    """
    q_ch = torch.sigmoid(f_q.mean(dim=1))  # [B_q, D] (Eq.15)
    s_ch = torch.sigmoid(f_s.mean(dim=(0, 1, 2)))  # [D] (Eq.15)
    m_q = (q_ch > DIFFUSION_TAU).to(q_ch.dtype)  # [B_q, D]
    m_s = (s_ch > DIFFUSION_TAU).to(s_ch.dtype)  # [D]
    m_common = m_q * m_s  # [B_q, D]
    c_common = (q_ch + s_ch) / 2.0 * m_common  # (Eq.16)
    c_unique = (q_ch * (m_q - m_common) + s_ch * (m_s - m_common)) / 2.0  # (Eq.17)
    return DIFFUSION_ALPHA * c_common + (1.0 - DIFFUSION_ALPHA) * c_unique  # [B_q, D] (Eq.18)


# ------------------------------------------------ fusion, SE, class weights, output (02 §5.4)

SE_REDUCTION = 4  # r [DECISION D-16]
BACKGROUND_CLASS_WEIGHT = 0.8  # w_cls = [0.8, 1.0, ..., 1.0] [PAPER §3.4]
FUSION_WEIGHTS = ("per_query", "per_class")  # [DECISION D-11]


def class_weights(n_classes: int, like: torch.Tensor) -> torch.Tensor:
    """w_cls [N+1] = [0.8, 1, ..., 1]; fixed, not a parameter [PAPER §3.4]."""
    w = torch.ones(n_classes, dtype=like.dtype, device=like.device)
    w[0] = BACKGROUND_CLASS_WEIGHT
    return w


class FusionOutput(nn.Module):
    """Eq.19-21: fusion weights, SE recalibration, class weights, W_out, residual and LayerNorm.

    `eq19_self="gated"` adds a third, channel-preserving summand to Eq.19's P_combined, the way
    VIP-Seg sums `proto_self` with `proto_cross` [VIPSEG models/vipseg.py:274-277]. It is **not in the
    paper** [DECISION D-19]: both summands Eq.19 prints are class-poor, which measurably dilutes the
    only part of a prototype that can change a prediction.
    """

    def __init__(self, dim: int = 128, fusion_weight: str = "per_query", eq19_self: str = "none"):
        super().__init__()
        if fusion_weight not in FUSION_WEIGHTS:
            raise ValueError(f"fusion_weight must be one of {FUSION_WEIGHTS}, got {fusion_weight!r}")
        if eq19_self not in EQ19_SELF:
            raise ValueError(f"eq19_self must be one of {EQ19_SELF}, got {eq19_self!r}")
        self.fusion_weight, self.eq19_self = fusion_weight, eq19_self
        self.fusion = nn.Sequential(nn.Linear(2 * dim, dim), nn.ReLU(), nn.Linear(dim, 2))  # [DECISION D-16]
        self.se_down = nn.Linear(dim, dim // SE_REDUCTION)  # W_1 [DECISION D-16]
        self.se_up = nn.Linear(dim // SE_REDUCTION, dim)  # W_2
        self.w_out = nn.Linear(dim, dim)  # [DECISION D-16]
        self.norm = nn.LayerNorm(dim)

    def weights(self, p_cross: torch.Tensor, p_diffuse: torch.Tensor) -> torch.Tensor:
        """Eq.19 softmax weights: [B_q, 2] (per_query, pooled over classes) or [B_q, N+1, 2] (per_class)."""
        both = torch.cat([p_cross, p_diffuse], dim=-1)  # [B_q, N+1, 2D]
        if self.fusion_weight == "per_query":
            both = both.mean(dim=1)  # [B_q, 2D] [DECISION D-11]
        return torch.softmax(self.fusion(both), dim=-1)

    def excitation(self, p_combined: torch.Tensor) -> torch.Tensor:
        """Eq.20 a = σ(W_2 ReLU(W_1 AvgPool_c(P_combined))): [B_q, D]."""
        return torch.sigmoid(self.se_up(torch.relu(self.se_down(p_combined.mean(dim=1)))))

    def forward(self, p_cross: torch.Tensor, p_diffuse: torch.Tensor, p_prev: torch.Tensor,
                p_self: Optional[torch.Tensor] = None) -> torch.Tensor:
        """All inputs [B_q, N+1, D]; `p_prev` is the ungated P^{t-1} [DECISION D-02] -> P^t [B_q, N+1, D].

        `p_self` is the channel-preserving summand of [DECISION D-19]; it must be None unless
        `eq19_self="gated"`, and is added to P_combined before Eq.20.
        """
        if (p_self is None) == (self.eq19_self == "gated"):
            raise ValueError(f"eq19_self={self.eq19_self!r} does not match p_self={type(p_self).__name__}")
        w = self.weights(p_cross, p_diffuse)  # [B_q, 2] or [B_q, N+1, 2]
        if self.fusion_weight == "per_query":
            w = w[:, None, :]  # [B_q, 1, 2]
        p_combined = w[..., 0:1] * p_cross + w[..., 1:2] * p_diffuse  # (Eq.19)
        if p_self is not None:
            p_combined = p_combined + p_self  # [DECISION D-19]
        p_attended = p_combined * self.excitation(p_combined)[:, None, :]  # (Eq.20)
        p_weighted = p_attended * class_weights(p_prev.shape[1], p_prev)[None, :, None]  # [PAPER §3.4]
        return self.norm(self.w_out(torch.relu(p_weighted)) + p_prev)  # (Eq.21)


class EPPMStage(nn.Module):
    """One EPPM stage (§3.4): P^t = Out(Fuse(CrossAttn(Gate(P^{t-1})), Diffuse(F^s, F^q)), P^{t-1})."""

    def __init__(self, dim: int = 128, use_gate: bool = True, cross_attn_scale: str = "sqrt_d",
                 fusion_weight: str = "per_query", cross_attn_norm: str = "none",
                 gate_target: str = "prototype", eq19_self: str = "none"):
        super().__init__()
        if gate_target not in GATE_TARGETS:
            raise ValueError(f"gate_target must be one of {GATE_TARGETS}, got {gate_target!r}")
        self.gate_target, self.eq19_self = gate_target, eq19_self
        self.gate = EntropyGate(use_gate)
        self.cross = CrossAttention(dim, cross_attn_scale, cross_attn_norm)
        self.out = FusionOutput(dim, fusion_weight, eq19_self)

    def forward(self, p_prev: torch.Tensor, f_s: torch.Tensor, f_q: torch.Tensor) -> torch.Tensor:
        """p_prev [B_q, N+1, D], f_s [N, K, 2048, D], f_q [B_q, 2048, D] -> P^t [B_q, N+1, D].

        `gate_target="features"` is the literal reading of Eq.13-14: "After entropy gating, we apply
        cross-attention" and Eq.14 multiplies ψ(P^{t-1}), the **ungated** prototype, exactly as printed.
        `gate_target="prototype"` gates P^{t-1} instead and feeds ψ the gated prototype, which departs
        from the argument Eq.14 prints. Eq.15-18 read F^q and F^s with no mention of gating, so the
        diffusion branch uses the ungated features either way [DECISION D-02].
        """
        if p_prev.dim() != 3 or p_prev.shape[0] != f_q.shape[0] or p_prev.shape[1] != f_s.shape[0] + 1:
            raise ValueError(f"P^(t-1) {tuple(p_prev.shape)} does not match F^s {tuple(f_s.shape)} and "
                             f"F^q {tuple(f_q.shape)}")
        if self.gate_target == "features":
            p_in, f_s_in, f_q_in = p_prev, self.gate(f_s), self.gate(f_q)  # (Eq.12 on F, then Eq.13)
        else:
            p_in, f_s_in, f_q_in = self.gate(p_prev), f_s, f_q  # (Eq.12 on P^{t-1})
        p_cross = self.cross(p_in, f_s_in, f_q_in)  # [B_q, N+1, D] (Eq.13-14)
        p_diffuse = prototype_diffusion(f_s, f_q)[:, None, :].expand_as(p_cross)  # (Eq.15-18) [DECISION D-16]
        # psi(P_gated), channel by channel, so the class structure of P^{t-1} survives [DECISION D-19]
        p_self = self.cross.psi(self.gate(p_prev)) if self.eq19_self == "gated" else None
        return self.out(p_cross, p_diffuse, p_prev, p_self)  # (Eq.19-21)


def stage_logits(f_q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Eq.23 L^t = F^q (P^t)ᵀ per query, no temperature [DECISION D-10]: [B_q, 2048, D] x [B_q, N+1, D] -> [B_q, 2048, N+1]."""
    return torch.einsum("bpd,bcd->bpc", f_q, p)
