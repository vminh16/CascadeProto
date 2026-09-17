"""Entropy-Aware Prototype Purification Module (EPPM).

Implementation for CascadeProto (ECCV 2026).
Spec references:
- docs/spec/01_ARCHITECTURE_SPEC.md (Section 3)
- docs/spec/02_TENSOR_MATH_SPEC.md (Section 4)
- AGENTS.md (Invariants & Mathematical bounds)
"""

from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_shannon_entropy(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Computes channel-wise Shannon entropy with numerical stability clamping.
    
    Formula:
        p = clamp(sigmoid(x), 1e-7, 1 - 1e-7)
        H = -p * log(p + eps) - (1 - p) * log(1 - p + eps)
    
    Invariant bounds:
        0.0 <= H <= log(2) ~= 0.69314718
    
    Args:
        x: Input tensor of shape [..., D].
        eps: Small stability constant (1e-8).
        
    Returns:
        Entropy tensor of shape [..., D] strictly bounded in [0.0, log(2)].
    """
    # [..., D]
    p = torch.sigmoid(x)
    # Numerical stability clamping as specified in 02_TENSOR_MATH_SPEC.md Section 7
    p_clamped = torch.clamp(p, min=1e-7, max=1.0 - 1e-7)  # [..., D]
    entropy = -p_clamped * torch.log(p_clamped + eps) - (1.0 - p_clamped) * torch.log(1.0 - p_clamped + eps)  # [..., D]
    return entropy


class InformationTheoreticGating(nn.Module):
    """Sub-Module 1: Information-Theoretic Gating.
    
    Calculates per-channel Shannon entropy, modulated by a learnable
    scalar threshold theta initialized to 0.5.
    
    Formula:
        H_i = -p_i log(p_i + eps) - (1 - p_i) log(1 - p_i + eps)
        g_i = sigmoid(2 * (theta - H_i))
        x_gated = x * g
    """

    def __init__(self, init_theta: float = 0.5) -> None:
        super().__init__()
        # Learnable scalar threshold initialized to 0.5
        self.theta = nn.Parameter(torch.tensor(float(init_theta), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for Information-Theoretic Gating.
        
        Args:
            x: Input prototype tensor of shape [B, N+1, D].
            
        Returns:
            Tuple of:
                - x_gated: Gated prototype tensor of shape [B, N+1, D]
                - entropy: Calculated Shannon entropy tensor of shape [B, N+1, D]
        """
        # [B, N+1, D]
        entropy = compute_shannon_entropy(x, eps=1e-8)  # [B, N+1, D]
        gate = torch.sigmoid(2.0 * (self.theta - entropy))  # [B, N+1, D]
        x_gated = x * gate  # [B, N+1, D]
        return x_gated, entropy


class CrossAttentionRefinement(nn.Module):
    """Sub-Module 2: Cross-Attention Refinement.
    
    Projects features into lower-dimensional subspace d = 72 via linear layers,
    computes query-support correlation A_qs, propagates support context F_qs,
    and refines prototypes via cross-attention with query-aligned representations.
    
    Dimensions:
        D = 128 (feature dimension)
        d = 72 (subspace dimension)
    """

    def __init__(self, dim: int = 128, subspace_dim: int = 72) -> None:
        super().__init__()
        self.dim = dim
        self.subspace_dim = subspace_dim
        self.scale = subspace_dim ** 0.5  # sqrt(72)

        # Projections into subspace d = 72
        self.proj_q = nn.Linear(dim, subspace_dim)
        self.proj_s = nn.Linear(dim, subspace_dim)
        self.proj_proto = nn.Linear(dim, subspace_dim)
        self.proj_qs = nn.Linear(dim, subspace_dim)

    def forward(
        self,
        P: torch.Tensor,
        F_s: torch.Tensor,
        F_q: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for Cross-Attention Refinement.
        
        Args:
            P: Input (gated) prototype of shape [B, N+1, D].
            F_s: Support point features of shape [B, N_s, D].
            F_q: Query point features of shape [B, N_q, D].
            
        Returns:
            Tuple of:
                - P_cross: Refined prototype of shape [B, N+1, D]
                - A_qs: Query-support affinity matrix of shape [B, N_q, N_s]
        """
        # 1. Project query and support to subspace d = 72
        Q_prime = self.proj_q(F_q)  # [B, N_q, 72]
        S_prime = self.proj_s(F_s)  # [B, N_s, 72]

        # 2. Pairwise cross-correlation matrix A_qs
        # [B, N_q, 72] x [B, 72, N_s] -> [B, N_q, N_s]
        attn_logits = torch.matmul(Q_prime, S_prime.transpose(1, 2)) / self.scale  # [B, N_q, N_s]
        A_qs = torch.softmax(attn_logits, dim=-1)  # [B, N_q, N_s]

        # 3. Support context propagation into query coordinate frame
        # [B, N_q, N_s] x [B, N_s, D] -> [B, N_q, D]
        F_qs = torch.matmul(A_qs, F_s)  # [B, N_q, D]

        # 4. Cross-attention prototype refinement over query-aligned features
        P_prime = self.proj_proto(P)  # [B, N+1, 72]
        F_qs_prime = self.proj_qs(F_qs)  # [B, N_q, 72]

        # [B, N+1, 72] x [B, 72, N_q] -> [B, N+1, N_q]
        proto_attn_logits = torch.matmul(P_prime, F_qs_prime.transpose(1, 2)) / self.scale  # [B, N+1, N_q]
        A_proto = torch.softmax(proto_attn_logits, dim=-1)  # [B, N+1, N_q]

        # [B, N+1, N_q] x [B, N_q, D] -> [B, N+1, D]
        P_cross = torch.matmul(A_proto, F_qs)  # [B, N+1, D]

        return P_cross, A_qs


class PrototypeDiffusion(nn.Module):
    """Sub-Module 3: Prototype Diffusion.
    
    Decomposes point-averaged channel statistics into common and unique
    components using threshold tau = 0.5, then blends them with alpha = 0.5.
    
    Parameters:
        tau = 0.5 (activation threshold)
        alpha = 0.5 (blend mixing parameter)
    """

    def __init__(self, tau: float = 0.5, alpha: float = 0.5) -> None:
        super().__init__()
        self.tau = tau
        self.alpha = alpha

    def forward(
        self,
        F_s: torch.Tensor,
        F_q: torch.Tensor,
        num_classes: int
    ) -> torch.Tensor:
        """Forward pass for Prototype Diffusion.
        
        Args:
            F_s: Support point features of shape [B, N_s, D].
            F_q: Query point features of shape [B, N_q, D].
            num_classes: Total number of classes (N+1).
            
        Returns:
            P_diffuse: Diffused prototype tensor of shape [B, N+1, D].
        """
        # 1. Spatial channel-mean activations across query and support
        q_mean = F_q.mean(dim=1)  # [B, D]
        s_mean = F_s.mean(dim=1)  # [B, D]
        q_ch = torch.sigmoid(q_mean)  # [B, D]
        s_ch = torch.sigmoid(s_mean)  # [B, D]

        # 2. Indicator masks with activation threshold tau = 0.5
        m_q = (q_ch > self.tau).float()  # [B, D]
        m_s = (s_ch > self.tau).float()  # [B, D]
        m_common = m_q * m_s  # [B, D]

        # 3. Channel feature components: common and unique
        c_common = 0.5 * (q_ch + s_ch) * m_common  # [B, D]
        c_unique = 0.5 * (q_ch * (m_q - m_common) + s_ch * (m_s - m_common))  # [B, D]

        # 4. Blended diffusion prototype with mixing parameter alpha = 0.5
        P_diff = self.alpha * c_common + (1.0 - self.alpha) * c_unique  # [B, D]

        # Broadcast along category dimension to span [B, N+1, D]
        P_diffuse = P_diff.unsqueeze(1).expand(-1, num_classes, -1)  # [B, N+1, D]
        return P_diffuse


class AdaptiveFusionRecalibration(nn.Module):
    """Sub-Module 4: Adaptive Fusion & Class Recalibration.
    
    Fuses P_cross and P_diffuse via a 2-layer MLP, applies Squeeze-and-Excitation
    (SE) channel recalibration, modulates with fixed class attenuation vector
    w_cls = [0.8, 1.0, ..., 1.0], and connects via residual link and LayerNorm.
    
    Formula:
        w = softmax(f_fusion([P_cross; P_diffuse])) in R^2
        P_combined = w_1 P_cross + w_2 P_diffuse
        a = sigmoid(W2_se * ReLU(W1_se * AvgPool(P_combined)))
        P_attended = P_combined * a
        P_weighted = P_attended * w_cls
        P_t = LayerNorm(W_out * ReLU(P_weighted) + P_{t-1})
    """

    def __init__(self, dim: int = 128, se_reduction: int = 4) -> None:
        super().__init__()
        self.dim = dim
        se_mid = max(1, dim // se_reduction)

        # 2-layer MLP fusion network
        self.fusion_mlp = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, 2)
        )

        # Squeeze-and-Excitation (SE) channel recalibration
        self.se_w1 = nn.Linear(dim, se_mid)
        self.se_w2 = nn.Linear(se_mid, dim)

        # Output projection and normalization
        self.w_out = nn.Linear(dim, dim)
        self.layer_norm = nn.LayerNorm(dim)

    def forward(
        self,
        P_cross: torch.Tensor,
        P_diffuse: torch.Tensor,
        P_prev: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass for Adaptive Fusion & Class Recalibration.
        
        Args:
            P_cross: Cross-attention refined prototype of shape [B, N+1, D].
            P_diffuse: Diffused prototype of shape [B, N+1, D].
            P_prev: Previous stage prototype P_{t-1} of shape [B, N+1, D].
            
        Returns:
            P_t: Purified prototype tensor of shape [B, N+1, D].
        """
        # 1. 2-layer MLP fusion weights
        # [B, N+1, 2*D] -> [B, N+1, 2]
        cat_features = torch.cat([P_cross, P_diffuse], dim=-1)  # [B, N+1, 2*D]
        w = torch.softmax(self.fusion_mlp(cat_features), dim=-1)  # [B, N+1, 2]

        # Convex combination
        P_combined = w[..., 0:1] * P_cross + w[..., 1:2] * P_diffuse  # [B, N+1, D]

        # 2. Squeeze-and-Excitation (SE) recalibration
        # Average pooling across categories [B, N+1, D] -> [B, D]
        pool = P_combined.mean(dim=1)  # [B, D]
        a = torch.sigmoid(self.se_w2(F.relu(self.se_w1(pool)))).unsqueeze(1)  # [B, 1, D]
        P_attended = P_combined * a  # [B, N+1, D]

        # 3. Class-specific attenuation vector w_cls = [0.8, 1.0, ..., 1.0]
        num_classes = P_attended.shape[1]
        w_cls = torch.ones(num_classes, device=P_attended.device, dtype=P_attended.dtype)  # [N+1]
        w_cls[0] = 0.8  # Background attenuation
        w_cls = w_cls.view(1, -1, 1)  # [1, N+1, 1]
        P_weighted = P_attended * w_cls  # [B, N+1, D]

        # 4. Residual connection and LayerNorm
        P_out = self.layer_norm(self.w_out(F.relu(P_weighted)) + P_prev)  # [B, N+1, D]
        return P_out


class EPPMStage(nn.Module):
    """Single stage of Entropy-Aware Prototype Purification Module (EPPM).
    
    Refines incoming prototype P_{t-1} to P_t and computes intermediate logits L_t.
    """

    def __init__(
        self,
        dim: int = 128,
        subspace_dim: int = 72,
        init_theta: float = 0.5,
        tau: float = 0.5,
        alpha: float = 0.5,
        se_reduction: int = 4
    ) -> None:
        super().__init__()
        self.dim = dim
        self.subspace_dim = subspace_dim

        # Sub-modules
        self.gating = InformationTheoreticGating(init_theta=init_theta)
        self.cross_attention = CrossAttentionRefinement(dim=dim, subspace_dim=subspace_dim)
        self.diffusion = PrototypeDiffusion(tau=tau, alpha=alpha)
        self.fusion = AdaptiveFusionRecalibration(dim=dim, se_reduction=se_reduction)

    def forward(
        self,
        P_prev: torch.Tensor,
        F_s: torch.Tensor,
        F_q: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for a single EPPM stage.
        
        Args:
            P_prev: Previous stage prototype P_{t-1} of shape [B, N+1, D].
            F_s: Support point features of shape [B, N_s, D].
            F_q: Query point features of shape [B, N_q, D].
            
        Returns:
            Tuple of:
                - P_t: Purified stage prototype of shape [B, N+1, D]
                - L_t: Intermediate logits of shape [B, N_q, N+1]
                - entropy: Shannon entropy tensor of shape [B, N+1, D]
        """
        num_classes = P_prev.shape[1]

        # 1. Shannon entropy gating
        P_gated, entropy = self.gating(P_prev)  # [B, N+1, D]

        # 2. Cross-attention refinement in subspace d = 72
        P_cross, _ = self.cross_attention(P_gated, F_s, F_q)  # [B, N+1, D]

        # 3. Prototype diffusion with common/unique decomposition
        P_diffuse = self.diffusion(F_s, F_q, num_classes=num_classes)  # [B, N+1, D]

        # 4. Adaptive fusion & SE recalibration & residual link
        P_t = self.fusion(P_cross, P_diffuse, P_prev)  # [B, N+1, D]

        # 5. Intermediate stage logits matching: L_t = F_q * (P_t)^T
        # [B, N_q, D] x [B, D, N+1] -> [B, N_q, N+1]
        L_t = torch.matmul(F_q, P_t.transpose(1, 2))  # [B, N_q, N+1]

        return P_t, L_t, entropy


class EPPMCascade(nn.Module):
    """Cascaded EPPM modules with depth T = 4.
    
    Sequentially refines prototype P_0 -> P_1 -> P_2 -> P_3 -> P_4
    and collects intermediate logits [L_1, L_2, L_3, L_4] for dynamic routing.
    
    Hardcoded invariants:
        T = 4 stages
        D = 128 feature dimension
        d = 72 subspace dimension
        theta = 0.5, tau = 0.5, alpha = 0.5
    """

    def __init__(
        self,
        num_stages: int = 4,
        dim: int = 128,
        subspace_dim: int = 72,
        init_theta: float = 0.5,
        tau: float = 0.5,
        alpha: float = 0.5,
        se_reduction: int = 4
    ) -> None:
        super().__init__()
        self.num_stages = num_stages
        self.dim = dim
        self.subspace_dim = subspace_dim

        self.stages = nn.ModuleList([
            EPPMStage(
                dim=dim,
                subspace_dim=subspace_dim,
                init_theta=init_theta,
                tau=tau,
                alpha=alpha,
                se_reduction=se_reduction
            )
            for _ in range(num_stages)
        ])

    def forward(
        self,
        P_0: torch.Tensor,
        F_s: torch.Tensor,
        F_q: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """Forward pass through the T = 4 EPPM cascade.
        
        Args:
            P_0: Initial fused prototype of shape [B, N+1, D].
            F_s: Support point features of shape [B, N_s, D].
            F_q: Query point features of shape [B, N_q, D].
            
        Returns:
            Tuple of:
                - P_final: Final purified prototype P_4 of shape [B, N+1, D]
                - stage_prototypes: List of prototypes [P_1, P_2, P_3, P_4]
                - stage_logits: List of intermediate logits [L_1, L_2, L_3, L_4],
                                each of shape [B, N_q, N+1]
        """
        stage_prototypes: List[torch.Tensor] = []
        stage_logits: List[torch.Tensor] = []

        P_curr = P_0
        for stage in self.stages:
            P_curr, L_curr, _ = stage(P_curr, F_s, F_q)
            stage_prototypes.append(P_curr)
            stage_logits.append(L_curr)

        return P_curr, stage_prototypes, stage_logits
