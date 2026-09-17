"""CascadeProto: Few-shot 3D Point Cloud Semantic Segmentation Network.

End-to-end network assembly for CascadeProto (ECCV 2026).
Components:
1. Shared VIP-Seg backbone (feature dim D = 128)
2. Learnable Modality Adapter (LMA) & GMMN Generator
3. Initial prototype fusion (P_0 = P_point + P_modal)
4. T = 4 Cascaded Entropy-Aware Prototype Purification Modules (EPPM)
5. Attention-Based Dynamic Routing Module (ADRM)
6. Decoupled GMMN distribution alignment loss & weighted segmentation loss

Spec references:
- docs/spec/01_ARCHITECTURE_SPEC.md
- docs/spec/02_TENSOR_MATH_SPEC.md
- docs/spec/03_MULTIMODAL_SPEC.md
- AGENTS.md
"""

from typing import Dict, List, NamedTuple, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vipseg_backbone import VIPSegBackbone, extract_point_prototypes
from models.lma import LearnableModalityAdapter, fuse_initial_prototypes
from models.eppm import EPPMCascade
from models.adrm import AttentionDynamicRouting
from loss.segmentation_loss import CascadeProtoLoss


class CascadeProtoForwardOutput(NamedTuple):
    """Named tuple output for CascadeProto forward pass.
    
    Allows direct tuple unpacking:
        logits, p_modal, p_point = model(...)
    As well as attribute access:
        out.logits, out.w_gate, out.stage_logits, out.stage_prototypes
    """
    logits: torch.Tensor             # [B, N_q, N+1]
    p_modal: torch.Tensor            # [B, N+1, D]
    p_point: torch.Tensor            # [B, N+1, D]
    w_gate: torch.Tensor             # [B, T]
    stage_logits: List[torch.Tensor] # T tensors each [B, N_q, N+1]
    stage_prototypes: List[torch.Tensor] # T tensors each [B, N+1, D]


class CascadeProto(nn.Module):
    """CascadeProto End-to-End Few-Shot 3D Point Cloud Segmentation Architecture.
    
    Invariants:
        D = 128 (feature dimension)
        d = 72 (cross-attention subspace dimension)
        T = 4 (cascade depth)
        theta = 0.5 (initial learnable Shannon gating threshold)
        tau = 0.5, alpha = 0.5 (diffusion parameters)
        lambda = 1.0 (loss balancing factor)
        w_cls = [0.8, 1.0, ..., 1.0] (class weighting)
    """

    def __init__(
        self,
        input_points: int = 2048,
        d_feature: int = 128,
        d_subspace: int = 72,
        num_stages: int = 4,
        text_dim: int = 512,
        hidden_dim: int = 256,
        init_theta: float = 0.5,
        tau: float = 0.5,
        alpha: float = 0.5,
        lambda_gmmn: float = 1.0,
        w_bg: float = 0.8,
        w_fg: float = 1.0
    ) -> None:
        super().__init__()
        self.d_feature = d_feature
        self.num_stages = num_stages

        # 1. Point Cloud Backbone (VIP-Seg encoder, D = 128)
        self.backbone = VIPSegBackbone(input_points=input_points, out_dim=d_feature)

        # 2. Learnable Modality Adapter & GMMN Generator
        self.lma = LearnableModalityAdapter(
            clip_dim=text_dim,
            feat_dim=d_feature
        )

        # 3. Cascaded EPPM Modules (T = 4)
        self.cascade = EPPMCascade(
            num_stages=num_stages,
            dim=d_feature,
            subspace_dim=d_subspace,
            init_theta=init_theta,
            tau=tau,
            alpha=alpha
        )

        # 4. Attention-Based Dynamic Routing Module (ADRM)
        self.adrm = AttentionDynamicRouting(
            dim=d_feature,
            num_stages=num_stages
        )

        # 5. Joint Loss Criterion
        self.criterion = CascadeProtoLoss(
            lambda_gmmn=lambda_gmmn,
            w_bg=w_bg,
            w_fg=w_fg
        )

    def _normalize_support_input(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Normalizes various support input shapes to canonical [B, N_s, C] and [B, N_s].
        
        Supports:
            - [B, N_way * K_shot, N_pts, C] -> [B, N_s, C]
            - [N_way, K_shot, C, N_pts] -> [1, N_s, C]
            - [B, N_s, C] -> [B, N_s, C]
        """
        # Case 1: [N_way, K_shot, C, N_pts] (VIP-Seg dataloader style)
        if support_x.dim() == 4 and support_x.shape[2] in (3, 6, 9):
            n_way, k_shot, C, N_pts = support_x.shape
            # Transpose to [N_way, K_shot, N_pts, C]
            x_perm = support_x.permute(0, 1, 3, 2).contiguous()
            x_flat = x_perm.view(1, n_way * k_shot * N_pts, C)
            y_flat = support_y.view(1, n_way * k_shot * N_pts)
            return x_flat, y_flat, n_way

        # Case 2: [B, N_way, K_shot, N_pts, C]
        if support_x.dim() == 5:
            B, n_way, k_shot, N_pts, C = support_x.shape
            x_flat = support_x.view(B, n_way * k_shot * N_pts, C)
            y_flat = support_y.view(B, n_way * k_shot * N_pts)
            return x_flat, y_flat, n_way

        # Case 3: [B, N_shots, N_pts, C] (spec style [B, N, 2048, 3])
        if support_x.dim() == 4:
            B, n_shots, N_pts, C = support_x.shape
            if C not in (3, 6, 9) and N_pts in (3, 6, 9):
                # Permute [B, n_shots, C, N_pts] -> [B, n_shots, N_pts, C]
                support_x = support_x.permute(0, 1, 3, 2).contiguous()
                C = support_x.shape[-1]
                N_pts = support_x.shape[2]
            x_flat = support_x.view(B, n_shots * N_pts, C)
            y_flat = support_y.view(B, n_shots * N_pts)
            return x_flat, y_flat, n_shots

        # Case 4: Canonical [B, N_s, C]
        if support_x.dim() == 3:
            B, N_s, C = support_x.shape
            if C in (3, 6, 9):
                pass
            elif support_x.shape[1] in (3, 6, 9):
                support_x = support_x.permute(0, 2, 1).contiguous()
            # Estimate n_way from max mask value
            n_way = int(support_y.max().item())
            return support_x, support_y, max(n_way, 1)

        raise ValueError(f"Unsupported support_x shape: {support_x.shape}")

    def _normalize_query_input(self, query_x: torch.Tensor) -> torch.Tensor:
        """Normalizes query coordinates/features to [B, N_q, C]."""
        if query_x.dim() == 3:
            # [B, N_q, C] or [B, C, N_q]
            if query_x.shape[1] in (3, 6, 9) and query_x.shape[2] > 9:
                return query_x.permute(0, 2, 1).contiguous()
            return query_x
        if query_x.dim() == 2:
            # [N_q, C] -> [1, N_q, C]
            return query_x.unsqueeze(0)
        raise ValueError(f"Unsupported query_x shape: {query_x.shape}")

    def forward(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
        text_embeddings: Optional[torch.Tensor] = None,
        query_y: Optional[torch.Tensor] = None,
        n_way: Optional[int] = None
    ) -> Union[CascadeProtoForwardOutput, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass of CascadeProto.
        
        Args:
            support_x: Support point clouds.
            support_y: Support segmentation masks.
            query_x: Query point clouds.
            text_embeddings: Normalized CLIP text embeddings of shape [B, N+1, 512]
                             or [1, N+1, 512]. If None, a learnable fallback is used.
            query_y: Optional ground-truth query labels of shape [B, N_q].
            n_way: Optional explicit number of foreground categories.
            
        Returns:
            CascadeProtoForwardOutput: NamedTuple containing:
                - logits: Final aggregated query logits [B, N_q, N+1]
                - p_modal: Synthesized modal prototypes [B, N+1, D]
                - p_point: Support point cloud prototypes [B, N+1, D]
                - w_gate: Dynamic routing weights [B, T]
                - stage_logits: List of intermediate logits [L_1, ..., L_T]
                - stage_prototypes: List of intermediate prototypes [P_1, ..., P_T]
            (If query_y is provided and called in training mode, unpacking
             (logits, loss) can also be supported via helper).
        """
        # 1. Normalize input dimensions
        s_x, s_y, inferred_way = self._normalize_support_input(support_x, support_y)
        q_x = self._normalize_query_input(query_x)
        B, N_s, _ = s_x.shape
        _, N_q, _ = q_x.shape
        way = n_way if n_way is not None else inferred_way
        num_classes = way + 1

        # 2. Extract 3D backbone features (D = 128)
        # s_x: [B, N_s, C] -> F_s: [B, N_s, 128]
        # q_x: [B, N_q, C] -> F_q: [B, N_q, 128]
        F_s = self.backbone(s_x)  # [B, N_s, 128]
        F_q = self.backbone(q_x)  # [B, N_q, 128]

        # 3. Support Point Prototype Extraction via Masked Average Pooling
        P_point = extract_point_prototypes(F_s, s_y, n_way=way)  # [B, N+1, 128]

        # 4. Modality Prototype Synthesis via LMA (CLIP projection + GMMN generator)
        if text_embeddings is None:
            # Fallback if text embeddings not provided: initialize from zero/noise
            text_embeddings = torch.zeros(
                (B, num_classes, 512),
                dtype=F_s.dtype,
                device=F_s.device
            )
        elif text_embeddings.dim() == 2:
            text_embeddings = text_embeddings.unsqueeze(0)
        if text_embeddings.shape[0] != B:
            text_embeddings = text_embeddings.expand(B, -1, -1)

        P_modal = self.lma(text_embeddings)  # [B, N+1, 128]

        # 5. Enriched Initial Prototype: P_0 = P_point + P_modal
        P_0 = fuse_initial_prototypes(P_point, P_modal)  # [B, N+1, 128]

        # 6. Cascaded Prototype Purification (T = 4 EPPM stages)
        P_final, stage_protos, stage_logits = self.cascade(P_0, F_s, F_q)

        # 7. Attention-Based Dynamic Routing (ADRM)
        L_final, w_gate = self.adrm(F_q, stage_logits)  # [B, N_q, N+1], [B, 4]

        # Return standard output
        output = CascadeProtoForwardOutput(
            logits=L_final,
            p_modal=P_modal,
            p_point=P_point,
            w_gate=w_gate,
            stage_logits=stage_logits,
            stage_prototypes=stage_protos
        )

        return output
