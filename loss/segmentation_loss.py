"""Segmentation and Joint Objective Loss Functions for CascadeProto.

Spec references:
- docs/spec/01_ARCHITECTURE_SPEC.md (Section 3, class weighting vector w_cls)
- docs/spec/02_TENSOR_MATH_SPEC.md (Section 6, total training objective)
- AGENTS.md (Invariants: lambda = 1.0, w_bg = 0.8, w_fg = 1.0)
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from loss.gmmn_loss import DecoupledGMMNLoss


class SegmentationLoss(nn.Module):
    """Class-weighted Cross Entropy Loss for Few-Shot Point Cloud Segmentation.
    
    Formula:
        L_seg = - (1 / N_q) * sum_{i=1}^{N_q} log( exp(L[i, Y_i]) / sum_c exp(L[i, c]) )
        
    Class attenuation:
        w_cls = [w_bg, w_fg, ..., w_fg] in R^{N+1}
        where w_bg = 0.8 (index 0) and w_fg = 1.0 (indices 1..N).
    """

    def __init__(self, w_bg: float = 0.8, w_fg: float = 1.0) -> None:
        super().__init__()
        self.w_bg = w_bg
        self.w_fg = w_fg

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute weighted cross-entropy loss.
        
        Args:
            logits: Query predicted logits of shape [B, N_q, num_classes].
            targets: Ground-truth class labels of shape [B, N_q].
            
        Returns:
            loss: Scalar cross-entropy loss.
        """
        # [B, N_q, C] -> [B, C, N_q]
        logits_permuted = logits.permute(0, 2, 1)  # [B, C, N_q]
        num_classes = logits.shape[-1]

        # Construct class weighting vector w_cls = [0.8, 1.0, ..., 1.0]
        weights = torch.full(
            (num_classes,),
            fill_value=self.w_fg,
            device=logits.device,
            dtype=logits.dtype
        )
        weights[0] = self.w_bg

        loss = F.cross_entropy(logits_permuted, targets, weight=weights)
        return loss


class CascadeProtoLoss(nn.Module):
    """Joint CascadeProto Training Objective.
    
    Formula:
        L_total = L_seg + lambda * L_GMMN
        where lambda = 1.0.
    """

    def __init__(
        self,
        lambda_gmmn: float = 1.0,
        w_bg: float = 0.8,
        w_fg: float = 1.0,
        gmmn_bg_weight: float = 0.1,
        gmmn_fg_weight: float = 1.0
    ) -> None:
        super().__init__()
        self.lambda_gmmn = lambda_gmmn
        self.seg_loss = SegmentationLoss(w_bg=w_bg, w_fg=w_fg)
        self.gmmn_loss = DecoupledGMMNLoss(
            bg_weight=gmmn_bg_weight,
            fg_weight=gmmn_fg_weight
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        p_modal: torch.Tensor,
        p_point: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for joint loss calculation.
        
        Args:
            logits: Query prediction logits of shape [B, N_q, num_classes].
            targets: Query ground-truth labels of shape [B, N_q].
            p_modal: Synthesized modal prototypes of shape [B, num_classes, D].
            p_point: Extracted point cloud prototypes of shape [B, num_classes, D].
            
        Returns:
            Tuple of:
                - total_loss: L_total = L_seg + lambda * L_GMMN
                - seg_loss: Cross-entropy segmentation loss
                - gmmn_loss: Decoupled GMMN distribution matching loss
        """
        l_seg = self.seg_loss(logits, targets)
        l_gmmn = self.gmmn_loss(p_modal, p_point)
        total_loss = l_seg + self.lambda_gmmn * l_gmmn
        return total_loss, l_seg, l_gmmn
