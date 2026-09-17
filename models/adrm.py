"""Attention-Based Dynamic Routing Module (ADRM).

Implementation for CascadeProto (ECCV 2026).
Spec references:
- docs/spec/01_ARCHITECTURE_SPEC.md (Section 4)
- docs/spec/02_TENSOR_MATH_SPEC.md (Section 5)
- AGENTS.md (Invariants & Mathematical bounds)
"""

from typing import List, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionDynamicRouting(nn.Module):
    """Attention-Based Dynamic Routing Module (ADRM).
    
    Rather than evaluating predictions exclusively from final stage P_4,
    ADRM treats every cascade stage t in {1, ..., T} as a specialized predictor.
    
    Mathematical Formulation:
        1. Query Global Average Pooling:
           v_q = (1 / N_q) * sum_{j=1}^{N_q} F_{q, j}  in R^D
        2. Routing Gate Vector:
           w_gate = softmax(W_g * v_q) in (0, 1)^T,  sum_{t=1}^T w_gate^(t) = 1.0
        3. Final Aggregated Logits:
           L_final = sum_{t=1}^T w_gate^(t) * L_t  in R^{N_q x (N+1)}
    
    Parameters:
        dim: Feature dimension D = 128.
        num_stages: Cascade depth T = 4.
    """

    def __init__(self, dim: int = 128, num_stages: int = 4) -> None:
        super().__init__()
        self.dim = dim
        self.num_stages = num_stages

        # Learnable projection matrix W_g in R^{T x D}
        # In PyTorch: Linear layer with input dim D, output dim T
        self.W_g = nn.Linear(dim, num_stages, bias=False)

    def forward(
        self,
        F_q: torch.Tensor,
        stage_logits: Union[List[torch.Tensor], torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for dynamic routing and logit aggregation.
        
        Args:
            F_q: Query point features of shape [B, N_q, D].
            stage_logits: Either a list of T tensors each of shape [B, N_q, N+1],
                          or a single stacked tensor of shape [B, T, N_q, N+1].
                          
        Returns:
            Tuple of:
                - L_final: Aggregated logits of shape [B, N_q, N+1]
                - w_gate: Routing gating weights of shape [B, T]
        """
        # 1. Query scene summary via Global Average Pooling
        # [B, N_q, D] -> [B, D]
        v_q = F_q.mean(dim=1)  # [B, D]

        # 2. Routing gate vector on probability simplex
        # [B, D] -> [B, T]
        gate_logits = self.W_g(v_q)  # [B, T]
        w_gate = torch.softmax(gate_logits, dim=-1)  # [B, T]

        # 3. Dynamic convex combination of stage logits
        if isinstance(stage_logits, list):
            # [B, T, N_q, N+1]
            stacked_logits = torch.stack(stage_logits, dim=1)
        else:
            stacked_logits = stage_logits

        # Expand gate weights to [B, T, 1, 1] for broadcasting
        # [B, T] -> [B, T, 1, 1]
        w_gate_expanded = w_gate.unsqueeze(-1).unsqueeze(-1)

        # Convex sum over stage dimension: [B, T, N_q, N+1] -> [B, N_q, N+1]
        L_final = (w_gate_expanded * stacked_logits).sum(dim=1)  # [B, N_q, N+1]

        return L_final, w_gate
