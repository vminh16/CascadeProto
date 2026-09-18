"""
VIP-Seg Backbone and Point Prototype Extraction for CascadeProto.
Implements:
1. Shared VIP-Seg Encoder mapping point clouds to dense geometric features D = 128.
2. Point-based Prototype Extraction (Module B from 01_ARCHITECTURE_SPEC.md).

Reference:
- 01_ARCHITECTURE_SPEC.md (Sections 2 & 4)
- 02_TENSOR_MATH_SPEC.md (Sections 1 & 2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.encoder import Encoder


class VIPSegBackbone(nn.Module):
    """
    Shared VIP-Seg Point Cloud Backbone Encoder.
    Produces feature representations of dimension D = 128 from raw 3D coordinates.
    """
    def __init__(self, input_points: int = 2048, out_dim: int = 128):
        super().__init__()
        self.input_points = input_points
        self.out_dim = out_dim
        
        # Encoder consists of 3 stages of DyPowerConv + Mamba/Transformer blocks
        self.encoder = Encoder(
            input_points=input_points,
            num_stages=3,
            embed_dim=60,
            k_neighbors=16,
            de_neighbors=10,
            alpha=1000,
            beta=30,
            num_experts=3,
        )
        
        # Projection head to D = 128
        self.Dim = 900
        self.bn = nn.Sequential(
            nn.BatchNorm1d(self.Dim),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Conv1d(self.Dim, 196, 1),
            nn.BatchNorm1d(196),
            nn.ReLU(inplace=True),
            nn.Conv1d(196, out_dim, 1),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract dense geometric features from point cloud.
        Args:
            x: Input point cloud tensor of shape [B, N, C] or [B, C, N].
               If C == 3 (xyz), it is padded to 9 channels (xyz + rgb + normalized XYZ).
        Returns:
            features: [B, N, D] where D = 128.
        """
        # Ensure x is [B, N, C]
        if x.dim() == 3 and x.shape[1] < x.shape[2] and x.shape[1] in (3, 6, 9):
            x = x.permute(0, 2, 1)

        B, N, C = x.shape
        if C < 9:
            # Pad with normalized coordinates and neutral colors if only xyz given
            xyz = x[:, :, :3]
            # Neutral RGB (zeros)
            rgb = torch.zeros_like(xyz)
            # Normalized coordinates [-1, 1]
            xyz_min = xyz.min(dim=1, keepdim=True)[0]
            xyz_max = xyz.max(dim=1, keepdim=True)[0]
            xyz_norm = 2.0 * (xyz - xyz_min) / torch.clamp(xyz_max - xyz_min, min=1e-6) - 1.0
            x = torch.cat([xyz, rgb, xyz_norm], dim=-1)

        # Encoder forward: expects [B, N, 9] -> returns [B, Dim, N]
        features = self.encoder(x)  # [B, 900, N]
        features = features / torch.clamp(features.norm(dim=1, keepdim=True), min=1e-8)
        
        # BN + FC projection: [B, 900, N] -> [B, 128, N]
        features = self.bn(features)
        features = self.fc(features)  # [B, 128, N]
        
        # Permute to [B, N, D]
        features = features.permute(0, 2, 1)  # [B, N, 128]
        return features


def extract_point_prototypes(
    support_features: torch.Tensor,
    support_masks: torch.Tensor,
    n_way: int,
) -> torch.Tensor:
    """
    Extract foreground and background point prototypes via Masked Average Pooling.
    Args:
        support_features: [B, N_s, D] or [N_way, K_shot, N_s, D] or [B, N_way * K_shot, N_s, D]
        support_masks: [B, N_s] or [N_way, K_shot, N_s] binary masks (1 for target class, 0 for bg)
        n_way: Number of foreground classes N
    Returns:
        P_point: [B, N + 1, D] where index 0 is background, indices 1..N are foreground classes.
    """
    # Normalize input shape to [B, num_support_points, D]
    if support_features.dim() == 4:
        # [N_way, K_shot, N_s, D]
        nway, kshot, ns, D = support_features.shape
        support_features = support_features.view(1, nway * kshot * ns, D)
        support_masks = support_masks.view(1, nway * kshot * ns)
    elif support_features.dim() == 3 and support_masks.dim() == 2:
        pass
    else:
        raise ValueError(f"Unexpected shape for support_features: {support_features.shape}")

    B, total_pts, D = support_features.shape
    prototypes_batch = []

    for b in range(B):
        feat = support_features[b]  # [total_pts, D]
        mask = support_masks[b]     # [total_pts]

        proto_list = []
        
        # 1. Background Prototype (class 0: mask == 0)
        bg_mask = (mask == 0)
        if bg_mask.sum() > 0:
            p_bg = feat[bg_mask].mean(dim=0, keepdim=True)  # [1, D]
        else:
            p_bg = torch.zeros(1, D, device=feat.device)
        proto_list.append(p_bg)

        # 2. Foreground Prototypes (classes 1..N)
        for k in range(1, n_way + 1):
            fg_mask = (mask == k)
            if fg_mask.sum() > 0:
                p_fg = feat[fg_mask].mean(dim=0, keepdim=True)  # [1, D]
            else:
                p_fg = torch.zeros(1, D, device=feat.device)
            proto_list.append(p_fg)

        p_point = torch.cat(proto_list, dim=0)  # [N+1, D]
        prototypes_batch.append(p_point)

    P_point = torch.stack(prototypes_batch, dim=0)  # [B, N+1, D]
    return P_point
