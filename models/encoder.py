"""
Encoder-Decoder Network with DyPowerConv-Mamba Blocks for Point Cloud Feature Extraction

Paper: "Reasoning Beyond Points: A Visual Introspective Approach for Few-Shot 3D Segmentation"
NeurIPS 2025

Main Components:
1. DyPowerConv (Dynamic Power Convolution) - Section 3.3
2. Mamba Blocks for long-range dependencies
3. Encoder-Decoder architecture - Section 3.2
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from pointnet2_ops_lib.pointnet2_ops import pointnet2_utils
except (ImportError, ModuleNotFoundError):
    pointnet2_utils = None

from models.model_utils import *

# Mamba imports
from functools import partial
try:
    from mamba_ssm.modules.mamba_simple import Mamba
    from .mamba_block import MambaBlock as mamblock
except (ImportError, ModuleNotFoundError):
    Mamba = None
    mamblock = None
from .mamba_block import TransBlock

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except (ImportError, ModuleNotFoundError):
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

torch.pi = math.pi

# ============================================================
# Mamba Block Creation
# ============================================================

def create_mamblock(
        d_model,
        ssm_cfg=None,
        norm_epsilon=1e-5,
        rms_norm=False,
        residual_in_fp32=False,
        fused_add_norm=False,
        layer_idx=None,
        drop_path=0.,
        device=None,
        dtype=None,
):
    if Mamba is None or mamblock is None:
        # Pure-PyTorch fallback transformer block when mamba_ssm is unavailable
        return TransBlock(embed_dim=d_model, num_heads=4)

    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}

    mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = mamblock(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
        drop_path=drop_path,
    )
    block.layer_idx = layer_idx
    return block

# ============================================================
# Point Cloud Sampling and Grouping
# ============================================================

# FPS + k-NN
class FPS_kNN(nn.Module):
    def __init__(self, group_num, k_neighbors):
        super().__init__()
        self.group_num = group_num
        self.k_neighbors = k_neighbors

    def forward(self, xyz, x, rgb):
        
        # FPS
        if pointnet2_utils is not None and xyz.is_cuda:
            fps_idx = pointnet2_utils.furthest_point_sample(xyz.contiguous(), self.group_num).long()
        else:
            fps_idx = furthest_point_sample_py(xyz, self.group_num)
        lc_xyz = index_points(xyz, fps_idx)
        lc_x = index_points(x, fps_idx)
        
        lc_rgb = index_points(rgb, fps_idx)
        
        # kNN
        knn_idx = knn_point(self.k_neighbors, xyz, lc_xyz)
        knn_xyz = index_points(xyz, knn_idx)
        knn_x = index_points(x, knn_idx)
        
        knn_rgb = index_points(rgb, knn_idx)

        return lc_xyz, lc_x, lc_rgb, knn_xyz, knn_x, knn_rgb

class Pooling(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, knn_x_w):
        lc_x = knn_x_w.max(-1)[0]
        lc_x = self.relu(lc_x)
        
        return lc_x

# ============================================================
# Position Encoding
# ============================================================

class PosE_Initial(nn.Module):
    def __init__(self, in_dim, out_dim, alpha, beta):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha, self.beta = alpha, beta
        
    def forward(self, x, rgbx):
        B, _, N = x.shape
        feat_dim = self.out_dim // (self.in_dim * 2)
        
        feat_range = torch.arange(feat_dim, device=x.device).float() / feat_dim   
        dim_embed = torch.pow(self.alpha, feat_range)
        x_div = torch.div(self.beta * x.unsqueeze(-1), dim_embed)
        rgbx_div = torch.div(self.beta * rgbx.unsqueeze(-1), dim_embed)
        
        sin_x, cos_x = torch.sin(x_div), torch.cos(x_div)
        sin_rgbx, cos_rgbx = torch.sin(rgbx_div), torch.cos(rgbx_div)
        
        position_embed = torch.stack([sin_x, cos_x], dim=4).flatten(3)
        position_embed = position_embed.permute(0, 1, 3, 2).reshape(B, self.out_dim, N)
        
        rgbx_embed = torch.stack([sin_rgbx, cos_rgbx], dim=4).flatten(3)
        rgbx_embed = rgbx_embed.permute(0, 1, 3, 2).reshape(B, self.out_dim, N)
                
        position_embed = position_embed * 0.8 + rgbx_embed * 0.2
        
        return position_embed, rgbx_embed
    
# ============================================================
# DyPowerConv Module
# ============================================================

class DyPowerConv(nn.Module):
    """
    Dynamic Power Convolution
    
    Combines Low-order Convolution (LoConv) and Dynamic High-order Convolution (DyHiConv)
    to adaptively model complex local geometric features through learnable power functions.
    
    Reference: Section 3.3 - Dynamic Power Convolution
    Formula: g_i = g^L_i + g^{DH}_i (Eq. 1)
    """
    def __init__(self, out_dim, alpha, beta, vv, ww, num_experts):
        super().__init__()
        alpha, beta = 1, 1

        # Low-order Convolution (Eq. 2-4)
        self.LoConv = LowOrderConvolution(3, out_dim, alpha, beta, vv)

        # Dynamic High-order Convolution (Eq. 5-8)      
        self.DyHiConv = DynamicHighOrderConvolution(3, out_dim, alpha, beta, ww, num_experts)

    def forward(self, lc_xyz, lc_x, lc_rgb, knn_xyz, knn_x, knn_rgb):
        """
        Args:
            lc_xyz: Local center coordinates, shape [B, G, 3]
            lc_x: Local center features, shape [B, G, C]
            lc_rgb: Local center RGB, shape [B, G, 3]
            knn_xyz: Neighbor coordinates, shape [B, G, K, 3]
            knn_x: Neighbor features, shape [B, G, K, C]
            knn_rgb: Neighbor RGB, shape [B, G, K, 3]
            
        Returns:
            output: Combined features g_i = g^L_i + g^{DH}_i, shape [B, out_dim, G]
        """
        # ============================================================
        # Feature Normalization
        # ============================================================
        
        # Normalize features
        mean_x = lc_x.unsqueeze(dim=-2)
        std_x = torch.std(knn_x - mean_x)
        knn_x = (knn_x - mean_x) / (std_x + 1e-5)
        
        # Normalize coordinates
        mean_xyz = lc_xyz.unsqueeze(dim=-2)
        std_xyz = torch.std(knn_xyz - mean_xyz)
        knn_xyz = (knn_xyz - mean_xyz) / (std_xyz + 1e-5)

        # ============================================================
        # Feature Expansion
        # ============================================================
        # Concatenate center features with neighbor features

        B, G, K, C = knn_x.shape
        knn_x = torch.cat([knn_x, lc_x.reshape(B, G, 1, -1).repeat(1, 1, K, 1)], dim=-1)
        
        # ============================================================
        # Geometry Extraction
        # ============================================================
        # Permute to [B, C, G, K] format for convolution operations
        knn_xyz = knn_xyz.permute(0, 3, 1, 2)
        knn_x = knn_x.permute(0, 3, 1, 2)
        knn_rgb = knn_rgb.permute(0, 3, 1, 2)
        
        # Low-order Convolution (captures basic geometric structures)
        x_lo = self.LoConv(knn_xyz, knn_x, knn_rgb)
        
        # Dynamic High-order Convolution (captures fine-grained details)
        x_hi = self.DyHiConv(lc_xyz, knn_xyz, knn_x, knn_rgb)

        # Combine (Eq. 1)
        return x_lo + x_hi

# ============================================================
# Low-order Convolution for Basic Geometric Extraction
# ============================================================

class LowOrderConvolution(nn.Module):
    def __init__(self, in_dim, out_dim, alpha, beta, vv):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha, self.beta = alpha, beta
        self.vv = vv

        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=1, bias=True),
            nn.BatchNorm2d(out_dim),
            nn.ReLU()
        )

        self.pooling = Pooling()
        
    def forward(self, knn_xyz, knn_x, knn_rgb):
        """
        Args:
            knn_xyz: Neighbor coordinates, shape [B, 3, G, K]
            knn_x: Neighbor features, shape [B, C, G, K]
            knn_rgb: Neighbor RGB, shape [B, 3, G, K]
            
        Returns:
            knn_x_new: Processed features, shape [B, out_dim, G]
        """

        B, _, G, K = knn_xyz.shape
        feat_dim = self.out_dim // (self.in_dim * 2)

        # Create frequency embeddings
        feat_range = torch.arange(feat_dim, device=knn_xyz.device).float()     
        dim_embed = torch.pow(self.alpha, feat_range / feat_dim)
        
        # Encode coordinates
        knn_xyz_new = knn_xyz.permute(0, 2, 3, 1)[..., None]
        div_xyz = torch.div(self.beta * knn_xyz_new.unsqueeze(-1), dim_embed) #* 0.1
        sin_xyz, cos_xyz = torch.sin(div_xyz), torch.cos(div_xyz)
        xyz_embed = torch.cat([sin_xyz, cos_xyz], dim=4).flatten(3)
        xyz_embed = xyz_embed.permute(0, 3, 1, 2)
        
        # Encode RGB
        knn_rgb_1 = knn_rgb.permute(0, 2, 3, 1)[..., None]
        div_rgb = torch.div(self.beta * knn_rgb_1.unsqueeze(-1), dim_embed) #* 0.1
        sin_rgb, cos_rgb = torch.sin(div_rgb), torch.cos(div_rgb)
        rgb_embed = torch.cat([sin_rgb, cos_rgb], dim=4).flatten(3)
        rgb_embed = rgb_embed.permute(0, 3, 1, 2)

        # Combine features (equal weighting: 1/3 each)
        knn_x_new = knn_x / 3 + xyz_embed / 3 + rgb_embed / 3
        
        # MLP transformation
        knn_x_new = self.mlp(knn_x_new)
        
        # Apply unlearnable parameterless weight projection
        knn_x_new = knn_x_new.permute(0, 2, 3, 1)
        device = knn_x_new.device
        pos = self.vv[:, :self.out_dim].to(device).T @ torch.arange(self.out_dim, device=device).unsqueeze(0).float()
        W_l = torch.cos(pos * 2 * torch.pi).to(device)  

        knn_x_new = knn_x_new @ W_l

        # Normalization
        mean_knn_x_new = knn_x_new.mean()
        std_knn_x_new = torch.std(knn_x_new - mean_knn_x_new)
        knn_x_new = (knn_x_new - mean_knn_x_new) / (std_knn_x_new + 1e-6)
        knn_x_new = knn_x_new.permute(0, 3, 1, 2)

         # Max pooling
        knn_x_new = self.pooling(knn_x_new)

        return knn_x_new

# ============================================================
# Dynamic High-order Convolution for Fine-grained Geometry Extraction
# ============================================================

class DynamicHighOrderConvolution(nn.Module):
    """
    Dynamic High-order Convolution
    
    Captures fine-grained details through multiple high-order convolutions with
    learnable power functions and dynamic weight generation.
    
    Reference: Section 3.3.3 (Eq. 5-8)
    """
    
    def __init__(self, in_dim, out_dim, alpha, beta, ww, num_experts):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha, self.beta = alpha, beta
        self.ww = ww
        self.num_experts = num_experts

        # Expert networks - each learns a different power function
        self.weight_generator = nn.Sequential(
            nn.Conv2d(10, 10 * num_experts, 1, bias=False),
            nn.BatchNorm2d(10 * num_experts),
            nn.ReLU(inplace=True),
            nn.Conv2d(10 * num_experts, out_dim * num_experts, 1, bias=False),
            nn.BatchNorm2d(out_dim * num_experts),
            nn.ReLU(inplace=True)
        )

        # Attention network for expert selection
        self.expert_selector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(10, num_experts, 1, bias=False),
            nn.Softmax(dim=1)
        )

        # Learnable power exponents for each expert (p_t in Eq. 6)
        self.power_p = nn.Parameter(torch.full((num_experts, 1), fill_value=1.0))

        self.HiConv = HighOrderConvolution()
 
    def forward(self, lc_xyz, knn_xyz, knn_x, knn_rgb):
        B, _, N, K = knn_xyz.shape
        feat_dim = self.out_dim // (self.in_dim * 2)
        
        feat_range = torch.arange(feat_dim, device=knn_xyz.device).float()     
        dim_embed = torch.pow(self.alpha, feat_range / feat_dim)
        
        knn_xyz_1 = knn_xyz.permute(0, 2, 3, 1)[..., None]
        div_xyz = torch.div(self.beta * knn_xyz_1.unsqueeze(-1), dim_embed) #* 0.1
        sin_xyz, cos_xyz = torch.sin(div_xyz), torch.cos(div_xyz)
        xyz_embed = torch.cat([sin_xyz, cos_xyz], dim=4).flatten(3)
        xyz_embed = xyz_embed.permute(0, 3, 1, 2)
        
        #print(knn_rgb.shape)
        knn_rgb_1 = knn_rgb.permute(0, 2, 3, 1)[..., None]
        div_rgb = torch.div(self.beta * knn_rgb_1.unsqueeze(-1), dim_embed) #* 0.1
        sin_rgb, cos_rgb = torch.sin(div_rgb), torch.cos(div_rgb)
        rgb_embed = torch.cat([sin_rgb, cos_rgb], dim=4).flatten(3)
        rgb_embed = rgb_embed.permute(0, 3, 1, 2)
        
        knn_x_new = knn_x / 3 + xyz_embed / 3 + rgb_embed / 3

        mean_knn_x = torch.mean(knn_x_new, dim=1, keepdim=True) 
        #std_x = torch.std(knn_x_new - mean_knn_x)
        #knn_x_new = (knn_x_new - mean_knn_x) / (std_x + 1e-5)
        relative_knn_x_new = knn_x_new - mean_knn_x

        #生成权重
        mean_xyz = lc_xyz.unsqueeze(dim=-2) #B, N, K, C
        coord_xi = mean_xyz.permute(0, 3, 1, 2).repeat(1, 1, 1, K)   # (B, 3, npoint, nsample),  centroid point
        abs_coord = knn_xyz  # (B, 3, npoint, nsample+1), absolute coordinates
        delta_x = knn_xyz - coord_xi   # (B, 3, npoint, nsample+1), normalized coordinates  
        abs_xi_xj = torch.norm(delta_x, p = 2, dim = 1).unsqueeze(1)
        h_xi_xj = torch.cat((coord_xi, abs_coord, delta_x, abs_xi_xj), dim = 1) # [B, 10, npoint, k]

        # ============================================================
        # Compute Attention Weights (Eq. 7)
        # ============================================================
        # ϕ_t = exp(W_t * h_j) / Σ exp(W_t * h_j)
        expert_weights = self.weight_generator(h_xi_xj)  # [B, out_dim * num_experts, N, K]
        expert_weights = expert_weights.view(B, self.num_experts, -1, N, K)  # [B, num_experts, out_dim, N, K]

        expert_scores = self.expert_selector(h_xi_xj)  # [B, num_experts, 1, 1]

        outputs = []
        for t in range(self.num_experts):
           # Dynamic weight generation (Eq. 8)
            w_t = expert_weights[:, t] # [B, out_dim, N, K]
            power_p = self.power_p[t]

            # ============================================================
            # Compute High-order Features for Each Expert (Eq. 6)
            # ============================================================
            # g^t_i = A({w_t(p_j) ⊙ (|f_j - f_i| + ε)^{p_t}})

            aggregated_f = self.HiConv(w_t, relative_knn_x_new, power_p)  # B, C, N
            
            # Apply attention weights
            f = aggregated_f * expert_scores[:, t]  # 加权

            outputs.append(f)

        knn_x_new = torch.stack(outputs, dim=-1)

        # Average attention weights over K neighbors
        knn_x_new = knn_x_new.sum(-1)  # [B, out_dim, N]



        #knn_x_new = self.HiConv(h_xi_xj, relative_knn_x_new)  # B, C, N

        knn_x_new = knn_x_new.unsqueeze(-1).permute(0, 2, 3, 1)

        device = knn_x_new.device
        pos = self.ww[:, :self.out_dim].to(device).T @ torch.arange(self.out_dim, device=device).unsqueeze(0).float()
        W_l = torch.cos(pos * 2 * torch.pi).to(device)    

        knn_x_new = knn_x_new @ W_l
        mean_knn_x_new = knn_x_new.mean()
        std_knn_x_new = torch.std(knn_x_new - mean_knn_x_new)
        knn_x_new = (knn_x_new - mean_knn_x_new) / (std_knn_x_new + 1e-6)
        knn_x_new = knn_x_new.permute(0, 3, 1, 2).squeeze(-1)

        return knn_x_new

# ============================================================
# High-order Convolution
# ============================================================

class HighOrderConvolution(nn.Module):
    """
    High-order Convolution with Multiple Experts
    
    Generates convolution kernels dynamically based on local geometric priors.
    Multiple expert networks are combined with attention weights.
    
    Reference: Section 3.3.3
    """

    def __init__(self):
        super().__init__()
        
    def forward(self, geometric_priors, relative_features, power_p):
        """
        Args:
            geometric_priors: Geometric information h_j, shape [B, in_channels, N, K]
                             Contains: [p_i, p_j, p_j - p_i, ||p_j - p_i||]
            relative_features: Relative features, shape [B, C, N, K]
            
        Returns:
            aggregated_features: Shape [B, out_dim, N]
        """
        B, _, N, K = geometric_priors.shape

        # Power function with learnable exponent (Eq. 6)
        # Note: Using smooth power function (|f_j - f_i| + ε)^{p_t}
        knn_exp = (torch.abs(relative_features) + 1e-6)**power_p

        # Element-wise multiplication
        weighted_f = torch.mul(geometric_priors, knn_exp)  # [B, out_dim, N, K]

        # Aggregation with max pooling
        aggregated_f = F.max_pool2d(weighted_f, kernel_size=(1, K)).squeeze(3)  # [B, out_dim, N]

        return aggregated_f

# ============================================================
# Parametric Encoder
# ============================================================

class ParametricEncoder(nn.Module):
    """
    Parametric Encoder with DyPowerConv-Mamba Blocks
    
    Multi-stage hierarchical encoder that progressively downsamples the point cloud
    while extracting increasingly abstract features.
    
    """
    def __init__(self, input_points, num_stages, embed_dim, k_neighbors, alpha, beta, vv, ww, num_experts):
        super().__init__()
        self.input_points = input_points
        self.num_stages = num_stages
        self.embed_dim = embed_dim
        self.alpha, self.beta = alpha, beta

        # Raw-point Embedding
        self.Embedding_layer = nn.Sequential(
            nn.Conv1d(3, self.embed_dim, 1, bias=False),
            nn.BatchNorm1d(self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.embed_dim, self.embed_dim, 1, bias=False),
        )

        self.FPS_kNN_list = nn.ModuleList() # FPS, kNN
        self.DyPowerConv_list = nn.ModuleList() # Local Geometry Aggregation
        self.Mamba_list = nn.ModuleList() # Mamba Block
        
        out_dim = self.embed_dim
        group_num = self.input_points

        # Multi-stage Hierarchy
        for i in range(self.num_stages):
            if i < 2:
                out_dim = out_dim * 2
                group_num = group_num // 2
            else:
                out_dim = out_dim * 2
                group_num = group_num // 2
            self.FPS_kNN_list.append(FPS_kNN(group_num, k_neighbors))
            self.DyPowerConv_list.append(DyPowerConv(out_dim, self.alpha, self.beta, vv, ww, num_experts))

            self.Mamba_list.append(
                create_mamblock(out_dim,
                    ssm_cfg=None,
                    norm_epsilon=1e-5,
                    rms_norm=False,
                    residual_in_fp32=False,
                    fused_add_norm=False,
                    layer_idx=i,
                    drop_path=0.))

    def forward(self, xyz, x, rgb, rgbx):

        # Initial embedding
        x = self.Embedding_layer(x)

        xyz_list = [xyz]  # [B, N, 3]
        x_list = [x]  # [B, C, N]
                
        # Multi-stage Hierarchy
        for i in range(self.num_stages):

            # 1. FPS + kNN grouping
            xyz, lc_x, rgb, knn_xyz, knn_x, knn_rgb = self.FPS_kNN_list[i](xyz, x.permute(0, 2, 1), rgb)

            # 2. DyPowerConv for local geometry aggregation
            x = self.DyPowerConv_list[i](xyz, lc_x, rgb, knn_xyz, knn_x, knn_rgb)

            # 3. Mamba block for long-range dependencies (with residual connection)
            x = self.Mamba_list[i](x.transpose(2, 1))[0].transpose(2, 1) + x
    
            xyz_list.append(xyz)
            x_list.append(x)

        return xyz_list, x_list

# ============================================================
# Non-Parametric Decoder
# ============================================================

class NonParametricDecoder(nn.Module):  
    def __init__(self, num_stages, de_neighbors):
        super().__init__()
        self.num_stages = num_stages
        self.de_neighbors = de_neighbors

    def propagate(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, N, 3]
            xyz2: sampled input points position data, [B, S, 3]
            points1: input points data, [B, D', N]
            points2: input points data, [B, D'', S]
        Return:
            new_points: upsampled points data, [B, D''', N]
        """
        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            #dists = dists ** 2
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :self.de_neighbors], idx[:, :, :self.de_neighbors]  # [B, N, 3]
            
            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
                        
            weight = weight.view(B, N, self.de_neighbors, 1)
            points2_indexed = index_points(points2, idx)

            interpolated_points = torch.sum(points2_indexed * weight, dim=2)
            
            mean_interpolated_points = interpolated_points.mean()
            std_interpolated_points = torch.std(interpolated_points - mean_interpolated_points)
            interpolated_points = (interpolated_points - mean_interpolated_points) / (std_interpolated_points + 1e-5)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        return new_points

    def forward(self, xyz_list, x_list):
        xyz_list.reverse()
        x_list.reverse()

        x = x_list[0]
        for i in range(self.num_stages):
            # Propagate point features to neighbors
            x = self.propagate(xyz_list[i+1], xyz_list[i], x_list[i+1], x)
        return x

# ============================================================
# Parametric VIP-Seg Backbone
# ============================================================

class Encoder(nn.Module):
    """
    Complete Encoder-Decoder Network with DyPowerConv-Mamba Blocks
    This is the main feature extraction backbone for VIP-Seg.
    """
    def __init__(self, input_points=2048, num_stages=5, embed_dim=144, k_neighbors=128, de_neighbors=6, alpha=1000, beta=50, num_experts=3):
        super().__init__()
        
        ## Ordered vv is slightly worse than its unordered version        
        #vv = torch.sort(torch.abs(torch.randn(1, 1000)))[0]
        vv = torch.randn(1, 5000)
        ww = torch.randn(1, 5000)

        # Parametric encoder with DyPowerConv-Mamba blocks
        self.EncNP = ParametricEncoder(input_points, num_stages, embed_dim, k_neighbors, alpha, beta, vv, ww, num_experts)

        # Non-parametric decoder
        self.DecNP = NonParametricDecoder(num_stages, de_neighbors)
        
        print(f"[Encoder] Initialized with:")
        print(f"  - Input points: {input_points}")
        print(f"  - Stages: {num_stages}")
        print(f"  - HiConv number: {num_experts}")
        print(f"  - Base dimension: {embed_dim}")
        print(f"  - Final dimension: {embed_dim * (2 ** num_stages)}")

    def forward(self, x):
        """
        Args:
            x: Input point cloud, shape [B, N, 9]
               Channels: [xyz (3), rgb (3), xyz (3)]
               
        Returns:
            features: Extracted features, shape [B, C, N]
        """

        pos, rgb = x[:, :, 6:], x[:, :, 3:6]
        xyz, pos_x = pos, pos.permute(0, 2, 1)
        rgb, rgbx = rgb, rgb.permute(0, 2, 1)

        xyz_list, x_list = self.EncNP(xyz, pos_x, rgb, rgbx)
        xyz_feat = self.DecNP(xyz_list, x_list)
        
        return xyz_feat
    


# ============================================================
# Test Code
# ============================================================

if __name__ == "__main__":
    print("Testing Encoder-Decoder Network...\n")
    
    # Create model
    model = Encoder(
        input_points=2048,
        num_stages=3,
        embed_dim=60,
        k_neighbors=16,
        de_neighbors=10,
        alpha=1000,
        beta=30, 
        num_experts=3
    ).cuda()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB\n")
    
    # Create dummy input
    batch_size = 2
    num_points = 2048
    x = torch.randn(batch_size, num_points, 9).cuda()
    
    # Forward pass
    print("Running forward pass...")
    features = model(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {features.shape}")
    print(f"Expected output dim: {60 * (2 ** 3)} = 480")
    
    print("\n✓ Test passed!")
