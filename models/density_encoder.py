"""A density-invariant variant of VIP-Seg's encoder [DECISION D-43]. **Beyond the paper.**

P8 (D-41) showed that the sampling density sets the recall through the features. This encoder keeps VIP-Seg's
modules, parameter names and shapes, and changes the four places where density enters [DECISION D-43]:

1. neighbourhoods: the points within a metric radius of each FPS centre (`ball_group`), not the k nearest;
2. offsets: divided by that radius, not by a std of the whole batch;
3. every batch-global mean / std: the block's own (`per_block_scale`, `per_block_standardize`), so a block's
   features no longer depend on the other blocks encoded with it;
4. coordinates: the metric channels 0-2 instead of the per-axis normalised channels 6-8.

The inherited classes of `models/encoder.py` are subclassed, never edited (AGENTS guardrail 2); each overridden
`forward` repeats the inherited computation line by line (cited) with only the changes above. Imports the VIP-Seg
encoder, so it needs `mamba_ssm` and `pointnet2_ops` (GPU environment, 05 §2 gate G2).
"""

import torch
import torch.nn as nn

from models.density_ops import BALL_K, BALL_RADII, METRIC_CHANNELS, ball_group, per_block_scale, per_block_standardize
from models.encoder import (DynamicHighOrderConvolution, DyPowerConv, Encoder, LowOrderConvolution,
                            NonParametricDecoder)
from models.model_utils import index_points, square_distance
from pointnet2_ops_lib.pointnet2_ops import pointnet2_utils


class MetricBallGrouping(nn.Module):
    """`FPS_kNN` with a ball of fixed radius in place of the k nearest points [VIPSEG models/encoder.py:72-93]."""

    def __init__(self, group_num: int, k: int, radius: float):
        super().__init__()
        self.group_num, self.k, self.radius = group_num, k, radius

    def forward(self, xyz, x, rgb):
        fps_idx = pointnet2_utils.furthest_point_sample(xyz.contiguous(), self.group_num).long()  # [B, G]
        lc_xyz, lc_x, lc_rgb = index_points(xyz, fps_idx), index_points(x, fps_idx), index_points(rgb, fps_idx)
        idx = ball_group(xyz, lc_xyz, self.radius, self.k)  # [B, G, K]
        return lc_xyz, lc_x, lc_rgb, index_points(xyz, idx), index_points(x, idx), index_points(rgb, idx)


class DensityLoConv(LowOrderConvolution):
    """LoConv with the block's own standardisation [VIPSEG models/encoder.py:236-291]."""

    def forward(self, knn_xyz, knn_x, knn_rgb):
        dev = knn_xyz.device
        feat_dim = self.out_dim // (self.in_dim * 2)
        feat_range = torch.arange(feat_dim, device=dev).float()
        dim_embed = torch.pow(self.alpha, feat_range / feat_dim)
        knn_xyz_new = knn_xyz.permute(0, 2, 3, 1)[..., None]
        div_xyz = torch.div(self.beta * knn_xyz_new.unsqueeze(-1), dim_embed)
        xyz_embed = torch.cat([torch.sin(div_xyz), torch.cos(div_xyz)], dim=4).flatten(3).permute(0, 3, 1, 2)
        knn_rgb_1 = knn_rgb.permute(0, 2, 3, 1)[..., None]
        div_rgb = torch.div(self.beta * knn_rgb_1.unsqueeze(-1), dim_embed)
        rgb_embed = torch.cat([torch.sin(div_rgb), torch.cos(div_rgb)], dim=4).flatten(3).permute(0, 3, 1, 2)
        knn_x_new = self.mlp(knn_x / 3 + xyz_embed / 3 + rgb_embed / 3)  # [B, C, G, K]
        knn_x_new = knn_x_new.permute(0, 2, 3, 1)  # [B, G, K, C]
        pos = self.vv[:, :self.out_dim].T.to(dev) @ torch.arange(self.out_dim, device=dev).unsqueeze(0).float()
        knn_x_new = knn_x_new @ torch.cos(pos * 2 * torch.pi)  # [B, G, K, C]
        knn_x_new = per_block_standardize(knn_x_new, 1e-6).permute(0, 3, 1, 2)  # [B, C, G, K], was batch-global
        return self.pooling(knn_x_new)  # [B, C, G]


class DensityDyHiConv(DynamicHighOrderConvolution):
    """DyHiConv with the block's own standardisation [VIPSEG models/encoder.py:336-421]."""

    def forward(self, lc_xyz, knn_xyz, knn_x, knn_rgb):
        dev = knn_xyz.device
        B, _, N, K = knn_xyz.shape
        feat_dim = self.out_dim // (self.in_dim * 2)
        feat_range = torch.arange(feat_dim, device=dev).float()
        dim_embed = torch.pow(self.alpha, feat_range / feat_dim)
        knn_xyz_1 = knn_xyz.permute(0, 2, 3, 1)[..., None]
        div_xyz = torch.div(self.beta * knn_xyz_1.unsqueeze(-1), dim_embed)
        xyz_embed = torch.cat([torch.sin(div_xyz), torch.cos(div_xyz)], dim=4).flatten(3).permute(0, 3, 1, 2)
        knn_rgb_1 = knn_rgb.permute(0, 2, 3, 1)[..., None]
        div_rgb = torch.div(self.beta * knn_rgb_1.unsqueeze(-1), dim_embed)
        rgb_embed = torch.cat([torch.sin(div_rgb), torch.cos(div_rgb)], dim=4).flatten(3).permute(0, 3, 1, 2)
        knn_x_new = knn_x / 3 + xyz_embed / 3 + rgb_embed / 3  # [B, C, N, K]
        relative_knn_x_new = knn_x_new - torch.mean(knn_x_new, dim=1, keepdim=True)
        coord_xi = lc_xyz.unsqueeze(dim=-2).permute(0, 3, 1, 2).repeat(1, 1, 1, K)  # [B, 3, N, K]
        delta_x = knn_xyz - coord_xi  # [B, 3, N, K], as VIP-Seg (knn_xyz holds the scaled offsets)
        abs_xi_xj = torch.norm(delta_x, p=2, dim=1).unsqueeze(1)  # [B, 1, N, K]
        h_xi_xj = torch.cat((coord_xi, knn_xyz, delta_x, abs_xi_xj), dim=1)  # [B, 10, N, K]
        expert_weights = self.weight_generator(h_xi_xj).view(B, self.num_experts, -1, N, K)
        expert_scores = self.expert_selector(h_xi_xj)  # [B, num_experts, 1, 1]
        outputs = []
        for t in range(self.num_experts):
            aggregated_f = self.HiConv(expert_weights[:, t], relative_knn_x_new, self.power_p[t])  # [B, C, N]
            outputs.append(aggregated_f * expert_scores[:, t])
        knn_x_new = torch.stack(outputs, dim=-1).sum(-1)  # [B, C, N]
        knn_x_new = knn_x_new.unsqueeze(-1).permute(0, 2, 3, 1)  # [B, N, 1, C]
        pos = self.ww[:, :self.out_dim].T.to(dev) @ torch.arange(self.out_dim, device=dev).unsqueeze(0).float()
        knn_x_new = knn_x_new @ torch.cos(pos * 2 * torch.pi)  # [B, N, 1, C]
        knn_x_new = per_block_standardize(knn_x_new, 1e-6)  # was batch-global
        return knn_x_new.permute(0, 3, 1, 2).squeeze(-1)  # [B, C, N]


class DensityDyPowerConv(DyPowerConv):
    """DyPowerConv with offsets in units of the ball radius and the block's own feature scale
    [VIPSEG models/encoder.py:164-218]."""

    def __init__(self, out_dim, vv, ww, num_experts, radius: float):
        super().__init__(out_dim, 1, 1, vv, ww, num_experts)  # alpha = beta = 1 inside, as VIP-Seg
        self.LoConv = DensityLoConv(3, out_dim, 1, 1, vv)
        self.DyHiConv = DensityDyHiConv(3, out_dim, 1, 1, ww, num_experts)
        self.radius = radius

    def forward(self, lc_xyz, lc_x, lc_rgb, knn_xyz, knn_x, knn_rgb):
        knn_x = per_block_scale(knn_x - lc_x.unsqueeze(dim=-2), 1e-5)  # [B, G, K, C], was / batch std
        knn_xyz = (knn_xyz - lc_xyz.unsqueeze(dim=-2)) / self.radius  # [B, G, K, 3], was / batch std
        B, G, K, C = knn_x.shape
        knn_x = torch.cat([knn_x, lc_x.reshape(B, G, 1, -1).repeat(1, 1, K, 1)], dim=-1)  # [B, G, K, 2C]
        knn_xyz, knn_x, knn_rgb = knn_xyz.permute(0, 3, 1, 2), knn_x.permute(0, 3, 1, 2), knn_rgb.permute(0, 3, 1, 2)
        return self.LoConv(knn_xyz, knn_x, knn_rgb) + self.DyHiConv(lc_xyz, knn_xyz, knn_x, knn_rgb)  # [B, C, G]


class DensityDecoder(NonParametricDecoder):
    """The non-parametric decoder with the block's own standardisation [VIPSEG models/encoder.py:549-603]."""

    def propagate(self, xyz1, xyz2, points1, points2):
        points2 = points2.permute(0, 2, 1)  # [B, S, C]
        B, N, _ = xyz1.shape
        S = xyz2.shape[1]
        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists, idx = square_distance(xyz1, xyz2).sort(dim=-1)
            dists, idx = dists[:, :, :self.de_neighbors], idx[:, :, :self.de_neighbors]  # [B, N, k]
            dist_recip = 1.0 / (dists + 1e-8)
            weight = (dist_recip / torch.sum(dist_recip, dim=2, keepdim=True)).view(B, N, self.de_neighbors, 1)
            interpolated_points = torch.sum(index_points(points2, idx) * weight, dim=2)  # [B, N, C]
            interpolated_points = per_block_standardize(interpolated_points, 1e-5)  # was batch-global
        if points1 is not None:
            new_points = torch.cat([points1.permute(0, 2, 1), interpolated_points], dim=-1)
        else:
            new_points = interpolated_points
        return new_points.permute(0, 2, 1)


class DensityEncoder(Encoder):
    """VIP-Seg's encoder with the four changes of [DECISION D-43]; same constructor arguments as `Encoder`."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        enc = self.EncNP
        if enc.num_stages != len(BALL_RADII):
            raise ValueError(f"{enc.num_stages} stages but {len(BALL_RADII)} radii [DECISION D-43]")
        for i in range(enc.num_stages):
            old_group, old_conv = enc.FPS_kNN_list[i], enc.DyPowerConv_list[i]
            if old_group.k_neighbors != BALL_K:
                raise ValueError(f"stage {i}: k = {old_group.k_neighbors}, the ball keeps {BALL_K}")
            enc.FPS_kNN_list[i] = MetricBallGrouping(old_group.group_num, BALL_K, BALL_RADII[i])
            enc.DyPowerConv_list[i] = DensityDyPowerConv(old_conv.LoConv.out_dim, old_conv.LoConv.vv,
                                                         old_conv.DyHiConv.ww, old_conv.DyHiConv.num_experts,
                                                         BALL_RADII[i])
        self.DecNP = DensityDecoder(self.DecNP.num_stages, self.DecNP.de_neighbors)

    def forward(self, x):
        """x [B, N, 9] (xyz, rgb, XYZ) -> [B, C, N]; reads the metric channels [VIPSEG models/encoder.py:635-650]."""
        pos, rgb = x[:, :, METRIC_CHANNELS], x[:, :, 3:6]
        xyz_list, x_list = self.EncNP(pos, pos.permute(0, 2, 1), rgb, rgb.permute(0, 2, 1))
        return self.DecNP(xyz_list, x_list)
