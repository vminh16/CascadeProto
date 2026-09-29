"""Arm A of [DECISION D-48] (amendment 6): the correlation neck that replaces the PEM/PDM head. **Beyond the paper.**

Per episode:
  cells       adaptive support cells per row (background over every mask-0 point, each way over its K shots),
              raw unit-feature space (P11.4's frozen space), caps 16 / 16            `models/correlation.py`
  descriptor  (max, top-2 mean, mean) of the cosines of each query point to each row's cells     [B_q, P, R, 3]
  embedding   W_d d + e_type(row) + W_x xyz                                                        [B_q, P, R, W]
  L layers    pre-norm attention across the P points of each (block, row), a feed-forward, and an equivariant
              mixing across the rows  h_r <- h_r + W_1 h_r + W_2 mean_r' h_r'
  readout     a shared linear map W -> 1 after every layer: logits l_1 .. l_L                     [B_q, P, R]
Only the row type (background or foreground) tells the rows apart, so the head is equivariant to the order of the
ways. No tensor is reshaped across the block, row or point axes: attention runs on [B_q, R, H, P, d].
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import correlation as corr

WIDTH, HEADS, FFN = 64, 4, 128
CAP_FG, CAP_BG = 16, 16  # [DECISION D-48 amendment 2]


class RowAttentionLayer(nn.Module):
    """Pre-norm self-attention over the points of each (block, row), a feed-forward, and the row mixing."""

    def __init__(self, width: int = WIDTH, heads: int = HEADS, ffn: int = FFN):
        super().__init__()
        if width % heads:
            raise ValueError(f"width {width} is not a multiple of heads {heads}")
        self.heads = heads
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.out = nn.Linear(width, width)
        self.ffn = nn.Sequential(nn.Linear(width, ffn), nn.GELU(), nn.Linear(ffn, width))
        self.mix_self, self.mix_mean = nn.Linear(width, width), nn.Linear(width, width, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h [B, P, R, W] -> [B, P, R, W]."""
        b, p, r, w = h.shape
        q, k, v = self.qkv(self.norm1(h)).chunk(3, dim=-1)  # 3 x [B, P, R, W]
        split = lambda x: x.unflatten(-1, (self.heads, w // self.heads)).permute(0, 2, 3, 1, 4)  # noqa: E731
        att = F.scaled_dot_product_attention(split(q), split(k), split(v))  # [B, R, H, P, W/H]
        h = h + self.out(att.permute(0, 3, 1, 2, 4).flatten(-2))  # [B, P, R, W], heads merged on the channel axis
        h = h + self.ffn(self.norm2(h))  # [B, P, R, W]
        return h + self.mix_self(h) + self.mix_mean(h.mean(dim=2, keepdim=True))  # equivariant across rows


class CorrelationHead(nn.Module):
    def __init__(self, layers: int, width: int = WIDTH):
        super().__init__()
        if layers < 1:
            raise ValueError(f"the correlation head needs at least one layer, got {layers}")
        self.embed = nn.Linear(3, width)
        self.row_type = nn.Embedding(2, width)  # 0 background, 1 foreground
        self.pos = nn.Linear(3, width)
        self.layers = nn.ModuleList(RowAttentionLayer(width) for _ in range(layers))
        self.readout = nn.Linear(width, 1)

    def forward(self, f_q: torch.Tensor, f_s: torch.Tensor, support_y: torch.Tensor,
                xyz: torch.Tensor) -> List[torch.Tensor]:
        """Logits of every layer, each [B_q, P, N+1]; f_q [B_q, P, D], f_s [N, K, P, D], support_y [N, K, P],
        xyz [B_q, P, 3] the query blocks' xyz channels."""
        u_q, u_s = F.normalize(f_q, dim=-1), F.normalize(f_s, dim=-1)
        rows = corr.row_cells(u_s, support_y, CAP_FG, CAP_BG)  # N+1 x (prototypes, sizes)
        desc = corr.row_descriptors(u_q, rows)  # [B_q, P, N+1, 3]
        types = torch.ones(desc.shape[2], dtype=torch.long, device=desc.device)  # [N+1]
        types[0] = 0
        h = self.embed(desc) + self.row_type(types) + self.pos(xyz).unsqueeze(2)  # [B_q, P, N+1, W]
        out = []
        for layer in self.layers:
            h = layer(h)
            out.append(self.readout(h).squeeze(-1))  # [B_q, P, N+1]
        return out


def deep_supervision(logits: List[torch.Tensor], labels: torch.Tensor) -> torch.Tensor:
    """(1/L) sum_l CE(l_l, Y) [DECISION D-48 amendment 1]; logits L x [B_q, P, N+1], labels [B_q, P]."""
    return torch.stack([F.cross_entropy(l.movedim(-1, 1), labels) for l in logits]).mean()
