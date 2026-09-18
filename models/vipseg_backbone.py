"""Shared point feature extractor: VIP-Seg encoder + feature head (Eq.2, spec 01 §2.1, 02 §2).

Layers, names and order follow VIP-Seg exactly [VIPSEG models/vipseg.py:34-53,79-97], so the
`encoder.*`, `bn.*` and `fc.*` weights of a VIP-Seg checkpoint load with strict=True (test ENC-6).
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

NUM_POINT = 2048
IN_CHANNELS = 9  # xyz, rgb, XYZ; the encoder reads rgb (3-5) and XYZ (6-8) [VIPSEG models/encoder.py:645]
ENCODER_DIM = 900
FEATURE_DIM = 128  # D [PAPER §4.1]

# [VIPSEG models/vipseg.py:34-43]
ENCODER_CONFIG = dict(input_points=NUM_POINT, num_stages=3, embed_dim=60, k_neighbors=16, de_neighbors=10,
                      alpha=1000, beta=30, num_experts=3)

# Fixed random projections of the encoder's positional encodings: `vv = torch.randn(1, 5000)` and
# `ww = torch.randn(1, 5000)`, drawn once and shared by every low-/high-order convolution
# [VIPSEG models/encoder.py:619-620,226,311]. They are plain attributes, so a state_dict would not
# store them and a reloaded model would draw new ones. VIP-Seg avoids this by pickling the whole
# model [VIPSEG runs/training.py:93-96]; here they become buffers instead.
FIXED_PROJECTIONS = ("vv", "ww")


def persist_fixed_projections(encoder: nn.Module) -> None:
    """Turn the plain-tensor attributes vv / ww of every submodule into buffers (same values)."""
    for module in encoder.modules():
        for name in FIXED_PROJECTIONS:
            value = module.__dict__.get(name)
            if isinstance(value, torch.Tensor):
                del module.__dict__[name]
                module.register_buffer(name, value)


def fixed_projections_of(model: nn.Module, prefix: str) -> dict:
    """{state_dict key: tensor} for vv / ww stored as plain attributes (e.g. a pickled VIP-Seg model)."""
    found = {}
    for path, module in model.named_modules():
        for name in FIXED_PROJECTIONS:
            value = module.__dict__.get(name)
            if isinstance(value, torch.Tensor):
                found[".".join(p for p in (prefix, path, name) if p)] = value
    return found


class PointFeatureExtractor(nn.Module):
    """f_enc: [B, 2048, 9] -> [B, 2048, 128], every block encoded as its own sample.

    `encoder` defaults to the VIP-Seg encoder (needs mamba_ssm and pointnet2_ops, raises if they are
    missing). Tests on CPU pass a stand-in with the same contract: [B, 2048, 9] -> [B, 900, 2048].
    """

    def __init__(self, encoder: Optional[nn.Module] = None):
        super().__init__()
        if encoder is None:
            from models.encoder import Encoder

            encoder = Encoder(**ENCODER_CONFIG)
        persist_fixed_projections(encoder)
        self.encoder = encoder
        # [VIPSEG models/vipseg.py:45-53]
        self.bn = nn.Sequential(nn.BatchNorm1d(ENCODER_DIM), nn.ReLU())
        self.fc = nn.Sequential(nn.Conv1d(ENCODER_DIM, 196, 1), nn.BatchNorm1d(196), nn.ReLU(),
                                nn.Conv1d(196, FEATURE_DIM, 1), nn.BatchNorm1d(FEATURE_DIM), nn.ReLU())

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        if points.dim() != 3 or points.shape[1:] != (NUM_POINT, IN_CHANNELS):
            raise ValueError(f"points must be [B, {NUM_POINT}, {IN_CHANNELS}], got {tuple(points.shape)}")
        feat = self.encoder(points)  # [B, 900, 2048]
        if feat.shape != (points.shape[0], ENCODER_DIM, NUM_POINT):
            raise ValueError(f"encoder output {tuple(feat.shape)} != {(points.shape[0], ENCODER_DIM, NUM_POINT)}")
        feat = feat / feat.norm(dim=1, keepdim=True)  # [B, 900, 2048], unit norm per point [VIPSEG models/vipseg.py:86]
        feat = self.fc(self.bn(feat))  # [B, 128, 2048] [VIPSEG models/vipseg.py:91-97]
        return feat.transpose(1, 2)  # [B, 2048, 128]

    def load_vipseg_weights(self, vipseg_model: nn.Module) -> None:
        """Copy encoder, feature head and fixed projections from a pickled VIP-Seg model (strict)."""
        state = {k: v for k, v in vipseg_model.state_dict().items() if k.split(".")[0] in ("encoder", "bn", "fc")}
        state.update(fixed_projections_of(vipseg_model.encoder, "encoder"))
        self.load_state_dict(state, strict=True)

    def encode_episode(self, support_x: torch.Tensor, query_x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """F^s [N, K, 2048, 128] and F^q [B_q, 2048, 128] (02 §2).

        The N·K support blocks form one batch and the queries another, as in VIP-Seg
        [VIPSEG models/vipseg.py:79-89]; flatten/unflatten only merge the independent way and shot
        axes into the batch axis and restore them in the same order.
        """
        if support_x.dim() != 4:
            raise ValueError(f"support_x must be [N, K, {NUM_POINT}, {IN_CHANNELS}], got {tuple(support_x.shape)}")
        n_way, k_shot = support_x.shape[:2]
        f_s = self(support_x.flatten(0, 1)).unflatten(0, (n_way, k_shot))  # [N, K, 2048, 128]
        f_q = self(query_x)  # [B_q, 2048, 128]
        return f_s, f_q
