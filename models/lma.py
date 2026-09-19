"""Learnable Modality Adapter and prototype generator of Eq.4-6 (spec 02 §4.1-4.2, spec 03 §3).

One episode at a time: E_CLIP [N+1, 512] -> Adapter -> E_adapted [N+1, D] -> G([E_adapted; z]) ->
P_modal [N+1, D], row 0 = background. The CLIP front-end that produces E_CLIP is models/clip_text.py.
"""

from typing import Optional

import torch
import torch.nn as nn

CLIP_DIM = 512  # [PAPER §4.1]
FEATURE_DIM = 128  # D [PAPER Eq.2]
ADAPTER_DROPOUT = 0.1  # [DECISION D-16]
EVAL_NOISE = ("zero", "sample", "mean_of_M")  # [DECISION D-06]


class ModalityAdapter(nn.Module):
    """Eq.5: W_2 Dropout(ReLU(LN(W_1 E + b_1))) + b_2, [N+1, 512] -> [N+1, D] [DECISION D-16]."""

    def __init__(self, in_dim: int = CLIP_DIM, dim: int = FEATURE_DIM):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(ADAPTER_DROPOUT)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(self.relu(self.norm(self.fc1(e)))))  # [N+1, D]


class PrototypeGenerator(nn.Module):
    """G of Eq.6, a three-layer MLP on [E_fused; z]: [N+1, 2D] -> [N+1, D] [DECISION D-05] [DECISION D-16]."""

    def __init__(self, dim: int = FEATURE_DIM):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * dim, dim), nn.ReLU(), nn.Linear(dim, dim), nn.ReLU(),
                                 nn.Linear(dim, dim))

    def forward(self, e_fused_z: torch.Tensor) -> torch.Tensor:
        return self.net(e_fused_z)  # [N+1, D]


class LearnableModalityAdapter(nn.Module):
    """Adapter^(m) of the run's single modality followed by G [PAPER Eq.4-6] [DECISION D-05].

    Noise z ~ N(0, I) of shape [N+1, D] is drawn afresh on every forward in training; in evaluation
    z = 0 (eval_noise='zero', default) or a fresh draw (eval_noise='sample') [DECISION D-06].
    """

    def __init__(self, eval_noise: str = "zero", dim: int = FEATURE_DIM):
        super().__init__()
        if eval_noise not in EVAL_NOISE:
            raise ValueError(f"eval_noise must be one of {EVAL_NOISE}, got {eval_noise!r}")
        if eval_noise == "mean_of_M":
            raise NotImplementedError("eval_noise='mean_of_M' needs a value of M, which the paper does not give (D-06)")
        self.eval_noise = eval_noise
        self.adapter = ModalityAdapter(CLIP_DIM, dim)
        self.generator = PrototypeGenerator(dim)

    def forward(self, e_clip: torch.Tensor, z: Optional[torch.Tensor] = None) -> torch.Tensor:
        """e_clip [N+1, 512] -> P_modal [N+1, D]; `z` overrides the noise (tests only)."""
        if e_clip.dim() != 2 or e_clip.shape[1] != CLIP_DIM:
            raise ValueError(f"expected E_CLIP [N+1, {CLIP_DIM}], got {tuple(e_clip.shape)}")
        e_fused = self.adapter(e_clip)  # [N+1, D]; E_fused := E_adapted^(m) [DECISION D-05]
        if z is None:
            if self.training or self.eval_noise == "sample":
                z = torch.randn_like(e_fused)  # [N+1, D]
            else:
                z = torch.zeros_like(e_fused)  # [N+1, D]
        if z.shape != e_fused.shape:
            raise ValueError(f"z {tuple(z.shape)} != {tuple(e_fused.shape)}")
        return self.generator(torch.cat([e_fused, z], dim=-1))  # [N+1, 2D] -> [N+1, D]
