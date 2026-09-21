"""VIP-Seg's released checkpoint behind the episode contract, for the pipeline sanity check (05 §4).

VIP-Seg saves the whole pickled model [VIPSEG runs/training.py:93-96]; loading it needs the inherited
`models.vipseg` module and weights_only=False, so only load checkpoints from a trusted source
(the official VIP-Seg repository).
"""

import torch
import torch.nn as nn

from pipeline.episodes import Episode
from pipeline.model_api import EpisodeOutput


def load_vipseg_model(checkpoint_path: str) -> nn.Module:
    """The pickled VIP-Seg model of a released checkpoint (trusted source only)."""
    import models.vipseg  # noqa: F401  (class definitions needed to unpickle the model)

    return torch.load(checkpoint_path, map_location="cuda", weights_only=False)["model"]


def init_features_from_vipseg(model: nn.Module, checkpoint_path: str) -> None:
    """Copy VIP-Seg's trained encoder, feature head and fixed projections into `model.features`.

    Diagnostic only [DECISION D-20]: it breaks guardrail #1 (no pre-trained point-cloud weights) and
    the paper's "without any pre-trained weights" (§1). It tests whether the paper's baseline row was
    built on VIP-Seg's trained encoder rather than one trained from scratch.
    """
    model.features.load_vipseg_weights(load_vipseg_model(checkpoint_path))  # strict


class VIPSegBaseline(nn.Module):
    def __init__(self, checkpoint_path: str):
        super().__init__()
        self.model = load_vipseg_model(checkpoint_path)

    def forward(self, episode: Episode) -> EpisodeOutput:
        support_x = episode.support_x.permute(0, 1, 3, 2)  # [N, K, 9, 2048], VIP-Seg layout
        query_x = episode.query_x.permute(0, 2, 1)  # [B_q, 9, 2048]
        logits, _ = self.model(support_x, episode.support_y, query_x, episode.query_y)  # [B_q, 2048, N+1]
        return EpisodeOutput(logits=logits, loss_gmmn=logits.new_zeros(()))
