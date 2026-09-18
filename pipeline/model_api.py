"""Contract between the episode pipeline and any segmentation model (CascadeProto, VIP-Seg baseline).

A model is an nn.Module whose forward takes one Episode and returns EpisodeOutput. Training and
evaluation never touch model internals, so the model can change (phases 10-13) without touching
train.py / eval.py.
"""

from typing import NamedTuple

import torch
import torch.nn.functional as F

from pipeline.episodes import Episode

GMMN_WEIGHT = 1.0  # lambda of Eq.26 [PAPER §4.1] (02 §7)


class EpisodeOutput(NamedTuple):
    logits: torch.Tensor  # [B_q, 2048, N+1] final logits L_final (02 §6)
    loss_gmmn: torch.Tensor  # scalar L_GMMN (02 §4.4); a zero tensor for models without it


def episode_loss(output: EpisodeOutput, episode: Episode) -> torch.Tensor:
    """L_total = CE(L_final, Y_q) + lambda * L_GMMN, unweighted CE averaged over all query points (02 §7)."""
    n_classes = episode.n_way + 1
    logits = output.logits  # [B_q, 2048, N+1]
    if logits.shape != (*episode.query_y.shape, n_classes):
        raise ValueError(f"logits {tuple(logits.shape)} != {(*episode.query_y.shape, n_classes)}")
    seg = F.cross_entropy(logits.reshape(-1, n_classes), episode.query_y.reshape(-1))  # scalar
    return seg + GMMN_WEIGHT * output.loss_gmmn


def predict(output: EpisodeOutput) -> torch.Tensor:
    """Y_hat = argmax_c L_final (02 §8)."""
    return output.logits.argmax(dim=-1)  # [B_q, 2048]
