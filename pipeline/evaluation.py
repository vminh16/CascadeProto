"""Evaluation over fixed episodes with VIP-Seg's metric, unchanged (04 §6, [DECISION D-08])."""

from typing import List, Optional

import torch
from torch.utils.data import DataLoader, Subset

from pipeline.episodes import EpisodeCollate
from pipeline.model_api import predict


def accumulated_miou(logger, preds: List, gts: List, sampled_classes: List, test_classes: List[int]) -> float:
    """TP/FP/FN accumulated over all episodes per global test class, background excluded.

    Calls VIP-Seg's `evaluate_metric` [VIPSEG runs/training_free.py:14-61]. That module imports the
    VIP-Seg encoder, so this needs mamba_ssm and pointnet2_ops (the GPU environment of gate G0).
    """
    from runs.training_free import evaluate_metric

    return float(evaluate_metric(logger, preds, gts, sampled_classes, list(test_classes)))


@torch.no_grad()
def evaluate(model, dataset, class_names, logger, device, max_episodes: Optional[int] = None) -> float:
    """Run `model` in eval mode on every episode of a MyTestDataset and return the accumulated mIoU."""
    if max_episodes is not None:
        dataset_view = Subset(dataset, range(min(max_episodes, len(dataset))))
    else:
        dataset_view = dataset
    loader = DataLoader(dataset_view, batch_size=1, shuffle=False, collate_fn=EpisodeCollate(class_names))
    model.eval()
    preds, gts, label2class = [], [], []
    for (episode,) in loader:
        output = model(episode.to(device))
        preds.append(predict(output).cpu().numpy())  # [B_q, 2048]
        gts.append(episode.query_y.numpy())  # [B_q, 2048]
        label2class.append(episode.sampled_classes)  # [N]
    return accumulated_miou(logger, preds, gts, label2class, list(dataset.classes))
