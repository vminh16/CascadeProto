"""Short differential training harness for the phase-14 gap (DEBUG, not part of the pipeline).

Symptom: after the full schedule our baseline reaches 49.5 and the full model 58.0 valid mIoU, while
VIP-Seg trained with its own code reaches 68.9 on the same valid episodes after 2,000 episodes.

Every variant trains from scratch on the same seeded episodes with AdamW (lr 1e-3, wd 0.1) and no LR
decay (both schedules decay only after this budget), then scores 300 valid episodes spread over all
class combinations. Only the variant changes, one variable at a time.

Each run also reports the regime of Eq.14 on real encoder features before and after training
[DECISION D-18]: `attn_width` out of D = 128, and the channel variation of `P_cross`. A width near 1
or near 128 both mean `P_cross` is constant along D, i.e. the stage can only pass the residual
through. That measurement is the one the CPU cannot make.

    python experiments/diag_short.py --data_path datasets/S3DIS/blocks_bs1_s1 --variants vipseg baseline
"""

import argparse
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import train  # noqa: E402
from pipeline.episodes import (EpisodeCollate, SeededEpisodes, build_eval_dataset, build_train_dataset,  # noqa: E402
                               read_class_names)
from pipeline.evaluation import evaluate  # noqa: E402
from models.prototypes import point_prototypes  # noqa: E402
from pipeline.model_api import EpisodeOutput, episode_loss  # noqa: E402

VARIANTS = {  # name: train.py switches (None = VIP-Seg's own model)
    "vipseg": None,
    "baseline": ["--use_lma", "false", "--num_stages", "0"],
    "baseline_l2": ["--use_lma", "false", "--num_stages", "0", "--l2norm_point_proto", "true"],
    "full": [],
    "full_l2": ["--l2norm_point_proto", "true"],
    "full_norm": ["--cross_attn_norm", "layernorm"],  # [DECISION D-18]
}


class VIPSegFresh(nn.Module):
    """VIP-Seg's model built from scratch, behind the episode contract (same call as VIPSegBaseline)."""

    def __init__(self, n_way, k_shot):
        super().__init__()
        from models.vipseg import VIPSeg

        self.model = VIPSeg(SimpleNamespace(n_way=n_way, k_shot=k_shot, pc_npts=2048))

    def forward(self, episode):
        logits, _ = self.model(episode.support_x.permute(0, 1, 3, 2), episode.support_y,
                               episode.query_x.permute(0, 2, 1), episode.query_y)
        return EpisodeOutput(logits=logits, loss_gmmn=logits.new_zeros(()))


class StridedView(torch.utils.data.Dataset):
    """Every `stride`-th valid episode, so all class combinations are represented."""

    def __init__(self, base, stride):
        self.base, self.idx, self.classes = base, list(range(0, len(base), stride)), base.classes

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        return self.base[self.idx[i]]


class _Print:
    def cprint(self, text):
        pass


def attention_regime(model, episode, device):
    """Softmax width and channel variation of P_cross in stage 1, on real encoder features [D-18].

    width = exp(H(row)) averaged over the rows of A: 1 means every row copies one channel, D = 128
    means uniform; both ends leave P_cross constant along D. chan_var = std_D / |mean_D| of P_cross.
    Returns (nan, nan) for a model without EPPM stages.
    """
    stages = getattr(model, "stages", None)
    if not stages:
        return float("nan"), float("nan")
    stage, was_training = stages[0], model.training
    model.eval()  # the probe must not move the BatchNorm running statistics
    with torch.no_grad():
        ep = episode.to(device)
        f_s, f_q = model.features.encode_episode(ep.support_x, ep.query_x)
        p = point_prototypes(f_s, ep.support_y).unsqueeze(0).expand(f_q.shape[0], -1, -1)  # P^0 (Eq.3)
        a = stage.cross.attention(f_s, f_q)
        width = torch.exp(-(a * (a + 1e-30).log()).sum(-1)).mean()
        p_cross = stage.cross(stage.gate(p), f_s, f_q)
        chan = (p_cross[0].std(-1) / p_cross[0].mean(-1).abs()).max()
    model.train(was_training)
    return float(width), float(chan)


def build(variant, device):
    if VARIANTS[variant] is None:
        return VIPSegFresh(2, 1).to(device)
    args = train.parse_args(["--dataset", "s3dis", "--data_path", "x", "--cvfold", "0", "--n_way", "2",
                             "--k_shot", "1"] + VARIANTS[variant])
    return train.build_model(train.model_config(args)).to(device)


def run(variant, data_path, episodes, batch, stride, seed, device):
    train.seed_everything(seed)
    model = build(variant, device)
    names = read_class_names(data_path, "s3dis")
    data = SeededEpisodes(build_train_dataset(data_path, "s3dis", 0, 2, 1, num_episode=episodes), seed)
    loader = DataLoader(data, batch_size=batch, shuffle=False, num_workers=4, collate_fn=EpisodeCollate(names))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    t0, losses, first = time.time(), [], None
    w0 = c0 = float("nan")
    model.train()
    for eps in loader:
        if first is None:
            first = eps[0]
            w0, c0 = attention_regime(model, first, device)
        eps = [ep.to(device) for ep in eps]
        loss = torch.stack([episode_loss(model(ep), ep) for ep in eps]).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    w1, c1 = attention_regime(model, first, device)
    valid = build_eval_dataset(data_path, "s3dis", 0, 2, 1, mode="valid", seed=0)
    with torch.no_grad():
        miou = evaluate(model, StridedView(valid, stride), names, _Print(), device)
    return {"variant": variant, "batch": batch, "steps": len(losses), "episodes": episodes,
            "loss_last100": float(np.mean(losses[-100:])), "valid_miou": miou,
            "attn_width_init": w0, "attn_width_end": w1, "p_cross_chan_var_init": c0,
            "p_cross_chan_var_end": c1, "seconds": time.time() - t0}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=True)
    p.add_argument("--variants", nargs="+", default=["vipseg", "baseline"], choices=sorted(VARIANTS))
    p.add_argument("--batches", nargs="+", type=int, default=[4])
    p.add_argument("--episodes", type=int, default=2400)
    p.add_argument("--stride", type=int, default=5, help="1500 valid episodes / 5 = 300")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    device = torch.device("cuda")
    for variant in args.variants:
        for batch in args.batches:
            r = run(variant, args.data_path, args.episodes, batch, args.stride, args.seed, device)
            print("[diag] " + " | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()),
                  flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
