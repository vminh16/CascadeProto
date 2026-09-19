"""Parameters, FLOPs and time per episode of one configuration (Table 6, [DECISION D-09]).

    python experiments/complexity.py --dataset s3dis --data_path x --cvfold 0 --n_way 2 --k_shot 1 [switches]

Takes the same switches as train.py. Fixed setting of D-09: 2-way 1-shot, one query per way (the
loader's), 9 channels, fvcore for FLOPs. fvcore does not count custom CUDA kernels (pointnet2
sampling/grouping, the Mamba selective scan); they are listed as unsupported, so the FLOPs are a
lower bound. Table 6 is not an acceptance criterion (D-09); the numbers are for the report.
"""

import json
import os
import statistics
import sys
import time
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import train  # noqa: E402
from pipeline.episodes import Episode  # noqa: E402

S3DIS_NAMES = ["ceiling", "floor", "wall", "beam", "column", "window", "door", "table", "chair", "sofa",
               "bookcase", "board", "clutter"]


def parameter_groups(model: nn.Module) -> Dict[str, int]:
    """Parameter counts of the parts named in 01 §4; the groups add up to the total."""
    count = lambda mod: sum(p.numel() for p in mod.parameters()) if mod is not None else 0
    encoder = count(getattr(model.features, "encoder", None))
    groups = {"encoder": encoder, "feature_head": count(model.features) - encoder,
              "lma": count(getattr(model, "lma", None)), "eppm_stages": count(model.stages),
              "adrm": count(model.routing)}
    groups["added_modules"] = groups["lma"] + groups["eppm_stages"] + groups["adrm"]
    groups["total"] = count(model)
    return groups


def synthetic_episode(n_way: int, k_shot: int, device, dtype=torch.float32, seed: int = 0) -> Episode:
    """Loader-shaped random episode (02 §1); only shapes matter for FLOPs and time."""
    g = np.random.default_rng(seed)
    support_y = torch.from_numpy((g.random((n_way, k_shot, 2048)) < 0.3).astype(np.int64))
    support_y[..., 0] = 1  # every support block has foreground points
    return Episode(support_x=torch.from_numpy(g.random((n_way, k_shot, 2048, 9))).to(dtype),
                   support_y=support_y,
                   query_x=torch.from_numpy(g.random((n_way, 2048, 9))).to(dtype),
                   query_y=torch.from_numpy(g.integers(0, n_way + 1, (n_way, 2048))),
                   sampled_classes=np.arange(3, 3 + n_way), class_names=S3DIS_NAMES[3:3 + n_way]).to(device)


class _TensorInputs(nn.Module):
    """fvcore traces tensor arguments only; rebuild the Episode inside."""

    def __init__(self, model, episode):
        super().__init__()
        self.model, self.episode = model, episode

    def forward(self, support_x, support_y, query_x, query_y):
        ep = Episode(support_x, support_y, query_x, query_y, self.episode.sampled_classes, self.episode.class_names)
        return self.model(ep).logits


def flops(model: nn.Module, episode: Episode) -> Dict[str, object]:
    from fvcore.nn import FlopCountAnalysis

    analysis = FlopCountAnalysis(_TensorInputs(model, episode),
                                 (episode.support_x, episode.support_y, episode.query_x, episode.query_y))
    analysis.unsupported_ops_warnings(False).uncalled_modules_warnings(False)
    return {"gflops": analysis.total() / 1e9, "unsupported_ops": dict(analysis.unsupported_ops())}


def time_per_episode(model: nn.Module, episode: Episode, repeats: int = 20, warmup: int = 5) -> float:
    """Median wall time of one evaluation forward in milliseconds."""
    sync = torch.cuda.synchronize if episode.query_x.is_cuda else (lambda: None)
    times = []
    with torch.no_grad():
        for i in range(warmup + repeats):
            sync()
            t0 = time.perf_counter()
            model(episode)
            sync()
            if i >= warmup:
                times.append(1000.0 * (time.perf_counter() - t0))
    return statistics.median(times)


def measure(model: nn.Module, n_way: int, k_shot: int, device, repeats: int = 20) -> Dict[str, object]:
    model = model.eval()
    ep = synthetic_episode(n_way, k_shot, device, dtype=next(model.parameters()).dtype)
    return {"parameters": parameter_groups(model), **flops(model, ep),
            "ms_per_episode": time_per_episode(model, ep, repeats=repeats), "device": str(device)}


def main(argv=None) -> int:
    args = train.parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = train.build_model(train.model_config(args)).to(device)
    result = measure(model, args.n_way, args.k_shot, device)
    result["config"] = train.model_config(args).to_dict()
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
