"""C3: does a model's prediction follow the query block's position in the episode? (GPU, no training; diagnostic)

The inherited loader appends the query block sampled for class k at position k [VIPSEG dataloaders/loader.py:181-222],
in training and in the cached test episodes alike, so "query block b holds class b + 1" is always true. This script
scores each checkpoint on the fixed100 episodes twice: in the stored order and with the two query blocks swapped
(labels travel with their blocks; predictions are mapped back). A model that segments by the support prototypes is
unaffected by the swap; a model that learned the position is not. Reported: mIoU in both orders, and for every
query block's own class the share predicted as that class, as the label of the position the block moved to, and as
background. That label is the other episode class's, which an order-free model also predicts by confusion, so the
stored-order share of that label (`normal_other`) is recorded too; the relabel shift is `swap_position - normal_other`
[DECISION D-36]. The support rule without head (`F^q n(P_point)ᵀ`) is scored the same way as a reference.

    python experiments/c3_query_order.py --data_path datasets/S3DIS/blocks_bs1_s1 \
        --checkpoint e1:ours:1:log_r2/s3dis_S1_N2_K1_point_T4_vip_b1/last.pt --checkpoint vipseg:vipseg:1:vipseg_S1_N2_K1.pt
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from experiments import p0_em_probe as p0  # noqa: E402
from experiments import p5_condition_probe as p5  # noqa: E402
from experiments import r2_distill_eval as r2  # noqa: E402

OUT = "results/phase16_p5/c3_query_order.json"


def swapped(item):
    """The same episode with its two query blocks exchanged (labels move with their blocks)."""
    sx, sy, qx, qy, sc = item
    return sx, sy, qx[::-1].copy(), qy[::-1].copy(), sc


@torch.no_grad()
def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=True)
    p.add_argument("--checkpoint", action="append", required=True)
    p.add_argument("--max_episodes", type=int, default=None)
    p.add_argument("--out", default=OUT, help="results/phase16_d37/... for the arms of D-37")
    args = p.parse_args(argv)
    from pipeline.episodes import make_episode, read_class_names

    dev = torch.device("cuda")
    names = read_class_names(args.data_path, "s3dis")
    ds, test_classes = r2.episodes_of("fixed100", args.data_path, 1)
    n_ep = len(ds) if args.max_episodes is None else min(args.max_episodes, len(ds))
    out = {"episodes": n_ep, "checkpoints": {}}
    for spec in args.checkpoint:
        ck = p0.parse_checkpoint(spec)
        rule, protocol, _ = r2.load(ck, 1, dev)  # protocol guard [DECISION D-22]
        counts = {k: [] for k in ("model", "model_swapped", "support_rule", "support_rule_swapped")}
        own = {"points": 0, "normal_own": 0, "normal_other": 0, "swap_own": 0, "swap_position": 0, "swap_bg": 0,
               "sr_normal_own": 0, "sr_swap_own": 0}
        for i in range(n_ep):
            item = ds[i]
            qy = item[3]  # [2, P]
            for tag, it in (("", item), ("_swapped", swapped(item))):
                e = make_episode(it, names).to(dev)
                f_q, m_eff, _, logits = rule(e)
                p0.check_identity(f_q, m_eff, logits)
                f_s = p5.support_features(rule, e)
                pm = logits.argmax(-1).cpu().numpy()  # [2, P] in this order
                ps = p5.support_rule_logits(f_q, f_s, e.support_y).argmax(-1).cpu().numpy()  # [2, P]
                if tag:
                    pm, ps = pm[::-1], ps[::-1]  # back to the stored order
                counts["model" + tag].append(p0.episode_counts(pm, qy, item[4], test_classes))
                counts["support_rule" + tag].append(p0.episode_counts(ps, qy, item[4], test_classes))
                for b in range(2):
                    m = qy[b] == b + 1  # the block's own (sampled-for) class
                    if not tag:
                        own["points"] += int(m.sum())
                        own["normal_own"] += int((pm[b][m] == b + 1).sum())
                        own["normal_other"] += int((pm[b][m] == 2 - b).sum())  # honest confusion [D-36]
                        own["sr_normal_own"] += int((ps[b][m] == b + 1).sum())
                    else:  # block b sat at position 1 - b, whose own label is 2 - b
                        own["swap_own"] += int((pm[b][m] == b + 1).sum())
                        own["swap_position"] += int((pm[b][m] == 2 - b).sum())
                        own["swap_bg"] += int((pm[b][m] == 0).sum())
                        own["sr_swap_own"] += int((ps[b][m] == b + 1).sum())
        rule.close()
        stacked = {k: np.stack(v) for k, v in counts.items()}
        n = max(own["points"], 1)
        res = {"protocol": protocol,
               "miou": {k: float(p0.miou_from_counts(v.sum(0))) for k, v in stacked.items()},
               "class_iou": {k: p0.class_ious(v.sum(0)).tolist() for k, v in stacked.items()},
               "own_class_share": {k: v / n for k, v in own.items() if k != "points"}, "own_points": own["points"],
               "paired_swap": p0.paired_bootstrap(stacked["model"], stacked["model_swapped"])}
        out["checkpoints"][ck.name] = res
        s = res["own_class_share"]
        print(f"[c3] {ck.name}: mIoU stored order {100 * res['miou']['model']:.2f}, swapped {100 * res['miou']['model_swapped']:.2f} | "
              f"support rule {100 * res['miou']['support_rule']:.2f} / {100 * res['miou']['support_rule_swapped']:.2f} | "
              f"own class kept {s['normal_own']:.3f} -> {s['swap_own']:.3f}, relabelled by position {s['swap_position']:.3f}, "
              f"background {s['swap_bg']:.3f}", flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out if args.max_episodes is None else args.out.replace(".json", "_smoke.json"), "w") as f:
        json.dump(out, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
