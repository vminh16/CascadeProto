"""P10.1 of [DECISION D-46]: is there a base-class prototype gap left on the training episodes? No training.

    python experiments/p10_align_probe.py base --data_path datasets/S3DIS/blocks_bs1_s1 \
        --checkpoint cr:ours:1:log_d37/s3dis_S1_N2_K1_point_T4_vip_clean_b1_qrandom/last.pt
    python experiments/p10_align_probe.py gate --name cr      # P10.1: exit 0 holds (part B runs), 3 fails
    python experiments/p10_align_probe.py decide              # D46.1-D46.4 after part B's trainings and tests

On P1's seeded training episodes (no augmentation, the fold's base classes scored), U's rule, the cosine oracle
(the query's own unit class directions in every present row) and the model; the gap oracle − U with its paired
bootstrap, and the mean prototype-alignment loss of `models/proto_align.py` at τ = 0.1. A small gap means CE already
aligns the base-class prototypes, and an alignment loss would have nothing to act on (D-29's reading).
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from experiments import d39_eval as d39  # noqa: E402
from experiments import p0_em_probe as p0  # noqa: E402
from experiments import p1_bpc_probe as p1  # noqa: E402
from experiments import p5_condition_probe as p5  # noqa: E402
from experiments import p6_prototype_probe as p6  # noqa: E402
from models.proto_align import TAU, alignment_loss  # noqa: E402

OUT_DIR = "results/phase16_d46"
EPISODES = 1000  # P1's bank size of training episodes
GAP_BAR = 5.0  # P10.1: oracle − U on the base classes, points [DECISION D-46]
RULES = ("model", "U", "oracle")
ARMS = ("m5a", "m5b")  # λ 0.25, 1.0 [DECISION D-46]
CR_VALID_GAP = 24.88  # CR's cosine oracle − U on valid, 80.84 − 55.96 (P9, results/phase16_p9/SUMMARY.md)
MECH_DROP = 2.0  # D46.1: the arm's valid gap must shrink by at least this much [DECISION D-46]
BASE_GAIN = 1.0  # D46.2


def gap_holds(gap_points: float) -> bool:
    """[DECISION D-46] P10.1: part B runs iff the base-class gap reaches the bar."""
    return gap_points >= GAP_BAR


def decide(parts: Dict[str, Optional[Dict]], draws: Dict) -> List[Tuple[str, str]]:
    """D46.1-D46.4 [DECISION D-46]. parts: P9 part B's result per arm on valid (None if missing)."""
    v, mech = [], {}
    for arm in ARMS:
        part = parts.get(arm)
        if part is None:  # a missing result is never a verdict
            return v + [("incomplete", f"{arm}: no part B result")]
        gap = 100.0 * (part["miou"]["cos_oracle"] - part["miou"]["U"])
        mech[arm] = gap <= CR_VALID_GAP - MECH_DROP
        v.append((f"D46.1 {arm} mechanism {'holds' if mech[arm] else 'fails'}",
                  f"valid gap {gap:.2f} (CR {CR_VALID_GAP:.2f}, bar {CR_VALID_GAP - MECH_DROP:.2f}); U "
                  f"{100 * part['miou']['U']:.2f}, cosine oracle {100 * part['miou']['cos_oracle']:.2f}"))
    if not any(mech.values()):
        return v + [("D46.4 stop: base-class alignment does not transfer to novel classes at these weights", "")]
    missing = [d for d in d39.DRAWS if d not in draws]
    if missing:
        return v + [("incomplete", f"missing draws {missing}")]
    base = []
    for arm in ARMS:
        if not mech[arm]:
            continue
        p_fx, rand = d39.paired(draws, "cr:U", f"{arm}:U")
        ok = d39.holds(p_fx, rand, BASE_GAIN)
        v.append((f"D46.2 {arm} U {'holds' if ok else 'fails'} at +{BASE_GAIN:g}",
                  f"{arm} - cr fixed100 {p_fx['gain']:+.2f} [{p_fx['ci_low']:+.2f}, {p_fx['ci_high']:+.2f}]; "
                  f"random600 {[round(g, 2) for g in rand]}"))
        if ok:
            base.append(arm)
    if base:
        best = max(base, key=lambda a: parts[a]["miou"]["U"])
        v.append(("D46.2 base", f"{best} (valid U {100 * parts[best]['miou']['U']:.2f})"))
    else:
        v.append(("D46.3 prototypes align on valid without a score gain: the next decision reads part B", ""))
    return v


@torch.no_grad()
def score_base(rule, data_path: str, device, n: int, max_episodes: Optional[int]) -> Tuple[Dict, Dict]:
    from pipeline.episodes import make_episode, read_class_names

    names = read_class_names(data_path, "s3dis")
    ds = p1.training_episodes(data_path, 1, n)
    base = sorted(int(c) for c in np.asarray(ds.classes))
    counts: Dict[str, List[np.ndarray]] = {r: [] for r in RULES}
    losses = []
    m = len(ds) if max_episodes is None else min(max_episodes, len(ds))
    for i in range(m):
        e = make_episode(ds[i], names).to(device)
        f_q, m_eff, _, logits = rule(e)
        p0.check_identity(f_q, m_eff, logits)  # the rule reads the tensors the model scores with
        f_s = p5.support_features(rule, e)  # [N, K, P, D]
        rows = p6.base_rows(f_q, f_s, e.support_y)  # [B_q, N+1, D]
        preds = {"model": logits.argmax(-1), "U": p6.rule_logits(f_q, rows).argmax(-1),
                 "oracle": p6.rule_logits(f_q, p6.oracle_rows(f_q, e.query_y, rows, "all")).argmax(-1)}
        gt = e.query_y.cpu().numpy()
        for r in RULES:
            counts[r].append(p0.episode_counts(preds[r].cpu().numpy(), gt, e.sampled_classes, base))
        losses.append(float(alignment_loss(f_s, e.support_y, f_q, e.query_y, TAU)))
    stacked = {k: np.stack(v) for k, v in counts.items()}
    miou = {k: float(p0.miou_from_counts(v.sum(0))) for k, v in stacked.items()}
    paired = p0.paired_bootstrap(stacked["U"], stacked["oracle"])
    return {"episodes": m, "base_classes": base, "miou": miou, "gap": paired,
            "align_loss_mean": float(np.mean(losses)), "align_tau": TAU,
            "class_iou": {k: p0.class_ious(v.sum(0)).tolist() for k, v in stacked.items()}}, stacked


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["base", "gate", "decide"])
    p.add_argument("--data_path")
    p.add_argument("--checkpoint", help="name:ours:1:path of a clean-base checkpoint")
    p.add_argument("--name", default="cr", help="gate: the checkpoint name")
    p.add_argument("--episodes", type=int, default=EPISODES)
    p.add_argument("--max_episodes", type=int, default=None, help="smoke runs only")
    p.add_argument("--tag", default="")
    p.add_argument("--out_dir", default=OUT_DIR)
    args = p.parse_args(argv)
    out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(REPO, args.out_dir)
    if args.stage == "decide":
        parts = {}
        for a in ARMS:
            path = os.path.join(out_dir, f"modules_cr_valid_{a}{args.tag}.json")
            parts[a] = json.load(open(path)) if os.path.isfile(path) else None
        draws = {}
        for d in d39.DRAWS:
            path = os.path.join(out_dir, p6.stem(d, args.tag) + ".json")
            if os.path.isfile(path):
                draws[d] = (json.load(open(path)), dict(np.load(path.replace(".json", "_counts.npz"))))
        for name, text in decide(parts, draws):
            print(f"{name:72s} {text}")
        return 0
    if args.stage == "gate":
        with open(os.path.join(out_dir, f"p10_base_{args.name}{args.tag}.json")) as f:
            res = json.load(f)
        g = res["gap"]
        verdict = "holds: part B runs" if gap_holds(g["gain"]) else "fails: part B does not run"
        print(f"P10.1 {verdict}  base-class gap {g['gain']:+.2f} [{g['ci_low']:+.2f}, {g['ci_high']:+.2f}] "
              f"(bar {GAP_BAR:g}); U {100 * res['miou']['U']:.2f}, oracle {100 * res['miou']['oracle']:.2f}, "
              f"model {100 * res['miou']['model']:.2f}; align loss {res['align_loss_mean']:.3f}")
        return 0 if gap_holds(g["gain"]) else 3
    if not (args.data_path and args.checkpoint):
        p.error("--data_path and --checkpoint are required")
    device = torch.device("cuda")
    rule, _, _, ck = p6.load_rule(args.checkpoint, device)
    res, stacked = score_base(rule, args.data_path, device, args.episodes, args.max_episodes)
    res["checkpoint"] = vars(ck)
    p6.save(res, stacked, f"p10_base_{ck.name}{args.tag}", out_dir)
    print(f"[p10] {ck.name}: base U {100 * res['miou']['U']:.2f} | oracle {100 * res['miou']['oracle']:.2f} | "
          f"model {100 * res['miou']['model']:.2f} | gap {res['gap']['gain']:+.2f} | align loss "
          f"{res['align_loss_mean']:.3f} | {res['episodes']} episodes", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
