"""P1 of [DECISION D-27]: base-class calibration of the background on trained checkpoints, no training.

    python experiments/p1_bpc_probe.py select --data_path datasets/S3DIS/blocks_bs1_s1 \
        --checkpoint vipseg_S1:vipseg:1:vipseg_S1_N2_K1.pt --checkpoint ours_S1:ours:1:log_s1/.../last.pt
    python experiments/p1_bpc_probe.py test --data_path ... --checkpoint ...              # frozen q
    python experiments/p1_bpc_probe.py decide                                             # rules P1.1-P1.5

Question: a query point that resembles a base (training) class is background in a novel-class episode
[COSeg §4.3]. Does moving to the background the foreground predictions that lie most clearly nearer to a
base class than to their own support class, in CL2N geometry (models/base_calibration.py), raise the
mIoU, and does that base margin separate the model's false foreground from its true foreground at all?

Protocol, fixed before any run [DECISION D-27], the same as P0 [DECISION D-26]:
* the bank of each checkpoint is built from 1,000 seeded training episodes of **its own fold**
  (training classes only, no augmentation), with the checkpoint's frozen features, in two passes (the
  CL2N centre, then the base prototypes);
* `select` scores the flip fraction q on the S1 **valid** draw of the S1 checkpoints only and freezes
  the best mean gain;
* `test` scores the frozen q once on fixed100: S1 checkpoints on S1, S0 checkpoints on S0; paired
  bootstrap over episodes;
* `decide` applies the rules of D-27.
Every checkpoint passes eval.py's protocol guard for the episodes it is scored on.
"""

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from experiments import p0_em_probe as p0  # noqa: E402
from models.base_calibration import (BasePrototypeBank, auc_from_histogram, base_margin,  # noqa: E402
                                     calibrate_background, separability_histogram, support_prototypes,
                                     top_fraction_flips)

OUT_DIR = "results/phase16_p1"
# Fraction of a query's foreground predictions that may be moved to the background, around the share of
# false foreground measured in the second smoke run (3.7 % of VIP-Seg's S1 foreground predictions), where
# absolute margin thresholds flipped 33-73 % [DECISION D-27].
FRACTIONS = (0.005, 0.01, 0.02, 0.04)
BANK_EPISODES = 1000  # >= 5x COSeg's EMA memory, 1 / (1 - 0.995) = 200 updates [COSeg Eq.10, T6]
MIN_OCCURRENCES = 100


def arm(q: float) -> str:
    return f"bpc_q{q:g}"


# ------------------------------------------------------------------ bank

def training_episodes(data_path: str, cvfold: int, n: int):
    """Seeded training episodes of the fold's base classes, without augmentation (04 §4)."""
    from dataloaders.loader import MyDataset
    from pipeline.episodes import (N_QUERIES, NUM_POINT, PC_ATTRIBS, RANDOM_SAMPLE, WAY_NUM, WAY_RATIO,
                                   SeededEpisodes)

    ds = MyDataset(data_path, "s3dis", cvfold=cvfold, num_episode=n, n_way=2, k_shot=1, n_queries=N_QUERIES,
                   phase=None, mode="train", num_point=NUM_POINT, pc_attribs=PC_ATTRIBS, pc_augm=False,
                   way_ratio=WAY_RATIO, way_num=WAY_NUM, random_sample=RANDOM_SAMPLE)
    return SeededEpisodes(ds, seed=0)


@torch.no_grad()
def build_bank(rule, data_path: str, fold: int, device, n: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """CL2N centre [D], base prototypes [J, D] from the checkpoint's own training episodes, and bank facts."""
    from torch.utils.data import DataLoader

    from pipeline.episodes import EpisodeCollate, read_class_names

    episodes = training_episodes(data_path, fold, n)
    names = read_class_names(data_path, "s3dis")
    bank = None
    for phase in ("centre", "prototypes"):  # SeededEpisodes makes both passes see the same episodes
        loader = DataLoader(episodes, batch_size=1, shuffle=False, collate_fn=EpisodeCollate(names))
        for (episode,) in loader:
            episode = episode.to(device)
            f_q, prior, logits = rule(episode)  # [B_q,P,D], [B_q,N+1,D], [B_q,P,N+1]
            p0.check_identity(f_q, prior, logits)
            bank = bank or BasePrototypeBank(f_q.shape[-1])
            if phase == "centre":
                bank.add_centre(rule.f_s, f_q)
            else:
                bank.add_episode(rule.f_s, episode.support_y, f_q, episode.query_y, episode.sampled_classes)
        if phase == "centre":
            bank.freeze_centre()
    protos = bank.prototypes(min_count=MIN_OCCURRENCES if n >= BANK_EPISODES else 1)  # smoke runs: any
    base_classes = sorted(int(c) for c in np.asarray(episodes.classes))
    if sorted(protos) != base_classes:
        raise RuntimeError(f"bank classes {sorted(protos)} differ from the fold's training classes {base_classes}")
    base = torch.stack([protos[c] for c in base_classes]).float()  # [J, D]
    return bank.mu.float(), base, {"classes": base_classes, "episodes": n,
                                   "occurrences": {str(c): bank.counts[c] for c in base_classes},
                                   "mean_raw_cos_to_centre": bank.common_cos_sum / max(bank.common_cos_count, 1)}


def load_or_build_bank(ck, rule, data_path, device, n) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    path = os.path.join(OUT_DIR, f"bank_{ck.name}_{n}.pt")  # a smoke bank never stands in for the full one
    if os.path.isfile(path):
        saved = torch.load(path, weights_only=True)
        return saved["mu"], saved["base"], json.loads(saved["info"])
    mu, base, info = build_bank(rule, data_path, ck.fold, device, n)
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.save({"mu": mu.cpu(), "base": base.cpu(), "info": json.dumps(info)}, path)
    return mu, base, info


# ------------------------------------------------------------------ scoring

@torch.no_grad()
def score_checkpoint(ck, data_path: str, cvfold: int, mode: str, fractions, device, max_episodes=None,
                     bank_episodes: int = BANK_EPISODES) -> Tuple[Dict, Dict[str, np.ndarray]]:
    from torch.utils.data import DataLoader, Subset

    from pipeline.episodes import EpisodeCollate, build_eval_dataset, read_class_names
    from pipeline.evaluation import accumulated_miou

    rule, protocol = p0.load_rule(ck, cvfold, device)
    mu, base, bank_info = load_or_build_bank(ck, rule, data_path, device, bank_episodes)
    mu, base = mu.to(device), base.to(device)  # [D], [J, D]
    names = read_class_names(data_path, "s3dis")
    dataset = build_eval_dataset(data_path, "s3dis", cvfold, 2, 1, mode=mode, seed=0)
    test_classes = [int(c) for c in np.asarray(dataset.classes)]
    if set(test_classes) & set(bank_info["classes"]):
        raise RuntimeError("base prototypes of a scored class: the bank must hold training classes only")
    view = dataset if max_episodes is None else Subset(dataset, range(min(max_episodes, len(dataset))))
    loader = DataLoader(view, batch_size=1, shuffle=False, collate_fn=EpisodeCollate(names))
    counts: Dict[str, List[np.ndarray]] = {}
    base_preds, gts, l2c = [], [], []
    hist_fp_tp = torch.zeros(2, 200, dtype=torch.long)
    flips = {arm(q): 0 for q in fractions}
    fg_total, positive_margin = 0, 0
    for (episode,) in loader:
        episode = episode.to(device)
        f_q, prior, logits0 = rule(episode)  # [B_q,P,D], [B_q,N+1,D], [B_q,P,N+1]
        p0.check_identity(f_q, prior, logits0)
        labels = episode.query_y  # [B_q, P]
        preds = {"model": logits0.argmax(dim=-1)}  # [B_q, P]
        support = support_prototypes(rule.f_s, episode.support_y, mu)  # [N, D]
        margin = base_margin(f_q, logits0, support, base, mu)  # [B_q, P]
        for q in fractions:
            flip = top_fraction_flips(margin, q)  # [B_q, P]
            preds[arm(q)] = calibrate_background(logits0, flip).argmax(dim=-1)  # [B_q, P]
            flips[arm(q)] += int(flip.sum())
        gt = labels.cpu().numpy()
        for name, pred in preds.items():
            counts.setdefault(name, []).append(p0.episode_counts(pred.cpu().numpy(), gt, episode.sampled_classes,
                                                                 test_classes))
        base_preds.append(preds["model"].cpu().numpy()), gts.append(gt), l2c.append(episode.sampled_classes)
        fg_pred = preds["model"] > 0  # [B_q, P]
        fg_total += int(fg_pred.sum())
        positive_margin += int((margin > 0).sum())
        hist_fp_tp += separability_histogram(margin, fg_pred & (labels == 0), fg_pred & (labels > 0)).cpu()
    rule.close()

    class _Quiet:
        def cprint(self, text):
            pass

    stacked = {k: np.stack(v) for k, v in counts.items()}  # name -> [E, 3, C+1]
    primary = accumulated_miou(_Quiet(), base_preds, gts, l2c, test_classes)  # VIP-Seg's metric [D-08]
    if abs(primary - p0.miou_from_counts(stacked["model"].sum(0))) > 1e-9:
        raise RuntimeError("count-based mIoU differs from VIP-Seg's evaluate_metric")
    fp, tp = int(hist_fp_tp[0].sum()), int(hist_fp_tp[1].sum())
    return {
        "checkpoint": vars(ck), "cvfold": cvfold, "mode": mode, "protocol": protocol,
        "episodes": len(view), "test_classes": test_classes, "bank": bank_info,
        "miou": {k: float(p0.miou_from_counts(v.sum(0))) for k, v in stacked.items()},
        "class_iou": {k: p0.class_ious(v.sum(0)).tolist() for k, v in stacked.items()},
        "separability": {"auc_false_fg_vs_true_fg": auc_from_histogram(hist_fp_tp),
                         "false_fraction_of_fg_predictions": fp / max(fp + tp, 1)},
        "flipped_fraction_of_fg_predictions": {k: v / max(fg_total, 1) for k, v in flips.items()},
        "positive_margin_fraction_of_fg_predictions": positive_margin / max(fg_total, 1),
    }, stacked


# ------------------------------------------------------------------ stages

def select_fraction(results: List[Dict]) -> Dict:
    gains = {q: float(np.mean([100.0 * (r["miou"][arm(q)] - r["miou"]["model"]) for r in results]))
             for q in FRACTIONS}
    best = max(FRACTIONS, key=lambda q: gains[q])
    return {"fraction": best, "name": arm(best), "mean_valid_gain": gains[best],
            "all_gains": {arm(q): g for q, g in gains.items()}}


def cmd_select(args, device) -> int:
    results = []
    for spec in args.checkpoint:
        ck = p0.parse_checkpoint(spec)
        if ck.fold != 1:
            raise ValueError(f"{ck.name}: selection uses S1 checkpoints only, S0 stays held out [DECISION D-22]")
        result, stacked = score_checkpoint(ck, args.data_path, 1, "valid", FRACTIONS, device, args.max_episodes,
                                           args.bank_episodes)
        save(result, stacked, f"select_{ck.name}")
        results.append(result)
        sep = result["separability"]
        print(f"[select] {ck.name}: model {result['miou']['model']:.4f} | "
              + " ".join(f"{arm(q)} {result['miou'][arm(q)]:.4f}" for q in FRACTIONS)
              + f" | AUC false-vs-true fg {sep['auc_false_fg_vs_true_fg']}"
              + f" | false share of fg {sep['false_fraction_of_fg_predictions']:.3f}"
              + f" | raw cos to centre {result['bank']['mean_raw_cos_to_centre']:.3f}", flush=True)
    choice = select_fraction(results)
    with open(os.path.join(OUT_DIR, "selection.json"), "w") as f:
        json.dump(choice, f, indent=1)
    print(f"[select] frozen: {choice['name']} (mean valid gain {choice['mean_valid_gain']:+.2f})", flush=True)
    return 0


def cmd_test(args, device) -> int:
    with open(os.path.join(OUT_DIR, "selection.json")) as f:
        q = json.load(f)["fraction"]
    for spec in args.checkpoint:
        ck = p0.parse_checkpoint(spec)
        result, stacked = score_checkpoint(ck, args.data_path, ck.fold, "test", [q], device, args.max_episodes,
                                           args.bank_episodes)
        result["frozen"] = arm(q)
        result["paired"] = {"frozen_vs_model": p0.paired_bootstrap(stacked["model"], stacked[arm(q)])}
        save(result, stacked, f"test_{ck.name}")
        p = result["paired"]["frozen_vs_model"]
        print(f"[test] {ck.name} S{ck.fold}: model {result['miou']['model']:.4f} -> "
              f"{result['miou'][arm(q)]:.4f} ({p['gain']:+.2f}, 95% CI {p['ci_low']:+.2f}..{p['ci_high']:+.2f})"
              f" | AUC false-vs-true fg {result['separability']['auc_false_fg_vs_true_fg']}", flush=True)
    return 0


def save(result: Dict, stacked: Dict[str, np.ndarray], stem: str) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, stem + ".json"), "w") as f:
        json.dump(result, f, indent=1)
    np.savez_compressed(os.path.join(OUT_DIR, stem + "_counts.npz"), **stacked)


def ci_side(ci: Dict) -> str:
    return "above 0" if ci["ci_low"] > 0 else "below 0" if ci["ci_high"] < 0 else "contains 0"


def decide(tests: List[Dict]) -> List[Tuple[str, str]]:
    """Rules P1.1-P1.5 of [DECISION D-27], applied to the test JSONs."""
    s1 = [t for t in tests if t["cvfold"] == 1]
    s0 = [t for t in tests if t["cvfold"] == 0]
    if len(s1) < 2 or len(s0) < 2:
        return [("incomplete", f"need two S1 and two S0 test results, have {len(s1)} and {len(s0)}")]
    ci = {t["checkpoint"]["name"]: t["paired"]["frozen_vs_model"] for t in tests}
    verdicts = []
    if all(c["gain"] >= 0.5 and c["ci_low"] > 0 for c in ci.values()):
        verdicts.append(("P1.1 go", "gain >= +0.5 with CI above 0 on all four: train the calibration (R3)"))
    elif all(t["paired"]["frozen_vs_model"]["ci_low"] <= 0 for t in s1):
        verdicts.append(("P1.2 stop (training-free form)", "no S1 checkpoint gains with CI above 0"))
    else:
        verdicts.append(("P1.3 in between", "some but not all checkpoints gain: R3 only with P1.4's signal"))
    for name, c in ci.items():
        verdicts.append((f"P1.1 gain {name}",
                         f"{c['gain']:+.2f} [{c['ci_low']:+.2f}, {c['ci_high']:+.2f}]: {ci_side(c)}"))
    aucs = [t["separability"]["auc_false_fg_vs_true_fg"] for t in tests if t["checkpoint"]["kind"] == "vipseg"]
    if aucs and all(a is not None and a >= 0.70 for a in aucs):
        verdicts.append(("P1.4 base signal", f"AUC {aucs} >= 0.70 on VIP-Seg: the base margin flags false "
                                              "foreground; a trained calibration (COSeg Eq.12) is justified"))
    elif aucs and all(a is not None and a < 0.60 for a in aucs):
        verdicts.append(("P1.4 base signal", f"AUC {aucs} < 0.60 on VIP-Seg: no usable signal, drop D-27"))
    else:
        verdicts.append(("P1.4 base signal", f"AUC {aucs}: weak, report only"))
    for t in tests:
        frozen = t["frozen"]
        drop = [round(100 * (b - a), 2) for a, b in zip(t["class_iou"]["model"][1:], t["class_iou"][frozen][1:])]
        if ci[t["checkpoint"]["name"]]["gain"] > 0 and min(drop) < -3.0:
            verdicts.append((f"P1.5 collapse watch {t['checkpoint']['name']}", f"per-class change {drop}"))
    return verdicts


def cmd_decide(args) -> int:
    tests = []
    for path in sorted(glob.glob(os.path.join(OUT_DIR, "test_*.json"))):
        with open(path) as f:
            tests.append(json.load(f))
    for rule, text in decide(tests):
        print(f"{rule:40s} {text}")
    return 0 if tests else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["select", "test", "decide"])
    p.add_argument("--data_path")
    p.add_argument("--checkpoint", action="append", default=[], help="name:kind:trained_fold:path")
    p.add_argument("--max_episodes", type=int, default=None, help="smoke runs only")
    p.add_argument("--bank_episodes", type=int, default=BANK_EPISODES, help="smaller only for smoke runs")
    args = p.parse_args(argv)
    if args.stage == "decide":
        return cmd_decide(args)
    if not args.data_path or not args.checkpoint:
        p.error("select and test need --data_path and at least one --checkpoint")
    device = torch.device("cuda")
    return cmd_select(args, device) if args.stage == "select" else cmd_test(args, device)


if __name__ == "__main__":
    raise SystemExit(main())
