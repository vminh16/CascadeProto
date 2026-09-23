"""P2 of [DECISION D-28]: query-side EM whose foreground M-step is filtered by the base margin, no training.

    python experiments/p2_fused_probe.py select --data_path datasets/S3DIS/blocks_bs1_s1 \
        --checkpoint vipseg_S1:vipseg:1:vipseg_S1_N2_K1.pt --checkpoint ours_S1:ours:1:log_s1/.../last.pt
    python experiments/p2_fused_probe.py test --data_path ... --checkpoint ...              # frozen arm
    python experiments/p2_fused_probe.py decide                                             # rules P2.0-P2.5

Question: P0 (D-26) failed because the foreground M-step averages the points the model already assigns
to a class, false ones included, so the prototype drifts toward its own errors. P1 (D-27) found a signal
independent of the model's posterior, the base margin, too weak to decide single points (AUC 0.65-0.72
on VIP-Seg). Removing a few true points from a mean over hundreds costs variance only, removing false
ones reduces its bias: does the base margin, used as a filter on the M-step, clean the prototypes (the
false share epsilon of the M-step mass falls) and turn P0's refinement into a gain?

Protocol, fixed before any run [DECISION D-28]:
* banks: P1's, `results/phase16_p1/bank_<name>_1000.pt` (built from each checkpoint's own training fold);
* `select` scores the grid on the S1 **valid** draw and freezes the arm with the best gain on the
  **VIP-Seg** S1 checkpoint (route B builds on VIP-Seg's head, D-25); our S1 checkpoint is scored for
  the report;
* `test` scores the frozen arm, the same arm without the filter, and the model, once on fixed100;
* `decide` applies the rules of D-28.
"""

import argparse
import glob
import itertools
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from experiments import p0_em_probe as p0  # noqa: E402
from experiments import p1_bpc_probe as p1  # noqa: E402
from models.base_calibration import base_margin, support_prototypes, top_fraction_flips  # noqa: E402
from models.transductive import em_step, responsibilities  # noqa: E402

OUT_DIR = "results/phase16_p2"
# weight: SSP won P0's selection, entropy is D-26's own; kappa as P0 without 16, which lost everywhere;
# T <= 2, P0's best arms were T = 1-2; r: the filtered fraction of foreground-assigned points, around
# the false share P1 measured (VIP-Seg 10-18 %, ours 19-27 %), r = 0 is P0 itself [DECISION D-28].
WEIGHTS = ("ssp", "entropy")
KAPPAS = (0.5, 1.0, 2.0, 4.0, 8.0)
MAX_STEPS = 2
FILTERS = (0.0, 0.1, 0.2, 0.3)


def arm_name(a: Dict) -> str:
    return f"{a['weight']}_k{a['kappa']:g}_T{a['steps']}_r{a['r']:g}"


def grid_arms() -> List[Dict]:
    return [dict(weight=w, kappa=k, steps=t, r=r)
            for w, k, r in itertools.product(WEIGHTS, KAPPAS, FILTERS) for t in range(1, MAX_STEPS + 1)]


def fg_keep_mask(f_q, logits, support, base, mu, r: float) -> torch.Tensor:
    """Points kept in the foreground M-step: all but the top fraction r (positive margins) [B_q, P]."""
    if r == 0:
        return torch.ones(logits.shape[:2], dtype=torch.bool, device=logits.device)  # [B_q, P]
    return ~top_fraction_flips(base_margin(f_q, logits, support, base, mu), r)  # [B_q, P]


def false_share_terms(wr: torch.Tensor, labels: torch.Tensor) -> Tuple[float, float]:
    """(false mass, total mass) of the foreground M-step: wr [B_q, P, N+1], labels [B_q, P]."""
    fg = wr[..., 1:]  # [B_q, P, N]
    cls = torch.arange(1, wr.shape[-1], device=wr.device)  # [N]
    wrong = (labels.unsqueeze(-1) != cls).to(wr.dtype)  # [B_q, P, N]
    return float((fg * wrong).sum()), float(fg.sum())


@torch.no_grad()
def run_arms(f_q, prior, logits0, support, base, mu, arms, labels) -> Tuple[Dict, Dict]:
    """Predictions of every arm, and the first-step false mass of every (weight, r)."""
    preds = {"model": logits0.argmax(dim=-1)}  # [B_q, P]
    eps: Dict[str, List[float]] = {}
    wanted = {arm_name(a) for a in arms}
    for weight, kappa, r in sorted({(a["weight"], a["kappa"], a["r"]) for a in arms}):
        logits = logits0  # [B_q, P, N+1]
        for t in range(1, MAX_STEPS + 1):
            keep = fg_keep_mask(f_q, logits, support, base, mu, r)  # [B_q, P]
            if t == 1:
                key = f"{weight}_r{r:g}"
                if key not in eps:
                    wr = responsibilities(logits, weight)  # [B_q, P, N+1]
                    wr = torch.cat([wr[..., :1], wr[..., 1:] * keep.unsqueeze(-1).to(wr.dtype)], dim=-1)
                    eps[key] = list(false_share_terms(wr, labels))
            _, logits = em_step(f_q, prior, logits, kappa, weight, fg_keep=keep)  # [B_q, P, N+1]
            name = arm_name(dict(weight=weight, kappa=kappa, steps=t, r=r))
            if name in wanted:
                preds[name] = logits.argmax(dim=-1)  # [B_q, P]
    return preds, eps


@torch.no_grad()
def score_checkpoint(ck, data_path: str, cvfold: int, mode: str, arms, device, max_episodes=None,
                     bank_episodes: int = p1.BANK_EPISODES):
    from torch.utils.data import DataLoader, Subset

    from pipeline.episodes import EpisodeCollate, build_eval_dataset, read_class_names
    from pipeline.evaluation import accumulated_miou

    rule, protocol = p0.load_rule(ck, cvfold, device)
    mu, base, bank_info = p1.load_or_build_bank(ck, rule, data_path, device, bank_episodes)  # P1's bank
    mu, base = mu.to(device), base.to(device)  # [D], [J, D]
    names = read_class_names(data_path, "s3dis")
    dataset = build_eval_dataset(data_path, "s3dis", cvfold, 2, 1, mode=mode, seed=0)
    test_classes = [int(c) for c in np.asarray(dataset.classes)]
    if set(test_classes) & set(bank_info["classes"]):
        raise RuntimeError("base prototypes of a scored class: the bank must hold training classes only")
    view = dataset if max_episodes is None else Subset(dataset, range(min(max_episodes, len(dataset))))
    loader = DataLoader(view, batch_size=1, shuffle=False, collate_fn=EpisodeCollate(names))
    counts: Dict[str, List[np.ndarray]] = {}
    eps_sum: Dict[str, List[float]] = {}
    base_preds, gts, l2c = [], [], []
    for (episode,) in loader:
        episode = episode.to(device)
        f_q, prior, logits0 = rule(episode)  # [B_q,P,D], [B_q,N+1,D], [B_q,P,N+1]
        p0.check_identity(f_q, prior, logits0)
        labels = episode.query_y  # [B_q, P], for the counts and the false-share diagnostic only
        support = support_prototypes(rule.f_s, episode.support_y, mu)  # [N, D]
        preds, eps = run_arms(f_q, prior, logits0, support, base, mu, arms, labels)
        for k, (wrong, total) in eps.items():
            acc = eps_sum.setdefault(k, [0.0, 0.0])
            acc[0] += wrong
            acc[1] += total
        gt = labels.cpu().numpy()
        for name, pred in preds.items():
            counts.setdefault(name, []).append(p0.episode_counts(pred.cpu().numpy(), gt, episode.sampled_classes,
                                                                 test_classes))
        base_preds.append(preds["model"].cpu().numpy()), gts.append(gt), l2c.append(episode.sampled_classes)
    rule.close()

    class _Quiet:
        def cprint(self, text):
            pass

    stacked = {k: np.stack(v) for k, v in counts.items()}  # name -> [E, 3, C+1]
    primary = accumulated_miou(_Quiet(), base_preds, gts, l2c, test_classes)  # VIP-Seg's metric [D-08]
    if abs(primary - p0.miou_from_counts(stacked["model"].sum(0))) > 1e-9:
        raise RuntimeError("count-based mIoU differs from VIP-Seg's evaluate_metric")
    return {
        "checkpoint": vars(ck), "cvfold": cvfold, "mode": mode, "protocol": protocol,
        "episodes": len(view), "test_classes": test_classes,
        "miou": {k: float(p0.miou_from_counts(v.sum(0))) for k, v in stacked.items()},
        "class_iou": {k: p0.class_ious(v.sum(0)).tolist() for k, v in stacked.items()},
        "false_share": {k: w / max(t, 1e-12) for k, (w, t) in eps_sum.items()},
    }, stacked


def save(result: Dict, stacked: Dict[str, np.ndarray], stem: str) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, stem + ".json"), "w") as f:
        json.dump(result, f, indent=1)
    np.savez_compressed(os.path.join(OUT_DIR, stem + "_counts.npz"), **stacked)


def select_arm(results: List[Dict]) -> Dict:
    """The arm with the best gain on the VIP-Seg S1 checkpoint (route B, D-25)."""
    vip = [r for r in results if r["checkpoint"]["kind"] == "vipseg"]
    if len(vip) != 1:
        raise ValueError(f"selection needs exactly one VIP-Seg S1 checkpoint, got {len(vip)}")
    m = vip[0]["miou"]
    gains = {arm_name(a): 100.0 * (m[arm_name(a)] - m["model"]) for a in grid_arms()}
    best = max(grid_arms(), key=lambda a: gains[arm_name(a)])
    return {"frozen": best, "name": arm_name(best), "valid_gain": gains[arm_name(best)],
            "all_gains": dict(sorted(gains.items(), key=lambda kv: -kv[1]))}


def test_arms(frozen: Dict) -> List[Dict]:
    return [frozen, dict(frozen, r=0.0)]


def cmd_select(args, device) -> int:
    results = []
    for spec in args.checkpoint:
        ck = p0.parse_checkpoint(spec)
        if ck.fold != 1:
            raise ValueError(f"{ck.name}: selection uses S1 checkpoints only, S0 stays held out [DECISION D-22]")
        result, stacked = score_checkpoint(ck, args.data_path, 1, "valid", grid_arms(), device, args.max_episodes,
                                           args.bank_episodes)
        save(result, stacked, f"select_{ck.name}")
        results.append(result)
        fs = result["false_share"]
        shares = " ".join("%.3f" % fs["ssp_r%g" % r] for r in FILTERS)
        print(f"[select] {ck.name}: model {result['miou']['model']:.4f} | false share ssp r0/.1/.2/.3 {shares}",
              flush=True)
    choice = select_arm(results)
    with open(os.path.join(OUT_DIR, "selection.json"), "w") as f:
        json.dump(choice, f, indent=1)
    print(f"[select] frozen: {choice['name']} (VIP-Seg S1 valid gain {choice['valid_gain']:+.2f})", flush=True)
    return 0


def cmd_test(args, device) -> int:
    with open(os.path.join(OUT_DIR, "selection.json")) as f:
        frozen = json.load(f)["frozen"]
    arms = test_arms(frozen)
    for spec in args.checkpoint:
        ck = p0.parse_checkpoint(spec)
        result, stacked = score_checkpoint(ck, args.data_path, ck.fold, "test", arms, device, args.max_episodes,
                                           args.bank_episodes)
        name, unfiltered = arm_name(arms[0]), arm_name(arms[1])
        result["frozen"] = name
        result["paired"] = {"frozen_vs_model": p0.paired_bootstrap(stacked["model"], stacked[name]),
                            "frozen_vs_unfiltered": p0.paired_bootstrap(stacked[unfiltered], stacked[name])}
        save(result, stacked, f"test_{ck.name}")
        p = result["paired"]["frozen_vs_model"]
        before = result["false_share"]["%s_r0" % frozen["weight"]]
        after = result["false_share"]["%s_r%g" % (frozen["weight"], frozen["r"])]
        print(f"[test] {ck.name} S{ck.fold}: model {result['miou']['model']:.4f} -> {result['miou'][name]:.4f} "
              f"({p['gain']:+.2f}, 95% CI {p['ci_low']:+.2f}..{p['ci_high']:+.2f}) | false share "
              f"{before:.3f} -> {after:.3f}", flush=True)
    return 0


def decide(tests: List[Dict]) -> List[Tuple[str, str]]:
    """Rules P2.0-P2.5 of [DECISION D-28], applied to the test JSONs."""
    vip = {t["cvfold"]: t for t in tests if t["checkpoint"]["kind"] == "vipseg"}
    if set(vip) != {0, 1}:
        return [("incomplete", f"need the VIP-Seg S1 and S0 test results, have folds {sorted(vip)}")]
    frozen = vip[1]["frozen"]
    weight, r = frozen.split("_")[0], float(frozen.split("_r")[-1])
    verdicts = []
    ratios = [t["false_share"][f"{weight}_r{r:g}"] / max(t["false_share"][f"{weight}_r0"], 1e-12)
              for t in vip.values()]
    if r == 0:
        verdicts.append(("P2.0 mechanism", "selection froze an unfiltered arm: the filter adds nothing"))
    elif all(x <= 0.75 for x in ratios):
        verdicts.append(("P2.0 mechanism",
                         f"false share x{[round(x, 2) for x in ratios]} (<= 0.75): the filter cleans the M-step"))
    else:
        verdicts.append(("P2.0 mechanism", f"false share x{[round(x, 2) for x in ratios]}: the filter does not "
                                           "clean the M-step enough (> 0.75); stop whatever the mIoU"))
    ci = {f: t["paired"]["frozen_vs_model"] for f, t in vip.items()}
    mech_ok = r > 0 and all(x <= 0.75 for x in ratios)
    if mech_ok and all(c["gain"] >= 0.5 and c["ci_low"] > 0 for c in ci.values()):
        verdicts.append(("P2.1 go",
                         "VIP-Seg S1 and S0 gain >= +0.5 with CI above 0: train the fused refinement (R2)"))
    elif ci[1]["ci_low"] <= 0 or not mech_ok:
        verdicts.append(("P2.2 stop", "VIP-Seg S1 does not gain with CI above 0, or the mechanism fails"))
    else:
        verdicts.append(("P2.3 in between", "S1 gains, S0 does not reach the go bar"))
    for t in tests:
        c = t["paired"]["frozen_vs_model"]
        verdicts.append((f"P2 gain {t['checkpoint']['name']}",
                         f"{c['gain']:+.2f} [{c['ci_low']:+.2f}, {c['ci_high']:+.2f}]: {p1.ci_side(c)}"))
    u = vip[0]["paired"]["frozen_vs_unfiltered"]
    verdicts.append(("P2.4 filter on held-out S0", f"{u['gain']:+.2f} [{u['ci_low']:+.2f}, {u['ci_high']:+.2f}]: "
                     + ("claimable" if u["ci_low"] > 0 else "not claimable")))
    for t in tests:
        ious = t["class_iou"]
        drop = [round(100 * (b - a), 2) for a, b in zip(ious["model"][1:], ious[t["frozen"]][1:])]
        if t["paired"]["frozen_vs_model"]["gain"] > 0 and min(drop) < -3.0:
            verdicts.append((f"P2.5 collapse watch {t['checkpoint']['name']}", f"per-class change {drop}"))
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
    p.add_argument("--bank_episodes", type=int, default=p1.BANK_EPISODES, help="smaller only for smoke runs")
    args = p.parse_args(argv)
    if args.stage == "decide":
        return cmd_decide(args)
    if not args.data_path or not args.checkpoint:
        p.error("select and test need --data_path and at least one --checkpoint")
    device = torch.device("cuda")
    return cmd_select(args, device) if args.stage == "select" else cmd_test(args, device)


if __name__ == "__main__":
    raise SystemExit(main())
