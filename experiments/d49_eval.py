"""D-49 and D-48 arm A: one reader for every head (CR's PEM/PDM, condition-balanced CB, the correlation head A).

    python experiments/d49_eval.py mu --data_path datasets/S3DIS/blocks_bs1_s1 --checkpoint cb:ours:1:<last.pt>
    python experiments/d49_eval.py test --data_path ... --checkpoint cr:ours:1:<CR last.pt> [--checkpoint ...]
    python experiments/d49_eval.py decide49 --cr cr --arm cb      # rules D49.1-D49.5 [DECISION D-49]
    python experiments/d49_eval.py decideA --cr cr --arm a0 --arm a1   # D48.2' [DECISION D-48 amendments 3, 6]

Rows per checkpoint: the model's logits, U and U + both on its features, and LP on the model's and on U + both's
predictions, with LP re-selected on the checkpoint's own valid over P7's 24 arms (D-48 amendment 1, change 6).
Draws: valid (selection), fixed100, random600 seeds 0-2, leak-free, and valid_raw, where P11.6's oracle abundance of
other-condition points is read in the checkpoint's centred space (mu from its base-class training episodes).
Every checkpoint passes eval.py's protocol guard [DECISION D-22].
"""

import argparse
import json
import os
import sys
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from experiments import d39_eval as d39  # noqa: E402
from experiments import p0_em_probe as p0  # noqa: E402
from experiments import p5_condition_probe as p5  # noqa: E402
from experiments import p6_prototype_probe as p6  # noqa: E402
from experiments import p7_propagation_probe as p7  # noqa: E402
from experiments import p11_precheck as p11  # noqa: E402

OUT_DIR = "results/phase16_d49"
MU_PER_PAIR = 14  # 15 base pairs x 14 = 210 training episodes for the centring mean
ROWS = ("model", "U", "both", "model+lp", "both+lp")
DRAWS = ("valid",) + p6.TEST_DRAWS + ("valid_raw",)
ALPHA_K = 16  # P11.6's context size of the oracle check
# [DECISION D-49]
ALPHA_BAR, RECALL_GAIN, LEAKFREE_GAIN, STANDARD_GAIN = 0.10, 0.05, 1.0, 1.0
# [DECISION D-48] D48.2'
NECK_GAIN = 1.0


def load(spec: str, device):
    import eval as eval_cli

    ck = p0.parse_checkpoint(spec)
    if ck.kind != "ours" or ck.fold != 1:
        raise ValueError(f"{ck.name}: D-49 reads our S1 checkpoints")
    args = SimpleNamespace(checkpoint=ck.path, model="cascadeproto", cvfold=1, checkpoint_cvfold=1,
                           allow_seen_classes=False)
    model = eval_cli.load_model(args, device)
    protocol = eval_cli.protocol_check(args, model.checkpoint_args)  # raises on seen classes [D-22]
    model.eval()
    return model, ck, protocol


@torch.no_grad()
def base_mean(model, data_path: str, device, max_episodes: Optional[int]) -> torch.Tensor:
    """Mean unit feature of the base-labelled points of seeded training episodes (P11's centring)."""
    from dataloaders.s3dis import S3DISDataset
    from pipeline.episodes import make_episode, read_class_names

    names = read_class_names(data_path, "s3dis")
    base = torch.tensor(sorted(int(c) for c in S3DISDataset(1, data_path).train_classes), device=device)
    total, count = None, 0
    for ep in p11.raw_episodes(data_path, "train", p11.TRAIN_SEED, MU_PER_PAIR, max_episodes):
        e = make_episode(ep["item"], names).to(device)
        f_s, f_q = model.features.encode_episode(e.support_x, e.query_x)
        for u, raw in ((F.normalize(f_q, dim=-1), ep["query_raw"]), (F.normalize(f_s[:, 0], dim=-1), ep["support_raw"][:, 0])):
            sel = torch.isin(torch.from_numpy(raw).to(device), base)  # [B, P]
            s = u[sel].double().sum(0)
            total = s if total is None else total + s
            count += int(sel.sum())
    return (total / max(count, 1)).float()


def lp_seeds(model, e, f_q, f_s) -> Dict[str, torch.Tensor]:
    return {"model": model(e).logits.argmax(-1), "U": p6.rule_logits(f_q, p6.base_rows(f_q, f_s, e.support_y)).argmax(-1),
            "both": p7.both_logits(f_q, f_s, e.support_y).argmax(-1)}


@torch.no_grad()
def select_lp(model, data_path: str, device, max_episodes: Optional[int]) -> Dict[str, str]:
    """The P7 arm with the highest valid mIoU for each seed (model, both); ties to the earlier arm."""
    from pipeline.episodes import make_episode, read_class_names

    names = read_class_names(data_path, "s3dis")
    items, test_classes = p6.episodes("valid", data_path, max_episodes)
    arms = p7.lp_arms()
    counts = {s: {p7.lp_name(a): [] for a in arms} for s in ("model", "both")}
    for item in items:
        e = make_episode(item, names).to(device)
        f_s, f_q = model.features.encode_episode(e.support_x, e.query_x)
        seeds = lp_seeds(model, e, f_q, f_s)
        gt = e.query_y.cpu().numpy()
        for a in arms:
            out = p11.spread_many(e.query_x[..., :3], F.normalize(f_q, dim=-1), [seeds["model"], seeds["both"]],
                                  e.n_way + 1, a)
            for s, pr in zip(("model", "both"), out):
                counts[s][p7.lp_name(a)].append(p0.episode_counts(pr.cpu().numpy(), gt, e.sampled_classes, test_classes))
    frozen = {}
    for s in ("model", "both"):
        miou = {k: float(p0.miou_from_counts(np.stack(v).sum(0))) for k, v in counts[s].items()}
        frozen[s] = max((p7.lp_name(a) for a in arms), key=lambda k: miou[k])
    return frozen


@torch.no_grad()
def score(model, name: str, draw: str, data_path: str, device, frozen: Dict[str, str], mu: Optional[torch.Tensor],
          max_episodes: Optional[int]) -> Tuple[Dict, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    from models import unmix as um
    from pipeline.episodes import make_episode, read_class_names

    names = read_class_names(data_path, "s3dis")
    items, test_classes = p11.draw_items(draw, data_path, max_episodes)
    counts: Dict[str, List[np.ndarray]] = {}
    cond: Dict[str, List[np.ndarray]] = {}
    alphas: Dict[str, List[torch.Tensor]] = {"own": [], "other": []}
    skipped = 0
    for it in items:
        item = it["item"]
        if (item[1].reshape(item[1].shape[0], -1).sum(1) == 0).any():
            skipped += 1
            continue
        e = make_episode(item, names).to(device)
        gt = e.query_y.cpu().numpy()
        f_s, f_q = model.features.encode_episode(e.support_x, e.query_x)
        preds = lp_seeds(model, e, f_q, f_s)
        u_q = F.normalize(f_q, dim=-1)
        for s in ("model", "both"):
            preds[f"{s}+lp"] = p11.spread_many(e.query_x[..., :3], u_q, [preds[s]], e.n_way + 1,
                                               p7.arm_of(frozen[s]))[0]
        for r in ROWS:
            pr = preds[r].cpu().numpy()
            counts.setdefault(r, []).append(p0.episode_counts(pr, gt, e.sampled_classes, test_classes))
            cond.setdefault(r, []).append(p5.condition_counts(pr, gt, e.sampled_classes, test_classes))
        if draw == "valid_raw" and mu is not None:
            v = u_q - mu
            ctx = um.context_means(v, e.query_x[..., :3], ALPHA_K)
            for b in range(gt.shape[0]):
                for c in range(1, e.n_way + 1):
                    pts = e.query_y[b] == c
                    if int(pts.sum()) < p6.MIN_POINTS:
                        continue
                    own_mean = (u_q[b][pts].mean(0) - mu).double()
                    _, al = um.mixture_fit(v[b][pts].double(), own_mean.expand(int(pts.sum()), -1), ctx[b][pts].double())
                    alphas["own" if c == b + 1 else "other"].append(al.cpu())
    stacked = {f"{name}:{k}": np.stack(v) for k, v in counts.items()}
    cstack = {f"{name}:{k}": np.stack(v) for k, v in cond.items()}
    res = {"draw": draw, "name": name, "episodes": len(counts["model"]), "skipped": skipped, "lp_frozen": frozen,
           "miou": {k: 100.0 * float(p0.miou_from_counts(v.sum(0))) for k, v in stacked.items()},
           "condition": {k: p5.split_summary(v.sum(0)) for k, v in cstack.items()}}
    if draw == "valid_raw" and mu is not None:
        res["alpha_quantiles"] = {k: [float(x) for x in np.quantile(torch.cat(v).numpy(), [0.1, 0.25, 0.5, 0.75, 0.9])]
                                  for k, v in alphas.items() if v}
    return res, stacked, cstack


# ------------------------------------------------------------------ rules

def load_draws(out_dir: str, names: List[str], tag: str) -> Dict:
    draws = {}
    for d in DRAWS:
        merged, counts, ok = {"miou": {}, "condition": {}}, {}, True
        for n in names:
            path = os.path.join(out_dir, f"{n}_{d.replace(':', '_seed')}{tag}.json")
            if not os.path.isfile(path):
                ok = False
                break
            r = json.load(open(path))
            merged["miou"].update(r["miou"]), merged["condition"].update(r["condition"])
            merged.setdefault("alpha", {})[n] = r.get("alpha_quantiles")
            counts.update(dict(np.load(path.replace(".json", "_counts.npz"))))
        if ok:
            draws[d] = (merged, counts)
    return draws


def other_recall(res: Dict, key: str) -> float:
    return float(np.nanmean(res["condition"][key]["recall_other"]))


def decide49(draws: Dict, cr: str, arm: str) -> List[Tuple[str, str]]:
    missing = [d for d in DRAWS if d not in draws]
    if missing:
        return [("incomplete", f"missing draws {missing}")]
    v = []
    al = draws["valid_raw"][0]["alpha"]
    a_arm = al[arm]["other"][2] if al.get(arm) else float("nan")
    a_cr = al[cr]["other"][2] if al.get(cr) else float("nan")
    fx = draws["fixed100"][0]
    r_arm, r_cr = other_recall(fx, f"{arm}:U"), other_recall(fx, f"{cr}:U")
    mech = a_arm >= ALPHA_BAR and r_arm >= r_cr + RECALL_GAIN
    v.append((f"D49.1 mechanism {'holds' if mech else 'fails'}",
              f"median own-class abundance of other points {a_cr:+.3f} -> {a_arm:+.3f} (bar {ALPHA_BAR:g}); U other "
              f"recall {r_cr:.3f} -> {r_arm:.3f} (bar +{RECALL_GAIN:g})"))
    lf = draws["leakfree"][1]
    p_lf = p0.paired_bootstrap(lf[f"{cr}:U"], lf[f"{arm}:U"])
    ok2 = p_lf["gain"] >= LEAKFREE_GAIN and p_lf["ci_low"] > 0
    v.append((f"D49.2 leak-free {'holds' if ok2 else 'fails'}",
              f"{arm} - {cr} (U) {p_lf['gain']:+.2f} [{p_lf['ci_low']:+.2f}, {p_lf['ci_high']:+.2f}]"))
    ok3, text3 = p11.holds_at(draws, f"{cr}:U", f"{arm}:U", STANDARD_GAIN)
    fixed_gain = fx["miou"][f"{arm}:U"] - fx["miou"][f"{cr}:U"]
    if not mech:
        v.append(("D49.5 stop: condition balance does not restore minority points at this q", text3))
    elif ok3:
        v.append((f"D49.3 {arm} is the base on both protocols", text3))
    elif ok2 and fixed_gain < 0:
        v.append((f"D49.4 {arm} is the base for the leak-free protocol; the main table's protocol to the maintainer", text3))
    else:
        v.append((f"D49.3 fails: {cr} stays the base", text3))
    for n in (cr, arm):
        v.append((f"reported {n}", " | ".join(f"{r} {fx['miou'][f'{n}:{r}']:.2f}" for r in ROWS)
                  + " | leak-free " + " ".join(f"{r} {draws['leakfree'][0]['miou'][f'{n}:{r}']:.2f}" for r in ROWS)))
    return v


def decide_a(draws: Dict, cr: str, arms: List[str]) -> List[Tuple[str, str]]:
    missing = [d for d in DRAWS if d not in draws]
    if missing:
        return [("incomplete", f"missing draws {missing}")]
    v, gains = [], {}
    for a in arms:
        ok, text = p11.holds_at(draws, f"{cr}:both+lp", f"{a}:model+lp", NECK_GAIN)
        gains[a] = (ok, draws["fixed100"][0]["miou"][f"{a}:model+lp"] - draws["fixed100"][0]["miou"][f"{cr}:both+lp"])
        lf = draws["leakfree"][0]["miou"]
        v.append((f"D48.2' {a} {'holds' if ok else 'fails'}", f"{a} model+LP - {cr} U+both+LP {text}; leak-free "
                  f"{lf[f'{a}:model+lp'] - lf[f'{cr}:both+lp']:+.2f}; other recall "
                  f"{other_recall(draws['fixed100'][0], f'{cr}:both+lp'):.3f} -> "
                  f"{other_recall(draws['fixed100'][0], f'{a}:model+lp'):.3f}"))
    spread = abs(draws["fixed100"][0]["miou"][f"{arms[0]}:model+lp"] - draws["fixed100"][0]["miou"][f"{arms[-1]}:model+lp"])
    effect = min(g for _, g in gains.values())
    kept = all(ok for ok, _ in gains.values()) and spread <= effect
    v.append((f"D48.2' neck {'kept' if kept else 'not kept'}", f"seed spread {spread:.2f}, smallest effect {effect:+.2f}"))
    return v


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["mu", "test", "decide49", "decideA"])
    p.add_argument("--data_path")
    p.add_argument("--checkpoint", action="append", default=[])
    p.add_argument("--draw", action="append")
    p.add_argument("--cr", default="cr")
    p.add_argument("--arm", action="append", default=[])
    p.add_argument("--max_episodes", type=int, default=None)
    p.add_argument("--tag", default="")
    p.add_argument("--out_dir", default=OUT_DIR)
    args = p.parse_args(argv)
    out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(REPO, args.out_dir)
    if args.stage in ("decide49", "decideA"):
        draws = load_draws(out_dir, [args.cr] + args.arm, args.tag)
        rows = decide49(draws, args.cr, args.arm[0]) if args.stage == "decide49" else decide_a(draws, args.cr, args.arm)
        for name, text in rows:
            print(f"{name:72s} {text}", flush=True)
        return 0
    device = torch.device("cuda")
    os.makedirs(out_dir, exist_ok=True)
    for spec in args.checkpoint:
        model, ck, protocol = load(spec, device)
        mu_path = os.path.join(out_dir, f"mu_{ck.name}{args.tag}.pt")
        if args.stage == "mu":
            torch.save(base_mean(model, args.data_path, device, args.max_episodes).cpu(), mu_path)
            print(f"[mu] {ck.name} saved", flush=True)
            continue
        mu = torch.load(mu_path, weights_only=True).to(device) if os.path.isfile(mu_path) else None
        sel_path = os.path.join(out_dir, f"lp_{ck.name}{args.tag}.json")
        if os.path.isfile(sel_path):
            frozen = json.load(open(sel_path))
        else:
            frozen = select_lp(model, args.data_path, device, args.max_episodes)
            json.dump(frozen, open(sel_path, "w"), indent=1)
        for draw in args.draw or DRAWS:
            res, stacked, cstack = score(model, ck.name, draw, args.data_path, device, frozen, mu, args.max_episodes)
            res.update(checkpoint=vars(ck), protocol=protocol)
            stem = f"{ck.name}_{draw.replace(':', '_seed')}{args.tag}"
            p6.save(res, stacked, stem, out_dir)
            np.savez_compressed(os.path.join(out_dir, stem + "_cond.npz"), **cstack)
            print(f"[test] {ck.name} {draw}: " + " | ".join(f"{k.split(':', 1)[1]} {m:.2f}" for k, m in res["miou"].items()),
                  flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
