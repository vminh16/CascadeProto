"""D-43: the density-invariant encoder (M1) against the clean base CR, each checkpoint with its own features.

  test       model / U / U + both / U + both + LP (P7's frozen arm) on fixed100, random600 seeds 0-2, leak-free
  condition  P8's part A (valid) and arm B (seed 3) on M1
  decide     rules D43.1-D43.5

    python experiments/d43_eval.py test --data_path datasets/S3DIS/blocks_bs1_s1 --checkpoint cr:ours:1:<CR last.pt> \
        --checkpoint m1:ours:1:<M1 last.pt>
    python experiments/d43_eval.py condition --data_path datasets/S3DIS/blocks_bs1_s1 --checkpoint m1:ours:1:<M1 last.pt>
    python experiments/d43_eval.py decide
"""

import argparse
import json
import os
import sys
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
from experiments import p8_condition_probe as p8  # noqa: E402
from experiments import r2_distill_eval as r2  # noqa: E402

OUT_DIR = "results/phase16_d43"
DRAWS = p6.TEST_DRAWS  # fixed100, random600:0-2, leakfree
ARMS = ("model", "U", "both", "lp")
P7_FIXED = "results/phase16_p7/test_S1_fixed100.json"
P8_INTERVENE = "results/phase16_p8/intervene_S1.json"
MECHANISM_SHARE = 0.5  # D43.1: at most half of CR's deficit remains
HONEST_GAIN, STANDARD_GAIN = 1.0, 1.0  # D43.2, D43.3
CR_REFS = {"model": ("D-37's CR", p6.D37_FIXED, "cr"), "U": ("D-39's cr:base", p7.D39_FIXED, "cr:base"),
           "both": ("D-39's cr:both", p7.D39_FIXED, "cr:both"), "lp": ("P7's lp", P7_FIXED, "lp")}


def stack_predictions(f_q: torch.Tensor, f_s: torch.Tensor, support_y: torch.Tensor, xyz: torch.Tensor,
                      arm: Dict) -> Dict[str, torch.Tensor]:
    """Predictions [B_q, P] of U, U + both and U + both + LP (one propagation arm) on one episode; no label read."""
    n_cls = support_y.shape[0] + 1
    u_logits = p6.rule_logits(f_q, p6.base_rows(f_q, f_s, support_y))  # [B_q, P, N+1]
    seed = p7.both_logits(f_q, f_s, support_y).argmax(dim=-1)  # [B_q, P]
    u = F.normalize(f_q.to(torch.float64), dim=-1)  # [B_q, P, D]
    idx, a = p7.knn_graph(xyz.to(torch.float64), u, arm["graph"], arm["k"])  # [B_q, P, k]
    s = p7.normalized(p7.affinity(idx, a, xyz.shape[1]))  # [B_q, P, P]
    lp = p7.spread(s, F.one_hot(seed, n_cls).to(torch.float64), arm["beta"]).argmax(dim=-1)  # [B_q, P]
    return {"U": u_logits.argmax(dim=-1), "both": seed, "lp": lp}


def mechanism(cr: Dict, m1: Dict) -> Tuple[bool, str]:
    """D43.1 on P8's arm-B summaries under U: both deficits at most MECHANISM_SHARE of CR's."""
    other = lambda s: s["r_own"] - s["r_v0"]  # noqa: E731, other-condition deficit
    uniform = lambda s: s["r_a_v0"] - s["r_a_v2"]  # noqa: E731, own class under uniform sampling
    ok = other(m1) <= MECHANISM_SHARE * other(cr) and uniform(m1) <= MECHANISM_SHARE * uniform(cr)
    return ok, (f"other deficit CR {other(cr):.3f} -> M1 {other(m1):.3f}; uniform drop CR {uniform(cr):.3f} -> "
                f"M1 {uniform(m1):.3f} (limit {MECHANISM_SHARE:g} of CR)")


def decide(draws: Dict, cr_int: Optional[Dict], m1_int: Optional[Dict], m1_split: Optional[Dict]) -> List[Tuple[str, str]]:
    """Rules D43.1-D43.5 of [DECISION D-43] on the `last` checkpoints cr and m1, stack U + both + LP."""
    missing = [d for d in DRAWS if d not in draws]
    if missing or m1_int is None or cr_int is None:
        return [("incomplete", f"missing draws {missing}, intervention cr {cr_int is not None} m1 {m1_int is not None}")]
    v = []
    ok1, text = mechanism(cr_int["U"], m1_int["U"])
    v.append((f"D43.1 mechanism {'holds' if ok1 else 'fails'}", text))
    lf = draws["leakfree"][1]
    p_lf = p0.paired_bootstrap(lf["cr:lp"], lf["m1:lp"])
    ok2 = p_lf["gain"] >= HONEST_GAIN and p_lf["ci_low"] > 0
    v.append((f"D43.2 leak-free {'holds' if ok2 else 'fails'}", f"m1 - cr (U + both + LP) {r2.ci_text(p_lf)}"))
    p_fx, rand = d39.paired(draws, "cr:lp", "m1:lp")
    ok3 = d39.holds(p_fx, rand, STANDARD_GAIN)
    text = f"m1 - cr fixed100 {r2.ci_text(p_fx)}; random600 {[round(g, 2) for g in rand]}"
    if not ok1:
        v.append(("D43.4 stop: density still enters; P9 locates it", text))
    elif ok3:
        v.append(("D43.3 M1 is the base on both protocols", text))
    elif ok2 and p_fx["gain"] < 0:
        v.append(("D43.3 M1 is the base for the leak-free protocol; protocol of the main table to the maintainer", text))
    else:
        v.append(("D43.3 CR stays the base", text))
    m = draws["fixed100"][0]["miou"]
    lfm = draws["leakfree"][0]["miou"]
    for n in ("cr", "m1"):
        v.append((f"D43.5 {n}", " | ".join(f"{a} {100 * m[f'{n}:{a}']:.2f}" for a in ARMS)
                  + " | leak-free " + " ".join(f"{a} {100 * lfm[f'{n}:{a}']:.2f}" for a in ARMS)))
    s = m1_int["U"]
    v.append(("D43.5 m1 arm B", f"R_V0 {s['r_v0']:.3f} R_V1 {s['r_v1']:.3f} R_own {s['r_own']:.3f} phi {s['phi']:.3f}; "
                                f"a V0 / V2 {s['r_a_v0']:.3f} / {s['r_a_v2']:.3f}"))
    if m1_split is not None:
        sm = m1_split["miou"]
        v.append(("D43.5 m1 condition oracles (valid)",
                  f"U {100 * sm['U']:.2f}, g_own {100 * (sm['oracle_own'] - sm['U']):+.2f}, "
                  f"g_other {100 * (sm['oracle_other'] - sm['U']):+.2f}, g_fg {100 * (sm['oracle_fg'] - sm['U']):+.2f}"))
    return v


@torch.no_grad()
def score(rules: Dict[str, object], draw: str, data_path: str, device, arm: Dict,
          max_episodes: Optional[int]) -> Tuple[Dict, Dict[str, np.ndarray]]:
    from pipeline.episodes import make_episode, read_class_names

    names = read_class_names(data_path, "s3dis")
    items, test_classes = p6.episodes(draw, data_path, max_episodes)
    counts: Dict[str, List[np.ndarray]] = {}
    preds_model: Dict[str, List[np.ndarray]] = {n: [] for n in rules}
    gts, l2c, skipped = [], [], 0
    for item in items:
        if (item[1].reshape(item[1].shape[0], -1).sum(1) == 0).any():
            skipped += 1  # leak-free draw only
            continue
        e = make_episode(item, names).to(device)
        gt = e.query_y.cpu().numpy()
        for name, rule in rules.items():
            f_q, m_eff, _, logits = rule(e)
            p0.check_identity(f_q, m_eff, logits)
            preds = stack_predictions(f_q, p5.support_features(rule, e), e.support_y, e.query_x[..., :3], arm)
            preds["model"] = logits.argmax(dim=-1)
            for a in ARMS:
                counts.setdefault(f"{name}:{a}", []).append(
                    p0.episode_counts(preds[a].cpu().numpy(), gt, e.sampled_classes, test_classes))
            preds_model[name].append(preds["model"].cpu().numpy())
        gts.append(gt), l2c.append(e.sampled_classes)
    stacked = {k: np.stack(v) for k, v in counts.items()}
    miou = {k: float(p0.miou_from_counts(v.sum(0))) for k, v in stacked.items()}
    if draw == "fixed100":
        from pipeline.evaluation import accumulated_miou

        class _Quiet:
            def cprint(self, text):
                pass

        for name in rules:
            if abs(accumulated_miou(_Quiet(), preds_model[name], gts, l2c, test_classes) - miou[f"{name}:model"]) > 1e-9:
                raise RuntimeError(f"{name}: the count-based mIoU differs from VIP-Seg's evaluate_metric")
        if "cr" in rules and max_episodes is None:
            for a, (what, path, key) in CR_REFS.items():
                with open(os.path.join(REPO, path)) as f:
                    p7.check_ref(f"cr:{a}", miou, json.load(f)["miou"][key], what)
    return {"draw": draw, "episodes": len(gts), "skipped": skipped, "test_classes": test_classes, "miou": miou,
            "class_iou": {k: p0.class_ious(v.sum(0)).tolist() for k, v in stacked.items()},
            "recall": {k: (v.sum(0)[2] / np.maximum(v.sum(0)[0], 1)).tolist() for k, v in stacked.items()},
            "precision": {k: (v.sum(0)[2] / np.maximum(v.sum(0)[1], 1)).tolist() for k, v in stacked.items()}}, stacked


def load_rules(specs: List[str], device) -> Tuple[Dict[str, object], Dict]:
    rules, meta = {}, {}
    for spec in specs:
        rule, protocol, config, ck = p6.load_rule(spec, device)  # the clean head [DECISION D-38]
        rules[ck.name] = rule
        meta[ck.name] = {"checkpoint": vars(ck), "protocol": protocol, "config": config}
    return rules, meta


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["test", "condition", "decide"])
    p.add_argument("--data_path")
    p.add_argument("--checkpoint", action="append", default=[])
    p.add_argument("--max_episodes", type=int, default=None, help="smoke runs only")
    p.add_argument("--tag", default="")
    p.add_argument("--out_dir", default=OUT_DIR)
    args = p.parse_args(argv)
    if args.stage == "decide":
        draws = {}
        for d in DRAWS:
            path = os.path.join(args.out_dir, p6.stem(d, args.tag) + ".json")
            if os.path.isfile(path):
                with open(path) as f:
                    draws[d] = (json.load(f), dict(np.load(path.replace(".json", "_counts.npz"))))

        def read(path):
            if os.path.isfile(path):
                with open(path) as f:
                    return json.load(f)
            return None

        cr_int = read(os.path.join(REPO, P8_INTERVENE))
        m1_int = read(os.path.join(args.out_dir, f"intervene_m1{args.tag}.json"))
        m1_split = read(os.path.join(args.out_dir, f"split_m1_valid{args.tag}.json"))
        for rule_name, text in decide(draws, cr_int, m1_int, m1_split):
            print(f"{rule_name:60s} {text}")
        return 0
    if not args.data_path or not args.checkpoint:
        p.error("test and condition need --data_path and the checkpoints")
    device = torch.device("cuda")
    rules, meta = load_rules(args.checkpoint, device)
    os.makedirs(args.out_dir, exist_ok=True)
    try:
        if args.stage == "test":
            arm = p7.arm_of(json.load(open(os.path.join(REPO, P7_FIXED)))["frozen"])  # P7's frozen arm
            for draw in DRAWS:
                result, stacked = score(rules, draw, args.data_path, device, arm, args.max_episodes)
                result.update(models=meta, lp_arm=arm)
                p6.save(result, stacked, p6.stem(draw, args.tag), args.out_dir)
                print(f"[test] {draw}: " + " | ".join(f"{k} {100 * v:.2f}" for k, v in result["miou"].items())
                      + f" | {result['episodes']} episodes", flush=True)
        else:
            for name, rule in rules.items():
                split, arrays = p8.score_split(rule, "valid", args.data_path, device, args.max_episodes,
                                               check_refs=False)
                split.update(models=meta[name])
                p6.save(split, arrays, f"split_{name}_valid{args.tag}", args.out_dir)
                interv = p8.score_intervention(rule, args.data_path, device, args.max_episodes)
                interv.update(models=meta[name])
                p6.save(interv, None, f"intervene_{name}{args.tag}", args.out_dir)
                s = interv["U"]
                print(f"[condition] {name}: U {100 * split['miou']['U']:.2f} oracle_own "
                      f"{100 * split['miou']['oracle_own']:.2f} oracle_other {100 * split['miou']['oracle_other']:.2f}"
                      f" | arm B R_V0 {s.get('r_v0', float('nan')):.3f} R_V1 {s.get('r_v1', float('nan')):.3f} "
                      f"R_own {s.get('r_own', float('nan')):.3f}", flush=True)
    finally:
        for rule in rules.values():
            rule.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
