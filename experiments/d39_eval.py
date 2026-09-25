"""D-39: trained self-support prototypes (A0, A1) against the clean base CR, and the combined background rule.

Every checkpoint is scored on fixed100, random600 seeds 0-2 and P5's leak-free draw (seed 4) with its own
prediction and the inference background rules of [DECISION D-39], on its rows (CR: U's rows, as in P6):
  model    the checkpoint's logits
  base     F^q Rᵀ on the rows (CR: U; A0: equal to the model, checked)
  ssp_bg   P6's frozen background self-support (rho 1, alpha 0.25, T 2)
  km3      background logit = max(row, 3 spherical k-means directions of the support background)
  both     ssp_bg, then km3 on its rows
  oracle   P6's presence-fair oracle on U's rows (the model's features), a bound

    python experiments/d39_eval.py test --data_path datasets/S3DIS/blocks_bs1_s1 --checkpoint cr:ours:1:<CR last.pt> \
        --checkpoint a0:ours:1:<A0 last.pt> --checkpoint a1:ours:1:<A1 last.pt>
    python experiments/d39_eval.py decide
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

from experiments import p0_em_probe as p0  # noqa: E402
from experiments import p5_condition_probe as p5  # noqa: E402
from experiments import p6_prototype_probe as p6  # noqa: E402
from experiments import r2_distill_eval as r2  # noqa: E402

OUT_DIR = "results/phase16_d39"
DRAWS = p6.TEST_DRAWS  # fixed100, random600:0-2, leakfree
ARMS = ("model", "base", "ssp_bg", "km3", "both", "oracle")
SSP_BG = dict(rho=1.0, alpha=0.25, steps=2, which="bg")  # P6's frozen arm [DECISION D-38]
KM = 3
ROWS_TOL = 1e-4  # A0's rows against U's
BG_GAIN, SSP_GAIN, STOP_GAIN, BASE_GAIN = 0.5, 1.0, 0.5, 1.0  # D39.1, D39.2, D39.3


def km_logits(f_q: torch.Tensor, rows: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    """F^q Rᵀ with the background logit = max over the background row and the centroid directions [B_q, P, N+1]."""
    logits = p6.rule_logits(f_q, rows)  # [B_q, P, N+1]
    extra = torch.einsum("bpd,kd->bpk", f_q, centroids).max(dim=-1).values  # [B_q, P]
    return torch.cat([torch.maximum(logits[..., :1], extra.unsqueeze(-1)), logits[..., 1:]], dim=-1)


def arm_logits(f_q: torch.Tensor, f_s: torch.Tensor, support_y: torch.Tensor, rows: torch.Tensor,
               labels: torch.Tensor) -> Dict[str, torch.Tensor]:
    """The inference rules of [DECISION D-39] on one episode (`model` is added by the caller)."""
    cent = p6.spherical_kmeans(p6.background_units(f_s, support_y), KM)  # [3, D]
    adapted = p6.self_support(f_q, rows, SSP_BG["rho"], SSP_BG["alpha"], SSP_BG["steps"], SSP_BG["which"])
    u_rows = p6.base_rows(f_q, f_s, support_y)  # [B_q, N+1, D]
    return {"base": p6.rule_logits(f_q, rows), "ssp_bg": p6.rule_logits(f_q, adapted),
            "km3": km_logits(f_q, rows, cent), "both": km_logits(f_q, adapted, cent),
            "oracle": p6.rule_logits(f_q, p6.oracle_rows(f_q, labels, u_rows, "all"))}


def holds(p: Dict[str, float], rand: List[float], gain: float) -> bool:
    return p["gain"] >= gain and p["ci_low"] > 0 and min(rand) > 0


def paired(draws: Dict, a: str, b: str) -> Tuple[Dict[str, float], List[float]]:
    """b - a: paired bootstrap on fixed100 and the three random600 differences, in points."""
    fixed = draws["fixed100"][1]
    rand = [100.0 * (draws[d][0]["miou"][b] - draws[d][0]["miou"][a]) for d in DRAWS[1:4]]
    return p0.paired_bootstrap(fixed[a], fixed[b]), rand


def decide(draws: Dict, alphas: Dict[str, Dict[str, float]]) -> List[Tuple[str, str]]:
    """Rules D39.1-D39.4 on the `last` checkpoints cr, a0, a1."""
    missing = [d for d in DRAWS if d not in draws]
    if missing:
        return [("incomplete", f"missing draws {missing}")]
    m = draws["fixed100"][0]["miou"]
    if any(f"{n}:model" not in m for n in ("cr", "a0", "a1")):
        return [("incomplete", "cr, a0 and a1 must all be scored")]
    v = []
    # D39.1 background combination on CR (U rows)
    single = max(("ssp_bg", "km3"), key=lambda a: m[f"cr:{a}"])
    p, rand = paired(draws, f"cr:{single}", "cr:both")
    bg_rule = "both" if holds(p, rand, BG_GAIN) else single
    v.append((f"D39.1 background -> {bg_rule}", f"cr both - cr {single} fixed100 {r2.ci_text(p)}; random600 "
                                               f"{[round(g, 2) for g in rand]}"))
    # D39.2 trained self-support
    p, rand = paired(draws, "a0:model", "a1:model")
    ssp = "go" if holds(p, rand, SSP_GAIN) else "stop" if p["gain"] < STOP_GAIN else "between"
    v.append((f"D39.2 self-support {ssp}", f"a1 - a0 fixed100 {r2.ci_text(p)}; random600 {[round(g, 2) for g in rand]}"))
    # D39.3 new base, candidate fixed by D39.1 and D39.2
    cand = f"{'a1' if ssp == 'go' else 'a0'}:{bg_rule}"
    p, rand = paired(draws, "cr:model", cand)
    v.append((f"D39.3 {'new base ' + cand if holds(p, rand, BASE_GAIN) else 'CR stays (+ ' + bg_rule + ')'}",
              f"{cand} - cr:model fixed100 {r2.ci_text(p)}; random600 {[round(g, 2) for g in rand]}"))
    # D39.4 reported
    lf = draws["leakfree"][0]["miou"]
    for n in ("cr", "a0", "a1"):
        v.append((f"D39.4 {n}", " | ".join(f"{a} {100 * m[f'{n}:{a}']:.2f}" for a in ARMS)
                  + f" | leak-free model {100 * lf[f'{n}:model']:.2f}, {bg_rule} {100 * lf[f'{n}:{bg_rule}']:.2f}, "
                    f"oracle {100 * lf[f'{n}:oracle']:.2f}"))
    if "a1" in alphas:
        v.append(("D39.4 learned alpha", f"a1 alpha_bg {alphas['a1']['bg']:.3f}, alpha_fg {alphas['a1']['fg']:.3f}"))
    return v


@torch.no_grad()
def score(rules: Dict[str, Tuple[object, str]], draw: str, data_path: str, device,
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
        for name, (rule, source) in rules.items():
            f_q, m_eff, _, logits = rule(e)
            p0.check_identity(f_q, m_eff, logits)
            f_s = p5.support_features(rule, e)
            u_rows = p6.base_rows(f_q, f_s, e.support_y)  # [B_q, N+1, D]
            if source == "unit" and rule.model.self_support is None and (m_eff - u_rows).abs().max().item() > ROWS_TOL:
                raise RuntimeError(f"{name}: A0's rows differ from U's")
            rows = u_rows if source == "U" else m_eff  # [B_q, N+1, D]
            out = arm_logits(f_q, f_s, e.support_y, rows, e.query_y)
            out["model"] = logits
            for arm in ARMS:
                pred = out[arm].argmax(dim=-1).cpu().numpy()
                counts.setdefault(f"{name}:{arm}", []).append(p0.episode_counts(pred, gt, e.sampled_classes, test_classes))
            preds_model[name].append(out["model"].argmax(dim=-1).cpu().numpy())
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
            with open(os.path.join(REPO, p6.D37_FIXED)) as f:
                ref = json.load(f)["miou"]["cr"]
            if abs(ref - miou["cr:model"]) > p6.CR_TOL:
                raise RuntimeError(f"cr scores {miou['cr:model']:.6f} on fixed100, D-37's CR {ref:.6f}")
    return {"draw": draw, "episodes": len(gts), "skipped": skipped, "test_classes": test_classes, "miou": miou,
            "class_iou": {k: p0.class_ious(v.sum(0)).tolist() for k, v in stacked.items()},
            "recall": {k: (v.sum(0)[2] / np.maximum(v.sum(0)[0], 1)).tolist() for k, v in stacked.items()},
            "precision": {k: (v.sum(0)[2] / np.maximum(v.sum(0)[1], 1)).tolist() for k, v in stacked.items()}}, stacked


def load_rules(specs: List[str], device):
    rules, meta, alphas = {}, {}, {}
    for spec in specs:
        ck = p0.parse_checkpoint(spec)
        if ck.fold != 1 or ck.kind != "ours":
            raise ValueError(f"{ck.name}: D-39 reads our S1 checkpoints [DECISION D-39]")
        rule, protocol, config = r2.load(ck, 1, device)  # eval.py's protocol guard [DECISION D-22]
        if config.get("prototype_rule", "mean") == "unit":
            source = "unit"  # the model's own rows
        elif config["stage_type"] == "vip_clean":
            source = "U"  # CR: P6's rules act on U's rows
        else:
            raise ValueError(f"{ck.name}: D-39 reads the clean base (vip_clean) or the unit rule [DECISION D-39]")
        ss = getattr(rule.model, "self_support", None)
        if ss is not None:
            a = ss.alphas(2)
            alphas[ck.name] = {"bg": float(a[0]), "fg": float(a[1])}
        rules[ck.name] = (rule, source)
        meta[ck.name] = {"checkpoint": vars(ck), "protocol": protocol, "config": config, "rows": source}
    return rules, meta, alphas


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["test", "decide"])
    p.add_argument("--data_path")
    p.add_argument("--checkpoint", action="append", default=[])
    p.add_argument("--max_episodes", type=int, default=None, help="smoke runs only")
    p.add_argument("--tag", default="")
    p.add_argument("--out_dir", default=OUT_DIR)
    args = p.parse_args(argv)
    if args.stage == "decide":
        draws, alphas = {}, {}
        for d in DRAWS:
            path = os.path.join(args.out_dir, p6.stem(d, args.tag) + ".json")
            if os.path.isfile(path):
                with open(path) as f:
                    r = json.load(f)
                alphas = r.get("alphas", alphas)
                draws[d] = (r, dict(np.load(path.replace(".json", "_counts.npz"))))
        for rule_name, text in decide(draws, alphas):
            print(f"{rule_name:34s} {text}")
        return 0
    if not args.data_path or not args.checkpoint:
        p.error("test needs --data_path and the checkpoints")
    device = torch.device("cuda")
    rules, meta, alphas = load_rules(args.checkpoint, device)
    os.makedirs(args.out_dir, exist_ok=True)
    try:
        for draw in DRAWS:
            result, stacked = score(rules, draw, args.data_path, device, args.max_episodes)
            result.update(models=meta, alphas=alphas)
            stem = p6.stem(draw, args.tag)
            with open(os.path.join(args.out_dir, stem + ".json"), "w") as f:
                json.dump(result, f, indent=1)
            np.savez_compressed(os.path.join(args.out_dir, stem + "_counts.npz"), **stacked)
            print(f"[test] {draw}: " + " | ".join(f"{k} {100 * v:.2f}" for k, v in result["miou"].items()
                                                  if k.endswith((":model", ":both", ":oracle")))
                  + f" | {result['episodes']} episodes", flush=True)
    finally:
        for rule, _ in rules.values():
            rule.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
