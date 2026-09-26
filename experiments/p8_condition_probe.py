"""P8 of [DECISION D-41]: the clean base's oracle gap split by sampling condition (own / other, D-35), and D-35's
condition intervention (arm B) re-run on it. Inference only, CR `last.pt` (D-37's base), S1; rules on U (D-38's unit
geometry).

  A  oracle_own / oracle_other / oracle_fg: U with the query's own direction in the foreground rows of the blocks
     where the class is own / other / either (presence-fair, one gauge); bounds, never results
  B  arm B of D-35 (seed 3): V0 the protocol block, V1 the same scan sampled for the other class c, V2 sampled
     uniformly; recall of c and of the block's own class a under U, phi = (R_V1 - R_V0) / (R_own - R_V0)
  C  cos(s_c, o_c) per query block and present class, by condition, with the block's recall under U

    python experiments/p8_condition_probe.py split --data_path datasets/S3DIS/blocks_bs1_s1 --checkpoint cr:ours:1:<last.pt>
    python experiments/p8_condition_probe.py intervene --data_path datasets/S3DIS/blocks_bs1_s1 --checkpoint cr:ours:1:<last.pt>
    python experiments/p8_condition_probe.py decide
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
from experiments import p7_propagation_probe as p7  # noqa: E402
from models.oracle_distill import oracle_directions  # noqa: E402

OUT_DIR = "results/phase16_p8"
P6_FIXED = "results/phase16_p6/test_S1_fixed100.json"
SPLIT_DRAWS = ("valid", "fixed100")  # decisions on valid, fixed100 reported [DECISION D-41]
ARMS = ("model", "U", "both", "oracle_own", "oracle_other", "oracle_fg")
WHICH = ("own", "other", "fg")
BOUND_GAIN = 1.0  # P8.1, P8.3: twice the single-run sd
PHI_CAUSAL, PHI_NOT = 0.5, 0.2  # D-35 P5.1a / P5.1b
REF_TOL = p6.CR_TOL
RULES = ("U", "both")  # intervention predictions; P8.2 reads U
ALIGN_FIELDS = ("class", "cond", "cos", "gt", "tp")


# ------------------------------------------------------------------ pure functions

def own_mask(n_blocks: int, n_cls: int, device=None) -> torch.Tensor:
    """[B, N+1] True where local class k is the one block b was sampled for (k = b + 1) [VIPSEG loader.py:181-222]."""
    if n_blocks != n_cls - 1:
        raise ValueError(f"P8 needs one query block per way, got {n_blocks} blocks for {n_cls - 1} ways")
    m = torch.zeros(n_blocks, n_cls, dtype=torch.bool, device=device)
    m[torch.arange(n_blocks), torch.arange(n_blocks) + 1] = True
    return m


def condition_oracle_rows(f_q: torch.Tensor, labels: torch.Tensor, rows: torch.Tensor, which: str) -> torch.Tensor:
    """rows [B_q, N+1, D] with the query's own unit direction in the present foreground rows of the blocks where the
    class is `own`, `other` or either (`fg`); absent rows and the background row untouched [DECISION D-41]."""
    if which not in WHICH:
        raise ValueError(f"which must be one of {WHICH}, got {which!r}")
    n_cls = rows.shape[1]
    o, present = oracle_directions(f_q, labels, n_cls)  # [B_q, N+1, D], [B_q, N+1]
    own = own_mask(rows.shape[0], n_cls, rows.device)  # [B_q, N+1]
    fg = torch.arange(n_cls, device=rows.device).unsqueeze(0) >= 1  # [1, N+1]
    cond = own if which == "own" else ~own if which == "other" else torch.ones_like(own)
    take = present & fg & cond  # [B_q, N+1]
    return torch.where(take.unsqueeze(-1), o, rows)  # [B_q, N+1, D]


def alignment(f_q: torch.Tensor, labels: torch.Tensor, rows: torch.Tensor, pred: torch.Tensor,
              label2class) -> np.ndarray:
    """One record per (query block, present foreground class): global class, condition (0 own, 1 other),
    cos(s_c, o_c), GT and TP of the class in the block under `pred` -> [R, 5] (ALIGN_FIELDS)."""
    n_cls = rows.shape[1]
    o, present = oracle_directions(f_q, labels, n_cls)  # [B_q, N+1, D], [B_q, N+1]
    cos = torch.einsum("bcd,bcd->bc", o, rows)  # [B_q, N+1], both unit
    own = own_mask(rows.shape[0], n_cls, rows.device)  # [B_q, N+1]
    out = []
    for b in range(rows.shape[0]):
        for k in range(1, n_cls):
            if not bool(present[b, k]):
                continue
            is_k = labels[b] == k  # [P]
            out.append([int(label2class[k - 1]), 0 if bool(own[b, k]) else 1, float(cos[b, k]),
                        float(is_k.sum()), float((is_k & (pred[b] == k)).sum())])
    return np.array(out, dtype=np.float64).reshape(-1, len(ALIGN_FIELDS))


def split_logits(f_q: torch.Tensor, f_s: torch.Tensor, support_y: torch.Tensor,
                 labels: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Logits [B_q, P, N+1] of U, U + both (D-39) and the three condition oracles on one episode."""
    rows = p6.base_rows(f_q, f_s, support_y)  # [B_q, N+1, D]
    out = {"U": p6.rule_logits(f_q, rows), "both": p7.both_logits(f_q, f_s, support_y)}
    for w in WHICH:
        out[f"oracle_{w}"] = p6.rule_logits(f_q, condition_oracle_rows(f_q, labels, rows, w))
    return out


def has_support_background(support_y: np.ndarray, min_points: int = 3) -> bool:
    """True when the support blocks hold at least `min_points` background points in total: U's background row needs
    one and the 3-component background of D-39 needs three [DECISION D-41, amended]."""
    return int((np.asarray(support_y) == 0).sum()) >= min_points


def class_counts(y: np.ndarray, pred: np.ndarray, k: int) -> Tuple[float, float]:
    """(GT, TP) of local class k in one block."""
    is_k = y == k
    return float(is_k.sum()), float((is_k & (pred == k)).sum())


def summarize_alignment(rec: np.ndarray, test_classes: List[int]) -> Dict:
    """Mean / quartiles of cos and pooled recall per class and condition, and Spearman(cos, block recall)."""
    out = {}
    for cname, cond in (("own", 0), ("other", 1)):
        for c in [None] + list(test_classes):
            sel = rec[:, 1] == cond
            if c is not None:
                sel &= rec[:, 0] == c
            if not sel.any():
                continue
            r = rec[sel]
            key = f"{'all' if c is None else c}_{cname}"
            recall = r[:, 4] / np.maximum(r[:, 3], 1.0)  # per block
            out[key] = {"blocks": int(sel.sum()), "cos_mean": float(r[:, 2].mean()),
                        "cos_q": np.percentile(r[:, 2], [25, 50, 75]).tolist(),
                        "recall_pooled": float(r[:, 4].sum() / max(r[:, 3].sum(), 1.0)),
                        "spearman_cos_recall": spearman(r[:, 2], recall)}
    return out


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Rank correlation with average ranks for ties; nan below 3 records or with a constant input."""
    if x.size < 3:
        return float("nan")

    def rank(a: np.ndarray) -> np.ndarray:
        order = np.argsort(a, kind="mergesort")
        r = np.empty(a.size)
        r[order] = np.arange(a.size, dtype=np.float64)
        for v in np.unique(a):
            tie = a == v
            r[tie] = r[tie].mean()
        return r

    rx, ry = rank(x), rank(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def decide(split: Dict[str, Dict], interv: Optional[Dict]) -> List[Tuple[str, str]]:
    """Rules P8.1-P8.5 of [DECISION D-41]."""
    if "valid" not in split:
        return [("incomplete", "the valid split is missing")]
    m = split["valid"]["miou"]
    g_own, g_other = 100.0 * (m["oracle_own"] - m["U"]), 100.0 * (m["oracle_other"] - m["U"])
    g_fg = 100.0 * (m["oracle_fg"] - m["U"])
    other_ok, own_ok = g_other >= BOUND_GAIN, g_own >= BOUND_GAIN
    v = [(f"P8.1 other-condition bound {'holds' if other_ok else 'below'}",
          f"g_other {g_other:+.2f} (valid; U {100 * m['U']:.2f})"),
         (f"P8.3 own-condition bound {'holds' if own_ok else 'below'}", f"g_own {g_own:+.2f}, g_fg {g_fg:+.2f}")]
    if interv is None or interv.get("U", {}).get("events", 0) == 0:
        return v + [("incomplete", "the intervention is missing")]
    s = interv["U"]
    phi, ci = s["phi"], s["v1_minus_v0_ci"]
    if phi != phi:
        dens = "undefined"  # no own / other gap under V0
    elif phi >= PHI_CAUSAL and ci[0] > 0:
        dens = "causal"
    elif phi < PHI_NOT:
        dens = "not density"
    else:
        dens = "partial"
    v.append((f"P8.2 density {dens}", f"phi {phi:.3f}; R_V0 {s['r_v0']:.3f}, R_V1 {s['r_v1']:.3f}, R_V2 "
                                      f"{s['r_v2']:.3f}, R_own {s['r_own']:.3f}; R_V1 - R_V0 CI "
                                      f"[{ci[0]:+.3f}, {ci[1]:+.3f}]; {s['events']} blocks"))
    arms = []
    if other_ok and dens in ("causal", "partial"):
        arms.append(("condition-balanced training", g_other))
    if own_ok:
        arms.append(("instance-alignment training", g_own))
    if arms:
        first = max(arms, key=lambda a: a[1])[0]
        v.append(("P8.4 next run: " + first, "admissible: " + ", ".join(f"{n} (bound {g:+.2f})" for n, g in arms)))
    else:
        v.append(("P8.4 next run: none", "the prototype gap is not the lever; revisit the backbone"))
    rep = f"a under V0 / V2 {s['r_a_v0']:.3f} / {s['r_a_v2']:.3f}"
    if "both" in interv and interv["both"].get("events", 0):
        rep += f"; phi under U + both {interv['both']['phi']:.3f}"
    if "fixed100" in split:
        f = split["fixed100"]["miou"]
        rep += "; fixed100 " + ", ".join(f"{a} {100 * f[a]:.2f}" for a in ARMS)
    v.append(("P8.5 reported", rep))
    for draw in SPLIT_DRAWS:
        if draw in split:
            al = split[draw]["alignment"]
            v.append((f"P8.5 cos {draw}", " | ".join(
                f"{k} cos {al[k]['cos_mean']:.3f} recall {al[k]['recall_pooled']:.3f} rho "
                f"{al[k]['spearman_cos_recall']:.2f}" for k in ("all_own", "all_other") if k in al)))
    return v


# ------------------------------------------------------------------ GPU passes

@torch.no_grad()
def score_split(rule, draw: str, data_path: str, device, max_episodes: Optional[int]) -> Tuple[Dict, Dict]:
    from pipeline.episodes import make_episode, read_class_names

    names = read_class_names(data_path, "s3dis")
    items, test_classes = p6.episodes(draw, data_path, max_episodes)
    counts: Dict[str, List[np.ndarray]] = {}
    split: Dict[str, np.ndarray] = {}
    records = []
    preds_model, gts, l2c = [], [], []
    for item in items:
        e = make_episode(item, names).to(device)
        f_q, m_eff, _, logits = rule(e)
        p0.check_identity(f_q, m_eff, logits)
        f_s = p5.support_features(rule, e)
        out = split_logits(f_q, f_s, e.support_y, e.query_y)
        preds = {k: v.argmax(dim=-1) for k, v in out.items()}
        preds["model"] = logits.argmax(dim=-1)
        rows = p6.base_rows(f_q, f_s, e.support_y)  # [B_q, N+1, D]
        changed = {w: (condition_oracle_rows(f_q, e.query_y, rows, w) != rows).any(-1) for w in WHICH}  # [B_q, N+1]
        if (changed["own"] & changed["other"]).any() or not torch.equal(changed["own"] | changed["other"],
                                                                        changed["fg"]):
            raise RuntimeError("oracle_own and oracle_other do not partition oracle_fg's rows")
        records.append(alignment(f_q, e.query_y, rows, preds["U"], e.sampled_classes))
        gt = e.query_y.cpu().numpy()
        for name, pred in preds.items():
            pr = pred.cpu().numpy()
            counts.setdefault(name, []).append(p0.episode_counts(pr, gt, e.sampled_classes, test_classes))
            split[name] = split.get(name, 0) + p5.condition_counts(pr, gt, e.sampled_classes, test_classes)
        preds_model.append(preds["model"].cpu().numpy()), gts.append(gt), l2c.append(e.sampled_classes)
    stacked = {k: np.stack(v) for k, v in counts.items()}
    miou = {k: float(p0.miou_from_counts(v.sum(0))) for k, v in stacked.items()}
    rec = np.concatenate(records)
    full = max_episodes is None
    if draw == "fixed100":
        from pipeline.evaluation import accumulated_miou

        class _Quiet:
            def cprint(self, text):
                pass

        if abs(accumulated_miou(_Quiet(), preds_model, gts, l2c, test_classes) - miou["model"]) > 1e-9:
            raise RuntimeError("the count-based mIoU differs from VIP-Seg's evaluate_metric")
        if full:
            with open(os.path.join(REPO, p6.D37_FIXED)) as f:
                p7.check_ref("model", miou, json.load(f)["miou"]["cr"], "D-37's CR")
            with open(os.path.join(REPO, p7.D39_FIXED)) as f:
                ref = json.load(f)["miou"]
            p7.check_ref("U", miou, ref["cr:base"], "D-39's cr:base")
            p7.check_ref("both", miou, ref["cr:both"], "D-39's cr:both")
            with open(os.path.join(REPO, P6_FIXED)) as f:
                p7.check_ref("oracle_fg", miou, json.load(f)["miou"]["oracle_fg"], "P6's oracle_fg")
    if draw == "valid" and full:
        with open(os.path.join(REPO, p7.P6_SELECT)) as f:
            ref = json.load(f)["miou"]
        p7.check_ref("model", miou, ref["model"], "P6's valid model")
        p7.check_ref("U", miou, ref["U"], "P6's valid U")
    return {"draw": draw, "episodes": len(gts), "test_classes": test_classes, "miou": miou,
            "class_iou": {k: p0.class_ious(v.sum(0)).tolist() for k, v in stacked.items()},
            "recall": {k: (v.sum(0)[2] / np.maximum(v.sum(0)[0], 1)).tolist() for k, v in stacked.items()},
            "precision": {k: (v.sum(0)[2] / np.maximum(v.sum(0)[1], 1)).tolist() for k, v in stacked.items()},
            "split": {k: p5.split_summary(v) for k, v in split.items()},
            "alignment": summarize_alignment(rec, test_classes)}, dict(stacked, alignment=rec)


@torch.no_grad()
def score_intervention(rule, data_path: str, device, max_episodes: Optional[int]) -> Dict:
    """D-35 arm B on the clean base under U and U + both: V0 / V1 / V2 per eligible query block (seed 3)."""
    from pipeline.episodes import make_episode, read_class_names

    names = read_class_names(data_path, "s3dis")
    seed = p5.DRAW_SEEDS["intervene"]
    own_gt = {r: {} for r in RULES}
    own_tp = {r: {} for r in RULES}
    events = {r: [] for r in RULES}
    cos_rec = []  # [class, version, cos]

    def predict(item):
        e = make_episode(item, names).to(device)
        f_q, m_eff, _, logits = rule(e)
        p0.check_identity(f_q, m_eff, logits)
        f_s = p5.support_features(rule, e)
        lg = split_logits(f_q, f_s, e.support_y, e.query_y)
        rows = p6.base_rows(f_q, f_s, e.support_y)  # [B_q, N+1, D]
        o, present = oracle_directions(f_q, e.query_y, rows.shape[1])  # [B_q, N+1, D], [B_q, N+1]
        cos = torch.einsum("bcd,bcd->bc", o, rows).where(present, torch.full_like(present, float("nan"),
                                                                                   dtype=rows.dtype))
        return {r: lg[r].argmax(-1).cpu().numpy() for r in RULES}, cos.cpu().numpy()  # [B_q, P], [B_q, N+1]

    n_ep, no_bg = 0, 0
    for ep in p5.draw_episodes(data_path, 1, seed, p5.EPISODES_PER_PAIR, False, max_episodes):
        sx, sy, qx, qy, classes = ep["item"]
        if not has_support_background(sy):
            no_bg += 1  # the background prototype is undefined for every rule [DECISION D-41, amended]
            continue
        pred0, cos0 = predict(ep["item"])
        n_ep += 1
        for b in range(qy.shape[0]):
            a = int(classes[b])
            for r in RULES:
                g, t = class_counts(qy[b], pred0[r][b], b + 1)
                own_gt[r][a] = own_gt[r].get(a, 0.0) + g
                own_tp[r][a] = own_tp[r].get(a, 0.0) + t
        for b in range(qy.shape[0]):
            a, c_local = int(classes[b]), 2 - b  # the other way of a 2-way episode
            c = int(classes[c_local - 1])
            if p5.raw_count(data_path, ep["q_scans"][b], c) < p5.MIN_RAW_POINTS:
                continue
            per = {r: {"gt_c": [], "tp_c": [], "gt_a": [], "tp_a": []} for r in RULES}
            for v, (cls, uniform) in enumerate(((None, None), (c, False), (a, True))):
                if v == 0:
                    y, preds, cos = qy[b], {r: pred0[r][b] for r in RULES}, cos0[b]
                else:
                    x, y, _ = p5.sample_block(data_path, ep["q_scans"][b], classes, cls, uniform,
                                              [seed, ep["pair"], ep["index"], 2 + v, b])
                    qx2, qy2 = qx.copy(), qy.copy()
                    qx2[b], qy2[b] = x, y
                    p2, c2 = predict((sx, sy, qx2, qy2, classes))
                    preds, cos = {r: p2[r][b] for r in RULES}, c2[b]
                for r in RULES:
                    g, t = class_counts(y, preds[r], c_local)
                    per[r]["gt_c"].append(g), per[r]["tp_c"].append(t)
                    if v != 1:
                        g, t = class_counts(y, preds[r], b + 1)
                        per[r]["gt_a"].append(g), per[r]["tp_a"].append(t)
                cos_rec.append([c, v, float(cos[c_local])])
            for r in RULES:
                events[r].append({"episode": ep["index"], "c": c, "a": a, **per[r]})
    out = {"seed": seed, "episodes": n_ep, "skipped_no_background": no_bg, "min_raw_points": p5.MIN_RAW_POINTS}
    for r in RULES:
        r_own = {k: own_tp[r][k] / own_gt[r][k] for k in own_gt[r] if own_gt[r][k] > 0}
        out[r] = p5.intervention_summary(events[r], r_own)
        out[r]["r_own_by_class"] = {int(k): float(x) for k, x in r_own.items()}
    cr = np.array(cos_rec, dtype=np.float64).reshape(-1, 3)
    out["cos_by_version"] = {f"V{v}": float(np.nanmean(cr[cr[:, 1] == v, 2])) if (cr[:, 1] == v).any() else None
                             for v in range(3)}
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["split", "intervene", "decide"])
    p.add_argument("--data_path")
    p.add_argument("--checkpoint")
    p.add_argument("--max_episodes", type=int, default=None, help="smoke runs only")
    p.add_argument("--tag", default="", help="suffix of the output files (smoke runs)")
    p.add_argument("--out_dir", default=OUT_DIR)
    args = p.parse_args(argv)
    if args.stage == "decide":
        split = {}
        for d in SPLIT_DRAWS:
            path = os.path.join(args.out_dir, f"split_S1_{d}{args.tag}.json")
            if os.path.isfile(path):
                with open(path) as f:
                    split[d] = json.load(f)
        path = os.path.join(args.out_dir, f"intervene_S1{args.tag}.json")
        interv = None
        if os.path.isfile(path):
            with open(path) as f:
                interv = json.load(f)
        for rule_name, text in decide(split, interv):
            print(f"{rule_name:44s} {text}")
        return 0
    if not args.data_path or not args.checkpoint:
        p.error("split and intervene need --data_path and --checkpoint")
    device = torch.device("cuda")
    rule, protocol, config, ck = p6.load_rule(args.checkpoint, device)  # the clean base [DECISION D-38]
    try:
        if args.stage == "split":
            for draw in SPLIT_DRAWS:
                result, arrays = score_split(rule, draw, args.data_path, device, args.max_episodes)
                result.update(checkpoint=vars(ck), protocol=protocol)
                p6.save(result, arrays, f"split_S1_{draw}{args.tag}", args.out_dir)
                print(f"[split] {draw}: " + " | ".join(f"{a} {100 * result['miou'][a]:.2f}" for a in ARMS)
                      + f" | {result['episodes']} episodes", flush=True)
        else:
            result = score_intervention(rule, args.data_path, device, args.max_episodes)
            result.update(checkpoint=vars(ck), protocol=protocol)
            p6.save(result, None, f"intervene_S1{args.tag}", args.out_dir)
            s = result["U"]
            print(f"[intervene] {s.get('events', 0)} blocks | U: R_V0 {s.get('r_v0', float('nan')):.3f} R_V1 "
                  f"{s.get('r_v1', float('nan')):.3f} R_own {s.get('r_own', float('nan')):.3f} phi "
                  f"{s.get('phi', float('nan')):.3f}", flush=True)
    finally:
        rule.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
