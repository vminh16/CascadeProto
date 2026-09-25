"""P6 of [DECISION D-38]: where the clean base's prototype gap lies, and whether a training-free query adaptation
recovers it. Inference only, CR `last.pt` (D-37's base: `stage_type=vip_clean`, random query order), S1.

Every rule is `L = F^q Rᵀ` with unit rows R in the oracle's geometry (normalised sums of unit features); rows of
classes absent from a query block keep their support direction (D-35's presence-fair form).
  U            support rows only (the base of every rule)
  oracle_x     U with the query's own direction in the background row / present foreground rows / every present row
  ssp_*        entropy-gated self-support: predict, keep the lowest-entropy fraction rho of each class's predicted
               points (>= 16), mix their direction into the row with weight 1 - alpha, repeat T times
  km*          spherical k-means of the support background into k directions, background logit = max over them

    python experiments/p6_prototype_probe.py select --data_path datasets/S3DIS/blocks_bs1_s1 --checkpoint cr:ours:1:<last.pt>
    python experiments/p6_prototype_probe.py test   --data_path datasets/S3DIS/blocks_bs1_s1 --checkpoint cr:ours:1:<last.pt>
    python experiments/p6_prototype_probe.py decide
"""

import argparse
import itertools
import json
import os
import sys
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from experiments import p0_em_probe as p0  # noqa: E402
from experiments import p5_condition_probe as p5  # noqa: E402
from experiments import r2_distill_eval as r2  # noqa: E402
from models.oracle_distill import oracle_directions  # noqa: E402

OUT_DIR = "results/phase16_p6"
D37_FIXED = "results/phase16_d37/test_S1_fixed100.json"
TEST_DRAWS = ("fixed100", "random600:0", "random600:1", "random600:2", "leakfree")
RHOS, ALPHAS, STEPS, ROW_SETS = (0.25, 0.5, 1.0), (0.25, 0.5, 0.75), (1, 2), ("bg", "fg", "all")
KS = (3, 5)
MIN_POINTS = 16  # fewer selected points keep the row [DECISION D-38]
KMEANS_ITERS, KMEANS_SEED = 20, 0
GO_GAIN, STOP_GAIN = 1.0, 0.5  # P6.2 / P6.4
LOCATION_SHARE = 2.0 / 3.0  # P6.1
CR_TOL = 1e-4  # the model's fixed100 mIoU against D-37's CR (0.01 points)


# ------------------------------------------------------------------ arms

def ssp_arms() -> List[Dict]:
    """The 54 self-support arms in grid order (ties in the selection go to the earlier one)."""
    return [dict(rho=r, alpha=a, steps=t, rows=s) for s, t, a, r in itertools.product(ROW_SETS, STEPS, ALPHAS, RHOS)]


def ssp_name(arm: Dict) -> str:
    return f"ssp_{arm['rows']}_T{arm['steps']}_a{arm['alpha']:g}_r{arm['rho']:g}"


def km_name(k: int) -> str:
    return f"km{k}"


def row_set(which: str, n_cls: int, device) -> torch.Tensor:
    """[N+1] bool: the rows an arm may change (background = row 0)."""
    m = torch.zeros(n_cls, dtype=torch.bool, device=device)
    if which in ("bg", "all"):
        m[0] = True
    if which in ("fg", "all"):
        m[1:] = True
    if which not in ("bg", "fg", "all"):
        raise ValueError(f"row set must be bg, fg or all, got {which!r}")
    return m


# ------------------------------------------------------------------ rules (pure, per episode)

def rule_logits(f_q: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """f_q [B_q, P, D], rows [B_q, N+1, D] -> L [B_q, P, N+1]."""
    return torch.einsum("bpd,bcd->bpc", f_q, rows)


def base_rows(f_q: torch.Tensor, f_s: torch.Tensor, support_y: torch.Tensor) -> torch.Tensor:
    """U's rows: the support directions of D-35, the same for every query block [B_q, N+1, D]."""
    return p5.support_directions(f_s, support_y).unsqueeze(0).expand(f_q.shape[0], -1, -1)


def oracle_rows(f_q: torch.Tensor, labels: torch.Tensor, rows: torch.Tensor, which: str) -> torch.Tensor:
    """rows with the query's own unit direction in the present rows of `which`; absent rows untouched."""
    o, present = oracle_directions(f_q, labels, rows.shape[1])  # [B_q, N+1, D], [B_q, N+1]
    take = present & row_set(which, rows.shape[1], rows.device).unsqueeze(0)  # [B_q, N+1]
    return torch.where(take.unsqueeze(-1), o, rows)  # [B_q, N+1, D]


def entropy(logits: torch.Tensor) -> torch.Tensor:
    """Shannon entropy of softmax(L) per point [B_q, P]."""
    logp = F.log_softmax(logits, dim=-1)  # [B_q, P, N+1]
    return -(logp.exp() * logp).sum(dim=-1)


def self_support(f_q: torch.Tensor, rows: torch.Tensor, rho: float, alpha: float, steps: int, which: str,
                 min_points: int = MIN_POINTS) -> torch.Tensor:
    """Entropy-gated self-support rows [B_q, N+1, D] [DECISION D-38]; no label is read.

    Per query block and class c in `which`: among the points predicted c, those whose entropy is at most the
    rho-quantile of theirs (rho = 1 keeps all); with at least `min_points` of them, S_c = n(sum of their unit
    features) and R_c <- n(alpha R_c + (1 - alpha) S_c); otherwise R_c is kept. Repeated `steps` times.
    """
    u = F.normalize(f_q, dim=-1)  # [B_q, P, D]
    allowed = row_set(which, rows.shape[1], rows.device)  # [N+1]
    for _ in range(steps):
        logits = rule_logits(f_q, rows)  # [B_q, P, N+1]
        pred, h = logits.argmax(dim=-1), entropy(logits)  # [B_q, P], [B_q, P]
        new = rows.clone()  # [B_q, N+1, D]
        for b in range(rows.shape[0]):
            for c in torch.nonzero(allowed).flatten().tolist():
                cand = pred[b] == c  # [P]
                if int(cand.sum()) < min_points:
                    continue
                sel = cand & (h[b] <= torch.quantile(h[b][cand], rho))  # [P]
                if int(sel.sum()) < min_points:
                    continue
                s_c = F.normalize(u[b][sel].sum(dim=0), dim=-1)  # [D]
                new[b, c] = F.normalize(alpha * rows[b, c] + (1.0 - alpha) * s_c, dim=-1)
        rows = new
    return rows


def spherical_kmeans(x: torch.Tensor, k: int, iters: int = KMEANS_ITERS, seed: int = KMEANS_SEED) -> torch.Tensor:
    """k unit centroids [k, D] of unit vectors x [M, D]: k-means++ seeding on 1 - cos, max-cosine assignment,
    centroid = normalised member sum; an empty cluster keeps its centroid. Deterministic for a seed."""
    if x.shape[0] < k:
        raise ValueError(f"{x.shape[0]} points cannot give {k} centroids")
    g = torch.Generator(device="cpu").manual_seed(seed)
    first = int(torch.randint(x.shape[0], (1,), generator=g))
    cent = [x[first]]
    for _ in range(1, k):
        d = (1.0 - x @ torch.stack(cent).T).clamp_min(0).min(dim=1).values  # [M]
        p = (d / d.sum()).cpu() if float(d.sum()) > 0 else torch.full((x.shape[0],), 1.0 / x.shape[0])
        cent.append(x[int(torch.multinomial(p, 1, generator=g))])
    c = torch.stack(cent)  # [k, D]
    for _ in range(iters):
        assign = (x @ c.T).argmax(dim=1)  # [M]
        members = F.one_hot(assign, k).to(x.dtype)  # [M, k]
        sums = members.T @ x  # [k, D]
        filled = members.sum(dim=0) > 0  # [k]
        c = torch.where(filled.unsqueeze(-1), F.normalize(sums, dim=-1), c)
    return c


def background_units(f_s: torch.Tensor, support_y: torch.Tensor) -> torch.Tensor:
    """Unit features of every support background point [M, D]."""
    return F.normalize(f_s[support_y == 0], dim=-1)  # [M, D]


def multi_bg_logits(f_q: torch.Tensor, rows: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    """U's logits with the background logit = max over the centroid directions [B_q, P, N+1]."""
    logits = rule_logits(f_q, rows)  # [B_q, P, N+1]
    bg = torch.einsum("bpd,kd->bpk", f_q, centroids).max(dim=-1).values  # [B_q, P]
    return torch.cat([bg.unsqueeze(-1), logits[..., 1:]], dim=-1)


def arm_predictions(f_q: torch.Tensor, f_s: torch.Tensor, support_y: torch.Tensor, names: List[str],
                    labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
    """Predictions [B_q, P] of the named arms on one episode; oracle arms need `labels`."""
    rows = base_rows(f_q, f_s, support_y)  # [B_q, N+1, D]
    by_name = {ssp_name(a): a for a in ssp_arms()}
    out, bg = {}, None
    for name in names:
        if name == "U":
            logits = rule_logits(f_q, rows)
        elif name.startswith("oracle_"):
            if labels is None:
                raise ValueError("oracle arms read the query labels")
            logits = rule_logits(f_q, oracle_rows(f_q, labels, rows, name[len("oracle_"):]))
        elif name in by_name:
            a = by_name[name]
            logits = rule_logits(f_q, self_support(f_q, rows, a["rho"], a["alpha"], a["steps"], a["rows"]))
        elif name.startswith("km"):
            bg = background_units(f_s, support_y) if bg is None else bg
            logits = multi_bg_logits(f_q, rows, spherical_kmeans(bg, int(name[2:])))
        else:
            raise ValueError(f"unknown arm {name!r}")
        out[name] = logits.argmax(dim=-1)
    return out


# ------------------------------------------------------------------ selection and rules (pure)

def select(miou: Dict[str, float], family: List[str], base: str = "U") -> Tuple[str, float]:
    """The arm of `family` with the largest valid gain over `base` (points); ties go to the earlier arm."""
    best = max(family, key=lambda n: (miou[n], -family.index(n)))
    return best, 100.0 * (miou[best] - miou[base])


def location(gains: Dict[str, float]) -> str:
    """P6.1 on the bounds g_bg, g_fg, g_all (points)."""
    fg, bg = gains["fg"] >= LOCATION_SHARE * gains["all"], gains["bg"] >= LOCATION_SHARE * gains["all"]
    if fg and bg:
        return "either row suffices"
    return "foreground" if fg else "background" if bg else "joint"


def decide(draws: Dict[str, Tuple[Dict, Dict[str, np.ndarray]]], frozen: Dict[str, str]) -> List[Tuple[str, str]]:
    """Rules P6.1-P6.5 of [DECISION D-38]."""
    missing = [d for d in TEST_DRAWS if d not in draws]
    if missing:
        return [("incomplete", f"missing draws {missing}")]
    fixed_r, fixed_c = draws["fixed100"]
    v = []
    for draw in ("fixed100", "leakfree"):
        r = draws[draw][0]["miou"]
        gains = {w: 100.0 * (r[f"oracle_{w}"] - r["U"]) for w in ("bg", "fg", "all")}
        v.append((f"P6.1 location {draw}", f"{location(gains)}: oracle bg {gains['bg']:+.2f}, fg {gains['fg']:+.2f}, "
                                           f"all {gains['all']:+.2f} over U {100 * r['U']:.2f}"))
    verdicts = {}
    for family in ("ssp", "km"):
        arm = frozen[family]
        p = p0.paired_bootstrap(fixed_c["U"], fixed_c[arm])
        rand = [100.0 * (draws[d][0]["miou"][arm] - draws[d][0]["miou"]["U"]) for d in TEST_DRAWS[1:4]]
        text = f"{arm} - U fixed100 {r2.ci_text(p)}; random600 {[round(g, 2) for g in rand]}"
        if p["gain"] >= GO_GAIN and p["ci_low"] > 0 and min(rand) > 0:
            verdicts[family] = "go"
        elif p["gain"] < STOP_GAIN:
            verdicts[family] = "stop"
        else:
            verdicts[family] = "between"
        rule = {"ssp": "P6.2 self-support", "km": "P6.3 multi-background"}[family]
        v.append((f"{rule} {verdicts[family]}", text))
    if verdicts["ssp"] == "stop" and verdicts["km"] == "stop":
        v.append(("P6.4 stop", "training-free query adaptation does not recover the gap; D-39 is a trained module"))
    r = fixed_r["miou"]
    closed = {a: (r[a] - r["U"]) / max(r["oracle_all"] - r["U"], 1e-9) for a in (frozen["ssp"], frozen["km"])}
    v.append(("P6.5 reported", f"model {100 * r['model']:.2f}, support rule {100 * r['support_rule']:.2f}, U "
                               f"{100 * r['U']:.2f}; share of the oracle gap closed "
                               + ", ".join(f"{a} {100 * s:.1f} %" for a, s in closed.items())
                               + "; leak-free " + ", ".join(f"{a} {100 * draws['leakfree'][0]['miou'][a]:.2f}"
                                                           for a in ("model", "U", frozen['ssp'], frozen['km'],
                                                                     'oracle_all'))))
    return v


# ------------------------------------------------------------------ GPU passes

def episodes(draw: str, data_path: str, max_episodes: Optional[int]) -> Tuple[Iterator, List[int]]:
    """(iterator of loader items, test classes) of a draw; `valid` is D-15's selection draw of S1."""
    if draw == "valid":
        from pipeline.episodes import build_eval_dataset

        ds = build_eval_dataset(data_path, "s3dis", 1, 2, 1, mode="valid", seed=0)
        classes = [int(c) for c in np.asarray(ds.classes)]
    elif draw == "leakfree":
        classes = p5.test_classes_of(1, data_path)
        gen = (ep["item"] for ep in p5.draw_episodes(data_path, 1, p5.DRAW_SEEDS["leakfree"], p5.EPISODES_PER_PAIR,
                                                     True, max_episodes))
        return gen, classes
    else:
        ds, classes = r2.episodes_of(draw, data_path, 1)
    n = len(ds) if max_episodes is None else min(max_episodes, len(ds))
    return (ds[i] for i in range(n)), classes


@torch.no_grad()
def score(rule, draw: str, data_path: str, names: List[str], device, max_episodes: Optional[int]) -> Tuple[Dict, Dict]:
    from pipeline.episodes import make_episode, read_class_names

    cls_names = read_class_names(data_path, "s3dis")
    items, test_classes = episodes(draw, data_path, max_episodes)
    counts: Dict[str, List[np.ndarray]] = {}
    preds_model, gts, l2c, skipped = [], [], [], 0
    for item in items:
        if (item[1].reshape(item[1].shape[0], -1).sum(1) == 0).any():
            skipped += 1  # a uniformly sampled support block without its class (leak-free draw only)
            continue
        e = make_episode(item, cls_names).to(device)
        f_q, m_eff, _, logits = rule(e)
        p0.check_identity(f_q, m_eff, logits)  # the rule reads the tensors the model scores with
        f_s = p5.support_features(rule, e)
        labels = e.query_y if any(n.startswith("oracle_") for n in names) else None
        preds = arm_predictions(f_q, f_s, e.support_y, [n for n in names if n not in ("model", "support_rule")], labels)
        preds["model"] = logits.argmax(dim=-1)
        preds["support_rule"] = p5.support_rule_logits(f_q, f_s, e.support_y).argmax(dim=-1)
        gt = e.query_y.cpu().numpy()
        for n in names:
            counts.setdefault(n, []).append(p0.episode_counts(preds[n].cpu().numpy(), gt, e.sampled_classes, test_classes))
        preds_model.append(preds["model"].cpu().numpy()), gts.append(gt), l2c.append(e.sampled_classes)
    stacked = {k: np.stack(v) for k, v in counts.items()}
    miou = {k: float(p0.miou_from_counts(v.sum(0))) for k, v in stacked.items()}
    if draw == "fixed100":  # the model's count-based mIoU is VIP-Seg's metric and D-37's CR number
        from pipeline.evaluation import accumulated_miou

        class _Quiet:
            def cprint(self, text):
                pass

        primary = accumulated_miou(_Quiet(), preds_model, gts, l2c, test_classes)
        if abs(primary - miou["model"]) > 1e-9:
            raise RuntimeError("the count-based mIoU differs from VIP-Seg's evaluate_metric")
        if max_episodes is None:
            with open(os.path.join(REPO, D37_FIXED)) as f:
                ref = json.load(f)["miou"]["cr"]
            if abs(ref - miou["model"]) > CR_TOL:
                raise RuntimeError(f"the model scores {miou['model']:.6f} on fixed100, D-37's CR {ref:.6f}")
    return {"draw": draw, "episodes": len(stacked["model"]), "skipped": skipped, "test_classes": test_classes,
            "miou": miou, "class_iou": {k: p0.class_ious(v.sum(0)).tolist() for k, v in stacked.items()},
            "recall": {k: (v.sum(0)[2] / np.maximum(v.sum(0)[0], 1)).tolist() for k, v in stacked.items()},
            "precision": {k: (v.sum(0)[2] / np.maximum(v.sum(0)[1], 1)).tolist() for k, v in stacked.items()}}, stacked


def stem(draw: str, tag: str) -> str:
    return f"test_S1_{draw.replace(':', '_seed')}{tag}"


def load_rule(spec: str, device):
    ck = p0.parse_checkpoint(spec)
    if ck.fold != 1 or ck.kind != "ours":
        raise ValueError(f"{ck.name}: P6 reads our S1 checkpoint of D-37's base [DECISION D-38]")
    rule, protocol, config = r2.load(ck, 1, device)  # eval.py's protocol guard [DECISION D-22]
    if config["stage_type"] != "vip_clean" or config["use_lma"] or not config["l2norm_point_proto"]:
        raise ValueError(f"{ck.name}: P6 reads the clean base (vip_clean, no LMA, L2 prototypes) [DECISION D-38]")
    return rule, protocol, config, ck


def save(obj: Dict, arrays: Optional[Dict[str, np.ndarray]], name: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, name + ".json"), "w") as f:
        json.dump(obj, f, indent=1)
    if arrays is not None:
        np.savez_compressed(os.path.join(out_dir, name + "_counts.npz"), **arrays)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["select", "test", "decide"])
    p.add_argument("--data_path")
    p.add_argument("--checkpoint")
    p.add_argument("--max_episodes", type=int, default=None, help="smoke runs only")
    p.add_argument("--tag", default="", help="suffix of the output files (smoke runs)")
    p.add_argument("--out_dir", default=OUT_DIR)
    args = p.parse_args(argv)
    sel_path = os.path.join(args.out_dir, f"select{args.tag}.json")
    if args.stage == "decide":
        with open(sel_path) as f:
            frozen = json.load(f)["frozen"]
        draws = {}
        for d in TEST_DRAWS:
            path = os.path.join(args.out_dir, stem(d, args.tag) + ".json")
            if os.path.isfile(path):
                with open(path) as f:
                    draws[d] = (json.load(f), dict(np.load(path.replace(".json", "_counts.npz"))))
        for rule_name, text in decide(draws, frozen):
            print(f"{rule_name:30s} {text}")
        return 0
    if not args.data_path or not args.checkpoint:
        p.error("select and test need --data_path and --checkpoint")
    device = torch.device("cuda")
    rule, protocol, config, ck = load_rule(args.checkpoint, device)
    try:
        if args.stage == "select":
            names = ["model", "support_rule", "U"] + [ssp_name(a) for a in ssp_arms()] + [km_name(k) for k in KS]
            result, stacked = score(rule, "valid", args.data_path, names, device, args.max_episodes)
            ssp, g_ssp = select(result["miou"], [ssp_name(a) for a in ssp_arms()])
            km, g_km = select(result["miou"], [km_name(k) for k in KS])
            result.update(frozen={"ssp": ssp, "km": km}, valid_gain={"ssp": g_ssp, "km": g_km},
                          checkpoint=vars(ck), protocol=protocol)
            save(result, stacked, f"select{args.tag}", args.out_dir)
            print(f"[select] U {100 * result['miou']['U']:.2f}, model {100 * result['miou']['model']:.2f} | frozen "
                  f"{ssp} ({g_ssp:+.2f}), {km} ({g_km:+.2f}) on valid", flush=True)
        else:
            with open(sel_path) as f:
                frozen = json.load(f)["frozen"]
            names = ["model", "support_rule", "U", "oracle_bg", "oracle_fg", "oracle_all", frozen["ssp"], frozen["km"]]
            for draw in TEST_DRAWS:
                result, stacked = score(rule, draw, args.data_path, names, device, args.max_episodes)
                result.update(frozen=frozen, checkpoint=vars(ck), protocol=protocol)
                save(result, stacked, stem(draw, args.tag), args.out_dir)
                print(f"[test] {draw}: " + " | ".join(f"{n} {100 * result['miou'][n]:.2f}" for n in names)
                      + f" | {result['episodes']} episodes", flush=True)
    finally:
        rule.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
