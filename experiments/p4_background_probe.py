"""P4 of [DECISION D-32]: is E1's background prototype contaminated by the episode's own classes, causally?

    python experiments/p4_background_probe.py select --data_path datasets/S3DIS/blocks_bs1_s1 \
        --checkpoint e1:ours:1:log_r2/s3dis_S1_N2_K1_point_T4_vip_b1/last.pt          # draw A, freezes q
    python experiments/p4_background_probe.py test --data_path ... --checkpoint ...   # draw B
    python experiments/p4_background_probe.py decide                                  # rules P4.1-P4.3

Episodes are drawn like the loader's test episodes, from the inherited functions, with one addition: the
support scans are also read with `support=False`, which labels every support point by its index among the
episode's classes with the same point sampling [VIPSEG dataloaders/loader.py:31-87]. So each support point
is known to be this way's class, the other way's class, or neither. E1's head is re-run from a modified
first prototype `P^0`; an identity check reproduces E1's own logits from the unmodified one.
"""

import argparse
import itertools
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from experiments import p0_em_probe as p0  # noqa: E402
from experiments import r2_distill_eval as r2  # noqa: E402

OUT_DIR = "results/phase16_p4"
FRACTIONS = (0.05, 0.1, 0.2, 0.3)  # [DECISION D-32]
EPISODES_PER_PAIR = 100  # as fixed100
DRAW_SEEDS = {"select": 1, "test": 2}  # draws A and B [DECISION D-32]
CAUSAL_GAIN, FIX_GAIN, NECK_GAP = 1.0, 0.5, 1.0  # [DECISION D-32]


# ------------------------------------------------------------------ pure functions (tested on the CPU)

def bg_prototype(f_s: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """Mean of the kept support points [D]; f_s [N, K, P, D], keep [N, K, P] bool (mask-0 points to use)."""
    if not keep.any():
        raise ValueError("no background support point left to average")
    w = keep.to(f_s.dtype)  # [N, K, P]
    return torch.einsum("nkp,nkpd->d", w, f_s) / w.sum()  # [D]


def fg_prototypes(f_s: torch.Tensor, support_y: torch.Tensor) -> torch.Tensor:
    """Masked mean per way [N, D] (Eq.3's foreground rows)."""
    w = support_y.to(f_s.dtype)  # [N, K, P]
    return torch.einsum("nkp,nkpd->nd", w, f_s) / w.sum(dim=(1, 2)).unsqueeze(-1)  # [N, D]


def other_way_score(f_s: torch.Tensor, fg: torch.Tensor) -> torch.Tensor:
    """[N, K, P]: for a point of way j's block, max over ways k != j of cos(f, fg_k)."""
    cos = torch.einsum("nkpd,md->nkpm", F.normalize(f_s, dim=-1), F.normalize(fg, dim=-1))  # [N, K, P, N]
    n = f_s.shape[0]
    own = torch.eye(n, dtype=torch.bool, device=f_s.device).view(n, 1, 1, n)  # [N, 1, 1, N]
    return cos.masked_fill(own, float("-inf")).amax(dim=-1)  # [N, K, P]


def purify_keep(f_s: torch.Tensor, support_y: torch.Tensor, q: float) -> torch.Tensor:
    """Mask-0 points to keep [N, K, P]: in each block, drop the fraction q of its mask-0 points that are most
    similar to another way's foreground prototype. q = 0 keeps every mask-0 point (E1's background)."""
    bg = support_y == 0  # [N, K, P]
    if q <= 0:
        return bg
    score = other_way_score(f_s, fg_prototypes(f_s, support_y))  # [N, K, P]
    keep = bg.clone()
    for j, k in itertools.product(range(f_s.shape[0]), range(f_s.shape[1])):
        idx = bg[j, k].nonzero().flatten()  # [M]
        n_drop = int(q * idx.numel())
        if n_drop:
            drop = idx[score[j, k, idx].topk(n_drop).indices]  # [n_drop]
            keep[j, k, drop] = False
    return keep


def first_prototype(f_s: torch.Tensor, support_y: torch.Tensor, keep_bg: torch.Tensor) -> torch.Tensor:
    """P^0 [N+1, D] with L2 rows, route B's form [VIPSEG models/vipseg.py:142] [DECISION D-10]."""
    rows = torch.cat([bg_prototype(f_s, keep_bg).unsqueeze(0), fg_prototypes(f_s, support_y)], dim=0)  # [N+1, D]
    return F.normalize(rows, dim=-1)  # [N+1, D]


def run_head(model, f_s: torch.Tensor, f_q: torch.Tensor, p0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """(M_eff [B_q, N+1, D], logits [B_q, P, N+1]) of the model's head started from P^0 [N+1, D]."""
    p = p0.unsqueeze(0).expand(f_q.shape[0], -1, -1)  # [B_q, N+1, D]
    steps = []
    for stage in model.stages:
        p = stage(p, f_s, f_q)  # [B_q, N+1, D]
        steps.append(p)
    m_eff = model.effective_prototype(f_q, p0, steps)  # [B_q, N+1, D]
    return m_eff, torch.einsum("bpd,bcd->bpc", f_q, m_eff)  # [B_q, N+1, D], [B_q, P, N+1]


def row_oracle_logits(f_q: torch.Tensor, m_eff: torch.Tensor, labels: torch.Tensor, rows) -> torch.Tensor:
    """Logits with only `rows` of M_eff replaced by the query's own direction, each keeping its norm."""
    from models.oracle_distill import oracle_directions

    o, present = oracle_directions(f_q, labels, m_eff.shape[1])  # [B_q, N+1, D], [B_q, N+1]
    sel = torch.zeros_like(present)
    sel[:, list(rows)] = True
    sel = sel & present  # [B_q, N+1]
    m = torch.where(sel.unsqueeze(-1), m_eff.norm(dim=-1, keepdim=True) * o, m_eff)  # [B_q, N+1, D]
    return torch.einsum("bpd,bcd->bpc", f_q, m)  # [B_q, P, N+1]


def auc_terms(score: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, bins: int = 400):
    """Histograms of the purifier score for other-way (positive) and clean (negative) mask-0 points."""
    edges = torch.linspace(-1.0, 1.0, bins + 1, device=score.device)
    idx = torch.bucketize(score.clamp(-1, 1), edges[1:-1])  # [..], bin index 0..bins-1
    return (torch.bincount(idx[positive], minlength=bins).cpu().numpy(),
            torch.bincount(idx[negative], minlength=bins).cpu().numpy())


def auc(pos_hist: np.ndarray, neg_hist: np.ndarray) -> float:
    """P(score of a positive > score of a negative), ties counted half, from two histograms."""
    npos, nneg = pos_hist.sum(), neg_hist.sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    below = np.concatenate([[0], np.cumsum(neg_hist)[:-1]])  # negatives in lower bins
    return float((pos_hist * (below + 0.5 * neg_hist)).sum() / (npos * nneg))


def class_rates(counts: np.ndarray) -> Dict[str, List[float]]:
    """Recall and precision per column (background first) from summed counts [3, C+1] (gt, pred, tp)."""
    gt, pr, tp = counts
    return {"recall": (tp / np.maximum(gt, 1)).tolist(), "precision": (tp / np.maximum(pr, 1)).tolist()}


# ------------------------------------------------------------------ episodes with the other way's labels

def draw_episodes(data_path: str, cvfold: int, seed: int, per_pair: int, max_episodes=None):
    """Yield (item, support_multi) with item the loader's tuple and support_multi [N, K, P] in {0..N}."""
    from dataloaders.loader import MyDataset, sample_K_pointclouds
    from pipeline.episodes import N_QUERIES, NUM_POINT, PC_ATTRIBS, WAY_NUM, WAY_RATIO

    ds = MyDataset(data_path, "s3dis", cvfold=cvfold, num_episode=1, n_way=2, k_shot=1, n_queries=N_QUERIES,
                   mode="test", num_point=NUM_POINT, pc_attribs=PC_ATTRIBS, pc_augm=False,
                   way_ratio=WAY_RATIO, way_num=WAY_NUM)
    pairs = list(itertools.combinations(sorted(int(c) for c in ds.classes), 2))
    count = 0
    for pi, pair in enumerate(pairs):
        for e in range(per_pair):
            if max_episodes is not None and count >= max_episodes:
                return
            np.random.seed([seed, pi, e])  # the loader draws scans and points from the global np.random
            classes = np.array(pair)
            used, sx, sm, qx, qy = [], [], [], [], []
            for c in classes:  # generate_one_episode's order and black list [VIPSEG loader.py:174-218]
                q_scans = np.random.choice([s for s in ds.query_class2scans[c] if s not in used], N_QUERIES,
                                           replace=False)
                used.extend(q_scans)
                s_scans = np.random.choice([s for s in ds.support_class2scans[c] if s not in used], 1, replace=False)
                used.extend(s_scans)
                x, y = sample_K_pointclouds(data_path, NUM_POINT, PC_ATTRIBS, False, None, q_scans, c, classes,
                                            is_support=False, is_random_sample=False)
                qx.append(x), qy.append(y)
                x, multi = sample_K_pointclouds(data_path, NUM_POINT, PC_ATTRIBS, False, None, s_scans, c, classes,
                                                is_support=False, is_random_sample=False)
                sx.append(x), sm.append(multi)
            support_multi = np.stack(sm)  # [N, K, P], 0 or the 1-based index of an episode class
            support_y = np.stack([(support_multi[k] == k + 1) for k in range(len(classes))]).astype(np.int32)
            yield (np.stack(sx), support_y, np.concatenate(qx), np.concatenate(qy).astype(np.int64), classes), support_multi
            count += 1


# ------------------------------------------------------------------ one pass

@torch.no_grad()
def score(ck, data_path: str, stage: str, fractions, device, per_pair: int, max_episodes=None) -> Tuple[Dict, Dict]:
    from pipeline.episodes import make_episode, read_class_names

    rule, protocol, _ = r2.load(ck, ck.fold, device)
    model = rule.model
    if model.config.use_lma or model.config.logit_scale != "none" or not model.config.l2norm_point_proto:
        raise ValueError("P4 re-runs route B's head (no LMA, no logit scale, L2 prototypes) [DECISION D-32]")
    from dataloaders.s3dis import S3DISDataset

    names = read_class_names(data_path, "s3dis")
    test_classes = [int(c) for c in S3DISDataset(ck.fold, data_path).test_classes]  # the loader's order
    counts: Dict[str, List[np.ndarray]] = {}
    hist_pos, hist_neg = np.zeros(400, dtype=np.int64), np.zeros(400, dtype=np.int64)
    contamination = [0, 0]
    for n, (item, multi) in enumerate(draw_episodes(data_path, ck.fold, DRAW_SEEDS[stage], per_pair, max_episodes)):
        episode = make_episode(item, names).to(device)
        multi_t = torch.as_tensor(multi, device=device)  # [N, K, P]
        f_s, f_q = model.features.encode_episode(episode.support_x, episode.query_x)  # [N,K,P,D], [B_q,P,D]
        labels = episode.query_y  # [B_q, P]
        gt = labels.cpu().numpy()
        bg = episode.support_y == 0  # [N, K, P]
        n_way = f_s.shape[0]
        own = torch.arange(1, n_way + 1, device=device).view(n_way, 1, 1)  # [N, 1, 1]
        other = (multi_t > 0) & (multi_t != own) & bg  # [N, K, P], the other way's points labelled background
        contamination[0] += int(other.sum())
        contamination[1] += int(bg.sum())
        p0_e1 = first_prototype(f_s, episode.support_y, bg)  # E1's own P^0
        m_eff, logits = run_head(model, f_s, f_q, p0_e1)
        if n < 5 or n % 50 == 0:
            p0.check_identity(f_q, m_eff, model(episode).logits)  # the re-run head is E1
        preds = {"model": logits.argmax(-1)}
        _, lc = run_head(model, f_s, f_q, first_prototype(f_s, episode.support_y, bg & ~other))
        preds["clean_bg"] = lc.argmax(-1)
        for q in fractions:
            _, lq = run_head(model, f_s, f_q, first_prototype(f_s, episode.support_y, purify_keep(f_s, episode.support_y, q)))
            preds[f"purify_q{q:g}"] = lq.argmax(-1)
        preds["oracle_bg_only"] = row_oracle_logits(f_q, m_eff, labels, [0]).argmax(-1)
        preds["oracle_fg_only"] = row_oracle_logits(f_q, m_eff, labels, range(1, n_way + 1)).argmax(-1)
        preds["oracle_all"] = row_oracle_logits(f_q, m_eff, labels, range(n_way + 1)).argmax(-1)
        score_ = other_way_score(f_s, fg_prototypes(f_s, episode.support_y))  # [N, K, P]
        hp, hn = auc_terms(score_, other, bg & ~other)
        hist_pos += hp
        hist_neg += hn
        for k, pred in preds.items():
            counts.setdefault(k, []).append(p0.episode_counts(pred.cpu().numpy(), gt, episode.sampled_classes,
                                                              test_classes))
    stacked = {k: np.stack(v) for k, v in counts.items()}  # name -> [E, 3, C+1]
    result = {"checkpoint": vars(ck), "stage": stage, "protocol": protocol, "episodes": len(counts["model"]),
              "test_classes": test_classes, "seed": DRAW_SEEDS[stage],
              "miou": {k: float(p0.miou_from_counts(v.sum(0))) for k, v in stacked.items()},
              "class_iou": {k: p0.class_ious(v.sum(0)).tolist() for k, v in stacked.items()},
              "rates": {k: class_rates(v.sum(0)) for k, v in stacked.items()},
              "contamination": contamination[0] / max(contamination[1], 1),
              "purifier_auc": auc(hist_pos, hist_neg)}
    return result, stacked


def save(result: Dict, stacked: Dict[str, np.ndarray], stem: str) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, stem + ".json"), "w") as f:
        json.dump(result, f, indent=1)
    np.savez_compressed(os.path.join(OUT_DIR, stem + "_counts.npz"), **stacked)


def select_fraction(result: Dict) -> Dict:
    gains = {q: 100.0 * (result["miou"][f"purify_q{q:g}"] - result["miou"]["model"]) for q in FRACTIONS}
    best = max(FRACTIONS, key=lambda q: gains[q])
    return {"q": best, "valid_gain": gains[best], "gains": {f"{q:g}": g for q, g in gains.items()}}


def cmd_select(args, device) -> int:
    ck = p0.parse_checkpoint(args.checkpoint)
    if ck.fold != 1:
        raise ValueError(f"{ck.name}: P4 runs on S1 only, S0 stays held out [DECISION D-22]")
    result, stacked = score(ck, args.data_path, "select", FRACTIONS, device, args.per_pair, args.max_episodes)
    result["selection"] = select_fraction(result)
    save(result, stacked, f"select_{ck.name}")
    print(f"[select] model {result['miou']['model']:.4f} clean_bg {result['miou']['clean_bg']:.4f} | frozen q "
          f"{result['selection']['q']:g} ({result['selection']['valid_gain']:+.2f}) | contamination "
          f"{result['contamination']:.3f} | purifier AUC {result['purifier_auc']:.3f}", flush=True)
    return 0


def cmd_test(args, device) -> int:
    ck = p0.parse_checkpoint(args.checkpoint)
    with open(os.path.join(OUT_DIR, f"select_{ck.name}.json")) as f:
        q = json.load(f)["selection"]["q"]
    result, stacked = score(ck, args.data_path, "test", (q,), device, args.per_pair, args.max_episodes)
    frozen = f"purify_q{q:g}"
    result["frozen"] = frozen
    result["paired"] = {name: p0.paired_bootstrap(stacked["model"], stacked[name])
                        for name in ("clean_bg", frozen, "oracle_bg_only", "oracle_fg_only", "oracle_all")}
    result["paired"]["clean_bg_vs_frozen"] = p0.paired_bootstrap(stacked[frozen], stacked["clean_bg"])
    save(result, stacked, f"test_{ck.name}")
    for k, v in result["paired"].items():
        print(f"[test] {k:20s} {v['gain']:+.2f} [{v['ci_low']:+.2f}, {v['ci_high']:+.2f}]", flush=True)
    print(f"[test] contamination {result['contamination']:.3f} | purifier AUC {result['purifier_auc']:.3f}", flush=True)
    return 0


def decide(test: Dict) -> List[Tuple[str, str]]:
    """Rules P4.1-P4.3 of [DECISION D-32]."""
    names = test["test_classes"]
    c = test["paired"]["clean_bg"]
    rec_m, rec_c = test["rates"]["model"]["recall"], test["rates"]["clean_bg"]["recall"]
    s3dis = ["ceiling", "floor", "wall", "beam", "column", "window", "door", "table", "chair", "sofa", "bookcase", "board"]
    cols = {s3dis[g]: i + 1 for i, g in enumerate(names)}
    planar = [n for n in ("floor", "wall") if n in cols]
    rec_up = all(rec_c[cols[n]] > rec_m[cols[n]] for n in planar)
    rec_text = ", ".join(f"{n} recall {rec_m[cols[n]]:.3f} -> {rec_c[cols[n]]:.3f}" for n in planar)
    v = []
    causal = c["gain"] >= CAUSAL_GAIN and c["ci_low"] > 0 and rec_up
    if causal:
        v.append(("P4.1 causal", f"clean_bg {c['gain']:+.2f} [{c['ci_low']:+.2f}, {c['ci_high']:+.2f}]; {rec_text}"))
    elif c["gain"] < 0.5:
        v.append(("P4.1 not causal", f"clean_bg {c['gain']:+.2f}: stop this line; {rec_text}"))
    else:
        v.append(("P4.1 in between", f"clean_bg {c['gain']:+.2f} [{c['ci_low']:+.2f}, {c['ci_high']:+.2f}]; {rec_text}"))
    f = test["paired"][test["frozen"]]
    v.append(("P4.2 label-free fix", f"{test['frozen']} {f['gain']:+.2f} [{f['ci_low']:+.2f}, {f['ci_high']:+.2f}]: "
              + ("a training-free candidate" if f["gain"] >= FIX_GAIN and f["ci_low"] > 0 else "not enough")))
    gap = test["paired"]["clean_bg_vs_frozen"]["gain"]
    if causal and gap >= NECK_GAP:
        v.append(("P4.3 neck", f"clean_bg - {test['frozen']} {gap:+.2f} >= +{NECK_GAP}: a learned purification is warranted"))
    elif causal:
        v.append(("P4.3 neck", f"clean_bg - {test['frozen']} {gap:+.2f} < +{NECK_GAP}: the rule suffices, no neck"))
    else:
        v.append(("P4.3 neck", "not reached (P4.1 does not hold)"))
    for k in ("oracle_bg_only", "oracle_fg_only", "oracle_all"):
        g = test["paired"][k]
        v.append((f"split {k}", f"{g['gain']:+.2f} [{g['ci_low']:+.2f}, {g['ci_high']:+.2f}]"))
    v.append(("diagnostics", f"contamination {test['contamination']:.3f}, purifier AUC {test['purifier_auc']:.3f}"))
    return v


def cmd_decide(args) -> int:
    with open(os.path.join(OUT_DIR, f"test_{args.name}.json")) as f:
        test = json.load(f)
    for rule, text in decide(test):
        print(f"{rule:22s} {text}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["select", "test", "decide"])
    p.add_argument("--data_path")
    p.add_argument("--checkpoint")
    p.add_argument("--name", default="e1")
    p.add_argument("--per_pair", type=int, default=EPISODES_PER_PAIR)
    p.add_argument("--max_episodes", type=int, default=None, help="smoke runs only")
    args = p.parse_args(argv)
    if args.stage == "decide":
        return cmd_decide(args)
    if not args.data_path or not args.checkpoint:
        p.error("select and test need --data_path and --checkpoint")
    device = torch.device("cuda")
    return cmd_select(args, device) if args.stage == "select" else cmd_test(args, device)


if __name__ == "__main__":
    raise SystemExit(main())
