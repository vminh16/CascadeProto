"""P3 of [DECISION D-31]: where E1's prototype error sits, and a training-free text prior. No training.

    python experiments/p3_probe.py select --data_path datasets/S3DIS/blocks_bs1_s1 \
        --checkpoint e1:ours:1:log_r2/s3dis_S1_N2_K1_point_T4_vip_b1/last.pt
    python experiments/p3_probe.py test --data_path ... --checkpoint e1:ours:1:...     # fixed100, frozen arm
    python experiments/p3_probe.py decide                                               # rules P3.0-P3.5

Part A (test stage) scores, on the same fixed100 episodes, the support-prototype rule without the head, E1
and the equal-norm oracle rule, and splits the oracle's fixable points by boundary/interior and by support
size. Part B scores the text prior of `models/text_prior.py` over its grid on the S1 valid draw, freezes
the best arm, and tests it once on fixed100 with its entropy sibling and an oracle-weight upper bound.
Every comparison is paired on identical episodes and weights; the checkpoint passes eval.py's protocol
guard [DECISION D-22].
"""

import argparse
import itertools
import json
import os
import sys
from types import SimpleNamespace
from typing import Dict, List, Tuple

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from experiments import p0_em_probe as p0  # noqa: E402
from experiments import p1_bpc_probe as p1  # noqa: E402
from experiments import r2_distill_eval as r2  # noqa: E402
from models import text_prior as tp  # noqa: E402

OUT_DIR = "results/phase16_p3"
SOURCES = ("ridge", "retrieval_t30", "retrieval_t100")
WEIGHTS = ("one", "entropy")
KAPPAS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
N_CLASSES = 12  # S3DIS's fold classes; clutter is never sampled as a way
MECH_ACC, GO_GAIN, COLLAPSE = 0.60, 0.5, -3.0  # [DECISION D-31]
BOUNDARY_ENRICHED, REGION_LEVEL, SUPPORT_RHO = 1.5, 1.2, -0.3  # interpretation bands of Part A [D-31]


def grid() -> List[Dict]:
    return [dict(source=s, prompt=p, weight=w, kappa=k)
            for s, p, w, k in itertools.product(SOURCES, tp.PROMPT_SETS, WEIGHTS, KAPPAS)]


def name_of(a: Dict) -> str:
    return tp.arm_name(a["source"], a["prompt"], a["weight"], a["kappa"])


def directions(source: str, e_all: torch.Tensor, novel, base_idx, base: torch.Tensor) -> torch.Tensor:
    if source == "ridge":
        return tp.ridge_directions(e_all, novel, base_idx, base)
    if source.startswith("retrieval_t"):
        return tp.retrieval_directions(e_all, novel, base_idx, base, float(source[len("retrieval_t"):]))
    raise ValueError(f"unknown text source {source!r}")


class BankRule:
    """E1 behind P1's bank builder: (F^q, M_eff, logits) per episode, with F^s kept in `self.f_s`."""

    def __init__(self, rule):
        self.rule = rule

    def __call__(self, episode):
        self.f_s, _ = self.rule.model.features.encode_episode(episode.support_x, episode.query_x)  # [N,K,P,D]
        f_q, m_eff, _, logits = self.rule(episode)
        return f_q, m_eff, logits


def clip_encoder(device):
    import clip

    from models.clip_text import DEFAULT_CLIP_VARIANT, load_clip

    model = load_clip(DEFAULT_CLIP_VARIANT, device)

    def encode(prompts):
        with torch.no_grad():
            return model.encode_text(clip.tokenize(prompts).to(device)).float().cpu()

    return encode


def load(ck, device):
    rule, protocol, config = r2.load(ck, ck.fold, device)
    if ck.kind != "ours":
        raise ValueError("P3 reads our E1 checkpoint (route B's base) [DECISION D-31]")
    return rule, protocol, config


# ------------------------------------------------------------------ one pass over a draw

@torch.no_grad()
def score(ck, data_path: str, mode: str, arms: List[Dict], device, bank_n: int, max_episodes=None,
          oracle_arm: Dict = None) -> Tuple[Dict, Dict[str, np.ndarray]]:
    from torch.utils.data import DataLoader, Subset

    from pipeline.episodes import EpisodeCollate, build_eval_dataset, read_class_names

    rule, protocol, config = load(ck, device)
    mu, base, info = p1.load_or_build_bank(ck, BankRule(rule), data_path, device, bank_n)
    mu, base = mu.to(device), base.to(device)
    names = read_class_names(data_path, "s3dis")
    encode = clip_encoder(device)
    emb = {p: tp.class_embeddings([names[c] for c in range(N_CLASSES)], p, encode).to(device) for p in tp.PROMPT_SETS}
    base_idx = info["classes"]
    dataset = build_eval_dataset(data_path, "s3dis", ck.fold, 2, 1, mode=mode, seed=0)
    test_classes = [int(c) for c in np.asarray(dataset.classes)]
    view = dataset if max_episodes is None else Subset(dataset, range(min(max_episodes, len(dataset))))
    loader = DataLoader(view, batch_size=1, shuffle=False, collate_fn=EpisodeCollate(names))
    counts: Dict[str, List[np.ndarray]] = {}
    mech = {f"{s}_{p}": {"acc": [0, 0], "align": []} for s in SOURCES for p in tp.PROMPT_SETS}
    part_a = {"episodes": []}
    feat = rule.model.features

    def add(key, pred, gt, ep):
        counts.setdefault(key, []).append(p0.episode_counts(pred.cpu().numpy(), gt, ep.sampled_classes, test_classes))

    for (episode,) in loader:
        episode = episode.to(device)
        f_s, _ = feat.encode_episode(episode.support_x, episode.query_x)  # [N, K, P, D]
        f_q, m_eff, steps, logits = rule(episode)  # [B_q,P,D], [B_q,N+1,D], T+1 x [B_q,N+1,D], [B_q,P,N+1]
        p0.check_identity(f_q, m_eff, logits)
        labels = episode.query_y  # [B_q, P], diagnostics, oracle arms and Part A only
        gt = labels.cpu().numpy()
        oracle = tp.unit_oracle_logits(f_q, m_eff, labels)  # [B_q, P, N+1]
        add("model", logits.argmax(-1), gt, episode)
        add("oracle_unit", oracle.argmax(-1), gt, episode)
        support_logits = torch.einsum("bpd,bcd->bpc", f_q, steps[0])  # [B_q, P, N+1], P^0 without the head
        add("support_rule", support_logits.argmax(-1), gt, episode)
        w = {"one": torch.ones_like(labels, dtype=logits.dtype), "entropy": tp.entropy_weight(logits)}  # [B_q, P]
        novel = [int(c) for c in episode.sampled_classes]
        cache = {}
        for s, p in itertools.product(SOURCES, tp.PROMPT_SETS):
            t_hat = directions(s, emb[p], novel, base_idx, base)  # [N, D]
            t = tp.text_logits(f_q, mu, t_hat)  # [B_q, P, N]
            gamma, _ = tp.support_gamma(f_s, episode.support_y, mu, t_hat)  # scalar
            cache[(s, p)] = (t, gamma)
            right, total = tp.text_query_accuracy(t, labels)
            mech[f"{s}_{p}"]["acc"][0] += right
            mech[f"{s}_{p}"]["acc"][1] += total
            mech[f"{s}_{p}"]["align"].append(tp.alignment_terms(t, logits, oracle, labels))
        for a in arms:
            t, gamma = cache[(a["source"], a["prompt"])]
            add(name_of(a), tp.apply_prior(logits, t, a["kappa"], gamma, w[a["weight"]]).argmax(-1), gt, episode)
        if oracle_arm is not None:  # kappa per episode chosen with the query labels: an upper bound
            t, gamma = cache[(oracle_arm["source"], oracle_arm["prompt"])]
            best, best_acc = logits.argmax(-1), -1
            for k in (0.0,) + KAPPAS:
                pred = tp.apply_prior(logits, t, k, gamma, w[oracle_arm["weight"]]).argmax(-1)
                acc = int((pred == labels).sum())
                if acc > best_acc:
                    best, best_acc = pred, acc
            add("oracle_gamma", best, gt, episode)
        if mode == "test":  # Part A
            pred_m, pred_o = logits.argmax(-1), oracle.argmax(-1)
            fixable = (pred_m != labels) & (pred_o == labels)  # [B_q, P]
            boundary = tp.boundary_mask(episode.query_x[..., :3], labels)  # [B_q, P]
            fg = labels > 0
            part_a["episodes"].append({
                "support_fg": float(episode.support_y.float().mean()),
                "fixable": int(fixable.sum()), "points": int(labels.numel()),
                "fixable_boundary": int((fixable & boundary).sum()), "boundary": int(boundary.sum()),
                "errors_model": int((pred_m != labels).sum()),
                "errors_model_boundary": int(((pred_m != labels) & boundary).sum()),
                "fg_acc_model": tp.point_accuracy(pred_m, labels, fg), "fg_acc_oracle": tp.point_accuracy(pred_o, labels, fg)})
    stacked = {k: np.stack(v) for k, v in counts.items()}  # name -> [E, 3, C+1]
    result = {"checkpoint": vars(ck), "mode": mode, "protocol": protocol, "episodes": len(view),
              "test_classes": test_classes, "bank": info,
              "miou": {k: float(p0.miou_from_counts(v.sum(0))) for k, v in stacked.items()},
              "class_iou": {k: p0.class_ious(v.sum(0)).tolist() for k, v in stacked.items()},
              "mechanism": {k: {"text_query_acc": v["acc"][0] / max(v["acc"][1], 1), **alignment(v["align"])}
                            for k, v in mech.items()}}
    if mode == "test":
        result["part_a"] = summarise_part_a(part_a["episodes"], result["miou"])
    return result, stacked


def alignment(terms: List[Tuple[float, float, float]], boot: int = p0.BOOTSTRAP, seed: int = 0) -> Dict[str, float]:
    """Pooled cosine Σab / sqrt(Σa² Σb²) and its episode-bootstrap 95 % CI."""
    arr = np.array(terms, dtype=np.float64)  # [E, 3]
    if arr.size == 0:
        return {"align": float("nan"), "align_ci_low": float("nan"), "align_ci_high": float("nan")}

    def cos(s):
        return float(s[0] / np.sqrt(max(s[1] * s[2], 1e-30)))

    rng = np.random.default_rng(seed)
    w = rng.multinomial(len(arr), np.full(len(arr), 1.0 / len(arr)), size=boot).astype(np.float64)  # [B, E]
    sums = w @ arr  # [B, 3]
    cs = sums[:, 0] / np.sqrt(np.maximum(sums[:, 1] * sums[:, 2], 1e-30))
    return {"align": cos(arr.sum(0)), "align_ci_low": float(np.percentile(cs, 2.5)),
            "align_ci_high": float(np.percentile(cs, 97.5))}


def summarise_part_a(eps: List[Dict], miou: Dict[str, float]) -> Dict:
    s, m, o = (100.0 * miou[k] for k in ("support_rule", "model", "oracle_unit"))
    fix = sum(e["fixable"] for e in eps)
    pts = sum(e["points"] for e in eps)
    fix_b = sum(e["fixable_boundary"] for e in eps)
    bnd = sum(e["boundary"] for e in eps)
    enrich = (fix_b / max(fix, 1)) / max(bnd / max(pts, 1), 1e-12)
    sup = np.array([e["support_fg"] for e in eps])
    cuts = np.quantile(sup, [1 / 3, 2 / 3])
    tercile = np.digitize(sup, cuts)  # 0 small, 1 middle, 2 large support masks
    fix_share = [sum(e["fixable"] for e, t in zip(eps, tercile) if t == k) / max(fix, 1) for k in range(3)]
    gain = [(e["fg_acc_oracle"][0] - e["fg_acc_model"][0]) / max(e["fg_acc_model"][1], 1) for e in eps]
    rho = tp.spearman(list(sup), gain)
    return {"support_rule": s, "model": m, "oracle_unit": o,
            "head_recovery": (m - s) / (o - s) if o != s else float("nan"),
            "fixable_share_of_points": fix / max(pts, 1), "boundary_share_of_points": bnd / max(pts, 1),
            "boundary_share_of_fixable": fix_b / max(fix, 1), "boundary_enrichment": enrich,
            "boundary_share_of_model_errors": sum(e["errors_model_boundary"] for e in eps) /
                                              max(sum(e["errors_model"] for e in eps), 1),
            "fixable_share_by_support_tercile": fix_share, "support_tercile_cuts": cuts.tolist(),
            "spearman_oracle_gain_vs_support_fg": rho}


def save(result: Dict, stacked: Dict[str, np.ndarray], stem: str) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, stem + ".json"), "w") as f:
        json.dump(result, f, indent=1)
    np.savez_compressed(os.path.join(OUT_DIR, stem + "_counts.npz"), **stacked)


def select_arm(result: Dict) -> Dict:
    gains = {name_of(a): 100.0 * (result["miou"][name_of(a)] - result["miou"]["model"]) for a in grid()}
    best = max(grid(), key=lambda a: gains[name_of(a)])
    return {"frozen": best, "name": name_of(best), "valid_gain": gains[name_of(best)],
            "top10": dict(sorted(gains.items(), key=lambda kv: -kv[1])[:10])}


def sibling(frozen: Dict) -> Dict:
    return dict(frozen, weight="one" if frozen["weight"] == "entropy" else "entropy")


def cmd_select(args, device) -> int:
    ck = p0.parse_checkpoint(args.checkpoint)
    if ck.fold != 1:
        raise ValueError(f"{ck.name}: selection uses S1 only, S0 stays held out [DECISION D-22]")
    result, stacked = score(ck, args.data_path, "valid", grid(), device, args.bank_episodes, args.max_episodes)
    choice = select_arm(result)
    result["selection"] = choice
    save(result, stacked, f"select_{ck.name}")
    print(f"[select] model {result['miou']['model']:.4f} | frozen {choice['name']} ({choice['valid_gain']:+.2f})",
          flush=True)
    for k, v in sorted(result["mechanism"].items()):
        print(f"[select] mechanism {k:28s} text acc {v['text_query_acc']:.3f} | align {v['align']:+.3f} "
              f"[{v['align_ci_low']:+.3f}, {v['align_ci_high']:+.3f}]", flush=True)
    return 0


def cmd_test(args, device) -> int:
    ck = p0.parse_checkpoint(args.checkpoint)
    with open(os.path.join(OUT_DIR, f"select_{ck.name}.json")) as f:
        sel = json.load(f)
    frozen = sel["selection"]["frozen"]
    arms = [frozen, sibling(frozen)]
    result, stacked = score(ck, args.data_path, "test", arms, device, args.bank_episodes, args.max_episodes,
                            oracle_arm=frozen)
    names = [name_of(a) for a in arms]
    result["frozen"], result["sibling"] = names
    result["valid_mechanism"] = sel["mechanism"][f"{frozen['source']}_{frozen['prompt']}"]
    result["paired"] = {"frozen_vs_model": p0.paired_bootstrap(stacked["model"], stacked[names[0]]),
                        "frozen_vs_sibling": p0.paired_bootstrap(stacked[names[1]], stacked[names[0]]),
                        "oracle_gamma_vs_model": p0.paired_bootstrap(stacked["model"], stacked["oracle_gamma"]),
                        "support_rule_vs_model": p0.paired_bootstrap(stacked["model"], stacked["support_rule"]),
                        "oracle_unit_vs_model": p0.paired_bootstrap(stacked["model"], stacked["oracle_unit"])}
    save(result, stacked, f"test_{ck.name}")
    p = result["paired"]["frozen_vs_model"]
    print(f"[test] model {result['miou']['model']:.4f} -> {names[0]} {result['miou'][names[0]]:.4f} "
          f"({p['gain']:+.2f} [{p['ci_low']:+.2f}, {p['ci_high']:+.2f}]) | oracle-gamma "
          f"{result['paired']['oracle_gamma_vs_model']['gain']:+.2f}", flush=True)
    print("[test] part A " + json.dumps(result["part_a"]), flush=True)
    return 0


def ci_side(ci: Dict) -> str:
    return "above 0" if ci["ci_low"] > 0 else "below 0" if ci["ci_high"] < 0 else "contains 0"


def decide(test: Dict) -> List[Tuple[str, str]]:
    """Rules P3.0-P3.5 and the Part A interpretation bands of [DECISION D-31]."""
    v = []
    mech = test["valid_mechanism"]
    mech_ok = mech["text_query_acc"] >= MECH_ACC and mech["align_ci_low"] > 0
    v.append(("P3.0 mechanism", f"text acc {mech['text_query_acc']:.3f} (bar {MECH_ACC}), alignment "
                                f"{mech['align']:+.3f} [{mech['align_ci_low']:+.3f}, {mech['align_ci_high']:+.3f}]: "
                                + ("holds" if mech_ok else "fails, stop text on this feature space")))
    g = test["paired"]["frozen_vs_model"]
    o = test["paired"]["oracle_gamma_vs_model"]["gain"]
    if o < GO_GAIN:
        v.append(("P3.2 stop", f"oracle-gamma gain {o:+.2f} < +{GO_GAIN}"))
    elif mech_ok and g["gain"] >= GO_GAIN and g["ci_low"] > 0:
        v.append(("P3.1 go", f"frozen {g['gain']:+.2f} [{g['ci_low']:+.2f}, {g['ci_high']:+.2f}], oracle-gamma "
                             f"{o:+.2f}: a trained text prior gets its own decision"))
    else:
        v.append(("P3.3 in between", f"frozen {g['gain']:+.2f} ({ci_side(g)}), oracle-gamma {o:+.2f}"))
    if test["frozen"].split("_")[-2] == "entropy":
        s = test["paired"]["frozen_vs_sibling"]
        v.append(("P3.4 entropy weight", f"{s['gain']:+.2f} [{s['ci_low']:+.2f}, {s['ci_high']:+.2f}]: "
                                         + ("claimable" if s["ci_low"] > 0 else "not claimable")))
    else:
        v.append(("P3.4 entropy weight", f"not claimable: selection froze {test['frozen']}"))
    drop = [round(100 * (b - a), 2) for a, b in zip(test["class_iou"]["model"][1:], test["class_iou"][test["frozen"]][1:])]
    if g["gain"] > 0 and min(drop) < COLLAPSE:
        v.append(("P3.5 collapse watch", f"per-class change {drop}"))
    a = test["part_a"]
    band = ("boundary-enriched: a local refinement problem" if a["boundary_enrichment"] >= BOUNDARY_ENRICHED else
            "region-level confusions: a global prototype shift" if a["boundary_enrichment"] < REGION_LEVEL else
            "between the bands")
    v.append(("A head recovery", f"support rule {a['support_rule']:.2f} -> E1 {a['model']:.2f} -> oracle "
                                 f"{a['oracle_unit']:.2f}: the head recovers {a['head_recovery']:.2f} of the gap"))
    v.append(("A fixable points", f"{a['fixable_share_of_points']:.3f} of points; boundary share "
                                  f"{a['boundary_share_of_fixable']:.3f} vs {a['boundary_share_of_points']:.3f} of all "
                                  f"points, enrichment {a['boundary_enrichment']:.2f}: {band}"))
    rho = a["spearman_oracle_gain_vs_support_fg"]
    v.append(("A support quality", f"Spearman(oracle gain, support foreground fraction) {rho:+.2f}; fixable share "
                                   f"by support tercile small/mid/large {[round(x, 3) for x in a['fixable_share_by_support_tercile']]}"
                                   + (": support quality matters" if rho <= SUPPORT_RHO else "")))
    return v


def cmd_decide(args) -> int:
    path = os.path.join(OUT_DIR, f"test_{args.name}.json")
    with open(path) as f:
        test = json.load(f)
    for rule, text in decide(test):
        print(f"{rule:22s} {text}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["select", "test", "decide"])
    p.add_argument("--data_path")
    p.add_argument("--checkpoint", help="name:ours:1:path of E1")
    p.add_argument("--name", default="e1", help="decide: the checkpoint name used in select/test")
    p.add_argument("--bank_episodes", type=int, default=p1.BANK_EPISODES)
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
