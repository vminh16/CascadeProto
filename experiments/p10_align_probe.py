"""P10.1 of [DECISION D-46]: is there a base-class prototype gap left on the training episodes? No training.

    python experiments/p10_align_probe.py base --data_path datasets/S3DIS/blocks_bs1_s1 \
        --checkpoint cr:ours:1:log_d37/s3dis_S1_N2_K1_point_T4_vip_clean_b1_qrandom/last.pt
    python experiments/p10_align_probe.py gate --name cr      # P10.1: exit 0 holds (part B runs), 3 fails
    python experiments/p10_align_probe.py decide              # D46.1-D46.4 after part B's trainings and tests
    python experiments/p10_align_probe.py stats --data_path ... --checkpoint cr:...   # Σ_w, Σ_η (amendment 1)
    python experiments/p10_align_probe.py metric --data_path ... --checkpoint cr:...  # P10.4
    python experiments/p10_align_probe.py kcurve --data_path ... --checkpoint cr:...  # P10.3
    python experiments/p10_align_probe.py amend                                        # P10.3 band, P10.4 rule

On P1's seeded training episodes (no augmentation, the fold's base classes scored), U's rule, the cosine oracle
(the query's own unit class directions in every present row) and the model; the gap oracle − U with its paired
bootstrap, and the mean prototype-alignment loss of `models/proto_align.py` at τ = 0.1. A small gap means CE already
aligns the base-class prototypes, and an alignment loss would have nothing to act on (D-29's reading).

Amendment 1: P10.4 scores the support's unit-feature means under C = shrink(Σ_w + Σ_η / K, λ), with Σ_w and Σ_η
from the base classes of the same training episodes (the Bayes rule for a prototype estimated from K instances);
P10.3 scores 2-way 5-shot test episodes with their first k shots and fits the prototype error e(k) = a + c / k.
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
from experiments import p1_bpc_probe as p1  # noqa: E402
from experiments import p5_condition_probe as p5  # noqa: E402
from experiments import p6_prototype_probe as p6  # noqa: E402
from experiments import p9_placement_probe as p9  # noqa: E402
from models.oracle_distill import oracle_directions  # noqa: E402
from models.proto_align import TAU, alignment_loss  # noqa: E402

OUT_DIR = "results/phase16_d46"
EPISODES = 1000  # P1's bank size of training episodes
GAP_BAR = 5.0  # P10.1: oracle − U on the base classes, points [DECISION D-46]
RULES = ("model", "U", "oracle")
ARMS = ("m5a", "m5b")  # λ 0.25, 1.0 [DECISION D-46]
CR_VALID_GAP = 24.88  # CR's cosine oracle − U on valid, 80.84 − 55.96 (P9, results/phase16_p9/SUMMARY.md)
MECH_DROP = 2.0  # D46.1: the arm's valid gap must shrink by at least this much [DECISION D-46]
BASE_GAIN = 1.0  # D46.2
# Amendment 1 [DECISION D-46]
LAMBDAS = (0.1, 0.3, 0.5, 0.7, 0.9, 1.0)  # shrinkage of the base-class metric; 1.0 = isotropic control
KS = (1, 2, 3, 5)  # shots of the K-curve, nested in 5-shot episodes
K_MAX, K_SEED = 5, 0
METRIC_GAIN = 0.5  # P10.4, P9's bar for a label-free rule
BIAS_HIGH, BIAS_LOW = 0.5, 0.25  # P10.3 bands of the bias share
METRIC_DRAWS = ("valid",) + tuple(d39.DRAWS[:4])  # selection draw, then fixed100 and random600 seeds 0-2


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


# ------------------------------------------------------------------ amendment 1, pure

class ClassStats:
    """Σ_w (pooled within-class covariance of unit features around their block's class mean) and Σ_η (pooled
    covariance of a class's block means across blocks), float64, from blocks' base-class foregrounds."""

    def __init__(self, d: int, min_points: int = p6.MIN_POINTS):
        self.rr = torch.zeros(d, d, dtype=torch.float64)
        self.n_points, self.min_points = 0, min_points
        self.means: Dict[int, List[torch.Tensor]] = {}

    def add(self, u: torch.Tensor, mask: torch.Tensor, cls: int) -> None:
        """u [P, D] unit features of one block, mask [P] bool of class `cls`."""
        if int(mask.sum()) < self.min_points:
            return
        x = u[mask].double().cpu()  # [n, D]
        m = x.mean(0)  # [D]
        r = x - m  # [n, D]
        self.rr += r.T @ r
        self.n_points += x.shape[0]
        self.means.setdefault(int(cls), []).append(m)

    def finalize(self) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        sigma_w = self.rr / max(self.n_points, 1)  # [D, D]
        eta, n = torch.zeros_like(self.rr), 0
        for ms in self.means.values():
            if len(ms) < 2:
                continue
            r = torch.stack(ms) - torch.stack(ms).mean(0)  # [n_c, D]
            eta += r.T @ r
            n += r.shape[0]
        sigma_eta = eta / max(n, 1)  # [D, D]
        info = {"points": self.n_points, "instances": {str(c): len(v) for c, v in sorted(self.means.items())},
                "trace_w": float(sigma_w.trace()), "trace_eta": float(sigma_eta.trace()),
                "pr_w": p9.participation_ratio(sigma_w), "pr_eta": p9.participation_ratio(sigma_eta)}
        return sigma_w, sigma_eta, info


def metric_cov(sigma_w: torch.Tensor, sigma_eta: torch.Tensor, k: int, lam: float) -> torch.Tensor:
    """C = shrink(Σ_w + Σ_η / k, λ) [D, D]: the covariance of f − m̂ for a mean estimated from k instances."""
    return p9.shrink(sigma_w + sigma_eta / k, lam)


def metric_logits(f_q: torch.Tensor, f_s: torch.Tensor, support_y: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
    """Shared-metric scores of the support's unit-feature means [B_q, P, N+1] (argmax = nearest in C⁻¹)."""
    u = F.normalize(f_q, dim=-1)  # [B_q, P, D]
    m_s = p9.support_means(f_s, support_y)  # [N+1, D]
    return torch.stack([p9.lda_logits(u[b], m_s, cov.to(u.device)) for b in range(u.shape[0])])


def fit_curve(ks, es) -> Dict[str, float]:
    """Least squares e(k) = a + c / k; bias share β = max(a, 0) / e(1)."""
    k = np.asarray(ks, dtype=np.float64)
    e = np.asarray(es, dtype=np.float64)
    x = np.stack([np.ones_like(k), 1.0 / k], axis=1)  # [n, 2]
    (a, c), *_ = np.linalg.lstsq(x, e, rcond=None)
    fit = x @ np.array([a, c])
    ss = float(((e - e.mean()) ** 2).sum())
    r2 = 1.0 - float(((e - fit) ** 2).sum()) / ss if ss > 0 else 1.0
    e1 = float(e[list(k).index(1.0)]) if 1.0 in k else float(a + c)
    return {"a": float(a), "c": float(c), "r2": r2, "bias_share": max(float(a), 0.0) / e1 if e1 > 0 else 0.0}


def bias_band(beta: float) -> str:
    """P10.3 bands [DECISION D-46 amendment 1]."""
    if beta >= BIAS_HIGH:
        return "bias-dominated: training that removes the systematic shift is the lever"
    if beta <= BIAS_LOW:
        return "variance-dominated: the 1-shot gap is the single support instance; read the goal per shot count"
    return "mixed"


def subset_shots(e, k: int):
    """The episode with its first k support shots (same queries)."""
    from pipeline.episodes import Episode

    if not 1 <= k <= e.k_shot:
        raise ValueError(f"k must be in 1..{e.k_shot}, got {k}")
    return Episode(support_x=e.support_x[:, :k], support_y=e.support_y[:, :k], query_x=e.query_x,
                   query_y=e.query_y, sampled_classes=e.sampled_classes, class_names=e.class_names)


def prototype_errors(f_q: torch.Tensor, f_s: torch.Tensor, support_y: torch.Tensor,
                     query_y: torch.Tensor) -> List[float]:
    """1 − cos(p̂_c, μ_bc) for every foreground class c present in query block b (U's rows, oracle directions)."""
    rows = p5.support_directions(f_s, support_y)  # [N+1, D]
    o, present = oracle_directions(f_q, query_y, rows.shape[0])  # [B_q, N+1, D], [B_q, N+1]
    cos = torch.einsum("bcd,cd->bc", o, rows)  # [B_q, N+1]
    return [1.0 - float(cos[b, c]) for b in range(o.shape[0]) for c in range(1, rows.shape[0]) if present[b, c]]


def metric_verdict(p_fx: Dict[str, float], rand: List[float]) -> bool:
    """P10.4: the frozen metric − U holds at +0.5."""
    return d39.holds(p_fx, rand, METRIC_GAIN)


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


# ------------------------------------------------------------------ amendment 1, GPU passes

@torch.no_grad()
def base_statistics(rule, data_path: str, device, n: int, max_episodes: Optional[int]):
    """Σ_w, Σ_η from the base-class foregrounds of P10.1's training episodes (query and support blocks)."""
    from pipeline.episodes import make_episode, read_class_names

    names = read_class_names(data_path, "s3dis")
    ds = p1.training_episodes(data_path, 1, n)
    feats = p5.module_of(rule).features
    stats = None
    m = len(ds) if max_episodes is None else min(max_episodes, len(ds))
    for i in range(m):
        e = make_episode(ds[i], names).to(device)
        f_s, f_q = feats.encode_episode(e.support_x, e.query_x)  # [N, K, P, D], [B_q, P, D]
        stats = stats or ClassStats(f_q.shape[-1])
        u_q, u_s = F.normalize(f_q, dim=-1), F.normalize(f_s, dim=-1)
        for b in range(u_q.shape[0]):
            for w in range(e.n_way):
                stats.add(u_q[b], e.query_y[b] == w + 1, int(e.sampled_classes[w]))
        for w in range(e.n_way):
            for k in range(e.k_shot):
                stats.add(u_s[w, k], e.support_y[w, k] == 1, int(e.sampled_classes[w]))
    return stats.finalize()


def load_stats(out_dir: str, name: str, tag: str) -> Tuple[torch.Tensor, torch.Tensor]:
    saved = torch.load(os.path.join(out_dir, f"p10_stats_{name}{tag}.pt"), weights_only=True)
    return saved["sigma_w"], saved["sigma_eta"]


@torch.no_grad()
def score_metric(rule, draw: str, data_path: str, device, sigma_w, sigma_eta, max_episodes) -> Tuple[Dict, Dict]:
    """U and the base-class metric at every λ (K = 1) on one draw."""
    from pipeline.episodes import make_episode, read_class_names

    names = read_class_names(data_path, "s3dis")
    items, test_classes = p6.episodes(draw, data_path, max_episodes)
    covs = {lam: metric_cov(sigma_w, sigma_eta, 1, lam).to(device) for lam in LAMBDAS}
    counts: Dict[str, List[np.ndarray]] = {}
    for item in items:
        e = make_episode(item, names).to(device)
        f_q, m_eff, _, logits = rule(e)
        p0.check_identity(f_q, m_eff, logits)
        f_s = p5.support_features(rule, e)
        preds = {"U": p6.rule_logits(f_q, p6.base_rows(f_q, f_s, e.support_y)).argmax(-1)}
        for lam, cov in covs.items():
            preds[f"maha_{lam}"] = metric_logits(f_q, f_s, e.support_y, cov).argmax(-1)
        gt = e.query_y.cpu().numpy()
        for k, pr in preds.items():
            counts.setdefault(k, []).append(p0.episode_counts(pr.cpu().numpy(), gt, e.sampled_classes, test_classes))
    stacked = {k: np.stack(v) for k, v in counts.items()}
    return {"draw": draw, "episodes": len(stacked["U"]), "test_classes": test_classes,
            "miou": {k: float(p0.miou_from_counts(v.sum(0))) for k, v in stacked.items()}}, stacked


def frozen_lambda(valid: Dict) -> float:
    """The λ < 1 with the highest valid mIoU (λ = 1 is the isotropic control, never selected)."""
    return max((lam for lam in LAMBDAS if lam < 1.0), key=lambda lam: valid["miou"][f"maha_{lam}"])


@torch.no_grad()
def score_kcurve(rule, data_path: str, device, sigma_w, sigma_eta, lam: float, max_episodes) -> Tuple[Dict, Dict]:
    """2-way 5-shot test episodes scored with their first k shots: U, model, cosine oracle, the frozen metric, and
    the prototype error e(k)."""
    from pipeline.episodes import build_eval_dataset, make_episode, read_class_names

    names = read_class_names(data_path, "s3dis")
    ds = build_eval_dataset(data_path, "s3dis", 1, 2, K_MAX, mode="test", seed=K_SEED)
    test_classes = [int(c) for c in np.asarray(ds.classes)]
    m = len(ds) if max_episodes is None else min(max_episodes, len(ds))
    counts: Dict[str, List[np.ndarray]] = {}
    errors: Dict[int, List[float]] = {k: [] for k in KS}
    for i in range(m):
        full = make_episode(ds[i], names).to(device)
        gt = full.query_y.cpu().numpy()
        for k in KS:
            e = subset_shots(full, k)
            f_q, m_eff, _, logits = rule(e)
            p0.check_identity(f_q, m_eff, logits)
            f_s = p5.support_features(rule, e)  # [N, k, P, D]
            rows = p6.base_rows(f_q, f_s, e.support_y)
            cov = metric_cov(sigma_w, sigma_eta, k, lam).to(device)
            preds = {"model": logits.argmax(-1), "U": p6.rule_logits(f_q, rows).argmax(-1),
                     "oracle": p6.rule_logits(f_q, p6.oracle_rows(f_q, e.query_y, rows, "all")).argmax(-1),
                     "maha": metric_logits(f_q, f_s, e.support_y, cov).argmax(-1)}
            for r, pr in preds.items():
                counts.setdefault(f"{r}_k{k}", []).append(
                    p0.episode_counts(pr.cpu().numpy(), gt, e.sampled_classes, test_classes))
            errors[k].extend(prototype_errors(f_q, f_s, e.support_y, e.query_y))
    stacked = {k: np.stack(v) for k, v in counts.items()}
    miou = {k: float(p0.miou_from_counts(v.sum(0))) for k, v in stacked.items()}
    e_mean = [float(np.mean(errors[k])) for k in KS]
    gap = [100.0 * (miou[f"oracle_k{k}"] - miou[f"U_k{k}"]) for k in KS]
    return {"episodes": m, "shots": list(KS), "test_classes": test_classes, "lambda": lam, "miou": miou,
            "proto_error": e_mean, "fit_error": fit_curve(KS, e_mean), "miou_gap": gap,
            "fit_gap": fit_curve(KS, gap)}, stacked


def amend(out_dir: str, tag: str) -> List[Tuple[str, str]]:
    """P10.3 band and P10.4 rule [DECISION D-46 amendment 1]."""
    v = []
    kpath = os.path.join(out_dir, f"p10_kcurve{tag}.json")
    if os.path.isfile(kpath):
        kc = json.load(open(kpath))
        f = kc["fit_error"]
        shots = kc["shots"]
        v.append(("P10.3 " + bias_band(f["bias_share"]),
                  f"e(k) {[round(x, 4) for x in kc['proto_error']]} for k {shots}; a {f['a']:.4f}, c {f['c']:.4f}, "
                  f"R2 {f['r2']:.3f}, bias share {f['bias_share']:.2f}; U by k "
                  f"{[round(100 * kc['miou']['U_k%d' % k], 2) for k in shots]}, oracle "
                  f"{[round(100 * kc['miou']['oracle_k%d' % k], 2) for k in shots]}, metric "
                  f"{[round(100 * kc['miou']['maha_k%d' % k], 2) for k in shots]}"))
    else:
        v.append(("incomplete", "P10.3: no K-curve"))
    draws = {}
    for d in METRIC_DRAWS:
        path = os.path.join(out_dir, f"p10_metric_{d.replace(':', '_seed')}{tag}.json")
        if os.path.isfile(path):
            draws[d] = (json.load(open(path)), dict(np.load(path.replace(".json", "_counts.npz"))))
    missing = [d for d in METRIC_DRAWS if d not in draws]
    if missing:
        return v + [("incomplete", f"P10.4: missing draws {missing}")]
    lam = frozen_lambda(draws["valid"][0])
    p_fx, rand = d39.paired(draws, "U", f"maha_{lam}")
    ok = metric_verdict(p_fx, rand)
    iso = 100.0 * (draws["fixed100"][0]["miou"]["maha_1.0"] - draws["fixed100"][0]["miou"]["U"])
    v.append((f"P10.4 {'holds' if ok else 'fails'} (lambda {lam:g})",
              f"metric - U fixed100 {p_fx['gain']:+.2f} [{p_fx['ci_low']:+.2f}, {p_fx['ci_high']:+.2f}]; random600 "
              f"{[round(g, 2) for g in rand]}; isotropic control {iso:+.2f}"))
    return v


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["base", "gate", "decide", "stats", "metric", "kcurve", "amend"])
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
    if args.stage == "amend":
        for name, text in amend(out_dir, args.tag):
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
    if args.stage == "stats":
        sigma_w, sigma_eta, info = base_statistics(rule, args.data_path, device, args.episodes, args.max_episodes)
        os.makedirs(out_dir, exist_ok=True)
        torch.save({"sigma_w": sigma_w, "sigma_eta": sigma_eta},
                   os.path.join(out_dir, f"p10_stats_{ck.name}{args.tag}.pt"))
        p6.save(info, None, f"p10_stats_{ck.name}{args.tag}", out_dir)
        print(f"[stats] {ck.name}: " + json.dumps(info), flush=True)
        return 0
    if args.stage == "metric":
        sigma_w, sigma_eta = load_stats(out_dir, ck.name, args.tag)
        for d in METRIC_DRAWS:
            res, stacked = score_metric(rule, d, args.data_path, device, sigma_w, sigma_eta, args.max_episodes)
            p6.save(res, stacked, f"p10_metric_{d.replace(':', '_seed')}{args.tag}", out_dir)
            print(f"[metric] {d}: " + " | ".join(f"{k} {100 * v:.2f}" for k, v in res["miou"].items()), flush=True)
        return 0
    if args.stage == "kcurve":
        sigma_w, sigma_eta = load_stats(out_dir, ck.name, args.tag)
        with open(os.path.join(out_dir, f"p10_metric_valid{args.tag}.json")) as f:
            valid = json.load(f)
        res, stacked = score_kcurve(rule, args.data_path, device, sigma_w, sigma_eta, frozen_lambda(valid),
                                    args.max_episodes)
        p6.save(res, stacked, f"p10_kcurve{args.tag}", out_dir)
        print(f"[kcurve] e(k) {res['proto_error']} fit {res['fit_error']} | "
              + " | ".join(f"{k} {100 * v:.2f}" for k, v in res["miou"].items()), flush=True)
        return 0
    res, stacked = score_base(rule, args.data_path, device, args.episodes, args.max_episodes)
    res["checkpoint"] = vars(ck)
    p6.save(res, stacked, f"p10_base_{ck.name}{args.tag}", out_dir)
    print(f"[p10] {ck.name}: base U {100 * res['miou']['U']:.2f} | oracle {100 * res['miou']['oracle']:.2f} | "
          f"model {100 * res['miou']['model']:.2f} | gap {res['gap']['gain']:+.2f} | align loss "
          f"{res['align_loss_mean']:.3f} | {res['episodes']} episodes", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
