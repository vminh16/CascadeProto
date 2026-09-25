"""P7 of [DECISION D-40]: label propagation on the query's own point graph, seeded by D-39's U + both prediction,
with its preconditions measured (P7a). Inference only, CR `last.pt` (D-37's base: `stage_type=vip_clean`, random
query order), S1.

  Y0       one-hot of the U + both prediction (D39.1's rule on U's rows)
  lp_*     Z = (1 - beta)(I - beta S)^-1 Y0, S = D^-1/2 (A + Aᵀ) D^-1/2 on a kNN graph of one query block
           [Zhou et al., NIPS 2003; Iscen et al., CVPR 2019, Eq. 9]; graph xyz / xyzf / feat x k x beta = 24 arms
  P7a      per class and sampling condition: homophily, local seed recall and false rate around missed points,
           the one-hop vote (fixable / breakable points), same-class reachability of a correct seed

    python experiments/p7_propagation_probe.py select --data_path datasets/S3DIS/blocks_bs1_s1 --checkpoint cr:ours:1:<last.pt>
    python experiments/p7_propagation_probe.py test   --data_path datasets/S3DIS/blocks_bs1_s1 --checkpoint cr:ours:1:<last.pt>
    python experiments/p7_propagation_probe.py decide
"""

import argparse
import itertools
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
from experiments import r2_distill_eval as r2  # noqa: E402

OUT_DIR = "results/phase16_p7"
D39_FIXED = "results/phase16_d39/test_S1_fixed100.json"
P6_SELECT = "results/phase16_p6/select.json"
DRAWS = p6.TEST_DRAWS  # fixed100, random600:0-2, leakfree
GRAPHS, KS, BETAS = ("xyz", "xyzf", "feat"), (8, 16), (0.5, 0.8, 0.9, 0.99)  # [DECISION D-40]
GAMMA = 3  # a_ij = [u_iᵀu_j]_+^gamma [Iscen et al., CVPR 2019, Eq. 9 and §5]
GATE_GAIN, ADOPT_GAIN, TRAIN_GAIN = 0.5, 0.5, 1.0  # P7.1, P7.2 / P7.4, P7.3
RESIDUAL_TOL = 1e-8  # ‖(I - beta S) Z - (1 - beta) Y0‖_inf per block, float64
REF_TOL = p6.CR_TOL  # 0.01 points against D-37's CR, D-39's U and U + both, P6's valid numbers
OWN, OTHER = 0, 1  # condition axis of the P7a arrays (D-35); the background (column 0) is counted under OWN
DIAG_STATS = ("n", "hit", "iso", "p_hit", "p_miss", "a_miss", "n_a_miss", "b_miss", "n_b_miss", "fixable",
              "breakable", "reach_miss")
ARM_STATS = ("fixed", "broken", "p_fixed", "p_broken")


# ------------------------------------------------------------------ arms

def lp_arms() -> List[Dict]:
    """The 24 propagation arms in grid order (ties in the selection go to the earlier one)."""
    return [dict(graph=g, k=k, beta=b) for g, k, b in itertools.product(GRAPHS, KS, BETAS)]


def lp_name(arm: Dict) -> str:
    return f"lp_{arm['graph']}_k{arm['k']}_b{arm['beta']:g}"


def arm_of(name: str) -> Dict:
    by_name = {lp_name(a): a for a in lp_arms()}
    if name not in by_name:
        raise ValueError(f"unknown propagation arm {name!r}")
    return by_name[name]


def graph_keys() -> List[Tuple[str, int]]:
    return list(itertools.product(GRAPHS, KS))


def graph_name(graph: str, k: int) -> str:
    return f"{graph}_k{k}"


# ------------------------------------------------------------------ graph and propagation (pure, per query block)

def knn_graph(xyz: torch.Tensor, u: torch.Tensor, graph: str, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Neighbour indices [B, P, k] and affinities a_ij [B, P, k] within each query block [DECISION D-40].

    xyz [B, P, 3] metric coordinates, u [B, P, D] unit features. `xyz`: nearest in space, a = 1; `xyzf`: the same
    neighbours, a = [u_iᵀu_j]_+^3; `feat`: nearest by cosine, a = [u_iᵀu_j]_+^3. A point is never its own neighbour.
    """
    if graph not in GRAPHS:
        raise ValueError(f"graph must be one of {GRAPHS}, got {graph!r}")
    n_points = xyz.shape[1]
    if not 0 < k < n_points:
        raise ValueError(f"k = {k} needs 0 < k < {n_points}")
    sim = torch.einsum("bpd,bqd->bpq", u, u)  # [B, P, P]
    eye = torch.eye(n_points, dtype=torch.bool, device=xyz.device)  # [P, P]
    if graph == "feat":
        idx = sim.masked_fill(eye, float("-inf")).topk(k, dim=-1).indices  # [B, P, k]
    else:
        dist = torch.cdist(xyz, xyz, compute_mode="donot_use_mm_for_euclid_dist")  # [B, P, P]
        idx = dist.masked_fill(eye, float("inf")).topk(k, dim=-1, largest=False).indices  # [B, P, k]
    if graph == "xyz":
        return idx, torch.ones(idx.shape, dtype=u.dtype, device=u.device)
    return idx, sim.gather(-1, idx).clamp_min(0.0) ** GAMMA  # [B, P, k]


def affinity(idx: torch.Tensor, a: torch.Tensor, n_points: int) -> torch.Tensor:
    """W = A + Aᵀ [B, P, P], A_ij = a_ij for j in NN_k(i): symmetric, zero diagonal [Iscen et al., Eq. 9]."""
    w = torch.zeros(idx.shape[0], n_points, n_points, dtype=a.dtype, device=a.device)  # [B, P, P]
    w.scatter_(2, idx, a)
    return w + w.transpose(1, 2)


def normalized(w: torch.Tensor) -> torch.Tensor:
    """S = D^-1/2 W D^-1/2 [B, P, P]; an isolated point (zero degree) gets a zero row and column [Zhou et al.]."""
    deg = w.sum(dim=-1)  # [B, P]
    inv = torch.where(deg > 0, deg.clamp_min(torch.finfo(w.dtype).tiny).rsqrt(), torch.zeros_like(deg))  # [B, P]
    return inv.unsqueeze(-1) * w * inv.unsqueeze(-2)


def spread(s: torch.Tensor, y0: torch.Tensor, beta: float) -> torch.Tensor:
    """Z = (1 - beta)(I - beta S)^-1 Y0 [B, P, C], the fixed point of F <- beta S F + (1 - beta) Y0 [Zhou et al.].

    Solved exactly by Cholesky (I - beta S is positive definite for 0 <= beta < 1); the residual is checked.
    """
    if not 0.0 <= beta < 1.0:
        raise ValueError(f"beta must lie in [0, 1), got {beta}")
    eye = torch.eye(s.shape[-1], dtype=s.dtype, device=s.device)  # [P, P]
    m = eye - beta * s  # [B, P, P]
    rhs = (1.0 - beta) * y0  # [B, P, C]
    z = torch.cholesky_solve(rhs, torch.linalg.cholesky(m))  # [B, P, C]
    residual = (m @ z - rhs).abs().max().item()
    if residual > RESIDUAL_TOL:
        raise RuntimeError(f"label spreading residual {residual:.3e} > {RESIDUAL_TOL:.0e} (beta {beta})")
    return z


def both_logits(f_q: torch.Tensor, f_s: torch.Tensor, support_y: torch.Tensor) -> torch.Tensor:
    """D-39's U + both [B_q, P, N+1]: P6's background self-support on U's rows, then the 3-component background."""
    rows = p6.base_rows(f_q, f_s, support_y)  # [B_q, N+1, D]
    adapted = p6.self_support(f_q, rows, d39.SSP_BG["rho"], d39.SSP_BG["alpha"], d39.SSP_BG["steps"],
                              d39.SSP_BG["which"])  # [B_q, N+1, D]
    cent = p6.spherical_kmeans(p6.background_units(f_s, support_y), d39.KM)  # [3, D]
    return d39.km_logits(f_q, adapted, cent)


# ------------------------------------------------------------------ P7a diagnostics (pure)

def point_stats(w: torch.Tensor, y: torch.Tensor, seed: torch.Tensor, n_cls: int) -> Dict[str, torch.Tensor]:
    """Per-point P7a quantities [B, P] on one graph W [B, P, P]; y the labels, seed Y0's prediction [DECISION D-40].

    p: share of the point's edge weight on its own class (0 when isolated); a: share of its same-class weight
    predicted y_i (defined when that weight is > 0); b: share of its other-class weight predicted y_i (defined when
    > 0); fixable / breakable: the one-hop vote Σ_j W_ij [ŷ_j = c] has y_i strictly above / strictly below the best
    other class, for a missed / hit point.
    """
    oy = F.one_hot(y, n_cls).to(w.dtype)  # [B, P, C]
    oh = F.one_hot(seed, n_cls).to(w.dtype)  # [B, P, C]

    def at_label(m: torch.Tensor) -> torch.Tensor:  # (W m)[i, y_i]  [B, P]
        return torch.einsum("bpq,bqc->bpc", w, m).gather(-1, y.unsqueeze(-1)).squeeze(-1)

    same, other = at_label(oy), at_label(1.0 - oy)  # [B, P], [B, P]
    same_hit, other_pred = at_label(oy * oh), at_label((1.0 - oy) * oh)  # [B, P], [B, P]
    votes = torch.einsum("bpq,bqc->bpc", w, oh)  # [B, P, C]
    v_y = votes.gather(-1, y.unsqueeze(-1)).squeeze(-1)  # [B, P]
    rival = votes.scatter(-1, y.unsqueeze(-1), float("-inf")).max(dim=-1).values  # [B, P]
    deg = w.sum(dim=-1)  # [B, P]
    iso, hit = deg <= 0, seed == y  # [B, P], [B, P]
    zero = torch.zeros_like(deg)
    return {"hit": hit, "iso": iso,
            "p": torch.where(iso, zero, same / deg.clamp_min(torch.finfo(w.dtype).tiny)),
            "has_a": same > 0, "a": torch.where(same > 0, same_hit / same.clamp_min(torch.finfo(w.dtype).tiny), zero),
            "has_b": other > 0,
            "b": torch.where(other > 0, other_pred / other.clamp_min(torch.finfo(w.dtype).tiny), zero),
            "fixable": ~hit & ~iso & (v_y > rival), "breakable": hit & ~iso & (rival > v_y)}


def components(idx: torch.Tensor, a: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """[B, P] component id of every point in its block's same-class graph (edges j in NN_k(i), a_ij > 0, y_i = y_j,
    undirected): the smallest point id of the component, by min-label propagation with pointer jumping."""
    b_q, n_points, k = idx.shape
    offset = (torch.arange(b_q, device=idx.device) * n_points).view(b_q, 1, 1)  # [B, 1, 1]
    rows = (torch.arange(n_points, device=idx.device).view(1, n_points, 1) + offset).expand(b_q, n_points, k)
    cols = idx + offset  # [B, P, k]
    flat_y = y.reshape(-1)  # [B*P]
    keep = (a > 0).reshape(-1) & (flat_y[rows.reshape(-1)] == flat_y[cols.reshape(-1)])  # [B*P*k]
    r, c = rows.reshape(-1)[keep], cols.reshape(-1)[keep]  # [E], [E]
    lab = torch.arange(b_q * n_points, device=idx.device)  # [B*P]
    while True:
        new = lab.clone()
        new.scatter_reduce_(0, r, lab[c], reduce="amin")
        new.scatter_reduce_(0, c, lab[r], reduce="amin")
        new = new[new]  # pointer jumping: every label is a point of the same component with a smaller or equal id
        if torch.equal(new, lab):
            return (lab - offset.view(b_q, 1).expand(b_q, n_points).reshape(-1)).view(b_q, n_points)
        lab = new


def reachable(idx: torch.Tensor, a: torch.Tensor, y: torch.Tensor, hit: torch.Tensor) -> torch.Tensor:
    """[B, P] bool: the point's same-class component holds a correctly predicted point."""
    comp = components(idx, a, y)  # [B, P]
    seeded = torch.zeros_like(comp).scatter_reduce(1, comp, hit.long(), reduce="amax")  # [B, P] by component id
    return seeded.gather(1, comp) > 0


def diag_values(st: Dict[str, torch.Tensor], reach: torch.Tensor) -> Dict[str, torch.Tensor]:
    """The DIAG_STATS summands per point [B, P]."""
    hit, miss = st["hit"], ~st["hit"]
    f = lambda m: m.to(torch.float64)  # noqa: E731
    return {"n": torch.ones_like(st["p"]), "hit": f(hit), "iso": f(st["iso"]), "p_hit": st["p"] * f(hit),
            "p_miss": st["p"] * f(miss), "a_miss": st["a"] * f(miss & st["has_a"]), "n_a_miss": f(miss & st["has_a"]),
            "b_miss": st["b"] * f(miss & st["has_b"]), "n_b_miss": f(miss & st["has_b"]),
            "fixable": f(st["fixable"]), "breakable": f(st["breakable"]), "reach_miss": f(reach & miss)}


def arm_values(st: Dict[str, torch.Tensor], after: torch.Tensor, y: torch.Tensor) -> Dict[str, torch.Tensor]:
    """The ARM_STATS summands per point [B, P]: missed points the arm fixes, hit points it breaks."""
    fixed = (~st["hit"] & (after == y)).to(torch.float64)  # [B, P]
    broken = (st["hit"] & (after != y)).to(torch.float64)  # [B, P]
    return {"fixed": fixed, "broken": broken, "p_fixed": st["p"] * fixed, "p_broken": st["p"] * broken}


def point_columns(y: np.ndarray, label2class, test_classes: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Column (0 background, 1 + index of the test class) and condition (OWN / OTHER of D-35) of every point [B, P].

    One query block per way, block b sampled for local class b + 1 [VIPSEG dataloaders/loader.py:181-222].
    """
    n_way = len(label2class)
    if y.shape[0] != n_way:
        raise ValueError(f"P7 needs one query block per way, got {y.shape[0]} blocks for {n_way} ways")
    lut = np.array([0] + [test_classes.index(int(c)) + 1 for c in label2class])  # [N+1]
    own = np.arange(1, n_way + 1).reshape(n_way, 1)  # [B, 1]
    return lut[y], np.where((y == 0) | (y == own), OWN, OTHER)


def accumulate(out: np.ndarray, col: np.ndarray, cond: np.ndarray, values: Dict[str, torch.Tensor],
               stats: Tuple[str, ...]) -> None:
    """out [S, C+1, 2] += the per-point values summed by (column, condition)."""
    for i, s in enumerate(stats):
        np.add.at(out[i], (col.ravel(), cond.ravel()), values[s].cpu().numpy().ravel())


def episode_pass(f_q: torch.Tensor, f_s: torch.Tensor, support_y: torch.Tensor, xyz: torch.Tensor,
                 labels: torch.Tensor, arms: Dict[str, Dict], lp_u: Optional[Dict]):
    """One episode: predictions [B_q, P] of model-free rules, per-graph P7a values and per-arm values.

    arms maps an output name to a propagation arm on Y0; lp_u (optional) is one arm applied to U's prediction.
    Every graph is built from one query block; no label enters a prediction.
    """
    n_cls = support_y.shape[0] + 1
    u_logits = p6.rule_logits(f_q, p6.base_rows(f_q, f_s, support_y))  # [B_q, P, N+1]
    seed = both_logits(f_q, f_s, support_y).argmax(dim=-1)  # [B_q, P]
    preds = {"U": u_logits.argmax(dim=-1), "both": seed}
    u = F.normalize(f_q.to(torch.float64), dim=-1)  # [B_q, P, D]
    xyz = xyz.to(torch.float64)  # [B_q, P, 3]
    y0 = F.one_hot(seed, n_cls).to(torch.float64)  # [B_q, P, N+1]
    diag, arm_vals = {}, {}
    for graph, k in graph_keys():
        idx, a = knn_graph(xyz, u, graph, k)  # [B_q, P, k], [B_q, P, k]
        w = affinity(idx, a, xyz.shape[1])  # [B_q, P, P]
        st = point_stats(w, labels, seed, n_cls)
        diag[graph_name(graph, k)] = diag_values(st, reachable(idx, a, labels, st["hit"]))
        mine = {n: arm for n, arm in arms.items() if (arm["graph"], arm["k"]) == (graph, k)}
        u_here = lp_u is not None and (lp_u["graph"], lp_u["k"]) == (graph, k)
        if not mine and not u_here:
            continue
        s = normalized(w)  # [B_q, P, P]
        for name, arm in mine.items():
            preds[name] = spread(s, y0, arm["beta"]).argmax(dim=-1)  # [B_q, P]
            arm_vals[name] = arm_values(st, preds[name], labels)
        if u_here:
            yu = F.one_hot(preds["U"], n_cls).to(torch.float64)  # [B_q, P, N+1]
            preds["lp_u"] = spread(s, yu, lp_u["beta"]).argmax(dim=-1)
    return preds, diag, arm_vals


# ------------------------------------------------------------------ summaries and rules (pure)

def ratios(a: np.ndarray) -> Dict[str, float]:
    """P7a ratios of one DIAG_STATS vector [S] (one class and condition, or pooled)."""
    s = dict(zip(DIAG_STATS, a.tolist()))
    miss = s["n"] - s["hit"]
    q = lambda x, y: x / y if y > 0 else float("nan")  # noqa: E731
    return {"n": s["n"], "recall": q(s["hit"], s["n"]), "p_hit": q(s["p_hit"], s["hit"]), "p_miss": q(s["p_miss"], miss),
            "a_miss": q(s["a_miss"], s["n_a_miss"]), "b_miss": q(s["b_miss"], s["n_b_miss"]),
            "fixable": q(s["fixable"], miss), "breakable": q(s["breakable"], s["hit"]),
            "reach": q(s["reach_miss"], miss), "iso": q(s["iso"], s["n"])}


def diag_summary(arr: np.ndarray, test_classes: List[int]) -> Dict:
    """Ratios per class and condition, the foreground pooled per condition, and the background [S, C+1, 2]."""
    out = {"background": ratios(arr[:, 0, OWN])}
    for cond, cname in ((OWN, "own"), (OTHER, "other")):
        out[f"fg_{cname}"] = ratios(arr[:, 1:, cond].sum(axis=1))
        for i, c in enumerate(test_classes):
            out[f"{c}_{cname}"] = ratios(arr[:, 1 + i, cond])
    return out


def mechanism(arm_arr: np.ndarray) -> Dict[str, float]:
    """P7.6 on one arm's ARM_STATS [4, C+1, 2]: foreground points fixed and broken, mean homophily of each set."""
    s = {n: float(arm_arr[i, 1:, :].sum()) for i, n in enumerate(ARM_STATS)}
    q = lambda x, y: x / y if y > 0 else float("nan")  # noqa: E731
    return {"fixed": s["fixed"], "broken": s["broken"], "p_fixed": q(s["p_fixed"], s["fixed"]),
            "p_broken": q(s["p_broken"], s["broken"])}


def passes_gate(valid_gain: float) -> bool:
    """P7.1: the frozen arm's valid gain over Y0 in points reaches the gate."""
    return valid_gain >= GATE_GAIN


def decide(draws: Dict[str, Tuple[Dict, Dict[str, np.ndarray]]], sel: Dict) -> List[Tuple[str, str]]:
    """Rules P7.1-P7.7 of [DECISION D-40]."""
    v = [(f"P7.1 gate {'pass' if sel['gate'] else 'stop'}",
          f"{sel['frozen']} - Y0 on valid {sel['valid_gain']:+.2f} (gate +{GATE_GAIN})")]
    missing = [d for d in DRAWS if d not in draws]
    if missing:
        return v + [("incomplete", f"missing draws {missing}")]
    fixed = draws["fixed100"][0]
    if sel["gate"]:
        if "lp" not in fixed["miou"]:
            return v + [("incomplete", "the gate passed but lp was not scored")]
        p, rand = d39.paired(draws, "both", "lp")
        text = f"lp - Y0 fixed100 {r2.ci_text(p)}; random600 {[round(g, 2) for g in rand]}"
        adopted = d39.holds(p, rand, ADOPT_GAIN)
        if d39.holds(p, rand, TRAIN_GAIN):
            v.append(("P7.2 adopt + P7.3 trained form admissible", text))
        elif adopted:
            v.append(("P7.2 adopt", text))
        elif p["gain"] < ADOPT_GAIN:
            v.append(("P7.4 stop", text))
        else:
            v.append(("P7 between (not adopted)", text))
        lf = draws["leakfree"][0]["miou"]
        g_lf = 100.0 * (lf["lp"] - lf["both"])
        reading = "protocol-dependent" if adopted and g_lf <= 0 else "reported"
        v.append((f"P7.5 leak-free {reading}", f"lp - Y0 {g_lf:+.2f} (Y0 {100 * lf['both']:.2f})"))
        mech = mechanism(np.asarray(fixed["arm_stats"]["lp"]))
        ok = mech["fixed"] > mech["broken"] and mech["p_fixed"] > mech["p_broken"]
        state = "confirmed" if ok else "unexplained" if adopted else "not confirmed"
        v.append((f"P7.6 mechanism {state}", f"foreground fixed {mech['fixed']:.0f} (mean p {mech['p_fixed']:.3f}), "
                                             f"broken {mech['broken']:.0f} (mean p {mech['p_broken']:.3f})"))
        m = fixed["miou"]
        v.append(("P7.7 composition", f"lp_u - U {100 * (m['lp_u'] - m['U']):+.2f}, lp - Y0 "
                                      f"{100 * (m['lp'] - m['both']):+.2f}, Y0 - U {100 * (m['both'] - m['U']):+.2f}"))
    else:
        v.append(("P7.4 stop at the gate", "no propagation arm scored on a test draw"))
    m = fixed["miou"]
    v.append(("P7.7 levels fixed100", f"model {100 * m['model']:.2f}, U {100 * m['U']:.2f}, Y0 {100 * m['both']:.2f}"
                                      + (f", lp {100 * m['lp']:.2f}" if "lp" in m else "")))
    for gname, summ in fixed["diag_summary"].items():
        parts = []
        for key in ("fg_own", "fg_other", "background"):
            r = summ[key]
            parts.append(f"{key} R {r['recall']:.3f} p_miss {r['p_miss']:.3f} p_hit {r['p_hit']:.3f} a_miss "
                         f"{r['a_miss']:.3f} fixable {r['fixable']:.3f} breakable {r['breakable']:.3f} reach "
                         f"{r['reach']:.3f}")
        v.append((f"P7a {gname}", " | ".join(parts)))
    return v


# ------------------------------------------------------------------ GPU passes

@torch.no_grad()
def score(rule, draw: str, data_path: str, arms: Dict[str, Dict], lp_u: Optional[Dict], device,
          max_episodes: Optional[int]) -> Tuple[Dict, Dict[str, np.ndarray]]:
    from pipeline.episodes import make_episode, read_class_names

    names = read_class_names(data_path, "s3dis")
    items, test_classes = p6.episodes(draw, data_path, max_episodes)
    n_col = len(test_classes) + 1
    counts: Dict[str, List[np.ndarray]] = {}
    split: Dict[str, np.ndarray] = {}
    diag = {graph_name(g, k): np.zeros((len(DIAG_STATS), n_col, 2)) for g, k in graph_keys()}
    arm_stats = {n: np.zeros((len(ARM_STATS), n_col, 2)) for n in arms}
    preds_model, gts, l2c, skipped = [], [], [], 0
    for item in items:
        if (item[1].reshape(item[1].shape[0], -1).sum(1) == 0).any():
            skipped += 1  # a uniformly sampled support block without its class (leak-free draw only)
            continue
        e = make_episode(item, names).to(device)
        f_q, m_eff, _, logits = rule(e)
        p0.check_identity(f_q, m_eff, logits)  # the rule reads the tensors the model scores with
        f_s = p5.support_features(rule, e)
        preds, dvals, avals = episode_pass(f_q, f_s, e.support_y, e.query_x[..., :3], e.query_y, arms, lp_u)
        preds["model"] = logits.argmax(dim=-1)
        gt = e.query_y.cpu().numpy()
        col, cond = point_columns(gt, e.sampled_classes, test_classes)
        for gname, vals in dvals.items():
            accumulate(diag[gname], col, cond, vals, DIAG_STATS)
        for name, vals in avals.items():
            accumulate(arm_stats[name], col, cond, vals, ARM_STATS)
        for name, pred in preds.items():
            pr = pred.cpu().numpy()
            counts.setdefault(name, []).append(p0.episode_counts(pr, gt, e.sampled_classes, test_classes))
            sc = p5.condition_counts(pr, gt, e.sampled_classes, test_classes)
            split[name] = split.get(name, 0) + sc
        preds_model.append(preds["model"].cpu().numpy()), gts.append(gt), l2c.append(e.sampled_classes)
    stacked = {k: np.stack(v) for k, v in counts.items()}
    miou = {k: float(p0.miou_from_counts(v.sum(0))) for k, v in stacked.items()}
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
                check_ref("model", miou, json.load(f)["miou"]["cr"], "D-37's CR")
            with open(os.path.join(REPO, D39_FIXED)) as f:
                ref = json.load(f)["miou"]
            check_ref("U", miou, ref["cr:base"], "D-39's cr:base")
            check_ref("both", miou, ref["cr:both"], "D-39's cr:both")
    if draw == "valid" and full:
        with open(os.path.join(REPO, P6_SELECT)) as f:
            ref = json.load(f)["miou"]
        check_ref("model", miou, ref["model"], "P6's valid model")
        check_ref("U", miou, ref["U"], "P6's valid U")
    return {"draw": draw, "episodes": len(gts), "skipped": skipped, "test_classes": test_classes, "miou": miou,
            "class_iou": {k: p0.class_ious(v.sum(0)).tolist() for k, v in stacked.items()},
            "recall": {k: (v.sum(0)[2] / np.maximum(v.sum(0)[0], 1)).tolist() for k, v in stacked.items()},
            "precision": {k: (v.sum(0)[2] / np.maximum(v.sum(0)[1], 1)).tolist() for k, v in stacked.items()},
            "split": {k: p5.split_summary(v) for k, v in split.items()},
            "diag_stats": list(DIAG_STATS), "diag": {k: v.tolist() for k, v in diag.items()},
            "diag_summary": {k: diag_summary(v, test_classes) for k, v in diag.items()},
            "arm_stats_names": list(ARM_STATS), "arm_stats": {k: v.tolist() for k, v in arm_stats.items()},
            "mechanism": {k: mechanism(v) for k, v in arm_stats.items()}}, stacked


def check_ref(name: str, miou: Dict[str, float], ref: float, what: str) -> None:
    if abs(miou[name] - ref) > REF_TOL:
        raise RuntimeError(f"{name} scores {miou[name]:.6f}, {what} {ref:.6f}")


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
            sel = json.load(f)
        draws = {}
        for d in DRAWS:
            path = os.path.join(args.out_dir, p6.stem(d, args.tag) + ".json")
            if os.path.isfile(path):
                with open(path) as f:
                    draws[d] = (json.load(f), dict(np.load(path.replace(".json", "_counts.npz"))))
        for rule_name, text in decide(draws, sel):
            print(f"{rule_name:44s} {text}")
        return 0
    if not args.data_path or not args.checkpoint:
        p.error("select and test need --data_path and --checkpoint")
    device = torch.device("cuda")
    rule, protocol, config, ck = p6.load_rule(args.checkpoint, device)  # the clean base [DECISION D-38]
    try:
        if args.stage == "select":
            arms = {lp_name(a): a for a in lp_arms()}
            result, stacked = score(rule, "valid", args.data_path, arms, None, device, args.max_episodes)
            frozen, gain = p6.select(result["miou"], list(arms), base="both")
            result.update(frozen=frozen, valid_gain=gain, gate=passes_gate(gain), checkpoint=vars(ck),
                          protocol=protocol)
            p6.save(result, stacked, f"select{args.tag}", args.out_dir)
            m = result["miou"]
            print(f"[select] model {100 * m['model']:.2f}, U {100 * m['U']:.2f}, Y0 {100 * m['both']:.2f} | frozen "
                  f"{frozen} ({gain:+.2f}) | gate {'pass' if result['gate'] else 'stop'}", flush=True)
        else:
            with open(sel_path) as f:
                sel = json.load(f)
            arm = arm_of(sel["frozen"])
            arms, lp_u = ({"lp": arm}, arm) if sel["gate"] else ({}, None)
            for draw in DRAWS:
                result, stacked = score(rule, draw, args.data_path, arms, lp_u, device, args.max_episodes)
                result.update(frozen=sel["frozen"], gate=sel["gate"], checkpoint=vars(ck), protocol=protocol)
                p6.save(result, stacked, p6.stem(draw, args.tag), args.out_dir)
                print(f"[test] {draw}: " + " | ".join(f"{n} {100 * v:.2f}" for n, v in result["miou"].items())
                      + f" | {result['episodes']} episodes", flush=True)
    finally:
        rule.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
