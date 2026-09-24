"""P5 of [DECISION D-35]: E1's oracle gap split by sampling condition and class presence. No training.

    python experiments/p5_condition_probe.py fixed --data_path datasets/S3DIS/blocks_bs1_s1 \
        --checkpoint e1:ours:1:log_r2/s3dis_S1_N2_K1_point_T4_vip_b1/last.pt --checkpoint vipseg:vipseg:1:vipseg_S1_N2_K1.pt
                                     # arms A (split, counterfactuals, oracles), C (batch statistics), E (dual views)
    python experiments/p5_condition_probe.py intervene --data_path ... --checkpoint e1:ours:1:...   # arm B
    python experiments/p5_condition_probe.py leakfree --data_path ... --checkpoint e1:ours:1:...    # arm D
    python experiments/p5_condition_probe.py decide                                                 # P5.1-P5.5

The inherited sampler [VIPSEG dataloaders/loader.py:31-89] over-samples the class a block is sampled for; in a
2-way episode query block b is sampled for local class b + 1, so the other episode class, when present, is at
background density. Arm A tags every (query block, episode class) as own / other / absent and splits the counts of
VIP-Seg's metric by that tag. Arm B re-samples the same scans. Arm C evaluates with the batch statistics training
uses. Arm D draws both sides uniformly (COSeg Tab.1's "w/o FG"). Arm E adds support views with the foreground thinned
to background density. Query labels are read for the split, the oracles and the counterfactuals only; no arm that
could be a method reads them.
"""

import argparse
import contextlib
import itertools
import json
import os
import sys
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from experiments import p0_em_probe as p0  # noqa: E402
from experiments import p4_background_probe as p4  # noqa: E402
from experiments import r2_distill_eval as r2  # noqa: E402

OUT_DIR = "results/phase16_p5"
EPISODES_PER_PAIR = 100  # as fixed100
DRAW_SEEDS = {"intervene": 3, "leakfree": 4}  # fresh draws, P4 used 1 and 2 [DECISION D-35]
EXTRA_DRAW = "random600:0"  # arm C's second draw [DECISION D-35]
MIN_RAW_POINTS = 100  # the other class must hold this many raw points of the scan to be re-sampled for [D-35]
JITTER_SIGMA, JITTER_CLIP = 0.01, 0.05  # the training jitter [VIPSEG dataloaders/loader.py:110-112]
GO_GAIN, FOLLOW_GAIN = 1.0, 0.5  # [DECISION D-35], the thresholds of R2/E1/N1 and P0-P4
PHI_CAUSAL, PHI_NOT = 0.5, 0.2  # [DECISION D-35]
CLUTTER = 12  # S3DIS class id of clutter, in no fold [VIPSEG dataloaders/s3dis.py:30-31]

OWN, OTHER, ABSENT = 0, 1, 2
# rows of the split counts, per test class column (column 0, background, stays zero)
GT_OWN, TP_OWN, GT_OTHER, TP_OTHER, FP_OWN, FP_OTHER, FP_ABSENT = range(7)
SPLIT_ROWS = ("gt_own", "tp_own", "gt_other", "tp_other", "fp_own", "fp_other", "fp_absent")
FP_KINDS = ("episode", "base", "clutter", "novel_out")


# ------------------------------------------------------------------ pure functions (tested on the CPU)

def block_status(gt: np.ndarray, n_way: int) -> np.ndarray:
    """[B_q, N]: OWN if query block b was sampled for local class k (k = b + 1), else OTHER if k occurs in it, else ABSENT.

    gt [B_q, P] in {0..N}. One query per way [VIPSEG dataloaders/loader.py:181-222], so B_q must equal N.
    """
    if gt.shape[0] != n_way:
        raise ValueError(f"P5 needs one query block per way, got {gt.shape[0]} blocks for {n_way} ways")
    status = np.full((gt.shape[0], n_way), ABSENT, dtype=np.int64)  # [B_q, N]
    for b in range(gt.shape[0]):
        for k in range(1, n_way + 1):
            if k == b + 1:
                status[b, k - 1] = OWN
            elif (gt[b] == k).any():
                status[b, k - 1] = OTHER
    return status


def condition_counts(pred: np.ndarray, gt: np.ndarray, label2class, test_classes: List[int]) -> np.ndarray:
    """[7, C+1] split counts of one episode (rows SPLIT_ROWS, column = 1 + index of the global class).

    pred, gt [B_q, P] in {0..N}. GT and TP are split by the status of the class in the block that holds the point;
    FP (predicted k, labelled otherwise) by the status of k in the block where it is predicted.
    """
    n_way = len(label2class)
    status = block_status(gt, n_way)  # [B_q, N]
    out = np.zeros((7, len(test_classes) + 1), dtype=np.float64)
    for b in range(gt.shape[0]):
        for k in range(1, n_way + 1):
            col = test_classes.index(int(label2class[k - 1])) + 1
            is_k, pr_k = gt[b] == k, pred[b] == k  # [P], [P]
            g, tp, fp = int(is_k.sum()), int((is_k & pr_k).sum()), int((~is_k & pr_k).sum())
            s = status[b, k - 1]
            if s == OWN:
                out[GT_OWN, col] += g
                out[TP_OWN, col] += tp
                out[FP_OWN, col] += fp
            elif s == OTHER:
                out[GT_OTHER, col] += g
                out[TP_OTHER, col] += tp
                out[FP_OTHER, col] += fp
            else:
                out[FP_ABSENT, col] += fp  # g = tp = 0 by definition
    return out


def counterfactual(pooled: np.ndarray, split: np.ndarray, kind: str) -> np.ndarray:
    """Pooled counts [3, C+1] (gt, pred, tp) of a counterfactual, from summed pooled and split counts [7, C+1].

    'a': each class's other-condition TP raised to recall_own * GT_other, the predicted count raised by the same
    amount, every FP unchanged. 'b': each class's FP in absent blocks removed. Bounds, never results [D-35].
    """
    gt, pr, tp = (pooled[i].copy() for i in range(3))
    if kind == "a":
        r_own = np.divide(split[TP_OWN], split[GT_OWN], out=np.zeros_like(split[TP_OWN]), where=split[GT_OWN] > 0)
        add = np.maximum(0.0, r_own * split[GT_OTHER] - split[TP_OTHER])  # [C+1]
        add[0] = 0.0
        tp, pr = tp + add, pr + add
    elif kind == "b":
        fp_abs = split[FP_ABSENT].copy()
        fp_abs[0] = 0.0
        pr = pr - fp_abs
    else:
        raise ValueError(f"counterfactual kind must be 'a' or 'b', got {kind!r}")
    return np.stack([gt, pr, tp])


def split_summary(split: np.ndarray) -> Dict[str, List[float]]:
    """Recall per condition and FP per status, relative to each class's total GT, from summed split counts."""
    gt = split[GT_OWN] + split[GT_OTHER]  # [C+1]
    safe = lambda a, b: np.divide(a, b, out=np.full_like(a, np.nan), where=b > 0).tolist()  # noqa: E731
    return {"recall_own": safe(split[TP_OWN], split[GT_OWN]), "recall_other": safe(split[TP_OTHER], split[GT_OTHER]),
            "other_share": safe(split[GT_OTHER], gt), "fp_own": safe(split[FP_OWN], gt),
            "fp_other": safe(split[FP_OTHER], gt), "fp_absent": safe(split[FP_ABSENT], gt),
            "counts": {k: split[i].tolist() for i, k in enumerate(SPLIT_ROWS)}}


def support_directions(f_s: torch.Tensor, support_y: torch.Tensor) -> torch.Tensor:
    """[N+1, D] unit directions of the support in the oracle's geometry: background (all mask-0 points of all
    ways) then each way's foreground, as normalised sums of unit features (`oracle_directions`' form)."""
    u = F.normalize(f_s, dim=-1)  # [N, K, P, D]
    fg = (support_y == 1).to(u.dtype)  # [N, K, P]
    s_fg = torch.einsum("nkp,nkpd->nd", fg, u)  # [N, D]
    s_bg = torch.einsum("nkp,nkpd->d", 1.0 - fg, u)  # [D]
    return F.normalize(torch.cat([s_bg.unsqueeze(0), s_fg], dim=0), dim=-1)  # [N+1, D]


def presence_fair_logits(f_q: torch.Tensor, f_s: torch.Tensor, support_y: torch.Tensor,
                         labels: torch.Tensor) -> torch.Tensor:
    """Oracle whose absent classes still compete [DECISION D-35]: present rows are the query's own unit-feature
    means O, absent rows the support's directions, all in one geometry and one norm. f_q [B_q, P, D] ->
    logits [B_q, P, N+1]."""
    from models.oracle_distill import oracle_directions

    n_cls = support_y.shape[0] + 1
    o, present = oracle_directions(f_q, labels, n_cls)  # [B_q, N+1, D], [B_q, N+1]
    s = support_directions(f_s, support_y).unsqueeze(0).expand_as(o)  # [B_q, N+1, D]
    rows = torch.where(present.unsqueeze(-1), o, s)  # [B_q, N+1, D]
    return torch.einsum("bpd,bcd->bpc", f_q, rows)  # [B_q, P, N+1]


def support_rule_logits(f_q: torch.Tensor, f_s: torch.Tensor, support_y: torch.Tensor) -> torch.Tensor:
    """P3's rule without head, `F^q n(P_point)ᵀ` [B_q, P, N+1] (Eq.3 with L2 rows) [DECISION D-31]."""
    from models.prototypes import point_prototypes

    p = F.normalize(point_prototypes(f_s, support_y), dim=-1)  # [N+1, D]
    return torch.einsum("bpd,cd->bpc", f_q, p)  # [B_q, P, N+1]


@contextlib.contextmanager
def batch_statistics(module: nn.Module) -> Iterator[int]:
    """Every BatchNorm of `module` normalises with its batch's statistics, as in training, while the running
    statistics stay untouched (momentum 0); everything else keeps its mode [DECISION D-35]. Yields the count."""
    bns = [m for m in module.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not bns:
        raise ValueError("no BatchNorm layer: batch statistics would change nothing")
    saved = [(m.training, m.momentum, m.num_batches_tracked.clone() if m.num_batches_tracked is not None else None,
              m.running_mean.clone(), m.running_var.clone()) for m in bns]
    try:
        for m in bns:
            m.train()
            m.momentum = 0.0
        yield len(bns)
    finally:
        for m, (training, momentum, tracked, mean, var) in zip(bns, saved):
            m.train(training)
            m.momentum = momentum
            if tracked is not None:
                m.num_batches_tracked.copy_(tracked)
            if not (torch.equal(m.running_mean, mean) and torch.equal(m.running_var, var)):
                raise RuntimeError("batch statistics mode changed a running statistic")


def sampled_indices(data: np.ndarray, num_point: int, sampled_class: int, random_sample: bool) -> np.ndarray:
    """The inherited sampler's point indices, with its exact calls to the global np.random
    [VIPSEG dataloaders/loader.py:35-58]; `sample_block` checks the copy against the original on every call."""
    n = data.shape[0]
    if random_sample:
        idx = np.random.choice(np.arange(n), num_point, replace=(n < num_point))
    else:
        valid = np.nonzero(data[:, 6] == sampled_class)[0]
        if n < num_point:
            n_valid = len(valid)
        else:
            n_valid = int(len(valid) / float(n) * num_point)
        a = np.random.choice(valid, n_valid, replace=False)
        b = np.random.choice(np.arange(n), num_point - n_valid, replace=(n < num_point))
        idx = np.concatenate([a, b])
    np.random.shuffle(idx)
    return idx


def episode_labels(raw: np.ndarray, classes) -> np.ndarray:
    """Query labels of the loader [VIPSEG dataloaders/loader.py:84-88]: 1 + index among the episode's classes, else 0."""
    out = np.zeros(raw.shape, dtype=np.int64)
    for i, c in enumerate(classes):
        out[raw == int(c)] = i + 1
    return out


def fp_kind(raw: int, classes, train_classes, test_classes) -> str:
    """Raw class of a false-positive point: the episode's other class, a base class, clutter, or a novel class
    outside the episode."""
    if raw in [int(c) for c in classes]:
        return "episode"
    if raw in train_classes:
        return "base"
    if raw == CLUTTER:
        return "clutter"
    if raw in test_classes:
        return "novel_out"
    raise ValueError(f"raw class {raw} is in no category")


def sparse_view(block: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """A support block [P, 9] (xyz, rgb, XYZ) with its foreground thinned to background density [DECISION D-35].

    With foreground share f = r(2 - r) under the sampler, r the raw share, keeping each foreground point with
    probability (1 - r)/(2 - r) leaves a foreground share r among the kept points. The block is refilled to P points
    with copies of uniformly drawn kept points plus the training jitter, then re-normalised like the loader
    (xyz shifted to its minimum, XYZ = xyz / max) and shuffled. mask [P] in {0, 1}.
    """
    p = block.shape[0]
    f = float(mask.mean())
    if not 0.0 < f < 1.0:
        raise ValueError(f"a sparse view needs foreground and background points, got foreground share {f}")
    r = 1.0 - np.sqrt(1.0 - f)
    keep = (mask == 0) | (rng.random(p) < (1.0 - r) / (2.0 - r))  # [P]
    if not (mask[keep] == 1).any():
        keep[np.nonzero(mask == 1)[0][0]] = True  # at least one foreground point survives
    kept = np.nonzero(keep)[0]
    extra = rng.choice(kept, p - kept.size, replace=True)
    jitter = np.clip(JITTER_SIGMA * rng.standard_normal((extra.size, 3)), -JITTER_CLIP, JITTER_CLIP)
    xyz = np.concatenate([block[kept, :3], block[extra, :3] + jitter])  # [P, 3]
    rgb = np.concatenate([block[kept, 3:6], block[extra, 3:6]])  # [P, 3]
    m = np.concatenate([mask[kept], mask[extra]])  # [P]
    xyz = xyz - xyz.min(axis=0)  # [VIPSEG dataloaders/loader.py:65-66]
    xyz_n = xyz / xyz.max(axis=0)  # [VIPSEG dataloaders/loader.py:70-74]
    order = rng.permutation(p)
    return np.concatenate([xyz, rgb, xyz_n], axis=1)[order].astype(block.dtype), m[order].astype(mask.dtype)


def match_fg_count(logits: torch.Tensor, target: int, iters: int = 60) -> torch.Tensor:
    """logits [B_q, P, N+1] plus the constant on the foreground columns whose foreground-prediction count is
    closest to `target` (the count is non-decreasing in the constant): the dual arm's control [DECISION D-35]."""
    def shifted(delta: float) -> torch.Tensor:
        out = logits.clone()  # [B_q, P, N+1]
        out[..., 1:] += delta
        return out

    def count(delta: float) -> int:
        return int((shifted(delta).argmax(-1) != 0).sum())

    span = float(logits.max() - logits.min()) + 1.0
    lo, hi = -span, span  # count(lo) = 0 and count(hi) = all points
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if count(mid) < target:
            lo = mid
        else:
            hi = mid
    best = min((lo, hi), key=lambda d: abs(count(d) - target))
    return shifted(best)


def dual_logits(dense: torch.Tensor, sparse: torch.Tensor) -> torch.Tensor:
    """Background from the dense run, foreground class k = max(dense_k, sparse_k) [B_q, P, N+1]."""
    return torch.cat([dense[..., :1], torch.maximum(dense[..., 1:], sparse[..., 1:])], dim=-1)


def phi(r_v0: float, r_v1: float, r_own: float) -> float:
    """Causal fraction (R_V1 - R_V0) / (R_own - R_V0) [DECISION D-35]; nan without an own/other gap."""
    gap = r_own - r_v0
    return float((r_v1 - r_v0) / gap) if gap > 0 else float("nan")


def intervention_summary(events: List[Dict], r_own: Dict[int, float], seed: int = 0, boot: int = 2000) -> Dict:
    """Pooled recalls of the re-sampled class under V0/V1/V2, phi, and episode-bootstrap CIs.

    events: one per re-sampled query block, keys episode, c, a, gt_c/tp_c per version (lists V0, V1, V2),
    gt_a/tp_a (V0, V2). r_own: recall of each class in the blocks sampled for it, same draw.
    """
    if not events:
        return {"events": 0}
    ep = np.array([e["episode"] for e in events])
    gt_c, tp_c = np.array([e["gt_c"] for e in events], float), np.array([e["tp_c"] for e in events], float)  # [E, 3]
    gt_a, tp_a = np.array([e["gt_a"] for e in events], float), np.array([e["tp_a"] for e in events], float)  # [E, 2]
    w_own = np.array([r_own.get(e["c"], np.nan) for e in events])  # [E]

    def stats(sel: np.ndarray) -> Dict[str, float]:
        g, t = gt_c[sel].sum(0), tp_c[sel].sum(0)  # [3]
        r = t / np.maximum(g, 1.0)
        ok = sel & np.isfinite(w_own)
        ro = float((gt_c[ok, 0] * w_own[ok]).sum() / max(gt_c[ok, 0].sum(), 1.0))
        ra = tp_a[sel].sum(0) / np.maximum(gt_a[sel].sum(0), 1.0)  # [2]
        return {"r_v0": float(r[0]), "r_v1": float(r[1]), "r_v2": float(r[2]), "r_own": ro,
                "phi": phi(r[0], r[1], ro), "r_a_v0": float(ra[0]), "r_a_v2": float(ra[1])}

    out = stats(np.ones(len(events), bool))
    rng = np.random.default_rng(seed)
    uniq = np.unique(ep)
    d, ph = [], []
    for _ in range(boot):
        pick = rng.choice(uniq, uniq.size, replace=True)
        sel = np.concatenate([np.nonzero(ep == u)[0] for u in pick])
        g, t = gt_c[sel].sum(0), tp_c[sel].sum(0)
        r = t / np.maximum(g, 1.0)
        ok = np.isfinite(w_own[sel])
        ro = (gt_c[sel][ok, 0] * w_own[sel][ok]).sum() / max(gt_c[sel][ok, 0].sum(), 1.0)
        d.append(r[1] - r[0])
        ph.append(phi(r[0], r[1], ro))
    out.update({"events": len(events), "episodes": int(uniq.size),
                "v1_minus_v0_ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                "phi_ci": [float(np.nanpercentile(ph, 2.5)), float(np.nanpercentile(ph, 97.5))]})
    out["per_class"] = {int(c): stats(np.array([e["c"] == c for e in events])) for c in sorted({e["c"] for e in events})}
    return out


def decide(fixed: Dict, intervene: Optional[Dict], leakfree: Optional[Dict], name: str = "e1") -> List[Tuple[str, str]]:
    """Rules P5.1-P5.5 of [DECISION D-35] on E1 (`name`)."""
    a = fixed["checkpoints"][name]
    v = []
    gain_a = a["cf_a_gain"]
    p51 = gain_a >= GO_GAIN
    v.append(("P5.1 E-a", f"cf-a - E1 {gain_a:+.2f}: " + ("matters" if p51 else "below +1.0, the other-condition line stops")))
    if p51:
        if intervene is None or intervene.get("events", 0) == 0:
            v.append(("P5.1a/b", "incomplete: arm B not run"))
        else:
            f, lo = intervene["phi"], intervene["v1_minus_v0_ci"][0]
            if f >= PHI_CAUSAL and lo > 0:
                v.append(("P5.1a causal", f"phi {f:.2f} >= {PHI_CAUSAL}, V1 - V0 CI low {lo:+.3f} > 0: decisions for M1 and M2"))
            elif f < PHI_NOT:
                v.append(("P5.1b not density", f"phi {f:.2f} < {PHI_NOT}: M1 only, M2 dropped"))
            else:
                v.append(("P5.1 in between", f"phi {f:.2f}, V1 - V0 CI low {lo:+.3f}: report"))
    gain_b = a["cf_b_gain"]
    p52 = gain_b >= GO_GAIN
    v.append(("P5.2 E-b", f"cf-b - E1 {gain_b:+.2f}: " + ("a decision for presence estimation (M3)" if p52 else "below +1.0")))
    t, t600 = a["paired"]["tbn_vs_model"], a.get("tbn_gain_extra")
    if t600 is None:
        v.append(("P5.3 batch statistics", "incomplete: the random600 draw is missing"))
    else:
        ok = t["gain"] >= GO_GAIN and t["ci_low"] > 0 and t600 > 0
        v.append(("P5.3 batch statistics", f"fixed100 {t['gain']:+.2f} [{t['ci_low']:+.2f}, {t['ci_high']:+.2f}], "
                  f"{EXTRA_DRAW} {t600:+.2f}: " + ("the test-time rule of later arms (transductive)" if ok else "not adopted")))
    if "dual_vs_control" in a["paired"]:
        d = a["paired"]["dual_vs_control"]
        ok = d["gain"] >= FOLLOW_GAIN and d["ci_low"] > 0
        v.append(("P5.4 dual prototypes", f"dual - control {d['gain']:+.2f} [{d['ci_low']:+.2f}, {d['ci_high']:+.2f}]: "
                  + ("M2 is trained" if ok else "not enough")))
    else:
        v.append(("P5.4 dual prototypes", "incomplete: arm E not run"))
    pf = a["paired"]["presence_fair_vs_model"]
    orc = a["paired"]["oracle_unit_vs_model"]
    rep = f"oracle {orc['gain']:+.2f}, presence-fair oracle {pf['gain']:+.2f}"
    if leakfree is not None:
        m = leakfree["miou"]
        rep += (f"; leak-free: E1 {100 * m['model']:.2f}, support rule {100 * m['support_rule']:.2f}, "
                f"oracle {100 * m['oracle_unit']:.2f}; protocol: E1 {100 * a['miou']['model']:.2f}, "
                f"support rule {100 * a['miou']['support_rule']:.2f}")
    if intervene is not None and "fp_kinds" in intervene:
        tot = sum(intervene["fp_kinds"].values())
        rep += "; FP kinds " + ", ".join(f"{k} {intervene['fp_kinds'][k] / max(tot, 1):.2f}" for k in FP_KINDS)
    v.append(("P5.5 report", rep))
    if not p51 and not p52:
        v.append(("neither", "condition and presence readings dropped; the gap is own-condition (M4)"))
    return v


# ------------------------------------------------------------------ episodes with scan names

def sample_block(data_path: str, scan: str, classes, sampled_class: int, random_sample: bool,
                 seed) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(ptcloud [P, 9], episode labels [P], raw labels [P]) of one query block; the ptcloud and labels come from the
    inherited `sample_pointcloud`, the raw labels from `sampled_indices` under the same seed, checked equal."""
    from dataloaders.loader import sample_pointcloud
    from pipeline.episodes import NUM_POINT, PC_ATTRIBS

    data = np.load(os.path.join(data_path, "data", f"{scan}.npy"))
    np.random.seed(seed)
    idx = sampled_indices(data, NUM_POINT, sampled_class, random_sample)  # [P]
    np.random.seed(seed)
    pc, gt = sample_pointcloud(data_path, NUM_POINT, PC_ATTRIBS, False, None, scan, classes, sampled_class,
                               support=False, random_sample=random_sample)
    raw = data[idx, 6].astype(np.int64)  # [P]
    xyz = data[idx, :3] - data[idx, :3].min(axis=0)  # [P, 3], the loader's shift
    if not (np.allclose(pc[:, :3], xyz) and np.array_equal(episode_labels(raw, classes), gt)):
        raise RuntimeError(f"{scan}: the index copy of the sampler differs from the inherited one")
    return pc, gt.astype(np.int64), raw


def raw_count(data_path: str, scan: str, cls: int) -> int:
    return int((np.load(os.path.join(data_path, "data", f"{scan}.npy"))[:, 6] == cls).sum())


def draw_episodes(data_path: str, cvfold: int, seed: int, per_pair: int, random_sample: bool,
                  max_episodes: Optional[int] = None) -> Iterator[Dict]:
    """Seeded test episodes with their scan names and the raw labels of the query points. Scans are chosen as in
    `generate_one_episode` (query then support per way, no scan reused) [VIPSEG dataloaders/loader.py:174-218];
    `random_sample` samples support and query uniformly (arm D)."""
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
            np.random.seed([seed, pi, e])
            classes = np.array(pair)
            used, scans = [], []
            for c in classes:
                q = np.random.choice([s for s in ds.query_class2scans[c] if s not in used], 1, replace=False)[0]
                used.append(q)
                s = np.random.choice([s for s in ds.support_class2scans[c] if s not in used], 1, replace=False)[0]
                used.append(s)
                scans.append((q, s))
            sx, sy, qx, qy, qraw = [], [], [], [], []
            for w, (c, (q, s)) in enumerate(zip(classes, scans)):
                x, y, raw = sample_block(data_path, q, classes, int(c), random_sample, [seed, pi, e, 0, w])
                qx.append(x), qy.append(y), qraw.append(raw)
                np.random.seed([seed, pi, e, 1, w])
                x, y = sample_K_pointclouds(data_path, NUM_POINT, PC_ATTRIBS, False, None, [s], c, classes,
                                            is_support=True, is_random_sample=random_sample)
                sx.append(x), sy.append(y.astype(np.int32))
            item = (np.stack(sx), np.stack(sy), np.stack(qx), np.stack(qy), classes)  # [N,1,P,9] [N,1,P] [N,P,9] [N,P]
            yield {"item": item, "q_scans": [q for q, _ in scans], "query_raw": np.stack(qraw), "pair": pi,
                   "index": count}
            count += 1


# ------------------------------------------------------------------ GPU passes

def module_of(rule) -> nn.Module:
    return rule.model if isinstance(rule, r2.OursRule) else rule.rule.baseline


def support_features(rule, episode) -> torch.Tensor:
    """F^s [N, K, P, D] as the model computes it (VIP-Seg: the hook of its last call; ours: the extractor)."""
    if isinstance(rule, r2.OursRule):
        if rule.model.neck is not None:
            raise ValueError("P5 reads E1-type checkpoints without a neck [DECISION D-35]")
        return rule.model.features.encode_episode(episode.support_x, episode.query_x)[0]
    return rule.rule.f_s


def test_classes_of(fold: int, data_path: str) -> List[int]:
    from dataloaders.s3dis import S3DISDataset

    return [int(c) for c in S3DISDataset(fold, data_path).test_classes]  # the loader's order


@torch.no_grad()
def score_fixed(rules: Dict[str, object], draw: str, data_path: str, device, full: bool,
                max_episodes: Optional[int] = None) -> Tuple[Dict, Dict[str, np.ndarray]]:
    """Arms A, C and E on one draw. full=False (the extra draw) scores the model and batch statistics only."""
    from torch.utils.data import DataLoader, Subset

    from pipeline.episodes import EpisodeCollate, read_class_names

    names = read_class_names(data_path, "s3dis")
    ds, test_classes = r2.episodes_of(draw, data_path, 1)
    view = ds if max_episodes is None else Subset(ds, range(min(max_episodes, len(ds))))
    loader = DataLoader(view, batch_size=1, shuffle=False, collate_fn=EpisodeCollate(names))
    counts: Dict[str, List[np.ndarray]] = {}
    split: Dict[str, List[np.ndarray]] = {}
    preds_of = {n: [] for n in rules}
    gts, l2c = [], []

    def add(key: str, pred: torch.Tensor, gt: np.ndarray, episode) -> None:
        p = pred.cpu().numpy()  # [B_q, P]
        counts.setdefault(key, []).append(p0.episode_counts(p, gt, episode.sampled_classes, test_classes))
        split.setdefault(key, []).append(condition_counts(p, gt, episode.sampled_classes, test_classes))

    for i, (episode,) in enumerate(loader):
        episode = episode.to(device)
        labels = episode.query_y  # [B_q, P], split, oracles and counterfactuals only
        gt = labels.cpu().numpy()
        for name, rule in rules.items():
            f_q, m_eff, _, logits = rule(episode)  # [B_q,P,D], [B_q,N+1,D], _, [B_q,P,N+1]
            p0.check_identity(f_q, m_eff, logits)
            add(name, logits.argmax(-1), gt, episode)
            preds_of[name].append(logits.argmax(-1).cpu().numpy())
            f_s = support_features(rule, episode) if full else None  # [N, K, P, D], before the batch-statistics call
            module = module_of(rule)
            with batch_statistics(module):
                _, _, _, logits_t = rule(episode)  # [B_q, P, N+1]
            if torch.equal(logits_t, logits):
                raise RuntimeError(f"{name}: batch statistics left the logits unchanged; no BatchNorm was switched")
            add(name + "_tbn", logits_t.argmax(-1), gt, episode)
            if not full:
                continue
            add(name + "_oracle_unit", torch.as_tensor(r2.oracle_replaced(f_q, m_eff, labels, unit=True)), gt, episode)
            add(name + "_presence_fair", presence_fair_logits(f_q, f_s, episode.support_y, labels).argmax(-1), gt, episode)
            add(name + "_support_rule", support_rule_logits(f_q, f_s, episode.support_y).argmax(-1), gt, episode)
            if isinstance(rule, r2.OursRule):  # arm E re-runs our head [DECISION D-35]
                model = rule.model
                dense_p0 = p4.first_prototype(f_s, episode.support_y, episode.support_y == 0)  # [N+1, D]
                _, dense = p4.run_head(model, f_s, f_q, dense_p0)  # [B_q, P, N+1]
                p0.check_identity(f_q, m_eff, dense)  # the re-run head is the model
                rng = np.random.default_rng([5, i])
                sx = episode.support_x.cpu().numpy()  # [N, K, P, 9]
                sy = episode.support_y.cpu().numpy()  # [N, K, P]
                views = [[sparse_view(sx[n, k], sy[n, k], rng) for k in range(sx.shape[1])] for n in range(sx.shape[0])]
                vx = torch.as_tensor(np.stack([[v[0] for v in w] for w in views]), device=device,
                                     dtype=episode.support_x.dtype)  # [N, K, P, 9]
                vy = torch.as_tensor(np.stack([[v[1] for v in w] for w in views]), device=device,
                                     dtype=episode.support_y.dtype)  # [N, K, P]
                f_sv = model.features.encode_episode(vx, episode.query_x)[0]  # [N, K, P, D]
                _, sparse = p4.run_head(model, f_sv, f_q, p4.first_prototype(f_sv, vy, vy == 0))  # [B_q, P, N+1]
                dual = dual_logits(dense, sparse)  # [B_q, P, N+1]
                target = int((dual.argmax(-1) != 0).sum())
                add(name + "_dual", dual.argmax(-1), gt, episode)
                add(name + "_control", match_fg_count(dense, target).argmax(-1), gt, episode)
        gts.append(gt), l2c.append(episode.sampled_classes)
    stacked = {k: np.stack(v) for k, v in counts.items()}  # [E, 3, C+1]
    stacked_split = {k: np.stack(v) for k, v in split.items()}  # [E, 7, C+1]
    if draw == "fixed100":  # the count-based mIoU must be VIP-Seg's own metric [D-08]
        from pipeline.evaluation import accumulated_miou

        class _Quiet:
            def cprint(self, text):
                pass

        for name in rules:
            primary = accumulated_miou(_Quiet(), preds_of[name], gts, l2c, test_classes)
            if abs(primary - p0.miou_from_counts(stacked[name].sum(0))) > 1e-9:
                raise RuntimeError(f"{name}: count-based mIoU differs from VIP-Seg's evaluate_metric")
    for key in stacked:  # the split is a partition of the pooled counts
        pooled, sp = stacked[key].sum(0), stacked_split[key].sum(0)
        fp = pooled[1] - pooled[2]
        if not (np.allclose((sp[GT_OWN] + sp[GT_OTHER])[1:], pooled[0][1:])
                and np.allclose((sp[TP_OWN] + sp[TP_OTHER])[1:], pooled[2][1:])
                and np.allclose((sp[FP_OWN] + sp[FP_OTHER] + sp[FP_ABSENT])[1:], fp[1:])):
            raise RuntimeError(f"{key}: the condition split does not add up to the pooled counts")
    result = {"draw": draw, "episodes": len(view), "test_classes": test_classes,
              "miou": {k: float(p0.miou_from_counts(v.sum(0))) for k, v in stacked.items()},
              "class_iou": {k: p0.class_ious(v.sum(0)).tolist() for k, v in stacked.items()},
              "split": {k: split_summary(v.sum(0)) for k, v in stacked_split.items()}}
    return result, {**stacked, **{k + "__split": v for k, v in stacked_split.items()}}


def fixed_result(rules: Dict[str, object], data_path: str, device, max_episodes: Optional[int]) -> Tuple[Dict, Dict]:
    main, arrays = score_fixed(rules, "fixed100", data_path, device, True, max_episodes)
    extra, extra_arrays = score_fixed(rules, EXTRA_DRAW, data_path, device, False, max_episodes)
    out = {"fixed100": main, EXTRA_DRAW: extra, "checkpoints": {}}
    for name in rules:
        pooled, sp = arrays[name].sum(0), arrays[name + "__split"].sum(0)
        base = float(p0.miou_from_counts(pooled))
        cf = {k: float(p0.miou_from_counts(counterfactual(pooled, sp, k))) for k in ("a", "b")}
        paired = {"tbn_vs_model": p0.paired_bootstrap(arrays[name], arrays[name + "_tbn"]),
                  "oracle_unit_vs_model": p0.paired_bootstrap(arrays[name], arrays[name + "_oracle_unit"]),
                  "presence_fair_vs_model": p0.paired_bootstrap(arrays[name], arrays[name + "_presence_fair"]),
                  "support_rule_vs_model": p0.paired_bootstrap(arrays[name], arrays[name + "_support_rule"])}
        if name + "_dual" in arrays:
            paired["dual_vs_control"] = p0.paired_bootstrap(arrays[name + "_control"], arrays[name + "_dual"])
            paired["dual_vs_model"] = p0.paired_bootstrap(arrays[name], arrays[name + "_dual"])
        e = extra_arrays
        out["checkpoints"][name] = {
            "miou": {k[len(name) + 1:] if k != name else "model": v for k, v in main["miou"].items()
                     if k == name or k.startswith(name + "_")},
            "cf_a": cf["a"], "cf_b": cf["b"], "cf_a_gain": 100.0 * (cf["a"] - base), "cf_b_gain": 100.0 * (cf["b"] - base),
            "paired": paired,
            "tbn_gain_extra": 100.0 * float(p0.miou_from_counts(e[name + "_tbn"].sum(0))
                                                                     - p0.miou_from_counts(e[name].sum(0)))}
    return out, {**arrays, **{EXTRA_DRAW.replace(":", "") + "__" + k: v for k, v in extra_arrays.items()}}


@torch.no_grad()
def score_intervention(rule, data_path: str, device, per_pair: int, max_episodes: Optional[int]) -> Dict:
    """Arm B: the other class of every query block, re-sampled in its own scan (V1) and uniformly (V2)."""
    from pipeline.episodes import make_episode, read_class_names

    names = read_class_names(data_path, "s3dis")
    test_classes = test_classes_of(1, data_path)
    train_classes = [c for c in range(12) if c not in test_classes]
    model = rule.model
    seed = DRAW_SEEDS["intervene"]
    own_gt: Dict[int, float] = {}
    own_tp: Dict[int, float] = {}
    events, fp_kinds = [], {k: 0 for k in FP_KINDS}
    for ep in draw_episodes(data_path, 1, seed, per_pair, False, max_episodes):
        sx, sy, qx, qy, classes = ep["item"]
        episode = make_episode(ep["item"], names).to(device)
        pred = model(episode).logits.argmax(-1).cpu().numpy()  # [B_q, P], V0
        for b in range(qy.shape[0]):
            a = int(classes[b])
            own_gt[a] = own_gt.get(a, 0.0) + float((qy[b] == b + 1).sum())
            own_tp[a] = own_tp.get(a, 0.0) + float(((qy[b] == b + 1) & (pred[b] == b + 1)).sum())
            fp = (pred[b] > 0) & (pred[b] != qy[b])  # [P]
            for r in ep["query_raw"][b][fp]:
                fp_kinds[fp_kind(int(r), classes, train_classes, test_classes)] += 1
        for b in range(qy.shape[0]):
            a, c_local = int(classes[b]), 2 - b  # the other way of a 2-way episode
            c = int(classes[c_local - 1])
            if raw_count(data_path, ep["q_scans"][b], c) < MIN_RAW_POINTS:
                continue
            gt_c, tp_c, gt_a, tp_a = [], [], [], []
            for v, (cls, uniform) in enumerate(((None, None), (c, False), (a, True))):
                if v == 0:
                    x, y, p = qx[b], qy[b], pred[b]
                else:
                    x, y, _ = sample_block(data_path, ep["q_scans"][b], classes, cls, uniform,
                                           [seed, ep["pair"], ep["index"], 2 + v, b])
                    qx2, qy2 = qx.copy(), qy.copy()
                    qx2[b], qy2[b] = x, y
                    e2 = make_episode((sx, sy, qx2, qy2, classes), names).to(device)
                    p = model(e2).logits.argmax(-1)[b].cpu().numpy()  # [P]
                gt_c.append(float((y == c_local).sum()))
                tp_c.append(float(((y == c_local) & (p == c_local)).sum()))
                if v != 1:
                    gt_a.append(float((y == b + 1).sum()))
                    tp_a.append(float(((y == b + 1) & (p == b + 1)).sum()))
            events.append({"episode": ep["index"], "c": c, "a": a, "gt_c": gt_c, "tp_c": tp_c, "gt_a": gt_a, "tp_a": tp_a})
    r_own = {k: own_tp[k] / own_gt[k] for k in own_gt if own_gt[k] > 0}
    out = intervention_summary(events, r_own)
    out.update({"seed": seed, "r_own_by_class": {int(k): v for k, v in r_own.items()}, "fp_kinds": fp_kinds,
                "min_raw_points": MIN_RAW_POINTS})
    return out


@torch.no_grad()
def score_leakfree(rule, data_path: str, device, per_pair: int, max_episodes: Optional[int]) -> Tuple[Dict, Dict]:
    """Arm D: support and query sampled uniformly; E1, the support rule and the oracle."""
    from pipeline.episodes import make_episode, read_class_names

    names = read_class_names(data_path, "s3dis")
    test_classes = test_classes_of(1, data_path)
    counts: Dict[str, List[np.ndarray]] = {}
    skipped = 0
    for ep in draw_episodes(data_path, 1, DRAW_SEEDS["leakfree"], per_pair, True, max_episodes):
        if (ep["item"][1].reshape(ep["item"][1].shape[0], -1).sum(1) == 0).any():
            skipped += 1  # a uniformly sampled support block without its class
            continue
        episode = make_episode(ep["item"], names).to(device)
        f_q, m_eff, _, logits = rule(episode)
        p0.check_identity(f_q, m_eff, logits)
        f_s = support_features(rule, episode)
        gt = episode.query_y.cpu().numpy()
        for key, pred in (("model", logits.argmax(-1)),
                          ("support_rule", support_rule_logits(f_q, f_s, episode.support_y).argmax(-1)),
                          ("oracle_unit", torch.as_tensor(r2.oracle_replaced(f_q, m_eff, episode.query_y, unit=True)))):
            counts.setdefault(key, []).append(p0.episode_counts(pred.cpu().numpy(), gt, episode.sampled_classes,
                                                                test_classes))
    stacked = {k: np.stack(v) for k, v in counts.items()}
    return ({"seed": DRAW_SEEDS["leakfree"], "episodes": len(stacked["model"]), "skipped": skipped,
             "test_classes": test_classes,
             "miou": {k: float(p0.miou_from_counts(v.sum(0))) for k, v in stacked.items()},
             "class_iou": {k: p0.class_ious(v.sum(0)).tolist() for k, v in stacked.items()}}, stacked)


# ------------------------------------------------------------------ CLI

def save(obj: Dict, arrays: Optional[Dict[str, np.ndarray]], stem: str) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, stem + ".json"), "w") as f:
        json.dump(obj, f, indent=1)
    if arrays:
        np.savez_compressed(os.path.join(OUT_DIR, stem + "_counts.npz"), **arrays)


def load_rules(specs: List[str], device) -> Tuple[Dict[str, object], Dict[str, str]]:
    rules, protocols = {}, {}
    for spec in specs:
        ck = p0.parse_checkpoint(spec)
        if ck.fold != 1:
            raise ValueError(f"{ck.name}: P5 runs on S1 only, S0 stays held out [DECISION D-22]")
        rule, protocol, config = r2.load(ck, 1, device)  # eval.py's protocol guard [DECISION D-22]
        if config is not None and (config["use_lma"] or config["logit_scale"] != "none"
                                   or not config["l2norm_point_proto"] or config["stage_type"] != "vip"):
            raise ValueError(f"{ck.name}: P5 reads route B's head (vip stages, L2 prototypes, no LMA) [D-35]")
        rules[ck.name], protocols[ck.name] = rule, protocol
    return rules, protocols


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["fixed", "intervene", "leakfree", "decide"])
    p.add_argument("--data_path")
    p.add_argument("--checkpoint", action="append", default=[])
    p.add_argument("--name", default="e1")
    p.add_argument("--per_pair", type=int, default=EPISODES_PER_PAIR)
    p.add_argument("--max_episodes", type=int, default=None, help="smoke runs only")
    p.add_argument("--tag", default="", help="suffix of the output files (smoke runs)")
    args = p.parse_args(argv)
    if args.stage == "decide":
        def read(stem):
            path = os.path.join(OUT_DIR, stem + args.tag + ".json")
            return json.load(open(path)) if os.path.isfile(path) else None
        fixed = read("fixed")
        if fixed is None:
            raise FileNotFoundError("run the fixed stage first")
        for rule, text in decide(fixed, read("intervene"), read("leakfree"), args.name):
            print(f"{rule:24s} {text}")
        return 0
    if not args.data_path or not args.checkpoint:
        p.error("fixed, intervene and leakfree need --data_path and --checkpoint")
    device = torch.device("cuda")
    rules, protocols = load_rules(args.checkpoint, device)
    if args.stage == "fixed":
        out, arrays = fixed_result(rules, args.data_path, device, args.max_episodes)
        out["protocol"] = protocols
        save(out, arrays, "fixed" + args.tag)
        for name, c in out["checkpoints"].items():
            pr = c["paired"]
            print(f"[fixed] {name}: model {100 * c['miou']['model']:.2f} | cf-a {c['cf_a_gain']:+.2f} | cf-b "
                  f"{c['cf_b_gain']:+.2f} | tbn {pr['tbn_vs_model']['gain']:+.2f} | oracle "
                  f"{pr['oracle_unit_vs_model']['gain']:+.2f} | presence-fair {pr['presence_fair_vs_model']['gain']:+.2f}"
                  + (f" | dual-control {pr['dual_vs_control']['gain']:+.2f}" if "dual_vs_control" in pr else ""),
                  flush=True)
        return 0
    if len(rules) != 1 or not isinstance(next(iter(rules.values())), r2.OursRule):
        raise ValueError("intervene and leakfree take one of our checkpoints (E1) [DECISION D-35]")
    name, rule = next(iter(rules.items()))
    if args.stage == "intervene":
        out = score_intervention(rule, args.data_path, device, args.per_pair, args.max_episodes)
        out.update({"checkpoint": name, "protocol": protocols[name]})
        save(out, None, "intervene" + args.tag)
        nan = float("nan")
        print(f"[intervene] events {out.get('events', 0)} | recall of the re-sampled class V0/V1/V2 "
              f"{out.get('r_v0', nan):.3f} / {out.get('r_v1', nan):.3f} / {out.get('r_v2', nan):.3f} | own "
              f"{out.get('r_own', nan):.3f} | phi {out.get('phi', nan):.2f}", flush=True)
        return 0
    out, arrays = score_leakfree(rule, args.data_path, device, args.per_pair, args.max_episodes)
    out.update({"checkpoint": name, "protocol": protocols[name]})
    save(out, arrays, "leakfree" + args.tag)
    print("[leakfree] " + " | ".join(f"{k} {100 * v:.2f}" for k, v in out["miou"].items()), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
