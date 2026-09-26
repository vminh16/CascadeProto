"""P9 of [DECISION D-44]: where density still enters M1 (D-43), and which module the one remaining run can use.
Inference only, S1.

  geometry  step 0 (CPU, no model): M1's grouping replayed on P8's arm-B events (seed 3); m = distinct points among
            the 16 a ball keeps, per stage, for the centres of the re-sampled class c (V0, V1) and the own class a
            (V0, V2)
  trace     part A (GPU, M1 `last.pt`): A1 the cosine between the V0 and V1 (V0 and V2) features of the same raw
            points per encoder slice; A2 the ball-count intervention (caps drawn from the sparse version's m) and
            psi = (R_ref - R_cap) / (R_ref - R_low)
  modules   part B (GPU, CR `last.pt`, valid): D-42's P9.3-P9.8 (metric oracles and label-free LDA, the base-class
            nuisance subspace, head shares, foreground components, collapse, retrieval purity)
  decide    rules D44.0-D44.4

    python experiments/p9_placement_probe.py geometry --data_path datasets/S3DIS/blocks_bs1_s1
    python experiments/p9_placement_probe.py trace --data_path datasets/S3DIS/blocks_bs1_s1 --checkpoint m1:ours:1:<M1 last.pt>
    python experiments/p9_placement_probe.py modules --data_path datasets/S3DIS/blocks_bs1_s1 --checkpoint cr:ours:1:<CR last.pt>
    python experiments/p9_placement_probe.py decide
"""

import argparse
import json
import os
import sys
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from experiments import p0_em_probe as p0  # noqa: E402
from experiments import p5_condition_probe as p5  # noqa: E402
from experiments import p6_prototype_probe as p6  # noqa: E402
from experiments import p8_condition_probe as p8  # noqa: E402
from models.density_ops import BALL_K, BALL_RADII, ball_group  # noqa: E402
from models.oracle_distill import oracle_directions  # noqa: E402

OUT_DIR = "results/phase16_p9"
GROUPS = (1024, 512, 256)  # FPS centres per stage: 2,048 halved three times [VIPSEG models/encoder.py:493-503]
SLICES = {"emb": (0, 60), "stage1": (60, 180), "stage2": (180, 420), "stage3": (420, 900)}  # decoder concat order
FPS_SKIP_MAG = 1e-3  # pointnet2's FPS never selects a point with x² + y² + z² <= 1e-3 [sampling_gpu.cu:100-101]
VERSIONS = ("V0", "V1", "V2")
CAP_ARMS = {"s1": (0,), "s2": (1,), "s3": (2,), "all": (0, 1, 2)}  # stages capped (0-based)
DIRECTIONS = ("other", "own")  # other: V1's c-centres capped to V0's m; own: V0's a-centres capped to V2's m
CAUSAL, NOT_CAUSAL = p8.PHI_CAUSAL, p8.PHI_NOT  # D-35's bands, 0.5 / 0.2 [DECISION D-44]
STEP0_FULL = 15.5  # D44.0: mean m of V0's c-centres at least this at every stage -> no cap arms
ENCODE_RTOL = 1e-3  # a block encoded alone against the model's pair encoding (DE-10's floor 1.3e-4)
REF_TOL = 0.005  # part A's reference recalls against D-43's intervene_m1.json
D43_INTERVENE = "results/phase16_d43/intervene_m1.json"
P8_VALID = "results/phase16_p8/split_S1_valid.json"
U_TOL = p6.CR_TOL  # U on valid against P8's split of CR (0.01 points)
LAMBDAS = (0.1, 0.3, 0.5)  # covariance shrinkage [DECISION D-42]
RANKS = (4, 8, 16)  # nuisance subspace ranks [DECISION D-42]
HEADS = (4, 8)  # head partitions [DECISION D-42]
KS = (2, 3)  # foreground k-means components [DECISION D-42]
PURITY_K = 16  # support neighbours per query point [DECISION D-42]
MIN_POINTS = p6.MIN_POINTS  # a class needs this many points in a block for o_{c,beta}
TRAIN_EPISODES = 1500  # base-class episodes for Sigma_eta [DECISION D-42]
TRAIN_SEED = 0
GAIN = 0.5  # P9.3, P9.4, P9.6: a label-free rule over U on valid
ORACLE_GAIN = 1.0  # P9.3: LDA oracle over the cosine oracle
CV_MIN, SHARE_MAX, PURITY_MIN = 0.5, 0.5, 0.5  # P9.5, P9.7, P9.8, conventions [DECISION D-42]
BOOT = 2000


# ------------------------------------------------------------------ step 0: geometry (pure)

def fps(xyz: np.ndarray, g: int) -> np.ndarray:
    """pointnet2's `furthest_point_sample` replayed in float32 [sampling_gpu.cu:70-110]: start at index 0, then the
    point farthest from the chosen set; points with x² + y² + z² <= 1e-3 are never chosen. xyz [N, 3] -> [g]."""
    x = np.asarray(xyz, dtype=np.float32)  # [N, 3]
    n = x.shape[0]
    if not 0 < g <= n:
        raise ValueError(f"cannot choose {g} of {n} points")
    eligible = (x * x).sum(axis=1) > np.float32(FPS_SKIP_MAG)  # [N]
    temp = np.full(n, np.float32(1e10), dtype=np.float32)  # [N]
    out = np.zeros(g, dtype=np.int64)
    old = 0
    for j in range(1, g):
        d = ((x - x[old]) ** 2).sum(axis=1, dtype=np.float32)  # [N]
        temp = np.where(eligible, np.minimum(temp, d), temp)
        cand = np.where(eligible, temp, np.float32(-1.0))  # [N]
        old = int(np.argmax(cand))  # first index of the maximum
        out[j] = old
    return out


def distinct_mask(xyz: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """[B, G, K] True where entry j of a ball holds coordinates that no earlier entry holds. xyz [B, N, 3],
    idx [B, G, K]."""
    b, g, k = idx.shape
    coords = torch.gather(xyz.unsqueeze(1).expand(b, g, -1, -1), 2, idx.unsqueeze(-1).expand(b, g, k, 3))  # [B,G,K,3]
    same = (coords.unsqueeze(3) == coords.unsqueeze(2)).all(dim=-1)  # [B, G, K, K], [j, i]
    earlier = torch.ones(k, k, dtype=torch.bool, device=idx.device).tril(diagonal=-1)  # [K, K], i < j
    return ~(same & earlier).any(dim=-1)  # [B, G, K]


def distinct_count(xyz: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """m [B, G]: distinct points among the entries of each ball."""
    return distinct_mask(xyz, idx).sum(dim=-1)  # [B, G]


def cap_indices(xyz: torch.Tensor, idx: torch.Tensor, cap: torch.Tensor) -> torch.Tensor:
    """idx [B, G, K] with each ball cut to its first `cap` [B, G] distinct points, the rest padded with the first
    entry, as `ball_group` pads [DECISION D-44]. cap >= m leaves the ball unchanged."""
    if bool((cap < 1).any()):
        raise ValueError("a cap keeps at least one point")
    new = distinct_mask(xyz, idx)  # [B, G, K]
    keep = new.long().cumsum(dim=-1) <= cap.unsqueeze(-1)  # [B, G, K]
    return torch.where(keep, idx, idx[..., :1].expand_as(idx))  # [B, G, K]


def replay(xyz: np.ndarray, labels: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    """M1's grouping on one block [DECISION D-44]: per stage (m [G], centre labels [G]). xyz [P, 3] metric, labels
    [P]; stage s+1 groups the centres of stage s."""
    pts = torch.from_numpy(np.asarray(xyz, dtype=np.float32))  # [N, 3]
    lab = np.asarray(labels)
    out = []
    for g, r in zip(GROUPS, BALL_RADII):
        c = fps(pts.numpy(), g)  # [G]
        centres = pts[torch.from_numpy(c)]  # [G, 3]
        nb = ball_group(pts.unsqueeze(0), centres.unsqueeze(0), r, BALL_K)  # [1, G, K]
        m = distinct_count(pts.unsqueeze(0), nb)[0].numpy()  # [G]
        out.append((m, lab[c]))
        pts, lab = centres, lab[c]
    return out


def m_stats(ms: List[np.ndarray]) -> Dict[str, float]:
    """Mean of m, share below K and at most K/2, over the centres pooled from `ms`."""
    m = np.concatenate(ms) if ms else np.zeros(0)
    if m.size == 0:
        return {"centres": 0}
    return {"centres": int(m.size), "mean": float(m.mean()), "share_lt_k": float((m < BALL_K).mean()),
            "share_le_half": float((m <= BALL_K // 2).mean())}


# ------------------------------------------------------------------ part A (pure)

def psi(r_ref: float, r_cap: float, r_low: float) -> float:
    """Share of the density gap R_ref - R_low that the cap reproduces [DECISION D-44]."""
    gap = r_ref - r_low
    return float("nan") if abs(gap) < 1e-12 else (r_ref - r_cap) / gap


def psi_summary(ep: np.ndarray, gt: np.ndarray, tp: np.ndarray, seed: int = 0, boot: int = BOOT) -> Dict[str, float]:
    """gt, tp [E, 3]: (reference, capped, low) counts per event; psi pooled over events, CI by resampling episodes."""
    def value(sel: np.ndarray) -> float:
        r = tp[sel].sum(0) / np.maximum(gt[sel].sum(0), 1.0)  # [3]
        return psi(r[0], r[1], r[2])

    everything = np.ones(len(ep), dtype=bool)
    r = tp.sum(0) / np.maximum(gt.sum(0), 1.0)
    rng = np.random.default_rng(seed)
    uniq = np.unique(ep)
    rows = {u: np.nonzero(ep == u)[0] for u in uniq}
    vals = []
    for _ in range(boot):
        sel = np.concatenate([rows[u] for u in rng.choice(uniq, uniq.size, replace=True)])
        vals.append(value(sel))
    return {"r_ref": float(r[0]), "r_cap": float(r[1]), "r_low": float(r[2]), "psi": value(everything),
            "psi_ci": [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))], "events": len(ep)}


def draw_caps(centre_labels: np.ndarray, target: int, source_m: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """[G] caps: K everywhere, and for each centre labelled `target` an m drawn from `source_m` (the m of the same
    class's centres in the sparse version). An empty source leaves every cap at K."""
    cap = np.full(centre_labels.shape[0], BALL_K, dtype=np.int64)
    sel = np.nonzero(centre_labels == target)[0]
    if sel.size and source_m.size:
        cap[sel] = rng.choice(source_m, sel.size, replace=True)
    return cap


def shared_positions(idx_a: np.ndarray, idx_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Positions in two blocks of the raw points both hold (first occurrence in each). idx_* [P] raw indices."""
    ua, pa = np.unique(idx_a, return_index=True)
    ub, pb = np.unique(idx_b, return_index=True)
    _, ia, ib = np.intersect1d(ua, ub, assume_unique=True, return_indices=True)
    return pa[ia], pb[ib]


def slice_cosines(enc_a: torch.Tensor, enc_b: torch.Tensor, f_a: torch.Tensor, f_b: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Per point cosine of two encodings of the same points, per encoder slice and for the feature.
    enc_* [900, M], f_* [M, 128] -> {name: [M]}."""
    out = {name: F.cosine_similarity(enc_a[lo:hi], enc_b[lo:hi], dim=0) for name, (lo, hi) in SLICES.items()}
    out["feature"] = F.cosine_similarity(f_a, f_b, dim=-1)  # [M]
    return out


# ------------------------------------------------------------------ part B (pure)

def shrink(cov: torch.Tensor, lam: float) -> torch.Tensor:
    """(1 - lam) cov + lam (tr cov / D) I for cov [..., D, D]."""
    d = cov.shape[-1]
    eye = torch.eye(d, dtype=cov.dtype, device=cov.device)
    tr = cov.diagonal(dim1=-2, dim2=-1).sum(-1)[..., None, None]  # [..., 1, 1]
    return (1.0 - lam) * cov + lam * tr / d * eye


def lda_logits(u: torch.Tensor, means: torch.Tensor, cov: torch.Tensor, centre: Optional[torch.Tensor] = None) -> torch.Tensor:
    """One block's LDA scores with one shared metric [DECISION D-42]: m_cᵀ C⁻¹ (u_i - c) - ½ (m_c - c)ᵀ C⁻¹ (m_c - c).
    u [P, D], means [N+1, D], cov [D, D], centre [D] (zero for the oracle form) -> [P, N+1]. float64 solve."""
    c = torch.zeros_like(u[0]) if centre is None else centre  # [D]
    cov64 = cov.double()
    a = torch.linalg.solve(cov64, (means - c).double().T)  # [D, N+1], C⁻¹ (m_c - c)
    if centre is None:
        lin = u.double() @ a  # [P, N+1], m_cᵀ C⁻¹ u_i
    else:
        lin = (u - c).double() @ torch.linalg.solve(cov64, means.double().T)  # [P, N+1], m_cᵀ C⁻¹ (u_i - c)
    quad = 0.5 * ((means - c).double() * a.T).sum(-1)  # [N+1]
    return (lin - quad).to(u.dtype)  # [P, N+1]


def class_means(u: torch.Tensor, labels: torch.Tensor, n_cls: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Means of unit features per label [N+1, D] and presence [N+1]. u [P, D], labels [P]."""
    onehot = F.one_hot(labels, n_cls).to(u.dtype)  # [P, N+1]
    cnt = onehot.sum(0)  # [N+1]
    return (onehot.T @ u) / cnt.clamp_min(1.0).unsqueeze(-1), cnt > 0  # [N+1, D], [N+1]


def within_cov(u: torch.Tensor, labels: torch.Tensor, means: torch.Tensor) -> torch.Tensor:
    """Pooled within-class covariance (1/P) Σ_i (u_i - m_{y_i})(u_i - m_{y_i})ᵀ [D, D]."""
    r = u - means[labels]  # [P, D]
    return r.T @ r / u.shape[0]


def total_cov(u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mean [D] and total covariance [D, D] of a block's unit features."""
    mu = u.mean(0)  # [D]
    r = u - mu  # [P, D]
    return mu, r.T @ r / u.shape[0]


def support_means(f_s: torch.Tensor, support_y: torch.Tensor) -> torch.Tensor:
    """[N+1, D] means of the support's unit features: background over every way and shot, then each way's foreground."""
    u = F.normalize(f_s, dim=-1)  # [N, K, P, D]
    fg = (support_y == 1).to(u.dtype)  # [N, K, P]
    m_fg = torch.einsum("nkp,nkpd->nd", fg, u) / fg.sum(dim=(1, 2)).clamp_min(1.0).unsqueeze(-1)  # [N, D]
    bg = 1.0 - fg
    m_bg = torch.einsum("nkp,nkpd->d", bg, u) / bg.sum().clamp_min(1.0)  # [D]
    return torch.cat([m_bg.unsqueeze(0), m_fg], dim=0)  # [N+1, D]


def subspace_share(v: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """‖Bᵀv‖² / ‖v‖² for v [..., D] and an orthonormal basis [D, r]."""
    return (v @ basis).pow(2).sum(-1) / v.pow(2).sum(-1).clamp_min(1e-30)


def projected_logits(f_q: torch.Tensor, rows: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """U with features and rows projected by P⊥ = I - BBᵀ and re-normalised [B_q, P, N+1] [DECISION D-42].
    f_q [B_q, P, D], rows [B_q, N+1, D], basis [D, r]."""
    def perp(x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x - (x @ basis) @ basis.T, dim=-1)

    return torch.einsum("bpd,bcd->bpc", perp(f_q), perp(rows))


def head_shares(v: torch.Tensor, basis: torch.Tensor, n_heads: int) -> torch.Tensor:
    """[..., H] shares ‖W_h v‖² / ‖v‖² for the partition of an orthonormal basis [D, D] into H equal column blocks."""
    d = basis.shape[1]
    if d % n_heads:
        raise ValueError(f"{d} columns do not split into {n_heads} heads")
    coef = (v @ basis).pow(2)  # [..., D]
    return coef.unflatten(-1, (n_heads, d // n_heads)).sum(-1) / v.pow(2).sum(-1, keepdim=True).clamp_min(1e-30)


def ratio_cv(e: np.ndarray, g: np.ndarray) -> float:
    """Coefficient of variation over heads of mean g_h / mean e_h. e, g [R, H]."""
    ratio = g.mean(0) / np.maximum(e.mean(0), 1e-30)  # [H]
    return float(ratio.std() / max(ratio.mean(), 1e-30))


def participation_ratio(cov: torch.Tensor) -> float:
    """(Σ λ)² / Σ λ² of a covariance [D, D]."""
    lam = torch.linalg.eigvalsh(cov.double()).clamp_min(0.0)
    return float(lam.sum() ** 2 / lam.pow(2).sum().clamp_min(1e-300))


def orthonormal(vectors: torch.Tensor) -> torch.Tensor:
    """[D, r] orthonormal basis of the span of the rows of `vectors` [r, D] (QR, float64)."""
    q, _ = torch.linalg.qr(vectors.double().T)
    return q  # [D, r]


def component_logits(f_q: torch.Tensor, rows: torch.Tensor, f_s: torch.Tensor, support_y: torch.Tensor,
                     k: int) -> torch.Tensor:
    """U with each foreground row the max over k spherical k-means components of the way's support foreground
    [B_q, P, N+1] (P6's `spherical_kmeans`) [DECISION D-42]."""
    logits = p6.rule_logits(f_q, rows).clone()  # [B_q, P, N+1]
    for w in range(f_s.shape[0]):
        units = F.normalize(f_s[w][support_y[w] == 1], dim=-1)  # [M, D]
        cent = p6.spherical_kmeans(units, k)  # [k, D]
        logits[..., w + 1] = torch.einsum("bpd,kd->bpk", f_q, cent).max(dim=-1).values
    return logits


def retrieval_purity(u_q: torch.Tensor, y_q: torch.Tensor, u_s: torch.Tensor, y_s: torch.Tensor,
                     k: int = PURITY_K) -> torch.Tensor:
    """[M] share of each query point's k nearest support points (cosine) with its label. u_q [M, D], y_q [M],
    u_s [S, D], y_s [S]; unit rows."""
    nn_idx = (u_q @ u_s.T).topk(k, dim=-1).indices  # [M, k]
    return (y_s[nn_idx] == y_q.unsqueeze(-1)).to(u_q.dtype).mean(-1)  # [M]


def shift_vectors(f_q: torch.Tensor, labels: torch.Tensor, rows: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per (query block, present foreground class with a present background): Δ_c - Δ_0 and d* = o_c - o_0 [R, D],
    Δ = o - s with o the query's own unit direction and s the support direction [DECISION D-42]."""
    o, present = oracle_directions(f_q, labels, rows.shape[1])  # [B_q, N+1, D], [B_q, N+1]
    delta = o - rows  # [B_q, N+1, D]
    dd, ds = [], []
    for b in range(f_q.shape[0]):
        if not bool(present[b, 0]):
            continue
        for c in range(1, rows.shape[1]):
            if bool(present[b, c]):
                dd.append(delta[b, c] - delta[b, 0]), ds.append(o[b, c] - o[b, 0])
    d = f_q.shape[-1]
    if not dd:
        return f_q.new_zeros(0, d), f_q.new_zeros(0, d)
    return torch.stack(dd), torch.stack(ds)  # [R, D] x 2


# ------------------------------------------------------------------ rules

def decide(geo: Optional[Dict], trace: Optional[Dict], modules: Optional[Dict]) -> List[Tuple[str, str]]:
    """Rules D44.0-D44.3 of [DECISION D-44]."""
    v = []
    if geo is not None:
        means = [geo["stages"][str(s)]["c_V0"].get("mean", float("inf")) for s in range(len(GROUPS))]
        full = all(x >= STEP0_FULL for x in means)
        v.append(("D44.0 " + ("ball counts full: no cap arms" if full else "sparse balls: cap arms run"),
                  "mean m of V0's c-centres per stage " + ", ".join(f"{x:.2f}" for x in means)
                  + f" (limit {STEP0_FULL})"))
    if trace is not None and "caps" in trace:
        p = {d: trace["caps"][d]["all"]["psi"] for d in DIRECTIONS}
        if any(x >= CAUSAL for x in p.values()):
            stages = [a for a in ("s1", "s2", "s3") if any(trace["caps"][d][a]["psi"] >= CAUSAL for d in DIRECTIONS)]
            v.append(("D44.1 the ball count carries density: M1b is a candidate",
                      f"psi all stages other {p['other']:.3f}, own {p['own']:.3f}; stages {stages or ['s1', 's2', 's3']}"))
        elif all(x <= NOT_CAUSAL for x in p.values()):
            v.append(("D44.1 the count is not the pathway: encoder branch closed",
                      f"psi all stages other {p['other']:.3f}, own {p['own']:.3f}"))
        else:
            v.append(("D44.1 undecided: encoder branch not trained on this budget",
                      f"psi all stages other {p['other']:.3f}, own {p['own']:.3f}"))
    elif trace is not None:
        v.append(("D44.1 not run", "step 0 found full balls"))
    if modules is not None:
        m, u = modules["miou"], modules["miou"]["U"]
        lda_or = max(m[f"lda_oracle_{lam}"] for lam in LAMBDAS)
        lda_free = max(m[f"lda_free_{lam}"] for lam in LAMBDAS)
        ok3 = lda_or >= m["cos_oracle"] + ORACLE_GAIN / 100 or lda_free >= u + GAIN / 100
        v.append((f"D44.2 P9.3 metric {'admissible' if ok3 else 'fails'}",
                  f"LDA oracle {100 * lda_or:.2f} vs cosine oracle {100 * m['cos_oracle']:.2f}; label-free LDA "
                  f"{100 * lda_free:.2f} vs U {100 * u:.2f}"))
        best_r = max(RANKS, key=lambda r: m[f"proj_{r}"])
        kr = modules["nuisance"][str(best_r)]
        ok4 = kr["kappa_median"] > kr["rho_median"] and m[f"proj_{best_r}"] >= u + GAIN / 100
        v.append((f"D44.2 P9.4 nuisance {'admissible' if ok4 else 'fails'}",
                  f"r {best_r}: kappa {kr['kappa_median']:.3f} rho {kr['rho_median']:.3f}, projected U "
                  f"{100 * m[f'proj_{best_r}']:.2f} vs U {100 * u:.2f}"))
        cvs = {h: modules["heads"]["pca"][str(h)]["cv"] for h in HEADS}
        ok5 = max(cvs.values()) >= CV_MIN
        v.append((f"D44.2 P9.5 heads {'admissible' if ok5 else 'fails'}",
                  "cv (PCA) " + ", ".join(f"H {h}: {c:.3f}" for h, c in cvs.items()) + f" (limit {CV_MIN})"))
        best_k = max(KS, key=lambda k: m[f"km_{k}"])
        ok6 = m[f"km_{best_k}"] >= u + GAIN / 100
        v.append((f"D44.2 P9.6 components {'admissible' if ok6 else 'fails'}",
                  f"k {best_k}: {100 * m[f'km_{best_k}']:.2f} vs U {100 * u:.2f}"))
        share = modules["collapse"]["span_share_median"]
        ok7 = share <= SHARE_MAX
        v.append((f"D44.2 P9.7 representation {'admissible' if ok7 else 'fails'}",
                  f"base-span share of d* {share:.3f} (limit {SHARE_MAX}); participation ratio "
                  f"{modules['collapse']['participation_ratio']:.1f}"))
        pur = modules["purity"]["miss"]
        ok8 = pur >= PURITY_MIN
        v.append((f"D44.2 P9.8 neck {'admissible' if ok8 else 'fails'}",
                  f"purity missed {pur:.3f}, hit {modules['purity']['hit']:.3f} (limit {PURITY_MIN})"))
    v.append(("D44.3", "P9 decides admissibility only; the run is chosen in a new decision"))
    return v


# ------------------------------------------------------------------ event iteration (data, no model)

def raw_indices(data_path: str, scan: str, cls: int, uniform: bool, seed, block: np.ndarray) -> np.ndarray:
    """The sampler's raw point indices of a block drawn by `sample_block` under `seed`, checked against its xyz."""
    from pipeline.episodes import NUM_POINT

    data = np.load(os.path.join(data_path, "data", f"{scan}.npy"))
    np.random.seed(seed)
    idx = p5.sampled_indices(data, NUM_POINT, cls, uniform)  # [P]
    if not np.allclose(data[idx, :3] - data[idx, :3].min(axis=0), block[:, :3]):
        raise RuntimeError(f"{scan}: the raw indices do not reproduce the block")
    return idx


def events(data_path: str, max_episodes: Optional[int]) -> Iterator[Dict]:
    """P8's arm-B events (seed 3): V0 / V1 / V2 of each eligible query block with raw indices and labels."""
    seed = p5.DRAW_SEEDS["intervene"]
    for ep in p5.draw_episodes(data_path, 1, seed, p5.EPISODES_PER_PAIR, False, max_episodes):
        sx, sy, qx, qy, classes = ep["item"]
        if not p8.has_support_background(sy):
            continue  # as P8 [DECISION D-41, amended]
        for b in range(qy.shape[0]):
            a, c_local = int(classes[b]), 2 - b
            c = int(classes[c_local - 1])
            if p5.raw_count(data_path, ep["q_scans"][b], c) < p5.MIN_RAW_POINTS:
                continue
            scan = ep["q_scans"][b]
            within = ep["index"] - ep["pair"] * p5.EPISODES_PER_PAIR  # draw_episodes seeds V0 by the index within the pair
            blocks = {"V0": (qx[b], qy[b].astype(np.int64),
                             raw_indices(data_path, scan, a, False, [seed, ep["pair"], within, 0, b], qx[b]))}
            for v, (cls, uniform) in ((1, (c, False)), (2, (a, True))):
                s = [seed, ep["pair"], ep["index"], 2 + v, b]
                x, y, _ = p5.sample_block(data_path, scan, classes, cls, uniform, s)
                blocks[f"V{v}"] = (x, y, raw_indices(data_path, scan, cls, uniform, s, x))
            yield {"episode": ep["index"], "pair": ep["pair"], "b": b, "a": a, "c": c, "c_local": c_local,
                   "a_local": b + 1, "item": ep["item"], "blocks": blocks}


def geometry(data_path: str, max_episodes: Optional[int]) -> Dict:
    """Step 0 [DECISION D-44]."""
    pools = {s: {k: [] for k in ("c_V0", "c_V1", "a_V0", "a_V2")} for s in range(len(GROUPS))}
    n = 0
    for ev in events(data_path, max_episodes):
        n += 1
        for v in VERSIONS:
            x, y, _ = ev["blocks"][v]
            for s, (m, lab) in enumerate(replay(x[:, :3], y)):
                if v in ("V0", "V1"):
                    pools[s][f"c_{v}"].append(m[lab == ev["c_local"]])
                if v in ("V0", "V2"):
                    pools[s][f"a_{v}"].append(m[lab == ev["a_local"]])
    return {"events": n, "radii": list(BALL_RADII), "groups": list(GROUPS), "k": BALL_K,
            "stages": {str(s): {k: m_stats(v) for k, v in pools[s].items()} for s in pools}}  # JSON keys


# ------------------------------------------------------------------ part A (GPU)

class Plan:
    """State shared by the capped groupings of one encoding: labels per stage, recorded m, caps per stage."""

    def __init__(self, labels: Optional[torch.Tensor] = None,
                 caps: Optional[Dict[int, Callable[[torch.Tensor], torch.Tensor]]] = None):
        self.labels = {0: labels}
        self.m: Dict[int, torch.Tensor] = {}
        self.centre_labels: Dict[int, torch.Tensor] = {}
        self.caps = caps or {}


def capped_grouping_class():
    """The capped grouping, defined on demand: it subclasses M1's grouping, which needs pointnet2_ops (GPU)."""
    from models.density_encoder import MetricBallGrouping
    from models.model_utils import index_points
    from pointnet2_ops_lib.pointnet2_ops import pointnet2_utils

    class CappedBallGrouping(MetricBallGrouping):
        """M1's `MetricBallGrouping` that records m and the centre labels and can cap balls [DECISION D-44]."""

        def __init__(self, base, stage: int):
            super().__init__(base.group_num, base.k, base.radius)
            self.stage, self.plan = stage, Plan()

        def forward(self, xyz, x, rgb):
            fps_idx = pointnet2_utils.furthest_point_sample(xyz.contiguous(), self.group_num).long()  # [B, G]
            lc_xyz, lc_x, lc_rgb = index_points(xyz, fps_idx), index_points(x, fps_idx), index_points(rgb, fps_idx)
            idx = ball_group(xyz, lc_xyz, self.radius, self.k)  # [B, G, K]
            plan = self.plan
            lab = plan.labels.get(self.stage)
            if lab is not None:
                c_lab = torch.gather(lab, 1, fps_idx)  # [B, G]
                plan.labels[self.stage + 1] = c_lab
                plan.centre_labels[self.stage] = c_lab
                if self.stage in plan.caps:
                    idx = cap_indices(xyz, idx, plan.caps[self.stage](c_lab))  # [B, G, K]
            plan.m[self.stage] = distinct_count(xyz, idx)  # [B, G]
            return lc_xyz, lc_x, lc_rgb, index_points(xyz, idx), index_points(x, idx), index_points(rgb, idx)

    return CappedBallGrouping


def install_capped(model) -> List:
    """Replace the three groupings of the loaded M1 instance by capped ones (instance attributes, no global patch)."""
    from models.density_encoder import DensityEncoder

    enc = model.features.encoder
    if not isinstance(enc, DensityEncoder):
        raise ValueError("part A reads M1 (encoder=density) [DECISION D-44]")
    cls = capped_grouping_class()
    mods = []
    for i in range(enc.EncNP.num_stages):
        mods.append(cls(enc.EncNP.FPS_kNN_list[i], i))
        enc.EncNP.FPS_kNN_list[i] = mods[-1]
    return mods


def encode_block(model, mods: List, x: torch.Tensor, plan: Plan) -> Tuple[torch.Tensor, torch.Tensor]:
    """One block encoded alone under a plan -> (encoder output [900, P], feature [P, 128])."""
    store = {}
    for m in mods:
        m.plan = plan
    hook = model.features.encoder.register_forward_hook(lambda mod, inp, out: store.__setitem__("enc", out))
    try:
        f = model.features(x.unsqueeze(0))  # [1, P, 128]
    finally:
        hook.remove()
        for m in mods:
            m.plan = Plan()
    return store["enc"][0], f[0]


@torch.no_grad()
def trace(rule, data_path: str, device, max_episodes: Optional[int], run_caps: bool) -> Dict:
    """Part A [DECISION D-44] on M1."""
    from pipeline.episodes import make_episode, read_class_names

    names = read_class_names(data_path, "s3dis")
    model = p5.module_of(rule)
    mods = install_capped(model)
    rec = {d: {a: [] for a in CAP_ARMS} for d in DIRECTIONS}  # [episode, gt_ref, gt_cap, gt_low, tp_ref, tp_cap, tp_low]
    refs = {k: [0.0, 0.0] for k in ("c_V0", "c_V1", "a_V0", "a_V2")}  # [gt, tp]
    cos_sum = {}  # (pair, group, slice) -> [sum, count]
    checks = {"encode_max_rel": 0.0, "uncapped_equal": 0}
    last_ep, rows, f_model, n_events = None, None, None, 0
    for n_ev, ev in enumerate(events(data_path, max_episodes)):
        n_events += 1
        if ev["episode"] != last_ep:
            e = make_episode(ev["item"], names).to(device)
            f_q, m_eff, _, logits = rule(e)
            p0.check_identity(f_q, m_eff, logits)
            rows = p5.support_directions(p5.support_features(rule, e), e.support_y)  # [N+1, D]
            f_model, last_ep = f_q, ev["episode"]

        def run(v: str, caps=None):
            x, y, _ = ev["blocks"][v]
            plan = Plan(torch.as_tensor(y, device=device).unsqueeze(0), caps)
            enc, f = encode_block(model, mods, torch.as_tensor(x, device=device), plan)
            pred = (f @ rows.T).argmax(-1).cpu().numpy()  # [P]
            return enc, f, pred, plan

        def counts(y: np.ndarray, pred: np.ndarray, k: int) -> Tuple[float, float]:
            return p8.class_counts(y, pred, k)

        out = {v: run(v) for v in VERSIONS}
        rel = float((out["V0"][1] - f_model[ev["b"]]).abs().max() / f_model[ev["b"]].abs().max())
        checks["encode_max_rel"] = max(checks["encode_max_rel"], rel)
        if rel > ENCODE_RTOL:
            raise RuntimeError(f"a block encoded alone differs from the model's encoding by {rel:.2e}")
        y = {v: ev["blocks"][v][1] for v in VERSIONS}
        g = {("c", v): counts(y[v], out[v][2], ev["c_local"]) for v in ("V0", "V1")}
        g.update({("a", v): counts(y[v], out[v][2], ev["a_local"]) for v in ("V0", "V2")})
        for (cls, v), (gt, tp) in g.items():
            refs[f"{cls}_{v}"][0] += gt
            refs[f"{cls}_{v}"][1] += tp
        for pair, other, cls_local in (("V0-V1", "V1", ev["c_local"]), ("V0-V2", "V2", ev["a_local"])):
            pa, pb = shared_positions(ev["blocks"]["V0"][2], ev["blocks"][other][2])
            ya = y["V0"][pa]
            ta, tb = torch.as_tensor(pa, device=device), torch.as_tensor(pb, device=device)
            cos = slice_cosines(out["V0"][0][:, ta], out[other][0][:, tb], out["V0"][1][ta], out[other][1][tb])
            for grp, sel in (("class", ya == cls_local), ("background", ya == 0)):
                for name, val in cos.items():
                    s = cos_sum.setdefault(f"{pair}|{grp}|{name}", [0.0, 0])
                    s[0] += float(val[torch.as_tensor(sel, device=device)].sum())
                    s[1] += int(sel.sum())
        if not run_caps:
            continue
        if n_ev < 5:  # the uncapped plan (every cap K) reproduces the plain grouping exactly
            full = {s: (lambda lab: torch.full_like(lab, BALL_K)) for s in range(len(GROUPS))}
            enc_u, f_u, _, _ = run("V1", full)
            if not (torch.equal(enc_u, out["V1"][0]) and torch.equal(f_u, out["V1"][1])):
                raise RuntimeError("the capped grouping with every cap at K differs from the plain one")
            checks["uncapped_equal"] += 1
        spec = {"other": ("V1", "V0", ev["c_local"], "c"), "own": ("V0", "V2", ev["a_local"], "a")}
        for d, (ref_v, src_v, t, cls) in spec.items():
            src_plan = out[src_v][3]
            src_m = {s: src_plan.m[s][0][src_plan.centre_labels[s][0] == t].cpu().numpy() for s in range(len(GROUPS))}
            for arm_i, (arm, stages) in enumerate(CAP_ARMS.items()):
                rng = np.random.default_rng([p5.DRAW_SEEDS["intervene"], ev["episode"], ev["b"], arm_i,
                                             DIRECTIONS.index(d)])

                def cap_fn(s):
                    return lambda lab: torch.as_tensor(draw_caps(lab[0].cpu().numpy(), t, src_m[s], rng),
                                                       device=lab.device).unsqueeze(0)

                _, _, pred_cap, _ = run(ref_v, {s: cap_fn(s) for s in stages})
                gt_ref, tp_ref = g[(cls, ref_v)]
                gt_cap, tp_cap = counts(y[ref_v], pred_cap, t)
                gt_low, tp_low = g[(cls, src_v)]
                rec[d][arm].append([ev["episode"], gt_ref, gt_cap, gt_low, tp_ref, tp_cap, tp_low])
    result = {"checks": checks, "events": n_events,
              "refs": {k: v[1] / max(v[0], 1.0) for k, v in refs.items()},
              "cos": {k: v[0] / max(v[1], 1) for k, v in cos_sum.items()},
              "cos_points": {k: v[1] for k, v in cos_sum.items()}}
    if run_caps:
        result["caps"] = {}
        for d in DIRECTIONS:
            result["caps"][d] = {}
            for arm in CAP_ARMS:
                a = np.array(rec[d][arm], dtype=np.float64).reshape(-1, 7)
                result["caps"][d][arm] = psi_summary(a[:, 0], a[:, 1:4], a[:, 4:7])
    return result


def p8_events() -> int:
    """Number of arm-B events of P8 and D-43 (1,045)."""
    with open(os.path.join(REPO, D43_INTERVENE)) as f:
        return int(json.load(f)["U"]["events"])


def check_trace_refs(result: Dict) -> None:
    """Part A's events and reference recalls against D-43's intervene_m1.json (per-block against pair encoding)."""
    with open(os.path.join(REPO, D43_INTERVENE)) as f:
        u = json.load(f)["U"]
    if result["events"] != u["events"]:
        raise RuntimeError(f"{result['events']} events, D-43 had {u['events']}")
    pairs = {"c_V0": u["r_v0"], "c_V1": u["r_v1"], "a_V0": u["r_a_v0"], "a_V2": u["r_a_v2"]}
    for k, ref in pairs.items():
        if abs(result["refs"][k] - ref) > REF_TOL:
            raise RuntimeError(f"{k}: {result['refs'][k]:.4f} against D-43's {ref:.4f}")


# ------------------------------------------------------------------ part B (GPU)

def train_statistics(rule, data_path: str, device, n_episodes: int) -> Dict[str, torch.Tensor]:
    """Base-class o_{c,beta} from training episodes without augmentation [DECISION D-44] -> Sigma_eta, the base-class
    means and the background mean (float64)."""
    from dataloaders.loader import MyDataset
    from pipeline.episodes import (N_QUERIES, NUM_POINT, PC_ATTRIBS, RANDOM_SAMPLE, WAY_NUM, WAY_RATIO,
                                   SeededEpisodes, make_episode, read_class_names)

    names = read_class_names(data_path, "s3dis")
    base = MyDataset(data_path, "s3dis", cvfold=1, num_episode=n_episodes, n_way=2, k_shot=1, n_queries=N_QUERIES,
                     phase=None, mode="train", num_point=NUM_POINT, pc_attribs=PC_ATTRIBS, pc_augm=False,
                     pc_augm_config=None, way_ratio=WAY_RATIO, way_num=WAY_NUM, random_sample=RANDOM_SAMPLE)
    ds = SeededEpisodes(base, TRAIN_SEED)
    feats = p5.module_of(rule).features
    per_class: Dict[int, List[torch.Tensor]] = {}
    bg: List[torch.Tensor] = []
    for i in range(min(n_episodes, len(ds))):
        e = make_episode(ds[i], names).to(device)
        f_s, f_q = feats.encode_episode(e.support_x, e.query_x)  # [N, K, P, D], [B_q, P, D]
        blocks = [(f_q[b], e.query_y[b]) for b in range(f_q.shape[0])]
        blocks += [(f_s[w, k], e.support_y[w, k] * (w + 1)) for w in range(f_s.shape[0]) for k in range(f_s.shape[1])]
        for f, y in blocks:
            u = F.normalize(f, dim=-1).double()  # [P, D]
            for lab in torch.unique(y).tolist():
                sel = y == lab
                if int(sel.sum()) < MIN_POINTS:
                    continue
                o = F.normalize(u[sel].sum(0), dim=0)  # [D]
                if lab == 0:
                    bg.append(o)
                else:
                    per_class.setdefault(int(e.sampled_classes[lab - 1]), []).append(o)
    means = {c: torch.stack(v).mean(0) for c, v in per_class.items()}
    dev = torch.cat([torch.stack(v) - means[c] for c, v in per_class.items()])  # [R, D]
    sigma = dev.T @ dev / dev.shape[0]  # [D, D]
    return {"sigma": sigma, "base_means": torch.stack([means[c] for c in sorted(means)]),
            "bg_mean": torch.stack(bg).mean(0), "classes": sorted(means), "vectors": dev.shape[0]}


@torch.no_grad()
def modules(rule, data_path: str, device, max_episodes: Optional[int], train_episodes: int) -> Tuple[Dict, Dict]:
    """Part B [DECISION D-44] on CR's valid draw."""
    from pipeline.episodes import make_episode, read_class_names

    stats = train_statistics(rule, data_path, device, train_episodes)
    evals, evecs = torch.linalg.eigh(stats["sigma"])  # ascending
    bases = {r: evecs[:, -r:].to(torch.float32) for r in RANKS}  # [D, r]
    names = read_class_names(data_path, "s3dis")
    items, test_classes = p6.episodes("valid", data_path, max_episodes)
    counts: Dict[str, List[np.ndarray]] = {}
    dd_all, ds_all = [], []
    s1, s2, n_pts = 0.0, 0.0, 0  # running sums of the valid query unit features: [D], [D, D]
    purity = {"hit": [0.0, 0], "miss": [0.0, 0]}
    common = []
    for item in items:
        e = make_episode(item, names).to(device)
        f_q, m_eff, _, logits = rule(e)
        p0.check_identity(f_q, m_eff, logits)
        f_s = p5.support_features(rule, e)
        y = e.query_y  # [B_q, P]
        rows = p6.base_rows(f_q, f_s, e.support_y)  # [B_q, N+1, D]
        n_cls = rows.shape[1]
        u = F.normalize(f_q, dim=-1)  # [B_q, P, D]
        preds = {"model": logits.argmax(-1), "U": p6.rule_logits(f_q, rows).argmax(-1)}
        o, present = oracle_directions(f_q, y, n_cls)  # [B_q, N+1, D], [B_q, N+1]
        preds["cos_oracle"] = p6.rule_logits(f_q, torch.where(present.unsqueeze(-1), o, rows)).argmax(-1)
        m_s = support_means(f_s, e.support_y)  # [N+1, D]
        for lam in LAMBDAS:
            p_or, p_free = [], []
            for b in range(u.shape[0]):
                mq, pres = class_means(u[b], y[b], n_cls)  # [N+1, D], [N+1]
                means = torch.where(pres.unsqueeze(-1), mq, m_s)  # absent rows: the support's means (presence-fair)
                p_or.append(lda_logits(u[b], means, shrink(within_cov(u[b], y[b], mq), lam)).argmax(-1))
                mu, t = total_cov(u[b])
                p_free.append(lda_logits(u[b], m_s, shrink(t, lam), centre=mu).argmax(-1))
            preds[f"lda_oracle_{lam}"], preds[f"lda_free_{lam}"] = torch.stack(p_or), torch.stack(p_free)
        for r in RANKS:
            preds[f"proj_{r}"] = projected_logits(f_q, rows, bases[r]).argmax(-1)
        for k in KS:
            preds[f"km_{k}"] = component_logits(f_q, rows, f_s, e.support_y, k).argmax(-1)
        gt = y.cpu().numpy()
        for name, pr in preds.items():
            counts.setdefault(name, []).append(p0.episode_counts(pr.cpu().numpy(), gt, e.sampled_classes, test_classes))
        dd, dstar = shift_vectors(f_q, y, rows)
        dd_all.append(dd.double()), ds_all.append(dstar.double())
        flat = u.reshape(-1, u.shape[-1]).double()  # [B_q·P, D], blocks and points only
        s1 += flat.sum(0)
        s2 += flat.T @ flat
        n_pts += flat.shape[0]
        common.append(float(u.mean(1).pow(2).sum(-1).mean()))
        u_s = F.normalize(f_s, dim=-1).reshape(-1, u.shape[-1])  # [N·K·P, D], ways, shots and points
        lab_s = (e.support_y * (torch.arange(e.support_y.shape[0], device=device) + 1).view(-1, 1, 1)).reshape(-1)
        for b in range(u.shape[0]):
            fg = y[b] > 0  # [P]
            if not bool(fg.any()):
                continue
            pur = retrieval_purity(u[b][fg], y[b][fg], u_s, lab_s)  # [M]
            hit = preds["U"][b][fg] == y[b][fg]
            for key, sel in (("hit", hit), ("miss", ~hit)):
                purity[key][0] += float(pur[sel].sum())
                purity[key][1] += int(sel.sum())
    stacked = {k: np.stack(v) for k, v in counts.items()}
    miou = {k: float(p0.miou_from_counts(v.sum(0))) for k, v in stacked.items()}
    if max_episodes is None:
        with open(os.path.join(REPO, P8_VALID)) as f:
            ref = json.load(f)["miou"]["U"]
        if abs(miou["U"] - ref) > U_TOL:
            raise RuntimeError(f"U scores {miou['U']:.6f} on valid, P8's split {ref:.6f}")
    dd, dstar = torch.cat(dd_all), torch.cat(ds_all)  # [R, D]
    mean = s1 / n_pts
    cov = s2 / n_pts - torch.outer(mean, mean)  # [D, D]
    nuisance = {}
    for r in RANKS:
        kap, rho = subspace_share(dd, bases[r].double()), subspace_share(dstar, bases[r].double())
        d_proj = dd - (dd @ bases[r].double()) @ bases[r].double().T
        s_proj = dstar - (dstar @ bases[r].double()) @ bases[r].double().T
        nuisance[str(r)] = {"kappa_median": float(kap.median()), "rho_median": float(rho.median()),
                            "cos_gain_median": float((F.cosine_similarity(s_proj - d_proj, s_proj, dim=-1)
                                                      - F.cosine_similarity(dstar - dd, dstar, dim=-1)).median())}
    _, pca = torch.linalg.eigh(cov)
    pca = pca.flip(-1)  # [D, D], descending variance
    g = torch.Generator(device="cpu").manual_seed(0)
    rand = orthonormal(torch.randn(cov.shape[0], cov.shape[0], generator=g, dtype=torch.float64)).to(device)
    heads = {}
    for kind, basis in (("pca", pca), ("random", rand)):
        heads[kind] = {}
        for h in HEADS:
            e_h = head_shares(dd, basis, h).cpu().numpy()  # [R, H]
            g_h = head_shares(dstar, basis, h).cpu().numpy()  # [R, H]
            heads[kind][str(h)] = {"cv": ratio_cv(e_h, g_h), "e_mean": e_h.mean(0).tolist(), "g_mean": g_h.mean(0).tolist()}
    span = orthonormal(torch.cat([stats["base_means"], stats["bg_mean"].unsqueeze(0)]).to(device))  # [D, 7]
    collapse = {"participation_ratio": participation_ratio(cov),
                "span_share_median": float(subspace_share(dstar, span).median()),
                "span_dim": int(span.shape[1]), "common_component": float(np.mean(common))}
    out = {"draw": "valid", "episodes": len(stacked["U"]), "test_classes": test_classes, "miou": miou,
           "nuisance": nuisance, "heads": heads, "collapse": collapse,
           "purity": {k: v[0] / max(v[1], 1) for k, v in purity.items()},
           "purity_points": {k: v[1] for k, v in purity.items()}, "shift_vectors": int(dd.shape[0]),
           "train": {"episodes": train_episodes, "vectors": stats["vectors"], "classes": stats["classes"],
                     "sigma_top_eigenvalues": evals.flip(0)[:16].tolist()}}
    return out, stacked


# ------------------------------------------------------------------ CLI

def read_json(path: str) -> Optional[Dict]:
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return None


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["geometry", "trace", "modules", "decide"])
    p.add_argument("--data_path")
    p.add_argument("--checkpoint")
    p.add_argument("--max_episodes", type=int, default=None, help="smoke runs only")
    p.add_argument("--train_episodes", type=int, default=TRAIN_EPISODES, help="lowered for smoke runs only")
    p.add_argument("--tag", default="")
    p.add_argument("--out_dir", default=OUT_DIR)
    args = p.parse_args(argv)
    out_dir = os.path.join(REPO, args.out_dir) if not os.path.isabs(args.out_dir) else args.out_dir
    if args.stage == "decide":
        geo = read_json(os.path.join(out_dir, f"geometry{args.tag}.json"))
        tr = read_json(os.path.join(out_dir, f"trace_m1{args.tag}.json"))
        mo = read_json(os.path.join(out_dir, f"modules_cr_valid{args.tag}.json"))
        for name, text in decide(geo, tr, mo):
            print(f"{name:60s} {text}")
        return 0
    if not args.data_path:
        p.error("--data_path is required")
    if args.stage == "geometry":
        res = geometry(args.data_path, args.max_episodes)
        if args.max_episodes is None and res["events"] != p8_events():
            raise RuntimeError(f"{res['events']} events, P8 and D-43 had {p8_events()}")
        p6.save(res, None, f"geometry{args.tag}", out_dir)
        for s, st in res["stages"].items():
            print(f"[geometry] stage {int(s) + 1} r {BALL_RADII[int(s)]}: " + " | ".join(
                f"{k} m {v.get('mean', float('nan')):.2f} <K {v.get('share_lt_k', float('nan')):.3f}"
                for k, v in st.items()) + f" | {res['events']} events", flush=True)
        return 0
    if not args.checkpoint:
        p.error("trace and modules need --checkpoint")
    device = torch.device("cuda")
    rule, protocol, config, ck = p6.load_rule(args.checkpoint, device)
    meta = {"checkpoint": vars(ck), "protocol": protocol, "config": config}
    if args.stage == "trace":
        if config.get("encoder") != "density":
            raise ValueError("trace reads M1 (encoder=density) [DECISION D-44]")
        geo = read_json(os.path.join(out_dir, f"geometry{args.tag}.json"))
        if geo is None:
            raise FileNotFoundError("run the geometry stage first (D44.0 decides the cap arms)")
        run_caps = not decide(geo, None, None)[0][0].startswith("D44.0 ball counts full")
        res = trace(rule, args.data_path, device, args.max_episodes, run_caps)
        if args.max_episodes is None:
            check_trace_refs(res)
        res.update(models=meta, run_caps=run_caps)
        p6.save(res, None, f"trace_m1{args.tag}", out_dir)
        print("[trace] refs " + " ".join(f"{k} {v:.3f}" for k, v in res["refs"].items()), flush=True)
        for d in res.get("caps", {}):
            print(f"[trace] {d}: " + " | ".join(f"{a} psi {s['psi']:.3f} [{s['psi_ci'][0]:.3f}, {s['psi_ci'][1]:.3f}]"
                                                 for a, s in res["caps"][d].items()), flush=True)
        return 0
    if config.get("encoder", "vipseg") != "vipseg":
        raise ValueError("modules reads CR (encoder=vipseg), the base after D-43 [DECISION D-44]")
    res, stacked = modules(rule, args.data_path, device, args.max_episodes, args.train_episodes)
    res.update(models=meta)
    p6.save(res, stacked, f"modules_cr_valid{args.tag}", out_dir)
    print("[modules] " + " | ".join(f"{k} {100 * v:.2f}" for k, v in res["miou"].items()), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
