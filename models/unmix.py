"""P11.6 of [DECISION D-48 amendment 5]: the mean rule, its transcription of D-39's "both", and label-free context
unmixing. **Beyond the paper.**

Mean rule (P10.4's isotropic control): class score s_c(x) = <u - mu, m_c - mu> - 1/2 ||m_c - mu||^2, the Euclidean
nearest mean up to a term shared by the classes. Centring by mu changes no decision; it lets every score be written
in the centred space v = u - mu, where unmixing works.

"both" in mean form (D-39's rule, with means instead of unit directions): the background score is the max over
  * the support background mean, adapted per query block by P6's frozen self-support (alpha 0.25, rho 1, 2 steps:
    m <- alpha m + (1 - alpha) mean of the block's points predicted background, when at least 16 are),
  * the 3 spherical k-means clusters of the support background (the mean of each cluster's members).

Unmixing (label-free). Context c(x) = mean of v over the k spatial neighbours of x (all of them, no prediction read).
For foreground row c with centred mean p_c, least squares v_x ~ alpha p_c + gamma c(x) gives
    alpha = (<c,c><v,p> - <p,c><v,c>) / (<p,p><c,c> - <p,c>^2).
With c(x) removed, the plain score <v,p> - 1/2 ||p||^2 = ||p||^2 (alpha_plain - 1/2) becomes ||p||^2 (alpha - 1/2).
The 2 x 2 system is singular when c(x) is parallel to p_c (a dense point inside its own class), so the unmixed score
is used only where cos(p_c, c(x)) < tau and ||c(x)|| is not negligible (the identifiability gate).
"""

from typing import Tuple

import torch
import torch.nn.functional as F

SSP_ALPHA, SSP_STEPS, SSP_MIN = 0.25, 2, 16  # P6's frozen background self-support [DECISION D-38]
KM = 3  # D-39's background components
CTX_EPS = 1e-6


def mean_scores(v: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """<v, p_r> - 1/2 ||p_r||^2 [B, P, R] for centred features v [B, P, D] and centred means p [B, R, D]."""
    return torch.einsum("bpd,brd->bpr", v, p) - 0.5 * (p * p).sum(-1).unsqueeze(1)


def cluster_means(u: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    """Mean [k, D] of the vectors u [M, D] assigned (max cosine) to each unit centroid [k, D]; an empty cluster
    keeps its centroid."""
    assign = (u @ centroids.T).argmax(dim=1)  # [M]
    members = F.one_hot(assign, centroids.shape[0]).to(u.dtype)  # [M, k]
    counts = members.sum(dim=0)  # [k]
    sums = members.T @ u  # [k, D]
    return torch.where((counts > 0).unsqueeze(-1), sums / counts.clamp_min(1.0).unsqueeze(-1), centroids)


def mean_both_logits(u_q: torch.Tensor, f_s: torch.Tensor, support_y: torch.Tensor, mu: torch.Tensor,
                     kmeans) -> torch.Tensor:
    """Mean rule with "both" on the background row, centred space [B_q, P, N+1].

    u_q [B_q, P, D] unit query features, f_s [N, K, P, D] support features, mu [D] the base mean,
    kmeans(x [M, D], k) -> unit centroids [k, D] (P6's spherical k-means).
    """
    u_s = F.normalize(f_s, dim=-1)  # [N, K, P, D]
    fg = (support_y == 1).to(u_s.dtype)  # [N, K, P]
    m_fg = torch.einsum("nkp,nkpd->nd", fg, u_s) / fg.sum(dim=(1, 2)).clamp_min(1.0).unsqueeze(-1)  # [N, D]
    bg_units = u_s[support_y == 0]  # [M, D]
    m_bg = bg_units.mean(dim=0)  # [D]
    comps = cluster_means(bg_units, kmeans(bg_units, KM))  # [3, D]
    v = u_q - mu  # [B_q, P, D]
    b_q = u_q.shape[0]
    fg_p = (m_fg - mu).unsqueeze(0).expand(b_q, -1, -1)  # [B_q, N, D]
    bg_p = (m_bg - mu).unsqueeze(0).repeat(b_q, 1)  # [B_q, D]
    for _ in range(SSP_STEPS):  # self-support of the background row, per block
        rows = torch.cat([bg_p.unsqueeze(1), fg_p], dim=1)  # [B_q, N+1, D]
        pred = mean_scores(v, rows).argmax(dim=-1)  # [B_q, P]
        for b in range(b_q):
            sel = pred[b] == 0  # [P]
            if int(sel.sum()) >= SSP_MIN:
                bg_p[b] = SSP_ALPHA * bg_p[b] + (1.0 - SSP_ALPHA) * v[b][sel].mean(dim=0)
    rows = torch.cat([bg_p.unsqueeze(1), fg_p], dim=1)  # [B_q, N+1, D]
    scores = mean_scores(v, rows)  # [B_q, P, N+1]
    comp = mean_scores(v, (comps - mu).unsqueeze(0).expand(b_q, -1, -1)).max(dim=-1).values  # [B_q, P]
    return torch.cat([torch.maximum(scores[..., :1], comp.unsqueeze(-1)), scores[..., 1:]], dim=-1)


def context_means(v: torch.Tensor, xyz: torch.Tensor, k: int) -> torch.Tensor:
    """c(x) [B, P, D]: the mean of v over the k nearest points of x in space (x itself excluded)."""
    n = xyz.shape[1]
    if not 0 < k < n:
        raise ValueError(f"k = {k} needs 0 < k < {n}")
    d = torch.cdist(xyz, xyz, compute_mode="donot_use_mm_for_euclid_dist")  # [B, P, P]
    d = d.masked_fill(torch.eye(n, dtype=torch.bool, device=xyz.device), float("inf"))
    idx = d.topk(k, dim=-1, largest=False).indices  # [B, P, k]
    gathered = torch.gather(v.unsqueeze(1).expand(-1, n, -1, -1), 2,
                            idx.unsqueeze(-1).expand(-1, -1, -1, v.shape[-1]))  # [B, P, k, D]
    return gathered.mean(dim=2)


def unmix_fg(scores: torch.Tensor, v: torch.Tensor, ctx: torch.Tensor, p: torch.Tensor,
             tau: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """(logits [B, P, N+1], gate [B, P, N]): the foreground columns of `scores` replaced by ||p||^2 (alpha - 1/2)
    where cos(p_c, c(x)) < tau; the background column untouched.

    v [B, P, D] centred query features, ctx [B, P, D] their contexts, p [N, D] centred foreground means.
    """
    vp = torch.einsum("bpd,nd->bpn", v, p)  # [B, P, N]
    vc = (v * ctx).sum(-1, keepdim=True)  # [B, P, 1]
    pc = torch.einsum("bpd,nd->bpn", ctx, p)  # [B, P, N]
    pp = (p * p).sum(-1)  # [N]
    cc = (ctx * ctx).sum(-1, keepdim=True)  # [B, P, 1]
    det = pp * cc - pc * pc  # [B, P, N]
    cos = pc / (pp.sqrt() * cc.sqrt()).clamp_min(1e-12)  # [B, P, N]
    gate = (cos < tau) & (cc > CTX_EPS) & (det > 1e-12)  # [B, P, N]
    alpha = (cc * vp - pc * vc) / det.clamp_min(1e-12)  # [B, P, N]
    fg = torch.where(gate, pp * (alpha - 0.5), scores[..., 1:])  # [B, P, N]
    return torch.cat([scores[..., :1], fg], dim=-1), gate


def mixture_fit(v: torch.Tensor, o: torch.Tensor, ctx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """(R^2 [M], alpha [M]) of the per-point least squares v ~ alpha o + gamma ctx (no intercept), rows [M, D].
    R^2 = 1 - ||residual||^2 / ||v||^2 (the oracle check of the linear mixture)."""
    a = torch.stack([o, ctx], dim=-1)  # [M, D, 2]
    g = a.transpose(1, 2) @ a  # [M, 2, 2]
    rhs = a.transpose(1, 2) @ v.unsqueeze(-1)  # [M, 2, 1]
    coef = torch.linalg.solve(g + 1e-12 * torch.eye(2, dtype=v.dtype, device=v.device), rhs)  # [M, 2, 1]
    res = v - (a @ coef).squeeze(-1)  # [M, D]
    r2 = 1.0 - (res * res).sum(-1) / (v * v).sum(-1).clamp_min(1e-30)
    return r2, coef[:, 0, 0]
