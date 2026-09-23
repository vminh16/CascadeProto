"""Base-class calibration of the background [DECISION D-27]. **Beyond the paper.**

COSeg's Base Prototypes Calibration [COSeg §4.3, Eq.9-12] in a form that needs no training, so it can
be probed on trained checkpoints first (P1 of D-27). A query point that resembles a base (training)
class is background in a novel-class episode [COSeg §4.3].

* **Geometry.** Base, support and query features are compared after centring on the mean feature of
  the base training data and L2 normalisation, SimpleShot's CL2N [SimpleShot §3]: `f~ = normalise(f - mu)`.
  Without the centring, post-ReLU features share a positive component that dominates every cosine (the
  first smoke run of P1 turned every point of VIP-Seg into background, D-27).
* **Bank.** For every base class j, `b_j = normalise(mean_o normalise(sum_{i in mask_o} f~_i))` over its
  occurrences o (support and query masks) in training episodes: the frozen-feature limit of COSeg's EMA
  of masked averages [COSeg Eq.9-10].
* **Calibration.** The support's foreground prototypes are built the same way, `u_c` from the episode's
  support masks. A point the model assigns to foreground class c is moved to the background when

      max_j cos(f~_i, b_j) - cos(f~_i, u_c) > delta        (delta >= 0)

  i.e. when, in one common geometry, it is nearer to a base class than to its own foreground class by a
  margin. Points the model assigns to the background are never touched, and the foreground logits are
  never changed; `delta = inf` returns the model's logits.
"""

from typing import Dict, Optional

import torch
import torch.nn.functional as F


def centred_unit(f: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    """CL2N [SimpleShot §3]: normalise(f - mu) for f [..., D], mu [D]."""
    return F.normalize(f - mu, dim=-1)  # [..., D]


def occurrence_prototypes(f: torch.Tensor, masks: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    """Unit masked averages of CL2N features: f [M, P, D], masks [M, P] in {0,1} -> [M, D] (0 rows if empty)."""
    s = torch.einsum("mp,mpd->md", masks.to(f.dtype), centred_unit(f, mu))  # [M, D]
    return F.normalize(s, dim=-1)  # [M, D]; an empty mask stays 0


def support_prototypes(f_s: torch.Tensor, support_y: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    """u_c of the episode's ways, built like the bank: f_s [N, K, P, D], support_y [N, K, P] -> [N, D]."""
    occ = torch.stack([occurrence_prototypes(f_s[n], support_y[n], mu) for n in range(f_s.shape[0])])  # [N,K,D]
    return F.normalize(occ.sum(dim=1), dim=-1)  # [N, D]


class BasePrototypeBank:
    """Two passes over training episodes: the CL2N centre, then per-occurrence base prototypes (COSeg Eq.9-10)."""

    def __init__(self, dim: int):
        self.dim = dim
        self.feature_sum = torch.zeros(dim, dtype=torch.float64)
        self.feature_count = 0
        self.mu: Optional[torch.Tensor] = None
        self.sums: Dict[int, torch.Tensor] = {}
        self.counts: Dict[int, int] = {}
        self.common_cos_sum, self.common_cos_count = 0.0, 0  # mean cos(f_i, mu) of the raw features

    def add_centre(self, f_s: torch.Tensor, f_q: torch.Tensor) -> None:
        """Pass 1: every support and query point, f_s [N, K, P, D], f_q [B_q, P, D]."""
        self.feature_sum += f_s.double().sum(dim=(0, 1, 2)).cpu() + f_q.double().sum(dim=(0, 1)).cpu()  # [D]
        self.feature_count += f_s[..., 0].numel() + f_q[..., 0].numel()

    def freeze_centre(self) -> torch.Tensor:
        if self.feature_count == 0:
            raise RuntimeError("no feature seen in pass 1")
        self.mu = self.feature_sum / self.feature_count  # [D]
        return self.mu

    def add_episode(self, f_s: torch.Tensor, support_y: torch.Tensor, f_q: torch.Tensor, query_y: torch.Tensor,
                    sampled_classes) -> None:
        """Pass 2: f_s [N, K, P, D], support_y [N, K, P] in {0,1}, f_q [B_q, P, D], query_y [B_q, P] in {0..N}."""
        if self.mu is None:
            raise RuntimeError("freeze_centre() before add_episode()")
        mu = self.mu.to(f_q.device, f_q.dtype)  # [D]
        cos = torch.einsum("bpd,d->bp", F.normalize(f_q, dim=-1), F.normalize(mu, dim=0))  # [B_q, P]
        self.common_cos_sum += float(cos.sum())
        self.common_cos_count += cos.numel()
        for way, cls in enumerate(sampled_classes):
            sup = occurrence_prototypes(f_s[way], support_y[way], mu)  # [K, D]
            qry = occurrence_prototypes(f_q, (query_y == way + 1).to(f_q.dtype), mu)  # [B_q, D]
            occ = torch.cat([sup, qry])  # [K + B_q, D]
            keep = occ.norm(dim=-1) > 0  # [K + B_q], occurrences with at least one point
            cls = int(cls)
            self.sums[cls] = self.sums.get(cls, torch.zeros(self.dim, dtype=torch.float64)) + \
                occ[keep].sum(0).double().cpu()  # [D]
            self.counts[cls] = self.counts.get(cls, 0) + int(keep.sum())

    def prototypes(self, min_count: int = 1) -> Dict[int, torch.Tensor]:
        """class -> unit prototype [D]; raises if a class has fewer than `min_count` occurrences."""
        low = {c: n for c, n in self.counts.items() if n < min_count}
        if low:
            raise RuntimeError(f"base classes with fewer than {min_count} occurrences: {low}")
        return {c: F.normalize(s, dim=0) for c, s in sorted(self.sums.items())}


def base_margin(f_q: torch.Tensor, logits: torch.Tensor, support: torch.Tensor, base: torch.Tensor,
                mu: torch.Tensor) -> torch.Tensor:
    """max_j cos(f~_i, b_j) - cos(f~_i, u_{c_i}) for the model's class c_i >= 1; -inf where c_i = 0.

    f_q [B_q, P, D], logits [B_q, P, N+1], support u [N, D], base [J, D], mu [D] -> [B_q, P].
    """
    f = centred_unit(f_q, mu)  # [B_q, P, D]
    to_base = torch.einsum("bpd,jd->bpj", f, base).max(dim=-1).values  # [B_q, P]
    to_fg = torch.einsum("bpd,nd->bpn", f, support)  # [B_q, P, N]
    pred = logits.argmax(dim=-1)  # [B_q, P]
    own = torch.gather(to_fg, -1, (pred - 1).clamp_min(0).unsqueeze(-1)).squeeze(-1)  # [B_q, P]
    return torch.where(pred > 0, to_base - own, torch.full_like(own, float("-inf")))  # [B_q, P]


def calibrate_background(logits: torch.Tensor, margin: torch.Tensor, delta: float) -> torch.Tensor:
    """Move to the background every point whose base margin exceeds delta; logits [B_q, P, N+1] -> same."""
    flip = margin > delta  # [B_q, P]; margin is -inf on background predictions
    out = logits.clone()  # [B_q, P, N+1]
    out[..., 0] = torch.where(flip, logits.max(dim=-1).values + 1.0, logits[..., 0])  # [B_q, P]
    return out


def separability_histogram(g: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor,
                           bins: int = 200, lo: float = -2.0, hi: float = 2.0) -> torch.Tensor:
    """Histograms [2, bins] over [lo, hi] of g for the positive and the negative points (for an AUC)."""
    idx = ((g.clamp(lo, hi) - lo) / (hi - lo) * (bins - 1)).round().long()  # [B_q, P]
    pos = torch.bincount(idx[positive], minlength=bins)  # [bins]
    neg = torch.bincount(idx[negative], minlength=bins)  # [bins]
    return torch.stack([pos, neg])  # [2, bins]


def auc_from_histogram(hist) -> Optional[float]:
    """P(g_pos > g_neg) + 0.5 P(tie) from histograms [2, bins]; None if a side is empty."""
    pos, neg = [torch.as_tensor(h, dtype=torch.float64) for h in hist]
    if pos.sum() == 0 or neg.sum() == 0:
        return None
    below = torch.cumsum(neg, 0) - neg  # [bins], negatives strictly below each bin
    return float(((pos * below).sum() + 0.5 * (pos * neg).sum()) / (pos.sum() * neg.sum()))
