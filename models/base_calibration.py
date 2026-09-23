"""Base-class calibration of the background [DECISION D-27]. **Beyond the paper.**

COSeg's Base Prototypes Calibration [COSeg §4.3, Eq.9-12] in a form that needs no training, so it can
be probed on trained checkpoints first (P1 of D-27):

* **Bank.** For every base (training) class j, the prototype `b_j` is the mean over its occurrences in
  training episodes of the per-occurrence masked average of unit features, normalised:
  `b_j = normalise(mean_o normalise(sum_{i in mask_o} f_i/||f_i||))`. COSeg keeps an EMA of per-episode
  masked averages over support and query (Eq.9-10, momentum 0.995, insensitive between 0.99 and 0.999,
  COSeg T6); with frozen features that EMA converges to this mean. Only base-class labels are read.
* **Calibration.** A query point that looks like a base class belongs to the background of a novel-class
  episode [COSeg §4.3]. The base prototypes join the background as extra prototypes of the model's own
  scoring rule, at the mean norm `s` of its foreground prototypes:

      L'_i0 = max(L_i0, omega * max_j s <f_i, b_j>)       [B_q, P]

  COSeg instead adds the base guidance to the background correlation through a trained layer (Eq.12);
  the max over prototypes is the multi-prototype background of AttMPTI, which needs no parameter.
  `omega = 0` returns the model's logits unchanged (for a positive rule the max can only raise L_i0).
"""

from typing import Dict, Optional

import torch
import torch.nn.functional as F


def occurrence_prototypes(f: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """Unit masked averages of unit features: f [M, P, D], masks [M, P] in {0,1} -> [M, D] (0 rows if empty)."""
    s = torch.einsum("mp,mpd->md", masks.to(f.dtype), F.normalize(f, dim=-1))  # [M, D]
    return F.normalize(s, dim=-1)  # [M, D]; an empty mask stays 0


class BasePrototypeBank:
    """Accumulates per-occurrence prototypes of the base classes of training episodes (COSeg Eq.9-10)."""

    def __init__(self, dim: int):
        self.dim = dim
        self.sums: Dict[int, torch.Tensor] = {}
        self.counts: Dict[int, int] = {}

    def add_episode(self, f_s: torch.Tensor, support_y: torch.Tensor, f_q: torch.Tensor, query_y: torch.Tensor,
                    sampled_classes) -> None:
        """f_s [N, K, P, D], support_y [N, K, P] in {0,1}, f_q [B_q, P, D], query_y [B_q, P] in {0..N}."""
        for way, cls in enumerate(sampled_classes):
            sup = occurrence_prototypes(f_s[way], support_y[way])  # [K, D]
            qry = occurrence_prototypes(f_q, (query_y == way + 1).to(f_q.dtype))  # [B_q, D]
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


def base_similarity(f_q: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    """g_i = max_j cos(f_i, b_j): f_q [B_q, P, D], base [J, D] (unit) -> [B_q, P]."""
    return torch.einsum("bpd,jd->bpj", F.normalize(f_q, dim=-1), base).max(dim=-1).values  # [B_q, P]


def calibrate_background(f_q: torch.Tensor, prior: torch.Tensor, logits: torch.Tensor, base: torch.Tensor,
                         omega: float) -> torch.Tensor:
    """L'_i0 = max(L_i0, omega * max_j s <f_i, b_j>), s = mean foreground prototype norm; -> [B_q, P, N+1].

    f_q [B_q, P, D] and prior [B_q, N+1, D] are the model's scoring rule, logits = F^q prior^T.
    """
    if omega == 0:
        return logits
    s = prior[:, 1:].norm(dim=-1).mean(dim=1)  # [B_q], mean foreground prototype norm
    base_logit = torch.einsum("bpd,jd->bpj", f_q, base).max(dim=-1).values * s[:, None]  # [B_q, P]
    out = logits.clone()  # [B_q, P, N+1]
    out[..., 0] = torch.maximum(logits[..., 0], omega * base_logit)  # [B_q, P]
    return out


def separability_histogram(g: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor,
                           bins: int = 200) -> torch.Tensor:
    """Histograms [2, bins] over [-1, 1] of g for the positive and the negative points (for an AUC)."""
    idx = ((g.clamp(-1, 1) + 1) / 2 * (bins - 1)).round().long()  # [B_q, P]
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
