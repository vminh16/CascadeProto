"""Attribution of [DECISION D-48 amendment 3]: add-one, leave-one-out, exact Shapley and pairwise interactions of
switchable inference blocks, from per-episode counts of every on/off combination, with paired bootstrap CIs.

m(S) is VIP-Seg's accumulated mIoU (points) of the combination S. For blocks B and S ⊆ B \\ {i}:
    add-one        m({i}) - m({})
    leave-one-out  m(B) - m(B \\ {i})
    Shapley        phi_i = sum_S |S|! (|B| - |S| - 1)! / |B|! [m(S ∪ {i}) - m(S)],   sum_i phi_i = m(B) - m({})
    interaction    I_ij = m({i, j}) - m({i}) - m({j}) + m({})
Every quantity is linear in the m's, so one set of bootstrap weights over episodes gives all of them jointly.
"""

import itertools
import math
from typing import Dict, FrozenSet, List, Sequence, Tuple

import numpy as np

from experiments import p0_em_probe as p0

Combo = FrozenSet[str]


def combos(blocks: Sequence[str]) -> List[Combo]:
    """Every subset of the blocks, smallest first."""
    return [frozenset(c) for r in range(len(blocks) + 1) for c in itertools.combinations(blocks, r)]


def combo_name(c: Combo, blocks: Sequence[str]) -> str:
    return "+".join(b for b in blocks if b in c) or "base"


def shapley_weights(blocks: Sequence[str], i: str) -> Dict[Combo, float]:
    """{S: weight} over S ⊆ B \\ {i}, weight |S|! (n - |S| - 1)! / n!."""
    rest = [b for b in blocks if b != i]
    n = len(blocks)
    return {frozenset(s): math.factorial(len(s)) * math.factorial(n - len(s) - 1) / math.factorial(n)
            for r in range(len(rest) + 1) for s in itertools.combinations(rest, r)}


def contrasts(blocks: Sequence[str], pairs: Sequence[Tuple[str, str]]) -> Dict[str, Dict[Combo, float]]:
    """Every registered quantity as a linear combination {combination: coefficient} of the m's."""
    full, empty = frozenset(blocks), frozenset()
    out: Dict[str, Dict[Combo, float]] = {}
    for i in blocks:
        out[f"add:{i}"] = {frozenset([i]): 1.0, empty: -1.0}
        out[f"loo:{i}"] = {full: 1.0, full - {i}: -1.0}
        coef: Dict[Combo, float] = {}
        for s, w in shapley_weights(blocks, i).items():
            coef[s | {i}] = coef.get(s | {i}, 0.0) + w
            coef[s] = coef.get(s, 0.0) - w
        out[f"shapley:{i}"] = coef
    for a, b in pairs:
        out[f"inter:{a}x{b}"] = {frozenset([a, b]): 1.0, frozenset([a]): -1.0, frozenset([b]): -1.0, empty: 1.0}
    out["total"] = {full: 1.0, empty: -1.0}
    return out


def evaluate(counts: Dict[Combo, np.ndarray], lin: Dict[Combo, float], weights: np.ndarray = None) -> np.ndarray:
    """sum_S coef_S * 100 m(S) [B] (or a scalar array with weights None); counts[S] [E, 3, C+1]."""
    total = 0.0
    for s, c in lin.items():
        if weights is None:
            m = p0.miou_from_counts(counts[s].sum(0))
        else:
            m = p0.miou_from_counts(np.einsum("be,exy->bxy", weights, counts[s]))  # [B]
        total = total + 100.0 * c * m
    return np.asarray(total)


def attribute(counts: Dict[Combo, np.ndarray], blocks: Sequence[str], pairs: Sequence[Tuple[str, str]],
              boot: int = p0.BOOTSTRAP, seed: int = 0) -> Dict[str, Dict[str, float]]:
    """{quantity: {value, ci_low, ci_high}} with one set of episode bootstrap weights for all quantities."""
    e = next(iter(counts.values())).shape[0]
    rng = np.random.default_rng(seed)
    w = rng.multinomial(e, np.full(e, 1.0 / e), size=boot).astype(np.float64)  # [B, E]
    out = {}
    for name, lin in contrasts(blocks, pairs).items():
        dist = evaluate(counts, lin, w)
        out[name] = {"value": float(evaluate(counts, lin)), "ci_low": float(np.percentile(dist, 2.5)),
                     "ci_high": float(np.percentile(dist, 97.5))}
    return out


def point_values(counts: Dict[Combo, np.ndarray], blocks: Sequence[str],
                 pairs: Sequence[Tuple[str, str]]) -> Dict[str, float]:
    """The registered quantities without CIs (the random600 draws)."""
    return {name: float(evaluate(counts, lin)) for name, lin in contrasts(blocks, pairs).items()}


def kept(fixed: Dict[str, Dict[str, float]], randoms: List[Dict[str, float]], i: str, bar: float) -> bool:
    """D48.1': phi_i >= bar with its fixed100 CI above 0 and phi_i > 0 on every random600 draw."""
    q = fixed[f"shapley:{i}"]
    return q["value"] >= bar and q["ci_low"] > 0 and all(r[f"shapley:{i}"] > 0 for r in randoms)
