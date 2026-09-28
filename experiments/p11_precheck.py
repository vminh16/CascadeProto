"""P11 of [DECISION D-48] (amendments 1-3): pre-checks on frozen CR features, and the 2^4 inference factorial.

    python experiments/p11_precheck.py fit --data_path datasets/S3DIS/blocks_bs1_s1 --checkpoint cr:ours:1:<last.pt>
    python experiments/p11_precheck.py select --data_path ... --checkpoint cr:...     # block parameters on valid
    python experiments/p11_precheck.py factorial --data_path ... --checkpoint cr:... [--draw d ...]
    python experiments/p11_precheck.py decide                                         # P11 gates, attribution

fit (training episodes of the fold's base classes, raw labels through P5's checked sampler copy):
  * the base-class moments (mean, covariance) of the unit features, for the descriptor spaces of amendment 2;
  * block [2]'s base learner, trained on frozen features (P11.1b);
  * P11.4's descriptor probe per space, and its held-out base-class score against U (the A2 reading).
select (valid): the parameters of [2] (psi), [6] (kappa) and OT (eps, rho) by coordinate ascent in the full
  combination without LP (LP is P7's frozen arm on CR); P11.2's rank correlation of gamma_e with kappa*_e.
factorial (valid, fixed100, random600 seeds 0-2, leak-free, and valid_raw with raw labels): the 16 combinations of
  {excl, text, ot, lp} on U + both, with P11.3's max-over-cells rules and P11.4's probe per space, the own / other
  split, and on valid_raw the mechanism readings (base false positives, g on dense novel foreground, OT mass on true
  foreground, background-cell contamination).
decide: P11.1b, P11.2, P11.4, P11.5 and the CR attribution (add-one, leave-one-out, Shapley, interactions, D48.1').

Order of the blocks inside a combination: U + both -> [2] exclusion -> [6] text -> OT -> LP. Query labels are read
only by counts and mechanism readings; no block reads them.
"""

import argparse
import itertools
import json
import os
import sys
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from experiments import attribution as att  # noqa: E402
from experiments import d39_eval as d39  # noqa: E402
from experiments import p0_em_probe as p0  # noqa: E402
from experiments import p1_bpc_probe as p1  # noqa: E402
from experiments import p3_probe as p3  # noqa: E402
from experiments import p5_condition_probe as p5  # noqa: E402
from experiments import p6_prototype_probe as p6  # noqa: E402
from experiments import p7_propagation_probe as p7  # noqa: E402
from models import base_learner as bl  # noqa: E402
from models import correlation as corr  # noqa: E402
from models import ot_assign as ota  # noqa: E402
from models import text_prior as tp  # noqa: E402

OUT_DIR = "results/phase16_d48"
P7_FIXED = "results/phase16_p7/test_S1_fixed100.json"  # P7's frozen LP arm on CR
TRAIN_SEED, TRAIN_PER_PAIR = 12, 70  # fresh seeds; 15 base pairs x 70 = 1,050 training episodes
RAW_SEED = 11  # valid_raw: a seeded test draw with raw labels (P5 used 3 and 4)
PTS_PER_BLOCK = 256  # points subsampled per block for the base learner and the probe
BL_EPOCHS, PROBE_EPOCHS, HOLDOUT = 10, 10, 0.2
TEXT_SOURCE, TEXT_PROMPT = "ridge", "descriptions"  # P3's frozen source and prompt on CR (D-46)
# grids in cosine units (the logits of U + both are cosines); conventions, selected on valid
PSIS = (0.01, 0.03, 0.1, 0.3, 1.0)
KAPPAS = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
OT_GRID = ((0.05, 0.1), (0.05, 1.0), (0.1, 0.1), (0.1, 1.0))  # (eps, rho) [DECISION D-48 amendment 2]
ASCENT_ROUNDS = 2
BLOCKS = ("excl", "text", "ot", "lp")
PAIRS = (("excl", "lp"), ("excl", "text"), ("ot", "lp"))  # the CR-side registered interactions (amendment 3)
BG_CAPS = (8, 16, 32)  # P11.3
PROBE_CAP = 16
# rules [DECISION D-48 amendments 1-3]
EXCL_GAIN, DENSE_G, DENSE_SHARE, DENSE_SHARE_REPORT = 1.0, 0.5, 0.10, 0.05  # P11.1b
TEXT_GAIN, OT_GAIN, PROBE_MARGIN, KEEP_BAR = 0.5, 0.5, 0.0, 0.5  # P11.2, P11.5, P11.4, D48.1'
FACTORIAL_DRAWS = ("valid",) + p6.TEST_DRAWS + ("valid_raw",)
DESC_DRAWS = ("valid", "fixed100", "leakfree", "valid_raw")  # P11.3 / P11.4 are scored on these
COND_COMBOS = ("base", "excl", "text", "ot", "lp", "excl+text+ot+lp")


# ------------------------------------------------------------------ episodes with raw labels

def sample_support(data_path: str, scan: str, cls: int, seed) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(ptcloud [P, 9], mask [P], raw labels [P]) of one support block: the inherited sampler under `seed`, and P5's
    index copy under the same seed for the raw labels, checked equal (one sampler call per draw, fix F18)."""
    from dataloaders.loader import sample_pointcloud
    from pipeline.episodes import NUM_POINT, PC_ATTRIBS

    data = np.load(os.path.join(data_path, "data", f"{scan}.npy"))
    np.random.seed(seed)
    idx = p5.sampled_indices(data, NUM_POINT, cls, False)  # [P]
    np.random.seed(seed)
    pc, mask = sample_pointcloud(data_path, NUM_POINT, PC_ATTRIBS, False, None, scan, [cls], cls, support=True,
                                 random_sample=False)
    raw = data[idx, 6].astype(np.int64)  # [P]
    xyz = data[idx, :3] - data[idx, :3].min(axis=0)  # [P, 3]
    if not (np.allclose(pc[:, :3], xyz) and np.array_equal(mask.astype(bool), raw == cls)):
        raise RuntimeError(f"{scan}: the index copy of the sampler differs from the inherited one")
    return pc, mask.astype(np.int32), raw


def raw_episodes(data_path: str, mode: str, seed: int, per_pair: int,
                 max_episodes: Optional[int] = None) -> Iterator[Dict]:
    """Seeded 2-way 1-shot episodes of the fold's `mode` classes ("train" = base, "test" = novel) with the raw
    labels of every query and support point. Scans are chosen as `generate_one_episode` does [VIPSEG
    dataloaders/loader.py:174-218], as in P5's `draw_episodes`, with no augmentation."""
    from dataloaders.loader import MyDataset
    from pipeline.episodes import N_QUERIES, NUM_POINT, PC_ATTRIBS, WAY_NUM, WAY_RATIO

    ds = MyDataset(data_path, "s3dis", cvfold=1, num_episode=1, n_way=2, k_shot=1, n_queries=N_QUERIES, mode=mode,
                   num_point=NUM_POINT, pc_attribs=PC_ATTRIBS, pc_augm=False, way_ratio=WAY_RATIO, way_num=WAY_NUM)
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
            sx, sy, sraw, qx, qy, qraw = [], [], [], [], [], []
            for w, (c, (q, s)) in enumerate(zip(classes, scans)):
                x, y, raw = p5.sample_block(data_path, q, classes, int(c), False, [seed, pi, e, 0, w])
                qx.append(x), qy.append(y), qraw.append(raw)
                x, m, raw = sample_support(data_path, s, int(c), [seed, pi, e, 1, w])
                sx.append(x[None]), sy.append(m[None]), sraw.append(raw[None])
            item = (np.stack(sx), np.stack(sy), np.stack(qx), np.stack(qy), classes)  # [N,1,P,9] [N,1,P] [N,P,9] [N,P]
            yield {"item": item, "query_raw": np.stack(qraw), "support_raw": np.stack(sraw), "index": count}
            count += 1


def draw_items(draw: str, data_path: str, max_episodes: Optional[int]) -> Tuple[Iterator[Dict], List[int]]:
    if draw == "valid_raw":
        return raw_episodes(data_path, "test", RAW_SEED, p5.EPISODES_PER_PAIR, max_episodes), \
            p5.test_classes_of(1, data_path)
    items, classes = p6.episodes(draw, data_path, max_episodes)
    return ({"item": it} for it in items), classes


# ------------------------------------------------------------------ per-episode quantities (pure)

def unit_both_logits(f_q: torch.Tensor, f_s: torch.Tensor, support_y: torch.Tensor) -> torch.Tensor:
    """D-39's U + both in cosine units [B_q, P, N+1] (unit query features; same argmax as on raw features, since
    every term is linear in f_q and P6's frozen self-support keeps every point, rho = 1)."""
    return p7.both_logits(F.normalize(f_q, dim=-1), f_s, support_y)


def combine(l0: torch.Tensor, combo, g: torch.Tensor, t: torch.Tensor, gamma: torch.Tensor,
            masses: torch.Tensor, params: Dict[str, float]) -> torch.Tensor:
    """Logits [B_q, P, N+1] of a combination of {excl, text, ot} on U + both (LP acts on the argmax afterwards)."""
    x = l0
    if "excl" in combo:
        x = bl.apply_exclusion(x, g, params["psi"])
    if "text" in combo:
        x = tp.apply_prior(x, t, params["kappa"], gamma, torch.ones_like(x[..., 0]))
    if "ot" in combo:
        x = ota.ot_logits(x, masses, params["eps"], params["rho"])
    return x


def spread_many(xyz: torch.Tensor, u: torch.Tensor, seeds: List[torch.Tensor], n_cls: int,
                arm: Dict) -> List[torch.Tensor]:
    """P7's label spreading of several seed predictions [B_q, P] on one graph, one Cholesky factor for all."""
    idx, a = p7.knn_graph(xyz.double(), u.double(), arm["graph"], arm["k"])  # [B_q, P, k]
    s = p7.normalized(p7.affinity(idx, a, xyz.shape[1]))  # [B_q, P, P]
    y0 = torch.cat([F.one_hot(sd, n_cls).double() for sd in seeds], dim=-1)  # [B_q, P, n_seeds * C]
    z = p7.spread(s, y0, arm["beta"])  # [B_q, P, n_seeds * C]
    return [z[..., i * n_cls:(i + 1) * n_cls].argmax(dim=-1) for i in range(len(seeds))]


def combo_label(combo) -> str:
    return att.combo_name(frozenset(combo), BLOCKS)


# ------------------------------------------------------------------ the checkpoint's context

class Context:
    """Frozen pieces of one checkpoint: the rule, the base learner, the text directions, the spaces and probes."""

    def __init__(self, rule, ck, data_path: str, device, fit: Optional[Dict], bank_n: int = p1.BANK_EPISODES):
        from pipeline.episodes import read_class_names

        self.rule, self.ck, self.device = rule, ck, device
        self.names = read_class_names(data_path, "s3dis")
        self.fit = fit
        if fit is not None:
            self.base_classes = fit["base_classes"]
            self.learner = bl.BaseLearner(fit["dim"], len(self.base_classes)).to(device)
            self.learner.load_state_dict(fit["learner"])
            self.learner.eval()
            self.spaces = corr.spaces_from(fit["mu"].to(device), fit["cov"].to(device))
            self.probes = {}
            for k, sd in fit["probes"].items():
                pr = corr.DescriptorProbe().to(device)
                pr.load_state_dict(sd)
                self.probes[k] = pr.eval()
        mu, base, info = p1.load_or_build_bank(ck, p3.BankRule(rule), data_path, device, bank_n)  # P3's text bank
        self.mu, self.bank, self.bank_idx = mu.to(device), base.to(device), info["classes"]
        emb = tp.class_embeddings([self.names[c] for c in range(p3.N_CLASSES)], TEXT_PROMPT, p3.clip_encoder(device))
        self.emb = emb.to(device)

    def features(self, e) -> Tuple[torch.Tensor, torch.Tensor]:
        f_q, m_eff, _, logits = self.rule(e)
        p0.check_identity(f_q, m_eff, logits)  # the rule reads the tensors the model scores with
        return f_q, p5.support_features(self.rule, e)  # [B_q, P, D], [N, K, P, D]

    def text(self, e, f_q: torch.Tensor, f_s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        novel = [int(c) for c in e.sampled_classes]
        t_hat = p3.directions(TEXT_SOURCE, self.emb, novel, self.bank_idx, self.bank)  # [N, D]
        gamma, _ = tp.support_gamma(f_s, e.support_y, self.mu, t_hat)
        return tp.text_logits(f_q, self.mu, t_hat), gamma  # [B_q, P, N], scalar

    def g(self, f_q: torch.Tensor) -> torch.Tensor:
        return bl.base_score(self.learner(f_q))  # [B_q, P], nothing excluded at test


# ------------------------------------------------------------------ fit

@torch.no_grad()
def collect_training(ctx: Context, data_path: str, max_episodes: Optional[int]) -> Dict:
    """Subsampled unit features, raw and episode labels, and support tensors of the training episodes."""
    from pipeline.episodes import make_episode

    gen = torch.Generator(device="cpu").manual_seed(TRAIN_SEED)
    feats, raws, eps = [], [], []
    for ep in raw_episodes(data_path, "train", TRAIN_SEED, TRAIN_PER_PAIR, max_episodes):
        e = make_episode(ep["item"], ctx.names).to(ctx.device)
        f_q, f_s = ctx.features(e)
        u_q, u_s = F.normalize(f_q, dim=-1), F.normalize(f_s, dim=-1)
        picks = [torch.randperm(u_q.shape[1], generator=gen)[:PTS_PER_BLOCK] for _ in range(u_q.shape[0])]
        q_idx = torch.stack(picks)  # [B_q, PTS]
        for b in range(u_q.shape[0]):
            feats.append(u_q[b, q_idx[b].to(u_q.device)].cpu()), raws.append(torch.from_numpy(ep["query_raw"][b])[q_idx[b]])
        for w in range(u_s.shape[0]):
            pick = torch.randperm(u_s.shape[2], generator=gen)[:PTS_PER_BLOCK]
            feats.append(u_s[w, 0, pick.to(u_s.device)].cpu()), raws.append(torch.from_numpy(ep["support_raw"][w, 0])[pick])
        # the query subsample and the full support (cells need every masked point), float16 to bound memory
        u_q_sub = torch.stack([u_q[b, q_idx[b].to(u_q.device)] for b in range(u_q.shape[0])])  # [B_q, PTS, D]
        y_sub = torch.stack([e.query_y[b, q_idx[b].to(u_q.device)] for b in range(u_q.shape[0])])  # [B_q, PTS]
        eps.append({"u_q": u_q_sub.half().cpu(), "u_s": u_s.half().cpu(), "support_y": e.support_y.cpu(),
                    "query_y": y_sub.cpu(), "classes": [int(c) for c in e.sampled_classes]})
    return {"x": torch.cat(feats), "raw": torch.cat(raws), "episodes": eps}


def probe_data(space: corr.Space, eps: List[Dict], device) -> Tuple[torch.Tensor, torch.Tensor]:
    ds, ys = [], []
    for ep in eps:
        z_q, z_s = space(ep["u_q"].to(device).float()), space(ep["u_s"].to(device).float())
        rows = corr.row_cells(z_s, ep["support_y"].to(device), PROBE_CAP, PROBE_CAP)
        desc = corr.row_descriptors(z_q, rows)  # [B_q, PTS, N+1, 3], the stored query subsample
        for b in range(desc.shape[0]):
            ds.append(desc[b]), ys.append(ep["query_y"][b].to(device))
    return torch.cat(ds), torch.cat(ys)


@torch.no_grad()
def heldout_base_scores(space: corr.Space, probe, eps: List[Dict], base_classes: List[int], device) -> Dict[str, float]:
    """Probe and U on held-out training episodes, base classes scored (the A2 reading of P11.4)."""
    counts = {"probe": [], "U": []}
    for ep in eps:
        u_q, u_s, sy = ep["u_q"].to(device).float(), ep["u_s"].to(device).float(), ep["support_y"].to(device)
        rows = corr.row_cells(space(u_s), sy, PROBE_CAP, PROBE_CAP)
        pred_p = probe(corr.row_descriptors(space(u_q), rows)).argmax(-1)
        pred_u = p6.rule_logits(u_q, p6.base_rows(u_q, u_s, sy)).argmax(-1)
        gt = ep["query_y"].numpy()
        counts["probe"].append(p0.episode_counts(pred_p.cpu().numpy(), gt, ep["classes"], base_classes))
        counts["U"].append(p0.episode_counts(pred_u.cpu().numpy(), gt, ep["classes"], base_classes))
    return {k: 100.0 * float(p0.miou_from_counts(np.stack(v).sum(0))) for k, v in counts.items()}


def run_fit(ctx: Context, data_path: str, max_episodes: Optional[int]) -> Tuple[Dict, Dict]:
    from dataloaders.s3dis import S3DISDataset

    base_classes = sorted(int(c) for c in S3DISDataset(1, data_path).train_classes)
    data = collect_training(ctx, data_path, max_episodes)
    x, raw = data["x"].to(ctx.device), data["raw"].to(ctx.device)
    is_base = torch.isin(raw, torch.tensor(base_classes, device=raw.device))
    mu, cov = corr.base_moments(x[is_base])  # base-class points only
    y = bl.base_targets(raw, base_classes)
    learner = bl.train_base_learner(x, y, len(base_classes), BL_EPOCHS)
    with torch.no_grad():
        acc = (learner(x).argmax(-1) == y).float()
    info = {"base_classes": base_classes, "points": int(x.shape[0]), "episodes": len(data["episodes"]),
            "base_share": float(is_base.float().mean()),
            "learner_train_acc": float(acc.mean()),
            "learner_train_acc_by_target": {str(j): float(acc[y == j].mean()) for j in range(len(base_classes) + 1)
                                            if bool((y == j).any())}}
    spaces = corr.spaces_from(mu.float().to(ctx.device), cov.float().to(ctx.device))
    info["spaces"] = {k: s.info for k, s in spaces.items()}
    n_hold = max(1, int(round(HOLDOUT * len(data["episodes"]))))
    train_eps, hold_eps = data["episodes"][:-n_hold], data["episodes"][-n_hold:]
    probes, info["probe_heldout_base"] = {}, {}
    for k, space in spaces.items():
        with torch.no_grad():
            d, yy = probe_data(space, train_eps, ctx.device)
        probes[k] = corr.train_probe(d, yy, PROBE_EPOCHS)
        info["probe_heldout_base"][k] = heldout_base_scores(space, probes[k], hold_eps, base_classes, ctx.device)
    fit = {"base_classes": base_classes, "dim": int(x.shape[-1]), "learner": learner.state_dict(),
           "mu": mu.float().cpu(), "cov": cov.float().cpu(), "probes": {k: p.state_dict() for k, p in probes.items()}}
    return fit, info


# ------------------------------------------------------------------ select (valid)

@torch.no_grad()
def cache_valid(ctx: Context, data_path: str, max_episodes: Optional[int]) -> Tuple[List[Dict], List[int]]:
    from pipeline.episodes import make_episode

    items, test_classes = draw_items("valid", data_path, max_episodes)
    cache = []
    for it in items:
        e = make_episode(it["item"], ctx.names).to(ctx.device)
        f_q, f_s = ctx.features(e)
        t, gamma = ctx.text(e, f_q, f_s)
        cache.append({"l0": unit_both_logits(f_q, f_s, e.support_y), "g": ctx.g(f_q), "t": t, "gamma": gamma,
                      "masses": ota.support_masses(e.support_y), "gt": e.query_y.cpu().numpy(),
                      "classes": list(e.sampled_classes)})
    return cache, test_classes


def cache_miou(cache: List[Dict], combo, params: Dict[str, float], test_classes: List[int]) -> float:
    c = [p0.episode_counts(combine(x["l0"], combo, x["g"], x["t"], x["gamma"], x["masses"], params)
                           .argmax(-1).cpu().numpy(), x["gt"], x["classes"], test_classes) for x in cache]
    return 100.0 * float(p0.miou_from_counts(np.stack(c).sum(0)))


def coordinate_ascent(score, grids: Dict[str, List], start: Dict, rounds: int = ASCENT_ROUNDS) -> Tuple[Dict, List]:
    """Maximise score(params) one coordinate at a time over its grid; ties keep the earlier grid value."""
    params, trace = dict(start), []
    for _ in range(rounds):
        for key, grid in grids.items():
            best, best_val = params[key], None
            for v in grid:
                val = score(dict(params, **{key: v}))
                trace.append({key: v, "score": val})
                if best_val is None or val > best_val:
                    best, best_val = v, val
            params[key] = best
    return params, trace


def kappa_star(x: Dict, test_kappas=(0.0,) + KAPPAS) -> float:
    """The kappa with the highest point accuracy of U + both + text on one episode (reads labels: a bound)."""
    gt = torch.from_numpy(x["gt"]).to(x["l0"].device)
    best, best_acc = 0.0, -1
    for k in test_kappas:
        pred = tp.apply_prior(x["l0"], x["t"], k, x["gamma"], torch.ones_like(x["l0"][..., 0])).argmax(-1)
        acc = int((pred == gt).sum())
        if acc > best_acc:
            best, best_acc = k, acc
    return best


def run_select(cache: List[Dict], test_classes: List[int]) -> Dict:
    full = ("excl", "text", "ot")
    start = {"psi": PSIS[2], "kappa": KAPPAS[2], "ot": OT_GRID[0]}

    def score(p):
        return cache_miou(cache, full, {"psi": p["psi"], "kappa": p["kappa"], "eps": p["ot"][0], "rho": p["ot"][1]},
                          test_classes)

    params, trace = coordinate_ascent(score, {"psi": list(PSIS), "kappa": list(KAPPAS), "ot": list(OT_GRID)}, start)
    gammas = [float(x["gamma"]) for x in cache]
    stars = [kappa_star(x) for x in cache]
    return {"psi": params["psi"], "kappa": params["kappa"], "eps": params["ot"][0], "rho": params["ot"][1],
            "trace": trace, "spearman_gamma_kappa_star": tp.spearman(gammas, stars),
            "episodes": len(cache)}


# ------------------------------------------------------------------ factorial

class RawReadings:
    """valid_raw mechanism sums: base-origin false positives, g by point kind, OT mass on true foreground,
    background-cell contamination."""

    KINDS = ("fg_own", "fg_other", "bg_base", "bg_clutter", "bg_novel")

    def __init__(self, base_classes: List[int], test_classes: List[int]):
        self.base, self.test = base_classes, test_classes
        self.fp_base = {}  # combo -> [false positives on base points, all false positives, TP]
        self.g = {k: [0.0, 0, 0] for k in self.KINDS}  # sum g, count, count with g > DENSE_G
        self.ot_mass = [0.0, 0.0]  # foreground-row mass on true foreground, all foreground-row mass
        self.cells = [0, 0]  # background cells dominated by an episode class, all background cells

    def kinds(self, gt: np.ndarray, raw: np.ndarray) -> np.ndarray:
        """[B_q, P] index into KINDS (-1 never occurs)."""
        k = np.full(gt.shape, -1, dtype=np.int64)
        for b in range(gt.shape[0]):
            own = gt[b] == b + 1
            k[b][own] = 0
            k[b][(gt[b] > 0) & ~own] = 1
            bg = gt[b] == 0
            k[b][bg & np.isin(raw[b], self.base)] = 2
            k[b][bg & (raw[b] == p5.CLUTTER)] = 3
            k[b][bg & np.isin(raw[b], self.test)] = 4
        if (k < 0).any():
            raise ValueError("a query point falls in no kind")
        return k

    def add(self, combo: str, pred: np.ndarray, gt: np.ndarray, raw: np.ndarray) -> None:
        fp = (pred > 0) & (gt != pred) & (gt == 0)
        acc = self.fp_base.setdefault(combo, [0, 0, 0])
        acc[0] += int((fp & np.isin(raw, self.base)).sum())
        acc[1] += int(fp.sum())
        acc[2] += int(((pred == gt) & (gt > 0)).sum())

    def add_g(self, g: np.ndarray, kinds: np.ndarray) -> None:
        for i, k in enumerate(self.KINDS):
            sel = kinds == i
            self.g[k][0] += float(g[sel].sum())
            self.g[k][1] += int(sel.sum())
            self.g[k][2] += int((g[sel] > DENSE_G).sum())

    def add_ot(self, log_t: torch.Tensor, gt: np.ndarray) -> None:
        mass = log_t[..., 1:].exp().sum(-1).cpu().numpy()  # [B_q, P]
        self.ot_mass[0] += float(mass[gt > 0].sum())
        self.ot_mass[1] += float(mass.sum())

    def add_cells(self, bg_proto: Tuple[torch.Tensor, torch.Tensor], z_bg: torch.Tensor, raw_bg: np.ndarray,
                  classes) -> None:
        proto, sizes = bg_proto
        assign = (z_bg @ proto.T).argmax(dim=1).cpu().numpy()  # [n]
        episode = np.isin(raw_bg, [int(c) for c in classes])
        for m in range(proto.shape[0]):
            members = assign == m
            if members.sum() == 0:
                continue
            self.cells[1] += 1
            self.cells[0] += int(episode[members].mean() > 0.5)

    def summary(self) -> Dict:
        return {"fp_base": {k: {"fp_base": v[0], "fp_all": v[1], "tp_fg": v[2]} for k, v in self.fp_base.items()},
                "g": {k: {"mean": v[0] / max(v[1], 1), "points": v[1], "share_above": v[2] / max(v[1], 1)}
                      for k, v in self.g.items()},
                "ot_mass_on_true_fg": self.ot_mass[0] / max(self.ot_mass[1], 1e-30),
                "bg_cells_contaminated": self.cells[0] / max(self.cells[1], 1)}


@torch.no_grad()
def run_factorial(ctx: Context, draw: str, data_path: str, params: Dict, arm: Dict,
                  max_episodes: Optional[int]) -> Tuple[Dict, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    from pipeline.episodes import make_episode

    items, test_classes = draw_items(draw, data_path, max_episodes)
    combos = att.combos(BLOCKS)
    counts: Dict[str, List[np.ndarray]] = {}
    cond: Dict[str, List[np.ndarray]] = {}
    raw_read = RawReadings(ctx.base_classes, test_classes) if draw == "valid_raw" else None
    contrasts: List[torch.Tensor] = []  # valid: oracle contrasts o_c - o_bg, for the top-r span share (F20)
    skipped, checked = 0, 0
    for it in items:
        item = it["item"]
        if (item[1].reshape(item[1].shape[0], -1).sum(1) == 0).any():
            skipped += 1  # leak-free draw only: a uniformly sampled support block without its class
            continue
        e = make_episode(item, ctx.names).to(ctx.device)
        gt = e.query_y.cpu().numpy()
        f_q, f_s = ctx.features(e)
        l0 = unit_both_logits(f_q, f_s, e.support_y)
        if checked < 20:  # U + both in cosine units keeps D-39's decisions
            if not torch.equal(l0.argmax(-1), p7.both_logits(f_q, f_s, e.support_y).argmax(-1)):
                raise RuntimeError("U + both on unit features differs from D-39's rule")
            checked += 1
        g = ctx.g(f_q)
        t, gamma = ctx.text(e, f_q, f_s)
        masses = ota.support_masses(e.support_y)
        preds = {}
        u_pred = p6.rule_logits(f_q, p6.base_rows(f_q, f_s, e.support_y)).argmax(-1).cpu().numpy()  # U alone
        counts.setdefault("U", []).append(p0.episode_counts(u_pred, gt, e.sampled_classes, test_classes))
        for c in combos:
            if "lp" in c:
                continue
            preds[combo_label(c)] = combine(l0, c, g, t, gamma, masses, params).argmax(-1)
        lp_keys = [combo_label(c - {"lp"}) for c in combos if "lp" in c]
        spread = spread_many(e.query_x[..., :3], F.normalize(f_q, dim=-1), [preds[k] for k in lp_keys],
                             l0.shape[-1], arm)
        for k, pr in zip(lp_keys, spread):
            preds["lp" if k == "base" else f"{k}+lp"] = pr
        for c in combos:
            name = combo_label(c)
            pr = preds[name].cpu().numpy()
            counts.setdefault(name, []).append(p0.episode_counts(pr, gt, e.sampled_classes, test_classes))
            if name in COND_COMBOS:
                cond.setdefault(name, []).append(p5.condition_counts(pr, gt, e.sampled_classes, test_classes))
        if draw in DESC_DRAWS:
            u_q, u_s = F.normalize(f_q, dim=-1), F.normalize(f_s, dim=-1)
            for k, space in ctx.spaces.items():
                z_q, z_s = space(u_q), space(u_s)
                for cap in BG_CAPS:
                    rows = corr.row_cells(z_s, e.support_y, PROBE_CAP, cap)
                    desc = corr.row_descriptors(z_q, rows)  # [B_q, P, N+1, 3]
                    pr = corr.max_rule_logits(desc).argmax(-1).cpu().numpy()
                    counts.setdefault(f"p3:{k}:bg{cap}", []).append(
                        p0.episode_counts(pr, gt, e.sampled_classes, test_classes))
                    if cap == PROBE_CAP:
                        pr = ctx.probes[k](desc).argmax(-1).cpu().numpy()
                        counts.setdefault(f"p4:{k}", []).append(
                            p0.episode_counts(pr, gt, e.sampled_classes, test_classes))
                        cond.setdefault(f"p4:{k}", []).append(
                            p5.condition_counts(pr, gt, e.sampled_classes, test_classes))
                        if raw_read is not None and k == "raw":
                            raw_read.add_cells(rows[0], z_s[e.support_y == 0],
                                               it["support_raw"][:, :, :][e.support_y.cpu().numpy() == 0],
                                               e.sampled_classes)
        if draw == "valid":
            from models.oracle_distill import oracle_directions

            o, present = oracle_directions(f_q, e.query_y, l0.shape[-1])  # [B_q, N+1, D], [B_q, N+1]
            for b in range(o.shape[0]):
                for c in range(1, o.shape[1]):
                    if present[b, c] and present[b, 0]:
                        contrasts.append((o[b, c] - o[b, 0]).cpu())
        if raw_read is not None:
            raw = it["query_raw"]
            for name in ("base", "excl", "excl+text+ot+lp", "text+ot+lp"):
                raw_read.add(name, preds[name].cpu().numpy(), gt, raw)
            raw_read.add_g(g.cpu().numpy(), raw_read.kinds(gt, raw))
            log_t, _ = ota.uot_log_plan(l0, masses, params["eps"], params["rho"])
            raw_read.add_ot(log_t, gt)
    stacked = {k: np.stack(v) for k, v in counts.items()}
    cstack = {k: np.stack(v) for k, v in cond.items()}
    res = {"draw": draw, "episodes": len(stacked["base"]), "skipped": skipped, "test_classes": test_classes,
           "params": params, "lp_arm": arm,
           "miou": {k: 100.0 * float(p0.miou_from_counts(v.sum(0))) for k, v in stacked.items()},
           "condition": {k: p5.split_summary(v.sum(0)) for k, v in cstack.items()}}
    if raw_read is not None:
        res["mechanism"] = raw_read.summary()
    if contrasts:
        d = torch.stack(contrasts)  # [n, D]
        res["contrast_share"] = {k: corr.contrast_share(s, d, torch.zeros_like(d)) for k, s in ctx.spaces.items()}
    return res, stacked, cstack


# ------------------------------------------------------------------ decide

def load_draw(out_dir: str, name: str, draw: str, tag: str):
    path = os.path.join(out_dir, f"p11_{draw.replace(':', '_seed')}_{name}{tag}.json")
    if not os.path.isfile(path):
        return None
    return json.load(open(path)), dict(np.load(path.replace(".json", "_counts.npz")))


def holds_at(draws: Dict, a: str, b: str, gain: float) -> Tuple[bool, str]:
    """'Holds at g' of D-48: fixed100 gain >= g with the paired CI above 0, and > 0 on every random600 draw."""
    fixed = p0.paired_bootstrap(draws["fixed100"][1][a], draws["fixed100"][1][b])
    rand = [draws[d][0]["miou"][b] - draws[d][0]["miou"][a] for d in d39.DRAWS[1:4]]
    ok = fixed["gain"] >= gain and fixed["ci_low"] > 0 and min(rand) > 0
    return ok, (f"fixed100 {fixed['gain']:+.2f} [{fixed['ci_low']:+.2f}, {fixed['ci_high']:+.2f}]; random600 "
                f"{[round(r, 2) for r in rand]}")


def as_combos(stacked: Dict[str, np.ndarray]) -> Dict:
    return {c: stacked[combo_label(c)] for c in att.combos(BLOCKS)}


def decide(out_dir: str, name: str, tag: str) -> List[Tuple[str, str]]:
    draws = {d: load_draw(out_dir, name, d, tag) for d in FACTORIAL_DRAWS}
    missing = [d for d, v in draws.items() if v is None]
    if missing:
        return [("incomplete", f"missing draws {missing}")]
    sel = json.load(open(os.path.join(out_dir, f"p11_select_{name}{tag}.json")))
    fit = json.load(open(os.path.join(out_dir, f"p11_fit_{name}{tag}.json")))
    v = []
    mech = draws["valid_raw"][0]["mechanism"]
    # P11.1b
    ok, text = holds_at(draws, "base", "excl", EXCL_GAIN)
    dense = mech["g"]["fg_own"]["share_above"]
    fp = mech["fp_base"]
    v.append((f"P11.1b exclusion {'holds' if ok and dense <= DENSE_SHARE else 'fails'} (psi {sel['psi']:g})",
              f"excl - base {text}; dense novel fg with g > {DENSE_G:g}: {dense:.3f} (limit {DENSE_SHARE:g}, "
              f"report {DENSE_SHARE_REPORT:g}); base-origin FP {fp['base']['fp_base']} -> {fp['excl']['fp_base']}, "
              f"TP {fp['base']['tp_fg']} -> {fp['excl']['tp_fg']}"))
    # P11.2
    vd = draws["valid"][1]
    p = p0.paired_bootstrap(vd["base"], vd["text"])
    ok2 = p["gain"] >= TEXT_GAIN and p["ci_low"] > 0
    v.append((f"P11.2 text prior {'holds' if ok2 else 'fails'} (kappa {sel['kappa']:g})",
              f"text - base valid {p['gain']:+.2f} [{p['ci_low']:+.2f}, {p['ci_high']:+.2f}]; "
              f"spearman(gamma_e, kappa*_e) {sel['spearman_gamma_kappa_star']:+.3f}"))
    # P11.3 (descriptive)
    m_v, m_f = draws["valid"][0]["miou"], draws["fixed100"][0]["miou"]
    rows = [f"{k} bg{cap} {m_v[f'p3:{k}:bg{cap}'] - m_v['base']:+.2f}" for k in corr.SPACES for cap in BG_CAPS]
    v.append(("P11.3 max over cells - U + both (valid)", "; ".join(rows)
              + f"; contaminated background cells {mech['bg_cells_contaminated']:.3f}; oracle contrast share in the "
                f"top-r span {json.dumps({k: round(x, 3) for k, x in draws['valid'][0].get('contrast_share', {}).items()})}"
                f"; energy share {json.dumps({k: round(s['energy_share'], 3) for k, s in fit['spaces'].items() if s})}"))
    # P11.4
    best = max(corr.SPACES, key=lambda k: m_v[f"p4:{k}"])
    gain4 = m_v[f"p4:{best}"] - m_v["U"]  # P11.4 compares with U (amendment 1)
    ok4 = gain4 >= PROBE_MARGIN
    lf = draws["leakfree"][0]["miou"]
    hb = fit["probe_heldout_base"][best]
    cond = draws["valid"][0]["condition"][f"p4:{best}"]
    v.append((f"P11.4 transfer probe {'holds: arm A is trained' if ok4 else 'fails: arm A is not trained'} ({best})",
              f"probe - U valid {gain4:+.2f}; fixed100 "
              f"{m_f[f'p4:{best}']:.2f}; leak-free {lf[f'p4:{best}']:.2f} vs base {lf['base']:.2f}; held-out base "
              f"probe {hb['probe']:.2f} vs U {hb['U']:.2f}; recall own {np.nanmean(cond['recall_own']):.3f} other "
              f"{np.nanmean(cond['recall_other']):.3f}"))
    # P11.5
    ok5, text5 = holds_at(draws, "base", "ot", OT_GAIN)
    lf_d = lf["ot"] - lf["base"]
    ok5 = ok5 and lf_d >= 0
    v.append((f"P11.5 OT {'holds' if ok5 else 'fails'} (eps {sel['eps']:g}, rho {sel['rho']:g})",
              f"ot - base {text5}; leak-free {lf_d:+.2f}; mass on true fg {mech['ot_mass_on_true_fg']:.3f}"))
    # CR attribution (amendment 3)
    fixed_att = att.attribute(as_combos(draws["fixed100"][1]), BLOCKS, PAIRS)
    rands = [att.point_values(as_combos(draws[d][1]), BLOCKS, PAIRS) for d in d39.DRAWS[1:4]]
    for i in BLOCKS:
        q = fixed_att[f"shapley:{i}"]
        v.append((f"D48.1' {i} {'kept' if att.kept(fixed_att, rands, i, KEEP_BAR) else 'not kept'} on CR",
                  f"shapley {q['value']:+.2f} [{q['ci_low']:+.2f}, {q['ci_high']:+.2f}], random600 "
                  f"{[round(r[f'shapley:{i}'], 2) for r in rands]}; add-one {fixed_att[f'add:{i}']['value']:+.2f}; "
                  f"leave-one-out {fixed_att[f'loo:{i}']['value']:+.2f}"))
    for a, b in PAIRS:
        q = fixed_att[f"inter:{a}x{b}"]
        v.append((f"interaction {a} x {b}", f"{q['value']:+.2f} [{q['ci_low']:+.2f}, {q['ci_high']:+.2f}]"))
    q = fixed_att["total"]
    v.append(("total full - base (fixed100)", f"{q['value']:+.2f} [{q['ci_low']:+.2f}, {q['ci_high']:+.2f}]; "
                                               f"full {m_f[combo_label(frozenset(BLOCKS))]:.2f}, base {m_f['base']:.2f}"))
    return v


# ------------------------------------------------------------------ main

def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["fit", "select", "factorial", "decide"])
    p.add_argument("--data_path")
    p.add_argument("--checkpoint", help="name:ours:1:path of a clean-base checkpoint")
    p.add_argument("--name", default="cr")
    p.add_argument("--draw", action="append", help="factorial: draws (default: all)")
    p.add_argument("--max_episodes", type=int, default=None, help="smoke runs only")
    p.add_argument("--bank_episodes", type=int, default=p1.BANK_EPISODES, help="smaller only for smoke runs")
    p.add_argument("--tag", default="")
    p.add_argument("--out_dir", default=OUT_DIR)
    args = p.parse_args(argv)
    out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(REPO, args.out_dir)
    if args.stage == "decide":
        for name, text in decide(out_dir, args.name, args.tag):
            print(f"{name:72s} {text}", flush=True)
        return 0
    if not (args.data_path and args.checkpoint):
        p.error("--data_path and --checkpoint are required")
    device = torch.device("cuda")
    rule, _, _, ck = p6.load_rule(args.checkpoint, device)
    os.makedirs(out_dir, exist_ok=True)
    fit_path = os.path.join(out_dir, f"p11_fit_{ck.name}{args.tag}.pt")
    if args.stage == "fit":
        ctx = Context(rule, ck, args.data_path, device, None, args.bank_episodes)
        fit, info = run_fit(ctx, args.data_path, args.max_episodes)
        torch.save(fit, fit_path)
        p6.save(info, None, f"p11_fit_{ck.name}{args.tag}", out_dir)
        print(f"[fit] {ck.name}: " + json.dumps({k: info[k] for k in ("points", "episodes", "learner_train_acc")})
              + " | held-out base probe vs U " + json.dumps(info["probe_heldout_base"]), flush=True)
        return 0
    ctx = Context(rule, ck, args.data_path, device, torch.load(fit_path, weights_only=True), args.bank_episodes)
    if args.stage == "select":
        cache, test_classes = cache_valid(ctx, args.data_path, args.max_episodes)
        sel = run_select(cache, test_classes)
        p6.save(sel, None, f"p11_select_{ck.name}{args.tag}", out_dir)
        print(f"[select] psi {sel['psi']} kappa {sel['kappa']} eps {sel['eps']} rho {sel['rho']} | spearman "
              f"{sel['spearman_gamma_kappa_star']:+.3f}", flush=True)
        return 0
    sel = json.load(open(os.path.join(out_dir, f"p11_select_{ck.name}{args.tag}.json")))
    params = {k: sel[k] for k in ("psi", "kappa", "eps", "rho")}
    arm = p7.arm_of(json.load(open(os.path.join(REPO, P7_FIXED)))["frozen"])  # P7's frozen arm, selected on CR
    for draw in args.draw or FACTORIAL_DRAWS:
        res, stacked, cstack = run_factorial(ctx, draw, args.data_path, params, arm, args.max_episodes)
        res["checkpoint"] = vars(ck)
        stem = f"p11_{draw.replace(':', '_seed')}_{ck.name}{args.tag}"
        p6.save(res, stacked, stem, out_dir)
        np.savez_compressed(os.path.join(out_dir, stem + "_cond.npz"), **cstack)
        m = res["miou"]
        print(f"[factorial] {draw}: base {m['base']:.2f} | " + " | ".join(
            f"{k} {m[k]:.2f}" for k in ("excl", "text", "ot", "lp", "excl+text+ot+lp")), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
