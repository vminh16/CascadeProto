"""D-48 debate: CPU-only checks from the S3DIS blocks and stored artefacts. Writes nothing in the repo.

1. RNG: a wrapper that re-implements generate_one_episode with support=False and a reordered class list
   reproduces the inherited episode bit for bit (points, query labels, support masks) with pc_augm on.
2. Composition of S1 test query blocks (seeded draw, the loader's sampler): base / clutter / novel-out shares.
3. Composition of S1 training query blocks: who is dense, what "none" is.
4. fixed100 stored episodes: support foreground counts (cells for M_fg = 16), background counts (M_bg = 32).
5. Proportional estimate of P11.1's Delta_base from CR's U+both+LP counts (estimate, not a measurement).
"""
import glob
import json
import os
import random
import sys
from collections import defaultdict

import h5py
import numpy as np

REPO = r"C:\Users\USER\Desktop\CascadeProto"
sys.path.insert(0, REPO)
from dataloaders.loader import MyDataset, sample_pointcloud  # noqa: E402

DATA = os.path.join(REPO, "datasets", "S3DIS", "blocks_bs1_s1")
NUM_POINT, PCA = 2048, "xyzrgbXYZ"
AUG = {"scale": 0, "rot": 1, "mirror_prob": 0, "jitter": 1, "shift": 0.1, "random_color": 0}
NAMES = [l.strip() for l in open(os.path.join(REPO, "datasets", "S3DIS", "meta", "s3dis_classnames.txt"))]
CLUTTER = 12
out = {}

# ---------------------------------------------------------------- 1. RNG / wrapper check
class BaseLabelDataset(MyDataset):
    """generate_one_episode with one call per block: support=False and classes [targets..., other base]."""
    def generate_one_episode(self, sampled_classes):
        others = [c for c in self.classes if c not in sampled_classes]
        order = list(sampled_classes) + others
        sx, sm, qx, ql, qb, sb = [], [], [], [], [], []
        black = []
        for c in sampled_classes:
            qs = [x for x in self.query_class2scans[c].copy() if x not in black] if black else self.query_class2scans[c].copy()
            q = np.random.choice(qs, self.n_queries, replace=False); black.extend(q)
            ss = [x for x in self.support_class2scans[c].copy() if x not in black] if black else self.support_class2scans[c].copy()
            s = np.random.choice(ss, self.k_shot, replace=False); black.extend(s)
            qp, qg = zip(*[sample_pointcloud(self.data_path, self.num_point, self.pc_attribs, self.pc_augm,
                                             self.pc_augm_config, n, order, c, support=False) for n in q])
            sp, sg = zip(*[sample_pointcloud(self.data_path, self.num_point, self.pc_attribs, self.pc_augm,
                                             self.pc_augm_config, n, order, c, support=False) for n in s])
            qg, sg = np.stack(qg), np.stack(sg)
            qx.append(np.stack(qp)); ql.append(np.where(qg <= len(sampled_classes), qg, 0)); qb.append(qg)
            k = list(sampled_classes).index(c) + 1
            sx.append(np.stack(sp)); sm.append((sg == k).astype(np.int32)); sb.append(sg)
        self._base = (np.concatenate(qb), np.stack(sb))
        return np.stack(sx), np.stack(sm), np.concatenate(qx), np.concatenate(ql)


def seeded(ds, i, seed=0):
    np.random.seed([seed, 2, i]); random.seed(f"{seed}-2-{i}")
    return ds[i]


ref = MyDataset(DATA, "s3dis", cvfold=1, num_episode=100, n_way=2, k_shot=1, n_queries=1, mode="train",
                num_point=NUM_POINT, pc_attribs=PCA, pc_augm=True, pc_augm_config=AUG)
wrp = BaseLabelDataset(DATA, "s3dis", cvfold=1, num_episode=100, n_way=2, k_shot=1, n_queries=1, mode="train",
                       num_point=NUM_POINT, pc_attribs=PCA, pc_augm=True, pc_augm_config=AUG)
ok = 0
N_RNG = 20
for i in range(N_RNG):
    a, b = seeded(ref, i), seeded(wrp, i)
    same = all(np.array_equal(x, y) for x, y in zip(a, b))
    ok += same
# the naive wrapper: a second call without restoring the RNG draws other points
np.random.seed(5); random.seed(5)
p1, _ = sample_pointcloud(DATA, NUM_POINT, PCA, True, AUG, ref.query_class2scans[8][0], [8, 0], 8)
p2, _ = sample_pointcloud(DATA, NUM_POINT, PCA, True, AUG, ref.query_class2scans[8][0], [0, 3, 4, 8, 10, 11], 8)
out["rng"] = {"episodes": N_RNG, "identical": ok, "naive_second_call_identical": bool(np.array_equal(p1, p2))}
print("RNG", out["rng"], flush=True)

# ---------------------------------------------------------------- helpers
cache = {}
def raw(scan):
    if scan not in cache:
        cache[scan] = np.load(os.path.join(DATA, "data", f"{scan}.npy"), mmap_mode="r")[:, 6].astype(np.int64)
    return cache[scan]


def idx_of(lab, c):  # the inherited sampler's calls (loader.py:38-58), labels only
    n = lab.shape[0]
    valid = np.nonzero(lab == c)[0]
    nv = len(valid) if n < NUM_POINT else int(len(valid) / float(n) * NUM_POINT)
    a = np.random.choice(valid, nv, replace=False)
    b = np.random.choice(np.arange(n), NUM_POINT - nv, replace=(n < NUM_POINT))
    return np.concatenate([a, b]), nv


def draw(ds, classes_pool, per_pair, seed):
    import itertools
    pairs = list(itertools.combinations(sorted(int(c) for c in classes_pool), 2))
    for pi, pair in enumerate(pairs):
        for e in range(per_pair):
            np.random.seed([seed, pi, e])
            used, blocks = [], []
            for c in pair:
                q = np.random.choice([s for s in ds.query_class2scans[c] if s not in used]); used.append(q)
                s = np.random.choice([s for s in ds.support_class2scans[c] if s not in used]); used.append(s)
                blocks.append((c, q, s))
            yield pair, blocks


def composition(mode, per_pair, seed):
    ds = MyDataset(DATA, "s3dis", cvfold=1, num_episode=1, n_way=2, k_shot=1, n_queries=1, mode=mode,
                   num_point=NUM_POINT, pc_attribs=PCA, pc_augm=False)
    train_c = list(ds.dataset.train_classes); test_c = list(ds.dataset.test_classes)
    pool = train_c if mode == "train" else test_c
    tot = defaultdict(float)
    per_base = defaultdict(float)
    per_cls = defaultdict(lambda: defaultdict(float))  # fg class -> raw class shares in its own query blocks
    dense_share = []
    beta_c = defaultdict(lambda: [0.0, 0.0])  # per episode class c: base points / non-c points in the episode's queries
    blocks_with_base = 0; nblocks = 0
    for pair, blocks in draw(ds, pool, per_pair, seed):
        ep_raw = []
        for w, (c, q, s) in enumerate(blocks):
            lab = raw(q)
            np.random.seed([seed, 99, __import__("zlib").crc32(str(q).encode()), w])
            idx, nv = idx_of(lab, c)
            r = lab[idx]
            ep_raw.append(r)
            nblocks += 1
            dense_share.append(float((r == c).mean()))
            other = pair[1 - w]
            tot["own_target"] += (r == c).sum(); tot["other_target"] += (r == other).sum()
            bg = (r != c) & (r != other)
            isbase = np.isin(r, train_c) & bg
            tot["base_bg"] += isbase.sum(); tot["clutter_bg"] += ((r == CLUTTER) & bg).sum()
            tot["novel_bg"] += (np.isin(r, test_c) & bg).sum(); tot["bg"] += bg.sum()
            blocks_with_base += isbase.sum() >= 20
            for b in np.unique(r[isbase]):
                per_base[NAMES[b]] += (r[isbase] == b).sum()
            for k in np.unique(r):
                per_cls[NAMES[c]][NAMES[k]] += (r == k).sum()
        er = np.concatenate(ep_raw)
        for c in pair:
            nonc = er != c
            beta_c[c][0] += (np.isin(er, train_c) & nonc & (er != pair[0]) & (er != pair[1])).sum()
            beta_c[c][1] += nonc.sum()
    res = {k: float(v) for k, v in tot.items()}
    res["base_share_of_bg"] = tot["base_bg"] / tot["bg"]
    res["clutter_share_of_bg"] = tot["clutter_bg"] / tot["bg"]
    res["novel_share_of_bg"] = tot["novel_bg"] / tot["bg"]
    res["bg_share_of_points"] = tot["bg"] / (tot["bg"] + tot["own_target"] + tot["other_target"])
    res["blocks_with_20plus_base_points"] = blocks_with_base / nblocks
    res["per_base_class_share_of_bg"] = {k: v / tot["bg"] for k, v in sorted(per_base.items(), key=lambda kv: -kv[1])}
    res["own_class_share_in_block_median"] = float(np.median(dense_share))
    res["per_fg_class_block_composition"] = {c: {k: round(v / sum(d.values()), 3) for k, v in sorted(d.items(), key=lambda kv: -kv[1])[:6]}
                                             for c, d in per_cls.items()}
    res["beta_c"] = {NAMES[c]: v[0] / v[1] for c, v in beta_c.items()}
    res["episodes"] = per_pair * len(list(__import__('itertools').combinations(pool, 2)))
    return res


out["test_composition"] = composition("test", 40, 11)
print(json.dumps({k: v for k, v in out["test_composition"].items() if k != "per_fg_class_block_composition"}, indent=1), flush=True)
out["train_composition"] = composition("train", 40, 12)
print(json.dumps({k: v for k, v in out["train_composition"].items() if k != "per_fg_class_block_composition"}, indent=1), flush=True)

# ---------------------------------------------------------------- 4. fixed100 stored episodes
fdir = os.path.join(DATA, "vipseg_eval_S_1_N_2_K_1_test_episodes_100_pts_2048")
fg_s, bg_s, fg_q = [], [], []
for f in glob.glob(os.path.join(fdir, "*.h5")):
    with h5py.File(f, "r") as h:
        sm = h["support_masks"][:]  # [N, K, P]
        ql = h["query_labels"][:]   # [N, P]
    fg_s.extend(sm.reshape(-1, sm.shape[-1]).sum(-1).tolist())
    bg_s.append(int((sm == 0).sum()))
    fg_q.append(float((ql > 0).mean()))
fg_s = np.array(fg_s); bg_s = np.array(bg_s)
out["fixed100_support"] = {
    "blocks": int(fg_s.size), "fg_min": int(fg_s.min()), "fg_p5": float(np.percentile(fg_s, 5)),
    "fg_median": float(np.median(fg_s)), "fg_max": int(fg_s.max()),
    "share_fg_below_320": float((fg_s < 320).mean()),  # < 20 points per cell at M_fg = 16
    "points_per_fg_cell_min_median": [float(fg_s.min() / 16), float(np.median(fg_s) / 16)],
    "bg_per_episode_min_median": [int(bg_s.min()), float(np.median(bg_s))],
    "query_fg_share_median": float(np.median(fg_q))}
print("fixed100", out["fixed100_support"], flush=True)

# ---------------------------------------------------------------- 5. proportional Delta_base estimate
z = np.load(os.path.join(REPO, "results", "phase16_d46", "test_S1_fixed100_counts.npz"))
j = json.load(open(os.path.join(REPO, "results", "phase16_d46", "test_S1_fixed100.json")))
tc = j["test_classes"]
est = {}
for arm in ("cr:U", "cr:lp"):
    c = z[arm].sum(0)
    gt, pr, tp = c[0], c[1], c[2]
    fp = pr - tp
    base = (tp / (gt + pr - tp))[1:].mean()
    beta = np.array([0.0] + [out["test_composition"]["beta_c"][NAMES[k]] for k in tc])
    prop = (tp / (gt + pr - tp - beta * fp))[1:].mean()
    upper = (tp / gt)[1:].mean()
    est[arm] = {"miou": 100 * base, "prop_estimate": 100 * prop, "delta_prop": 100 * (prop - base),
                "all_fp_removed": 100 * upper, "fp_share_of_pred": {NAMES[k]: float(fp[i + 1] / pr[i + 1]) for i, k in enumerate(tc)}}
out["delta_base_estimate"] = est
print(json.dumps(est, indent=1), flush=True)
json.dump(out, open(os.path.join(os.path.dirname(__file__), "d48_cpu_checks.json"), "w"), indent=1, default=float)
