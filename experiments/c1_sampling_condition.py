"""C1: the sampling condition of every class occurrence in AttMPTI/VIP-Seg episodes (CPU, no model).

The inherited sampler [VIPSEG dataloaders/loader.py:31-58] draws `int(ratio * 2048)` points of the class a
block is sampled for, then `2048 - that` points from the whole block, target class included. The class a
block is sampled for is therefore over-sampled (density factor 2 - pi relative to uniform sampling, pi its
share of the raw block) and every other class, including the other episode classes present in that block, is
under-sampled (factor 1 - pi). COSeg calls this "foreground leakage" [COSeg §3.2, Tab.1] but only for the
sampled-for class; in an N-way episode the query block of way k is sampled for class k, so a class c != k that
also appears in it is at background density ("other" condition), while the support always shows c in the
over-sampled ("own") condition.

For every query block of seeded test episodes this script records, per episode class present in it: whether
the block was sampled for it (own) or not (other), the number of points, the median distance to the 16th
nearest neighbour in xyz (the encoder's k [VIPSEG models/vipseg.py:39]) and the share of exact duplicate
points. Episodes come from the inherited `MyDataset.__getitem__` (read only, no augmentation, as in testing).

    python experiments/c1_sampling_condition.py --data_path datasets/S3DIS/blocks_bs1_s1 --cvfold 1
"""

import argparse
import itertools
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataloaders.loader import MyDataset  # noqa: E402  inherited, unchanged
from pipeline.episodes import NUM_POINT, PC_ATTRIBS, WAY_NUM, WAY_RATIO, read_class_names  # noqa: E402

K_NEIGHBOURS = 16  # [VIPSEG models/vipseg.py:39]


def knn_radius(xyz: np.ndarray, k: int = K_NEIGHBOURS) -> np.ndarray:
    """Distance of every point to its k-th nearest other point [P]; duplicates count as neighbours at 0."""
    d, _ = cKDTree(xyz).query(xyz, k=k + 1)  # [P, k+1], column 0 is the point itself
    return d[:, -1]


def duplicate_mask(xyz: np.ndarray) -> np.ndarray:
    """True for points whose xyz occurs more than once in the block [P] (the sampler draws some twice)."""
    _, inverse, counts = np.unique(xyz, axis=0, return_inverse=True, return_counts=True)
    return counts[inverse.ravel()] > 1


def run(data_path: str, cvfold: int, episodes_per_pair: int, seed: int) -> dict:
    names = read_class_names(data_path, "s3dis")
    np.random.seed(seed)
    ds = MyDataset(data_path, "s3dis", cvfold=cvfold, n_way=2, k_shot=1, n_queries=1, mode="test",
                   num_point=NUM_POINT, pc_attribs=PC_ATTRIBS, pc_augm=False, way_ratio=WAY_RATIO, way_num=WAY_NUM)
    classes = [int(c) for c in ds.classes]
    acc = {c: {k: [] for k in ("own_n", "other_n", "own_r", "other_r", "own_dup", "other_dup")} for c in classes}
    bg_r, sup_fg_r, sup_bg_r = [], [], []
    for pair in itertools.combinations(classes, 2):
        for _ in range(episodes_per_pair):
            sx, sy, qx, qy, sc = ds.__getitem__(0, list(pair))  # query b was sampled for sc[b] (label b+1)
            for b in range(qx.shape[0]):
                xyz = qx[b, :, :3]  # [P, 3], metres, shifted to min 0 [VIPSEG dataloaders/loader.py:65-66]
                r, dup = knn_radius(xyz), duplicate_mask(xyz)
                for w, c in enumerate(sc):
                    m = qy[b] == w + 1
                    if not m.any():
                        continue
                    key = "own" if w == b else "other"
                    acc[int(c)][key + "_n"].append(int(m.sum()))
                    acc[int(c)][key + "_r"].append(float(np.median(r[m])))
                    acc[int(c)][key + "_dup"].append(float(dup[m].mean()))
                bg = qy[b] == 0
                if bg.any():
                    bg_r.append(float(np.median(r[bg])))
            for w in range(sx.shape[0]):
                r = knn_radius(sx[w, 0, :, :3])
                sup_fg_r.append(float(np.median(r[sy[w, 0] == 1])))
                if (sy[w, 0] == 0).any():
                    sup_bg_r.append(float(np.median(r[sy[w, 0] == 0])))
    out = {"cvfold": cvfold, "episodes_per_pair": episodes_per_pair, "seed": seed, "classes": {},
           "support_fg_radius_median": float(np.median(sup_fg_r)),
           "support_bg_radius_median": float(np.median(sup_bg_r)),
           "query_bg_radius_median": float(np.median(bg_r))}
    tot_own = tot_other = 0
    for c in classes:
        a = acc[c]
        n_own, n_other = int(np.sum(a["own_n"])), int(np.sum(a["other_n"]))
        tot_own, tot_other = tot_own + n_own, tot_other + n_other
        out["classes"][names[c]] = {
            "other_share_of_points": n_other / max(n_own + n_other, 1),
            "blocks_own": len(a["own_n"]), "blocks_other": len(a["other_n"]),
            "median_points_own": float(np.median(a["own_n"])) if a["own_n"] else None,
            "median_points_other": float(np.median(a["other_n"])) if a["other_n"] else None,
            "knn16_radius_own": float(np.median(a["own_r"])) if a["own_r"] else None,
            "knn16_radius_other": float(np.median(a["other_r"])) if a["other_r"] else None,
            "duplicate_share_own": float(np.mean(a["own_dup"])) if a["own_dup"] else None,
            "duplicate_share_other": float(np.mean(a["other_dup"])) if a["other_dup"] else None,
        }
    out["other_share_all_foreground_points"] = tot_other / max(tot_own + tot_other, 1)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data_path", required=True)
    p.add_argument("--cvfold", type=int, required=True, choices=[0, 1])
    p.add_argument("--episodes_per_pair", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_json", default=None)
    args = p.parse_args(argv)
    out = run(args.data_path, args.cvfold, args.episodes_per_pair, args.seed)
    print(f"S{args.cvfold}, {args.episodes_per_pair} episodes per class pair; kNN-16 radius median [m]: support fg "
          f"{out['support_fg_radius_median']:.4f}, support bg {out['support_bg_radius_median']:.4f}, query bg "
          f"{out['query_bg_radius_median']:.4f}")
    print("class | other share | blocks own/other | median points own/other | kNN16 radius own/other | dup own/other")
    for name, s in out["classes"].items():
        fmt = lambda v, f: "-" if v is None else format(v, f)  # noqa: E731
        print(f"{name:8s} | {s['other_share_of_points']:.3f} | {s['blocks_own']}/{s['blocks_other']} | "
              f"{fmt(s['median_points_own'], '.0f')}/{fmt(s['median_points_other'], '.0f')} | "
              f"{fmt(s['knn16_radius_own'], '.4f')}/{fmt(s['knn16_radius_other'], '.4f')} | "
              f"{fmt(s['duplicate_share_own'], '.3f')}/{fmt(s['duplicate_share_other'], '.3f')}")
    print(f"other-condition share of all foreground query points: {out['other_share_all_foreground_points']:.3f}")
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
