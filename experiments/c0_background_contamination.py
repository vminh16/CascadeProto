"""C0: how much of a support block's background belongs to the episode's other way (CPU, data only).

    .venv/Scripts/python experiments/c0_background_contamination.py --cvfold 1 --episodes 3000

Hypothesis after P3 (`results/phase16_p3/SUMMARY.md`): floor and wall, present in almost every S3DIS block,
are labelled background in the other way's support block; the background prototype pools the mask-0 points
of all support blocks [VIPSEG models/vipseg.py:108-116], so it absorbs the episode's own foreground classes.

For random 2-way episodes of the fold's test classes, drawn like the loader's test episodes (query and
support blocks from its class2scans lists, no block reused within an episode [VIPSEG dataloaders/loader.py:
174-200]), this reads each support block's raw semantic labels (column 6 of the block file) and reports, per
class pair, the share of the pooled background points that carry the OTHER way's class. Blocks are used
whole: the loader keeps the class's share of points and samples the rest uniformly
[VIPSEG dataloaders/loader.py:36-53], so whole-block shares estimate the sampled ones. Writes nothing.
"""

import argparse
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from dataloaders.s3dis import S3DISDataset  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", default=os.path.join(REPO, "datasets", "S3DIS", "blocks_bs1_s1"))
    p.add_argument("--cvfold", type=int, default=1, choices=[0, 1])
    p.add_argument("--episodes", type=int, default=3000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    ds = S3DISDataset(args.cvfold, args.data_path)
    names = [line.strip() for line in open(os.path.join(os.path.dirname(args.data_path), "meta", "s3dis_classnames.txt"))]
    classes = list(ds.test_classes)
    rng = np.random.default_rng(args.seed)
    cache = {}

    def labels(scan):
        if scan not in cache:
            cache[scan] = np.load(os.path.join(args.data_path, "data", f"{scan}.npy"))[:, 6].astype(np.int64)
        return cache[scan]

    per_pair = {}
    pooled = {"other_way_in_bg": 0, "bg": 0, "blocks_with_other_way": 0, "blocks": 0}
    for _ in range(args.episodes):
        pair = rng.choice(classes, 2, replace=False)
        used, support = [], []
        for c in pair:  # the loader's order: query block first, then support block, none reused
            q = rng.choice([s for s in ds.query_class2scans[c] if s not in used])
            used.append(q)
            s = rng.choice([s for s in ds.support_class2scans[c] if s not in used])
            used.append(s)
            support.append((c, s))
        bg_total = other_total = 0
        for k, (c, scan) in enumerate(support):
            lab = labels(scan)
            other = pair[1 - k]
            bg = lab != c
            n_other = int((lab[bg] == other).sum())
            bg_total += int(bg.sum())
            other_total += n_other
            pooled["blocks"] += 1
            pooled["blocks_with_other_way"] += n_other > 0
        key = tuple(sorted(int(c) for c in pair))
        acc = per_pair.setdefault(key, [0, 0, 0])
        acc[0] += other_total
        acc[1] += bg_total
        acc[2] += 1
        pooled["other_way_in_bg"] += other_total
        pooled["bg"] += bg_total
    print(f"fold S{args.cvfold}, test classes {[names[c] for c in classes]}, {args.episodes} episodes")
    print(f"pooled background points that belong to the other way: "
          f"{pooled['other_way_in_bg'] / pooled['bg']:.3f}; support blocks containing the other way's class: "
          f"{pooled['blocks_with_other_way'] / pooled['blocks']:.3f}")
    for (a, b), (o, t, n) in sorted(per_pair.items(), key=lambda kv: -kv[1][0] / max(kv[1][1], 1)):
        print(f"  {names[a]:>8s} + {names[b]:<8s} episodes {n:4d}  other-way share of pooled background {o / t:.3f}")
    # per class: when class c is a way, how much of the OTHER support block's background is c
    print("per class: share of the other support block's background that is this class, when this class is a way")
    for c in classes:
        shares = []
        for _ in range(300):
            partner = rng.choice([x for x in classes if x != c])
            scan = rng.choice(ds.support_class2scans[partner])
            lab = labels(scan)
            bg = lab != partner
            shares.append((lab[bg] == c).mean() if bg.any() else 0.0)
        print(f"  {names[c]:>8s} mean {np.mean(shares):.3f}, blocks where present {np.mean(np.array(shares) > 0):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
