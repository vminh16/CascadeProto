"""Check a prepared S3DIS block folder against the reference manifest (spec 04 §2).

    python preprocess/verify_s3dis.py                                   # datasets/S3DIS/blocks_bs1_s1
    python preprocess/verify_s3dis.py --data_path <.../blocks_bs1_s1>
    python preprocess/verify_s3dis.py --write                           # (re)create the manifest

The manifest stores, for every room, the number of blocks and one signature per block made of
integers only: the point count and the point count of each of the 13 classes. Point order inside a
room depends on the file-system order of the annotation files, so coordinates or file hashes would
differ between machines; these counts do not. Two machines that ran preprocess/prepare_s3dis.py on
the same zip must produce identical signatures.

Exit code 0 only if every room is present, every block loads, and every signature matches.
"""

import argparse
import glob
import hashlib
import json
import os
import sys
from collections import defaultdict

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA = os.path.join(REPO, "datasets", "S3DIS", "blocks_bs1_s1")
MANIFEST = os.path.join(REPO, "preprocess", "s3dis_blocks_manifest.json")
CLASS_NAMES = ["ceiling", "floor", "wall", "beam", "column", "window", "door",
               "table", "chair", "sofa", "bookcase", "board", "clutter"]  # [VIPSEG dataloaders/s3dis.py:13-14]
MIN_POINTS = 1000  # room2blocks min_npts [VIPSEG preprocess/room2blocks.py:78]


def block_signature(path):
    """[n_points, count of class 0, ..., count of class 12] or None if the block does not load."""
    try:
        block = np.load(path, mmap_mode="r")
    except (OSError, ValueError, EOFError):
        return None
    if block.ndim != 2 or block.shape[1] != 7 or block.shape[0] < MIN_POINTS:
        return None
    labels = np.asarray(block[:, 6]).astype(np.int64)
    if labels.min() < 0 or labels.max() >= len(CLASS_NAMES):
        return None
    return [int(block.shape[0])] + np.bincount(labels, minlength=len(CLASS_NAMES)).tolist()


def scan(data_path):
    """{room: [signature of block 0, block 1, ...]} plus the list of unreadable files."""
    files = glob.glob(os.path.join(data_path, "data", "*.npy"))
    if not files:
        sys.exit(f"no blocks in {os.path.join(data_path, 'data')}")
    rooms, bad = defaultdict(dict), []
    for i, path in enumerate(files):
        name = os.path.basename(path)[:-4]
        room, index = name.rsplit("_block_", 1)
        signature = block_signature(path)
        if signature is None:
            bad.append(name)
        rooms[room][int(index)] = signature
        if (i + 1) % 1000 == 0:
            print(f"  read {i + 1}/{len(files)} blocks", flush=True)
    ordered = {room: [blocks.get(i) for i in range(max(blocks) + 1)] for room, blocks in rooms.items()}
    return ordered, sorted(bad)


def room_digest(signatures):
    return hashlib.sha256(json.dumps(signatures).encode()).hexdigest()[:16]


def summary(rooms):
    per_area = defaultdict(int)
    points = np.zeros(len(CLASS_NAMES), dtype=np.int64)
    for room, signatures in rooms.items():
        per_area[room.split("_")[0] + "_" + room.split("_")[1]] += 1
        for s in signatures:
            if s is not None:
                points += np.asarray(s[1:])
    return dict(sorted(per_area.items())), {c: int(n) for c, n in zip(CLASS_NAMES, points)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data_path", default=DEFAULT_DATA)
    parser.add_argument("--write", action="store_true", help="write the manifest from this folder")
    args = parser.parse_args()

    meta = os.path.join(os.path.dirname(os.path.normpath(args.data_path)), "meta", "s3dis_classnames.txt")
    names_ok = os.path.isfile(meta) and [l.strip() for l in open(meta) if l.strip()] == CLASS_NAMES
    print(f"data: {os.path.abspath(args.data_path)}")
    rooms, bad = scan(args.data_path)
    per_area, class_points = summary(rooms)
    n_blocks = sum(len(s) for s in rooms.values())
    print(f"rooms {len(rooms)} | blocks {n_blocks} | rooms per area {per_area}")
    print(f"points per class: {class_points}")

    if args.write:
        if bad or not names_ok:
            sys.exit(f"refusing to write a manifest: {len(bad)} unreadable blocks, class-name file ok={names_ok}")
        manifest = {"rooms": {r: {"blocks": len(s), "digest": room_digest(s), "signatures": s}
                              for r, s in sorted(rooms.items())},
                    "n_rooms": len(rooms), "n_blocks": n_blocks, "rooms_per_area": per_area,
                    "points_per_class": class_points}
        with open(MANIFEST, "w") as f:
            json.dump(manifest, f, separators=(",", ":"))
        print(f"wrote {MANIFEST}")
        return 0

    reference = json.load(open(MANIFEST))["rooms"]
    missing = sorted(set(reference) - set(rooms))
    extra = sorted(set(rooms) - set(reference))
    differ = sorted(r for r in set(rooms) & set(reference) if room_digest(rooms[r]) != reference[r]["digest"])
    problems = {
        "class-name file wrong or missing": [] if names_ok else [meta],
        "unreadable or too-small blocks": bad,
        "rooms missing": missing,
        "unexpected rooms": extra,
        "rooms whose blocks differ from the reference": differ,
    }
    for label, items in problems.items():
        print(f"{'FAIL' if items else 'ok  '} {label}: {len(items)}" + (f" {items[:10]}" if items else ""))
    for room in differ[:5]:
        got, want = rooms[room], reference[room]["signatures"]
        print(f"     {room}: {len(got)} blocks here vs {len(want)} in the reference; "
              f"first differing block {next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want)))}")
    failed = any(problems.values())
    print("RESULT:", "FAIL" if failed else f"OK, identical to the reference ({len(reference)} rooms, {n_blocks} blocks)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
