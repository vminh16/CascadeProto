"""Download S3DIS and build the AttMPTI/VIP-Seg block layout of spec 04 §2.

Steps (each is skipped when its output already exists):
  1. Download Stanford3dDataset_v1.2_Aligned_Version.zip from the Hugging Face backup
     `cminst/S3DIS` (copy of https://cvg-data.inf.ethz.ch/s3dis/), resumable, SHA-256 checked.
  2. Extract it into <root>/.
  3. Replace the stray byte in Area_5/hallway_6 (ceiling_1.txt line 180389, where a space belongs)
     that the inherited script leaves to be "fixed manually" [VIPSEG preprocess/collect_s3dis_data.py:94].
  4. Write <root>/meta/s3dis_classnames.txt in the loader's id order [VIPSEG dataloaders/s3dis.py:13-14].
  5. Run the inherited preprocess/collect_s3dis_data.py (rooms -> scenes/data/*.npy).
  6. Run the inherited preprocess/room2blocks.py with default arguments (-> blocks_bs1_s1/data).
  7. Check that all 272 rooms have blocks. Rooms the inherited script skipped are redone one by
     one with its own functions, so errors surface instead of being printed and ignored.

The inherited scripts are called unchanged; they hard-code <repo>/datasets/S3DIS as the output
root and split paths on '/', so run this on Linux/WSL2 from any directory.

Usage:
    python preprocess/prepare_s3dis.py                 # full pipeline
    python preprocess/prepare_s3dis.py --delete_raw    # also delete the zip and raw txt afterwards
    python preprocess/prepare_s3dis.py --blocks_only   # keep only blocks_bs1_s1 + meta

Peak disk use is about 50 GB (zip 4 GB, raw txt 17 GB, scenes 15 GB, blocks 14 GB). A rerun only
collects or splits rooms that are missing, so it never rewrites finished outputs.

The dataset is public; HF_TOKEN (read from the environment or <repo>/.env) is optional and only
raises Hugging Face's anonymous rate limits.
"""

import argparse
import glob
import hashlib
import os
import shutil
import subprocess
import sys
import time
import zipfile

import numpy as np
import requests

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.join(REPO, "datasets", "S3DIS")  # fixed by the inherited collect_s3dis_data.py:69
RAW_DIR = os.path.join(ROOT, "Stanford3dDataset_v1.2_Aligned_Version")
ZIP_NAME = "Stanford3dDataset_v1.2_Aligned_Version.zip"
ZIP_URL = f"https://huggingface.co/datasets/cminst/S3DIS/resolve/main/{ZIP_NAME}"
ZIP_SIZE = 4394701719
ZIP_SHA256 = "f16db73310983e6b4df9b3d5290d20315e0cdf56c14f0d795f96d372378fce94"

# Id order of the loader [VIPSEG dataloaders/s3dis.py:13-14]; clutter = 12.
CLASS_NAMES = ["ceiling", "floor", "wall", "beam", "column", "window", "door",
               "table", "chair", "sofa", "bookcase", "board", "clutter"]
N_ROOMS = 272  # [PAPER §4.1]


def log(msg):
    print(msg, flush=True)


def hf_token():
    token = os.environ.get("HF_TOKEN")
    env_file = os.path.join(REPO, ".env")
    if not token and os.path.isfile(env_file):
        for line in open(env_file, encoding="utf-8"):
            key, _, value = line.strip().partition("=")
            if key == "HF_TOKEN" and value:
                token = value.strip().strip('"').strip("'")
    return token or None


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(zip_path):
    part = zip_path + ".part"
    done = os.path.getsize(part) if os.path.exists(part) else 0
    headers = {"Range": f"bytes={done}-"} if done else {}
    token = hf_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    log(f"[1/7] Downloading {ZIP_NAME} ({ZIP_SIZE / 2**30:.2f} GiB), resuming at {done / 2**20:.0f} MiB")
    with requests.get(ZIP_URL, headers=headers, stream=True, timeout=60) as r:
        r.raise_for_status()
        mode = "ab" if r.status_code == 206 else "wb"
        if mode == "wb":
            done = 0
        last = time.time()
        with open(part, mode) as f:
            for chunk in r.iter_content(chunk_size=8 * 2**20):
                f.write(chunk)
                done += len(chunk)
                if time.time() - last > 30:
                    log(f"      {done / ZIP_SIZE:6.1%}  {done / 2**20:,.0f} MiB")
                    last = time.time()
    if os.path.getsize(part) != ZIP_SIZE:
        raise RuntimeError(f"size {os.path.getsize(part)} != {ZIP_SIZE}; rerun to resume")
    os.replace(part, zip_path)


def fix_hallway_6():
    """Replace bytes that cannot appear in a number by a space in Area_5/hallway_6 annotations.

    The stray byte in ceiling_1.txt stands between two colour values ("185<byte>187"); deleting
    it would merge them into one column, so it is replaced by a space.
    """
    allowed = set(b"0123456789.-+eE \t\r\n")
    fixed = []
    for path in glob.glob(os.path.join(RAW_DIR, "Area_5", "hallway_6", "Annotations", "*.txt")):
        data = open(path, "rb").read()
        clean = bytes(b if b in allowed else 0x20 for b in data)
        if clean != data:
            open(path, "wb").write(clean)
            fixed.append((os.path.basename(path), sum(a != b for a, b in zip(data, clean))))
    return fixed


def run_inherited(script, *args):
    cmd = [sys.executable, os.path.join(REPO, "preprocess", script), *args]
    log("      $ " + " ".join(cmd))
    subprocess.run(cmd, cwd=REPO, check=True, stdout=subprocess.DEVNULL)


def raw_rooms():
    """{Area_a_room: annotation dir} for every room of the extracted dataset ({} if deleted)."""
    return {f"{os.path.basename(os.path.dirname(d))}_{os.path.basename(d)}": os.path.join(d, "Annotations")
            for d in glob.glob(os.path.join(RAW_DIR, "Area_*", "*")) if os.path.isdir(d)}


def room_names(folder, suffix):
    return {os.path.basename(p)[:-len(suffix)] for p in glob.glob(os.path.join(folder, "*" + suffix))}


def rooms_with_blocks(blocks):
    return {os.path.basename(p).rsplit("_block_", 1)[0] for p in glob.glob(os.path.join(blocks, "*.npy"))}


def collect_rooms(rooms, scenes):
    """Collect the given rooms with the inherited collect_point_label (same output as its script)."""
    sys.path.insert(0, REPO)
    from preprocess import collect_s3dis_data as inherited

    inherited.CLASS_NAMES = CLASS_NAMES  # set by the script's __main__ block [VIPSEG ...collect_s3dis_data.py:73-74]
    inherited.CLASS2LABEL = {name: i for i, name in enumerate(CLASS_NAMES)}
    os.makedirs(scenes, exist_ok=True)
    for name, anno_path in rooms.items():
        inherited.collect_point_label(anno_path, os.path.join(scenes, f"{name}.npy"))


def split_rooms(rooms, scenes, blocks):
    """Blocks for the given rooms with the inherited room2blocks_wrapper and its default arguments."""
    sys.path.insert(0, REPO)
    from preprocess.room2blocks import room2blocks_wrapper

    os.makedirs(blocks, exist_ok=True)
    for name in rooms:
        for i, block in enumerate(room2blocks_wrapper(os.path.join(scenes, f"{name}.npy"),
                                                      block_size=1, stride=1, min_npts=1000)):
            np.save(os.path.join(blocks, f"{name}_block_{i}.npy"), block)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--delete_raw", action="store_true",
                        help="delete the zip and the extracted txt files after a successful run")
    parser.add_argument("--blocks_only", action="store_true",
                        help="also delete scenes/, keeping only blocks_bs1_s1 and meta (~15 GB instead of ~50 GB)")
    args = parser.parse_args()
    os.makedirs(ROOT, exist_ok=True)
    zip_path = os.path.join(ROOT, ZIP_NAME)
    scenes = os.path.join(ROOT, "scenes", "data")
    blocks = os.path.join(ROOT, "blocks_bs1_s1", "data")

    if not os.path.isdir(RAW_DIR) and not os.path.isfile(zip_path):
        download(zip_path)
    if os.path.isfile(zip_path) and not os.path.isdir(RAW_DIR):
        log("      verifying SHA-256")
        if sha256_of(zip_path) != ZIP_SHA256:
            raise RuntimeError(f"SHA-256 mismatch for {zip_path}; delete it and rerun")
        log(f"[2/7] Extracting into {ROOT}")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(ROOT)

    log("[3/7] Cleaning Area_5/hallway_6")
    for name, n in fix_hallway_6():
        log(f"      replaced {n} stray byte(s) by a space in {name}")

    log("[4/7] Writing meta/s3dis_classnames.txt")
    os.makedirs(os.path.join(ROOT, "meta"), exist_ok=True)
    with open(os.path.join(ROOT, "meta", "s3dis_classnames.txt"), "w") as f:
        f.write("\n".join(CLASS_NAMES) + "\n")

    raw = raw_rooms()
    if not glob.glob(os.path.join(scenes, "*.npy")):
        log("[5/7] Collecting rooms (inherited collect_s3dis_data.py, ~30 min)")
        run_inherited("collect_s3dis_data.py", "--data_path", RAW_DIR)
    missing = sorted(set(raw) - room_names(scenes, ".npy"))
    if missing:
        # collect_s3dis_data.py swallows per-room errors [VIPSEG preprocess/collect_s3dis_data.py:99-102];
        # redo only the missing rooms with its own function, letting errors surface.
        log(f"[5/7] Collecting {len(missing)} missing room(s): {missing}")
        collect_rooms({name: raw[name] for name in missing}, scenes)
    else:
        log("[5/7] All rooms collected")

    if not glob.glob(os.path.join(blocks, "*.npy")):
        log("[6/7] Splitting rooms into 1 m blocks (inherited room2blocks.py)")
        run_inherited("room2blocks.py", "--data_path", os.path.join(ROOT, "scenes"), "--dataset", "s3dis")
    no_blocks = sorted(room_names(scenes, ".npy") - rooms_with_blocks(blocks))
    if no_blocks:
        log(f"[6/7] Splitting {len(no_blocks)} room(s) without blocks: {no_blocks}")
        split_rooms(no_blocks, scenes, blocks)
    else:
        log("[6/7] Every room has blocks")

    log("[7/7] Checking outputs")
    covered = rooms_with_blocks(blocks)
    expected = set(raw) if raw else None
    if expected is not None and covered != expected:
        raise RuntimeError(f"rooms without blocks: {sorted(expected - covered)}")
    if len(covered) != N_ROOMS:
        raise RuntimeError(f"{len(covered)} rooms have blocks, expected {N_ROOMS}")
    n_blocks = len(glob.glob(os.path.join(blocks, "*.npy")))
    log(f"      {len(covered)} rooms, {n_blocks} blocks in {os.path.dirname(blocks)}")

    if args.delete_raw or args.blocks_only:
        log("      deleting the zip and the raw txt files")
        shutil.rmtree(RAW_DIR, ignore_errors=True)
        if os.path.isfile(zip_path):
            os.remove(zip_path)
    if args.blocks_only:
        log("      deleting scenes/ (only blocks_bs1_s1 and meta are read by the loader)")
        shutil.rmtree(os.path.dirname(scenes), ignore_errors=True)

    log(f"Done. Use --data_path {os.path.dirname(blocks)}")


if __name__ == "__main__":
    main()
