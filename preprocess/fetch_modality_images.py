"""Fetch and verify the class-level images of [DECISION D-47] (not in the paper).

Reads `assets/modality/images_s3dis.json` and downloads every original file from Wikimedia Commons into
`datasets/modality/images/<entry file>`. Each file is checked against the sha1 that Wikimedia publishes and against
the manifest's sha256; any mismatch raises. `--record_sha256` fills the sha256 of entries that have none (used
once, when the manifest is built) after the sha1 check has passed.

    python preprocess/fetch_modality_images.py [--manifest PATH] [--root DIR] [--record_sha256]
"""

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.clip_image import DEFAULT_IMAGE_ROOT, DEFAULT_MANIFEST, sha256_of  # noqa: E402

USER_AGENT = "CascadeProto-research/0.1 (https://github.com/vminh16/CascadeProto; D-47 image fetcher) python-urllib"


def sha1_of(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(path: str, entry: dict) -> None:
    """Raises unless the file matches the entry's Wikimedia sha1 and (when recorded) its sha256."""
    if sha1_of(path) != entry["sha1"]:
        raise ValueError(f"{path}: sha1 differs from Wikimedia's {entry['sha1']} ({entry['title']})")
    if entry.get("sha256") and sha256_of(path) != entry["sha256"]:
        raise ValueError(f"{path}: sha256 differs from the manifest ({entry['title']})")


def fetch(entry: dict, path: str, attempts: int = 6) -> None:
    """Download one original; on HTTP 429 wait as the server asks (Retry-After) or back off, then retry."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    req = urllib.request.Request(entry["url"], headers={"User-Agent": USER_AGENT})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            break
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == attempts - 1:
                raise
            wait = int(e.headers.get("Retry-After") or 0) or 30 * 2 ** attempt
            print(f"429 on {entry['title']}; waiting {wait}s", flush=True)
            time.sleep(wait)
    with open(path + ".tmp", "wb") as f:
        f.write(data)
    os.replace(path + ".tmp", path)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default=DEFAULT_MANIFEST)
    p.add_argument("--root", default=DEFAULT_IMAGE_ROOT)
    p.add_argument("--record_sha256", action="store_true")
    args = p.parse_args(argv)
    with open(args.manifest, encoding="utf-8") as f:
        manifest = json.load(f)
    recorded = 0
    for name, entries in manifest["classes"].items():
        for entry in entries:
            path = os.path.join(args.root, entry["file"])
            if not os.path.isfile(path):
                time.sleep(5.0)  # polite to the Wikimedia servers, which rate-limit bursts (HTTP 429)
                fetch(entry, path)
            verify(path, entry)
            if not entry.get("sha256"):
                if not args.record_sha256:
                    raise ValueError(f"{entry['title']}: no sha256 in the manifest; rerun with --record_sha256")
                entry["sha256"] = sha256_of(path)
                recorded += 1
        print(f"{name}: {len(entries)} images verified", flush=True)
    if recorded:
        with open(args.manifest, "w", encoding="utf-8", newline="\n") as f:
            json.dump(manifest, f, indent=1, ensure_ascii=False)
            f.write("\n")
        print(f"recorded {recorded} sha256 values in {args.manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
