"""Image front-end of the LMA at the class level: class names -> E_CLIP [N+1, 512] from M images per class through the
frozen CLIP image encoder [PAPER Fig.1 "Image -> CLIP"] [DECISION D-47]. **The image source is not in the paper.**

Row c is the mean of the M unit embeddings of class c's images, renormalised; row 0 uses the images of the
background entry. The images are listed in a manifest (`assets/modality/images_s3dis.json`: URL, licence, author,
Wikimedia sha1, sha256) and fetched by `preprocess/fetch_modality_images.py`; a missing or altered image raises.
CLIP stays outside the module tree, as for text (03 §2.1).
"""

import hashlib
import json
import os
from typing import Callable, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

from models.clip_text import CLIP_DIM, DEFAULT_CLIP_VARIANT, PREPROCESS, load_clip

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MANIFEST = os.path.join(REPO, "assets", "modality", "images_s3dis.json")
DEFAULT_IMAGE_ROOT = os.path.join(REPO, "datasets", "modality", "images")
BACKGROUND_KEY = "background"  # the manifest's entry for row 0 [DECISION D-47]


def load_manifest(path: str = DEFAULT_MANIFEST) -> Dict[str, List[dict]]:
    """{class name or 'background': [entry, ...]}; each entry has title, url, sha1, sha256, licence, author, file."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"image manifest not found: {path} [DECISION D-47]")
    with open(path, encoding="utf-8") as f:
        return json.load(f)["classes"]


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def image_paths(manifest: Dict[str, List[dict]], name: str, root: str) -> List[str]:
    """The verified local files of `name`; raises on an unknown name, a missing file or a hash mismatch."""
    if name not in manifest:
        raise KeyError(f"no images for class {name!r} in the manifest [DECISION D-47]")
    paths = []
    for entry in manifest[name]:
        path = os.path.join(root, entry["file"])
        if not os.path.isfile(path):
            raise FileNotFoundError(f"missing image {path}; run preprocess/fetch_modality_images.py")
        if sha256_of(path) != entry["sha256"]:
            raise ValueError(f"image {path} does not match its manifest sha256")
        paths.append(path)
    return paths


def clip_encode_images(variant: str) -> Callable[[List[str], torch.device], torch.Tensor]:
    """encode(paths, device) -> raw CLIP image features [len(paths), 512], CLIP's own preprocessing."""

    def encode(paths: List[str], device: torch.device) -> torch.Tensor:
        from PIL import Image

        model = load_clip(variant, device)
        prep = PREPROCESS[variant]
        dev = next(model.parameters()).device
        batch = torch.stack([prep(Image.open(p).convert("RGB")) for p in paths]).to(dev)  # [M, 3, 224, 224]
        with torch.no_grad():
            return model.encode_image(batch)  # [M, 512]

    return encode


class ClipImageEmbedding:
    """Class names -> E_CLIP [N+1, 512], float32, unit rows, row 0 = background [DECISION D-47].

    `encode` replaces CLIP in CPU tests (05 §1 principle 3). Rows are cached per class name.
    """

    def __init__(self, variant: str = DEFAULT_CLIP_VARIANT, manifest: Optional[Dict[str, List[dict]]] = None,
                 root: str = DEFAULT_IMAGE_ROOT,
                 encode: Optional[Callable[[List[str], torch.device], torch.Tensor]] = None):
        self.variant = variant
        self.manifest = manifest if manifest is not None else load_manifest()
        self.root = root
        self.encode = encode if encode is not None else clip_encode_images(variant)
        self.cache: Dict[str, torch.Tensor] = {}  # class name -> [512] float32 on CPU

    def row(self, name: str, device: torch.device) -> torch.Tensor:
        if name not in self.cache:
            paths = image_paths(self.manifest, name, self.root)
            raw = self.encode(paths, device)  # [M, 512]
            if raw.shape != (len(paths), CLIP_DIM):
                raise ValueError(f"CLIP image features {tuple(raw.shape)} != {(len(paths), CLIP_DIM)}")
            unit = F.normalize(raw.float(), dim=-1)  # [M, 512], each image on the sphere
            self.cache[name] = F.normalize(unit.mean(dim=0), dim=0).cpu()  # [512], mean then renormalise
        return self.cache[name]

    def __call__(self, class_names: Sequence[str], device: torch.device) -> torch.Tensor:
        names = [BACKGROUND_KEY] + list(class_names)  # row order of P_point (02 §0)
        return torch.stack([self.row(n, device) for n in names]).to(device)  # [N+1, 512]
