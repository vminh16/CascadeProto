"""D-48 debate, CPU: (a) P3's ridge coefficients a_c (novel text direction = sum_j a_cj * base prototype j),
(b) share of S1 test-class query blocks that are also training blocks, (c) extreme-value offset of a max over M."""
import os
import pickle
import sys

import numpy as np
import torch
import torch.nn.functional as F

REPO = r"C:\Users\USER\Desktop\CascadeProto"
sys.path.insert(0, REPO)
from models import text_prior as tp  # noqa: E402

NAMES = [l.strip() for l in open(os.path.join(REPO, "datasets", "S3DIS", "meta", "s3dis_classnames.txt"))][:12]
BASE = [0, 3, 4, 8, 10, 11]  # S1 base: ceiling beam column chair bookcase board
NOVEL = [6, 1, 9, 7, 2, 5]   # door floor sofa table wall window

import clip  # noqa: E402
from models.clip_text import DEFAULT_CLIP_VARIANT, load_clip  # noqa: E402
model = load_clip(DEFAULT_CLIP_VARIANT, "cpu")
def enc(prompts):
    with torch.no_grad():
        return model.encode_text(clip.tokenize(prompts)).float()

for kind in ("descriptions", "template"):
    e_all = tp.class_embeddings(NAMES, kind, enc)  # [12, 512]
    e = F.normalize(e_all - e_all.mean(0, keepdim=True), dim=-1).double()
    eb, en = e[BASE], e[NOVEL]
    k = eb @ eb.T + tp.RIDGE_LAMBDA * torch.eye(6, dtype=e.dtype)
    coef = torch.linalg.solve(k, eb @ en.T).T  # [6 novel, 6 base]
    print(f"\n== ridge coefficients a_c ({kind}), rows novel, cols {[NAMES[b] for b in BASE]}; cond(K) {torch.linalg.cond(k).item():.1f}")
    for i, c in enumerate(NOVEL):
        row = coef[i].numpy()
        print(f"{NAMES[c]:>7s} " + " ".join(f"{v:+6.2f}" for v in row) + f"   |a|_1 {np.abs(row).sum():.2f}  top {NAMES[BASE[int(np.argmax(row))]]}")
    raw = (e_all[NOVEL] @ e_all[BASE].T).numpy()
    print("raw CLIP cos novel x base (uncentred):")
    for i, c in enumerate(NOVEL):
        print(f"{NAMES[c]:>7s} " + " ".join(f"{v:6.3f}" for v in raw[i]))

# (b) block overlap between test-class query lists and training-class lists
c2s = pickle.load(open(os.path.join(REPO, "datasets", "S3DIS", "blocks_bs1_s1", "class2scans_100.pkl"), "rb"))
train_blocks = set().union(*[set(c2s[b]) for b in BASE])
for c in NOVEL:
    s = set(c2s[c]); print(f"{NAMES[c]:>7s}: {len(s)} query blocks, {len(s & train_blocks) / len(s):.3f} also in a training list")
allt = set().union(*[set(c2s[c]) for c in NOVEL])
print(f"all test-class blocks {len(allt)}, share also training blocks {len(allt & train_blocks) / len(allt):.3f}")

# (c) E[max of M iid N(0,1)] by simulation
rng = np.random.default_rng(0)
for m in (1, 3, 8, 16, 32):
    print(f"M {m:2d}: E[max] {rng.standard_normal((200000, m)).max(1).mean():.3f} sd")
