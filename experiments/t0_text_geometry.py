"""T0: CPU probes of the text path before any text arm is designed (phase 16, beyond the paper).

    .venv/Scripts/python experiments/t0_text_geometry.py      # needs CLIP ViT-B/16 cached and P1's banks

A. Geometry of the CLIP prompts the LMA sees: the repo's template (`models/clip_text.py`), bare class
   names and a prompt ensemble; pairwise cosine, energy in the common direction, residual rank.
B. The GMMN term of Eq.7-8 (`loss/gmmn_loss.py`) on unit-norm prototypes (route B's L2), compared with
   sum(1/sigma^2) * ||mean difference||^2, and its sensitivity to swapping the two text rows.
C. Relational probe on P1's base-prototype banks (`results/phase16_p1/bank_*.pt`, CL2N geometry): does
   CLIP similarity between class names predict similarity between visual prototypes (Spearman over the
   15 pairs, permutation p), and how well does a text-weighted mixture of the other five prototypes
   predict the sixth (leave-one-out cosine)?
D. A ridge map text -> prototype fitted on five base classes, predicting the sixth.
Diagnostic only; writes nothing. Results: docs/research/2026-09-24_text_integration_independent.md.
"""

import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import clip  # noqa: E402

from loss.gmmn_loss import RBF_BANDWIDTHS, mmd  # noqa: E402
from models.clip_text import BACKGROUND_PROMPT, class_prompt  # noqa: E402

ENSEMBLE = ("a photo of a {}.", "a point cloud of a {}.", "a 3D scan of a {} in a room.", "the {} in an indoor scene.",
            "a {} in an office.", "a rendering of a {}.")
BANKS = ("vipseg_S1", "vipseg_S0", "ours_S1")


def upper(s: np.ndarray) -> np.ndarray:
    i, j = np.triu_indices(s.shape[0], 1)
    return s[i, j]


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a.argsort().argsort(), b.argsort().argsort())[0, 1])


def load_bank(name: str):
    b = torch.load(os.path.join(REPO, "results", "phase16_p1", f"bank_{name}_1000.pt"), map_location="cpu",
                   weights_only=False)
    info = b["info"] if isinstance(b["info"], dict) else json.loads(b["info"])
    return info["classes"], F.normalize(b["base"].float(), dim=-1)  # [6], [6, 128]


def main() -> int:
    names = [line.strip() for line in open(os.path.join(REPO, "datasets", "S3DIS", "meta", "s3dis_classnames.txt"))]
    model, _ = clip.load("ViT-B/16", device="cpu", jit=False)
    model.eval()

    def enc(prompts):
        with torch.no_grad():
            return F.normalize(model.encode_text(clip.tokenize(prompts)).float(), dim=-1)

    texts = {"template": enc([class_prompt(n) for n in names]), "bare": enc(names),
             "ensemble": F.normalize(torch.stack([enc([t.format(n) for t in ENSEMBLE]).mean(0) for n in names]), dim=-1)}
    bg = enc([BACKGROUND_PROMPT])

    # A. prompt geometry
    for tag, e in texts.items():
        s = e @ e.T
        off = s[~torch.eye(len(names), dtype=torch.bool)]
        mean_vec = e.mean(0)
        sv = torch.linalg.svdvals(e - mean_vec)
        print(f"A {tag:9s} off-diag cos mean {off.mean():.3f} min {off.min():.3f} max {off.max():.3f} | "
              f"common-direction energy {float(mean_vec.norm() ** 2):.3f} | residual eff. rank "
              f"{float(sv.sum() ** 2 / (sv ** 2).sum()):.1f}")
    print(f"A background prompt vs template class prompts: cos mean {float((bg @ texts['template'].T).mean()):.3f}")

    # B. GMMN on unit-norm prototypes
    torch.manual_seed(0)
    coef = sum(1.0 / s ** 2 for s in RBF_BANDWIDTHS)
    errs, swap = [], []
    for _ in range(200):
        pp = F.normalize(torch.rand(3, 128) + 0.5, dim=-1)  # post-ReLU-like unit rows
        pm = F.normalize(torch.rand(3, 128) + 0.5, dim=-1) * (0.5 + torch.rand(1))
        exact = float(mmd(pm[1:], pp[1:]))
        approx = coef * float((pm[1:].mean(0) - pp[1:].mean(0)).norm() ** 2)
        errs.append(abs(exact - approx) / max(exact, 1e-12))
        swap.append(abs(float(mmd(pm[[2, 1]], pp[1:])) - exact))
    print(f"B sum 1/sigma^2 = {coef:.4f}; |MMD - coef*||mean diff||^2| / MMD: median {np.median(errs):.3f}, "
          f"max {np.max(errs):.3f}; swapping the two text rows changes MMD by at most {max(swap):.2e}")

    # C. relational probe
    for bank in BANKS:
        cls, b = load_bank(bank)
        v = (b @ b.T).numpy()
        for tag, e in texts.items():
            st = (e[cls] @ e[cls].T).numpy()
            rho = spearman(upper(st), upper(v))
            rng = np.random.default_rng(0)
            null = [spearman(upper(st), upper(v[np.ix_(p, p)])) for p in (rng.permutation(6) for _ in range(2000))]
            loo = {}
            for tau in (0.0, 10.0, 30.0, 100.0):
                cs = []
                for j in range(6):
                    others = [k for k in range(6) if k != j]
                    w = torch.softmax(tau * torch.tensor(st[j, others]), dim=0).float()
                    cs.append(float(F.normalize((w[:, None] * b[others]).sum(0), dim=-1) @ b[j]))
                loo[tau] = np.mean(cs)
            nearest = np.mean([max(v[j, k] for k in range(6) if k != j) for j in range(6)])
            print(f"C {bank:10s} {tag:9s} Spearman(text, visual) {rho:+.2f} (perm p {np.mean(np.array(null) >= rho):.3f}) | "
                  f"LOO cos: uniform {loo[0.0]:.3f}, tau10 {loo[10.0]:.3f}, tau30 {loo[30.0]:.3f}, "
                  f"tau100 {loo[100.0]:.3f}, oracle nearest {nearest:.3f}")
        print("   classes", [names[c] for c in cls])

    # D. ridge map from five names
    for bank in BANKS[:2]:
        cls, b = load_bank(bank)
        b = b.double()
        for tag, e in texts.items():
            x = e[cls].double()
            out = []
            for lam in (0.01, 0.1, 1.0):
                cs = []
                for j in range(6):
                    o = [k for k in range(6) if k != j]
                    xo = x[o] - x[o].mean(0)  # the common template direction carries nothing
                    w = torch.linalg.solve(xo.T @ xo + lam * torch.eye(512, dtype=torch.float64), xo.T @ b[o])
                    pred = (x[j] - x[o].mean(0)) @ w + b[o].mean(0)
                    cs.append(float(F.normalize(pred, dim=-1) @ b[j]))
                out.append(f"lam {lam}: {np.mean(cs):.3f}")
            base = np.mean([float(F.normalize(b[[k for k in range(6) if k != j]].mean(0), dim=-1) @ b[j])
                            for j in range(6)])
            print(f"D {bank:10s} {tag:9s} " + ", ".join(out) + f" | mean-of-others baseline {base:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
