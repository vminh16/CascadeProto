"""C2: which inputs VIP-Seg's cross-correlation attention depends on (CPU, no data, no checkpoint).

The inherited PEM computes its cross term after `reshape(72, -1)` on batched tensors
[VIPSEG models/vipseg.py:285-296]. This script reads the inherited `PrototypeEnhancementModule` source (the file is
parsed, never edited or imported, because importing `models.vipseg` needs pointnet2_ops), rebuilds the attention
matrix exactly as the module does, and perturbs one input at a time:

* only way 2's support block      -> does the attention of slot 1 (way 1) or slot 0 (background) change?
* only query 2                    -> does the attention of query 1 change?

A per-(query, slot) form (what D-01 implements) would change in neither case. The script also checks the index
formula of the research note 2026-09-24_condition_presence_gauge.md §1.4 against the module to machine precision.

    python experiments/c2_vipseg_crosscorr_check.py
"""

import ast
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P, D, PROJ = 2048, 128, 72


def load_pem():
    src = open(os.path.join(REPO, "models", "vipseg.py"), encoding="utf-8").read()
    ns = {"torch": torch, "nn": nn, "F": F}
    for node in ast.parse(src).body:
        if isinstance(node, ast.ClassDef) and node.name == "PrototypeEnhancementModule":
            exec(compile(ast.Module([node], []), "models/vipseg.py", "exec"), ns)
    return ns["PrototypeEnhancementModule"]()


def attention(pem, q, s):
    """VIP-Seg's cross attention [B, N+1, D, D], K = 1, as in models/vipseg.py:244-292."""
    b, n = q.shape[0], s.shape[0]
    sup = torch.cat([s.mean(0).unsqueeze(0), s], 0).reshape(-1, P, D)  # [N+1, P, D], slot 0 = mean of ways
    qq = pem.maxpool(q.transpose(1, 2)).transpose(1, 2)  # [B, 64, D]
    ss = pem.maxpool(sup.transpose(1, 2)).transpose(1, 2)  # [N+1, 64, D]
    que, su = pem.map(qq).reshape(PROJ, -1), pem.map(ss).reshape(PROJ, -1)  # [72, B*D], [72, (N+1)*D]
    cc = torch.matmul(que.transpose(0, 1) / D ** 0.5, su).reshape(b, D, n + 1, D).permute(0, 2, 1, 3)
    return F.softmax(cc, dim=-1), pem.map(qq), pem.map(ss)  # [B, N+1, D, D], Q' [B, 72, D], S' [N+1, 72, D]


def formula(qp, sp):
    """A[b', w'] from the index algebra: sum over r = 0..71 of Q'[r//36, 2(r%36)+b'] x S'[r//24, 3(r%24)+w']."""
    out = torch.zeros(2, 3, D, D, dtype=qp.dtype)
    for bb in range(2):
        for ww in range(3):
            for r in range(PROJ):
                out[bb, ww] += torch.outer(qp[r // 36, 2 * (r % 36) + bb], sp[r // 24, 3 * (r % 24) + ww])
    return F.softmax(out / D ** 0.5, dim=-1)


def main():
    torch.manual_seed(0)
    pem = load_pem().double().eval()
    with torch.no_grad():
        # the module's own weights are near-uniform at init; scale the projection so the softmax is not flat
        pem.map.weight.mul_(20.0)
        q = torch.rand(2, P, D, dtype=torch.double)
        s = torch.rand(2, 1, P, D, dtype=torch.double)
        base, qp, sp = attention(pem, q, s)
        err = (formula(qp, sp) - base).abs().max().item()
        s2 = s.clone(); s2[1] = torch.rand(1, P, D, dtype=torch.double)
        q2 = q.clone(); q2[1] = torch.rand(P, D, dtype=torch.double)
        a_s, _, _ = attention(pem, q, s2)
        a_q, _, _ = attention(pem, q2, s)
    lines = [
        f"index formula vs module, max |diff|: {err:.2e}",
        f"attention of (query 1, slot 1 = way 1) when only way 2's support changes, max |diff|: "
        f"{(a_s[0, 1] - base[0, 1]).abs().max().item():.3e}",
        f"attention of (query 1, slot 0 = background) when only way 2's support changes: "
        f"{(a_s[0, 0] - base[0, 0]).abs().max().item():.3e}",
        f"attention of query 1 (all slots) when only query 2 changes: {(a_q[0] - base[0]).abs().max().item():.3e}",
        f"typical attention entry (1/D): {1 / D:.3e}",
    ]
    print("\n".join(lines))
    return 0 if err < 1e-9 else 1


if __name__ == "__main__":
    sys.exit(main())
