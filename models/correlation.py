"""Blocks [3]-[4] of [DECISION D-48]: adaptive support cells, their correlation descriptor, and the descriptor spaces
of amendment 2. **Beyond the paper.**

Cells (fix F11): a set of n unit support features of one row (a way's masked points over its K shots, or every
mask-0 point) is split into M = clamp(floor(n / MIN_CELL), M_LO, M_HI) cells by spherical k-means from farthest-point
seeds, so every cell holds about MIN_CELL points or more except at the floor M_LO. A cell prototype is the normalised
member sum.

Descriptor: for a query point and a row with cells p_1..p_M, cos_m = <z_x, p_m> in the chosen space, and
d = (max_m cos_m, mean of the top 2, mean_m cos_m), defined for any M >= 2 and invariant to the order of the cells.

Spaces (fixes F12, F13, F20), fitted on base-class training features, all compared by cosine:
    raw       z = u
    centred   z = n(u - mu)                                   (unit features are non-negative, ||mean||^2 ~ 0.56)
    proj_r    z = n(U_r^T (u - mu))                            top-r eigenvectors of the base covariance
    white_r   z = n((Lambda_r + delta I)^-1/2 U_r^T (u - mu))  truncated whitening; the full inverse is never formed
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

MIN_CELL, M_LO, M_HI = 32, 2, 16  # [DECISION D-48 amendment 2]
KMEANS_ITERS = 5
WHITEN_DELTA = 1e-3  # relative to the largest kept eigenvalue
SPACES = ("raw", "centred", "proj6", "proj8", "white6", "white8")


def cell_count(n: int, cap: int = M_HI, min_cell: int = MIN_CELL, floor: int = M_LO) -> int:
    """M = clamp(floor(n / min_cell), floor, cap); n must reach the floor."""
    if n < floor:
        raise ValueError(f"{n} points cannot form {floor} cells")
    return max(floor, min(cap, n // min_cell))


def farthest_seeds(u: torch.Tensor, k: int) -> torch.Tensor:
    """k indices [k] of unit vectors u [n, D]: the point least similar to the mean direction, then repeatedly the
    point whose best cosine to the chosen seeds is lowest. Deterministic (ties to the lowest index)."""
    centre = F.normalize(u.sum(dim=0), dim=-1)  # [D]
    idx = [int((u @ centre).argmin())]
    best = u @ u[idx[0]]  # [n]
    for _ in range(1, k):
        nxt = int(best.argmin())
        idx.append(nxt)
        best = torch.maximum(best, u @ u[nxt])
    return torch.tensor(idx, device=u.device)


def cells(u: torch.Tensor, k: int, iters: int = KMEANS_ITERS) -> Tuple[torch.Tensor, torch.Tensor]:
    """(prototypes [k, D], sizes [k]) of unit vectors u [n, D]: spherical k-means from farthest seeds, max-cosine
    assignment; a cell that empties keeps its prototype with size 0."""
    if u.shape[0] < k:
        raise ValueError(f"{u.shape[0]} points cannot give {k} cells")
    c = u[farthest_seeds(u, k)]  # [k, D]
    sizes = torch.zeros(k, dtype=u.dtype, device=u.device)
    for _ in range(iters + 1):
        assign = (u @ c.T).argmax(dim=1)  # [n]
        members = F.one_hot(assign, k).to(u.dtype)  # [n, k]
        sizes = members.sum(dim=0)  # [k]
        c = torch.where((sizes > 0).unsqueeze(-1), F.normalize(members.T @ u, dim=-1), c)  # [k, D]
    return c, sizes


def row_cells(z_s: torch.Tensor, support_y: torch.Tensor, cap_fg: int = M_HI,
              cap_bg: int = M_HI) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """[(prototypes, sizes)] per row: background (every mask-0 point of every way), then each way over its K shots.

    z_s [N, K, P, E] support vectors in the chosen space (unit), support_y [N, K, P] binary.
    """
    out = [cells(z_s[support_y == 0], cell_count(int((support_y == 0).sum()), cap_bg))]
    for w in range(z_s.shape[0]):
        pts = z_s[w][support_y[w] == 1]  # [n_w, E]
        out.append(cells(pts, cell_count(pts.shape[0], cap_fg)))
    return out


def descriptor(cos: torch.Tensor) -> torch.Tensor:
    """d [..., 3] = (max, mean of the top 2, mean) over the last axis cos [..., M], M >= 2."""
    if cos.shape[-1] < 2:
        raise ValueError(f"the descriptor needs at least 2 cells, got {cos.shape[-1]}")
    top2 = cos.topk(2, dim=-1).values  # [..., 2]
    return torch.stack([top2[..., 0], top2.mean(dim=-1), cos.mean(dim=-1)], dim=-1)


def row_descriptors(z_q: torch.Tensor, rows: List[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
    """D [B_q, P, N+1, 3] for query vectors z_q [B_q, P, E] (unit); cells that emptied are skipped."""
    out = []
    for proto, sizes in rows:
        keep = proto[sizes > 0]  # [M', E]
        out.append(descriptor(torch.einsum("bpe,me->bpm", z_q, keep)))  # [B_q, P, 3]
    return torch.stack(out, dim=2)  # [B_q, P, N+1, 3]


def max_rule_logits(desc: torch.Tensor) -> torch.Tensor:
    """P11.3's rule [B_q, P, N+1]: each row scored by its best cell."""
    return desc[..., 0]


class Space:
    """A descriptor space of amendment 2, fitted once on base-class training features."""

    def __init__(self, kind: str, mu: Optional[torch.Tensor] = None, cov: Optional[torch.Tensor] = None):
        if kind not in SPACES:
            raise ValueError(f"space must be one of {SPACES}, got {kind!r}")
        self.kind = kind
        self.mu, self.proj, self.info = None, None, {}
        if kind == "raw":
            return
        if mu is None or (kind != "centred" and cov is None):
            raise ValueError(f"space {kind} needs the base mean (and covariance)")
        self.mu = mu
        if kind == "centred":
            return
        r = int(kind[-1])
        evals, evecs = torch.linalg.eigh(cov.double())  # ascending
        lam = evals.flip(0)[:r].clamp_min(0.0)  # [r]
        u_r = evecs.flip(1)[:, :r]  # [D, r]
        self.info = {"r": r, "energy_share": float(lam.sum() / evals.clamp_min(0).sum()),
                     "eigenvalues": lam.tolist()}
        if kind.startswith("white"):
            scale = (lam + WHITEN_DELTA * lam[0]).rsqrt()  # [r]
            self.proj = (u_r * scale).to(mu.dtype)  # [D, r]
        else:
            self.proj = u_r.to(mu.dtype)  # [D, r]

    def __call__(self, u: torch.Tensor) -> torch.Tensor:
        """Unit vectors [..., E] of unit features u [..., D]."""
        if self.kind == "raw":
            return u
        x = u - self.mu.to(u.device, u.dtype)  # [..., D]
        if self.proj is not None:
            x = torch.einsum("...d,dr->...r", x, self.proj.to(u.device, u.dtype))  # [..., r]
        return F.normalize(x, dim=-1)


def base_moments(u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """(mean [D], covariance [D, D]) of unit features u [M, D], float64."""
    x = u.double()
    mu = x.mean(dim=0)  # [D]
    r = x - mu  # [M, D]
    return mu, r.T @ r / max(x.shape[0] - 1, 1)


def contrast_share(space: Space, o_fg: torch.Tensor, o_bg: torch.Tensor) -> float:
    """Share of the oracle contrast energy ||P(o_c - o_bg)||^2 / ||o_c - o_bg||^2 kept by a projection space."""
    if space.proj is None or not space.kind.startswith("proj"):
        return 1.0
    d = (o_fg - o_bg).double()  # [..., D]
    q, _ = torch.linalg.qr(space.proj.double().to(d.device))  # [D, r] orthonormal basis of the span
    kept = torch.einsum("...d,dr->...r", d, q)  # [..., r]
    return float((kept ** 2).sum() / (d ** 2).sum().clamp_min(1e-30))


def spaces_from(mu: torch.Tensor, cov: torch.Tensor) -> Dict[str, Space]:
    return {k: Space(k, mu, cov) for k in SPACES}


class DescriptorProbe(torch.nn.Module):
    """P11.4's transfer probe: one small MLP scores every foreground row's descriptor, another the background row's.

    Shared weights across the ways make it equivariant to their order (amendment 1, change 8); only the row type
    (background or foreground) is distinguished.
    """

    def __init__(self, width: int = 16):
        super().__init__()
        mlp = lambda: torch.nn.Sequential(torch.nn.Linear(3, width), torch.nn.ReLU(),  # noqa: E731
                                          torch.nn.Linear(width, 1))
        self.fg, self.bg = mlp(), mlp()
        # input standardisation, set from the training descriptors: cosines sit in a narrow band (F12)
        self.register_buffer("shift", torch.zeros(3))
        self.register_buffer("scale", torch.ones(3))

    def forward(self, desc: torch.Tensor) -> torch.Tensor:
        """desc [..., N+1, 3] -> logits [..., N+1]."""
        x = (desc - self.shift) / self.scale  # [..., N+1, 3], the same map for every row
        return torch.cat([self.bg(x[..., :1, :]), self.fg(x[..., 1:, :])], dim=-2).squeeze(-1)


def train_probe(desc: torch.Tensor, y: torch.Tensor, epochs: int, lr: float = 1e-2, batch: int = 8192,
                seed: int = 0) -> DescriptorProbe:
    """CE of the probe on (descriptors [M, N+1, 3], episode labels [M]); deterministic for a seed."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    torch.manual_seed(seed)
    probe = DescriptorProbe().to(desc.device, desc.dtype)
    flat = desc.reshape(-1, desc.shape[-1])  # [M * (N+1), 3], every row's descriptor (the last axis is kept)
    probe.shift.copy_(flat.mean(dim=0))
    probe.scale.copy_(flat.std(dim=0).clamp_min(1e-6))
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    for _ in range(epochs):
        order = torch.randperm(desc.shape[0], generator=g).to(desc.device)  # [M]
        for i in range(0, desc.shape[0], batch):
            idx = order[i:i + batch]
            loss = F.cross_entropy(probe(desc[idx]), y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return probe.eval()
