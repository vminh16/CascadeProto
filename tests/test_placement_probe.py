"""P9-1..15 (05 §3.8u): the placement probe of [DECISION D-44], beyond the paper.

P9-1..14 run on the CPU (geometry replay, the cap, part A's and part B's pure functions, rules); P9-15 needs the GPU
encoder (marker `cuda`).
"""

import ast
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from experiments import p6_prototype_probe as p6
from experiments import p9_placement_probe as p9
from models.density_ops import BALL_K, ball_group

REPO = Path(__file__).resolve().parents[1]


def fps_brute(xyz, g):
    """pointnet2's FPS written as a plain loop over points [sampling_gpu.cu:85-110]."""
    x = np.asarray(xyz, dtype=np.float32)
    temp = [np.float32(1e10)] * len(x)
    out, old = [0], 0
    for _ in range(1, g):
        best, besti = np.float32(-1.0), 0
        for k in range(len(x)):
            if float((x[k] * x[k]).sum(dtype=np.float32)) <= 1e-3:
                continue
            d = np.float32(((x[k] - x[old]) ** 2).sum(dtype=np.float32))
            temp[k] = min(d, temp[k])
            if temp[k] > best:
                best, besti = temp[k], k
        out.append(besti)
        old = besti
    return np.array(out)


def cloud(n=300, seed=0, dup=40):
    rng = np.random.default_rng(seed)
    x = rng.random((n, 3)).astype(np.float32) * np.array([1.0, 1.0, 0.5], dtype=np.float32)
    x[n - dup:] = x[rng.integers(0, n - dup, dup)]  # duplicates, as the sampler draws a raw point twice
    return x


def test_p9_1_fps_replay_matches_the_kernel_loop():
    x = cloud()
    x[5] = [0.01, 0.01, 0.01]  # |x|² = 3e-4 <= 1e-3: never chosen
    got = p9.fps(x, 60)
    assert got.tolist() == fps_brute(x, 60).tolist()
    assert got[0] == 0 and 5 not in got.tolist() and len(set(got.tolist())) == 60
    with pytest.raises(ValueError, match="cannot choose"):
        p9.fps(x, 301)


def test_p9_2_distinct_count_by_hand():
    x = torch.from_numpy(cloud())
    centres = x[:20]
    idx = ball_group(x.unsqueeze(0), centres.unsqueeze(0), 0.2, BALL_K)  # [1, 20, K]
    m = p9.distinct_count(x.unsqueeze(0), idx)[0]
    for g in range(20):
        coords = {tuple(x[int(i)].tolist()) for i in idx[0, g]}
        assert int(m[g]) == len(coords)
    assert int(m.min()) >= 1 and int(m.max()) <= BALL_K


def test_p9_3_cap_keeps_the_first_distinct_points():
    x = torch.from_numpy(cloud(seed=1))
    idx = ball_group(x.unsqueeze(0), x[:30].unsqueeze(0), 0.3, BALL_K)
    m = p9.distinct_count(x.unsqueeze(0), idx)
    assert torch.equal(p9.cap_indices(x.unsqueeze(0), idx, torch.full_like(m, BALL_K)), idx)
    assert torch.equal(p9.cap_indices(x.unsqueeze(0), idx, m), idx)
    one = p9.cap_indices(x.unsqueeze(0), idx, torch.ones_like(m))
    assert torch.equal(one, idx[..., :1].expand_as(idx))
    cap = torch.randint(1, BALL_K + 1, m.shape, generator=torch.Generator().manual_seed(0))
    capped = p9.cap_indices(x.unsqueeze(0), idx, cap)
    assert torch.equal(p9.distinct_count(x.unsqueeze(0), capped), torch.minimum(m, cap))
    new = p9.distinct_mask(x.unsqueeze(0), idx)
    first_kept = (new.long().cumsum(-1) <= cap.unsqueeze(-1)) & new  # the first `cap` distinct entries stay in place
    assert torch.equal(capped[first_kept], idx[first_kept])
    with pytest.raises(ValueError, match="at least one"):
        p9.cap_indices(x.unsqueeze(0), idx, torch.zeros_like(m))


def test_p9_4_replay_follows_the_centres():
    rng = np.random.default_rng(2)
    xyz = rng.random((2048, 3)).astype(np.float32)
    labels = rng.integers(0, 3, 2048)
    out = p9.replay(xyz, labels)
    assert [len(m) for m, _ in out] == list(p9.GROUPS)
    c1 = p9.fps(xyz, p9.GROUPS[0])
    assert out[0][1].tolist() == labels[c1].tolist()
    c2 = p9.fps(xyz[c1], p9.GROUPS[1])
    assert out[1][1].tolist() == labels[c1][c2].tolist()
    x1 = torch.from_numpy(xyz[c1][c2])
    nb = ball_group(torch.from_numpy(xyz[c1]).unsqueeze(0), x1.unsqueeze(0), 0.2, BALL_K)
    assert out[1][0].tolist() == p9.distinct_count(torch.from_numpy(xyz[c1]).unsqueeze(0), nb)[0].tolist()
    assert all(1 <= m.min() and m.max() <= BALL_K for m, _ in out)


def test_p9_5_m_stats_by_hand():
    s = p9.m_stats([np.array([16, 8, 4]), np.array([16])])
    assert s == {"centres": 4, "mean": 11.0, "share_lt_k": 0.5, "share_le_half": 0.5}
    assert p9.m_stats([]) == {"centres": 0}


def test_p9_6_psi_by_hand():
    assert p9.psi(0.8, 0.5, 0.2) == pytest.approx(0.5)
    assert p9.psi(0.8, 0.8, 0.2) == 0.0 and np.isnan(p9.psi(0.5, 0.4, 0.5))
    ep = np.array([0, 0, 1, 2])
    gt = np.array([[10, 10, 10], [10, 10, 10], [20, 20, 20], [10, 10, 10]], float)
    tp = np.array([[8, 5, 2], [8, 5, 2], [16, 10, 4], [8, 5, 2]], float)
    s = p9.psi_summary(ep, gt, tp, boot=200)
    assert s["psi"] == pytest.approx(0.5) and s["r_ref"] == pytest.approx(0.8) and s["r_low"] == pytest.approx(0.2)
    assert s["psi_ci"][0] == pytest.approx(0.5) and s["psi_ci"][1] == pytest.approx(0.5) and s["events"] == 4
    tp2 = tp.copy()
    tp2[3, 1] = 2  # one episode reproduces the whole gap
    s2 = p9.psi_summary(ep, gt, tp2, boot=500)
    assert s2["psi"] == pytest.approx((0.8 - 22 / 50) / 0.6) and s2["psi_ci"][0] < s2["psi"] < s2["psi_ci"][1]


def test_p9_7_caps_only_on_the_target_class():
    rng = np.random.default_rng(0)
    lab = np.array([0, 2, 2, 1, 2])
    cap = p9.draw_caps(lab, 2, np.array([3, 5]), rng)
    assert cap[[0, 3]].tolist() == [BALL_K, BALL_K] and set(cap[[1, 2, 4]].tolist()) <= {3, 5}
    assert p9.draw_caps(lab, 2, np.array([], dtype=np.int64), rng).tolist() == [BALL_K] * 5


def test_p9_8_shared_positions_by_hand():
    a = np.array([7, 3, 9, 3, 1])
    b = np.array([1, 8, 7, 7, 4])
    pa, pb = p9.shared_positions(a, b)
    assert sorted(zip(a[pa].tolist(), pa.tolist(), pb.tolist())) == [(1, 4, 0), (7, 0, 2)]


def test_p9_9_slices_cover_the_encoder_output():
    lo = [s for s, _ in p9.SLICES.values()]
    hi = [e for _, e in p9.SLICES.values()]
    assert lo[0] == 0 and hi[-1] == 900 and lo[1:] == hi[:-1] and hi[0] - lo[0] == 60
    g = torch.Generator().manual_seed(0)
    enc, f = torch.randn(900, 7, generator=g), torch.randn(7, 128, generator=g)
    cos = p9.slice_cosines(enc, enc, f, f)
    assert set(cos) == set(p9.SLICES) | {"feature"} and all(torch.allclose(v, torch.ones(7)) for v in cos.values())
    cos2 = p9.slice_cosines(enc, -enc, f, f)
    assert torch.allclose(cos2["stage2"], -torch.ones(7))


def test_p9_10_lda_forms_by_hand():
    g = torch.Generator().manual_seed(1)
    d, p = 6, 50
    u = torch.randn(p, d, generator=g, dtype=torch.float64)
    means = torch.randn(3, d, generator=g, dtype=torch.float64)
    a = torch.randn(d, d, generator=g, dtype=torch.float64)
    cov = a @ a.T + 0.1 * torch.eye(d, dtype=torch.float64)
    inv = torch.linalg.inv(cov)
    ref = u @ inv @ means.T - 0.5 * torch.einsum("cd,de,ce->c", means, inv, means)
    assert torch.allclose(p9.lda_logits(u, means, cov), ref, atol=1e-10)
    c = u.mean(0)
    ref2 = (u - c) @ inv @ means.T - 0.5 * torch.einsum("cd,de,ce->c", means - c, inv, means - c)
    assert torch.allclose(p9.lda_logits(u, means, cov, centre=c), ref2, atol=1e-10)
    lam = 0.3
    s = p9.shrink(cov, lam)
    assert torch.allclose(s, 0.7 * cov + 0.3 * torch.trace(cov) / d * torch.eye(d, dtype=torch.float64))
    labels = torch.randint(0, 3, (p,), generator=g)
    labels[:3] = torch.tensor([0, 1, 2])
    m, pres = p9.class_means(u, labels, 4)
    assert pres.tolist() == [True, True, True, False]
    assert torch.allclose(m[1], u[labels == 1].mean(0))
    w = p9.within_cov(u, labels, m)
    ref_w = sum((u[i] - m[labels[i]]).outer(u[i] - m[labels[i]]) for i in range(p)) / p
    assert torch.allclose(w, ref_w, atol=1e-12)
    mu, t = p9.total_cov(u)
    assert torch.allclose(t, torch.cov(u.T, correction=0), atol=1e-12) and torch.allclose(mu, u.mean(0))


def test_p9_11_lda_oracle_uses_the_within_class_metric():
    """Two classes separated along a quiet axis, with a large common spread on another: the LDA oracle separates
    them, the cosine rule on the means does not."""
    g = torch.Generator().manual_seed(2)
    n = 400
    y = torch.arange(n) % 2
    u = torch.zeros(n, 3, dtype=torch.float64)
    u[:, 0] = 1.0
    u[:, 1] = 3.0 * torch.randn(n, generator=g, dtype=torch.float64)
    u[:, 2] = 0.2 * (2 * y - 1) + 0.02 * torch.randn(n, generator=g, dtype=torch.float64)
    m, _ = p9.class_means(u, y, 2)
    acc_lda = (p9.lda_logits(u, m, p9.shrink(p9.within_cov(u, y, m), 0.1)).argmax(-1) == y).double().mean()
    acc_cos = ((u @ F.normalize(m, dim=-1).T).argmax(-1) == y).double().mean()
    assert acc_lda > 0.95 and acc_cos < 0.8


def test_p9_12_subspaces_heads_and_collapse():
    g = torch.Generator().manual_seed(3)
    basis = p9.orthonormal(torch.randn(8, 8, generator=g, dtype=torch.float64))
    assert torch.allclose(basis.T @ basis, torch.eye(8, dtype=torch.float64), atol=1e-12)
    b2 = basis[:, :2]
    v = torch.randn(5, 8, generator=g, dtype=torch.float64)
    inside = v @ b2 @ b2.T
    assert torch.allclose(p9.subspace_share(inside, b2), torch.ones(5, dtype=torch.float64))
    assert torch.allclose(p9.subspace_share(v - inside, b2), torch.zeros(5, dtype=torch.float64), atol=1e-12)
    sh = p9.head_shares(v, basis, 4)
    assert sh.shape == (5, 4) and torch.allclose(sh.sum(-1), torch.ones(5, dtype=torch.float64))
    assert torch.allclose(sh[:, 0], p9.subspace_share(v, basis[:, :2]))
    with pytest.raises(ValueError, match="do not split"):
        p9.head_shares(v, basis, 3)
    e = np.array([[0.5, 0.5], [0.5, 0.5]])
    gg = np.array([[0.2, 0.8], [0.4, 0.6]])  # ratios 0.6, 1.4 -> mean 1.0, std 0.4
    assert p9.ratio_cv(e, gg) == pytest.approx(0.4)
    assert p9.participation_ratio(torch.eye(8)) == pytest.approx(8.0)
    assert p9.participation_ratio(torch.outer(v[0], v[0])) == pytest.approx(1.0)
    f_q = torch.randn(2, 10, 8, generator=g, dtype=torch.float64)
    rows = F.normalize(torch.randn(2, 3, 8, generator=g, dtype=torch.float64), dim=-1)
    lg = p9.projected_logits(f_q, rows, b2)
    perp = lambda x: F.normalize(x - x @ b2 @ b2.T, dim=-1)  # noqa: E731
    assert torch.allclose(lg, torch.einsum("bpd,bcd->bpc", perp(f_q), perp(rows)))
    assert torch.allclose(p9.projected_logits(f_q, rows + rows @ b2 @ b2.T, b2), lg)  # the subspace is ignored


def test_p9_13_support_side_and_retrieval_by_hand():
    g = torch.Generator().manual_seed(4)
    f_s = torch.rand(2, 1, 40, 6, generator=g, dtype=torch.float64)
    sy = (torch.rand(2, 1, 40, generator=g) < 0.4).long()
    u = F.normalize(f_s, dim=-1)
    m = p9.support_means(f_s, sy)
    assert torch.allclose(m[0], u[sy == 0].mean(0)) and torch.allclose(m[2], u[1][sy[1] == 1].mean(0))
    f_q = torch.rand(2, 30, 6, generator=g, dtype=torch.float64)
    rows = p6.base_rows(f_q, f_s, sy)
    k1 = p9.component_logits(f_q, rows, f_s, sy, 1)  # one component is the way's direction: U itself
    assert torch.allclose(k1, p6.rule_logits(f_q, rows), atol=1e-12)
    k2 = p9.component_logits(f_q, rows, f_s, sy, 2)
    cent = p6.spherical_kmeans(F.normalize(f_s[1][sy[1] == 1], dim=-1), 2)
    assert torch.allclose(k2[..., 2], (f_q @ cent.T).max(-1).values) and torch.equal(k2[..., 0], k1[..., 0])
    uq = F.normalize(torch.randn(4, 6, generator=g, dtype=torch.float64), dim=-1)
    us = F.normalize(torch.randn(20, 6, generator=g, dtype=torch.float64), dim=-1)
    ys, yq = torch.randint(0, 3, (20,), generator=g), torch.tensor([0, 1, 2, 1])
    pur = p9.retrieval_purity(uq, yq, us, ys, k=5)
    for i in range(4):
        nn_idx = (us @ uq[i]).argsort(descending=True)[:5]
        assert float(pur[i]) == pytest.approx(float((ys[nn_idx] == yq[i]).double().mean()))
    labels = torch.randint(0, 3, (2, 30), generator=g)
    labels[1][labels[1] == 2] = 1
    dd, ds = p9.shift_vectors(f_q, labels, rows)
    assert dd.shape == (3, 6)  # block 0: classes 1, 2; block 1: class 1
    o = lambda b, c: F.normalize(F.normalize(f_q[b][labels[b] == c], dim=-1).sum(0), dim=0)  # noqa: E731
    assert torch.allclose(ds[2], o(1, 1) - o(1, 0))
    assert torch.allclose(dd[1], (o(0, 2) - rows[0, 2]) - (o(0, 0) - rows[0, 0]))


def modules_result(**over):
    miou = {"U": 0.56, "cos_oracle": 0.80, "model": 0.55}
    miou.update({f"lda_oracle_{lam}": 0.80 for lam in p9.LAMBDAS})
    miou.update({f"lda_free_{lam}": 0.56 for lam in p9.LAMBDAS})
    miou.update({f"proj_{r}": 0.56 for r in p9.RANKS})
    miou.update({f"km_{k}": 0.56 for k in p9.KS})
    miou.update(over.pop("miou", {}))
    out = {"miou": miou,
           "nuisance": {str(r): {"kappa_median": 0.1, "rho_median": 0.2} for r in p9.RANKS},
           "heads": {"pca": {str(h): {"cv": 0.1} for h in p9.HEADS}},
           "collapse": {"span_share_median": 0.8, "participation_ratio": 10.0},
           "purity": {"hit": 0.9, "miss": 0.3}}
    for k, v in over.items():
        out[k] = v
    return out


def geo(means):
    return {"stages": {str(s): {"c_V0": {"mean": m}} for s, m in enumerate(means)}}  # JSON keys


def trace_result(other, own, single=None):
    single = single or {}
    caps = {d: {a: {"psi": single.get((d, a), 0.0)} for a in p9.CAP_ARMS} for d in p9.DIRECTIONS}
    caps["other"]["all"]["psi"], caps["own"]["all"]["psi"] = other, own
    return {"caps": caps}


def test_p9_14_rules():
    assert "full" in p9.decide(geo([15.6, 16, 16]), None, None)[0][0]
    assert "sparse" in p9.decide(geo([10.3, 15.4, 16]), None, None)[0][0]
    assert "sparse" in p9.decide(geo([15.6, 15.4, 16]), None, None)[0][0]  # every stage must be full
    v = dict(p9.decide(None, trace_result(0.6, 0.1, {("other", "s1"): 0.55}), None))
    key = [k for k in v if k.startswith("D44.1")][0]
    assert "carries density" in key and "['s1']" in v[key]
    v = dict(p9.decide(None, trace_result(0.1, 0.52), None))
    key = [k for k in v if k.startswith("D44.1")][0]
    assert "carries density" in key and "['s1', 's2', 's3']" in v[key]  # no single stage reaches 0.5
    assert any("not the pathway" in k for k in dict(p9.decide(None, trace_result(0.2, 0.15), None)))
    assert any("undecided" in k for k in dict(p9.decide(None, trace_result(0.3, 0.1), None)))
    assert any("D44.1 not run" in k for k in dict(p9.decide(None, {"refs": {}}, None)))
    names = [n for n, _ in p9.decide(None, None, modules_result())]
    assert all("fails" in n for n in names if n.startswith("D44.2"))
    ok = modules_result(miou={"lda_oracle_0.3": 0.811, "proj_8": 0.566, "km_3": 0.566},
                        nuisance={str(r): {"kappa_median": 0.3, "rho_median": 0.2} for r in p9.RANKS},
                        heads={"pca": {"4": {"cv": 0.1}, "8": {"cv": 0.5}}},
                        collapse={"span_share_median": 0.5, "participation_ratio": 10.0},
                        purity={"hit": 0.9, "miss": 0.5})
    names = [n for n, _ in p9.decide(None, None, ok)]
    assert all("admissible" in n for n in names if n.startswith("D44.2")) and len(names) == 7
    edge = modules_result(miou={"lda_oracle_0.1": 0.8099, "lda_free_0.5": 0.5649, "proj_4": 0.5649, "km_2": 0.5649})
    edge["nuisance"]["4"] = {"kappa_median": 0.3, "rho_median": 0.2}
    assert all("fails" in n for n, _ in p9.decide(None, None, edge) if n.startswith("D44.2"))
    free = modules_result(miou={"lda_free_0.1": 0.5655})
    assert "P9.3 metric admissible" in [n for n, _ in p9.decide(None, None, free)][0]
    rho = modules_result(miou={"proj_16": 0.57})  # gain holds, kappa <= rho
    assert any("P9.4 nuisance fails" in n for n, _ in p9.decide(None, None, rho))
    assert p9.decide(None, None, None) == [("D44.3", "P9 decides admissibility only; the run is chosen in a new decision")]


def test_p9_14b_files_parse_as_python_310():
    for f in ("experiments/p9_placement_probe.py", "tests/test_placement_probe.py"):
        ast.parse((REPO / f).read_text(encoding="utf-8"), feature_version=(3, 10))


# ------------------------------------------------------------------ GPU (gate G2)

@pytest.mark.cuda
def test_p9_15_capped_grouping_fps_replay_and_slices_on_the_gpu():
    from pointnet2_ops_lib.pointnet2_ops import pointnet2_utils

    from models.density_encoder import DensityEncoder, MetricBallGrouping
    from models.vipseg_backbone import ENCODER_CONFIG, persist_fixed_projections

    g = torch.Generator().manual_seed(5)
    xyz = (torch.rand(2, 2048, 3, generator=g) * torch.tensor([1.0, 1.0, 2.5])).cuda()
    xyz[0, 100:140] = xyz[0, 0:40]  # duplicated samples
    x, rgb = torch.randn(2, 2048, 60, generator=g).cuda(), torch.rand(2, 2048, 3, generator=g).cuda()
    base = MetricBallGrouping(1024, BALL_K, 0.1)
    capped = p9.capped_grouping_class()(base, 0)
    with torch.no_grad():
        for a, b in zip(base(xyz, x, rgb), capped(xyz, x, rgb)):
            assert torch.equal(a, b)
        replay = p9.fps(xyz[0].cpu().numpy(), 1024)
        cuda = pointnet2_utils.furthest_point_sample(xyz[:1].contiguous(), 1024).long()[0].cpu().numpy()
        assert (replay == cuda).mean() > 0.99, f"replay agrees on {(replay == cuda).mean():.4f} of the centres"
        labels = torch.randint(0, 3, (2, 2048), generator=g).cuda()
        capped.plan = p9.Plan(labels, {0: lambda lab: torch.where(lab == 1, torch.ones_like(lab),
                                                                  torch.full_like(lab, BALL_K))})
        out = capped(xyz, x, rgb)
        m = capped.plan.m[0]
        c_lab = capped.plan.centre_labels[0]
        assert bool((m[c_lab == 1] == 1).all()) and bool((m[c_lab != 1] > 1).any())
        assert torch.equal(out[0], base(xyz, x, rgb)[0])  # the centres do not move
        torch.manual_seed(0)
        enc = DensityEncoder(**ENCODER_CONFIG)
        persist_fixed_projections(enc)
        enc = enc.cuda().eval()
        pts = torch.cat([xyz[:1], rgb[:1], xyz[:1] / xyz[:1].max(1, keepdim=True).values], dim=-1)  # [1, 2048, 9]
        out = enc(pts)  # [1, 900, 2048]
        emb = enc.EncNP.Embedding_layer(pts[:, :, 0:3].permute(0, 2, 1))  # [1, 60, 2048]
        lo, hi = p9.SLICES["emb"]
        assert torch.equal(out[:, lo:hi], emb)
