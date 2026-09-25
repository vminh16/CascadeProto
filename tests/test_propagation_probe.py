"""P7-1..10 (05 §3.8r): the label propagation probe of [DECISION D-40], beyond the paper. CPU, float64, synthetic."""

import ast
import itertools
from collections import deque
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from experiments import d39_eval as d39
from experiments import p6_prototype_probe as p6
from experiments import p7_propagation_probe as p7

REPO = Path(__file__).resolve().parents[1]
ATOL = 1e-12


def block(bq=2, p=40, d=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    xyz = torch.rand(bq, p, 3, generator=g, dtype=torch.float64)
    u = F.normalize(torch.randn(bq, p, d, generator=g, dtype=torch.float64), dim=-1)
    return xyz, u


def episode(bq=2, p=64, d=8, n=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    f_q = torch.rand(bq, p, d, generator=g, dtype=torch.float64)
    f_s = torch.rand(n, 1, p, d, generator=g, dtype=torch.float64)
    support_y = (torch.rand(n, 1, p, generator=g) < 0.3).long()
    xyz = torch.rand(bq, p, 3, generator=g, dtype=torch.float64)
    labels = torch.randint(0, n + 1, (bq, p), generator=g)
    return f_q, f_s, support_y, xyz, labels


@pytest.mark.parametrize("graph", p7.GRAPHS)
def test_p7_1_knn_graph_by_brute_force(graph):
    xyz, u = block()
    k = 4
    idx, a = p7.knn_graph(xyz, u, graph, k)
    assert idx.shape == a.shape == (2, 40, k)
    for b, i in itertools.product(range(2), range(40)):
        if graph == "feat":
            key = sorted((-(u[b, i] @ u[b, j]).item(), j) for j in range(40) if j != i)
        else:
            key = sorted((((xyz[b, i] - xyz[b, j]) ** 2).sum().item(), j) for j in range(40) if j != i)
        assert set(idx[b, i].tolist()) == {j for _, j in key[:k]}
        for m, j in enumerate(idx[b, i].tolist()):
            want = 1.0 if graph == "xyz" else max(0.0, (u[b, i] @ u[b, j]).item()) ** 3
            assert abs(a[b, i, m].item() - want) < ATOL
    if graph == "xyzf":
        assert (a == 0).any() and (a > 0).any()  # the positive part matters on this block
    with pytest.raises(ValueError, match="graph"):
        p7.knn_graph(xyz, u, "rgb", k)
    with pytest.raises(ValueError, match="k ="):
        p7.knn_graph(xyz, u, graph, 40)


def test_p7_2_affinity_and_normalisation():
    xyz, u = block()
    idx, a = p7.knn_graph(xyz, u, "xyzf", 4)
    w = p7.affinity(idx, a, 40)
    ref = torch.zeros(2, 40, 40, dtype=torch.float64)
    for b, i, m in itertools.product(range(2), range(40), range(4)):
        ref[b, i, idx[b, i, m]] = a[b, i, m]
    assert torch.allclose(w, ref + ref.transpose(1, 2), atol=ATOL)
    assert torch.equal(w, w.transpose(1, 2)) and (w.diagonal(dim1=1, dim2=2) == 0).all()
    w[0, 3, :] = 0.0
    w[0, :, 3] = 0.0  # an isolated point
    s = p7.normalized(w)
    deg = w.sum(-1)
    for b, i, j in itertools.product(range(2), range(40), range(40)):
        want = w[b, i, j] / (deg[b, i] * deg[b, j]).sqrt() if deg[b, i] > 0 and deg[b, j] > 0 else 0.0
        assert abs(s[b, i, j].item() - float(want)) < ATOL
    assert torch.isfinite(s).all() and (s[0, 3] == 0).all() and (s[0, :, 3] == 0).all()


@pytest.mark.parametrize("beta", p7.BETAS)
def test_p7_3_spreading_is_zhous_fixed_point(beta):
    xyz, u = block()
    idx, a = p7.knn_graph(xyz, u, "xyz", 4)
    s = p7.normalized(p7.affinity(idx, a, 40))
    seeds = torch.randint(0, 3, (2, 40), generator=torch.Generator().manual_seed(1))
    y0 = F.one_hot(seeds, 3).double()
    z = p7.spread(s, y0, beta)
    f = y0.clone()
    for _ in range(6000):  # beta^6000 < 1e-26
        f = beta * s @ f + (1.0 - beta) * y0
    assert torch.allclose(z, f, atol=1e-10)
    assert torch.allclose(p7.spread(s, y0, 0.0), y0, atol=ATOL)
    with pytest.raises(ValueError, match="beta"):
        p7.spread(s, y0, 1.0)


def test_p7_4_isolated_points_keep_their_seed():
    w = torch.zeros(1, 5, 5, dtype=torch.float64)
    for i, j in ((0, 1), (1, 2), (2, 3)):
        w[0, i, j] = w[0, j, i] = 1.0  # point 4 has no edge
    y0 = F.one_hot(torch.tensor([[0, 0, 0, 0, 2]]), 3).double()
    z = p7.spread(p7.normalized(w), y0, 0.99)
    assert torch.allclose(z[0, 4], 0.01 * y0[0, 4], atol=ATOL) and int(z[0, 4].argmax()) == 2


def reference_stats(w, y, seed, n_cls):
    """Loop form of point_stats (one block)."""
    out = {k: [] for k in ("p", "a", "has_a", "b", "has_b", "fixable", "breakable")}
    n = len(w)
    for i in range(n):
        deg = sum(w[i][j] for j in range(n))
        same = sum(w[i][j] for j in range(n) if y[j] == y[i])
        other = sum(w[i][j] for j in range(n) if y[j] != y[i])
        same_hit = sum(w[i][j] for j in range(n) if y[j] == y[i] and seed[j] == y[i])
        other_pred = sum(w[i][j] for j in range(n) if y[j] != y[i] and seed[j] == y[i])
        votes = [sum(w[i][j] for j in range(n) if seed[j] == c) for c in range(n_cls)]
        rival = max(votes[c] for c in range(n_cls) if c != y[i])
        hit = seed[i] == y[i]
        out["p"].append(same / deg if deg > 0 else 0.0)
        out["has_a"].append(same > 0), out["a"].append(same_hit / same if same > 0 else 0.0)
        out["has_b"].append(other > 0), out["b"].append(other_pred / other if other > 0 else 0.0)
        out["fixable"].append(not hit and deg > 0 and votes[y[i]] > rival)
        out["breakable"].append(hit and deg > 0 and rival > votes[y[i]])
    return out


def bfs_reachable(edges, y, hit):
    n = len(y)
    adj = [[] for _ in range(n)]
    for i, j in edges:
        if y[i] == y[j]:
            adj[i].append(j), adj[j].append(i)
    out = []
    for s in range(n):
        seen, todo = {s}, deque([s])
        while todo:
            v = todo.popleft()
            for x in adj[v]:
                if x not in seen:
                    seen.add(x), todo.append(x)
        out.append(any(hit[v] for v in seen))
    return out


def test_p7_5_diagnostics_by_hand_and_by_loops():
    # a path 0-1-2-3-4-5, labels [1 1 1 0 0 0], seeds [1 0 0 0 0 1]
    w = torch.zeros(1, 6, 6, dtype=torch.float64)
    for i in range(5):
        w[0, i, i + 1] = w[0, i + 1, i] = 1.0
    y, seed = torch.tensor([[1, 1, 1, 0, 0, 0]]), torch.tensor([[1, 0, 0, 0, 0, 1]])
    st = p7.point_stats(w, y, seed, 3)
    assert st["p"][0].tolist() == [1.0, 1.0, 0.5, 0.5, 1.0, 1.0]
    assert st["a"][0, 1].item() == 0.5 and st["a"][0, 2].item() == 0.0 and st["a"][0, 5].item() == 1.0
    assert st["breakable"][0].tolist() == [True, False, False, False, False, False]  # 0's only neighbour votes 0
    assert st["fixable"][0].tolist() == [False, False, False, False, False, True]  # 1 is a tie: neither
    reach = torch.tensor([[True, True, False, True, True, True]])
    dv = p7.diag_values(st, reach)
    assert dv["hit"][0].tolist() == [1, 0, 0, 1, 1, 0] and dv["p_hit"][0].tolist() == [1, 0, 0, 0.5, 1, 0]
    assert dv["p_miss"][0].tolist() == [0, 1, 0.5, 0, 0, 1] and dv["reach_miss"][0].tolist() == [0, 1, 0, 0, 0, 1]
    assert dv["a_miss"][0].tolist() == [0, 0.5, 0, 0, 0, 1] and dv["n_a_miss"][0].tolist() == [0, 1, 1, 0, 0, 1]
    assert dv["n_b_miss"][0].tolist() == [0, 0, 1, 0, 0, 0] and dv["b_miss"][0].tolist() == [0] * 6
    assert dv["fixable"][0].tolist() == [0, 0, 0, 0, 0, 1] and dv["breakable"][0].tolist() == [1, 0, 0, 0, 0, 0]
    av = p7.arm_values(st, torch.tensor([[0, 1, 0, 0, 0, 0]]), y)  # fixes 1 and 5, breaks 0
    assert av["fixed"][0].tolist() == [0, 1, 0, 0, 0, 1] and av["broken"][0].tolist() == [1, 0, 0, 0, 0, 0]
    assert av["p_fixed"][0].tolist() == [0, 1, 0, 0, 0, 1] and av["p_broken"][0].tolist() == [1, 0, 0, 0, 0, 0]
    # random weighted graphs against the loop form; reachability against BFS
    for seed_ in range(3):
        xyz, u = block(p=30, seed=seed_)
        for graph in p7.GRAPHS:
            idx, a = p7.knn_graph(xyz, u, graph, 3)
            w = p7.affinity(idx, a, 30)
            g = torch.Generator().manual_seed(seed_)
            y = torch.randint(0, 3, (2, 30), generator=g)
            seed = torch.where(torch.rand(2, 30, generator=g) < 0.7, y, torch.randint(0, 3, (2, 30), generator=g))
            st = p7.point_stats(w, y, seed, 3)
            reach = p7.reachable(idx, a, y, st["hit"])
            for b in range(2):
                ref = reference_stats(w[b].tolist(), y[b].tolist(), seed[b].tolist(), 3)
                for key, vals in ref.items():
                    got = st[key][b].tolist()
                    assert all(abs(float(x) - float(r)) < 1e-12 for x, r in zip(got, vals)), (graph, key)
                edges = [(i, int(idx[b, i, m])) for i in range(30) for m in range(3) if a[b, i, m] > 0]
                assert reach[b].tolist() == bfs_reachable(edges, y[b].tolist(), st["hit"][b].tolist())
    # a class without any correct seed is unreachable
    idx = torch.tensor([[[1], [0], [3], [2]]])
    reach = p7.reachable(idx, torch.ones(1, 4, 1, dtype=torch.float64), torch.tensor([[1, 1, 2, 2]]),
                         torch.tensor([[False, False, True, False]]))
    assert reach.tolist() == [[False, False, True, True]]


def test_p7_6_columns_conditions_and_sums():
    y = np.array([[0, 1, 2, 1], [2, 0, 1, 2]])
    col, cond = p7.point_columns(y, [9, 5], [6, 1, 9, 7, 2, 5])
    assert col.tolist() == [[0, 3, 6, 3], [6, 0, 3, 6]]
    assert cond.tolist() == [[p7.OWN, p7.OWN, p7.OTHER, p7.OWN], [p7.OWN, p7.OWN, p7.OTHER, p7.OWN]]
    out = np.zeros((2, 7, 2))
    vals = {"x": torch.arange(8, dtype=torch.float64).view(2, 4), "y": torch.ones(2, 4, dtype=torch.float64)}
    p7.accumulate(out, col, cond, vals, ("x", "y"))
    assert out[0, 3, p7.OWN] == 1 + 3 and out[0, 3, p7.OTHER] == 6 and out[0, 6, p7.OWN] == 4 + 7
    assert out[1].sum() == 8
    with pytest.raises(ValueError, match="one query block per way"):
        p7.point_columns(np.zeros((3, 4), dtype=int), [9, 5], [6, 1, 9, 7, 2, 5])
    arr = np.zeros((len(p7.DIAG_STATS), 3, 2))
    s = {n: i for i, n in enumerate(p7.DIAG_STATS)}
    arr[s["n"]], arr[s["hit"]], arr[s["p_miss"]], arr[s["fixable"]] = 10.0, 6.0, 2.0, 1.0
    arr[s["n"], 2, p7.OTHER] = 30.0  # class 2, other condition
    summ = p7.diag_summary(arr, [6, 1])
    assert summ["fg_own"]["n"] == 20 and summ["fg_own"]["recall"] == 0.6 and summ["fg_own"]["p_miss"] == 0.5
    assert summ["fg_other"]["n"] == 40 and summ["fg_other"]["recall"] == 12 / 40
    assert summ["1_other"]["fixable"] == 1 / 24 and summ["background"]["fixable"] == 0.25
    assert np.isnan(summ["background"]["a_miss"])  # no missed point with a same-class neighbour counted


def test_p7_7_seeds_are_d39s_combined_rule_and_arms_are_the_spreading():
    f_q, f_s, sy, xyz, labels = episode()
    rows = p6.base_rows(f_q, f_s, sy)
    assert torch.equal(p7.both_logits(f_q, f_s, sy), d39.arm_logits(f_q, f_s, sy, rows, labels)["both"])
    arms = {"a": dict(graph="xyzf", k=8, beta=0.9), "b": dict(graph="feat", k=16, beta=0.5)}
    lp_u = dict(graph="xyz", k=8, beta=0.99)
    preds, diag, arm_vals = p7.episode_pass(f_q, f_s, sy, xyz, labels, arms, lp_u)
    assert set(preds) == {"U", "both", "a", "b", "lp_u"} and set(arm_vals) == {"a", "b"}
    assert torch.equal(preds["both"], p7.both_logits(f_q, f_s, sy).argmax(-1))
    assert not torch.equal(preds["both"], preds["U"])  # the two seeds differ on this episode
    assert set(diag) == {p7.graph_name(g, k) for g, k in p7.graph_keys()}
    u = F.normalize(f_q, dim=-1)
    seed = preds["both"]
    for name, arm in list(arms.items()) + [("lp_u", lp_u)]:
        idx, a = p7.knn_graph(xyz, u, arm["graph"], arm["k"])
        s = p7.normalized(p7.affinity(idx, a, 64))
        src = preds["U"] if name == "lp_u" else seed
        assert torch.equal(preds[name], p7.spread(s, F.one_hot(src, 3).double(), arm["beta"]).argmax(-1))
    assert torch.equal(preds["U"], p6.rule_logits(f_q, rows).argmax(-1))


def test_p7_8_no_label_enters_a_prediction_and_blocks_are_independent():
    f_q, f_s, sy, xyz, labels = episode(seed=3)
    arms = {p7.lp_name(a): a for a in p7.lp_arms()[::5]}
    lp_u = p7.lp_arms()[7]
    preds, diag, _ = p7.episode_pass(f_q, f_s, sy, xyz, labels, arms, lp_u)
    other_labels = torch.randint(0, 3, labels.shape, generator=torch.Generator().manual_seed(9))
    preds2, _, _ = p7.episode_pass(f_q, f_s, sy, xyz, other_labels, arms, lp_u)
    assert all(torch.equal(preds[n], preds2[n]) for n in preds)
    sw, _, _ = p7.episode_pass(f_q[[1, 0]], f_s, sy, xyz[[1, 0]], labels[[1, 0]], arms, lp_u)
    assert all(torch.equal(preds[n][[1, 0]], sw[n]) for n in preds)
    perm = torch.randperm(64, generator=torch.Generator().manual_seed(4))
    pp, dp, _ = p7.episode_pass(f_q[:, perm], f_s, sy, xyz[:, perm], labels[:, perm], arms, lp_u)
    assert all(torch.equal(preds[n][:, perm], pp[n]) for n in preds)
    for g in diag:
        assert torch.allclose(diag[g]["p_miss"][:, perm], dp[g]["p_miss"], atol=1e-12)


def fake_draws(t_both=70.0, t_lp=72.0, t_u=69.0, t_lpu=70.0):
    def counts(t, seed):
        rng = np.random.default_rng(seed)
        c = np.zeros((60, 3, 3))
        c[:, 0, :] = c[:, 1, :] = 100.0
        c[:, 2, 1:] = np.clip(t + rng.normal(0, 2, (60, 2)), 0, 100)
        c[:, 2, 0] = 90.0
        return c

    stats = np.zeros((len(p7.ARM_STATS), 3, 2))
    stats[0, 1:, :], stats[1, 1:, :] = 50.0, 25.0  # fixed 200, broken 100
    stats[2, 1:, :], stats[3, 1:, :] = 40.0, 10.0  # mean p 0.8 / 0.4
    summ = {"xyz_k8": p7.diag_summary(np.ones((len(p7.DIAG_STATS), 3, 2)), [6, 1])}
    draws = {}
    for i, d in enumerate(p7.DRAWS):
        cnt = {n: counts(t, i) for n, t in (("model", 68.0), ("U", t_u), ("both", t_both), ("lp", t_lp),
                                            ("lp_u", t_lpu))}
        res = {"miou": {k: float(p7.p0.miou_from_counts(v.sum(0))) for k, v in cnt.items()},
               "arm_stats": {"lp": stats.tolist()}, "diag_summary": summ}
        draws[d] = (res, cnt)
    return draws


def test_p7_9_rules():
    sel = {"frozen": "lp_xyz_k8_b0.9", "valid_gain": 1.2, "gate": True}
    v = dict(p7.decide(fake_draws(), sel))
    assert "P7.2 adopt + P7.3 trained form admissible" in v and "P7.6 mechanism confirmed" in v
    assert "P7.5 leak-free reported" in v and any(k.startswith("P7a xyz_k8") for k in v)
    draws = fake_draws()
    lf = draws["leakfree"][0]["miou"]
    lf["lp"] = lf["both"] - 0.001
    assert "P7.5 leak-free protocol-dependent" in dict(p7.decide(draws, sel))
    draws = fake_draws(t_lp=70.6)  # about +0.7
    stats = np.asarray(draws["fixed100"][0]["arm_stats"]["lp"])
    stats[1] = 2 * stats[0]  # breaks more than it fixes
    draws["fixed100"][0]["arm_stats"]["lp"] = stats.tolist()
    v = dict(p7.decide(draws, sel))
    assert "P7.2 adopt" in v and "P7.6 mechanism unexplained" in v
    draws = fake_draws(t_lp=70.6)
    draws["random600:1"][0]["miou"]["lp"] = draws["random600:1"][0]["miou"]["both"] - 0.001
    assert "P7 between (not adopted)" in dict(p7.decide(draws, sel))
    draws = fake_draws(t_lp=70.2)
    draws["leakfree"][0]["miou"]["lp"] = draws["leakfree"][0]["miou"]["both"] - 0.001
    v = dict(p7.decide(draws, sel))
    assert "P7.4 stop" in v and "P7.6 mechanism confirmed" in v and "P7.5 leak-free reported" in v
    assert p7.passes_gate(0.5) and not p7.passes_gate(0.49)
    v = dict(p7.decide(fake_draws(), dict(sel, gate=False, valid_gain=0.3)))
    assert "P7.1 gate stop" in v and "P7.4 stop at the gate" in v and not any(k.startswith("P7.2") for k in v)
    draws = fake_draws()
    del draws["leakfree"]
    assert p7.decide(draws, sel)[-1][0] == "incomplete"
    names = [p7.lp_name(a) for a in p7.lp_arms()]
    assert len(set(names)) == 24 and names[0] == "lp_xyz_k8_b0.5" and names[-1] == "lp_feat_k16_b0.99"
    assert all(p7.arm_of(n) == a for n, a in zip(names, p7.lp_arms()))
    with pytest.raises(ValueError):
        p7.arm_of("lp_rgb_k8_b0.5")
    miou = {"both": 0.50, names[3]: 0.51, names[9]: 0.51, **{n: 0.5 for n in names if n not in (names[3], names[9])}}
    assert p6.select(miou, names, base="both") == (names[3], pytest.approx(1.0))  # a tie goes to the earlier arm


def test_p7_10_files_parse_as_python_310():
    for f in ("experiments/p7_propagation_probe.py", "tests/test_propagation_probe.py"):
        ast.parse((REPO / f).read_text(encoding="utf-8"), feature_version=(3, 10))
