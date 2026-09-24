"""D37-T1..T11 (05 §3.8o): the cross-term forms of [DECISION D-36] and the 2 x 2 of [DECISION D-37], beyond the paper.

The inherited PEM / PDM are read from `models/vipseg.py` with `ast` (the file is parsed, never edited; importing it
needs `pointnet2_ops`), so T1-T5 run on the CPU in float64 against the real modules. T6 (marker `cuda`) repeats
the equivariance and gradient checks on the real model. The projection weights are scaled by 20 (as in C2) so
that the attention is far from uniform and a positional path, if present, is visible.
"""

import ast
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import models.vip_stage as vip_stage
from models.cascadeproto import STAGE_TYPES, CascadeProto, CascadeProtoConfig, build_stage
from models.vip_stage import CrossFormModule, VIPStage, cross_attention, vip_module_forward
from models.vipseg_backbone import PointFeatureExtractor
from pipeline.episodes import QUERY_ORDER_SEED_STREAM, QueryOrder, with_query_order
from pipeline.model_api import episode_loss
from tests.test_cascadeproto import episode
from tests.test_feature_extractor import StandInEncoder

REPO = Path(__file__).resolve().parents[1]
D, P = 128, 2048
ATOL = 1e-12
ROUTE_B = dict(use_lma=False, num_stages=4, l2norm_point_proto=True)  # E1's head [D-25] [D-30]


def inherited(name: str):
    src = (REPO / "models" / "vipseg.py").read_text(encoding="utf-8")
    ns = {"torch": torch, "nn": nn, "F": F}
    for node in ast.parse(src).body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            exec(compile(ast.Module([node], []), "models/vipseg.py", "exec"), ns)
    return ns[name]


PEM, PDM = inherited("PrototypeEnhancementModule"), inherited("PrototypeDifferenceModule")


def vip_modules(seed=0, dtype=torch.float64):
    torch.manual_seed(seed)
    mods = [(PEM if step % 2 == 0 else PDM)().to(dtype) for step in range(4)]
    with torch.no_grad():
        for m in mods:
            m.map.weight.mul_(20.0)
    return mods


def tensors(n=2, k=1, bq=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    f_s = torch.rand(n, k, P, D, generator=g, dtype=torch.float64)
    f_q = torch.rand(bq, P, D, generator=g, dtype=torch.float64)
    p0 = torch.randn(n + 1, D, generator=g, dtype=torch.float64)
    return f_s, f_q, p0.unsqueeze(0).expand(bq, -1, -1)  # P^0 is the same for every query (02 §4.5)


def chain(mods, form, f_s, f_q, p):
    for step, m in enumerate(mods):
        p = VIPStage(step, module=m, cross_form=form)(p, f_s, f_q)
    return p  # [B_q, N+1, D]


# ------------------------------------------------------------------ T1-T3 the forms

@pytest.mark.parametrize("cls,n,k,bq", [(PEM, 2, 1, 2), (PDM, 2, 1, 2), (PEM, 2, 2, 2), (PDM, 2, 2, 2),
                                        (PEM, 3, 1, 3), (PDM, 3, 2, 3)])
def test_d37_t1_scrambled_reproduces_the_inherited_module(cls, n, k, bq):
    torch.manual_seed(1)
    m = cls().double()
    with torch.no_grad():
        m.map.weight.mul_(20.0)
    f_s, f_q, p = tensors(n, k, bq, seed=2)
    p = p + 0.1 * torch.randn_like(p)  # per-query prototypes, as after the first stage
    assert torch.allclose(vip_module_forward(m, f_q, f_s, p, "scrambled"), m(f_q, f_s, p), atol=ATOL, rtol=0)
    assert torch.equal(CrossFormModule(m)(f_q, f_s, p), m(f_q, f_s, p))  # native calls the module unchanged
    assert torch.allclose(CrossFormModule(m, "scrambled")(f_q, f_s, p), m(f_q, f_s, p), atol=ATOL, rtol=0)


def test_d37_t2_cross_attention_forms():
    """scrambled = C2's index algebra (D-35 point 5); clean = softmax(Q'_bᵀ S'_w / √128) per query and slot."""
    from experiments.c2_vipseg_crosscorr_check import formula

    g = torch.Generator().manual_seed(3)
    qp = torch.randn(2, 72, D, generator=g, dtype=torch.float64)
    sp = torch.randn(3, 72, D, generator=g, dtype=torch.float64)
    assert torch.allclose(cross_attention(qp, sp, "scrambled"), formula(qp, sp), atol=ATOL, rtol=0)
    clean = cross_attention(qp, sp, "clean")
    for b in range(2):
        for w in range(3):
            ref = torch.softmax(qp[b].T @ sp[w] / D ** 0.5, dim=-1)  # [D, D]
            assert torch.allclose(clean[b, w], ref, atol=ATOL, rtol=0)
    assert not torch.allclose(clean, formula(qp, sp), atol=1e-3)
    with pytest.raises(ValueError, match="cross-term form"):
        cross_attention(qp, sp, "native")
    f_s, f_q, p = tensors()
    with pytest.raises(TypeError, match="expected one of"):
        vip_module_forward(nn.Linear(2, 2), f_q, f_s, p, "clean")


@pytest.mark.parametrize("bq,perm", [(2, [1, 0]), (3, [2, 0, 1])])
def test_d37_t3_clean_head_is_query_equivariant_and_query_local(bq, perm):
    mods = vip_modules()
    f_s, f_q, p = tensors(n=2, k=1, bq=bq, seed=4)
    perm = torch.tensor(perm)
    other = f_q.clone()
    other[1:] = torch.rand_like(other[1:])
    out = {form: chain(mods, form, f_s, f_q, p) for form in ("native", "scrambled", "clean")}
    assert torch.allclose(out["scrambled"], out["native"], atol=ATOL, rtol=0)
    clean_perm = chain(mods, "clean", f_s, f_q[perm], p)
    assert torch.allclose(clean_perm, out["clean"][perm], atol=ATOL, rtol=0)  # equivariant
    assert torch.allclose(chain(mods, "clean", f_s, other, p)[0], out["clean"][0], atol=ATOL, rtol=0)  # local
    # the same checks detect VIP-Seg's form: the position and the other queries change a query's prototypes
    assert (chain(mods, "native", f_s, f_q[perm], p) - out["native"][perm]).abs().max() > 1e-3
    assert (chain(mods, "native", f_s, other, p)[0] - out["native"][0]).abs().max() > 1e-3


# ------------------------------------------------------------------ T4 configuration and parameters

def route_b_model(stage_type, monkeypatch, seed=0):
    mods = iter(vip_modules(seed))
    monkeypatch.setattr(vip_stage, "build_vip_module", lambda step: next(mods))
    torch.manual_seed(seed)
    config = CascadeProtoConfig(stage_type=stage_type, **ROUTE_B)
    return CascadeProto(config, PointFeatureExtractor(encoder=StandInEncoder())).double()


def test_d37_t4_vip_clean_shares_vip_parameters_and_configuration(monkeypatch):
    assert "vip_clean" in STAGE_TYPES
    vip, clean = route_b_model("vip", monkeypatch), route_b_model("vip_clean", monkeypatch)
    assert [s.cross_form for s in vip.stages] == ["native"] * 4
    assert [s.cross_form for s in clean.stages] == ["clean"] * 4
    assert list(vip.state_dict()) == list(clean.state_dict())
    clean.load_state_dict(vip.state_dict(), strict=True)  # an E1 checkpoint loads into the clean form
    monkeypatch.setattr(vip_stage, "build_vip_module", lambda step: PEM())
    assert build_stage(CascadeProtoConfig(stage_type="vip_clean", **ROUTE_B), 0).cross_form == "clean"
    with pytest.raises(ValueError, match="ignores"):
        CascadeProtoConfig(stage_type="vip_clean", cross_attn_scale="sqrt_D").check_implemented()
    with pytest.raises(ValueError, match="cross_form"):
        VIPStage(0, module=PEM(), cross_form="other")
    assert not hasattr(VIPStage(0, module=PEM()), "cross")  # diag_short reads `cross` as an EPPM stage


# ------------------------------------------------------------------ T5 the lemma of D-37 (CF = CR)

def loss_and_grads(model, ep):
    model.zero_grad()
    loss = episode_loss(model(ep), ep)
    loss.backward()
    return loss.item(), {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}


def permuted(ep, perm):
    out = episode(n=2, k=1, bq=len(perm), seed=5)
    out.support_x, out.support_y = ep.support_x, ep.support_y
    out.query_x, out.query_y = ep.query_x[perm], ep.query_y[perm]
    return out


def test_d37_t5_clean_head_trains_identically_in_any_query_order(monkeypatch):
    """Training mode (BatchNorm on batch statistics): a permuted episode gives the same loss and every gradient."""
    ep = episode(n=2, k=1, bq=2, seed=5)
    swapped = permuted(ep, torch.tensor([1, 0]))
    for stage_type in ("vip_clean", "vip"):
        model = route_b_model(stage_type, monkeypatch).train()
        l0, g0 = loss_and_grads(model, ep)
        l1, g1 = loss_and_grads(model, swapped)
        scale = max(g.abs().max().item() for g in g0.values())  # biases before a BatchNorm have gradient ~0
        worst = max((g0[n] - g1[n]).abs().max().item() for n in g0) / scale
        if stage_type == "vip_clean":
            assert abs(l0 - l1) < 1e-12 and set(g0) == set(g1) and len(g0) > 0
            assert worst < 1e-10
        else:  # the test detects VIP-Seg's form
            assert abs(l0 - l1) > 1e-6 or worst > 1e-3


# ------------------------------------------------------------------ T6 the real model on the GPU

def gpu_episodes(dev):
    """(stored, swapped, stored with every query coordinate moved by one float32 ULP) on the GPU."""
    ep = episode(n=2, k=1, bq=2, seed=6)
    ep.support_x, ep.query_x = ep.support_x.float(), ep.query_x.float()
    sw = permuted(ep, torch.tensor([1, 0]))
    ulp = permuted(ep, torch.tensor([0, 1]))
    ulp.query_x = torch.nextafter(ep.query_x, torch.full_like(ep.query_x, 2.0))
    return ep.to(dev), sw.to(dev), ulp.to(dev)


def worst_grad(g0, g):
    """Largest gradient difference relative to the parameter's own maximum, over non-negligible gradients."""
    scale = max(v.abs().max().item() for v in g0.values())
    return max((g0[n] - g[n]).abs().max().item() / g0[n].abs().max().item()
               for n in g0 if g0[n].abs().max().item() > 1e-3 * scale)


@pytest.mark.cuda
def test_d37_t6_real_model_clean_head_is_order_free_and_scrambled_reproduces_native():
    """On the GPU the forward and backward are float32 (TF32 in cuDNN convolutions by default), so order-freedom is
    judged against the rounding floor: the change caused by moving every query coordinate by one ULP. The clean head's
    swap must stay within 3x that floor; VIP-Seg's form must exceed 5x it (measured 2026-09-25: clean 1.8e-3 against a
    floor of 5.4e-3 without TF32, native 0.66)."""
    from train import build_model

    dev = torch.device("cuda")
    models = {}
    for st in ("vip", "vip_clean"):
        torch.manual_seed(0)
        models[st] = build_model(CascadeProtoConfig(stage_type=st, **ROUTE_B)).to(dev)
    models["vip_clean"].load_state_dict(models["vip"].state_dict(), strict=True)
    ep, sw, ulp = gpu_episodes(dev)
    with torch.no_grad():
        m = models["vip"].eval()
        native = m(ep).logits
        for s in m.stages:
            s.cross_form = "scrambled"
        scrambled = m(ep).logits
        for s in m.stages:
            s.cross_form = "native"
        assert (scrambled - native).abs().max().item() <= 1e-4 * max(1.0, native.abs().max().item())  # C4.0a
        c = models["vip_clean"].eval()
        clean = c(ep).logits
        floor = (c(ulp).logits - clean).abs().max().item()
        assert (c(sw).logits - clean[[1, 0]]).abs().max().item() <= 3 * floor + 1e-6
    for st, m in models.items():
        m.train()
        l0, g0 = loss_and_grads(m, ep)
        l1, g1 = loss_and_grads(m, sw)
        l2, g2 = loss_and_grads(m, ulp)
        if st == "vip_clean":
            assert abs(l1 - l0) <= 3 * abs(l2 - l0) + 1e-7
            assert worst_grad(g0, g1) <= 3 * worst_grad(g0, g2)
        else:
            assert abs(l1 - l0) > 5 * abs(l2 - l0)  # the test detects VIP-Seg's positional form

# ------------------------------------------------------------------ T7 random query order

class FakeEpisodes:
    """Episode i: query block b filled with 10 i + b, labelled b + 1 (the loader's order)."""

    classes = np.array([3, 7])

    def __len__(self):
        return 4000

    def __getitem__(self, i):
        qx = np.stack([np.full((P, 9), 10.0 * i + b) for b in range(2)])  # [2, P, 9]
        qy = np.stack([np.full(P, b + 1) for b in range(2)])  # [2, P]
        return np.full((2, 1, P, 9), float(i)), np.ones((2, 1, P), np.int32), qx, qy, np.array([3, 7])


def test_d37_t7_query_order_permutes_blocks_with_labels_and_nothing_else():
    base = FakeEpisodes()
    assert with_query_order(base, "fixed", 0) is base
    with pytest.raises(ValueError, match="query_order"):
        with_query_order(base, "shuffled", 0)
    q = with_query_order(base, "random", 0)
    assert isinstance(q, QueryOrder) and len(q) == len(base) and (q.classes == base.classes).all()
    state = np.random.get_state()
    swaps = 0
    for i in range(2000):
        sx, sy, qx, qy, cl = q[i]
        bsx, bsy, bqx, bqy, bcl = base[i]
        order = np.random.default_rng([0, QUERY_ORDER_SEED_STREAM, i]).permutation(2)
        assert (qx == bqx[order]).all() and (qy == bqy[order]).all()
        assert (qx[:, 0, 0] - 10.0 * i == qy[:, 0] - 1).all()  # each label travelled with its block
        assert (sx == bsx).all() and (sy == bsy).all() and (cl == bcl).all()
        swaps += int(order[0] == 1)
    assert 0.45 < swaps / 2000 < 0.55
    after = np.random.get_state()
    assert state[0] == after[0] and (state[1] == after[1]).all() and state[2:] == after[2:]  # global RNG untouched
    np.random.seed(123)
    assert all((q[i][2] == with_query_order(base, "random", 0)[i][2]).all() for i in range(50))  # deterministic
    other = with_query_order(base, "random", 1)
    assert any(not (q[i][2] == other[i][2]).all() for i in range(50))


def test_d37_t8_train_flags_and_run_directories():
    import train

    e1 = ["--dataset", "s3dis", "--data_path", "x", "--cvfold", "1", "--n_way", "2", "--k_shot", "1",
          "--use_lma", "false", "--num_stages", "4", "--stage_type", "vip", "--l2norm_point_proto", "true",
          "--batch_size", "1", "--lr_step_epochs", "15", "--valid_every", "4", "--save_dir", "log_r2"]
    args = train.parse_args(e1)
    assert args.query_order == "fixed"
    assert train.run_dir(args).replace("\\", "/") == "log_r2/s3dis_S1_N2_K1_point_T4_vip_b1"  # E1 unchanged
    vr = train.parse_args(e1[:-2] + ["--save_dir", "log_d37", "--query_order", "random"])
    assert train.run_dir(vr).replace("\\", "/") == "log_d37/s3dis_S1_N2_K1_point_T4_vip_b1_qrandom"
    cr = train.parse_args([a if a != "vip" else "vip_clean" for a in e1[:-2]] + ["--save_dir", "log_d37",
                                                                                  "--query_order", "random"])
    assert train.run_dir(cr).replace("\\", "/") == "log_d37/s3dis_S1_N2_K1_point_T4_vip_clean_b1_qrandom"
    assert train.model_config(cr).stage_type == "vip_clean"
    old = {k: v for k, v in train.comparable_args(args).items() if k != "query_order"}  # a checkpoint before D-37
    assert train.resume_mismatch(old, train.comparable_args(args)) == []
    assert train.resume_mismatch(old, train.comparable_args(vr)) == ["query_order"]  # save_dir may differ


# ------------------------------------------------------------------ T9-T10 the readers of C4 and D-37

def test_d37_t9_c4_bookkeeping_and_rule():
    from experiments import c4_crossterm_ablation as c4

    qy = np.array([[1, 1, 0, 2], [2, 2, 1, 0]])
    pred = np.array([[1, 2, 0, 0], [0, 1, 1, 1]])
    own = dict(points=0, own=0, other=0, background=0)
    c4.own_update(own, pred, qy, swap=True)
    # block 0 own label 1 at points 0-1: predicted 1 (own), 2 (other); block 1 own label 2 at 0-1: 0 (bg), 1 (other)
    assert own == dict(points=4, own=1, other=2, background=1)
    stored = dict(points=0, own=0, other=0, background=0)
    c4.own_update(stored, pred, qy, swap=False)
    assert stored == own  # the counts do not depend on the order; the shift compares two passes
    with pytest.raises(ValueError, match="2-way"):
        c4.own_update(own, np.zeros((3, 4), int), np.zeros((3, 4), int), swap=False)

    def result(stored, swapped_, other_stored, other_swapped):
        miou = dict(native_stored=0.73, native_swapped=0.01, scrambled_stored=0.73, clean_stored=stored,
                    clean_swapped=swapped_)
        share = {k: dict(own=0.9, other=0.0, background=0.1) for k in miou}
        share["native_swapped"]["other"] = 0.9
        share["clean_stored"]["other"], share["clean_swapped"]["other"] = other_stored, other_swapped
        return dict(miou=miou, own_class_share=share, form_error=1e-7)

    assert c4.relabel_shift(result(0.5, 0.5, 0.04, 0.05), "clean") == pytest.approx(0.01)
    assert c4.relabel_shift(result(0.5, 0.5, 0.04, 0.05), "native") == pytest.approx(0.9)
    # honest confusion (0.2 in both orders) is not a positional relabel
    ok = {"checkpoints": {"e1": result(0.5, 0.5, 0.2, 0.2), "vipseg": result(0.4, 0.4005, 0.03, 0.035)}}
    assert c4.decide(ok)[1] is True
    bad = {"checkpoints": {"e1": result(0.5, 0.5, 0.0, 0.0), "vipseg": result(0.4, 0.398, 0.0, 0.0)}}
    assert c4.decide(bad)[1] is False
    assert c4.decide({"checkpoints": {"e1": result(0.5, 0.5, 0.01, 0.03)}})[1] is False
    assert c4.decide({"checkpoints": {"e1": result(0.5, 0.5, 0.03, 0.01)}})[1] is False
    assert c4.decide({"checkpoints": {}})[1] is False
    c3 = {"checkpoints": {"e1": {"miou": {"model": 0.73, "model_swapped": 0.01}}}}
    assert max(c4.check_c3(result(0.5, 0.5, 0, 0), c3, "e1").values()) < 1e-9
    with pytest.raises(RuntimeError, match="C4.0b"):
        c4.check_c3(result(0.5, 0.5, 0, 0), {"checkpoints": {"e1": {"miou": {"model": 0.7322, "model_swapped": 0.01}}}}, "e1")
    with pytest.raises(RuntimeError, match="not in"):
        c4.check_c3(result(0.5, 0.5, 0, 0), c3, "vipseg")


def counts(tp, episodes=60, seed=0):
    """[E, 3, 3] per-episode counts for two classes with IoU ≈ tp / (200 - tp)."""
    rng = np.random.default_rng(seed)
    c = np.zeros((episodes, 3, 3))
    c[:, 0, :] = 100.0
    c[:, 1, :] = 100.0
    c[:, 2, 1:] = np.clip(tp + rng.normal(0, 2, (episodes, 2)), 0, 100)
    c[:, 2, 0] = 90.0
    return c


def fake_draws(tps, swap_rows):
    """(draws, swap) with model name -> TP level; random600 draws keep the same ordering of the models."""
    from experiments import d37_eval as d37

    draws = {}
    for di, draw in enumerate(d37.DRAWS):
        cnt = {}
        for name, tp in tps.items():
            cnt[name] = counts(tp, seed=di)
            cnt[name + "_oracle"] = counts(tp + 5, seed=di)
            cnt[name + "_oracle_unit"] = counts(tp + 6, seed=di)
        miou = {k: float(d37.p0.miou_from_counts(v.sum(0))) for k, v in cnt.items()}
        paired = {f"{n}{o}_vs_{n}": d37.p0.paired_bootstrap(cnt[n], cnt[n + o]) for n in tps
                  for o in ("_oracle", "_oracle_unit")}
        draws[draw] = ({"miou": miou, "paired": paired}, cnt)
    swap = {"checkpoints": {n: {"miou": {"model": s, "model_swapped": w},
                                "own_class_share": {"swap_position": r + 0.2, "normal_other": 0.2}}
                            for n, (s, w, r) in swap_rows.items()}}  # 0.2 honest confusion in both orders
    return draws, swap


def test_d37_t10_rules():
    from experiments import d37_eval as d37

    tps = {"e1": 85, "e1_best": 86, "vr": 70, "vr_best": 71, "cr": 75, "cr_best": 76}
    rows = {"e1": (0.73, 0.01, 0.91), "vr": (0.55, 0.548, 0.01), "cr": (0.60, 0.60, 0.0)}
    draws, swap = fake_draws(tps, rows)
    e1_draws = {d: ({"miou": {k: v for k, v in r[0]["miou"].items() if k.startswith("e1")}}, None)
                for d, r in draws.items()}
    v = dict(d37.decide(draws, swap, {}, e1_draws))
    assert "D37.2 equivariance" in v and "D37.1 origin" in v and "D37.3 clean head" in v
    assert "D37.4 shortcut worth" in v and "D37.4 oracle gap_oracle_unit" in v
    # a clean head that is not order-free stops everything
    bad = dict(rows, cr=(0.60, 0.59, 0.0))
    assert list(dict(d37.decide(draws, fake_draws(tps, bad)[1], {}, e1_draws))) == ["D37.2 FAILED"]
    # VR still positional
    pos = dict(rows, vr=(0.55, 0.02, 0.9))
    assert "D37.1 STOP" in dict(d37.decide(draws, fake_draws(tps, pos)[1], {}, e1_draws))
    # tie -> clean; scrambled clearly better -> scrambled
    tie, swap_t = fake_draws(dict(tps, cr=70.1), rows)
    assert "D37.3 tie -> clean head" in dict(d37.decide(tie, swap_t, {}, e1_draws))
    worse, swap_w = fake_draws(dict(tps, cr=60), rows)
    assert "D37.3 scrambled head" in dict(d37.decide(worse, swap_w, {}, e1_draws))
    # the evaluation must reproduce E1 first
    off = {d: ({"miou": dict(r[0]["miou"], e1=r[0]["miou"]["e1"] + 0.001)}, None) for d, r in e1_draws.items()}
    assert list(dict(d37.decide(draws, swap, {}, off))) == ["E1 check failed"]
    assert list(dict(d37.decide({}, swap, {}, e1_draws))) == ["incomplete"]


def test_d37_t11_new_files_parse_as_python_310():
    for f in ("models/vip_stage.py", "experiments/c4_crossterm_ablation.py", "experiments/d37_eval.py",
              "experiments/c3_query_order.py", "pipeline/episodes.py", "train.py", "tests/test_d37_trace.py"):
        ast.parse((REPO / f).read_text(encoding="utf-8"), feature_version=(3, 10))
