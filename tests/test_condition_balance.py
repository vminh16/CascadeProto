"""CB-1..6 (05 §3.8y): condition-balanced training episodes of [DECISION D-49], and CH-1..6: the correlation head of
[DECISION D-48 amendment 6]. Beyond the paper. CPU.
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from pipeline import episodes as ep_mod
from pipeline.model_api import episode_loss


def block(p=2048, own_share=0.4, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.random((p, 9)).astype(np.float32)
    x[:, 3:6] = rng.random((p, 3))  # rgb, distinct per point: identifies a point after the shuffle
    lab = np.zeros(p, dtype=np.int64)
    lab[: int(own_share * p)] = 1
    lab[int(own_share * p): int(own_share * p) + 100] = 2
    return x, lab


# ------------------------------------------------------------------ D-49

def test_cb1_sparse_view_thins_the_own_class_to_its_raw_share():
    x, lab = block(own_share=0.5)
    f = 0.5
    r = 1.0 - np.sqrt(1.0 - f)
    shares = []
    for s in range(20):
        out_x, out_lab = ep_mod.sparse_query_view(x, lab, 1, np.random.default_rng(s))
        assert out_x.shape == x.shape and out_lab.shape == lab.shape
        shares.append(float((out_lab == 1).mean()))
    assert abs(np.mean(shares) - r) < 0.02  # the sampler's raw share, background density
    assert (out_lab == 2).sum() >= 100  # the other class keeps all its points (and gains copies)


def test_cb2_labels_follow_their_points_and_xyz_is_the_loaders():
    x, lab = block()
    out_x, out_lab = ep_mod.sparse_query_view(x, lab, 1, np.random.default_rng(3))
    by_rgb = {tuple(np.round(x[i, 3:6], 6)): lab[i] for i in range(x.shape[0])}
    assert all(by_rgb[tuple(np.round(out_x[i, 3:6], 6))] == out_lab[i] for i in range(out_x.shape[0]))
    kept_xyz = {tuple(np.round(v, 6)) for v in x[:, :3]}
    assert sum(tuple(np.round(v, 6)) in kept_xyz for v in out_x[:, :3]) >= 0.5 * x.shape[0]  # xyz not re-shifted
    shifted = out_x[:, :3] - out_x[:, :3].min(axis=0)
    assert np.allclose(out_x[:, 6:], shifted / shifted.max(axis=0), atol=1e-6)


def test_cb3_blocks_without_or_only_their_class_are_unchanged():
    x, lab = block()
    same_x, same_lab = ep_mod.sparse_query_view(x, np.zeros_like(lab), 1, np.random.default_rng(0))
    assert same_x is x
    same_x, _ = ep_mod.sparse_query_view(x, np.ones_like(lab), 1, np.random.default_rng(0))
    assert same_x is x


class Items:
    classes = np.array([3, 5])

    def __len__(self):
        return 4

    def __getitem__(self, i):
        rng = np.random.default_rng(i)
        qx = rng.random((2, 2048, 9)).astype(np.float32)
        qy = np.zeros((2, 2048), dtype=np.int64)
        qy[0, :900], qy[1, :700], qy[1, 700:760] = 1, 2, 1
        return (rng.random((2, 1, 2048, 9)).astype(np.float32), np.ones((2, 1, 2048), dtype=np.int32), qx, qy,
                np.array([3, 5]))


def test_cb4_wrapper_thins_only_queries_deterministically():
    base = Items()
    assert ep_mod.with_condition_balance(base, 0.0, 0) is base
    with pytest.raises(ValueError):
        ep_mod.ConditionBalance(base, 1.5, 0)
    state = np.random.get_state()
    cb = ep_mod.ConditionBalance(base, 1.0, 0)
    a, b = cb[2], cb[2]
    assert all(np.array_equal(u, v) for u, v in zip(a, b))  # a function of (seed, i)
    assert np.array_equal(np.random.get_state()[1], state[1])  # the global RNG is not drawn from
    orig = base[2]
    assert np.array_equal(a[0], orig[0]) and np.array_equal(a[1], orig[1])  # supports untouched
    assert (a[3][0] == 1).mean() < (orig[3][0] == 1).mean()  # block 0 thinned in class 1
    assert (a[3][1] == 2).mean() < (orig[3][1] == 2).mean()  # block 1 thinned in class 2
    assert all(np.array_equal(u, v) for u, v in zip(ep_mod.ConditionBalance(base, 0.0, 0)[2], orig))


def test_cb5_train_flag_and_run_tag():
    import train

    args = train.parse_args(["--dataset", "s3dis", "--data_path", "x", "--cvfold", "1", "--n_way", "2", "--k_shot",
                             "1", "--condition_balance", "0.5"])
    assert args.condition_balance == 0.5 and train.run_dir(args).endswith("_cb0.5")
    args = train.parse_args(["--dataset", "s3dis", "--data_path", "x", "--cvfold", "1", "--n_way", "2", "--k_shot", "1"])
    assert "_cb" not in train.run_dir(args)


def test_cb6_new_files_parse_as_python310():
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for f in ("pipeline/episodes.py", "models/corr_head.py", "models/cascadeproto.py", "experiments/d49_eval.py"):
        ast.parse((root / f).read_text(encoding="utf-8"), feature_version=(3, 10))


def synthetic_draws(delta_fixed: float, delta_leak: float, alpha_arm: float, recall_arm: float):
    """draws[d] = (merged, counts) for names cr / cb with row U; the arm's TP shifted per draw."""
    from experiments import d49_eval as d49

    rng = np.random.default_rng(0)
    base = rng.integers(200, 400, size=(60, 3, 7)).astype(np.float64)
    base[:, 2] = np.minimum(base[:, 0], base[:, 1]) * 0.6
    draws = {}
    for d in d49.DRAWS:
        delta = delta_leak if d == "leakfree" else delta_fixed
        arm = base.copy()
        arm[:, 2] = np.clip(base[:, 2] * (1 + delta), 0, np.minimum(base[:, 0], base[:, 1]))
        counts = {f"{n}:{r}": (base if n == "cr" else arm) for n in ("cr", "cb") for r in d49.ROWS}
        from experiments import p0_em_probe as p0

        miou = {k: 100.0 * float(p0.miou_from_counts(v.sum(0))) for k, v in counts.items()}
        cond = {"cr:U": {"recall_other": [np.nan, 0.10, 0.10]}, "cb:U": {"recall_other": [np.nan, recall_arm, recall_arm]}}
        alpha = {"cr": {"other": [0, 0, -0.02, 0, 0]}, "cb": {"other": [0, 0, alpha_arm, 0, 0]}}
        draws[d] = ({"miou": miou, "condition": cond, "alpha": alpha}, counts)
    return draws


def test_cb7_d49_rules_on_every_branch():
    from experiments import d49_eval as d49

    head = lambda rows: rows[2][0]  # noqa: E731, the verdict line after D49.1 and D49.2
    assert head(d49.decide49(synthetic_draws(0.2, 0.2, 0.0, 0.2), "cr", "cb")).startswith("D49.5")  # no mechanism
    assert head(d49.decide49(synthetic_draws(0.2, 0.2, 0.2, 0.2), "cr", "cb")).startswith("D49.3 cb is the base")
    assert head(d49.decide49(synthetic_draws(-0.05, 0.2, 0.2, 0.2), "cr", "cb")).startswith("D49.4")
    assert head(d49.decide49(synthetic_draws(0.0, 0.0, 0.2, 0.2), "cr", "cb")).startswith("D49.3 fails")
    # leak-free holds, fixed100 above 0 but below the bar: CR stays (D49.4 needs a fixed100 loss)
    assert head(d49.decide49(synthetic_draws(0.004, 0.2, 0.2, 0.2), "cr", "cb")).startswith("D49.3 fails")
    assert d49.decide49({}, "cr", "cb")[0][0] == "incomplete"


# ------------------------------------------------------------------ D-48 amendment 6, the correlation head

def corr_model(layers=2):
    from models.cascadeproto import CascadeProto, CascadeProtoConfig
    from models.vipseg_backbone import PointFeatureExtractor
    from tests.test_feature_extractor import StandInEncoder

    torch.manual_seed(0)
    cfg = CascadeProtoConfig(use_lma=False, num_stages=layers, use_adrm=False, stage_type="corr")
    return CascadeProto(cfg, PointFeatureExtractor(encoder=StandInEncoder())).double()


def test_ch1_configuration_and_shapes():
    from models.cascadeproto import CascadeProtoConfig
    from tests.test_cascadeproto import episode

    for bad in (dict(num_stages=0), dict(use_lma=True), dict(use_adrm=True), dict(l2norm_point_proto=True)):
        base = dict(use_lma=False, num_stages=2, use_adrm=False, stage_type="corr")
        with pytest.raises(ValueError):
            CascadeProtoConfig(**{**base, **bad})
    m = corr_model().eval()
    ep = episode(n=2, k=1, bq=2, seed=1)
    out = m(ep)
    assert out.logits.shape == (2, 2048, 3) and out.loss_reg is None
    assert len(m.stages) == 0 and m.routing is None


def test_ch2_equivariant_to_the_order_of_the_ways():
    from tests.test_cascadeproto import episode

    m = corr_model().eval()
    ep = episode(n=2, k=1, bq=2, seed=2)
    out = m(ep).logits
    ep.support_x, ep.support_y = ep.support_x[[1, 0]], ep.support_y[[1, 0]]
    swapped = m(ep).logits
    assert torch.allclose(swapped, out[..., [0, 2, 1]], atol=1e-8)


def test_ch3_training_loss_is_ce_plus_deep_supervision():
    from models.corr_head import deep_supervision
    from tests.test_cascadeproto import episode

    m = corr_model(layers=3).train()
    ep = episode(n=2, k=1, bq=2, seed=3)
    out = m(ep)
    layers = m.corr(*reversed(m.features.encode_episode(ep.support_x, ep.query_x)), ep.support_y, ep.query_x[..., :3])
    assert torch.allclose(out.logits, layers[-1])
    assert torch.allclose(out.loss_reg, deep_supervision(layers, ep.query_y))
    seg = F.cross_entropy(out.logits.reshape(-1, 3), ep.query_y.reshape(-1))
    loss = episode_loss(out, ep)
    assert torch.allclose(loss, seg + out.loss_reg)
    loss.backward()
    assert m.corr.readout.weight.grad is not None and m.features.fc[0].weight.grad.abs().sum() > 0


def test_ch4_deep_supervision_is_the_mean_of_the_layer_ce():
    from models.corr_head import deep_supervision

    g = torch.Generator().manual_seed(4)
    logits = [torch.randn(2, 30, 3, generator=g, dtype=torch.float64) for _ in range(3)]
    y = torch.randint(0, 3, (2, 30), generator=g)
    ref = sum(F.cross_entropy(l.reshape(-1, 3), y.reshape(-1)) for l in logits) / 3
    assert torch.allclose(deep_supervision(logits, y), ref)


def test_ch5_evaluation_reads_no_query_label():
    from tests.test_cascadeproto import episode

    m = corr_model().eval()
    ep = episode(n=2, k=1, bq=2, seed=5)
    out = m(ep).logits
    ep.query_y = torch.zeros_like(ep.query_y)
    assert torch.equal(m(ep).logits, out)


def test_ch6_layer_mixing_is_equivariant_across_rows():
    from models.corr_head import RowAttentionLayer

    torch.manual_seed(1)
    layer = RowAttentionLayer().double()
    h = torch.randn(2, 50, 3, 64, dtype=torch.float64)
    out = layer(h)
    perm = layer(h[:, :, [0, 2, 1]])
    assert torch.allclose(perm, out[:, :, [0, 2, 1]], atol=1e-10)
