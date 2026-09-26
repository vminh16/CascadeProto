"""DE-1..11 (05 §3.8t): the density-invariant encoder of [DECISION D-43] and its reader, beyond the paper.

DE-1..8 run on the CPU (pure operations, configuration, rules); DE-9..11 need the GPU encoder (marker `cuda`).
"""

import ast
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from experiments import d43_eval as d43
from experiments import p6_prototype_probe as p6
from experiments import p7_propagation_probe as p7
from models.cascadeproto import CascadeProtoConfig
from models.density_ops import ball_group, per_block_scale, per_block_standardize

REPO = Path(__file__).resolve().parents[1]


def test_de1_ball_group_against_brute_force():
    g = torch.Generator().manual_seed(0)
    xyz = torch.rand(2, 200, 3, generator=g)
    centers = xyz[:, :30]  # centres are points of the block, as FPS gives
    idx = ball_group(xyz, centers, 0.15, 8)
    assert idx.shape == (2, 30, 8)
    for b in range(2):
        for i in range(30):
            inside = [j for j in range(200) if float(((xyz[b, j] - centers[b, i]) ** 2).sum().sqrt()) <= 0.15]
            want = inside[:8] + [inside[0]] * (8 - len(inside[:8]))
            assert idx[b, i].tolist() == want
    far = torch.tensor([[[5.0, 5.0, 5.0]]])
    with pytest.raises(ValueError, match="no point"):
        ball_group(xyz[:1], far, 0.1, 4)
    with pytest.raises(ValueError, match="positive"):
        ball_group(xyz, centers, 0.0, 4)


def test_de2_per_block_statistics_do_not_depend_on_the_batch():
    g = torch.Generator().manual_seed(1)
    t = torch.randn(3, 5, 7, 4, generator=g, dtype=torch.float64)
    t[1] *= 10.0  # a block with another scale
    s, z = per_block_scale(t, 1e-5), per_block_standardize(t, 1e-6)
    for b in range(3):
        assert torch.allclose(s[b:b + 1], per_block_scale(t[b:b + 1], 1e-5), atol=1e-12)
        assert torch.allclose(z[b:b + 1], per_block_standardize(t[b:b + 1], 1e-6), atol=1e-12)
        assert abs(float(z[b].mean())) < 1e-12 and abs(float(z[b].std()) - 1.0) < 1e-5
        assert torch.allclose(s[b], t[b] / (t[b].std() + 1e-5), atol=1e-12)
    # the batch-global form VIP-Seg uses does depend on the other blocks
    glob = (t - t.mean()) / ((t - t.mean()).std() + 1e-6)
    assert not torch.allclose(glob[:1], per_block_standardize(t[:1], 1e-6), atol=1e-3)


def test_de3_config_field():
    assert CascadeProtoConfig().encoder == "vipseg"
    assert CascadeProtoConfig(encoder="density").encoder == "density"
    with pytest.raises(ValueError, match="encoder"):
        CascadeProtoConfig(encoder="pointnet")
    old = CascadeProtoConfig().to_dict()
    del old["encoder"]  # a checkpoint written before D-43
    assert CascadeProtoConfig(**old).encoder == "vipseg"


def test_de4_train_flags():
    import train

    base = ["--dataset", "s3dis", "--data_path", "x", "--cvfold", "1", "--n_way", "2", "--k_shot", "1",
            "--use_lma", "false", "--num_stages", "4", "--l2norm_point_proto", "true", "--stage_type", "vip_clean",
            "--batch_size", "1", "--lr_step_epochs", "15", "--valid_every", "4", "--seed", "0",
            "--query_order", "random", "--save_dir", "log_d43"]
    cr = train.parse_args(base)
    m1 = train.parse_args(base + ["--encoder", "density"])
    assert train.run_dir(cr).replace("\\", "/") == "log_d43/s3dis_S1_N2_K1_point_T4_vip_clean_b1_qrandom"
    assert train.run_dir(m1).replace("\\", "/") == "log_d43/s3dis_S1_N2_K1_point_T4_vip_clean_b1_qrandom_dens"
    assert train.model_config(m1).encoder == "density" and train.model_config(cr).encoder == "vipseg"
    old = {k: v for k, v in train.comparable_args(cr).items() if k != "encoder"}  # a checkpoint before D-43
    assert train.resume_mismatch(old, train.comparable_args(cr)) == []
    assert "encoder" in train.resume_mismatch(old, train.comparable_args(m1))


def interv(r_v0, r_own, a_v0, a_v2):
    return {"U": {"r_v0": r_v0, "r_v1": 0.8, "r_own": r_own, "r_a_v0": a_v0, "r_a_v2": a_v2, "phi": 1.0}}


def test_de5_mechanism_rule():
    cr = interv(0.245, 0.833, 0.837, 0.463)["U"]
    ok, _ = d43.mechanism(cr, interv(0.60, 0.85, 0.82, 0.70)["U"])  # deficits 0.25, 0.12: below half
    assert ok
    ok, _ = d43.mechanism(cr, interv(0.40, 0.85, 0.82, 0.70)["U"])  # other deficit 0.45 > 0.294
    assert not ok
    ok, _ = d43.mechanism(cr, interv(0.60, 0.85, 0.82, 0.50)["U"])  # uniform drop 0.32 > 0.187
    assert not ok


def fake_draws(t_cr=70.0, t_m1=72.0, t_lf_cr=40.0, t_lf_m1=42.0):
    def counts(t, seed):
        rng = np.random.default_rng(seed)
        c = np.zeros((60, 3, 3))
        c[:, 0, :] = c[:, 1, :] = 100.0
        c[:, 2, 1:] = np.clip(t + rng.normal(0, 2, (60, 2)), 0, 100)
        c[:, 2, 0] = 90.0
        return c

    draws = {}
    for i, d in enumerate(d43.DRAWS):
        tc, tm = (t_lf_cr, t_lf_m1) if d == "leakfree" else (t_cr, t_m1)
        cnt = {f"{n}:{a}": counts(t, i) for n, t in (("cr", tc), ("m1", tm)) for a in d43.ARMS}
        draws[d] = ({"miou": {k: float(d43.p0.miou_from_counts(v.sum(0))) for k, v in cnt.items()}}, cnt)
    return draws


def test_de6_rules():
    cr_i = interv(0.245, 0.833, 0.837, 0.463)
    good, bad = interv(0.60, 0.85, 0.82, 0.70), interv(0.30, 0.85, 0.82, 0.50)
    v = dict(d43.decide(fake_draws(), cr_i, good, None))
    assert "D43.1 mechanism holds" in v and "D43.2 leak-free holds" in v and "D43.3 M1 is the base on both protocols" in v
    v = dict(d43.decide(fake_draws(t_m1=68.0), cr_i, good, None))  # leak-free up, standard down
    assert any(k.startswith("D43.3 M1 is the base for the leak-free protocol") for k in v)
    v = dict(d43.decide(fake_draws(t_m1=70.3, t_lf_m1=40.3), cr_i, good, None))
    assert "D43.3 CR stays the base" in v and "D43.2 leak-free fails" in v
    draws = fake_draws(t_lf_m1=40.0)  # leak-free: a mean gain of about +1.8 with a CI across 0
    c = draws["leakfree"][1]["cr:lp"].copy()
    c[:30, 2, 1:] += 20.0
    c[30:, 2, 1:] -= 17.0
    draws["leakfree"][1]["m1:lp"] = c
    v = dict(d43.decide(draws, cr_i, good, None))
    assert "D43.2 leak-free fails" in v and "+1." in v["D43.2 leak-free fails"]
    v = dict(d43.decide(fake_draws(), cr_i, bad, None))
    assert "D43.1 mechanism fails" in v and any(k.startswith("D43.4 stop") for k in v)
    assert d43.decide({}, cr_i, good, None)[0][0] == "incomplete"
    assert d43.decide(fake_draws(), cr_i, None, None)[0][0] == "incomplete"


def test_de7_stack_predictions_are_p7s_composition():
    g = torch.Generator().manual_seed(2)
    f_q = torch.rand(2, 64, 8, generator=g, dtype=torch.float64)
    f_s = torch.rand(2, 1, 64, 8, generator=g, dtype=torch.float64)
    sy = (torch.rand(2, 1, 64, generator=g) < 0.3).long()
    xyz = torch.rand(2, 64, 3, generator=g, dtype=torch.float64)
    arm = dict(graph="feat", k=16, beta=0.99)
    out = d43.stack_predictions(f_q, f_s, sy, xyz, arm)
    labels = torch.zeros(2, 64, dtype=torch.long)
    preds, _, _ = p7.episode_pass(f_q, f_s, sy, xyz, labels, {"lp": arm}, None)
    for k in ("U", "both", "lp"):
        assert torch.equal(out[k], preds[k])
    assert torch.equal(out["U"], p6.rule_logits(f_q, p6.base_rows(f_q, f_s, sy)).argmax(-1))


def test_de8_files_parse_as_python_310():
    for f in ("models/density_ops.py", "models/density_encoder.py", "models/cascadeproto.py", "train.py",
              "experiments/d43_eval.py", "experiments/p8_condition_probe.py", "tests/test_density_encoder.py"):
        ast.parse((REPO / f).read_text(encoding="utf-8"), feature_version=(3, 10))


# ------------------------------------------------------------------ GPU (gate G2)

def encoders():
    from models.density_encoder import DensityEncoder
    from models.encoder import Encoder
    from models.vipseg_backbone import ENCODER_CONFIG, persist_fixed_projections

    torch.manual_seed(0)
    vip = Encoder(**ENCODER_CONFIG)
    torch.manual_seed(0)
    dens = DensityEncoder(**ENCODER_CONFIG)
    for e in (vip, dens):
        persist_fixed_projections(e)
    return vip.cuda().eval(), dens.cuda().eval()


def blocks(n=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    xyz = torch.rand(n, 2048, 3, generator=g) * torch.tensor([1.0, 1.0, 2.5])  # metres
    rgb = torch.rand(n, 2048, 3, generator=g)
    xyz_n = (xyz - xyz.min(1, keepdim=True).values) / (xyz - xyz.min(1, keepdim=True).values).max(1, keepdim=True).values
    return torch.cat([xyz, rgb, xyz_n], dim=-1).cuda()  # [n, 2048, 9]


@pytest.mark.cuda
def test_de9_same_parameters_as_vipseg():
    vip, dens = encoders()
    a, b = vip.state_dict(), dens.state_dict()
    assert list(a) == list(b) and all(a[k].shape == b[k].shape for k in a)


@pytest.mark.cuda
def test_de10_blocks_are_encoded_independently_only_by_the_density_encoder():
    """Block a encoded next to b and next to c: with the density encoder its features must not depend on the partner
    (measured 2026-09-26: exactly 0.0). Batch size 1 against 2 differs by rounding only (1.4e-4, equal to the change
    caused by a one-ULP input perturbation, 1.3e-4), so the partner, not the batch size, is compared."""
    vip, dens = encoders()
    x = blocks(3)
    ab, ac = x[[0, 1]], x[[0, 2]]
    with torch.no_grad(), torch.backends.cudnn.flags(enabled=True, benchmark=False, deterministic=True,
                                                     allow_tf32=False):
        d_rel = float((dens(ab)[:1] - dens(ac)[:1]).abs().max() / dens(ab)[:1].abs().max())
        v_rel = float((vip(ab)[:1] - vip(ac)[:1]).abs().max() / vip(ab)[:1].abs().max())
    print(f"DE-10 relative change of block a when its partner changes: density {d_rel:.2e}, VIP-Seg {v_rel:.2e}")
    assert d_rel < 1e-6
    assert v_rel > 1e-3  # VIP-Seg's batch-global statistics couple the blocks (D-42)


@pytest.mark.cuda
def test_de11_per_block_modules_equal_vipseg_on_one_block():
    """With one block the block's statistics are the batch's: each overridden forward must equal the inherited one."""
    from models.density_encoder import DensityDecoder, DensityDyHiConv, DensityLoConv
    from models.encoder import DynamicHighOrderConvolution, LowOrderConvolution, NonParametricDecoder

    g = torch.Generator().manual_seed(3)
    vv, ww = torch.randn(1, 5000, generator=g), torch.randn(1, 5000, generator=g)
    knn_xyz = torch.randn(1, 3, 64, 16, generator=g).cuda()
    knn_x = torch.randn(1, 120, 64, 16, generator=g).cuda()
    knn_rgb = torch.rand(1, 3, 64, 16, generator=g).cuda()
    lc_xyz = torch.rand(1, 64, 3, generator=g).cuda()
    with torch.no_grad():
        torch.manual_seed(0)
        lo = LowOrderConvolution(3, 120, 1, 1, vv).cuda().eval()
        dlo = DensityLoConv(3, 120, 1, 1, vv).cuda().eval()
        dlo.load_state_dict(lo.state_dict())
        assert torch.allclose(dlo(knn_xyz, knn_x, knn_rgb), lo(knn_xyz, knn_x, knn_rgb), atol=1e-5)
        hi = DynamicHighOrderConvolution(3, 120, 1, 1, ww, 3).cuda().eval()
        dhi = DensityDyHiConv(3, 120, 1, 1, ww, 3).cuda().eval()
        dhi.load_state_dict(hi.state_dict())
        assert torch.allclose(dhi(lc_xyz, knn_xyz, knn_x, knn_rgb), hi(lc_xyz, knn_xyz, knn_x, knn_rgb), atol=1e-5)
        xyz1, xyz2 = torch.rand(1, 256, 3, generator=g).cuda(), torch.rand(1, 64, 3, generator=g).cuda()
        p1, p2 = torch.randn(1, 30, 256, generator=g).cuda(), torch.randn(1, 60, 64, generator=g).cuda()
        dec, ddec = NonParametricDecoder(3, 10), DensityDecoder(3, 10)
        assert torch.allclose(ddec.propagate(xyz1, xyz2, p1, p2), dec.propagate(xyz1, xyz2, p1, p2), atol=1e-5)
