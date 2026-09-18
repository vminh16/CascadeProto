"""PROTO-1..10 (05 §3.2): point prototypes of Eq.3 / spec 02 §3.

All checks run in float64 and compare against independent computations to 1e-12, so any deviation
from the formula (not just a wrong shape) fails.
"""

import pytest
import torch

from models.prototypes import EMPTY_BACKGROUND_VALUE, point_prototypes

ATOL = 1e-12
N, K, P, D = 3, 2, 2048, 128


def fixture(n=N, k=K, p=P, d=D, seed=0):
    """Random features and binary masks whose foreground fraction differs per way."""
    g = torch.Generator().manual_seed(seed)
    feat = torch.randn(n, k, p, d, generator=g, dtype=torch.float64)  # [N, K, P, D]
    frac = torch.linspace(0.1, 0.6, n, dtype=torch.float64)[:, None, None]  # [N, 1, 1]
    mask = (torch.rand(n, k, p, generator=g, dtype=torch.float64) < frac).long()  # [N, K, P]
    return feat, mask


def vipseg_reference(feat, mask):
    """Line-by-line port of VIP-Seg's prototype loop [VIPSEG models/vipseg.py:108-130], without its
    later L2 normalisation (excluded by D-10)."""
    n_way = feat.shape[0]
    rows = []
    bg_features = feat[mask == 0]
    if bg_features.shape[0] < 1:
        rows.append(torch.ones(1, feat.shape[-1], dtype=feat.dtype) * 0.1)
    else:
        rows.append(bg_features.mean(0).unsqueeze(0))
    for i in range(n_way):
        rows.append(feat[i, mask[i] == 1].mean(0).unsqueeze(0))
    return torch.cat(rows, dim=0)


def test_proto1_foreground_rows_are_per_way_masked_means():
    feat, mask = fixture()
    proto = point_prototypes(feat, mask)
    assert proto.shape == (N + 1, D) and proto.dtype == torch.float64
    for k in range(N):
        points = feat[k][mask[k] == 1]  # [n_k, D], all K shots of way k
        manual = points.sum(0) / points.shape[0]
        assert torch.allclose(proto[k + 1], manual, atol=ATOL, rtol=0)


def test_proto2_background_row_pools_all_ways_and_shots():
    feat, mask = fixture()
    proto = point_prototypes(feat, mask)
    points = feat[mask == 0]  # [n_bg, D] over every way and shot
    assert torch.allclose(proto[0], points.sum(0) / points.shape[0], atol=ATOL, rtol=0)


def test_proto3_no_zero_row_and_distinct_rows():
    feat, mask = fixture()
    proto = point_prototypes(feat, mask)
    assert (proto.abs().sum(dim=1) > 0).all()
    for i in range(N + 1):
        for j in range(i + 1, N + 1):
            assert not torch.allclose(proto[i], proto[j]), (i, j)


def test_proto4_changing_one_way_leaves_other_ways_unchanged():
    feat, mask = fixture()
    before = point_prototypes(feat, mask)
    mask2 = mask.clone()
    mask2[1] = 1 - mask2[1]  # flip way 2 (row 2)
    after = point_prototypes(feat, mask2)
    assert torch.equal(before[1], after[1]) and torch.equal(before[3], after[3])
    assert not torch.allclose(before[2], after[2])
    assert not torch.allclose(before[0], after[0])  # background pools way 2's points too


def test_proto5_equals_vipseg_loop():
    for seed in range(5):
        feat, mask = fixture(seed=seed)
        assert torch.allclose(point_prototypes(feat, mask), vipseg_reference(feat, mask), atol=ATOL, rtol=0)


def test_proto6_invariances_and_way_equivariance():
    feat, mask = fixture()
    proto = point_prototypes(feat, mask)
    g = torch.Generator().manual_seed(1)
    # point order inside each block (the loader shuffles it [VIPSEG dataloaders/loader.py:58])
    perm = torch.randperm(P, generator=g)
    assert torch.allclose(point_prototypes(feat[:, :, perm], mask[:, :, perm]), proto, atol=ATOL, rtol=0)
    # shot order inside each way
    shots = torch.tensor([1, 0])
    assert torch.allclose(point_prototypes(feat[:, shots], mask[:, shots]), proto, atol=ATOL, rtol=0)
    # way order permutes the foreground rows and keeps the background row
    ways = torch.tensor([2, 0, 1])
    permuted = point_prototypes(feat[ways], mask[ways])
    assert torch.allclose(permuted[0], proto[0], atol=ATOL, rtol=0)
    assert torch.allclose(permuted[1:], proto[1:][ways], atol=ATOL, rtol=0)


def test_proto7_affine_equivariance():
    """Every row is a mean, so P(a·F + b) = a·P(F) + b."""
    feat, mask = fixture()
    a, b = 2.5, torch.randn(D, dtype=torch.float64)
    assert torch.allclose(point_prototypes(a * feat + b, mask), a * point_prototypes(feat, mask) + b,
                          atol=1e-10, rtol=0)


def test_proto8_gradient_is_mask_over_count():
    feat, mask = fixture(n=2, k=2, p=64, d=8)
    feat.requires_grad_(True)
    for row in range(3):
        grad, = torch.autograd.grad(point_prototypes(feat, mask)[row].sum(), feat)  # [N, K, P, D]
        if row == 0:
            weight = (mask == 0).double() / (mask == 0).sum()
        else:
            weight = torch.zeros_like(mask, dtype=torch.float64)
            weight[row - 1] = (mask[row - 1] == 1).double() / (mask[row - 1] == 1).sum()
        assert torch.allclose(grad, weight[..., None].expand_as(grad), atol=ATOL, rtol=0)


def test_proto9_empty_masks():
    feat, mask = fixture(n=2, k=1, p=32, d=4)
    all_fg = torch.ones_like(mask)  # no background point anywhere
    proto = point_prototypes(feat, all_fg)
    assert torch.equal(proto[0], torch.full((4,), EMPTY_BACKGROUND_VALUE, dtype=torch.float64))
    assert torch.allclose(proto, vipseg_reference(feat, all_fg), atol=ATOL, rtol=0)
    no_fg = mask.clone()
    no_fg[1] = 0
    with pytest.raises(ValueError, match=r"ways \[1\]"):
        point_prototypes(feat, no_fg)


@pytest.mark.parametrize("bad_mask", [
    lambda m: m + 1,  # values {1, 2}: the old convention that hid audit C2
    lambda m: m[..., :-1],  # wrong point count
    lambda m: m[0],  # missing way axis
])
def test_proto10_rejects_invalid_input(bad_mask):
    feat, mask = fixture(n=2, k=2, p=16, d=4)
    with pytest.raises(ValueError):
        point_prototypes(feat, bad_mask(mask))
    with pytest.raises(ValueError):
        point_prototypes(feat.reshape(4, 16, 4), mask)  # flattened ways and shots


def test_proto10_float32_matches_float64():
    feat, mask = fixture()
    p64 = point_prototypes(feat, mask)
    p32 = point_prototypes(feat.float(), mask)
    assert p32.dtype == torch.float32
    assert torch.allclose(p32.double(), p64, atol=1e-5, rtol=0)
