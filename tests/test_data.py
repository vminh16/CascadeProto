"""Gate G4 (05 §3.10): the inherited loader on real preprocessed S3DIS blocks.

Data path: $CASCADEPROTO_S3DIS, default datasets/S3DIS/blocks_bs1_s1 (layout of 04 §2.2).
"""

import glob
import os

import numpy as np
import pytest

from pipeline.episodes import (IN_CHANNELS, NUM_POINT, build_eval_dataset, build_train_dataset, make_episode,
                               read_class_names)

pytestmark = pytest.mark.data

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.environ.get("CASCADEPROTO_S3DIS", os.path.join(REPO, "datasets", "S3DIS", "blocks_bs1_s1"))
S0_TEST = ["beam", "board", "bookcase", "ceiling", "chair", "column"]  # [VIPSEG dataloaders/s3dis.py:20]


@pytest.fixture(scope="module")
def class_names():
    if not os.path.isdir(os.path.join(DATA_PATH, "data")):
        pytest.fail(f"no preprocessed blocks at {DATA_PATH}; run preprocess/prepare_s3dis.py")
    return read_class_names(DATA_PATH, "s3dis")


def test_data0_every_block_loads(class_names):
    """All 272 rooms [PAPER §4.1] have blocks, and every block is an [n >= 1000, 7] array."""
    blocks = sorted(glob.glob(os.path.join(DATA_PATH, "data", "*.npy")))
    rooms = {os.path.basename(p).rsplit("_block_", 1)[0] for p in blocks}
    assert len(rooms) == 272
    bad = []
    for path in blocks:
        try:
            array = np.load(path, mmap_mode="r")
            if array.ndim != 2 or array.shape[1] != 7 or array.shape[0] < 1000:
                bad.append(path)
        except (OSError, ValueError, EOFError):
            bad.append(path)
    assert not bad, f"{len(bad)} unreadable blocks, e.g. {bad[:3]}; rerun preprocess/prepare_s3dis.py"


def test_data1_class_names_file(class_names):
    assert class_names == ["ceiling", "floor", "wall", "beam", "column", "window", "door",
                           "table", "chair", "sofa", "bookcase", "board", "clutter"]


@pytest.mark.parametrize("k_shot", [1, 5])
def test_data2_training_episode_layout(class_names, k_shot):
    dataset = build_train_dataset(DATA_PATH, "s3dis", cvfold=0, n_way=2, k_shot=k_shot, num_episode=3)
    for i in range(3):
        ep = make_episode(dataset[i], class_names)  # validates 02 §1 shapes, binary masks, labels
        assert ep.support_x.shape == (2, k_shot, NUM_POINT, IN_CHANNELS)
        xyz_norm = ep.support_x[..., 6:9]  # XYZ columns
        assert xyz_norm.min() >= 0 and xyz_norm.max() <= 1 + 1e-6
        assert (ep.support_y.sum(dim=-1) > 0).all(), "every support block contains its class"
        assert set(ep.query_y.unique().tolist()) <= {0, 1, 2}
        assert not set(ep.class_names) & set(S0_TEST)


def test_data3_fold0_classes(class_names):
    dataset = build_train_dataset(DATA_PATH, "s3dis", cvfold=0, n_way=2, k_shot=1, num_episode=1)
    train = {class_names[c] for c in dataset.classes}
    assert train == {"floor", "wall", "window", "door", "table", "sofa"}
    assert "clutter" not in train and not train & set(S0_TEST)


def test_data4_fixed_test_episodes(class_names):
    dataset = build_eval_dataset(DATA_PATH, "s3dis", cvfold=0, n_way=2, k_shot=1, mode="test", seed=0)
    assert len(dataset) == 15 * 100  # C(6,2) combinations x 100 [DECISION D-08]
    assert sorted(class_names[c] for c in dataset.classes) == sorted(S0_TEST)
    ep = make_episode(dataset[0], class_names)
    assert set(ep.class_names) <= set(S0_TEST)
    assert np.isin(ep.sampled_classes, dataset.classes).all()
