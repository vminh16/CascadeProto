"""Episode construction on top of the inherited VIP-Seg loader (spec 04 §2-§4, spec 02 §1).

This module is the only place that builds `MyDataset` / `MyTestDataset`, so every run uses the
loader arguments of 04 §4.1 instead of the loader's own defaults (4096 points, 'xyz').
"""

import math
import os
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import torch

from dataloaders.loader import MyDataset, MyTestDataset

# 04 §4.1 [VIPSEG scripts/vipseg_s3dis.sh] [VIPSEG main.py:48-69]
NUM_POINT = 2048
PC_ATTRIBS = "xyzrgbXYZ"
IN_CHANNELS = len(PC_ATTRIBS)
WAY_RATIO = [0.05, 0.05]
WAY_NUM = [100, 100]
N_QUERIES = 1
RANDOM_SAMPLE = False
# Training augmentation [DECISION D-12] [VIPSEG main.py:56-66] [VIPSEG scripts/vipseg_s3dis.sh]
AUGMENT_CONFIG = {"scale": 0, "rot": 1, "mirror_prob": 0, "jitter": 1, "shift": 0.1, "random_color": 0}

# Fixed evaluation episodes are cached under <data_path>/<tag>_S_<fold>_N_<N>_K_<K>_[test_]episodes_...
# The tags are VIP-Seg's own [VIPSEG scripts/vipseg_eval_s3dis.sh] [VIPSEG runs/training.py:52-66], so
# the VIP-Seg sanity run (05 §4) and CascadeProto are scored on the same cached test episodes.
TEST_EPISODE_TAG = "vipseg_eval"
VALID_EPISODE_TAG = "vipseg"
N_EPISODES_PER_COMBINATION = 100  # 04 §6.1 [DECISION D-08]

# 04 §5 [PAPER §4.1] [DECISION D-12]
SCHEDULE = {
    "s3dis": {"epochs": 50, "episodes_per_epoch": 480},
    "scannet": {"epochs": 30, "episodes_per_epoch": 800},
}
EPISODES_PER_BATCH = 4


@dataclass
class Episode:
    """One N-way K-shot episode in the layout of spec 02 §1."""

    support_x: torch.Tensor  # [N, K, 2048, 9] float32
    support_y: torch.Tensor  # [N, K, 2048] int64, binary {0, 1}
    query_x: torch.Tensor  # [B_q, 2048, 9] float32
    query_y: torch.Tensor  # [B_q, 2048] int64, {0..N}
    sampled_classes: np.ndarray  # [N] global class ids; local label k+1 = sampled_classes[k]
    class_names: List[str]  # [N] names of sampled_classes, used for the CLIP prompts (03)

    @property
    def n_way(self) -> int:
        return self.support_x.shape[0]

    @property
    def k_shot(self) -> int:
        return self.support_x.shape[1]

    def to(self, device) -> "Episode":
        return Episode(
            support_x=self.support_x.to(device),
            support_y=self.support_y.to(device),
            query_x=self.query_x.to(device),
            query_y=self.query_y.to(device),
            sampled_classes=self.sampled_classes,
            class_names=self.class_names,
        )


def read_class_names(data_path: str, dataset: str) -> List[str]:
    """Class names in id order, from the file the loader itself reads [VIPSEG dataloaders/s3dis.py:15]."""
    path = os.path.join(os.path.dirname(os.path.normpath(data_path)), "meta", f"{dataset}_classnames.txt")
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def make_episode(item: Sequence, class_names: Sequence[str]) -> Episode:
    """Build an Episode from one loader item (MyDataset.__getitem__ or read_episode).

    The item is (support_x [N, K, 2048, 9], support_y [N, K, 2048], query_x [N*n_q, 2048, 9],
    query_y [N*n_q, 2048], sampled_classes [N]) as numpy arrays [VIPSEG dataloaders/loader.py:157-161].
    """
    support_x, support_y, query_x, query_y, sampled_classes = item
    n_way, k_shot = support_y.shape[:2]
    if support_x.shape != (n_way, k_shot, NUM_POINT, IN_CHANNELS):
        raise ValueError(f"support_x {support_x.shape} != {(n_way, k_shot, NUM_POINT, IN_CHANNELS)}")
    if query_x.shape[1:] != (NUM_POINT, IN_CHANNELS) or query_y.shape != query_x.shape[:2]:
        raise ValueError(f"query_x {query_x.shape} / query_y {query_y.shape} do not match 02 §1")
    if not np.isin(support_y, (0, 1)).all():
        raise ValueError("support masks must be binary (02 §1)")
    if query_y.min() < 0 or query_y.max() > n_way:
        raise ValueError(f"query labels outside 0..{n_way}")
    sampled_classes = np.asarray(sampled_classes).astype(np.int64)
    return Episode(
        support_x=torch.from_numpy(np.ascontiguousarray(support_x, dtype=np.float32)),  # [N, K, 2048, 9]
        support_y=torch.from_numpy(support_y.astype(np.int64)),  # [N, K, 2048]
        query_x=torch.from_numpy(np.ascontiguousarray(query_x, dtype=np.float32)),  # [B_q, 2048, 9]
        query_y=torch.from_numpy(query_y.astype(np.int64)),  # [B_q, 2048]
        sampled_classes=sampled_classes,
        class_names=[class_names[c] for c in sampled_classes],
    )


class EpisodeCollate:
    """DataLoader collate_fn: a list of loader items becomes a list of Episodes (one per item).

    A training batch of 4 episodes is 4 loader items, each forwarded separately (04 §4.3).
    """

    def __init__(self, class_names: Sequence[str]):
        self.class_names = list(class_names)

    def __call__(self, batch) -> List[Episode]:
        return [make_episode(item, self.class_names) for item in batch]


def build_train_dataset(data_path: str, dataset: str, cvfold: int, n_way: int, k_shot: int,
                        num_episode: int) -> MyDataset:
    """Random training episodes over the fold's training classes, with D-12 augmentation."""
    return MyDataset(data_path, dataset, cvfold=cvfold, num_episode=num_episode,
                     n_way=n_way, k_shot=k_shot, n_queries=N_QUERIES, phase=None, mode="train",
                     num_point=NUM_POINT, pc_attribs=PC_ATTRIBS,
                     pc_augm=True, pc_augm_config=AUGMENT_CONFIG,
                     way_ratio=WAY_RATIO, way_num=WAY_NUM, random_sample=RANDOM_SAMPLE)


def build_eval_dataset(data_path: str, dataset: str, cvfold: int, n_way: int, k_shot: int,
                       mode: str, seed: int) -> MyTestDataset:
    """Fixed episodes over the fold's test classes: mode 'test' (reported) or 'valid' (D-15 selection).

    Episodes are generated once and cached as .h5; `seed` only matters for that first generation.
    """
    if mode not in ("test", "valid"):
        raise ValueError(f"mode must be 'test' or 'valid', got {mode!r}")
    tag = TEST_EPISODE_TAG if mode == "test" else VALID_EPISODE_TAG
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        episodes = MyTestDataset(tag, data_path, dataset, cvfold=cvfold,
                                 num_episode_per_comb=N_EPISODES_PER_COMBINATION,
                                 n_way=n_way, k_shot=k_shot, n_queries=N_QUERIES,
                                 num_point=NUM_POINT, pc_attribs=PC_ATTRIBS,
                                 way_ratio=WAY_RATIO, way_num=WAY_NUM, mode=mode)
    finally:
        np.random.set_state(state)
    # The loader trusts any existing cache folder [VIPSEG dataloaders/loader.py:239-241]; a build that
    # crashed half-way would silently shrink the test set, so check the count of 04 §6.1.
    expected = math.comb(len(episodes.classes), n_way) * N_EPISODES_PER_COMBINATION
    if len(episodes) != expected:
        raise RuntimeError(f"{len(episodes)} cached {mode} episodes, expected {expected}; delete the "
                           f"'{tag}_S_{cvfold}_N_{n_way}_K_{k_shot}_*' folder in {data_path} and rerun")
    return episodes
