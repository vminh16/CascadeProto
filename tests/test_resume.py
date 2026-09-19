"""RES-1..5 (05 §3.8b, gate G1): seeded training episodes and exact resumption of an interrupted run.

The training loop of train.py runs on the CPU with the stand-in encoder and a stand-in loader that,
like the inherited one, draws from the global `np.random` and `random`.
"""

import os
import random

import numpy as np
import pytest
import torch

import train
from models.cascadeproto import CascadeProto
from models.clip_text import ClipTextEmbedding
from models.vipseg_backbone import PointFeatureExtractor
from pipeline.episodes import SeededEpisodes, make_episode
from tests.test_clip_text import RecordingEncoder
from tests.test_feature_extractor import StandInEncoder

CLASS_NAMES = ["ceiling", "floor", "wall", "beam", "column", "window", "door",
               "table", "chair", "sofa", "bookcase", "board", "clutter"]


class GlobalRandomLoader:
    """Stand-in for MyDataset: every item is drawn from the global np.random and random."""

    classes = np.array([1, 2, 5, 6, 7, 9])

    def __init__(self, n):
        self.n = n
        self.requested = []  # indices read, in order

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        self.requested.append(i)
        classes = np.random.choice(self.classes, 2, replace=False)
        support_x = np.random.random((2, 1, 2048, 9)).astype(np.float32)
        support_x[..., 3:6] += random.random()  # like the loader's `random`-based augmentation
        support_y = (np.random.random((2, 1, 2048)) < 0.4).astype(np.int32)
        query_x = np.random.random((2, 2048, 9)).astype(np.float32)
        query_y = np.random.randint(0, 3, (2, 2048))
        return support_x, support_y, query_x, query_y, classes


def args_for(tmp_path, name, **kw):
    argv = ["--dataset", "s3dis", "--data_path", "x", "--cvfold", "0", "--n_way", "2", "--k_shot", "1",
            "--epochs", "3", "--episodes_per_epoch", "8", "--valid_every", "2", "--num_workers", "0",
            "--save_dir", str(tmp_path / name)]
    for k, v in kw.items():
        argv += [f"--{k}", str(v)]
    return train.parse_args(argv)


def build(args, init_seed):
    torch.manual_seed(init_seed)
    config = train.model_config(args)
    m = CascadeProto(config, PointFeatureExtractor(encoder=StandInEncoder()),
                     text_embedding=ClipTextEmbedding(encode=RecordingEncoder()))
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # step 2 < 3 epochs: the decay falls after epoch 2, i.e. after the interruption at epoch 1
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=2, gamma=args.lr_gamma)
    return m, opt, sched, config


class _Log:
    def __init__(self):
        self.lines = []

    def cprint(self, text):
        self.lines.append(text)


def validate(m):
    """A deterministic function of the weights, so best.pt selection is exercised."""
    return float(sum(p.double().sum() for p in m.parameters()).sin())


def run(args, init_seed, stop_after_epoch=None):
    """One call of train_loop with a fixed global seed, as train.main does."""
    train.seed_everything(args.seed)
    out_dir = train.run_dir(args)
    os.makedirs(out_dir, exist_ok=True)
    m, opt, sched, config = build(args, init_seed)
    train.seed_everything(args.seed)
    loader = GlobalRandomLoader(args.epochs * args.episodes_per_epoch)
    log = _Log()
    log.requested = loader.requested
    state = train.train_loop(args, m, opt, sched, SeededEpisodes(loader, args.seed), CLASS_NAMES, validate, config,
                             out_dir, log, torch.device("cpu"), stop_after_epoch=stop_after_epoch)
    return m, opt, state, out_dir, log


def ckpt(out_dir, name):
    return torch.load(os.path.join(out_dir, name), map_location="cpu", weights_only=False)


# ------------------------------------------------------------------------------ seeded episodes

def test_res1_episode_i_depends_only_on_seed_and_index():
    data = SeededEpisodes(GlobalRandomLoader(10), seed=3)
    forward = [data[i] for i in range(10)]
    np.random.seed(123)
    random.seed(456)
    backward = [data[i] for i in reversed(range(10))][::-1]
    for a, b in zip(forward, backward):
        assert all(np.array_equal(x, y) for x, y in zip(a, b))
    other = SeededEpisodes(GlobalRandomLoader(10), seed=4)
    assert not np.array_equal(other[0][0], forward[0][0])
    assert not np.array_equal(forward[1][3], forward[0][3])  # query labels come from numpy alone


def test_res1_caller_random_state_is_restored():
    data = SeededEpisodes(GlobalRandomLoader(3), seed=0)
    np.random.seed(7)
    random.seed(8)
    expected = (np.random.random(), random.random())
    np.random.seed(7)
    random.seed(8)
    data[1]
    assert (np.random.random(), random.random()) == expected


def test_res1_training_episodes_are_valid_episodes():
    ep = make_episode(SeededEpisodes(GlobalRandomLoader(2), seed=0)[1], CLASS_NAMES)
    assert ep.support_x.shape == (2, 1, 2048, 9)


# ------------------------------------------------------------------------------ resumption

def test_res2_interrupted_and_resumed_run_equals_uninterrupted_run(tmp_path):
    """LMA row: dropout and the generator noise draw from the torch RNG, the loader from numpy/random."""
    kw = dict(use_lma="true", num_stages=0)
    full, full_opt, full_state, full_dir, _ = run(args_for(tmp_path, "a", **kw), init_seed=0)
    run(args_for(tmp_path, "b", **kw), init_seed=0, stop_after_epoch=1)
    resumed, res_opt, res_state, res_dir, log = run(args_for(tmp_path, "b", resume="true", **kw), init_seed=99)
    assert any("resumed" in line and "after epoch 1" in line for line in log.lines)
    assert log.requested == list(range(8, 24))  # epochs 2-3 read episodes 8..23, each once, in order
    assert res_opt.param_groups[0]["lr"] == full_opt.param_groups[0]["lr"] == 5e-4  # one decay, after epoch 2
    assert res_state == full_state and full_state["epoch"] == 3
    for name in ("last.pt", "best.pt"):
        a, b = ckpt(full_dir, name), ckpt(res_dir, name)
        assert a["epoch"] == b["epoch"]
        assert all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"])
    a, b = full_opt.state_dict()["state"], res_opt.state_dict()["state"]
    assert all(torch.equal(a[i][k], b[i][k]) for i in a for k in ("exp_avg", "exp_avg_sq"))


def test_res3_resume_refuses_different_arguments(tmp_path):
    kw = dict(use_lma="false", num_stages=0)
    run(args_for(tmp_path, "c", **kw), init_seed=0, stop_after_epoch=1)
    with pytest.raises(ValueError, match="lr"):
        run(args_for(tmp_path, "c", resume="true", lr=5e-4, **kw), init_seed=0)
    run(args_for(tmp_path, "c", resume="true", num_workers=0, **kw), init_seed=0)  # free argument: fine


def test_res4_resume_without_checkpoint_starts_fresh_and_finished_run_is_a_no_op(tmp_path):
    kw = dict(use_lma="false", num_stages=0)
    fresh, _, state, out_dir, log = run(args_for(tmp_path, "d", resume="true", **kw), init_seed=0)
    assert state["epoch"] == 3 and not any("resumed" in line for line in log.lines)
    assert log.requested == list(range(24))  # epoch e reads episodes 8e..8e+7
    before = ckpt(out_dir, "last.pt")["model"]
    _, _, state2, _, log2 = run(args_for(tmp_path, "d", resume="true", **kw), init_seed=5)
    assert state2 == state and not any(line.startswith("epoch ") for line in log2.lines)
    after = ckpt(out_dir, "last.pt")["model"]
    assert all(torch.equal(before[k], after[k]) for k in before)


def test_res5_checkpoints_are_complete_and_atomic(tmp_path):
    kw = dict(use_lma="false", num_stages=0)
    _, _, _, out_dir, _ = run(args_for(tmp_path, "e", **kw), init_seed=0, stop_after_epoch=2)
    r = ckpt(out_dir, "resume.pt")
    assert set(r) == {"model", "optimizer", "scheduler", "state", "random", "config", "args"}
    assert r["state"]["epoch"] == 2 and set(r["random"]) == {"torch", "numpy", "python", "cuda"}
    assert r["scheduler"]["last_epoch"] == 2
    assert os.path.isfile(os.path.join(out_dir, "best.pt"))  # validated at epoch 2
    assert not [f for f in os.listdir(out_dir) if f.endswith(".tmp")]
