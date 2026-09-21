"""Q14-1..5 (05 §3.11, gate G1): the phase-14 run queue maps the paper's tables to runs correctly."""

import os

import pytest

import train
from experiments import phase14
from models.cascadeproto import CascadeProtoConfig

EXPECTED = {  # Table 4 rows of D-17 as configurations
    "baseline": CascadeProtoConfig(use_lma=False, num_stages=0),
    "lma": CascadeProtoConfig(use_lma=True, num_stages=0),
    "gate": CascadeProtoConfig(use_lma=True, num_stages=1),
    "cascade": CascadeProtoConfig(use_lma=True, num_stages=4, use_adrm=False),
    "full": CascadeProtoConfig(),
}


def config_of(run):
    return train.model_config(train.parse_args(phase14.train_argv(run, "d", "s")))


def test_q1_queue_sizes_and_unique_run_dirs(tmp_path):
    runs = phase14.all_runs()
    count = {p: sum(r.priority == p for r in runs) for p in phase14.PRIORITIES}
    assert count == {"P1": 2, "P2": 8, "P2s": 2, "P3": 6, "P4": 8}
    dirs = [phase14.run_dir(r, str(tmp_path)) for r in runs]
    assert len(set(dirs)) == len(runs) == 26
    assert len({r.name for r in runs}) == 26


def test_q2_table4_rows_are_the_d17_configurations_on_both_folds():
    runs = phase14.all_runs()
    for row, expected in EXPECTED.items():
        cells = [r for r in runs if ("T4", row) in r.tables]
        assert sorted(r.cvfold for r in cells) == [0, 1], row
        for r in cells:
            assert config_of(r) == expected and (r.n_way, r.k_shot, r.seed) == (2, 1, 0)


def test_q3_table5_depths_and_table2_settings_are_covered():
    runs = phase14.all_runs()
    for t in range(1, 7):
        cells = [r for r in runs if ("T5", f"T{t}") in r.tables]
        assert sorted(r.cvfold for r in cells) == [0, 1], t
        for r in cells:
            c = config_of(r)
            assert c.num_stages == t and c.use_lma and c.use_gate and (r.n_way, r.k_shot) == (2, 1)
    for n, k in ((2, 1), (2, 5), (3, 1), (3, 5)):
        cells = [r for r in runs if ("T2", f"N{n}K{k}") in r.tables]
        assert sorted(r.cvfold for r in cells) == [0, 1]
        assert all(config_of(r) == EXPECTED["full"] and (r.n_way, r.k_shot) == (n, k) for r in cells)
    seeds = [r for r in runs if r.priority == "P2s"]
    assert sorted(r.seed for r in seeds) == [1, 2] and all(config_of(r) == EXPECTED["full"] for r in seeds)


def test_q4_pending_steps_skip_finished_outputs(tmp_path):
    run = phase14.all_runs()[0]  # P1 full S0, with random600
    save = str(tmp_path)
    labels = [label for label, _ in phase14.pending_steps(run, "d", save)]
    assert labels == [f"{run.name}: train"] + [f"{run.name}: eval {w} {p}" for w in ("best", "last")
                                                for p in ("fixed100", "random600")]
    out = phase14.run_dir(run, save)
    os.makedirs(out)
    open(os.path.join(out, "last.pt"), "w").close()
    open(phase14.result_path(run, save, "best", "fixed100"), "w").close()
    labels = [label for label, _ in phase14.pending_steps(run, "d", save)]
    assert labels == [f"{run.name}: eval best random600", f"{run.name}: eval last fixed100",
                      f"{run.name}: eval last random600"]


def test_q5_commands_resume_training_and_write_json_results(tmp_path):
    run = [r for r in phase14.all_runs() if r.name == "cascade_S1_N2K1"][0]
    steps = dict(phase14.pending_steps(run, "d", str(tmp_path), python="py"))
    train_cmd = steps["cascade_S1_N2K1: train"]
    assert train_cmd[0] == "py" and train_cmd[1].endswith("train.py")
    assert train_cmd[train_cmd.index("--resume") + 1] == "true"
    ev = steps["cascade_S1_N2K1: eval last fixed100"]
    assert ev[ev.index("--checkpoint") + 1].endswith(os.path.join("s3dis_S1_N2_K1_text_T4_noadrm", "last.pt"))
    assert ev[ev.index("--result_json") + 1].endswith("eval_last_fixed100.json")
    assert not any("random600" in label for label in steps)  # only P1 runs use the second protocol


def test_q5_seed_is_part_of_the_run_dir(tmp_path):
    seeds = [r for r in phase14.all_runs() if r.name.startswith("full_S0_N2K1")]
    dirs = {phase14.run_dir(r, str(tmp_path)) for r in seeds}
    assert len(dirs) == 3 and any(d.endswith("_seed2") for d in dirs)


def test_init_from_vipseg_gets_its_own_run_directory():
    """D-20: a VIP-Seg-initialised run must never resume from, or overwrite, a from-scratch run."""
    import train
    base = ["--dataset", "s3dis", "--data_path", "x", "--cvfold", "0", "--n_way", "2", "--k_shot", "1"]
    scratch = train.run_dir(train.parse_args(base))
    init = train.run_dir(train.parse_args(base + ["--init_from_vipseg", "vipseg_S0_N2_K1.pt"]))
    assert init == scratch + "_vipinit"
    assert train.parse_args(base).init_from_vipseg is None


def test_train_classes_all_widens_only_the_sampled_classes(monkeypatch):
    """D-21: the leakage diagnostic adds the fold's test classes to what training samples."""
    import types
    import numpy as np
    import pytest
    from pipeline import episodes

    class FakeMyDataset:
        def __init__(self, *a, **k):
            self.dataset = types.SimpleNamespace(train_classes=[1, 2, 5, 6, 7, 9], test_classes=[3, 11, 10, 0, 8, 4])
            self.classes = np.array(self.dataset.train_classes)

    monkeypatch.setattr(episodes, "MyDataset", FakeMyDataset)
    split = episodes.build_train_dataset("x", "s3dis", 0, 2, 1, num_episode=4)
    leak = episodes.build_train_dataset("x", "s3dis", 0, 2, 1, num_episode=4, train_classes="all")
    assert list(split.classes) == [1, 2, 5, 6, 7, 9]
    assert list(leak.classes) == list(range(12))  # all 12 classes, clutter excluded as in the loader
    with pytest.raises(ValueError):
        episodes.build_train_dataset("x", "s3dis", 0, 2, 1, num_episode=4, train_classes="some")


def test_train_classes_all_gets_its_own_run_directory():
    import train
    base = ["--dataset", "s3dis", "--data_path", "x", "--cvfold", "0", "--n_way", "2", "--k_shot", "1"]
    assert train.run_dir(train.parse_args(base + ["--train_classes", "all"])) == train.run_dir(train.parse_args(base)) + "_leak"


def test_batch_size_option_and_run_directory():
    """D-12 probe: batch 1 reproduces VIP-Seg's 24,000-step schedule and must not share a run directory."""
    import pytest
    import train
    base = ["--dataset", "s3dis", "--data_path", "x", "--cvfold", "0", "--n_way", "2", "--k_shot", "1"]
    assert train.parse_args(base).batch_size == 4
    b1 = train.parse_args(base + ["--batch_size", "1"])
    assert b1.batch_size == 1 and train.run_dir(b1) == train.run_dir(train.parse_args(base)) + "_b1"
    with pytest.raises(SystemExit):
        train.parse_args(base + ["--batch_size", "7"])  # 480 episodes per epoch is not a multiple of 7


def test_resume_accepts_flags_added_after_the_checkpoint_at_their_default():
    """A checkpoint written before --batch_size existed resumes at the default, and only there."""
    import train

    base = ["--dataset", "s3dis", "--data_path", "d", "--cvfold", "0", "--n_way", "2", "--k_shot", "1"]
    current = train.comparable_args(train.parse_args(base))
    saved = {k: v for k, v in current.items() if k != "batch_size"}
    assert train.resume_mismatch(saved, current) == []
    changed = train.comparable_args(train.parse_args(base + ["--batch_size", "1"]))
    assert train.resume_mismatch(saved, changed) == ["batch_size"]
    assert train.resume_mismatch(dict(saved, seed=1), current) == ["seed"]
