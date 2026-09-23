"""R1-1..5 (05 §3.8f, gate G1): the phase-16 R1 queue, its variants and its summary.

CPU only: the variants are checked as configurations, not by building the encoder (which needs
`pointnet2_ops` and `mamba_ssm`, gate G2).
"""

import os

import pytest

import train
from experiments import summarize_r1
from experiments.diag_short import VARIANTS

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R1 = {  # variant -> the configuration it must produce (16e, run_r1.sh)
    "r1_baseline_l2": dict(use_lma=False, num_stages=0, stage_type="eppm", l2norm_point_proto=True),
    "r1_eppm": dict(use_lma=False, num_stages=1, stage_type="eppm", cross_attn_support="class_slots",
                    l2norm_point_proto=False),
    "r1_pooled": dict(use_lma=False, num_stages=1, stage_type="eppm", cross_attn_support="pooled",
                      l2norm_point_proto=False),
    "r1_eppms": dict(use_lma=False, num_stages=1, stage_type="eppm_s", cross_attn_support="pooled",
                     l2norm_point_proto=True),
    "r1_eppms_slots": dict(use_lma=False, num_stages=1, stage_type="eppm_s",
                           cross_attn_support="class_slots", l2norm_point_proto=True),
    "r1_vippem": dict(use_lma=False, num_stages=1, stage_type="vip", l2norm_point_proto=True),
    "r1_vip4": dict(use_lma=False, num_stages=4, stage_type="vip", l2norm_point_proto=True),
}


def config_of(variant):
    args = train.parse_args(["--dataset", "s3dis", "--data_path", "x", "--cvfold", "1", "--n_way", "2",
                             "--k_shot", "1"] + VARIANTS[variant])
    return train.model_config(args)


@pytest.mark.parametrize("variant,expected", sorted(R1.items()))
def test_r1_1_each_variant_is_the_configuration_it_claims(variant, expected):
    config = config_of(variant)
    for field, value in expected.items():
        assert getattr(config, field) == value, field
    config.check_implemented()  # the switch guards of D-24 / D-25 accept it


def test_r1_2_the_variants_differ_in_exactly_one_thing_at_a_time():
    """R1 changes one variable per step: support reading, then the stage, then the reference head."""
    a, b = config_of("r1_eppm").to_dict(), config_of("r1_pooled").to_dict()
    assert {k for k in a if a[k] != b[k]} == {"cross_attn_support"}  # (a) -> (b)
    c_slots, c = config_of("r1_eppms_slots").to_dict(), config_of("r1_eppms").to_dict()
    assert {k for k in c_slots if c_slots[k] != c[k]} == {"cross_attn_support"}  # (c') -> (c)
    assert {k for k in c if c[k] != config_of("r1_vippem").to_dict()[k]} == {"stage_type",
                                                                             "cross_attn_support"}


def test_r1_3_summary_reproduces_hand_computed_statistics(tmp_path):
    log = tmp_path / "r1.log"
    log.write_text("\n".join([
        "=== noise",
        "[diag] variant=r1_baseline_l2 | cvfold=1 | seed=0 | valid_miou=0.5000 | seconds=1.0",
        "[diag] variant=r1_baseline_l2 | cvfold=1 | seed=1 | valid_miou=0.5200 | seconds=1.0",
        "[diag] variant=r1_baseline_l2 | cvfold=1 | seed=2 | valid_miou=0.5300 | seconds=1.0",
        "[diag] variant=r1_pooled | cvfold=1 | seed=0 | valid_miou=0.6000 | seconds=1.0",
        "[diag] variant=r1_pooled | cvfold=1 | seed=1 | valid_miou=0.6100 | seconds=1.0",
        "[diag] variant=r1_pooled | cvfold=1 | seed=2 | valid_miou=0.6200 | seconds=1.0",
    ]) + "\n", encoding="utf-8")
    runs = summarize_r1.parse(str(log))
    assert [s for s, _ in runs["r1_pooled"]] == [0, 1, 2]
    mean, sd, n = summarize_r1.stats([v for _, v in runs["r1_pooled"]])
    assert (round(mean, 6), round(sd, 6), n) == (0.61, 0.01, 3)
    d, se, t = summarize_r1.welch([0.60, 0.61, 0.62], [0.50, 0.52, 0.53])
    assert round(d, 6) == round(0.61 - 0.5166666666666667, 6)
    assert round(se, 6) == round((0.01 ** 2 / 3 + 0.015275252316519467 ** 2 / 3) ** 0.5, 6)
    assert t > 3  # this synthetic gap would satisfy rule R1.1


def test_r1_4_summary_reports_missing_variants_instead_of_guessing(tmp_path, capsys):
    log = tmp_path / "r1.log"
    log.write_text("[diag] variant=r1_eppm | seed=0 | valid_miou=0.5000\n", encoding="utf-8")
    assert summarize_r1.main([str(log)]) == 0
    out = capsys.readouterr().out
    assert "not run yet" in out and "r1_eppm" in out
    empty = tmp_path / "empty.log"
    empty.write_text("=== nothing yet\n", encoding="utf-8")
    assert summarize_r1.main([str(empty)]) == 1  # a log without results is an error, not an empty table


def test_r1_5_the_queue_screens_on_s1_and_states_its_decision_rules():
    with open(os.path.join(REPO, "experiments", "run_r1.sh"), encoding="utf-8") as f:
        script = f.read()
    assert "FOLD=${FOLD:-1}" in script  # S0 stays held out [DECISION D-22]
    assert 'SEEDS=${SEEDS:-"0 1 2"}' in script and "EPISODES=${EPISODES:-9600}" in script
    for rule in ("R1.1", "R1.2", "R1.3", "R1.4", "R1.5"):
        assert rule in script
    for variant in ("r1_eppm", "r1_pooled", "r1_eppms", "r1_vippem", "r1_baseline_l2"):
        assert variant in script and variant in VARIANTS
