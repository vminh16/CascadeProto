"""CPX-1..3 (05 §3.8e, gate G1): the complexity report of experiments/complexity.py on the stand-in encoder."""

import pytest
import torch

from experiments import complexity
from models.cascadeproto import CascadeProtoConfig
from pipeline.model_api import EpisodeOutput
from tests.test_cascadeproto import model


@pytest.mark.parametrize("config,added", [(CascadeProtoConfig(use_lma=False, num_stages=0), 0),
                                          (CascadeProtoConfig(), 466_444),
                                          (CascadeProtoConfig(num_stages=1), 148_352 + 79_395)])
def test_cpx1_parameter_groups_add_up(config, added):
    g = complexity.parameter_groups(model(config))
    assert g["added_modules"] == added
    assert g["encoder"] + g["feature_head"] + g["added_modules"] == g["total"]
    assert g["feature_head"] == 204_260  # 01 §4


def test_cpx2_flops_are_counted_and_grow_with_the_cascade():
    base = model(CascadeProtoConfig(use_lma=False, num_stages=0)).float().eval()
    full = model(CascadeProtoConfig()).float().eval()
    ep = complexity.synthetic_episode(2, 1, torch.device("cpu"))
    f_base, f_full = complexity.flops(base, ep), complexity.flops(full, ep)
    assert f_base["gflops"] > 0 and f_full["gflops"] > f_base["gflops"]
    assert isinstance(f_full["unsupported_ops"], dict)


def test_cpx3_synthetic_episode_and_timing():
    ep = complexity.synthetic_episode(3, 5, torch.device("cpu"))
    assert ep.support_x.shape == (3, 5, 2048, 9) and ep.query_x.shape == (3, 2048, 9)
    assert (ep.support_y.sum(-1) > 0).all()
    m = model(CascadeProtoConfig(use_lma=False, num_stages=0)).float().eval()
    assert isinstance(m(ep), EpisodeOutput)
    assert complexity.time_per_episode(m, ep, repeats=2, warmup=1) > 0
