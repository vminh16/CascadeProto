"""ABL-1..3 (05 §3.6, gate G1): the ablation switches of spec 01 §3 and [DECISION D-17].

Each Table 4 row must produce its prescribed prediction; the references are the written-out ones of
tests/test_cascadeproto.py.
"""

import pytest
import torch

from models.cascadeproto import CascadeProto, CascadeProtoConfig
from pipeline.model_api import episode_loss
from tests.test_cascadeproto import (BASELINE, CASCADE, FULL, GATE, LMA, cascade_ref, episode, full_ref, lma_terms,
                                     model)
from models.prototypes import point_prototypes


def prediction_ref(config, m, ep):
    if config == BASELINE:
        f_s, f_q = m.features.encode_episode(ep.support_x, ep.query_x)
        return torch.einsum("bpd,cd->bpc", f_q, point_prototypes(f_s, ep.support_y))  # F^q P_pointᵀ
    if config == LMA:
        f_q, p_point, p_modal = lma_terms(m, ep)
        return torch.einsum("bpd,cd->bpc", f_q, p_point + p_modal)  # F^q (P^0)ᵀ
    if config in (GATE, CASCADE):
        return cascade_ref(m, ep)  # L^1, L^4
    return full_ref(m, ep)[0]  # L_final


@pytest.mark.parametrize("config", [BASELINE, LMA, GATE, CASCADE, FULL],
                         ids=["baseline", "lma", "entropy_gate", "cascade", "adrm"])
def test_abl1_each_table4_row_gives_its_prediction_and_trains(config):
    m, ep = model(config), episode()
    m.eval()
    assert torch.allclose(m(ep).logits, prediction_ref(config, m, ep), atol=1e-11, rtol=0)
    m.train()
    episode_loss(m(ep), ep).backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())


@pytest.mark.parametrize("t", range(1, 7))
def test_abl2_num_stages_sets_the_routing_width(t):
    m = model(CascadeProtoConfig(num_stages=t))
    assert len(m.stages) == t and (t == 1 or m.routing.weights(torch.rand(2, 2048, 128).double()).shape == (2, t))


@pytest.mark.parametrize("modality", ["image", "audio"])
def test_abl3_unimplemented_modalities_raise(modality):
    with pytest.raises(NotImplementedError, match=modality):
        CascadeProto(CascadeProtoConfig(modality=modality))
