"""EP-1..4 (05 §3.9, gate G3, marker `clip`): the full model on one episode with the real frozen CLIP.

The VIP-Seg encoder needs CUDA; here it is the per-point stand-in of test_feature_extractor.py, so the
test runs on any machine that has the CLIP weights. float32, as in training. EP-5/6 are in
tests/test_clip_text.py.
"""

import copy

import pytest
import torch

from models.cascadeproto import CascadeProto, CascadeProtoConfig
from models.vipseg_backbone import PointFeatureExtractor
from pipeline.model_api import episode_loss, predict
from tests.test_cascadeproto import learnable_episode
from tests.test_feature_extractor import StandInEncoder

pytestmark = pytest.mark.clip
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="module")
def setup():
    torch.manual_seed(0)
    m = CascadeProto(CascadeProtoConfig(), PointFeatureExtractor(encoder=StandInEncoder())).to(DEVICE)
    ep = learnable_episode(n=2, k=1, bq=2)
    ep.support_x, ep.query_x = ep.support_x.float(), ep.query_x.float()
    return m, ep.to(DEVICE)


def test_ep1_full_model_output_contract(setup):
    m, ep = setup
    out = m.eval()(ep)
    assert out.logits.shape == (2, 2048, 3) and out.logits.dtype == torch.float32
    assert out.loss_gmmn.shape == () and torch.isfinite(out.loss_gmmn) and out.loss_gmmn > 0
    assert m.text.variant == "ViT-B/16" and len(m.stages) == 4 and m.routing.w_g.out_features == 4
    e = m.text(ep.class_names, DEVICE)
    assert e.shape == (3, 512) and not torch.allclose(e[1], e[2])  # real, distinct class embeddings


def test_ep2_loss_is_finite_and_reaches_every_parameter(setup):
    m, ep = setup
    m.train()
    m.zero_grad()
    loss = episode_loss(m(ep), ep)
    assert torch.isfinite(loss)
    loss.backward()
    for name, p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, name


def test_ep3_one_adamw_step_changes_parameters(setup):
    m, ep = setup
    m = copy.deepcopy(m).train()
    before = {n: p.detach().clone() for n, p in m.named_parameters()}
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.1)  # [PAPER §4.1]
    opt.zero_grad()
    episode_loss(m(ep), ep).backward()
    opt.step()
    for n, p in m.named_parameters():
        assert torch.isfinite(p).all() and not torch.equal(p, before[n]), n


def test_ep4_evaluation_is_deterministic(setup):
    m, ep = setup
    m.eval()
    with torch.no_grad():
        a, b = m(ep), m(ep)
    assert torch.equal(a.logits, b.logits) and torch.equal(predict(a), predict(b))
