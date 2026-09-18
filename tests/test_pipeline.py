"""Gate G1: episode pipeline, loss contract and schedule (spec 04 §4-§5, 02 §1 and §7).

Synthetic arrays only (05 §1 principle 3); real data is covered by tests/test_data.py.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.episodes import EpisodeCollate, make_episode
from pipeline.model_api import GMMN_WEIGHT, EpisodeOutput, episode_loss, predict

N, K, P, C = 2, 3, 2048, 9
CLASS_NAMES = ["ceiling", "floor", "wall", "beam", "column", "window", "door",
               "table", "chair", "sofa", "bookcase", "board", "clutter"]


def loader_item(n=N, k=K, seed=0):
    """A tuple shaped like MyDataset.__getitem__ [VIPSEG dataloaders/loader.py:157-161]."""
    rng = np.random.default_rng(seed)
    support_x = rng.random((n, k, P, C), dtype=np.float32)  # [N, K, 2048, 9]
    support_y = rng.integers(0, 2, (n, k, P)).astype(np.int32)  # [N, K, 2048]
    query_x = rng.random((n, P, C), dtype=np.float32)  # [N*n_q, 2048, 9]
    query_y = rng.integers(0, n + 1, (n, P)).astype(np.int64)  # [N*n_q, 2048]
    sampled_classes = np.array([3, 11, 10][:n], dtype=np.int32)  # [N]
    return support_x, support_y, query_x, query_y, sampled_classes


def test_pipe1_episode_follows_spec02_layout():
    ep = make_episode(loader_item(), CLASS_NAMES)
    assert ep.support_x.shape == (N, K, P, C) and ep.support_x.dtype == torch.float32
    assert ep.support_y.shape == (N, K, P) and ep.support_y.dtype == torch.int64
    assert ep.query_x.shape == (N, P, C) and ep.query_y.shape == (N, P)
    assert ep.n_way == N and ep.k_shot == K
    assert ep.class_names == ["beam", "board"]
    assert torch.equal(ep.support_y, torch.from_numpy(loader_item()[1]).long())


@pytest.mark.parametrize("field,bad", [
    (1, lambda y: y + 1),  # support mask with value 2 (the old {1, 2} convention of audit L8)
    (0, lambda x: x[..., :3]),  # 3 channels instead of 9
    (3, lambda y: y + N + 1),  # query label above N
])
def test_pipe1_episode_rejects_bad_layout(field, bad):
    item = list(loader_item())
    item[field] = bad(item[field])
    with pytest.raises(ValueError):
        make_episode(item, CLASS_NAMES)


def test_pipe2_collate_keeps_one_episode_per_item():
    batch = [loader_item(seed=s) for s in range(4)]
    episodes = EpisodeCollate(CLASS_NAMES)(batch)
    assert len(episodes) == 4
    assert not torch.equal(episodes[0].support_x, episodes[1].support_x)


def test_pipe3_loss_is_unweighted_ce_plus_gmmn():
    ep = make_episode(loader_item(), CLASS_NAMES)
    logits = torch.randn(N, P, N + 1)  # [B_q, 2048, N+1]
    gmmn = torch.tensor(0.37)
    loss = episode_loss(EpisodeOutput(logits, gmmn), ep)
    log_p = F.log_softmax(logits, dim=-1)  # [B_q, 2048, N+1]
    manual = -log_p.gather(-1, ep.query_y.unsqueeze(-1)).mean() + GMMN_WEIGHT * gmmn
    assert torch.allclose(loss, manual, atol=1e-6)
    assert torch.equal(predict(EpisodeOutput(logits, gmmn)), logits.argmax(-1))


def test_pipe3_loss_rejects_wrong_logit_shape():
    ep = make_episode(loader_item(), CLASS_NAMES)
    with pytest.raises(ValueError):
        episode_loss(EpisodeOutput(torch.randn(N, P, N + 2), torch.tensor(0.0)), ep)


class StubModel(nn.Module):
    """Per-point linear classifier over the 9 input channels; fixed N+1 outputs."""

    def __init__(self, n_way):
        super().__init__()
        self.head = nn.Linear(C, n_way + 1)

    def forward(self, episode):
        return EpisodeOutput(self.head(episode.query_x), torch.zeros(()))  # [B_q, 2048, N+1]


def test_pipe4_one_step_uses_the_batch_mean_loss():
    from train import train_steps

    episodes = EpisodeCollate(CLASS_NAMES)([loader_item(seed=s) for s in range(4)])
    model = StubModel(N)
    expected = torch.stack([episode_loss(model(ep), ep) for ep in episodes]).mean().item()
    before = model.head.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    (loss,) = list(train_steps(model, optimizer, [episodes], torch.device("cpu")))
    assert loss == pytest.approx(expected, rel=1e-6)
    assert not torch.equal(before, model.head.weight)


@pytest.mark.parametrize("dataset,epochs,episodes,steps", [("s3dis", 50, 480, 120), ("scannet", 30, 800, 200)])
def test_pipe5_schedule_defaults(dataset, epochs, episodes, steps):
    from train import parse_args

    args = parse_args(["--dataset", dataset, "--data_path", "x", "--cvfold", "0", "--n_way", "2", "--k_shot", "1"])
    assert (args.epochs, args.episodes_per_epoch) == (epochs, episodes)
    assert args.episodes_per_epoch // 4 == steps
    assert args.epochs * args.episodes_per_epoch == 24000  # [DECISION D-12]
    assert (args.lr, args.weight_decay, args.lr_step_epochs, args.lr_gamma) == (1e-3, 0.1, 10, 0.5)


def test_pipe5_rejects_partial_batches():
    from train import parse_args

    with pytest.raises(SystemExit):
        parse_args(["--dataset", "s3dis", "--data_path", "x", "--cvfold", "0", "--n_way", "2", "--k_shot", "1",
                    "--episodes_per_epoch", "481"])


def test_pipe6_eval_refuses_missing_checkpoint(tmp_path):
    from eval import load_model, parse_args

    args = parse_args(["--dataset", "s3dis", "--data_path", "x", "--cvfold", "0", "--n_way", "2", "--k_shot", "1",
                       "--checkpoint", str(tmp_path / "missing.pt")])
    with pytest.raises(FileNotFoundError):
        load_model(args, torch.device("cpu"))


def test_pipe6_unimplemented_modality_raises():
    from train import build_model, model_config, parse_args

    args = parse_args(["--dataset", "s3dis", "--data_path", "x", "--cvfold", "0", "--n_way", "2", "--k_shot", "1",
                       "--modality", "audio"])
    with pytest.raises(NotImplementedError, match="modality 'audio'"):
        build_model(model_config(args))


def test_pipe6_switch_defaults_are_the_full_model():
    from train import model_config, parse_args

    config = model_config(parse_args(["--dataset", "s3dis", "--data_path", "x", "--cvfold", "0",
                                      "--n_way", "2", "--k_shot", "1"]))
    assert (config.use_lma, config.num_stages, config.use_gate, config.use_adrm) == (True, 4, True, True)
    assert (config.logit_scale, config.l2norm_point_proto, config.modality) == ("none", False, "text")


def test_pipe7_eval_rebuilds_the_checkpoint_configuration(tmp_path, monkeypatch):
    """eval.py builds the architecture stored in the checkpoint, not one from its own CLI."""
    import eval as eval_script
    from models.cascadeproto import CascadeProto, CascadeProtoConfig
    from models.vipseg_backbone import PointFeatureExtractor
    from tests.test_feature_extractor import StandInEncoder

    def stand_in_model(config):
        return CascadeProto(config, PointFeatureExtractor(encoder=StandInEncoder()))

    monkeypatch.setattr(eval_script, "build_model", stand_in_model)
    config = CascadeProtoConfig(use_lma=False, num_stages=0, logit_scale="sqrt_D")
    torch.manual_seed(0)
    trained = stand_in_model(config).eval()
    path = tmp_path / "best.pt"
    torch.save({"model": trained.state_dict(), "config": config.to_dict()}, path)
    args = eval_script.parse_args(["--dataset", "s3dis", "--data_path", "x", "--cvfold", "0", "--n_way", "2",
                                   "--k_shot", "1", "--checkpoint", str(path)])
    torch.manual_seed(1)
    loaded = eval_script.load_model(args, torch.device("cpu")).eval()
    assert loaded.config == config
    ep = make_episode(loader_item(), CLASS_NAMES)
    assert torch.equal(loaded(ep).logits, trained(ep).logits)
