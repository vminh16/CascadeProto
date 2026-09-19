"""TXT-1..6 (05 §3.3b): the CLIP text front-end of spec 03 §2.1.

CPU tests replace CLIP with a recording stand-in encoder (05 §1 principle 3). Tests marked `clip`
load the real frozen CLIP text encoder (gate G3).
"""

import pytest
import torch
import torch.nn.functional as F

from models import clip_text
from models.clip_text import BACKGROUND_PROMPT, DEFAULT_CLIP_VARIANT, ClipTextEmbedding, episode_prompts

CPU = torch.device("cpu")
S3DIS = ["ceiling", "floor", "wall", "beam", "column", "window", "door", "table", "chair", "sofa",
         "bookcase", "board", "clutter"]


class RecordingEncoder:
    """Deterministic raw features per prompt, in float16 like CLIP on a GPU; records every call."""

    def __init__(self):
        self.calls = []

    def raw(self, prompt):
        g = torch.Generator().manual_seed(sum(map(ord, prompt)) * 7919 + len(prompt))
        return (3.0 * torch.randn(512, generator=g)).half()

    def __call__(self, prompts, device):
        self.calls.append(list(prompts))
        return torch.stack([self.raw(p) for p in prompts])


def test_txt1_prompts_follow_fig1_and_d13():
    assert BACKGROUND_PROMPT == "This point cloud represents the background."
    assert episode_prompts(["chair", "shower curtain"]) == [
        "This point cloud represents the background.",
        "This point cloud represents the chair.",
        "This point cloud represents the shower curtain.",
    ]
    assert DEFAULT_CLIP_VARIANT == "ViT-B/16"


def test_txt2_rows_are_float32_unit_vectors_in_prompt_order():
    enc = RecordingEncoder()
    e = ClipTextEmbedding(encode=enc)(["door", "chair"], CPU)
    assert e.shape == (3, 512) and e.dtype == torch.float32 and not e.requires_grad
    prompts = episode_prompts(["door", "chair"])
    expected = F.normalize(torch.stack([enc.raw(p) for p in prompts]).float(), dim=-1)
    assert torch.equal(e, expected)  # float32 before the norm, row 0 = background
    assert torch.allclose(e.norm(dim=-1), torch.ones(3), atol=1e-6, rtol=0)


def test_txt3_way_order_permutes_foreground_rows_only():
    emb = ClipTextEmbedding(encode=RecordingEncoder())
    a, b = emb(["door", "chair", "sofa"], CPU), emb(["sofa", "door", "chair"], CPU)
    assert torch.equal(b[0], a[0]) and torch.equal(b[1:], a[1:][[2, 0, 1]])


def test_txt4_each_prompt_is_encoded_once():
    enc = RecordingEncoder()
    emb = ClipTextEmbedding(encode=enc)
    first = emb(["door", "chair"], CPU)
    second = emb(["chair", "sofa"], CPU)
    assert enc.calls == [episode_prompts(["door", "chair"]), ["This point cloud represents the sofa."]]
    assert torch.equal(second[1], first[2])
    emb(["door", "sofa"], CPU)
    assert len(enc.calls) == 2


def test_txt5_wrong_feature_shape_raises():
    with pytest.raises(ValueError):
        ClipTextEmbedding(encode=lambda prompts, device: torch.zeros(len(prompts), 768))(["door"], CPU)


# ------------------------------------------------------------------------------- real CLIP (G3)

@pytest.mark.clip
def test_txt6_unknown_variant_raises_without_fallback():
    with pytest.raises(ValueError, match="unknown CLIP variant"):
        clip_text.load_clip("ViT-B/99", CPU)


@pytest.mark.clip
def test_ep5_clip_is_loaded_once_and_frozen():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    before = clip_text.LOAD_COUNT["n"]
    already = DEFAULT_CLIP_VARIANT in clip_text._LOADED
    a, b = ClipTextEmbedding(), ClipTextEmbedding()
    a(["door", "chair"], device)
    b(["table", "sofa"], device)
    assert clip_text.LOAD_COUNT["n"] - before == (0 if already else 1)
    model = clip_text._LOADED[DEFAULT_CLIP_VARIANT]
    assert not model.training and not any(p.requires_grad for p in model.parameters())


@pytest.mark.clip
def test_ep6_real_embeddings_match_direct_clip_call():
    import clip

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    e = ClipTextEmbedding()(S3DIS[3:6], device)  # beam, column, window
    assert e.shape == (4, 512) and e.dtype == torch.float32
    model = clip_text.load_clip(DEFAULT_CLIP_VARIANT, device)
    tokens = clip.tokenize(episode_prompts(S3DIS[3:6])).to(device)
    with torch.no_grad():
        direct = F.normalize(model.encode_text(tokens).float(), dim=-1)
    assert torch.equal(e, direct)
    sim = e @ e.T  # distinct prompts give distinct unit vectors
    assert torch.allclose(sim.diagonal(), torch.ones(4, device=device), atol=1e-5)
    assert (sim - torch.eye(4, device=device)).max() < 0.999
