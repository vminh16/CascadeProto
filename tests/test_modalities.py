"""MOD-1..8 (05 §3.8z): the class-level image and audio front-ends of [DECISION D-47]. Beyond the paper's text. CPU;
CLIP, Whisper and espeak-ng are replaced by stand-ins (05 §1 principle 3), the real run is `experiments/d47_build.py`.
"""

import json
import os
import re

import pytest
import torch

from models.clip_audio import ClipAudioEmbedding, normalise
from models.clip_image import BACKGROUND_KEY, DEFAULT_MANIFEST, ClipImageEmbedding, load_manifest, sha256_of
from models.clip_text import ClipTextEmbedding, episode_prompts

CPU = torch.device("cpu")
S3DIS = ["ceiling", "floor", "wall", "beam", "column", "window", "door", "table", "chair", "sofa", "bookcase",
         "board", "clutter"]
LICENCES = re.compile(r"^(CC0|Public domain|CC BY(-SA)? [0-9.]+( [a-z]{2})?)$")


def test_mod1_manifest_covers_every_class_with_open_licences():
    m = load_manifest()
    assert set(m) == set(S3DIS) | {BACKGROUND_KEY}
    titles = [e["title"] for v in m.values() for e in v]
    assert len(titles) == len(set(titles)) == 5 * 14
    for name, entries in m.items():
        assert len(entries) == 5
        for e in entries:
            assert LICENCES.match(e["licence"]), e["licence"]
            assert re.fullmatch(r"[0-9a-f]{40}", e["sha1"]) and e["url"].startswith("https://upload.wikimedia.org/")
            assert e["file"].startswith(name + "/") and e["author"]
            assert re.fullmatch(r"[0-9a-f]{64}", e["sha256"]), f"{e['file']}: sha256 not recorded"


def fake_images(tmp_path, names=("background", "door", "chair"), m=3):
    manifest = {}
    for n in names:
        entries = []
        for i in range(m):
            rel = f"{n}/{i}.bin"
            os.makedirs(tmp_path / n, exist_ok=True)
            (tmp_path / rel).write_bytes(f"{n}-{i}".encode())
            entries.append({"file": rel, "sha256": sha256_of(str(tmp_path / rel))})
        manifest[n] = entries
    return manifest


class PathEncoder:
    """Stand-in for the CLIP image encoder: a fixed random feature per file, and a record of the calls."""

    def __init__(self):
        self.calls = []

    def __call__(self, paths, device):
        self.calls.append(list(paths))
        feats = []
        for p in paths:
            g = torch.Generator().manual_seed(sum(open(p, "rb").read()))
            feats.append(3.0 * torch.randn(512, generator=g))
        return torch.stack(feats)  # [M, 512]


def test_mod2_image_rows_are_unit_means_in_row_order(tmp_path):
    manifest = fake_images(tmp_path)
    enc = PathEncoder()
    emb = ClipImageEmbedding(manifest=manifest, root=str(tmp_path), encode=enc)
    e = emb(["door", "chair"], CPU)
    assert e.shape == (3, 512) and e.dtype == torch.float32
    assert torch.allclose(e.norm(dim=-1), torch.ones(3), atol=1e-6)
    for row, name in zip(e, ["background", "door", "chair"]):
        paths = [str(tmp_path / x["file"]) for x in manifest[name]]
        unit = torch.nn.functional.normalize(enc(paths, CPU), dim=-1)
        assert torch.allclose(row, torch.nn.functional.normalize(unit.mean(0), dim=0), atol=1e-6)


def test_mod3_image_row_ignores_the_order_of_the_images_and_is_cached(tmp_path):
    manifest = fake_images(tmp_path)
    a = ClipImageEmbedding(manifest=manifest, root=str(tmp_path), encode=PathEncoder())(["door", "chair"], CPU)
    flipped = {k: list(reversed(v)) for k, v in manifest.items()}
    enc = PathEncoder()
    emb = ClipImageEmbedding(manifest=flipped, root=str(tmp_path), encode=enc)
    assert torch.allclose(emb(["door", "chair"], CPU), a, atol=1e-6)
    emb(["chair", "door"], CPU)
    assert len(enc.calls) == 3  # background, door, chair once each


def test_mod4_missing_or_altered_image_raises(tmp_path):
    manifest = fake_images(tmp_path)
    emb = ClipImageEmbedding(manifest=manifest, root=str(tmp_path), encode=PathEncoder())
    with pytest.raises(KeyError):
        emb(["sofa"], CPU)
    (tmp_path / "door/1.bin").write_bytes(b"changed")
    with pytest.raises(ValueError, match="sha256"):
        emb(["door"], CPU)
    os.remove(tmp_path / "chair/0.bin")
    with pytest.raises(FileNotFoundError):
        emb(["chair"], CPU)


class TextEncoder:
    def __call__(self, prompts, device):
        return torch.stack([torch.randn(512, generator=torch.Generator().manual_seed(hash(normalise(p)) % 2**31))
                            for p in prompts])


def test_mod5_audio_row_is_the_text_row_when_the_transcript_is_exact():
    heard = {}
    synth = lambda text, wav: heard.__setitem__(wav, text)  # noqa: E731
    exact = lambda wav, device: heard[wav].upper().replace(".", "")  # noqa: E731, case and punctuation only
    enc = TextEncoder()
    emb = ClipAudioEmbedding(synth=synth, transcribe=exact, encode=enc)
    audio = emb(["door", "chair"], CPU)
    text = ClipTextEmbedding(encode=lambda ps, d: enc([normalise(p) for p in ps], d))(["door", "chair"], CPU)
    assert torch.allclose(audio, text, atol=1e-6)
    assert torch.allclose(audio.norm(dim=-1), torch.ones(3), atol=1e-6)
    assert all(exact_flag for _, exact_flag in emb.log.values()) and len(emb.log) == 3


def test_mod6_audio_logs_the_transcript_and_whether_it_is_exact():
    heard = {}
    emb = ClipAudioEmbedding(synth=lambda text, wav: heard.__setitem__(wav, text),
                             transcribe=lambda wav, d: heard[wav].replace("chair", "chare"), encode=TextEncoder())
    rows = emb(["door", "chair"], CPU)
    prompts = episode_prompts(["door", "chair"])
    assert emb.log[prompts[1]] == (prompts[1], True)
    assert emb.log[prompts[2]][1] is False and "chare" in emb.log[prompts[2]][0]
    heard_rows = torch.nn.functional.normalize(TextEncoder()([emb.log[p][0] for p in prompts], CPU), dim=-1)
    assert torch.allclose(rows, heard_rows, atol=1e-6)  # the transcript is encoded, not the prompt
    said = torch.nn.functional.normalize(TextEncoder()(prompts, CPU), dim=-1)
    assert not torch.allclose(rows[2], said[2], atol=1e-3)


def test_mod7_missing_engine_raises(monkeypatch):
    import shutil

    from models import clip_audio

    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(FileNotFoundError, match="espeak-ng"):
        clip_audio.espeak_synth("x", "y.wav")


def test_mod8_unknown_modality_raises_and_the_factory_knows_all_three():
    from models.cascadeproto import modality_embedding

    with pytest.raises(ValueError):
        modality_embedding("video")
    assert isinstance(modality_embedding("image"), ClipImageEmbedding)
    assert isinstance(modality_embedding("audio"), ClipAudioEmbedding)
    with open(DEFAULT_MANIFEST, encoding="utf-8") as f:
        assert json.load(f)["decision"] == "D-47"


def test_mod9_fetcher_verifies_both_hashes(tmp_path):
    import hashlib

    from preprocess.fetch_modality_images import verify

    path = tmp_path / "x.jpg"
    path.write_bytes(b"image bytes")
    entry = {"title": "x", "sha1": hashlib.sha1(b"image bytes").hexdigest(), "sha256": sha256_of(str(path))}
    verify(str(path), entry)
    verify(str(path), {**entry, "sha256": ""})  # not yet recorded: only the sha1 is checked
    with pytest.raises(ValueError, match="sha1"):
        verify(str(path), {**entry, "sha1": "0" * 40})
    with pytest.raises(ValueError, match="sha256"):
        verify(str(path), {**entry, "sha256": "0" * 64})
