"""Audio front-end of the LMA at the class level [PAPER Fig.1 "Audio -> Whisper -> CLIP"] [DECISION D-47].
**The audio source is not in the paper.**

Per prompt of 03 §2.1 (`episode_prompts`, row 0 = background):
  speech      offline text-to-speech, `espeak-ng`, one voice and rate (logged)
  transcript  frozen Whisper `base`, English, greedy decoding (temperature 0, no fallback)
  row         the CLIP text embedding of the transcript, L2-normalised
Each transcript is logged with whether it equals the prompt after lower-casing and punctuation removal. Stated
before any run (D-47): the row is a noisy copy of the text row, I(audio; class) <= I(text; class), and it equals the
text row whenever the transcript is exact. A missing engine or model raises.
"""

import os
import re
import shutil
import subprocess
import tempfile
from typing import Callable, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

from models.clip_text import CLIP_DIM, DEFAULT_CLIP_VARIANT, clip_encode_text, episode_prompts

TTS_ENGINE, TTS_VOICE, TTS_RATE = "espeak-ng", "en-us", 150  # words per minute [DECISION D-47]
WHISPER_MODEL = "base"  # [DECISION D-47]
_WHISPER: Dict[str, object] = {}  # name -> frozen Whisper model, one per process


def normalise(text: str) -> str:
    """Lower case, punctuation removed, whitespace collapsed: the exact-transcript test of D-47."""
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


def espeak_synth(text: str, wav_path: str) -> None:
    """Speak `text` into `wav_path` with espeak-ng; raises when the engine is missing or fails."""
    if shutil.which(TTS_ENGINE) is None:
        raise FileNotFoundError(f"{TTS_ENGINE} not found; install it (apt install espeak-ng) [DECISION D-47]")
    subprocess.run([TTS_ENGINE, "-v", TTS_VOICE, "-s", str(TTS_RATE), "-w", wav_path, text], check=True,
                   capture_output=True)


def tts_version() -> str:
    if shutil.which(TTS_ENGINE) is None:
        raise FileNotFoundError(f"{TTS_ENGINE} not found [DECISION D-47]")
    return subprocess.run([TTS_ENGINE, "--version"], check=True, capture_output=True, text=True).stdout.strip()


def whisper_transcribe(name: str = WHISPER_MODEL) -> Callable[[str, torch.device], str]:
    """transcribe(wav_path, device) -> text, frozen Whisper `name`, English, greedy."""

    def transcribe(wav_path: str, device: torch.device) -> str:
        import whisper

        if name not in _WHISPER:
            model = whisper.load_model(name, device=device)
            model.eval()
            for p in model.parameters():
                p.requires_grad_(False)
            _WHISPER[name] = model
        model = _WHISPER[name]
        out = model.transcribe(wav_path, language="en", task="transcribe", temperature=0.0, beam_size=None,
                               best_of=None, condition_on_previous_text=False,
                               fp16=next(model.parameters()).is_cuda)
        return out["text"].strip()

    return transcribe


class ClipAudioEmbedding:
    """Class names -> E_CLIP [N+1, 512], float32, unit rows, row 0 = background [DECISION D-47].

    `synth`, `transcribe` and `encode` replace espeak-ng, Whisper and CLIP in CPU tests. Rows are cached per prompt;
    `log` keeps {prompt: (transcript, exact)}.
    """

    def __init__(self, variant: str = DEFAULT_CLIP_VARIANT,
                 synth: Optional[Callable[[str, str], None]] = None,
                 transcribe: Optional[Callable[[str, torch.device], str]] = None,
                 encode: Optional[Callable[[List[str], torch.device], torch.Tensor]] = None):
        self.variant = variant
        self.synth = synth if synth is not None else espeak_synth
        self.transcribe = transcribe if transcribe is not None else whisper_transcribe()
        self.encode = encode if encode is not None else clip_encode_text(variant)
        self.cache: Dict[str, torch.Tensor] = {}  # prompt -> [512] float32 on CPU
        self.log: Dict[str, tuple] = {}

    def transcript(self, prompt: str, device: torch.device) -> str:
        with tempfile.TemporaryDirectory() as tmp:
            wav = os.path.join(tmp, "prompt.wav")
            self.synth(prompt, wav)
            return self.transcribe(wav, device)

    def __call__(self, class_names: Sequence[str], device: torch.device) -> torch.Tensor:
        prompts = episode_prompts(class_names)
        missing = [p for p in dict.fromkeys(prompts) if p not in self.cache]
        if missing:
            heard = [self.transcript(p, device) for p in missing]
            raw = self.encode(heard, device)  # [M, 512]
            if raw.shape != (len(missing), CLIP_DIM):
                raise ValueError(f"CLIP text features {tuple(raw.shape)} != {(len(missing), CLIP_DIM)}")
            unit = F.normalize(raw.float(), dim=-1)  # [M, 512]
            for prompt, text, row in zip(missing, heard, unit.cpu()):
                self.cache[prompt] = row
                self.log[prompt] = (text, normalise(text) == normalise(prompt))
        return torch.stack([self.cache[p] for p in prompts]).to(device)  # [N+1, 512]
