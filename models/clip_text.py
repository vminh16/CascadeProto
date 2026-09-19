"""Text front-end of the LMA: class names -> frozen CLIP text embeddings E_CLIP [N+1, 512] (spec 03 §2.1).

CLIP is not a submodule of the model: it is frozen, never saved in a checkpoint, and loaded once per
process and variant [DECISION D-13]. Embeddings are cached per prompt string, so after the first few
episodes no text is encoded again.
"""

from typing import Callable, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

DEFAULT_CLIP_VARIANT = "ViT-B/16"  # [DECISION D-13]
CLIP_DIM = 512  # [PAPER §4.1]
BACKGROUND_PROMPT = "This point cloud represents the background."  # [DECISION D-13]

_LOADED: Dict[str, torch.nn.Module] = {}  # variant -> frozen CLIP model, one per process
LOAD_COUNT = {"n": 0}  # number of clip.load calls in this process (test EP-5)


def class_prompt(name: str) -> str:
    """Foreground prompt of [PAPER Fig.1]; `name` verbatim from the class-name file [DECISION D-13]."""
    return f"This point cloud represents the {name}."


def episode_prompts(class_names: Sequence[str]) -> List[str]:
    """Row order [background, class 1, ..., class N], the order of P_point (02 §0)."""
    return [BACKGROUND_PROMPT] + [class_prompt(name) for name in class_names]


def load_clip(variant: str, device: torch.device) -> torch.nn.Module:
    """The frozen CLIP model of `variant`, loaded on the first call only. No fallback variant."""
    if variant not in _LOADED:
        import clip

        if variant not in clip.available_models():
            raise ValueError(f"unknown CLIP variant {variant!r}; available: {clip.available_models()}")
        model, _ = clip.load(variant, device=device, jit=False)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        _LOADED[variant] = model
        LOAD_COUNT["n"] += 1
    return _LOADED[variant]


def clip_encode_text(variant: str) -> Callable[[List[str], torch.device], torch.Tensor]:
    """encode(prompts, device) -> raw CLIP text features [len(prompts), 512]."""

    def encode(prompts: List[str], device: torch.device) -> torch.Tensor:
        import clip

        model = load_clip(variant, device)
        tokens = clip.tokenize(prompts).to(next(model.parameters()).device)
        with torch.no_grad():
            return model.encode_text(tokens)

    return encode


class ClipTextEmbedding:
    """Class names -> E_CLIP [N+1, 512], float32, L2-normalised, row 0 = background (03 §2.1).

    `encode` replaces CLIP in CPU tests (05 §1 principle 3); by default the real CLIP text encoder of
    `variant` is used.
    """

    def __init__(self, variant: str = DEFAULT_CLIP_VARIANT,
                 encode: Optional[Callable[[List[str], torch.device], torch.Tensor]] = None):
        self.variant = variant
        self.encode = encode if encode is not None else clip_encode_text(variant)
        self.cache: Dict[str, torch.Tensor] = {}  # prompt -> [512] float32 on CPU

    def __call__(self, class_names: Sequence[str], device: torch.device) -> torch.Tensor:
        prompts = episode_prompts(class_names)
        missing = [p for p in dict.fromkeys(prompts) if p not in self.cache]
        if missing:
            raw = self.encode(missing, device)  # [M, 512]
            if raw.shape != (len(missing), CLIP_DIM):
                raise ValueError(f"CLIP text features {tuple(raw.shape)} != {(len(missing), CLIP_DIM)}")
            unit = F.normalize(raw.float(), dim=-1)  # float32 first, then L2 norm [DECISION D-13]
            for prompt, row in zip(missing, unit.cpu()):
                self.cache[prompt] = row
        return torch.stack([self.cache[p] for p in prompts]).to(device)  # [N+1, 512]
