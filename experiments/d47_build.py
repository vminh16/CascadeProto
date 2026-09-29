"""Build and describe the class-level rows of [DECISION D-47] with the real frozen models (GPU or CPU).

For the S3DIS class names and the background: the text row (D-13), the image row (M = 5 images through the CLIP
image encoder) and the audio row (espeak-ng -> Whisper base -> CLIP text). Written to results/phase16_d47/:
  rows_<variant>.npz   text, image, audio, each [14, 512], row 0 = background
  d47_build.json       engine and model versions, transcripts and their exactness, and descriptive tables:
                       per-class cosines between the modalities, the mean cosine between different classes within
                       each modality, and which text row each image / audio row is nearest to.
Descriptive only: D-47 has no rule on these numbers.

    python experiments/d47_build.py [--device cuda]
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.clip_audio import TTS_ENGINE, TTS_RATE, TTS_VOICE, WHISPER_MODEL, ClipAudioEmbedding, tts_version  # noqa
from models.clip_image import DEFAULT_MANIFEST, ClipImageEmbedding  # noqa: E402
from models.clip_text import DEFAULT_CLIP_VARIANT, ClipTextEmbedding, episode_prompts  # noqa: E402

OUT = "results/phase16_d47"


def class_names(path: str = "datasets/S3DIS/meta/s3dis_classnames.txt"):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def describe(rows: dict, names: list) -> dict:
    """Cosine tables between and within modalities; rows are unit vectors [C, 512]."""
    out = {"per_class": {}, "within": {}, "nearest_text": {}}
    for a, b in (("text", "image"), ("text", "audio"), ("image", "audio")):
        out["per_class"][f"{a}_{b}"] = dict(zip(names, (rows[a] * rows[b]).sum(1).round(4).tolist()))
    off = ~np.eye(len(names), dtype=bool)
    for m, r in rows.items():
        g = r @ r.T  # [C, C]
        out["within"][m] = {"mean_off_diagonal": round(float(g[off].mean()), 4),
                            "max_off_diagonal": round(float(g[off].max()), 4)}
    for m in ("image", "audio"):
        nearest = (rows[m] @ rows["text"].T).argmax(1)  # [C]
        out["nearest_text"][m] = {"correct": int((nearest == np.arange(len(names))).sum()), "of": len(names),
                                  "nearest": dict(zip(names, [names[i] for i in nearest]))}
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--variant", default=DEFAULT_CLIP_VARIANT)
    args = p.parse_args(argv)
    device = torch.device(args.device)
    os.makedirs(OUT, exist_ok=True)
    names = class_names()
    audio = ClipAudioEmbedding(args.variant)
    rows = {"text": ClipTextEmbedding(args.variant)(names, device),
            "image": ClipImageEmbedding(args.variant)(names, device),
            "audio": audio(names, device)}
    rows = {k: v.float().cpu().numpy() for k, v in rows.items()}  # each [14, 512]
    labels = ["background"] + names
    np.savez(os.path.join(OUT, f"rows_{args.variant.replace('/', '-')}.npz"), labels=np.array(labels), **rows)
    prompts = episode_prompts(names)
    report = {
        "variant": args.variant, "manifest": os.path.relpath(DEFAULT_MANIFEST),
        "tts": {"engine": TTS_ENGINE, "voice": TTS_VOICE, "rate_wpm": TTS_RATE, "version": tts_version()},
        "whisper": {"model": WHISPER_MODEL, "language": "en", "decoding": "greedy, temperature 0"},
        "transcripts": {lab: {"prompt": pr, "heard": audio.log[pr][0], "exact": audio.log[pr][1]}
                        for lab, pr in zip(labels, prompts)},
        "exact": sum(audio.log[pr][1] for pr in prompts), "of": len(prompts),
        **describe(rows, labels),
    }
    with open(os.path.join(OUT, "d47_build.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    print(json.dumps({k: report[k] for k in ("exact", "of", "within", "nearest_text")}, indent=1))
    print("text-image", report["per_class"]["text_image"])
    print("text-audio", report["per_class"]["text_audio"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
