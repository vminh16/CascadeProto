### D-13 — Modality scope, CLIP variant, prompts · `LOCKED`

* **Problem.** L1 names "CLIP [17]" and Fig.1 shows "Audio → Whisper → CLIP" and "Image → CLIP", but gives no CLIP variant, no background prompt, no image or audio source, and no Whisper usage details. The only prompt shown is "This point cloud represents the chair." [PAPER Fig.1].
* **Decision.**
  1. Implement **text first**; image and audio are deferred and must fail loudly if selected before they exist.
  2. CLIP variant fixed by config, default `ViT-B/16`, logged with every run. Embeddings are L2-normalised (standard CLIP usage; not stated in L1), computed once per class and cached; CLIP is never reloaded per episode.
  3. Foreground prompt `"This point cloud represents the {class}."` [PAPER Fig.1]. Background prompt `"This point cloud represents the background."` (assumption; not in L1). `{class}` is the class name exactly as written in the dataset's class-name file (e.g. `shower curtain`, `otherfurniture`, `refridgerator`).
  4. Audio (when implemented): Whisper transcription → CLIP text encoder, following the arrow order of Fig.1.
