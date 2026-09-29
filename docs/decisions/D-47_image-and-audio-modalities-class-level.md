### D-47 — Image and audio modalities at the class level, as in the paper's Fig. 1 · `PROPOSED`, maintainer request 2026-09-28

* **Problem.** D-13 implemented text only; selecting `modality=image` or `modality=audio` raises (03 §2.2). The
  maintainer asked for both. They serve two purposes:
  - the paper's Tables 2–3 report one row per modality;
  - the phase-16 architecture (D-48) takes class-level priors from every available modality.

  D-13 left the following open: the image source, the audio source, the CLIP image variant, and whether the audio
  path uses Whisper's transcript or its embeddings.
* **What the paper gives.** Fig. 1 draws "Image → CLIP" and "Audio → Whisper → CLIP" [PAPER Fig.1]. Eq. 4 has one
  adapter per modality, and D-05 fixes one modality per run for route A. The paper names no image or audio data.
* **Decision.**
  * **Image, class level.**
    - **Source.** M = 5 images per class name of the dataset (S3DIS: 13 names), taken from Wikimedia Commons with an
      open licence.
    - **Manifest.** `datasets/modality/images_s3dis.json` lists the URL, licence, author and sha256 of each image.
      A download script fetches and verifies the images. A missing or altered image raises.
    - **Background.** 5 images of empty indoor rooms. This is an assumption, as the background prompt is in D-13.
    - **Embedding.** The frozen CLIP image encoder of the variant fixed by D-13 (default ViT-B/16), with CLIP's own
      preprocessing. Each embedding is L2-normalised, the class row is the mean of its M embeddings, renormalised,
      and the result is cached.
    - **Why class-level images.** This follows the paper's arrow "Image → CLIP" into the same per-class slot as the
      text row.
    - **Not in scope.** Per-point 2D features from aligned scene images (the MM-FSS route) need S3DIS's 2D-3D-S
      images and a frozen 2D segmentation model. They wait for a separate maintainer decision.
  * **Audio, class level** (D-13 item 4).
    - **Speech.** The fixed prompt of D-13 ("This point cloud represents the {class}.") is synthesised by an offline
      text-to-speech engine: `espeak-ng`, one voice, fixed rate. The engine and voice are logged.
    - **Transcription.** The audio is transcribed by frozen Whisper (`base`, English, greedy decoding). The transcript
      is encoded by the CLIP text encoder, L2-normalised and cached.
    - **Logged per class:** the transcript, and whether it equals the prompt after lower-casing and punctuation
      removal.
    - **Expected result, stated before any run.** Under this pipeline the audio row is a noisy copy of the text row:
      I(audio; class) ≤ I(text; class) by the data-processing inequality. The audio row equals the text row exactly
      whenever the transcript is exact. The audio row is implemented because the paper reports it, not because it
      is expected to add information.
  * **Interface.** `modality ∈ {text, image, audio}` returns `E_CLIP [N+1, 512]` with the row order of 02 §0. Route A
    (LMA, Eq. 4–9) accepts any of the three, one per run (D-05). D-48 layer [6] may use several at once, each with its
    own per-episode weight.
  * **Guardrails.** Only frozen CLIP and Whisper weights are loaded (01 §2.1). No point-cloud weights. No silent
    fallback: a missing asset, engine or model raises.
* **Checks and tests** (05, new section; written before the code).
  - The manifest's hashes are verified.
  - Rows are unit-norm and in the correct order.
  - The cache round-trips.
  - The image row is invariant to the order of the M images.
  - An unknown modality raises.
  - The audio transcript equality is reported.
  - A text-vs-image-vs-audio cosine table per class is written to `results/phase16_d47/`, descriptive only.
* **Rules.** None on score. This decision implements inputs. The scores are read by D-48 (layer [6]) and, for
  route A's Table 2 rows, by whichever run uses them.
* **Cost** (not measured). Minutes on CPU: 70 images, 14 syntheses and transcriptions.
* **Affects.**
  - `models/clip_image.py` (new), `models/clip_audio.py` (new), `models/clip_text.py` (shared helpers);
  - the `modality` option in `models/cascadeproto.py`;
  - `datasets/modality/` (manifest, not the images), `scripts_modality/fetch_images.py` (new);
  - `requirements.txt` (`openai-whisper`; `espeak-ng` as a system package);
  - `tests/test_modalities.py`;
  - 03 §2.2 (deferred → implemented, with tags to this decision).

#### Amendment 1 (2026-09-29, implementation notes, before the rows were built)

* **Status.** Accepted by the maintainer's request of 2026-09-28 ("multimodal is not only text, implement image and
  audio"); implemented as below.
* **Manifest location.** `datasets/` is git-ignored, so the manifest is `assets/modality/images_s3dis.json` (tracked);
  the images go to `datasets/modality/images/<class>/<i>.<ext>` (not tracked). The fetch script is
  `preprocess/fetch_modality_images.py` (not `scripts_modality/`).
* **How the images were chosen.** For each class name, a contact sheet of up to 20 candidates was drawn from a
  Wikimedia Commons category (`Office chairs`, `Office desks`, `Couches`, `Bookcases`, `Whiteboards`,
  `Dropped ceilings`, `Floor tiles`, `White walls`, `Beams`, `Pillars`, `Windows`, `Interior doors`, `Offices`,
  `Empty rooms`) or, where the category gave outdoor or museum pictures (beam, column), from a Commons search. The
  agent picked 5 per class by eye, preferring indoor office scenes like S3DIS's; `clutter` uses cluttered desks. The
  choice is subjective and is recorded, not tuned: no score was looked at. Only CC0, public-domain, CC BY and
  CC BY-SA files were accepted, and each entry keeps its author and licence URL for attribution.
* **Verification.** Each download is checked against the sha1 that Wikimedia publishes; its sha256 is then recorded
  in the manifest, and the image front-end checks the sha256 at every load. Wikimedia rate-limits bursts (HTTP 429),
  so the fetcher waits 5 s between files, honours `Retry-After`, and names the repository in its User-Agent.
* **Audio settings.** `espeak-ng` 1.51, voice `en-us`, 150 words per minute; Whisper `base` (openai-whisper
  20250625), English, temperature 0 without fallback, no conditioning on previous text; fp16 on the GPU.
* **Code.** `models/clip_image.py`, `models/clip_audio.py`, `modality_embedding` in `models/cascadeproto.py` (the
  `NotImplementedError` for image and audio is removed), `models/clip_text.py` keeps CLIP's preprocessing,
  `experiments/d47_build.py` builds the three rows with the real models and writes the descriptive tables.
  Tests MOD-1…8 (05 §3.8z); ABL-3, CP-6 and PIPE-6 now expect the front-ends instead of an error.

#### Outcome (2026-09-29): rows built, descriptive only

`results/phase16_d47/SUMMARY.md`. All 70 images fetched and verified. Audio: 10 of 14 transcripts exact; the other
four (background → "markplund", ceiling → "reprimandes", wall → "world", clutter → "cloud") give rows at cosine
0.90–0.97 to the text rows, and the wall and clutter audio rows lie nearest the background's text row. Image: the
cosine to the same class's text row is 0.25–0.31 (CLIP's image–text gap); 12 of 14 image rows lie nearest their own
class's text row (beam → ceiling, table → clutter); between classes the image rows are more spread than the text rows
(mean cosine 0.821 against 0.900). No segmentation run uses them yet (staging: base first, multimodal second).
