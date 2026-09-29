# Phase 16 D-47: class-level image and audio rows (2026-09-29)

Built by `experiments/d47_build.py` on the RTX 3090 instance, on the CPU (the GPU was training D-49), from the 70
images of `assets/modality/images_s3dis.json` (all fetched, Wikimedia sha1 checked, sha256 recorded). Files:
`rows_ViT-B-16.npz` (text, image, audio, each [14, 512], row 0 = background), `d47_build.json`, `d47_build.log`.
D-47 has no rule on these numbers; they describe the inputs, nothing here is a segmentation score.

## Audio: espeak-ng 1.51 (en-us, 150 wpm) → Whisper base → CLIP text

10 of 14 transcripts are exact after lower-casing and punctuation removal. The four errors:

| Row | Heard | cos(audio, text) |
| :--- | :--- | ---: |
| background | "This point cloud represents the markplund." | 0.903 |
| ceiling | "This point cloud reprimandes the ceiling." | 0.967 |
| wall | "This point cloud represents the world." | 0.914 |
| clutter | "This point cloud represents the cloud." | 0.922 |

The other ten rows equal the text rows exactly (cosine 1.000), as D-47 stated before the run. Two of the errors hit
rows that S1's test episodes use: the background row (every episode) and wall (a test class of S1). The nearest text
row of the wall audio row and of the clutter audio row is the background's, so for these classes the audio row
carries the wrong class. This is the data-processing inequality made concrete: the audio path can only lose what
the text path has.

## Image: 5 images per class through the CLIP image encoder

| Quantity | text | image | audio |
| :--- | ---: | ---: | ---: |
| Mean cosine between different classes | 0.900 | 0.821 | 0.886 |
| Largest cosine between different classes | 0.952 | 0.920 | 0.945 |
| Rows whose nearest text row is their own class | 14 / 14 | 12 / 14 | 12 / 14 |

* cos(image row, text row of the same class) lies between 0.252 (table) and 0.311 (chair). This is CLIP's usual gap
  between its image and text embeddings, so image and text rows cannot be mixed as if they were one space; a
  consumer of both needs its own adapter per modality (as Eq. 4 has) or per-modality centring.
* Two image rows sit nearer another class's text: beam → ceiling (the beam photographs show ceilings with beams)
  and table → clutter (office desks carry objects).
* The image rows are more spread than the text rows (0.821 against 0.900 between classes). The prompt template of
  D-13 shares most of its tokens across classes; the photographs do not. This says nothing yet about segmentation.

## Reading

* The three front-ends work end to end, with the same [N+1, 512] contract; route A can now run one modality per run
  (D-05), and D-48 layer [6] can read several.
* Audio is text with four corrupted rows, as expected. Image adds a different view of each class; whether it helps
  prototypes is not measured here. By the project's staging (base first, multimodal second), no multimodal run is
  started from this phase.
