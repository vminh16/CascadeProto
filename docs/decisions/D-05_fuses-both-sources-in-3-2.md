### D-05 — "Fuses both sources" in §3.2 · `LOCKED`

* **Conflict inside L1.** §3.2: "Learnable Modality Adapters (LMA) project CLIP text and image embeddings into the point cloud feature space. A GMMN-based distribution matching module then fuses both sources". Abstract: "enabling flexible single-modality semantic enrichment"; Tables 2–3 report separate Text / Image / Audio rows; Eq.4 defines one adapter per modality.
* **Decision.** One modality per run. `E_fused := E_adapted^(m)` for the selected modality m [PAPER Eq.4, Eq.6]. "Both sources" is read as the point prototype and the modal prototype combined by Eq.9.
* **Sub-decision (generator input).** Eq.6 writes `G(E_fused, z)` without a noise dimension; use `z ∈ R^{(N+1)×D}` concatenated with `E_fused` (input 2D = 256) followed by a three-layer MLP [PAPER Eq.6 "three-layer MLP generator"].
