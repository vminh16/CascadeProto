### D-15 — Model selection · `LOCKED`

* **Problem.** L1 is silent. L2 evaluates every 2,000 iterations on a `MyTestDataset(mode='valid')` built from the **test classes** and keeps the best checkpoint [VIPSEG runs/training.py], i.e. model selection on the test classes.
* **Decision.** Primary = VIP-Seg-compatible selection, so numbers are comparable with Tables 2–3 (validate every 10 epochs on the `valid` episode set, keep the best). Always also log the **last-epoch** checkpoint's test mIoU so the selection effect is visible.
* **Ablation flag.** `model_selection = {best_valid (default), last}`.
