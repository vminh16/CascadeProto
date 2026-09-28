### D-12 — Epoch definition, augmentation · `LOCKED` (epoch) / `LOCKED` (augmentation)

* **Problem.** L1 gives epochs, batch size and StepLR period but not the number of episodes per epoch or any augmentation. L2 trains for `NUM_ITERS=24000` episodes with batch size 1 on both datasets [VIPSEG scripts/vipseg_s3dis.sh, scripts/vipseg_scannet.sh].
* **Decision (LOCKED).** Match VIP-Seg's total of **24,000 training episodes**, with L1's batch size and epochs:
  * S3DIS: 50 epochs × 480 episodes/epoch = 24,000 episodes = 120 optimiser steps/epoch (batch 4).
  * ScanNet: 30 epochs × 800 episodes/epoch = 24,000 episodes = 200 optimiser steps/epoch (batch 4).
  * StepLR(step = 10 epochs, γ = 0.5) [PAPER §4.1]. Episodes per epoch exposed on the CLI.
  * *Correction (2026-09-17):* the option text confirmed earlier said "≈480 steps/epoch"; 480 is the number of **episodes** per epoch, i.e. 120 steps at batch 4.
  * *Measured (2026-09-21):* VIP-Seg's own schedule (batch 1, 24,000 steps, halving every 7,200) gives the baseline 0.4901 on fixed100 against 0.4908 under this decision, so the choice of batch and decay is not a source of the gap to the paper (`train.py --batch_size`, report §3.3, CHANGELOG 15x).
* **Decision (LOCKED, augmentation).** Use VIP-Seg's training augmentation: `pc_augm` on, `shift 0.1`, `rot 1`, `jitter 1`, `scale 0`, `mirror 0`, `color 0` [VIPSEG scripts/vipseg_s3dis.sh] [VIPSEG main.py:56-66]. No augmentation at test time. Measured effect (2026-09-17): the encoder reads only the normalised `XYZ` and `rgb` columns [VIPSEG models/encoder.py:645], and `XYZ` is recomputed after subtracting the minimum [VIPSEG dataloaders/loader.py:70-74], so the shift augmentation does not change the model input; rotation and jitter do.
