### D-08 — Evaluation protocol and metric · `LOCKED`

* **Problem.** L1 gives no mIoU formula: "Performance is reported as mean IoU (mIoU) averaged over S0 and S1 across 600 randomly sampled episodes" [PAPER §4.1].
* **Decision.** Primary = the VIP-Seg protocol:
  * Test set = `MyTestDataset`, 100 episodes per class combination, fixed and cached as `.h5` [VIPSEG dataloaders/loader.py:230-267] [VIPSEG scripts/vipseg_eval_s3dis.sh `N_TEST_EPISODES=100`], `n_queries = 1`.
  * Metric = TP/FP/FN accumulated over **all** test episodes per global test class, IoU per class, mean over foreground classes only (background excluded) [VIPSEG runs/training_free.py:56-59].
  * Report S0, S1 and their average separately, as in Tables 2–3.
  * L2 evidence that this reproduces the paper's baseline numbers: VIP-Seg log directory `log_s3dis_VIPSeg/log_S0_N2_K1_0.722026` matches VIP-Seg's 72.20 in Table 6.
* **Ablation flag.** `eval_protocol = {vipseg_fixed (default), random600}`.
