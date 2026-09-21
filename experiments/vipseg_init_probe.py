"""Zero-training probe of D-20: the paper's baseline on VIP-Seg's trained encoder, no training at all.

    python experiments/vipseg_init_probe.py --data_path datasets/S3DIS/blocks_bs1_s1 --checkpoint vipseg_S0_N2_K1.pt

The paper's Table 4 baseline, "a plain VIP-Seg backbone with masked average pooling and single-step
prototype matching" (§4.3), scores 82.72 on S0, above VIP-Seg's own 72.20. Trained from scratch ours
scores 49.08. If the paper's baseline sat on VIP-Seg's trained encoder, masked average pooling and a
dot product over VIP-Seg's own features should land near the paper's row with no training at all.
VIP-Seg trained on the S0 training classes only, so scoring the S0 test classes leaks nothing.
Diagnostic only; it breaks guardrail #1 [DECISION D-20].
"""

import argparse
import json
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import train  # noqa: E402
from pipeline.episodes import build_eval_dataset, read_class_names  # noqa: E402
from pipeline.evaluation import accumulated_miou, collect_predictions  # noqa: E402
from pipeline.metrics_alt import alternative_metrics  # noqa: E402
from pipeline.vipseg_baseline import init_features_from_vipseg  # noqa: E402

CONFIGS = {  # name: train.py switches
    "baseline": ["--use_lma", "false", "--num_stages", "0"],
    "baseline_l2": ["--use_lma", "false", "--num_stages", "0", "--l2norm_point_proto", "true"],
}


class _Print:
    def cprint(self, text):
        pass


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=True)
    p.add_argument("--checkpoint", required=True, help="VIP-Seg's released S0 2-way 1-shot checkpoint")
    p.add_argument("--out_dir", default="results/vipinit")
    args = p.parse_args(argv)
    device = torch.device("cuda")
    os.makedirs(args.out_dir, exist_ok=True)
    names = read_class_names(args.data_path, "s3dis")
    dataset = build_eval_dataset(args.data_path, "s3dis", 0, 2, 1, mode="test", seed=0)  # fixed100 (D-08)
    for name, switches in CONFIGS.items():
        train.seed_everything(0)
        targs = train.parse_args(["--dataset", "s3dis", "--data_path", args.data_path, "--cvfold", "0",
                                  "--n_way", "2", "--k_shot", "1"] + switches)
        model = train.build_model(train.model_config(targs)).to(device)
        init_features_from_vipseg(model, args.checkpoint)
        preds, gts, l2c = collect_predictions(model, dataset, names, device)
        miou = accumulated_miou(_Print(), preds, gts, l2c, list(dataset.classes))
        extra = alternative_metrics(preds, gts, l2c, list(dataset.classes))
        result = {"run": f"{name}_vipinit_zeroshot", "miou": miou, "episodes": len(dataset), "extra": extra}
        with open(os.path.join(args.out_dir, f"{name}_zeroshot.json"), "w") as f:
            json.dump(result, f)
        print(f"[probe] {name} on VIP-Seg's trained encoder, no training: mIoU {miou:.4f} | "
              + " | ".join(f"{k}={v:.4f}" for k, v in extra.items()), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
