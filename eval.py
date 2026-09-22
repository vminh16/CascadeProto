"""Evaluation on the fixed test episodes of spec 04 §6.

    python eval.py --dataset s3dis --data_path datasets/S3DIS/blocks_bs1_s1 --cvfold 0 --n_way 2 --k_shot 1 \
        --checkpoint log_cascadeproto/s3dis_S0_N2_K1_text/best.pt

`--model vipseg` scores VIP-Seg's released checkpoint on the same episodes and metric (05 §4).

Protocol guard [DECISION D-22]: a checkpoint is scored only on the test classes of the fold it was
trained on. S0's training classes are fold 1's test classes and vice versa, so scoring a checkpoint
with the other `--cvfold`, or one trained with `--train_classes all` [D-21], scores classes it has
seen. That is refused unless `--allow_seen_classes true` marks the run as a diagnostic.
"""

import argparse
import json
import os

import numpy as np
import torch

from dataloaders.loader import MyDataset
from pipeline.episodes import (N_QUERIES, NUM_POINT, PC_ATTRIBS, SCHEDULE, WAY_NUM, WAY_RATIO,
                               build_eval_dataset, read_class_names)
from pipeline.evaluation import accumulated_miou, collect_predictions
from pipeline.metrics_alt import alternative_metrics
from train import build_model, seed_everything, str2bool
from utils.logger import IOStream

RANDOM_PROTOCOL_EPISODES = 600  # [PAPER §4.1] [DECISION D-08], ablation only
DRY_RUN_EPISODES = 5


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="CascadeProto evaluation (spec 04 §6)")
    p.add_argument("--dataset", required=True, choices=sorted(SCHEDULE))
    p.add_argument("--data_path", required=True)
    p.add_argument("--cvfold", type=int, required=True, choices=[0, 1])
    p.add_argument("--n_way", type=int, required=True, choices=[2, 3])
    p.add_argument("--k_shot", type=int, required=True, choices=[1, 5])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model", default="cascadeproto", choices=["cascadeproto", "vipseg"])
    p.add_argument("--eval_protocol", default="fixed100", choices=["fixed100", "random600"],
                   help="fixed100: 100 cached episodes per class combination (primary, D-08)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save_dir", default="log_eval")
    p.add_argument("--dry_run", type=str2bool, default=False)
    p.add_argument("--result_json", default=None, help="also write the result as JSON to this path")
    p.add_argument("--extra_metrics", type=str2bool, default=False,
                   help="also report alternative mIoU definitions (diagnostic only, pipeline/metrics_alt.py)")
    p.add_argument("--checkpoint_cvfold", type=int, default=None, choices=[0, 1],
                   help="fold the checkpoint was trained on; required when the checkpoint does not record it "
                        "(VIP-Seg's released checkpoints) [D-22]")
    p.add_argument("--allow_seen_classes", type=str2bool, default=False,
                   help="score classes the checkpoint was trained on; leakage diagnostic only, never a result [D-22]")
    return p.parse_args(argv)


def load_model(args, device) -> torch.nn.Module:
    """The model, with `checkpoint_args` = the training arguments stored in the checkpoint (None if absent)."""
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")
    if args.model == "vipseg":
        from pipeline.vipseg_baseline import VIPSegBaseline

        model = VIPSegBaseline(args.checkpoint).to(device)
        model.checkpoint_args = None  # VIP-Seg saves only the model, iteration and IoU [VIPSEG runs/training.py:97-100]
        return model
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    from models.cascadeproto import CascadeProtoConfig

    config = CascadeProtoConfig(**checkpoint["config"])  # the architecture the weights were trained with
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.checkpoint_epoch = checkpoint.get("epoch")
    model.checkpoint_args = checkpoint.get("args")
    return model


def training_fold(args, checkpoint_args) -> int:
    """The fold the checkpoint was trained on: recorded in it, or stated with --checkpoint_cvfold [D-22]."""
    recorded = (checkpoint_args or {}).get("cvfold")
    if recorded is not None and args.checkpoint_cvfold is not None and recorded != args.checkpoint_cvfold:
        raise ValueError(f"--checkpoint_cvfold {args.checkpoint_cvfold} contradicts the checkpoint, which was "
                         f"trained with --cvfold {recorded}")
    fold = recorded if recorded is not None else args.checkpoint_cvfold
    if fold is None:
        raise ValueError(f"{args.checkpoint} does not record the fold it was trained on; pass --checkpoint_cvfold "
                         f"so that the protocol guard can check it [D-22]")
    return fold


def protocol_check(args, checkpoint_args) -> str:
    """'clean', or the reason the run scores seen classes; raises unless --allow_seen_classes [D-22]."""
    fold = training_fold(args, checkpoint_args)
    problems = []
    if fold != args.cvfold:
        problems.append(f"trained on S{fold}, whose training classes are the test classes of S{args.cvfold}")
    if (checkpoint_args or {}).get("train_classes", "split") != "split":
        problems.append("trained with --train_classes all, i.e. on the test classes [D-21]")
    if not problems:
        return "clean"
    if not args.allow_seen_classes:
        raise ValueError("refusing to score classes the checkpoint has seen: " + "; ".join(problems) +
                         ". Pass --allow_seen_classes true only for a leakage diagnostic [D-22]")
    return "SEEN-CLASS DIAGNOSTIC, not a few-shot result: " + "; ".join(problems)


def main(argv=None):
    args = parse_args(argv)
    seed_everything(args.seed)
    device = torch.device("cuda")
    out_dir = os.path.join(args.save_dir, f"{args.model}_{args.dataset}_S{args.cvfold}_N{args.n_way}_K{args.k_shot}")
    os.makedirs(out_dir, exist_ok=True)
    logger = IOStream(os.path.join(out_dir, "log_eval.txt"))
    logger.cprint(f"args: {vars(args)}")

    model = load_model(args, device)
    protocol = protocol_check(args, model.checkpoint_args)  # before any episode is scored [D-22]
    logger.cprint(f"protocol: {protocol} (checkpoint trained on S{training_fold(args, model.checkpoint_args)}, "
                  f"scored on S{args.cvfold})")
    if hasattr(model, "config"):
        logger.cprint(f"model config (from checkpoint): {model.config.to_dict()}")
    class_names = read_class_names(args.data_path, args.dataset)
    if args.eval_protocol == "fixed100":
        dataset = build_eval_dataset(args.data_path, args.dataset, args.cvfold, args.n_way, args.k_shot,
                                     mode="test", seed=args.seed)
    else:
        dataset = MyDataset(args.data_path, args.dataset, cvfold=args.cvfold, num_episode=RANDOM_PROTOCOL_EPISODES,
                            n_way=args.n_way, k_shot=args.k_shot, n_queries=N_QUERIES, mode="test",
                            num_point=NUM_POINT, pc_attribs=PC_ATTRIBS, pc_augm=False,
                            way_ratio=WAY_RATIO, way_num=WAY_NUM)
    n = DRY_RUN_EPISODES if args.dry_run else len(dataset)
    logger.cprint(f"{args.eval_protocol}: {n} of {len(dataset)} episodes | test classes "
                  f"{[class_names[c] for c in np.asarray(dataset.classes)]}")
    preds, gts, label2class = collect_predictions(model, dataset, class_names, device, max_episodes=n)
    miou = accumulated_miou(logger, preds, gts, label2class, list(dataset.classes))
    logger.cprint(f"[TEST] {args.model} {args.dataset} S{args.cvfold} {args.n_way}-way {args.k_shot}-shot "
                  f"mIoU: {miou:.6f}")
    extra = None
    if args.extra_metrics:
        extra = alternative_metrics(preds, gts, label2class, list(dataset.classes))
        logger.cprint("[EXTRA] " + " | ".join(f"{k}={v:.6f}" for k, v in extra.items()))
    if args.result_json:
        with open(args.result_json, "w") as f:
            json.dump({"miou": miou, "protocol": args.eval_protocol, "episodes": n, "checkpoint": args.checkpoint,
                       "epoch": getattr(model, "checkpoint_epoch", None), "dry_run": args.dry_run,
                       "protocol_check": protocol, "cvfold": args.cvfold,
                       "checkpoint_cvfold": training_fold(args, model.checkpoint_args),
                       **({"extra": extra} if extra is not None else {})}, f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
