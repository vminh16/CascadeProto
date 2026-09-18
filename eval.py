"""Evaluation on the fixed test episodes of spec 04 §6.

    python eval.py --dataset s3dis --data_path datasets/S3DIS/blocks_bs1_s1 --cvfold 0 --n_way 2 --k_shot 1 \
        --checkpoint log_cascadeproto/s3dis_S0_N2_K1_text/best.pt

`--model vipseg` scores VIP-Seg's released checkpoint on the same episodes and metric (05 §4).
"""

import argparse
import os

import numpy as np
import torch

from dataloaders.loader import MyDataset
from pipeline.episodes import (N_QUERIES, NUM_POINT, PC_ATTRIBS, SCHEDULE, WAY_NUM, WAY_RATIO,
                               build_eval_dataset, read_class_names)
from pipeline.evaluation import evaluate
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
    p.add_argument("--modality", default="text", choices=["text", "image", "audio"])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model", default="cascadeproto", choices=["cascadeproto", "vipseg"])
    p.add_argument("--eval_protocol", default="fixed100", choices=["fixed100", "random600"],
                   help="fixed100: 100 cached episodes per class combination (primary, D-08)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save_dir", default="log_eval")
    p.add_argument("--dry_run", type=str2bool, default=False)
    return p.parse_args(argv)


def load_model(args, device) -> torch.nn.Module:
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")
    if args.model == "vipseg":
        from pipeline.vipseg_baseline import VIPSegBaseline

        return VIPSegBaseline(args.checkpoint).to(device)
    model = build_model(args).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)["model"]
    model.load_state_dict(state, strict=True)
    return model


def main(argv=None):
    args = parse_args(argv)
    seed_everything(args.seed)
    device = torch.device("cuda")
    out_dir = os.path.join(args.save_dir, f"{args.model}_{args.dataset}_S{args.cvfold}_N{args.n_way}_K{args.k_shot}")
    os.makedirs(out_dir, exist_ok=True)
    logger = IOStream(os.path.join(out_dir, "log_eval.txt"))
    logger.cprint(f"args: {vars(args)}")

    model = load_model(args, device)
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
    miou = evaluate(model, dataset, class_names, logger, device, max_episodes=n)
    logger.cprint(f"[TEST] {args.model} {args.dataset} S{args.cvfold} {args.n_way}-way {args.k_shot}-shot "
                  f"mIoU: {miou:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
