"""Episodic training of CascadeProto (spec 04 §4-§5).

Every episode comes from the inherited VIP-Seg loader through pipeline.episodes; there is no
synthetic data here. `--dry_run true` means real data, one optimiser step.

    python train.py --dataset s3dis --data_path datasets/S3DIS/blocks_bs1_s1 --cvfold 0 --n_way 2 --k_shot 1
"""

import argparse
import math
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from pipeline.episodes import (AUGMENT_CONFIG, EPISODES_PER_BATCH, NUM_POINT, PC_ATTRIBS, SCHEDULE,
                               EpisodeCollate, build_eval_dataset, build_train_dataset, read_class_names)
from pipeline.evaluation import evaluate
from pipeline.model_api import episode_loss
from utils.logger import IOStream

DRY_RUN_VALID_EPISODES = 5


def str2bool(v: str) -> bool:
    if v.lower() in ("true", "1", "yes"):
        return True
    if v.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"boolean expected, got {v!r}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="CascadeProto episodic training (spec 04)")
    p.add_argument("--dataset", required=True, choices=sorted(SCHEDULE))
    p.add_argument("--data_path", required=True, help="the blocks_bs1_s1 directory (04 §2.2)")
    p.add_argument("--cvfold", type=int, required=True, choices=[0, 1])
    p.add_argument("--n_way", type=int, required=True, choices=[2, 3])
    p.add_argument("--k_shot", type=int, required=True, choices=[1, 5])
    p.add_argument("--modality", default="text", choices=["text", "image", "audio"])
    # Ablation switches of spec 01 §3 / D-17; the defaults are the full model.
    p.add_argument("--use_lma", type=str2bool, default=True)
    p.add_argument("--num_stages", type=int, default=4)
    p.add_argument("--use_gate", type=str2bool, default=True)
    p.add_argument("--use_adrm", type=str2bool, default=True)
    p.add_argument("--logit_scale", default="none", choices=["none", "sqrt_D"])
    p.add_argument("--l2norm_point_proto", type=str2bool, default=False)
    p.add_argument("--clip_variant", default="ViT-B/16", help="frozen CLIP for the text prompts [D-13]")
    p.add_argument("--eval_noise", default="zero", choices=["zero", "sample", "mean_of_M"], help="[D-06]")
    p.add_argument("--gmmn_fg_mode", default="joint", choices=["joint", "per_class"], help="[D-04]")
    p.add_argument("--gmmn_detach_point", type=str2bool, default=False, help="[D-04]")
    p.add_argument("--cross_attn", default="channel", choices=["channel", "two_hop"], help="[D-01]")
    p.add_argument("--cross_attn_scale", default="sqrt_d", choices=["sqrt_d", "sqrt_D"], help="[D-01]")
    p.add_argument("--gate_target", default="prototype", choices=["prototype", "features"], help="[D-02]")
    p.add_argument("--fusion_weight", default="per_query", choices=["per_query", "per_class"], help="[D-11]")
    p.add_argument("--diffusion_input", default="post_relu", choices=["post_relu", "pre_relu"], help="[D-14]")
    p.add_argument("--epochs", type=int, default=None, help="default: 50 (S3DIS) / 30 (ScanNet) [D-12]")
    p.add_argument("--episodes_per_epoch", type=int, default=None, help="default: 480 / 800 [D-12]")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--lr_step_epochs", type=int, default=10)
    p.add_argument("--lr_gamma", type=float, default=0.5)
    p.add_argument("--valid_every", type=int, default=10, help="epochs between validations [D-15]")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--save_dir", default="log_cascadeproto")
    p.add_argument("--dry_run", type=str2bool, default=False)
    args = p.parse_args(argv)
    schedule = SCHEDULE[args.dataset]
    args.epochs = args.epochs or schedule["epochs"]
    args.episodes_per_epoch = args.episodes_per_epoch or schedule["episodes_per_epoch"]
    if args.episodes_per_epoch % EPISODES_PER_BATCH:
        p.error(f"--episodes_per_epoch must be a multiple of {EPISODES_PER_BATCH}")
    return args


def seed_everything(seed: int) -> None:
    random.seed(seed)  # loader augmentation uses `random` [VIPSEG dataloaders/loader.py:92-100]
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def model_config(args):
    from models.cascadeproto import CascadeProtoConfig

    return CascadeProtoConfig(use_lma=args.use_lma, num_stages=args.num_stages, use_gate=args.use_gate,
                              use_adrm=args.use_adrm, modality=args.modality, logit_scale=args.logit_scale,
                              l2norm_point_proto=args.l2norm_point_proto, clip_variant=args.clip_variant,
                              eval_noise=args.eval_noise, gmmn_fg_mode=args.gmmn_fg_mode,
                              gmmn_detach_point=args.gmmn_detach_point, cross_attn=args.cross_attn,
                              cross_attn_scale=args.cross_attn_scale, gate_target=args.gate_target,
                              fusion_weight=args.fusion_weight, diffusion_input=args.diffusion_input)


def build_model(config, feature_extractor=None) -> torch.nn.Module:
    """CascadeProto with the switches of `config`; unimplemented switches raise (01 §3)."""
    from models.cascadeproto import CascadeProto

    return CascadeProto(config, feature_extractor)


def run_dir(args) -> str:
    variant = args.modality if args.use_lma else "point"
    tag = f"_T{args.num_stages}" + ("" if args.use_adrm or args.num_stages == 0 else "_noadrm")
    tag += "" if args.use_gate or args.num_stages == 0 else "_nogate"
    return os.path.join(args.save_dir, f"{args.dataset}_S{args.cvfold}_N{args.n_way}_K{args.k_shot}_{variant}{tag}")


def train_steps(model, optimizer, batches, device):
    """One optimiser step per batch of episodes; the loss is the mean over the batch (02 §7, D-12).

    Yields (L_total, L_GMMN) of the batch, both averaged over its episodes.
    """
    model.train()
    for episodes in batches:
        episodes = [ep.to(device) for ep in episodes]
        outputs = [model(ep) for ep in episodes]
        loss = torch.stack([episode_loss(out, ep) for out, ep in zip(outputs, episodes)]).mean()  # scalar
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        yield loss.item(), torch.stack([out.loss_gmmn.detach() for out in outputs]).mean().item()


def main(argv=None):
    args = parse_args(argv)
    seed_everything(args.seed)
    device = torch.device("cuda")
    out_dir = run_dir(args)
    os.makedirs(out_dir, exist_ok=True)
    logger = IOStream(os.path.join(out_dir, "log_train.txt"))
    logger.cprint(f"args: {vars(args)}")
    logger.cprint(f"data: {os.path.abspath(args.data_path)} | num_point={NUM_POINT} pc_attribs={PC_ATTRIBS} "
                  f"augmentation={AUGMENT_CONFIG}")

    class_names = read_class_names(args.data_path, args.dataset)
    steps_per_epoch = args.episodes_per_epoch // EPISODES_PER_BATCH
    total_episodes = EPISODES_PER_BATCH if args.dry_run else args.epochs * args.episodes_per_epoch
    train_set = build_train_dataset(args.data_path, args.dataset, args.cvfold, args.n_way, args.k_shot,
                                    num_episode=total_episodes)
    train_loader = DataLoader(train_set, batch_size=EPISODES_PER_BATCH, shuffle=False,
                              num_workers=args.num_workers, worker_init_fn=seed_worker,
                              collate_fn=EpisodeCollate(class_names), drop_last=True)
    valid_set = build_eval_dataset(args.data_path, args.dataset, args.cvfold, args.n_way, args.k_shot,
                                   mode="valid", seed=args.seed)
    logger.cprint(f"train classes {list(train_set.classes)} | test classes {list(valid_set.classes)} | "
                  f"{total_episodes} training episodes, {steps_per_epoch} steps/epoch")

    config = model_config(args)
    model = build_model(config).to(device)
    logger.cprint(f"model config: {config.to_dict()}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_epochs, gamma=args.lr_gamma)

    best_miou, step, epoch, epoch_losses, epoch_gmmn, t0 = -math.inf, 0, 0, [], [], time.time()
    for loss, gmmn in train_steps(model, optimizer, train_loader, device):
        step += 1
        epoch_losses.append(loss)
        epoch_gmmn.append(gmmn)
        if args.dry_run:
            logger.cprint(f"[dry run] one step on {EPISODES_PER_BATCH} real episodes, loss {loss:.4f} "
                          f"(L_GMMN {gmmn:.4f})")
            miou = evaluate(model, valid_set, class_names, logger, device, max_episodes=DRY_RUN_VALID_EPISODES)
            logger.cprint(f"[dry run] valid mIoU on {DRY_RUN_VALID_EPISODES} episodes: {miou:.4f}")
            return 0
        if step % steps_per_epoch:
            continue
        epoch += 1
        scheduler.step()
        logger.cprint(f"epoch {epoch}/{args.epochs} | loss {np.mean(epoch_losses):.4f} | "
                      f"L_GMMN {np.mean(epoch_gmmn):.4f} | lr {scheduler.get_last_lr()[0]:.2e} | "
                      f"{time.time() - t0:.0f}s")
        epoch_losses, epoch_gmmn = [], []
        if epoch % args.valid_every == 0 or epoch == args.epochs:
            miou = evaluate(model, valid_set, class_names, logger, device)
            logger.cprint(f"epoch {epoch} | valid mIoU {miou:.4f}")
            if miou > best_miou:
                best_miou = miou
                torch.save({"model": model.state_dict(), "config": config.to_dict(), "epoch": epoch,
                            "valid_miou": miou, "args": vars(args)},
                           os.path.join(out_dir, "best.pt"))
    torch.save({"model": model.state_dict(), "config": config.to_dict(), "epoch": epoch, "args": vars(args)},
               os.path.join(out_dir, "last.pt"))
    logger.cprint(f"done: best valid mIoU {best_miou:.4f}; evaluate best.pt and last.pt with eval.py [D-15]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
