"""Episodic training of CascadeProto (spec 04 §4-§5).

Every episode comes from the inherited VIP-Seg loader through pipeline.episodes; there is no
synthetic data here. `--dry_run true` means real data, one optimiser step.

A resume checkpoint (`resume.pt`: weights, optimiser, scheduler, counters, random states) is written
after every epoch; `--resume true` continues an interrupted run from it and gives the same episodes
and random draws as an uninterrupted run (training episode i is seeded by (seed, i)).

    python train.py --dataset s3dis --data_path datasets/S3DIS/blocks_bs1_s1 --cvfold 0 --n_way 2 --k_shot 1
"""

import argparse
import math
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from pipeline.episodes import (AUGMENT_CONFIG, EPISODES_PER_BATCH, NUM_POINT, PC_ATTRIBS, SCHEDULE,
                               EpisodeCollate, SeededEpisodes, build_eval_dataset, build_train_dataset,
                               read_class_names)
from pipeline.evaluation import evaluate
from pipeline.model_api import episode_loss
from utils.logger import IOStream

DRY_RUN_VALID_EPISODES = 5
RESUME_FILE = "resume.pt"
# Arguments that may differ between an interrupted run and its resumption; all others must match.
RESUME_FREE_ARGS = ("resume", "num_workers", "save_dir", "data_path")


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
    p.add_argument("--cross_attn_norm", default="none", choices=["none", "layernorm"], help="[D-18]")
    p.add_argument("--gate_target", default="prototype", choices=["prototype", "features"], help="[D-02]")
    p.add_argument("--eq19_self", default="none", choices=["none", "gated"], help="[D-19], beyond the paper")
    p.add_argument("--train_classes", default="split", choices=["split", "all"],
                   help="all = also train on the test classes; leakage diagnostic only [D-21]")
    p.add_argument("--init_from_vipseg", default=None,
                   help="VIP-Seg checkpoint whose trained encoder + feature head initialise the model; "
                        "diagnostic only, breaks guardrail #1 [D-20]")
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
    p.add_argument("--resume", type=str2bool, default=False, help="continue from <run dir>/resume.pt if present")
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


def model_config(args):
    from models.cascadeproto import CascadeProtoConfig

    return CascadeProtoConfig(use_lma=args.use_lma, num_stages=args.num_stages, use_gate=args.use_gate,
                              use_adrm=args.use_adrm, modality=args.modality, logit_scale=args.logit_scale,
                              l2norm_point_proto=args.l2norm_point_proto, clip_variant=args.clip_variant,
                              eval_noise=args.eval_noise, gmmn_fg_mode=args.gmmn_fg_mode,
                              gmmn_detach_point=args.gmmn_detach_point, cross_attn=args.cross_attn,
                              cross_attn_scale=args.cross_attn_scale, cross_attn_norm=args.cross_attn_norm,
                              gate_target=args.gate_target, eq19_self=args.eq19_self,
                              fusion_weight=args.fusion_weight, diffusion_input=args.diffusion_input)


def build_model(config, feature_extractor=None) -> torch.nn.Module:
    """CascadeProto with the switches of `config`; unimplemented switches raise (01 §3)."""
    from models.cascadeproto import CascadeProto

    return CascadeProto(config, feature_extractor)


def run_dir(args) -> str:
    variant = args.modality if args.use_lma else "point"
    tag = f"_T{args.num_stages}" + ("" if args.use_adrm or args.num_stages == 0 else "_noadrm")
    tag += "" if args.use_gate or args.num_stages == 0 else "_nogate"
    tag += "" if args.seed == 0 else f"_seed{args.seed}"
    tag += "_vipinit" if getattr(args, "init_from_vipseg", None) else ""  # [D-20]
    tag += "_leak" if getattr(args, "train_classes", "split") == "all" else ""  # [D-21]
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


def random_states() -> dict:
    return {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def set_random_states(states: dict) -> None:
    torch.set_rng_state(states["torch"])
    np.random.set_state(states["numpy"])
    random.setstate(states["python"])
    if states["cuda"]:
        torch.cuda.set_rng_state_all(states["cuda"])


def atomic_save(obj, path: str) -> None:
    """Write to a temporary file and rename, so an interruption never leaves a truncated checkpoint."""
    torch.save(obj, path + ".tmp")
    os.replace(path + ".tmp", path)


def comparable_args(args) -> dict:
    return {k: v for k, v in vars(args).items() if k not in RESUME_FREE_ARGS}


def train_loop(args, model, optimizer, scheduler, train_set, class_names, validate, config, out_dir, logger,
               device, stop_after_epoch=None) -> dict:
    """Epochs state["epoch"]+1 .. args.epochs; validation every args.valid_every epochs (D-15).

    One DataLoader per epoch over the episodes of that epoch; `resume.pt` after every epoch.
    `stop_after_epoch` simulates an interruption (tests only).
    """
    state = {"epoch": 0, "best_miou": -math.inf}
    resume_path = os.path.join(out_dir, RESUME_FILE)
    if args.resume and os.path.isfile(resume_path):
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)  # RNG states must stay on the CPU
        if ckpt["args"] != comparable_args(args):
            diff = {k for k in ckpt["args"] if ckpt["args"][k] != comparable_args(args).get(k)}
            raise ValueError(f"cannot resume {resume_path}: arguments differ in {sorted(diff)}")
        model.load_state_dict(ckpt["model"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        state = ckpt["state"]
        set_random_states(ckpt["random"])
        logger.cprint(f"resumed from {resume_path} after epoch {state['epoch']}")
    t0 = time.time()
    while state["epoch"] < args.epochs:
        first = state["epoch"] * args.episodes_per_epoch
        loader = DataLoader(Subset(train_set, range(first, first + args.episodes_per_epoch)),
                            batch_size=EPISODES_PER_BATCH, shuffle=False, num_workers=args.num_workers,
                            collate_fn=EpisodeCollate(class_names), drop_last=True)
        losses = list(train_steps(model, optimizer, loader, device))
        state["epoch"] += 1
        scheduler.step()
        logger.cprint(f"epoch {state['epoch']}/{args.epochs} | loss {np.mean([l for l, _ in losses]):.4f} | "
                      f"L_GMMN {np.mean([g for _, g in losses]):.4f} | lr {scheduler.get_last_lr()[0]:.2e} | "
                      f"{time.time() - t0:.0f}s")
        if state["epoch"] % args.valid_every == 0 or state["epoch"] == args.epochs:
            miou = validate(model)
            logger.cprint(f"epoch {state['epoch']} | valid mIoU {miou:.4f}")
            if miou > state["best_miou"]:
                state["best_miou"] = miou
                atomic_save({"model": model.state_dict(), "config": config.to_dict(), "epoch": state["epoch"],
                             "valid_miou": miou, "args": vars(args)}, os.path.join(out_dir, "best.pt"))
        atomic_save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "scheduler": scheduler.state_dict(), "state": dict(state), "random": random_states(),
                     "config": config.to_dict(), "args": comparable_args(args)}, resume_path)
        if stop_after_epoch is not None and state["epoch"] >= stop_after_epoch:
            return state
    atomic_save({"model": model.state_dict(), "config": config.to_dict(), "epoch": state["epoch"],
                 "args": vars(args)}, os.path.join(out_dir, "last.pt"))
    return state


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
    train_set = SeededEpisodes(build_train_dataset(args.data_path, args.dataset, args.cvfold, args.n_way,
                                                   args.k_shot, num_episode=total_episodes,
                                                   train_classes=args.train_classes), args.seed)
    valid_set = build_eval_dataset(args.data_path, args.dataset, args.cvfold, args.n_way, args.k_shot,
                                   mode="valid", seed=args.seed)
    logger.cprint(f"train classes {list(train_set.classes)} | test classes {list(valid_set.classes)} | "
                  f"{total_episodes} training episodes, {steps_per_epoch} steps/epoch")

    config = model_config(args)
    model = build_model(config).to(device)
    if args.init_from_vipseg:  # before resume.pt is read, so a resumed run keeps its own weights [D-20]
        from pipeline.vipseg_baseline import init_features_from_vipseg

        init_features_from_vipseg(model, args.init_from_vipseg)
        logger.cprint(f"features initialised from VIP-Seg checkpoint {args.init_from_vipseg} [D-20]")
    logger.cprint(f"model config: {config.to_dict()}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_epochs, gamma=args.lr_gamma)

    if args.dry_run:
        loader = DataLoader(train_set, batch_size=EPISODES_PER_BATCH, shuffle=False, num_workers=args.num_workers,
                            collate_fn=EpisodeCollate(class_names), drop_last=True)
        loss, gmmn = next(train_steps(model, optimizer, loader, device))
        logger.cprint(f"[dry run] one step on {EPISODES_PER_BATCH} real episodes, loss {loss:.4f} "
                      f"(L_GMMN {gmmn:.4f})")
        miou = evaluate(model, valid_set, class_names, logger, device, max_episodes=DRY_RUN_VALID_EPISODES)
        logger.cprint(f"[dry run] valid mIoU on {DRY_RUN_VALID_EPISODES} episodes: {miou:.4f}")
        return 0

    state = train_loop(args, model, optimizer, scheduler, train_set, class_names,
                       lambda m: evaluate(m, valid_set, class_names, logger, device), config, out_dir, logger, device)
    logger.cprint(f"done: best valid mIoU {state['best_miou']:.4f}; evaluate best.pt and last.pt with eval.py [D-15]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
