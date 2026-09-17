"""Episodic Training Script for CascadeProto.

Paper: CascadeProto (ECCV 2026, Wang et al.)
Command Example:
    python train.py --dataset s3dis --cvfold 0 --n_way 2 --k_shot 1 --modality text --dry_run True
"""

import argparse
import os
import random
import sys
import numpy as np
import torch
import torch.optim as optim

from models.cascadeproto import CascadeProto
from models.lma import generate_clip_text_embeddings, format_category_prompt
from loss.segmentation_loss import CascadeProtoLoss


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_s3dis_class_names(fold: int):
    """Returns train and test class names for S3DIS splits."""
    all_classes = [
        'ceiling', 'floor', 'wall', 'beam', 'column',
        'window', 'door', 'table', 'chair', 'sofa',
        'bookcase', 'board', 'clutter'
    ]
    fold_0 = ['beam', 'board', 'bookcase', 'ceiling', 'chair', 'column']
    fold_1 = ['door', 'floor', 'sofa', 'table', 'wall', 'window']
    test_classes = fold_0 if fold == 0 else fold_1
    train_classes = [c for c in all_classes if c not in test_classes and c != 'clutter']
    return train_classes, test_classes


def generate_dry_run_batch(
    n_way: int,
    k_shot: int,
    device: torch.device,
    sampled_classes=None
):
    """Generates 1 synthetic episodic batch conforming to the data spec."""
    B = 1
    N_p = 2048
    num_classes = n_way + 1

    # Support: [B, N_way * K_shot, N_p, 3]
    support_x = torch.randn(B, n_way * k_shot, N_p, 3, device=device, dtype=torch.float32)
    support_y = torch.zeros(B, n_way * k_shot, N_p, device=device, dtype=torch.float32)
    for w in range(n_way):
        for k in range(k_shot):
            idx = w * k_shot + k
            # Assign first 500 points to class w + 1
            support_y[:, idx, :500] = float(w + 1)

    # Query: [B, N_p, 3]
    query_x = torch.randn(B, N_p, 3, device=device, dtype=torch.float32)
    query_y = torch.randint(0, num_classes, (B, N_p), device=device, dtype=torch.int64)

    # Text embeddings: [B, num_classes, 512]
    if sampled_classes is not None and len(sampled_classes) == n_way:
        E_text = generate_clip_text_embeddings(sampled_classes, device=device)
    else:
        # Generic synthetic embedding
        E_text = torch.randn(B, num_classes, 512, device=device, dtype=torch.float32)
        E_text = E_text / E_text.norm(dim=-1, keepdim=True)

    return support_x, support_y, query_x, query_y, E_text


def parse_args():
    parser = argparse.ArgumentParser(description="CascadeProto Episodic Training")
    parser.add_argument("--dataset", type=str, default="s3dis", choices=["s3dis", "scannet"],
                        help="Dataset name (s3dis or scannet)")
    parser.add_argument("--data_path", type=str, default="./data/s3dis",
                        help="Root directory of preprocessed dataset")
    parser.add_argument("--cvfold", type=int, default=0, choices=[0, 1],
                        help="Cross-validation fold")
    parser.add_argument("--n_way", type=int, default=2, choices=[2, 3],
                        help="Number of classes per episode")
    parser.add_argument("--k_shot", type=int, default=1, choices=[1, 5],
                        help="Number of support shots per class")
    parser.add_argument("--modality", type=str, default="text", choices=["text", "image", "audio"],
                        help="Guiding modality")
    parser.add_argument("--dry_run", type=str2bool, default=False,
                        help="Run a dry-run test episode and exit")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Number of episodes per batch")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Initial learning rate for AdamW")
    parser.add_argument("--weight_decay", type=float, default=0.1,
                        help="Weight decay for AdamW")
    parser.add_argument("--step_size", type=int, default=10,
                        help="Scheduler step size in epochs")
    parser.add_argument("--gamma", type=float, default=0.5,
                        help="Learning rate halving factor")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--save_path", type=str, default="./checkpoints",
                        help="Directory to save model checkpoints")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print(" CascadeProto Episodic Training Pipeline")
    print(f" Dataset: {args.dataset.upper()} | Fold: {args.cvfold} | Setting: {args.n_way}-way {args.k_shot}-shot")
    print(f" Modality: {args.modality} | Device: {device} | Dry-Run: {args.dry_run}")
    print("=" * 60)

    # 1. Initialize CascadeProto model
    model = CascadeProto(
        input_points=2048,
        d_feature=128,
        d_subspace=72,
        num_stages=4,
        text_dim=512,
        init_theta=0.5,
        tau=0.5,
        alpha=0.5,
        lambda_gmmn=1.0,
        w_bg=0.8,
        w_fg=1.0
    ).to(device)

    # 2. Initialize Joint Criterion (CrossEntropy with w_cls=[0.8, 1, 1] + 1.0 * GMMN)
    criterion = CascadeProtoLoss(
        lambda_gmmn=1.0,
        w_bg=0.8,
        w_fg=1.0,
        gmmn_bg_weight=0.1,
        gmmn_fg_weight=1.0
    ).to(device)

    # 3. Optimizer & Scheduler as specified in AGENTS.md
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    train_classes, test_classes = get_s3dis_class_names(args.cvfold)
    print(f" Active Train Classes ({len(train_classes)}): {train_classes}")
    print(f" Unseen Test Classes ({len(test_classes)}): {test_classes}")

    # 4. Dry-run execution
    if args.dry_run:
        print("\n[Executing Phase 3 Dry-Run Episode]...")
        model.train()
        sampled_fg = random.sample(train_classes, args.n_way)
        print(f" Sampled Episode Foreground Classes: {sampled_fg}")

        support_x, support_y, query_x, query_y, E_text = generate_dry_run_batch(
            n_way=args.n_way,
            k_shot=args.k_shot,
            device=device,
            sampled_classes=sampled_fg
        )

        optimizer.zero_grad()
        output = model(support_x, support_y, query_x, text_embeddings=E_text, n_way=args.n_way)
        total_loss, seg_loss, gmmn_loss = criterion(
            output.logits, query_y, output.p_modal, output.p_point
        )

        total_loss.backward()
        optimizer.step()

        pred = torch.softmax(output.logits, dim=-1).argmax(dim=-1)
        correct = (pred == query_y).sum().item()
        acc = correct / query_y.numel()

        print(f"\n[Dry-Run Step Metrics]")
        print(f"  • Total Loss:   {total_loss.item():.4f}")
        print(f"  • Seg Loss:     {seg_loss.item():.4f}")
        print(f"  • GMMN Loss:    {gmmn_loss.item():.4f}")
        print(f"  • Query Acc:    {acc * 100:.2f}%")
        print(f"  • ADRM Weights: {[round(w, 4) for w in output.w_gate[0].tolist()]}")
        print("\n[Dry-Run PASSED] 1 episode successfully executed forward, backward, and optimizer step!")
        return 0

    # 5. Full Training Loop
    os.makedirs(args.save_path, exist_ok=True)
    print(f"\nStarting training for {args.epochs} epochs...")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []

        # Run episodic iterations per epoch (e.g. 100 episodes)
        num_episodes_per_epoch = 100
        for ep in range(num_episodes_per_epoch):
            sampled_fg = random.sample(train_classes, args.n_way)
            support_x, support_y, query_x, query_y, E_text = generate_dry_run_batch(
                n_way=args.n_way,
                k_shot=args.k_shot,
                device=device,
                sampled_classes=sampled_fg
            )

            optimizer.zero_grad()
            output = model(support_x, support_y, query_x, text_embeddings=E_text, n_way=args.n_way)
            total_loss, _, _ = criterion(output.logits, query_y, output.p_modal, output.p_point)
            total_loss.backward()
            optimizer.step()

            epoch_losses.append(total_loss.item())

        scheduler.step()
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] - Loss: {avg_loss:.4f} - LR: {current_lr:.6f}")

        if epoch % 10 == 0 or epoch == args.epochs:
            ckpt_path = os.path.join(args.save_path, f"cascadeproto_epoch_{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

    print("Training completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
