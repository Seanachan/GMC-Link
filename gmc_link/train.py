"""
GMC-Link Training Script
========================
Trains the MotionLanguageAligner using Supervised InfoNCE loss with
motion-language pairs from refer-kitti sequences (test on 0011).
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from typing import Tuple, Optional
from tqdm import tqdm

import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gmc_link.losses import AlignmentLoss
from gmc_link.alignment import MotionLanguageAligner
from gmc_link.dataset import MotionLanguageDataset, collate_fn, build_training_data
from gmc_link.text_utils import TextEncoder


def train_one_epoch(
    model: MotionLanguageAligner,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    loss_func: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Train the model for a single epoch using CLIP-style cross-modal InfoNCE.

    For each batch of N (motion, language, expr_id) triples:
      - Compute NxN cosine similarity matrix via model.forward()
      - Pass sim_matrix and expr_ids to InfoNCE loss with false-negative masking

    Returns:
        Tuple of (average_loss, accuracy).
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for motion_features, language_features, expr_ids in dataloader:

        motion_features = motion_features.to(device)
        language_features = language_features.to(device)
        expr_ids = expr_ids.to(device)

        # ── NxN cosine similarity matrix ──
        sim_matrix = model(motion_features, language_features)

        # ── InfoNCE loss with false-negative masking ──
        loss = loss_func(sim_matrix, expr_ids)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # ── Track retrieval accuracy (motion→language) ──
        batch_size = sim_matrix.size(0)
        with torch.no_grad():
            nearest_lang_idx = sim_matrix.argmax(dim=1)
            predicted_ids = expr_ids[nearest_lang_idx]
            correct += (predicted_ids == expr_ids).sum().item()
            total += batch_size

    accuracy = correct / total if total > 0 else 0.0
    return total_loss / len(dataloader), accuracy


def setup_data(
    device: torch.device,
    data_root: str,
    sequences: list,
    batch_size: int,
) -> Optional[DataLoader]:
    """
    Initialize text encoder, build training dataset, and return a DataLoader.
    """
    print("Loading text encoder...")
    encoder = TextEncoder(device=str(device))

    print("Building training data...")
    all_motions, all_languages, all_labels = build_training_data(
        data_root=data_root,
        sequences=sequences,
        text_encoder=encoder,
    )

    print(f"Total training samples: {len(all_motions)}")
    if len(all_motions) == 0:
        return None

    dataset = MotionLanguageDataset(all_motions, all_languages, all_labels)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,  # Consistent batch size for contrastive learning
        num_workers=4,
        persistent_workers=True,
    )

    return dataloader


def setup_model_and_optimizer(
    device: torch.device, lang_dim: int, learning_rate: float, epochs: int
) -> Tuple[
    MotionLanguageAligner, nn.Module, optim.Optimizer, optim.lr_scheduler.LRScheduler
]:
    """
    Initialize the MotionLanguageAligner, InfoNCE loss, and AdamW optimizer.
    """
    model = MotionLanguageAligner(motion_dim=13, lang_dim=lang_dim, embed_dim=256).to(
        device
    )

    criterion = AlignmentLoss(temperature=0.07)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )

    return model, criterion, optimizer, scheduler


def save_training_curves(
    loss_history: list,
    accuracy_history: list,
    lr_history: list,
    save_path: str,
) -> None:
    """Save loss, accuracy, and LR curves to a PNG alongside the weights."""
    plot_path = save_path.replace(".pth", "_curves.png")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle(f"Training Curves — {os.path.basename(save_path)}", fontsize=12, fontweight="bold")
    epochs = range(1, len(loss_history) + 1)

    axes[0].plot(epochs, loss_history, color="#e74c3c", linewidth=1.5)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("InfoNCE Loss")
    axes[0].set_title("Loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, [a * 100 for a in accuracy_history], color="#27ae60", linewidth=1.5)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Retrieval Accuracy (M→L)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, lr_history, color="#3498db", linewidth=1.5)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Learning Rate")
    axes[2].set_title("LR Schedule")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Training curves saved to {plot_path}")


def train_loop(
    model: MotionLanguageAligner,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    criterion: nn.Module,
    device: torch.device,
    epochs: int,
    save_path: str = "gmc_link_weights.pth",
) -> None:
    """
    Execute the main training loop across all epochs and save the final weights.
    """
    print(f"Starting training on {device} | {len(dataloader)} batches/epoch...")
    loss_history = []
    accuracy_history = []
    lr_history = []

    for epoch in tqdm(range(epochs)):
        avg_loss, accuracy = train_one_epoch(
            model, dataloader, optimizer, criterion, device
        )
        scheduler.step()

        loss_history.append(avg_loss)
        accuracy_history.append(accuracy)
        lr_history.append(scheduler.get_last_lr()[0])

        if (epoch + 1) % 20 == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(
                f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f} | "
                f"Acc: {accuracy:.2%} | LR: {current_lr:.6f}"
            )

    # Save model weights + temperature
    save_dict = {
        "model": model.state_dict(),
        "temperature": criterion.temperature,
    }
    torch.save(save_dict, save_path)
    print(f"Training complete. Weights saved to {save_path} (τ={criterion.temperature:.4f})")

    # Save training curves
    save_training_curves(loss_history, accuracy_history, lr_history, save_path)


# ── Split definitions (sequences only, no paths) ─────────────────────
V1_TRAIN_SEQS = [
    "0001", "0002", "0003", "0004", "0006",
    "0007", "0008", "0009", "0010", "0012",
    "0014", "0015", "0016", "0018", "0020",
]
V2_TRAIN_SEQS = [
    "0000", "0001", "0002", "0003", "0004", "0005",
    "0006", "0007", "0008", "0009", "0010", "0011",
    "0012", "0013", "0014", "0015",
]


def main() -> None:
    """
    Main training execution block. All paths configurable via CLI args.

    Usage:
        python -m gmc_link.train --split v1 --data-root refer-kitti
        python -m gmc_link.train --split v2 --data-root refer-kitti-v2
    """
    import argparse

    parser = argparse.ArgumentParser(description="Train GMC-Link MotionLanguageAligner")
    parser.add_argument("--split", default=os.environ.get("TRAIN_SPLIT", "v2").lower(),
                        choices=["v1", "v2"], help="Dataset split (default: v2, or TRAIN_SPLIT env)")
    parser.add_argument("--data-root", default=None,
                        help="Path to Refer-KITTI dataset (default: refer-kitti for v1, refer-kitti-v2 for v2)")
    parser.add_argument("--save-path", default=None,
                        help="Output weights path (default: gmc_link_weights_v1train.pth / gmc_link_weights.pth)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size (default: 256 for v1, 512 for v2)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    # --- Configuration ---
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Device: {device}")

    lang_dim = 384

    if args.split == "v1":
        data_root = args.data_root or "refer-kitti"
        sequences = V1_TRAIN_SEQS
        batch_size = args.batch_size or 256
        save_path = args.save_path or "gmc_link_weights_v1train.pth"
        print(f"Training on Refer-KITTI V1 train split → {save_path}")
    else:
        data_root = args.data_root or "refer-kitti-v2"
        sequences = V2_TRAIN_SEQS
        batch_size = args.batch_size or 512
        save_path = args.save_path or "gmc_link_weights.pth"
        print(f"Training on Refer-KITTI V2, seqs 0000-0015 → {save_path}")

    print(f"  data_root={data_root}  batch_size={batch_size}  epochs={args.epochs}  lr={args.lr}")

    # --- Pipeline ---
    dataloader = setup_data(device, data_root, sequences, batch_size)
    if dataloader is None:
        print("ERROR: No training data found.")
        return

    model, criterion, optimizer, scheduler = setup_model_and_optimizer(
        device, lang_dim, args.lr, args.epochs
    )

    train_loop(model, dataloader, optimizer, scheduler, criterion, device, args.epochs,
               save_path=save_path)


if __name__ == "__main__":
    main()
