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
    for epoch in tqdm(range(epochs)):
        avg_loss, accuracy = train_one_epoch(
            model, dataloader, optimizer, criterion, device
        )
        scheduler.step()

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


def main() -> None:
    """
    Main training execution block.

    Set TRAIN_SPLIT env var to choose dataset:
      TRAIN_SPLIT=v2  (default) — Refer-KITTI V2 seqs 0000-0015 → gmc_link_weights.pth
      TRAIN_SPLIT=v1            — Refer-KITTI V1 seq 0011 only   → gmc_link_weights_v1train.pth
    """
    import os as _os
    train_split = _os.environ.get("TRAIN_SPLIT", "v2").lower()

    # --- Configuration ---
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Device: {device}")

    learning_rate = 1e-3
    epochs = 100
    lang_dim = 384

    if train_split == "v1":
        # Refer-KITTI V1: train on official train split, eval on val (0005, 0011, 0013)
        # Mirrors iKUN's VIDEOS['train'] — excludes 0005, 0011, 0013 (val/test seqs)
        data_root = "/home/seanachan/data/Dataset/refer-kitti"
        sequences = [
            "0001", "0002", "0003", "0004", "0006",
            "0007", "0008", "0009", "0010", "0012",
            "0014", "0015", "0016", "0018", "0020",
        ]
        batch_size = 256
        save_path = "gmc_link_weights_v1train.pth"
        print("Training on Refer-KITTI V1 train split → " + save_path)
    else:
        # Refer-KITTI V2 data path and official train/test split
        # Train: seqs 0000-0015 | Test: seqs 0016-0020
        data_root = "/home/seanachan/data/Dataset/refer-kitti-v2"
        sequences = [
            "0000", "0001", "0002", "0003", "0004", "0005",
            "0006", "0007", "0008", "0009", "0010", "0011",
            "0012", "0013", "0014", "0015",
        ]
        batch_size = 512  # Balanced: enough in-batch negatives without diluting unique classes
        save_path = "gmc_link_weights.pth"
        print("Training on Refer-KITTI V2, seqs 0000-0015 → " + save_path)

    # --- Pipeline ---
    dataloader = setup_data(device, data_root, sequences, batch_size)
    if dataloader is None:
        print("ERROR: No training data found.")
        return

    model, criterion, optimizer, scheduler = setup_model_and_optimizer(
        device, lang_dim, learning_rate, epochs
    )

    train_loop(model, dataloader, optimizer, scheduler, criterion, device, epochs,
               save_path=save_path)


if __name__ == "__main__":
    main()
