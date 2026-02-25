"""
GMC-Link Training Script
========================
Trains the MotionLanguageAligner using BCE loss with motion-language pairs
from refer-kitti sequences 0015/0016/0018 (test on 0011).
"""

import sys
import os
# pylint: disable=too-many-arguments,too-many-locals,wrong-import-position,too-many-positional-arguments

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
    Train the model for a single epoch.

    Returns:
        Tuple of (average_loss, accuracy).
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for _, (motion_features, language_features, labels) in enumerate(
        dataloader
    ):
        motion_features = motion_features.to(device)
        language_features = language_features.to(device)
        labels = labels.to(device)

        # Per-pair similarity scores
        scores = model.score_pairs(motion_features, language_features)
        loss = loss_func(scores, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Track accuracy
        preds = (torch.sigmoid(scores) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    accuracy = correct / total if total > 0 else 0.0
    return total_loss / len(dataloader), accuracy


def setup_data(
    device: torch.device,
    data_root: str,
    sequences: list,
    batch_size: int,
    frame_gap: int = 5,
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
        frame_gap=frame_gap,
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
    )

    return dataloader


def setup_model_and_optimizer(
    device: torch.device, lang_dim: int, learning_rate: float, epochs: int
) -> Tuple[
    MotionLanguageAligner, nn.Module, optim.Optimizer, optim.lr_scheduler.LRScheduler
]:
    """
    Initialize the MotionLanguageAligner, BCE loss function, and AdamW optimizer.
    """
    model = MotionLanguageAligner(motion_dim=8, lang_dim=lang_dim, embed_dim=256).to(
        device
    )

    criterion = AlignmentLoss()
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

    torch.save(model.state_dict(), save_path)
    print(f"Training complete. Weights saved to {save_path}")


def main() -> None:
    """
    Main training execution block.
    """
    # --- Configuration ---
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Device: {device}")

    learning_rate = 1e-3
    batch_size = 128
    epochs = 50
    lang_dim = 384

    # Refer-KITTI data paths (Train on all available seqs, test on 11)
    data_root = "refer-kitti"
    sequences = [
        "0001",
        "0002",
        "0003",
        "0004",
        "0005",
        "0006",
        "0007",
        "0008",
        "0009",
        "0010",
        "0012",
        "0013",
        "0014",
        "0015",
        "0016",
        "0018",
        "0020",
    ]

    # --- Pipeline ---
    dataloader = setup_data(device, data_root, sequences, batch_size)
    if dataloader is None:
        print("ERROR: No training data found.")
        return

    model, criterion, optimizer, scheduler = setup_model_and_optimizer(
        device, lang_dim, learning_rate, epochs
    )

    train_loop(model, dataloader, optimizer, scheduler, criterion, device, epochs)


if __name__ == "__main__":
    main()
