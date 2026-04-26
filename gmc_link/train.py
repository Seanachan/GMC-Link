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

from gmc_link.losses import AlignmentLoss, HardNegativeInfoNCE
from gmc_link.alignment import MotionLanguageAligner
from gmc_link.dataset import (
    MotionLanguageDataset, SequenceMotionLanguageDataset,
    collate_fn, sequence_collate_fn,
    build_training_data, compute_extra_dims,
)
from gmc_link.text_utils import TextEncoder


def train_one_epoch(
    model: MotionLanguageAligner,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    loss_func: nn.Module,
    device: torch.device,
    grad_clip: float = 0.0,
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

    for batch in dataloader:
        if len(batch) == 4:
            motion_features, padding_masks, language_features, expr_ids = batch
            motion_features = motion_features.to(device)
            padding_masks = padding_masks.to(device)
            language_features = language_features.to(device)
            expr_ids = expr_ids.to(device)
            sim_matrix = model(motion_features, language_features, padding_mask=padding_masks)
        else:
            motion_features, language_features, expr_ids = batch
            motion_features = motion_features.to(device)
            language_features = language_features.to(device)
            expr_ids = expr_ids.to(device)
            sim_matrix = model(motion_features, language_features)

        # ── InfoNCE loss with false-negative masking ──
        loss = loss_func(sim_matrix, expr_ids)

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
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
    data_root,
    sequences: list,
    batch_size: int,
    use_group_labels: bool = False,
    extra_features: list = None,
    seq_len: int = 0,
    text_encoder_name: str = "all-MiniLM-L6-v2",
    ego_router_name: str = "orb",
) -> Optional[DataLoader]:
    """
    Initialize text encoder, build training dataset, and return a DataLoader.

    data_root may be either:
      - str: single dataset root; `sequences` is the seq list for that root.
      - list of (root, seqs) pairs: multiple sources are loaded and
        concatenated; `sequences` is ignored in that case. Label IDs are
        offset per source so they remain distinct.
    """
    print(f"Loading text encoder: {text_encoder_name}")
    encoder = TextEncoder(model_name=text_encoder_name, device=str(device))

    sources = data_root if isinstance(data_root, (list, tuple)) else [(data_root, sequences)]

    print("Building training data...")
    if seq_len > 0:
        all_motion, all_masks, all_lang, all_labels = [], [], [], []
    else:
        all_motion, all_lang, all_labels = [], [], []
    label_offset = 0

    for root, seqs in sources:
        if len(sources) > 1:
            print(f"  === Source: {root} ({len(seqs)} seqs) ===")
        result = build_training_data(
            data_root=root,
            sequences=seqs,
            text_encoder=encoder,
            use_group_labels=use_group_labels,
            extra_features=extra_features,
            seq_len=seq_len,
            ego_router_name=ego_router_name,
        )
        if seq_len > 0:
            seq_motion, seq_masks, seq_language, seq_labels = result
            if len(seq_motion) == 0:
                continue
            offset_labels = [lbl + label_offset for lbl in seq_labels]
            all_motion.extend(seq_motion)
            all_masks.extend(seq_masks)
            all_lang.extend(seq_language)
            all_labels.extend(offset_labels)
            label_offset = max(offset_labels) + 1 if offset_labels else label_offset
        else:
            motion_data, language_data, label_ids = result
            if len(motion_data) == 0:
                continue
            offset_labels = [lbl + label_offset for lbl in label_ids]
            all_motion.extend(motion_data)
            all_lang.extend(language_data)
            all_labels.extend(offset_labels)
            label_offset = max(offset_labels) + 1 if offset_labels else label_offset

    if seq_len > 0:
        if len(all_motion) == 0:
            return None
        print(f"Total training sequences: {len(all_motion)}")
        dataset = SequenceMotionLanguageDataset(all_motion, all_masks, all_lang, all_labels)
        chosen_collate = sequence_collate_fn
    else:
        if len(all_motion) == 0:
            return None
        print(f"Total training samples: {len(all_motion)}")
        dataset = MotionLanguageDataset(all_motion, all_lang, all_labels)
        chosen_collate = collate_fn

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=chosen_collate,
        pin_memory=True,
        drop_last=True,
        num_workers=4,
        persistent_workers=True,
    )

    return dataloader


def setup_model_and_optimizer(
    device: torch.device, lang_dim: int, learning_rate: float, epochs: int,
    learnable_temp: bool = False, motion_dim: int = 13,
    architecture: str = "mlp", seq_len: int = 10,
    loss_name: str = "infonce", beta: float = 1.0,
) -> Tuple[
    MotionLanguageAligner, nn.Module, optim.Optimizer, optim.lr_scheduler.LRScheduler
]:
    """Initialize model, loss, and AdamW optimizer."""
    model = MotionLanguageAligner(
        motion_dim=motion_dim, lang_dim=lang_dim, embed_dim=256,
        architecture=architecture, seq_len=seq_len,
    ).to(device)

    if loss_name == "hninfo":
        if learnable_temp:
            raise ValueError("--learnable-temp is not supported with --loss hninfo")
        criterion = HardNegativeInfoNCE(temperature=0.07, beta=beta, fnm=True)
    else:
        criterion = AlignmentLoss(temperature=0.07, learnable=learnable_temp)

    params = list(model.parameters())
    if learnable_temp:
        params += list(criterion.parameters())

    optimizer = optim.AdamW(params, lr=learning_rate)
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
    warmup_epochs: int = 0,
    grad_clip: float = 0.0,
) -> None:
    """
    Execute the main training loop across all epochs and save the final weights.
    """
    print(f"Starting training on {device} | {len(dataloader)} batches/epoch...")
    if warmup_epochs > 0:
        print(f"  LR warmup: {warmup_epochs} epochs")
    if grad_clip > 0:
        print(f"  Gradient clipping: max_norm={grad_clip}")

    base_lr = optimizer.param_groups[0]["lr"]
    loss_history = []
    accuracy_history = []
    lr_history = []

    for epoch in tqdm(range(epochs)):
        # Linear warmup: scale LR from 0 to base_lr over warmup_epochs
        if warmup_epochs > 0 and epoch < warmup_epochs:
            warmup_lr = base_lr * (epoch + 1) / warmup_epochs
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr

        avg_loss, accuracy = train_one_epoch(
            model, dataloader, optimizer, criterion, device, grad_clip=grad_clip
        )
        # Only step cosine scheduler after warmup completes
        if epoch >= warmup_epochs:
            scheduler.step()

        loss_history.append(avg_loss)
        accuracy_history.append(accuracy)
        lr_history.append(optimizer.param_groups[0]["lr"])

        if (epoch + 1) % 20 == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            tau_str = f" | τ={criterion.temperature:.4f}" if hasattr(criterion, 'log_inv_temp') and criterion.log_inv_temp is not None else ""
            print(
                f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f} | "
                f"Acc: {accuracy:.2%} | LR: {current_lr:.6f}{tau_str}"
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


def _run_single_stage(
    device: torch.device,
    data_root: str,
    sequences: list,
    batch_size: int,
    lang_dim: int,
    lr: float,
    epochs: int,
    save_path: str,
    use_group_labels: bool = False,
    resume_path: str = None,
    warmup_epochs: int = 0,
    learnable_temp: bool = False,
    grad_clip: float = 0.0,
    extra_features: list = None,
    architecture: str = "mlp",
    seq_len: int = 10,
    loss_name: str = "infonce",
    beta: float = 1.0,
    text_encoder_name: str = "all-MiniLM-L6-v2",
    ego_router_name: str = "orb",
) -> None:
    """Run a single training stage."""
    if loss_name == "hninfo" and use_group_labels:
        raise ValueError(
            "--loss hninfo requires sentence-level labels; "
            "not compatible with --stage 1 (group labels)"
        )

    dataloader = setup_data(device, data_root, sequences, batch_size,
                            use_group_labels=use_group_labels,
                            extra_features=extra_features,
                            seq_len=seq_len if architecture == "temporal_transformer" else 0,
                            text_encoder_name=text_encoder_name,
                            ego_router_name=ego_router_name)
    if dataloader is None:
        print("ERROR: No training data found.")
        return

    motion_dim = 13 + compute_extra_dims(extra_features)
    model, criterion, optimizer, scheduler = setup_model_and_optimizer(
        device, lang_dim, lr, epochs, learnable_temp=learnable_temp,
        motion_dim=motion_dim,
        architecture=architecture, seq_len=seq_len,
        loss_name=loss_name, beta=beta,
    )

    if resume_path is not None:
        print(f"  Loading weights from {resume_path}...")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
        else:
            model.load_state_dict(checkpoint)

    train_loop(model, dataloader, optimizer, scheduler, criterion, device, epochs,
               save_path=save_path, warmup_epochs=warmup_epochs, grad_clip=grad_clip)

    checkpoint = torch.load(save_path, map_location=device, weights_only=False)
    checkpoint["motion_dim"] = motion_dim
    checkpoint["extra_features"] = extra_features
    checkpoint["architecture"] = architecture
    checkpoint["seq_len"] = seq_len if architecture == "temporal_transformer" else None
    checkpoint["loss_name"] = loss_name
    checkpoint["beta"] = beta if loss_name == "hninfo" else None
    checkpoint["text_encoder"] = text_encoder_name
    checkpoint["lang_dim"] = lang_dim
    torch.save(checkpoint, save_path)


def main() -> None:
    """
    Main training execution block. All paths configurable via CLI args.

    Usage:
        python -m gmc_link.train --split v1 --data-root refer-kitti
        python -m gmc_link.train --split v1 --stage curriculum
    """
    import argparse

    parser = argparse.ArgumentParser(description="Train GMC-Link MotionLanguageAligner")
    parser.add_argument("--split", default=os.environ.get("TRAIN_SPLIT", "v2").lower(),
                        choices=["v1", "v2", "v1v2"], help="Dataset split (default: v2, or TRAIN_SPLIT env; v1v2 joins V1 and V2 with held-out seqs filtered from V2)")
    parser.add_argument("--data-root", default=None,
                        help="Path to Refer-KITTI dataset (default: refer-kitti for v1, refer-kitti-v2 for v2)")
    parser.add_argument("--save-path", default=None,
                        help="Output weights path (default: gmc_link_weights_v1train.pth / gmc_link_weights.pth)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size (default: 256 for v1, 512 for v2)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--warmup-epochs", type=int, default=0, help="Linear LR warmup epochs")
    parser.add_argument("--learnable-temp", action="store_true", help="Make temperature a learnable parameter")
    parser.add_argument("--grad-clip", type=float, default=0.0, help="Gradient clipping max_norm (0=disabled)")
    parser.add_argument("--loss", default="infonce", choices=["infonce", "hninfo"],
                        help="Contrastive loss (default: infonce; hninfo = hard-negative InfoNCE with FNM)")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="Hard-negative concentration (only used when --loss hninfo; 0=uniform, 1.0 typical)")
    parser.add_argument("--stage", default=None, choices=["1", "2", "curriculum"],
                        help="Curriculum stage: 1=group-level, 2=fine-tune, curriculum=both")
    parser.add_argument("--resume", default=None, help="Path to pretrained weights (for stage 2)")
    parser.add_argument("--extra-features", default=None,
                        help="Comma-separated extra features (e.g., speed_m,ego_motion)")
    parser.add_argument("--architecture", default="mlp", choices=["mlp", "temporal_transformer"],
                        help="Motion encoder architecture (default: mlp)")
    parser.add_argument("--seq-len", type=int, default=10,
                        help="Sequence length T for temporal_transformer (default: 10)")
    parser.add_argument("--text-encoder", default="all-MiniLM-L6-v2",
                        help="Sentence-transformers model for language embeddings "
                             "(default: all-MiniLM-L6-v2; e.g., BAAI/bge-base-en-v1.5)")
    parser.add_argument("--ego", default="orb", choices=["orb", "recoverpose"],
                        help="Ego-motion source for dataset cache building (Exp 37 Stage A)")
    args = parser.parse_args()

    if args.stage == "curriculum" and args.loss == "hninfo":
        parser.error(
            "--loss hninfo is not compatible with --stage curriculum "
            "(stage 1 uses group labels; HN-InfoNCE requires sentence-level labels). "
            "Run stage 2 standalone with --resume if you want HN-InfoNCE finetuning."
        )

    # Parse extra features
    extra_features = None
    if args.extra_features:
        extra_features = [f.strip() for f in args.extra_features.split(",")]
        from gmc_link.dataset import EXTRA_FEATURE_DIMS
        for f in extra_features:
            if f not in EXTRA_FEATURE_DIMS:
                print(f"ERROR: Unknown feature '{f}'. Valid: {list(EXTRA_FEATURE_DIMS.keys())}")
                return
        extra_dims = compute_extra_dims(extra_features)
        print(f"Extra features: {extra_features} (+{extra_dims}D → {13 + extra_dims}D)")

    # --- Configuration ---
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Device: {device}")

    # Infer lang_dim from encoder. Covers MiniLM (384), BGE-base/E5-base (768), etc.
    from sentence_transformers import SentenceTransformer
    lang_dim = SentenceTransformer(args.text_encoder).get_sentence_embedding_dimension()
    if args.text_encoder != "all-MiniLM-L6-v2":
        print(f"Text encoder: {args.text_encoder} (lang_dim={lang_dim})")

    if args.split == "v1":
        data_root = args.data_root or "refer-kitti"
        sequences = V1_TRAIN_SEQS
        batch_size = args.batch_size or 256
        save_path = args.save_path or "gmc_link_weights_v1train.pth"
    elif args.split == "v1v2":
        # Joint V1+V2 training. Exclude V1 held-out eval seqs from V2.
        heldout = {"0005", "0011", "0013"}
        v2_clean = [s for s in V2_TRAIN_SEQS if s not in heldout]
        data_root = [
            ("refer-kitti", V1_TRAIN_SEQS),
            ("refer-kitti-v2", v2_clean),
        ]
        sequences = None  # unused when data_root is a list
        batch_size = args.batch_size or 256
        save_path = args.save_path or "gmc_link_weights_v1v2train.pth"
    else:
        data_root = args.data_root or "refer-kitti-v2"
        sequences = V2_TRAIN_SEQS
        batch_size = args.batch_size or 512
        save_path = args.save_path or "gmc_link_weights.pth"

    # --- Curriculum training ---
    if args.stage == "curriculum":
        stage1_path = save_path.replace(".pth", "_stage1.pth")
        curriculum_path = save_path.replace(".pth", "_curriculum.pth")

        print(f"═══ Stage 1: Group-level training (100 epochs) → {stage1_path}")
        print(f"  data_root={data_root}  batch_size={batch_size}  epochs=100  lr={args.lr}")
        _run_single_stage(
            device, data_root, sequences, batch_size, lang_dim,
            lr=args.lr, epochs=100, save_path=stage1_path,
            use_group_labels=True,
            warmup_epochs=args.warmup_epochs, grad_clip=args.grad_clip,
            extra_features=extra_features,
            architecture=args.architecture, seq_len=args.seq_len,
            text_encoder_name=args.text_encoder,
            ego_router_name=args.ego,
        )

        stage2_lr = args.lr * 0.1
        print(f"\n═══ Stage 2: Fine expression training (50 epochs) → {curriculum_path}")
        print(f"  data_root={data_root}  batch_size={batch_size}  epochs=50  lr={stage2_lr}")
        _run_single_stage(
            device, data_root, sequences, batch_size, lang_dim,
            lr=stage2_lr, epochs=50, save_path=curriculum_path,
            use_group_labels=False, resume_path=stage1_path,
            warmup_epochs=0, grad_clip=args.grad_clip,
            extra_features=extra_features,
            architecture=args.architecture, seq_len=args.seq_len,
            text_encoder_name=args.text_encoder,
            ego_router_name=args.ego,
        )
        return

    # --- Single stage ---
    if args.stage == "1":
        save_path = args.save_path or save_path.replace(".pth", "_stage1.pth")
        print(f"Stage 1: Group-level training → {save_path}")
        use_group = True
    elif args.stage == "2":
        save_path = args.save_path or save_path.replace(".pth", "_curriculum.pth")
        print(f"Stage 2: Fine expression training → {save_path}")
        use_group = False
    else:
        print(f"Training on {'V1' if args.split == 'v1' else 'V2'} → {save_path}")
        use_group = False

    print(f"  data_root={data_root}  batch_size={batch_size}  epochs={args.epochs}  lr={args.lr}")

    _run_single_stage(
        device, data_root, sequences, batch_size, lang_dim,
        lr=args.lr, epochs=args.epochs, save_path=save_path,
        use_group_labels=use_group, resume_path=args.resume,
        warmup_epochs=args.warmup_epochs, learnable_temp=args.learnable_temp,
        grad_clip=args.grad_clip, extra_features=extra_features,
        architecture=args.architecture, seq_len=args.seq_len,
        loss_name=args.loss, beta=args.beta,
        text_encoder_name=args.text_encoder,
        ego_router_name=args.ego,
    )


if __name__ == "__main__":
    main()
