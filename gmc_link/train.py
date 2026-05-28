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

from gmc_link.losses import AlignmentLoss, HardNegativeInfoNCE, StructuralConsensusLoss
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
    target_class_id: Optional[int] = None,
    struct_loss: Optional[nn.Module] = None,
    lam_struct: float = 0.0,
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

    # Side-channel: id_to_class tensor attached on dataloader by setup_data().
    # If target_class_id is set, build per-batch anchor_mask: True where the
    # sample's expr_id maps to target class. Negatives stay full-batch.
    id_to_class = getattr(dataloader, "id_to_class", None)
    if target_class_id is not None and id_to_class is None:
        raise RuntimeError(
            "target_class_id set but dataloader has no id_to_class lookup"
        )
    if id_to_class is not None:
        id_to_class = id_to_class.to(device)

    for batch in dataloader:
        # Batch shapes:
        #   3-tuple: (motion, lang, label)                  — MLP, no clip
        #   4-tuple: (motion, mask, lang, label)            — transformer, no clip
        #   4-tuple: (motion, lang, label, clip)            — MLP + clip
        #   5-tuple: (motion, mask, lang, label, clip)      — transformer + clip
        # Disambiguate 4-tuples by element[1] dtype: bool=mask, float=lang.
        clip_features = None
        padding_masks = None
        if len(batch) == 5:
            motion_features, padding_masks, language_features, expr_ids, clip_features = batch
        elif len(batch) == 4:
            if batch[1].dtype == torch.bool:
                motion_features, padding_masks, language_features, expr_ids = batch
            else:
                motion_features, language_features, expr_ids, clip_features = batch
        else:
            motion_features, language_features, expr_ids = batch

        motion_features = motion_features.to(device)
        language_features = language_features.to(device)
        expr_ids = expr_ids.to(device)
        if padding_masks is not None:
            padding_masks = padding_masks.to(device)
        if clip_features is not None:
            clip_features = clip_features.to(device)
        motion_emb, lang_emb = model.encode(
            motion_features, language_features,
            padding_mask=padding_masks, clip_feats=clip_features,
        )
        sim_matrix = torch.matmul(motion_emb, lang_emb.t())

        # ── InfoNCE loss with false-negative masking ──
        anchor_mask = None
        if target_class_id is not None:
            batch_class_ids = id_to_class[expr_ids]
            anchor_mask = (batch_class_ids == target_class_id)
            if anchor_mask.sum().item() == 0:
                # No target-class anchors in this batch — skip backward to
                # avoid div-by-zero / NaN gradient. Loss for logging = 0.
                continue
        loss = loss_func(sim_matrix, expr_ids, anchor_mask=anchor_mask)
        if struct_loss is not None and lam_struct > 0.0:
            loss = loss + lam_struct * struct_loss(motion_emb, lang_emb)

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
    motion_filter: str = "loose",
    class_filter: str = "all",
    clip_cache_path: str = None,
    use_depth: bool = False,
    depth_cache_dir: str = None,
    world_xy: bool = False,
    depth_source: str = "monocular",
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
        if clip_cache_path is not None:
            raise NotImplementedError(
                "--use-clip-feat with temporal_transformer (seq_len>0) not implemented"
            )
        all_motion, all_masks, all_lang, all_labels = [], [], [], []
    else:
        all_motion, all_lang, all_labels = [], [], []
    all_clip = [] if clip_cache_path is not None and seq_len == 0 else None
    all_id_to_class: list = []   # global id_to_class concatenated across sources (offset-aligned)
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
            motion_filter=motion_filter,
            class_filter=class_filter,
            clip_cache_path=clip_cache_path,
            use_depth=use_depth,
            depth_cache_dir=depth_cache_dir,
            world_xy=world_xy,
            depth_source=depth_source,
        )
        if seq_len > 0:
            seq_motion, seq_masks, seq_language, seq_labels, src_id_to_class = result
            if len(seq_motion) == 0:
                continue
            offset_labels = [lbl + label_offset for lbl in seq_labels]
            all_motion.extend(seq_motion)
            all_masks.extend(seq_masks)
            all_lang.extend(seq_language)
            all_labels.extend(offset_labels)
            all_id_to_class.extend(src_id_to_class)
            label_offset = max(offset_labels) + 1 if offset_labels else label_offset
        else:
            if all_clip is not None:
                motion_data, language_data, label_ids, clip_data, src_id_to_class = result
            else:
                motion_data, language_data, label_ids, src_id_to_class = result
                clip_data = None
            if len(motion_data) == 0:
                continue
            offset_labels = [lbl + label_offset for lbl in label_ids]
            all_motion.extend(motion_data)
            all_lang.extend(language_data)
            all_labels.extend(offset_labels)
            all_id_to_class.extend(src_id_to_class)
            if all_clip is not None:
                # clip_data is (N, dim) np array; extend row-wise
                all_clip.extend([clip_data[i] for i in range(clip_data.shape[0])])
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
        dataset = MotionLanguageDataset(all_motion, all_lang, all_labels, clip_data=all_clip)
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

    # Attach id_to_class as a tensor on the dataloader so train_one_epoch can
    # consume it without changing the DataLoader's tuple shape.
    dataloader.id_to_class = torch.tensor(all_id_to_class, dtype=torch.long)
    return dataloader


def setup_model_and_optimizer(
    device: torch.device, lang_dim: int, learning_rate: float, epochs: int,
    learnable_temp: bool = False, motion_dim: int = 13,
    architecture: str = "mlp", seq_len: int = 10,
    loss_name: str = "infonce", beta: float = 1.0,
    use_clip_feat: bool = False, clip_feat_dim: int = 512, clip_proj_dim: int = 64,
    fusion_site: str = "input_concat", lang_passthrough: bool = False,
    app_proj_dim: int = 256,
    identity_init_depth: bool = False,
) -> Tuple[
    MotionLanguageAligner, nn.Module, optim.Optimizer, optim.lr_scheduler.LRScheduler
]:
    """Initialize model, loss, and AdamW optimizer."""
    model = MotionLanguageAligner(
        motion_dim=motion_dim, lang_dim=lang_dim, embed_dim=256,
        architecture=architecture, seq_len=seq_len,
        use_clip_feat=use_clip_feat,
        clip_feat_dim=clip_feat_dim,
        clip_proj_dim=clip_proj_dim,
        fusion_site=fusion_site,
        lang_passthrough=lang_passthrough,
        app_proj_dim=app_proj_dim,
        identity_init_depth=identity_init_depth,
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
    target_class_id: Optional[int] = None,
    struct_loss: Optional[nn.Module] = None,
    lam_struct: float = 0.0,
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
            model, dataloader, optimizer, criterion, device,
            grad_clip=grad_clip, target_class_id=target_class_id,
            struct_loss=struct_loss, lam_struct=lam_struct,
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
    motion_filter: str = "loose",
    class_filter: str = "all",
    use_clip_feat: bool = False,
    clip_cache_path: str = None,
    clip_proj_dim: int = 64,
    fusion_site: str = "input_concat",
    lang_passthrough: bool = False,
    app_proj_dim: int = 256,
    use_depth: bool = False,
    depth_cache_dir: str = None,
    identity_init_depth: bool = False,
    world_xy: bool = False,
    depth_source: str = "monocular",
    lam_struct: float = 0.0,
    lam_angle: float = 0.5,
    struct_mode: str = "dist",
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
                            ego_router_name=ego_router_name,
                            motion_filter=motion_filter,
                            class_filter=class_filter,
                            clip_cache_path=clip_cache_path if use_clip_feat else None,
                            use_depth=use_depth,
                            depth_cache_dir=depth_cache_dir if use_depth else None,
                            world_xy=world_xy,
                            depth_source=depth_source)
    if dataloader is None:
        print("ERROR: No training data found.")
        return

    motion_dim = 13 + compute_extra_dims(extra_features) + (4 if use_depth else 0)
    model, criterion, optimizer, scheduler = setup_model_and_optimizer(
        device, lang_dim, lr, epochs, learnable_temp=learnable_temp,
        motion_dim=motion_dim,
        architecture=architecture, seq_len=seq_len,
        loss_name=loss_name, beta=beta,
        use_clip_feat=use_clip_feat,
        clip_feat_dim=512,
        clip_proj_dim=clip_proj_dim,
        fusion_site=fusion_site,
        lang_passthrough=lang_passthrough,
        app_proj_dim=app_proj_dim,
        identity_init_depth=identity_init_depth,
    )

    if resume_path is not None:
        print(f"  Loading weights from {resume_path}...")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
        else:
            model.load_state_dict(checkpoint)

    # Resolve target_class_id for anchor-mask mode (per-class specialist).
    # class_filter=='all' → no mask (target_class_id=None).
    target_class_id: Optional[int] = None
    if class_filter != "all":
        from gmc_link.expr_class import CLASS_LABELS
        target_class_id = CLASS_LABELS.index(class_filter)
        print(
            f"  Anchor-mask mode: class_filter={class_filter!r} "
            f"→ target_class_id={target_class_id}"
        )
        cls_counts = torch.bincount(
            dataloader.id_to_class, minlength=len(CLASS_LABELS)
        ).tolist()
        cls_summary = {c: cls_counts[i] for i, c in enumerate(CLASS_LABELS)}
        print(f"  id_to_class distribution: {cls_summary}")

    struct_loss = None
    if lam_struct > 0.0:
        struct_loss = StructuralConsensusLoss(
            lam_angle=lam_angle, mode=struct_mode,
        ).to(device)
        print(
            f"  Structural Consensus aux loss: lam_struct={lam_struct} "
            f"mode={struct_mode} lam_angle={lam_angle if struct_mode == 'dist_angle' else 'N/A'}"
        )

    train_loop(model, dataloader, optimizer, scheduler, criterion, device, epochs,
               save_path=save_path, warmup_epochs=warmup_epochs, grad_clip=grad_clip,
               target_class_id=target_class_id,
               struct_loss=struct_loss, lam_struct=lam_struct)

    checkpoint = torch.load(save_path, map_location=device, weights_only=False)
    checkpoint["motion_dim"] = motion_dim
    checkpoint["extra_features"] = extra_features
    checkpoint["architecture"] = architecture
    checkpoint["seq_len"] = seq_len if architecture == "temporal_transformer" else None
    checkpoint["loss_name"] = loss_name
    checkpoint["beta"] = beta if loss_name == "hninfo" else None
    checkpoint["text_encoder"] = text_encoder_name
    checkpoint["lang_dim"] = lang_dim
    checkpoint["use_clip_feat"] = use_clip_feat
    checkpoint["clip_feat_dim"] = 512 if use_clip_feat else None
    checkpoint["clip_proj_dim"] = clip_proj_dim if use_clip_feat else None
    checkpoint["fusion_site"] = fusion_site if use_clip_feat else None
    checkpoint["lang_passthrough"] = lang_passthrough
    checkpoint["app_proj_dim"] = app_proj_dim if (use_clip_feat and fusion_site == "late_concat") else None
    checkpoint["class_filter"] = class_filter
    checkpoint["target_class_id"] = target_class_id
    checkpoint["use_depth"] = use_depth
    checkpoint["depth_cache_dir"] = depth_cache_dir if use_depth else None
    checkpoint["identity_init_depth"] = identity_init_depth
    checkpoint["world_xy"] = world_xy
    checkpoint["depth_source"] = depth_source  # Path B: drives inference ego mode in manager
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
    parser.add_argument("--architecture", default="mlp",
                        choices=["mlp", "temporal_transformer", "shared_weight"],
                        help="Motion encoder architecture (default: mlp). "
                             "shared_weight: per-modality Linear adapter → shared 2-hidden MLP, "
                             "raw cos at inference, no input LN, no post-ops.")
    parser.add_argument("--seq-len", type=int, default=10,
                        help="Sequence length T for temporal_transformer (default: 10)")
    parser.add_argument("--text-encoder", default="all-MiniLM-L6-v2",
                        help="Sentence-transformers model for language embeddings "
                             "(default: all-MiniLM-L6-v2; e.g., BAAI/bge-base-en-v1.5)")
    parser.add_argument("--motion-filter", default="loose",
                        choices=["loose", "strict", "off"],
                        help="Train-expression filter: 'loose' (motion keyword present, default), "
                             "'strict' (motion keyword AND no appearance/spatial qualifier), "
                             "'off' (no filter — keep all expressions)")
    parser.add_argument("--class-filter", default="all",
                        choices=["all", "motion", "static", "appear"],
                        help="Per-class specialist filter (Tier B 1): apply AFTER motion-filter. "
                             "'all' (default; no class filter), 'motion' (active-motion exprs), "
                             "'static' (parked/stopped exprs), 'appear' (color/clothing/spatial). "
                             "For specialists set --motion-filter off + --class-filter <class>.")
    parser.add_argument("--ego", default="orb", choices=["orb", "recoverpose"],
                        help="Ego-motion source for dataset cache building (Exp 37 Stage A)")
    parser.add_argument("--use-clip-feat", action="store_true",
                        help="Concat per-bbox CLIP visual features onto motion vector")
    parser.add_argument("--clip-cache-path", default="gmc_link/clip_b32_datacomp_v1_gt_cache.npz",
                        help="Path to pre-computed CLIP feature .npz cache "
                             "(see gmc_link/clip_cache.py)")
    parser.add_argument("--fusion-site", default="input_concat",
                        choices=["input_concat", "late_concat"],
                        help="CLIP fusion site: input_concat (pre-MLP, Exp 39 path) "
                             "or late_concat (post-projector, Exp 41 path)")
    parser.add_argument("--lang-passthrough", action="store_true",
                        help="Skip lang_projector — use raw lang encoder output. "
                             "Requires lang_dim == out_dim (e.g. CLIP-text 512D + late_concat 256+256=512)")
    parser.add_argument("--app-proj-dim", type=int, default=256,
                        help="Output dim of app_projector arm in late_concat mode")
    parser.add_argument("--clip-dim", type=int, default=64,
                        help="Projected CLIP dim concatenated to motion (default: 64)")
    parser.add_argument("--use-depth", action="store_true",
                        help="Append [Z_n, dZ_2, dZ_5, dZ_10] to motion vector (13D→17D, "
                             "Approach B). Requires --depth-cache-dir.")
    parser.add_argument("--depth-cache-dir", default="gmc_link/depth_cache",
                        help="Per-track Z time-series cache dir, used when --use-depth set "
                             "(reads z_track_gt_<seq>.json per train sequence)")
    parser.add_argument("--identity-init-depth", action="store_true",
                        help="Zero-init the 4 depth weight columns of motion_projector[0] "
                             "so depth tail contributes zero at step 0 (bit-exact 13D init "
                             "until first gradient step).")
    parser.add_argument("--depth-source",
                        choices=["monocular", "lidar_oxts", "lidar_cohort"],
                        default="monocular",
                        help="Path B depth-axis signal. 'lidar_oxts' = LiDAR depth + GT oxts "
                             "ego-ΔZ (full Path B, setting A). 'lidar_cohort' = LiDAR depth + "
                             "stationary-cohort ego (ego ABLATION, setting B; isolates whether "
                             "the ego swap is the lever). 'monocular' (default) = DAv2 + cohort, "
                             "byte-identical to the prior depth path. lidar_* use "
                             "z_track_lidar_gt_<seq> caches; pass --depth-cache-dir gmc_link/depth_cache_lidar.")
    parser.add_argument("--world-xy", action="store_true",
                        help="Project image-plane (dx, dy) per scale to metric world "
                             "(dX, dY) via inverse pinhole `dX = dx_pixel * Z / f_x`. "
                             "Same dim, drop-in feature swap on top of --use-depth path. "
                             "Z lookup matches use_depth (default 30m fallback).")
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG seed for torch/numpy/random (default: nondeterministic)")
    parser.add_argument("--lam-struct", type=float, default=0.0,
                        help="Structural Consensus Constraint aux-loss weight "
                             "(CDRMOT lever, Lever A). 0=disabled. Typical sweep "
                             "{0.1, 0.5, 1.0}. Pairwise-distance MSE between motion "
                             "and language embedding manifolds.")
    parser.add_argument("--lam-angle", type=float, default=0.5,
                        help="Weight on triplet-angle term inside StructuralConsensusLoss "
                             "(only used when --struct-mode dist_angle).")
    parser.add_argument("--struct-mode", default="dist", choices=["dist", "dist_angle"],
                        help="Structural-loss variant: 'dist' (pairwise-distance only, "
                             "Phase A1 default) or 'dist_angle' (+ triplet-angle term).")
    args = parser.parse_args()

    if args.seed is not None:
        import random, numpy as _np
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        _np.random.seed(args.seed)
        random.seed(args.seed)
        print(f"Seeded RNG: torch/numpy/random = {args.seed}")

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

    # Infer lang_dim from encoder. Covers MiniLM (384), BGE-base/E5-base (768),
    # and CLIP text towers via "clip:<arch>:<pretrained>" prefix (512 for ViT-B-32).
    if args.text_encoder.startswith("clip:"):
        # Parse without instantiating the model twice — derive dim from arch.
        # ViT-B-32, ViT-B-16, ViT-L-14 all use shared embed_dim 512 in open_clip
        # for these checkpoints; probe via TextEncoder if uncertain.
        from gmc_link.text_utils import TextEncoder as _TE
        _probe = _TE(model_name=args.text_encoder, device=str(device))
        lang_dim = _probe.embedding_dim
        del _probe
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    else:
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
            motion_filter=args.motion_filter,
            class_filter=args.class_filter,
            use_clip_feat=args.use_clip_feat,
            clip_cache_path=args.clip_cache_path,
            clip_proj_dim=args.clip_dim,
            fusion_site=args.fusion_site,
            lang_passthrough=args.lang_passthrough,
            app_proj_dim=args.app_proj_dim,
            use_depth=args.use_depth,
            depth_cache_dir=args.depth_cache_dir,
            depth_source=args.depth_source,
            identity_init_depth=args.identity_init_depth,
            world_xy=args.world_xy,
            lam_struct=args.lam_struct,
            lam_angle=args.lam_angle,
            struct_mode=args.struct_mode,
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
            motion_filter=args.motion_filter,
            class_filter=args.class_filter,
            use_clip_feat=args.use_clip_feat,
            clip_cache_path=args.clip_cache_path,
            clip_proj_dim=args.clip_dim,
            fusion_site=args.fusion_site,
            lang_passthrough=args.lang_passthrough,
            app_proj_dim=args.app_proj_dim,
            use_depth=args.use_depth,
            depth_cache_dir=args.depth_cache_dir,
            depth_source=args.depth_source,
            identity_init_depth=args.identity_init_depth,
            world_xy=args.world_xy,
            lam_struct=args.lam_struct,
            lam_angle=args.lam_angle,
            struct_mode=args.struct_mode,
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
        motion_filter=args.motion_filter,
        class_filter=args.class_filter,
        use_clip_feat=args.use_clip_feat,
        clip_cache_path=args.clip_cache_path,
        clip_proj_dim=args.clip_dim,
        fusion_site=args.fusion_site,
        lang_passthrough=args.lang_passthrough,
        app_proj_dim=args.app_proj_dim,
        use_depth=args.use_depth,
        depth_cache_dir=args.depth_cache_dir,
            depth_source=args.depth_source,
        identity_init_depth=args.identity_init_depth,
        world_xy=args.world_xy,
        lam_struct=args.lam_struct,
        lam_angle=args.lam_angle,
        struct_mode=args.struct_mode,
    )


if __name__ == "__main__":
    main()
