#!/usr/bin/env python3
"""
Layer 1: Training Convergence Check
====================================
Loads the trained V1 weights and evaluates them on the V1 training data.
Reports: retrieval accuracy, InfoNCE loss, temperature, cosine sim stats.

Usage:
    python diagnostics/diag_training_convergence.py
    python diagnostics/diag_training_convergence.py --weights gmc_link_weights.pth  # test other weights
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gmc_link.alignment import MotionLanguageAligner
from gmc_link.dataset import MotionLanguageDataset, collate_fn, build_training_data
from gmc_link.text_utils import TextEncoder

# ── V1 config ────────────────────────────────────────────────────────────
DATA_ROOT = "refer-kitti"
WEIGHTS_PATH = "gmc_link_weights_v1train.pth"
TRAIN_SEQUENCES = [
    "0001", "0002", "0003", "0004", "0006",
    "0007", "0008", "0009", "0010", "0012",
    "0014", "0015", "0016", "0018", "0020",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=WEIGHTS_PATH)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Weights: {args.weights}")

    # ── Load model ────────────────────────────────────────────────────
    model = MotionLanguageAligner(motion_dim=13, lang_dim=384, embed_dim=256).to(device)
    checkpoint = torch.load(args.weights, map_location=device)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
        temperature = checkpoint.get("temperature", 1.0)
    else:
        model.load_state_dict(checkpoint)
        temperature = 1.0
    model.eval()
    print(f"Temperature (saved): {temperature:.4f}")

    # ── Build training data ───────────────────────────────────────────
    encoder = TextEncoder(device=str(device))
    all_motions, all_languages, all_labels = build_training_data(
        data_root=DATA_ROOT, sequences=TRAIN_SEQUENCES, text_encoder=encoder,
    )
    print(f"Training samples: {len(all_motions)}")

    n_classes = len(set(all_labels))
    print(f"Unique expression classes: {n_classes}")

    dataset = MotionLanguageDataset(all_motions, all_languages, all_labels)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=4)

    # ── Evaluate ──────────────────────────────────────────────────────
    total_loss = 0.0
    correct_m2l = 0
    correct_l2m = 0
    total = 0
    all_diag_cosines = []   # cosine sim of positive (diagonal) pairs
    all_offdiag_cosines = []  # cosine sim of negative (off-diagonal) pairs

    with torch.no_grad():
        for motion_feat, lang_feat, expr_ids in loader:
            motion_feat = motion_feat.to(device)
            lang_feat = lang_feat.to(device)
            B = motion_feat.size(0)

            sim_matrix = model(motion_feat, lang_feat)  # (B, B) cosine sim

            # InfoNCE loss
            logits = sim_matrix / temperature
            targets = torch.arange(B, device=device)
            m2l_loss = F.cross_entropy(logits, targets)
            l2m_loss = F.cross_entropy(logits.t(), targets)
            loss = (m2l_loss + l2m_loss) / 2.0
            total_loss += loss.item()

            # Retrieval accuracy
            correct_m2l += (sim_matrix.argmax(dim=1) == targets).sum().item()
            correct_l2m += (sim_matrix.argmax(dim=0) == targets).sum().item()
            total += B

            # Collect cosine sim stats
            sim_np = sim_matrix.cpu().numpy()
            for i in range(B):
                all_diag_cosines.append(sim_np[i, i])
                for j in range(B):
                    if i != j:
                        all_offdiag_cosines.append(sim_np[i, j])

    avg_loss = total_loss / len(loader)
    acc_m2l = correct_m2l / total
    acc_l2m = correct_l2m / total

    diag = np.array(all_diag_cosines)
    offdiag = np.array(all_offdiag_cosines)

    # ── Report ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("LAYER 1: TRAINING CONVERGENCE REPORT")
    print("=" * 60)
    print(f"  Samples:           {total}")
    print(f"  Classes:           {n_classes}")
    print(f"  Temperature:       {temperature:.4f}")
    print(f"  InfoNCE Loss:      {avg_loss:.4f}")
    print(f"  Accuracy (M→L):    {acc_m2l:.2%}")
    print(f"  Accuracy (L→M):    {acc_l2m:.2%}")
    print()
    print("  Cosine Similarity (positive/diagonal pairs):")
    print(f"    mean={diag.mean():.4f}  std={diag.std():.4f}  "
          f"min={diag.min():.4f}  max={diag.max():.4f}")
    print("  Cosine Similarity (negative/off-diagonal pairs):")
    print(f"    mean={offdiag.mean():.4f}  std={offdiag.std():.4f}  "
          f"min={offdiag.min():.4f}  max={offdiag.max():.4f}")
    print(f"  Mean separation (diag - offdiag): {diag.mean() - offdiag.mean():.4f}")
    print()

    # Interpretation hints
    if acc_m2l < 0.10:
        print("  ⚠ Very low retrieval accuracy — model may not have learned.")
    elif acc_m2l < 0.50:
        print("  ⚠ Moderate accuracy — model learned something but struggles.")
    else:
        print("  ✓ Good training accuracy.")

    if diag.mean() - offdiag.mean() < 0.05:
        print("  ⚠ Tiny mean separation — embeddings are poorly separated.")
    elif diag.mean() - offdiag.mean() < 0.15:
        print("  ⚠ Modest separation — some signal but overlapping distributions.")
    else:
        print("  ✓ Clear separation between positive and negative pairs.")

    print("=" * 60)

    # ── Visualization ─────────────────────────────────────────────
    os.makedirs("diagnostics/results", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("Layer 1: Training Convergence", fontsize=12, fontweight="bold")

    # Left: overlapping histograms of positive vs negative cosine sim
    axes[0].hist(diag, bins=80, alpha=0.7, label=f"Positive (diag) μ={diag.mean():.3f}",
                 color="#27ae60", density=True)
    axes[0].hist(offdiag, bins=80, alpha=0.5, label=f"Negative (off-diag) μ={offdiag.mean():.3f}",
                 color="#e74c3c", density=True)
    axes[0].set_xlabel("Cosine Similarity")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Positive vs Negative Cosine Similarity")
    axes[0].legend(fontsize=8)

    # Right: summary metrics as text box
    axes[1].axis("off")
    metrics_text = (
        f"Samples: {total}\n"
        f"Classes: {n_classes}\n"
        f"Temperature: {temperature:.4f}\n"
        f"InfoNCE Loss: {avg_loss:.4f}\n"
        f"Accuracy (M→L): {acc_m2l:.2%}\n"
        f"Accuracy (L→M): {acc_l2m:.2%}\n"
        f"\nPositive cosine: {diag.mean():.4f} ± {diag.std():.4f}\n"
        f"Negative cosine: {offdiag.mean():.4f} ± {offdiag.std():.4f}\n"
        f"Separation: {diag.mean() - offdiag.mean():.4f}"
    )
    axes[1].text(0.1, 0.5, metrics_text, transform=axes[1].transAxes,
                 fontsize=11, verticalalignment="center", fontfamily="monospace",
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    axes[1].set_title("Summary Metrics")

    plt.tight_layout()
    plot_path = "diagnostics/results/layer1_training_convergence.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"  Plot saved: {plot_path}")


if __name__ == "__main__":
    main()
