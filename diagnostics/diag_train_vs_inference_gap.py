#!/usr/bin/env python3
"""
Layer 4: Train vs Inference Distribution Gap
=============================================
Compares the distribution of 13D motion vectors from training data (GT centroids)
versus live inference (NeuralSORT tracker + GMCLinkManager) on the same sequence.

If the distributions differ significantly, the model is being asked to generalize
to inputs it never trained on — a likely cause of poor similarity scores.

Reports: per-dimension mean/std shift, overall KL divergence estimate.
Saves: diagnostics/results/layer4_distribution_gap.npz

Usage:
    python diagnostics/diag_train_vs_inference_gap.py
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
from collections import defaultdict

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gmc_link.manager import GMCLinkManager
from gmc_link.text_utils import TextEncoder
from gmc_link.dataset import build_training_data

# ── V1 config ────────────────────────────────────────────────────────────
DATA_ROOT = "refer-kitti"
WEIGHTS_PATH = "gmc_link_weights_v1train.pth"
TRACK_DIR = "NeuralSORT"
SEQUENCE = "0011"
TRAIN_SEQUENCES = [
    "0001", "0002", "0003", "0004", "0006",
    "0007", "0008", "0009", "0010", "0012",
    "0014", "0015", "0016", "0018", "0020",
]
RESULTS_DIR = "diagnostics/results"

DIM_NAMES = [
    "res_dx_short", "res_dy_short",
    "res_dx_mid", "res_dy_mid",
    "res_dx_long", "res_dy_long",
    "dw", "dh",
    "cx", "cy", "w", "h", "snr",
]


def collect_inference_vectors(sequence, weights_path, device):
    """Run GMCLinkManager on NeuralSORT tracks and collect motion vectors."""
    encoder = TextEncoder(device=str(device))
    # Use a dummy expression just to drive the pipeline
    lang_emb = encoder.encode("moving cars")

    linker = GMCLinkManager(weights_path=weights_path, device=str(device), lang_dim=384)

    frame_dir = os.path.join(DATA_ROOT, "KITTI", "training", "image_02", sequence)
    frame_files = sorted(f for f in os.listdir(frame_dir) if f.endswith(".png"))

    # Load NeuralSORT tracks
    car_path = os.path.join(TRACK_DIR, sequence, "car", "predict.txt")
    tracks_by_frame = defaultdict(list)
    if os.path.exists(car_path):
        data = np.loadtxt(car_path, delimiter=",")
        if data.ndim == 1:
            data = data[np.newaxis]
        for row in data:
            fid = int(row[0])
            tracks_by_frame[fid].append((int(row[1]), row[2], row[3], row[4], row[5]))

    all_vectors = []

    class _Track:
        def __init__(self, oid, x, y, bw, bh):
            self.id = oid
            self.bbox = [x, y, x + bw, y + bh]
            self.centroid = (x + bw / 2, y + bh / 2)

    for frame_0idx, fname in enumerate(frame_files):
        frame_1idx = frame_0idx + 1
        if frame_1idx not in tracks_by_frame:
            # Still process frame for ego-motion estimation
            frame_img = cv2.imread(os.path.join(frame_dir, fname))
            if frame_img is not None:
                linker.process_frame(frame_img, [], lang_emb)
            continue

        frame_img = cv2.imread(os.path.join(frame_dir, fname))
        if frame_img is None:
            continue

        detections = tracks_by_frame[frame_1idx]
        active = [_Track(o, x, y, w, h) for o, x, y, w, h in detections]
        det_arr = np.array([[x, y, x + w, y + h] for _, x, y, w, h in detections])

        _, velocities, _ = linker.process_frame(
            frame_img, active, lang_emb, detections=det_arr
        )

        for tid, vec in velocities.items():
            all_vectors.append(vec)

    return np.array(all_vectors, dtype=np.float32) if all_vectors else np.empty((0, 13))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=WEIGHTS_PATH)
    parser.add_argument("--seq", default=SEQUENCE)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Collect training vectors ──────────────────────────────────────
    print("Building training data vectors...")
    encoder = TextEncoder(device=str(device))
    train_motions, _, _ = build_training_data(
        data_root=DATA_ROOT, sequences=TRAIN_SEQUENCES, text_encoder=encoder,
    )
    train_vecs = np.array(train_motions, dtype=np.float32)
    print(f"Training vectors: {train_vecs.shape}")

    # ── Collect inference vectors ─────────────────────────────────────
    print(f"Collecting inference vectors from seq {args.seq}...")
    infer_vecs = collect_inference_vectors(args.seq, args.weights, device)
    print(f"Inference vectors: {infer_vecs.shape}")

    if len(infer_vecs) == 0:
        print("ERROR: No inference vectors collected.")
        return

    # ── Per-dimension comparison ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("LAYER 4: TRAIN vs INFERENCE DISTRIBUTION GAP")
    print("=" * 70)

    print(f"\n  {'Dimension':<15} {'Train μ':>9} {'Train σ':>9} {'Infer μ':>9} "
          f"{'Infer σ':>9} {'Δμ':>9} {'σ ratio':>9}")
    print("  " + "-" * 72)

    shifts = []
    for dim in range(13):
        t_mean = train_vecs[:, dim].mean()
        t_std = train_vecs[:, dim].std()
        i_mean = infer_vecs[:, dim].mean()
        i_std = infer_vecs[:, dim].std()
        delta_mean = i_mean - t_mean
        sigma_ratio = i_std / (t_std + 1e-8)
        shifts.append(abs(delta_mean) / (t_std + 1e-8))  # standardized shift

        flag = ""
        if abs(delta_mean) / (t_std + 1e-8) > 1.0:
            flag = " ⚠"
        elif sigma_ratio > 3.0 or sigma_ratio < 0.33:
            flag = " ⚠"

        print(f"  {DIM_NAMES[dim]:<15} {t_mean:>9.4f} {t_std:>9.4f} {i_mean:>9.4f} "
              f"{i_std:>9.4f} {delta_mean:>+9.4f} {sigma_ratio:>9.2f}{flag}")

    shifts = np.array(shifts)

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n  Standardized shift (|Δμ|/σ_train):")
    print(f"    mean={shifts.mean():.3f}  max={shifts.max():.3f} "
          f"(dim: {DIM_NAMES[shifts.argmax()]})")

    n_flagged = (shifts > 1.0).sum()
    print(f"    Dimensions with shift > 1σ: {n_flagged}/13")

    print()
    if shifts.mean() > 1.0:
        print("  ⚠ LARGE DISTRIBUTION GAP: inference vectors look very different from training.")
        print("    The model is being asked to generalize to an unseen input distribution.")
    elif shifts.mean() > 0.5:
        print("  ⚠ Moderate gap — some dimensions shifted meaningfully.")
    else:
        print("  ✓ Distributions are reasonably similar.")

    # ── Save ──────────────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_path = os.path.join(RESULTS_DIR, f"layer4_distribution_gap_{args.seq}.npz")
    np.savez(save_path, train_vecs=train_vecs, infer_vecs=infer_vecs,
             dim_names=DIM_NAMES, shifts=shifts)
    print(f"  Saved: {save_path}")

    # ── Visualization ─────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Layer 4: Train vs Inference Distribution Gap (seq {args.seq})",
                 fontsize=12, fontweight="bold")

    # Top-left: standardized mean shift
    colors_shift = ["#e74c3c" if s > 1.0 else "#f39c12" if s > 0.5 else "#27ae60"
                    for s in shifts]
    axes[0, 0].barh(range(13), shifts, color=colors_shift)
    axes[0, 0].set_yticks(range(13))
    axes[0, 0].set_yticklabels(DIM_NAMES, fontsize=8)
    axes[0, 0].set_xlabel("|Δμ| / σ_train")
    axes[0, 0].set_title("Standardized Mean Shift")
    axes[0, 0].axvline(x=1.0, color="red", linestyle="--", alpha=0.5)
    axes[0, 0].invert_yaxis()

    # Top-right: variance ratio (log scale)
    ratios = []
    for d in range(13):
        t_std = train_vecs[:, d].std()
        i_std = infer_vecs[:, d].std()
        ratios.append(i_std / (t_std + 1e-8))
    log_ratios = np.log10(np.array(ratios) + 1e-8)
    colors_var = ["#e74c3c" if (r > 3 or r < 0.1) else "#f39c12" if (r > 2 or r < 0.33)
                  else "#27ae60" for r in ratios]
    axes[0, 1].barh(range(13), log_ratios, color=colors_var)
    axes[0, 1].set_yticks(range(13))
    axes[0, 1].set_yticklabels(DIM_NAMES, fontsize=8)
    axes[0, 1].set_xlabel("log₁₀(σ_infer / σ_train)")
    axes[0, 1].set_title("Variance Ratio")
    axes[0, 1].axvline(x=0, color="gray", linestyle="-", alpha=0.3)
    axes[0, 1].invert_yaxis()

    # Bottom: overlapping histograms for worst dimensions
    worst_dims = np.argsort([-abs(np.log10(r + 1e-8)) for r in ratios])[:2]
    for plot_idx, dim in enumerate(worst_dims):
        ax = axes[1, plot_idx]
        t_vals = np.clip(train_vecs[:, dim], np.percentile(train_vecs[:, dim], 1),
                         np.percentile(train_vecs[:, dim], 99))
        i_vals = infer_vecs[:, dim]
        ax.hist(t_vals, bins=80, alpha=0.6, label="Train", color="#3498db", density=True)
        ax.hist(i_vals, bins=80, alpha=0.6, label="Inference", color="#e67e22", density=True)
        ax.set_xlabel(DIM_NAMES[dim])
        ax.set_ylabel("Density")
        ax.set_title(f"Distribution: {DIM_NAMES[dim]} (ratio={ratios[dim]:.3f})")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, f"layer4_distribution_gap_{args.seq}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"  Plot saved: {plot_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
