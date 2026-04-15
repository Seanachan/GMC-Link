#!/usr/bin/env python3
"""
Plot diagnostic results from Layers 1-5.
Generates a summary figure showing the key findings.

Usage:
    python diagnostics/plot_diagnostics.py
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt

RESULTS_DIR = "diagnostics/results"
OUTPUT_PATH = "diagnostics/results/diagnostic_summary.png"


def plot_layer4_distribution_gap(ax, data):
    """Bar chart of standardized mean shifts per dimension."""
    dim_names = list(data["dim_names"])
    shifts = data["shifts"]

    # Truncate long names
    short_names = [n[:10] for n in dim_names]

    colors = ["#e74c3c" if s > 1.0 else "#f39c12" if s > 0.5 else "#2ecc71" for s in shifts]
    bars = ax.barh(range(len(dim_names)), shifts, color=colors)
    ax.set_yticks(range(len(dim_names)))
    ax.set_yticklabels(short_names, fontsize=7)
    ax.set_xlabel("|Δμ| / σ_train", fontsize=8)
    ax.set_title("L4: Train vs Inference Shift", fontsize=9, fontweight="bold")
    ax.axvline(x=1.0, color="red", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.invert_yaxis()


def plot_layer4_variance_ratio(ax, data):
    """Bar chart of variance ratios (inference σ / train σ)."""
    train_vecs = data["train_vecs"]
    infer_vecs = data["infer_vecs"]
    dim_names = list(data["dim_names"])
    short_names = [n[:10] for n in dim_names]

    ratios = []
    for d in range(13):
        t_std = train_vecs[:, d].std()
        i_std = infer_vecs[:, d].std()
        ratios.append(i_std / (t_std + 1e-8))
    ratios = np.array(ratios)

    colors = ["#e74c3c" if (r > 3 or r < 0.1) else "#f39c12" if (r > 2 or r < 0.33) else "#2ecc71"
              for r in ratios]
    ax.barh(range(len(dim_names)), np.log10(ratios + 1e-8), color=colors)
    ax.set_yticks(range(len(dim_names)))
    ax.set_yticklabels(short_names, fontsize=7)
    ax.set_xlabel("log₁₀(σ_infer / σ_train)", fontsize=8)
    ax.set_title("L4: Variance Ratio", fontsize=9, fontweight="bold")
    ax.axvline(x=0, color="gray", linestyle="-", alpha=0.3, linewidth=0.8)
    ax.invert_yaxis()


def plot_layer5_scores(ax, data):
    """Per-expression GT vs non-GT cosine similarity."""
    results = list(data["results"])
    results = sorted(results, key=lambda x: -x["auc"])

    exprs = [r["sentence"][:25] for r in results]
    gt_means = [r["gt_cos_mean"] for r in results]
    nongt_means = [r["nongt_cos_mean"] for r in results]

    y = np.arange(len(exprs))
    height = 0.35
    ax.barh(y - height/2, gt_means, height, label="GT match", color="#27ae60", alpha=0.8)
    ax.barh(y + height/2, nongt_means, height, label="Non-GT", color="#e74c3c", alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(exprs, fontsize=6)
    ax.set_xlabel("Mean Cosine Similarity", fontsize=8)
    ax.set_title("L5: Live Inference — GT vs Non-GT Cosine", fontsize=9, fontweight="bold")
    ax.legend(fontsize=7, loc="lower right")
    ax.axvline(x=0, color="gray", linestyle="-", alpha=0.3)
    ax.invert_yaxis()


def plot_layer5_auc(ax, data):
    """Per-expression AUC."""
    results = list(data["results"])
    results = sorted(results, key=lambda x: -x["auc"])

    exprs = [r["sentence"][:25] for r in results]
    aucs = [r["auc"] for r in results]

    colors = ["#e74c3c" if a < 0.5 else "#f39c12" if a < 0.65 else "#2ecc71" for a in aucs]
    ax.barh(range(len(exprs)), aucs, color=colors, alpha=0.8)
    ax.set_yticks(range(len(exprs)))
    ax.set_yticklabels(exprs, fontsize=6)
    ax.set_xlabel("AUC", fontsize=8)
    ax.set_title("L5: Per-Expression AUC", fontsize=9, fontweight="bold")
    ax.axvline(x=0.5, color="red", linestyle="--", alpha=0.5, linewidth=0.8, label="Chance")
    ax.legend(fontsize=7)
    ax.invert_yaxis()


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load available results
    l4_path = os.path.join(RESULTS_DIR, "layer4_distribution_gap_0011.npz")
    l5_path = os.path.join(RESULTS_DIR, "layer5_live_scores_0011.npz")

    have_l4 = os.path.exists(l4_path)
    have_l5 = os.path.exists(l5_path)

    if not have_l4 and not have_l5:
        print("No result files found. Run diagnostic scripts first.")
        return

    # Layout: 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("GMC-Link Diagnostic Summary (V1, seq 0011)", fontsize=13, fontweight="bold")

    if have_l4:
        l4_data = np.load(l4_path, allow_pickle=True)
        plot_layer4_distribution_gap(axes[0, 0], l4_data)
        plot_layer4_variance_ratio(axes[0, 1], l4_data)
    else:
        axes[0, 0].text(0.5, 0.5, "Layer 4 not available", ha="center", va="center")
        axes[0, 1].text(0.5, 0.5, "Layer 4 not available", ha="center", va="center")

    if have_l5:
        l5_data = np.load(l5_path, allow_pickle=True)
        plot_layer5_scores(axes[1, 0], l5_data)
        plot_layer5_auc(axes[1, 1], l5_data)
    else:
        axes[1, 0].text(0.5, 0.5, "Layer 5 not available", ha="center", va="center")
        axes[1, 1].text(0.5, 0.5, "Layer 5 not available", ha="center", va="center")

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
