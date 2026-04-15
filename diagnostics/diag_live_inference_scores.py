#!/usr/bin/env python3
"""
Layer 5: Live Inference Score Analysis
=======================================
Runs the full GMCLinkManager pipeline on NeuralSORT tracks for each motion
expression in a sequence. Records cosine similarity per track per frame and
labels each as GT-match or non-match.

Reports: per-expression AUC, mean cosine for GT vs non-GT, score traces.
Saves: diagnostics/results/layer5_live_scores.npz

Usage:
    python diagnostics/diag_live_inference_scores.py
    python diagnostics/diag_live_inference_scores.py --seq 0005
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import json
from collections import defaultdict

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gmc_link.manager import GMCLinkManager
from gmc_link.text_utils import TextEncoder
from gmc_link.dataset import load_refer_kitti_expressions, is_motion_expression

# ── V1 config ────────────────────────────────────────────────────────────
DATA_ROOT = "refer-kitti"
WEIGHTS_PATH = "gmc_link_weights_v1train.pth"
TRACK_DIR = "NeuralSORT"
SEQUENCE = "0011"
RESULTS_DIR = "diagnostics/results"


class DummyTrack:
    def __init__(self, oid, x, y, bw, bh):
        self.id = oid
        self.bbox = [x, y, x + bw, y + bh]
        self.centroid = (x + bw / 2, y + bh / 2)


def load_neuralsort_tracks(seq):
    """Load car + pedestrian NeuralSORT tracks → {frame_1idx: [(oid, x, y, w, h)]}"""
    tracks = defaultdict(list)
    for cls in ["car", "pedestrian"]:
        path = os.path.join(TRACK_DIR, seq, cls, "predict.txt")
        if not os.path.exists(path):
            continue
        data = np.loadtxt(path, delimiter=",")
        if data.ndim == 1:
            data = data[np.newaxis]
        for row in data:
            fid = int(row[0])
            tracks[fid].append((int(row[1]), row[2], row[3], row[4], row[5]))
    return tracks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=WEIGHTS_PATH)
    parser.add_argument("--seq", default=SEQUENCE)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Seq: {args.seq} | Weights: {args.weights}")

    encoder = TextEncoder(device=str(device))

    frame_dir = os.path.join(DATA_ROOT, "KITTI", "training", "image_02", args.seq)
    frame_files = sorted(f for f in os.listdir(frame_dir) if f.endswith(".png"))
    ns_tracks = load_neuralsort_tracks(args.seq)

    expr_dir = os.path.join(DATA_ROOT, "expression", args.seq)
    expressions = load_refer_kitti_expressions(expr_dir)
    motion_exprs = [e for e in expressions if is_motion_expression(e["sentence"])]
    print(f"Motion expressions: {len(motion_exprs)}/{len(expressions)}")

    per_expr_results = []

    for expr_idx, expr in enumerate(motion_exprs):
        sentence = expr["sentence"]
        label_map = expr["label"]
        lang_emb = encoder.encode(sentence)

        # Build GT lookup: {frame_0idx: set(track_ids)}
        gt_by_frame = {}
        for fid_str, tids in label_map.items():
            gt_by_frame[int(fid_str)] = set(tids)

        # Fresh manager per expression
        linker = GMCLinkManager(weights_path=args.weights, device=str(device), lang_dim=384)

        gt_cosines = []
        nongt_cosines = []
        gt_sigmoid_scores = []
        nongt_sigmoid_scores = []

        for frame_0idx, fname in enumerate(frame_files):
            frame_1idx = frame_0idx + 1
            frame_img = cv2.imread(os.path.join(frame_dir, fname))
            if frame_img is None:
                continue

            if frame_1idx not in ns_tracks:
                linker.process_frame(frame_img, [], lang_emb)
                continue

            detections = ns_tracks[frame_1idx]
            active = [DummyTrack(o, x, y, w, h) for o, x, y, w, h in detections]
            det_arr = np.array([[x, y, x + w, y + h] for _, x, y, w, h in detections]) \
                if detections else None

            scores, _, cosines = linker.process_frame(
                frame_img, active, lang_emb, detections=det_arr
            )

            gt_tids = gt_by_frame.get(frame_0idx, set())
            for tid, cos_val in cosines.items():
                sig_val = scores.get(tid, 0.0)
                if tid in gt_tids:
                    gt_cosines.append(cos_val)
                    gt_sigmoid_scores.append(sig_val)
                else:
                    nongt_cosines.append(cos_val)
                    nongt_sigmoid_scores.append(sig_val)

        if gt_cosines and nongt_cosines:
            gt_arr = np.array(gt_cosines)
            nongt_arr = np.array(nongt_cosines)
            separation = gt_arr.mean() - nongt_arr.mean()

            from scipy.stats import mannwhitneyu
            try:
                U, _ = mannwhitneyu(gt_arr, nongt_arr, alternative="greater")
                auc = U / (len(gt_arr) * len(nongt_arr))
            except ValueError:
                auc = 0.5

            per_expr_results.append({
                "sentence": sentence,
                "n_gt": len(gt_cosines),
                "n_nongt": len(nongt_cosines),
                "gt_cos_mean": gt_arr.mean(),
                "gt_cos_std": gt_arr.std(),
                "nongt_cos_mean": nongt_arr.mean(),
                "nongt_cos_std": nongt_arr.std(),
                "separation": separation,
                "auc": auc,
                "gt_sig_mean": np.mean(gt_sigmoid_scores),
                "nongt_sig_mean": np.mean(nongt_sigmoid_scores),
            })

        if (expr_idx + 1) % 10 == 0:
            print(f"  {expr_idx + 1}/{len(motion_exprs)} expressions processed")

    # ── Report ────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("LAYER 5: LIVE INFERENCE SCORE ANALYSIS")
    print("=" * 78)

    if not per_expr_results:
        print("  No results — check data paths and NeuralSORT tracks.")
        return

    print(f"\n  {'Expression':<40} {'GT cos':>8} {'NonGT':>8} {'Sep':>7} "
          f"{'AUC':>6} {'GT sig':>7} {'NG sig':>7}")
    print("  " + "-" * 84)
    for r in sorted(per_expr_results, key=lambda x: -x["auc"]):
        name = r["sentence"][:39]
        print(f"  {name:<40} {r['gt_cos_mean']:>8.4f} {r['nongt_cos_mean']:>8.4f} "
              f"{r['separation']:>+7.4f} {r['auc']:>6.3f} "
              f"{r['gt_sig_mean']:>7.4f} {r['nongt_sig_mean']:>7.4f}")

    # Aggregate
    all_sep = [r["separation"] for r in per_expr_results]
    all_auc = [r["auc"] for r in per_expr_results]
    all_gt_cos = [r["gt_cos_mean"] for r in per_expr_results]
    all_nongt_cos = [r["nongt_cos_mean"] for r in per_expr_results]

    print(f"\n  Aggregate ({len(per_expr_results)} expressions):")
    print(f"    Mean GT cosine:     {np.mean(all_gt_cos):.4f}")
    print(f"    Mean non-GT cosine: {np.mean(all_nongt_cos):.4f}")
    print(f"    Mean separation:    {np.mean(all_sep):+.4f} (std={np.std(all_sep):.4f})")
    print(f"    Mean AUC:           {np.mean(all_auc):.3f} (std={np.std(all_auc):.3f})")

    # Compare with Layer 3 (if available)
    layer3_path = os.path.join(RESULTS_DIR, f"layer3_gt_cosine_{args.seq}.npz")
    if os.path.exists(layer3_path):
        l3 = np.load(layer3_path, allow_pickle=True)["results"]
        l3_auc = np.mean([r["auc"] for r in l3])
        print(f"\n  Layer 3 (GT centroids) AUC: {l3_auc:.3f}")
        print(f"  Layer 5 (NeuralSORT)   AUC: {np.mean(all_auc):.3f}")
        drop = l3_auc - np.mean(all_auc)
        if drop > 0.05:
            print(f"  ⚠ AUC drops {drop:.3f} from GT→tracker — tracker noise is hurting.")
        else:
            print(f"  ✓ Minimal AUC drop ({drop:+.3f}) — tracker quality is not the bottleneck.")

    # Interpretation
    print()
    mean_auc = np.mean(all_auc)
    if mean_auc < 0.55:
        print("  ⚠ Near-chance AUC at inference — scores are not discriminative.")
    elif mean_auc < 0.70:
        print("  ⚠ Weak discrimination at inference.")
    else:
        print("  ✓ Reasonable discrimination at inference.")

    # ── Save ──────────────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_path = os.path.join(RESULTS_DIR, f"layer5_live_scores_{args.seq}.npz")
    np.savez(save_path, results=per_expr_results)
    print(f"\n  Saved: {save_path}")

    # ── Visualization ─────────────────────────────────────────────────
    sorted_results = sorted(per_expr_results, key=lambda x: -x["auc"])
    exprs = [r["sentence"][:30] for r in sorted_results]
    y = np.arange(len(exprs))

    fig, axes = plt.subplots(1, 2, figsize=(14, max(5, len(exprs) * 0.35)))
    fig.suptitle(f"Layer 5: Live Inference Scores (seq {args.seq}, NeuralSORT)",
                 fontsize=12, fontweight="bold")

    # Left: GT vs non-GT cosine
    h = 0.35
    axes[0].barh(y - h/2, [r["gt_cos_mean"] for r in sorted_results], h,
                 label="GT match", color="#27ae60", alpha=0.8)
    axes[0].barh(y + h/2, [r["nongt_cos_mean"] for r in sorted_results], h,
                 label="Non-GT", color="#e74c3c", alpha=0.8)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(exprs, fontsize=7)
    axes[0].set_xlabel("Mean Cosine Similarity")
    axes[0].set_title("GT vs Non-GT Cosine")
    axes[0].legend(fontsize=8)
    axes[0].axvline(x=0, color="gray", linestyle="-", alpha=0.3)
    axes[0].invert_yaxis()

    # Right: AUC per expression
    aucs = [r["auc"] for r in sorted_results]
    colors = ["#e74c3c" if a < 0.5 else "#f39c12" if a < 0.65 else "#27ae60" for a in aucs]
    axes[1].barh(y, aucs, color=colors, alpha=0.8)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(exprs, fontsize=7)
    axes[1].set_xlabel("AUC")
    axes[1].set_title("Per-Expression AUC")
    axes[1].axvline(x=0.5, color="red", linestyle="--", alpha=0.5, label="Chance")
    axes[1].legend(fontsize=8)
    axes[1].invert_yaxis()

    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, f"layer5_live_scores_{args.seq}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"  Plot saved: {plot_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
