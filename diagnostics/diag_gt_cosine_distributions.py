#!/usr/bin/env python3
"""
Layer 3: GT vs Non-GT Cosine Similarity Distributions
======================================================
For sequence 0011, computes motion vectors from GT annotation centroids for
both GT-matching and non-matching tracks. Compares their cosine similarity
distributions against each expression.

Reports: per-expression mean separation, overlap, AUC.
Saves: diagnostics/results/layer3_gt_cosine.npz

Usage:
    python diagnostics/diag_gt_cosine_distributions.py
    python diagnostics/diag_gt_cosine_distributions.py --seq 0005
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

from gmc_link.alignment import MotionLanguageAligner
from gmc_link.dataset import (
    load_refer_kitti_expressions, load_labels_with_ids,
    is_motion_expression, FRAME_GAPS,
)
from gmc_link.utils import VELOCITY_SCALE, warp_points
from gmc_link.core import ORBHomographyEngine
from gmc_link.text_utils import TextEncoder

# ── V1 config ────────────────────────────────────────────────────────────
DATA_ROOT = "refer-kitti"
WEIGHTS_PATH = "gmc_link_weights_v1train.pth"
SEQUENCE = "0011"
RESULTS_DIR = "diagnostics/results"


def compute_motion_vector_for_track(
    track_centroids, frame_ids, frame_dir, orb_engine, frame_shape,
):
    """
    Given a track's centroid history, compute the 13D motion vector at the
    last available frame using the same logic as the training dataset.
    Returns list of (frame_id, motion_vector_13d) pairs.
    """
    h, w = frame_shape
    results = []
    sorted_frames = sorted(frame_ids)
    homography_cache = {}

    for i in range(len(sorted_frames)):
        curr_fid = sorted_frames[i]
        scale_velocities = []
        any_valid = False
        best_bg_residual = np.zeros(2, dtype=np.float32)

        for gap_idx, gap in enumerate(FRAME_GAPS):
            if i >= gap:
                past_fid = sorted_frames[i - gap]
                # Compute ego motion
                cache_key = (past_fid, curr_fid)
                if cache_key not in homography_cache:
                    img_past = cv2.imread(os.path.join(frame_dir, f"{past_fid:06d}.png"))
                    img_curr = cv2.imread(os.path.join(frame_dir, f"{curr_fid:06d}.png"))
                    if img_past is not None and img_curr is not None:
                        H, bg_res = orb_engine.estimate_homography(img_past, img_curr)
                        homography_cache[cache_key] = (H, bg_res)
                    else:
                        homography_cache[cache_key] = (None, np.zeros(2))
                H, bg_res = homography_cache[cache_key]

                cx1, cy1, _, _ = track_centroids[past_fid]
                cx2, cy2, _, _ = track_centroids[curr_fid]

                raw_dx = (cx2 - cx1) / w * VELOCITY_SCALE
                raw_dy = (cy2 - cy1) / h * VELOCITY_SCALE

                ego_dx, ego_dy = 0.0, 0.0
                if H is not None:
                    warped = warp_points(np.array([[cx1, cy1]], dtype=np.float32), H)[0]
                    ego_dx = (warped[0] - cx1) / w * VELOCITY_SCALE
                    ego_dy = (warped[1] - cy1) / h * VELOCITY_SCALE

                scale_velocities.append((raw_dx - ego_dx, raw_dy - ego_dy))
                any_valid = True
                if gap_idx == 1:
                    best_bg_residual = bg_res
            else:
                scale_velocities.append((0.0, 0.0))

        if not any_valid:
            continue

        # dw, dh from mid-scale
        mid_gap = FRAME_GAPS[1]
        if i >= mid_gap:
            _, _, bw1, bh1 = track_centroids[sorted_frames[i - mid_gap]]
            _, _, bw2, bh2 = track_centroids[curr_fid]
            dw = (bw2 - bw1) / w * VELOCITY_SCALE
            dh = (bh2 - bh1) / h * VELOCITY_SCALE
        else:
            dw, dh = 0.0, 0.0
            _, _, bw1, bh1 = track_centroids[curr_fid]

        cx, cy, bw, bh_val = track_centroids[curr_fid]
        mid_dx, mid_dy = scale_velocities[1]
        obj_speed = np.sqrt(mid_dx ** 2 + mid_dy ** 2)
        bg_mag = np.sqrt(
            (best_bg_residual[0] / w * VELOCITY_SCALE) ** 2
            + (best_bg_residual[1] / h * VELOCITY_SCALE) ** 2
        )
        snr = obj_speed / (bg_mag + 1e-6)

        vec = np.array([
            scale_velocities[0][0], scale_velocities[0][1],
            scale_velocities[1][0], scale_velocities[1][1],
            scale_velocities[2][0], scale_velocities[2][1],
            dw, dh, cx / w, cy / h, bw / w, bh_val / h, snr,
        ], dtype=np.float32)

        results.append((curr_fid, vec))

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=WEIGHTS_PATH)
    parser.add_argument("--seq", default=SEQUENCE)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Seq: {args.seq} | Weights: {args.weights}")

    # ── Load model ────────────────────────────────────────────────────
    model = MotionLanguageAligner(motion_dim=13, lang_dim=384, embed_dim=256).to(device)
    checkpoint = torch.load(args.weights, map_location=device)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # ── Load data ─────────────────────────────────────────────────────
    encoder = TextEncoder(device=str(device))
    orb_engine = ORBHomographyEngine(max_features=1500)

    frame_dir = os.path.join(DATA_ROOT, "KITTI", "training", "image_02", args.seq)
    labels_dir = os.path.join(DATA_ROOT, "KITTI", "labels_with_ids", "image_02", args.seq)
    expr_dir = os.path.join(DATA_ROOT, "expression", args.seq)

    labels_by_frame = load_labels_with_ids(labels_dir)
    expressions = load_refer_kitti_expressions(expr_dir)

    # Detect frame shape
    sample_img = cv2.imread(os.path.join(frame_dir, sorted(os.listdir(frame_dir))[0]))
    frame_shape = (sample_img.shape[0], sample_img.shape[1])
    h, w = frame_shape
    print(f"Frame shape: {w}x{h}")

    # Build track centroid database: {track_id: {frame_id: (cx, cy, w, h)}}
    all_track_centroids = defaultdict(dict)
    for fid, detections in labels_by_frame.items():
        for det in detections:
            tid = det["track_id"]
            x1 = det["x1_n"] * w
            y1 = det["y1_n"] * h
            bw = det["w_n"] * w
            bh = det["h_n"] * h
            all_track_centroids[tid][fid] = (x1 + bw / 2, y1 + bh / 2, bw, bh)

    # ── Per-expression analysis ───────────────────────────────────────
    motion_exprs = [e for e in expressions if is_motion_expression(e["sentence"])]
    print(f"Motion expressions: {len(motion_exprs)}/{len(expressions)}")

    per_expr_results = []

    for expr in motion_exprs:
        sentence = expr["sentence"]
        label_map = expr["label"]
        gt_track_ids = set()
        for fid_str, tids in label_map.items():
            gt_track_ids.update(tids)

        lang_emb = encoder.encode(sentence)

        # Determine frames where GT tracks are active
        gt_frames = set(int(f) for f in label_map.keys())

        # Compute motion vectors for GT tracks
        gt_cosines = []
        for tid in gt_track_ids:
            if tid not in all_track_centroids:
                continue
            centroids = all_track_centroids[tid]
            relevant_frames = sorted(f for f in centroids if f in gt_frames)
            if len(relevant_frames) < 2:
                continue
            track_subset = {f: centroids[f] for f in relevant_frames}
            frame_vecs = compute_motion_vector_for_track(
                track_subset, relevant_frames, frame_dir, orb_engine, frame_shape,
            )
            if not frame_vecs:
                continue
            vecs = np.array([v for _, v in frame_vecs])
            with torch.no_grad():
                m_emb, l_emb = model.encode(
                    torch.tensor(vecs, device=device),
                    lang_emb.to(device),
                )
                cos = (m_emb @ l_emb.t()).flatten().cpu().numpy()
            gt_cosines.extend(cos.tolist())

        # Compute motion vectors for non-GT tracks in the same frame range
        non_gt_cosines = []
        non_gt_tids = [t for t in all_track_centroids if t not in gt_track_ids]
        for tid in non_gt_tids:
            centroids = all_track_centroids[tid]
            relevant_frames = sorted(f for f in centroids if f in gt_frames)
            if len(relevant_frames) < 2:
                continue
            track_subset = {f: centroids[f] for f in relevant_frames}
            frame_vecs = compute_motion_vector_for_track(
                track_subset, relevant_frames, frame_dir, orb_engine, frame_shape,
            )
            if not frame_vecs:
                continue
            vecs = np.array([v for _, v in frame_vecs])
            with torch.no_grad():
                m_emb, l_emb = model.encode(
                    torch.tensor(vecs, device=device),
                    lang_emb.to(device),
                )
                cos = (m_emb @ l_emb.t()).flatten().cpu().numpy()
            non_gt_cosines.extend(cos.tolist())

        if gt_cosines and non_gt_cosines:
            gt_arr = np.array(gt_cosines)
            nongt_arr = np.array(non_gt_cosines)
            separation = gt_arr.mean() - nongt_arr.mean()

            # Simple AUC via Mann-Whitney U
            from scipy.stats import mannwhitneyu
            try:
                U, p = mannwhitneyu(gt_arr, nongt_arr, alternative="greater")
                auc = U / (len(gt_arr) * len(nongt_arr))
            except ValueError:
                auc = 0.5

            per_expr_results.append({
                "sentence": sentence,
                "n_gt": len(gt_cosines),
                "n_nongt": len(non_gt_cosines),
                "gt_mean": gt_arr.mean(),
                "gt_std": gt_arr.std(),
                "nongt_mean": nongt_arr.mean(),
                "nongt_std": nongt_arr.std(),
                "separation": separation,
                "auc": auc,
            })

    # ── Report ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("LAYER 3: GT vs NON-GT COSINE SIMILARITY DISTRIBUTIONS")
    print("=" * 70)

    if not per_expr_results:
        print("  No results — check data paths.")
        return

    print(f"\n  {'Expression':<45} {'GT mean':>8} {'NonGT':>8} {'Sep':>7} {'AUC':>6}")
    print("  " + "-" * 78)
    for r in sorted(per_expr_results, key=lambda x: -x["auc"]):
        name = r["sentence"][:44]
        print(f"  {name:<45} {r['gt_mean']:>8.4f} {r['nongt_mean']:>8.4f} "
              f"{r['separation']:>+7.4f} {r['auc']:>6.3f}")

    # Aggregate
    all_sep = [r["separation"] for r in per_expr_results]
    all_auc = [r["auc"] for r in per_expr_results]
    print(f"\n  Aggregate ({len(per_expr_results)} expressions):")
    print(f"    Mean separation: {np.mean(all_sep):+.4f} (std={np.std(all_sep):.4f})")
    print(f"    Mean AUC:        {np.mean(all_auc):.3f} (std={np.std(all_auc):.3f})")

    # Interpretation
    print()
    mean_auc = np.mean(all_auc)
    if mean_auc < 0.55:
        print("  ⚠ Near-chance AUC — cosine similarity has no discriminative power.")
    elif mean_auc < 0.70:
        print("  ⚠ Weak discrimination — signal exists but is noisy.")
    else:
        print("  ✓ Reasonable discrimination on GT data.")

    mean_sep = np.mean(all_sep)
    if mean_sep < 0.02:
        print("  ⚠ Negligible mean separation — GT and non-GT are indistinguishable.")
    elif mean_sep < 0.0:
        print("  ⚠ INVERTED: non-GT tracks score higher than GT tracks!")

    # ── Save results ──────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_path = os.path.join(RESULTS_DIR, f"layer3_gt_cosine_{args.seq}.npz")
    np.savez(save_path, results=per_expr_results)
    print(f"\n  Saved: {save_path}")

    # ── Visualization ─────────────────────────────────────────────────
    sorted_results = sorted(per_expr_results, key=lambda x: -x["auc"])
    exprs = [r["sentence"][:30] for r in sorted_results]
    y = np.arange(len(exprs))

    fig, axes = plt.subplots(1, 2, figsize=(14, max(5, len(exprs) * 0.35)))
    fig.suptitle(f"Layer 3: GT vs Non-GT Cosine (seq {args.seq}, GT centroids)",
                 fontsize=12, fontweight="bold")

    h = 0.35
    axes[0].barh(y - h/2, [r["gt_mean"] for r in sorted_results], h,
                 label="GT match", color="#27ae60", alpha=0.8)
    axes[0].barh(y + h/2, [r["nongt_mean"] for r in sorted_results], h,
                 label="Non-GT", color="#e74c3c", alpha=0.8)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(exprs, fontsize=7)
    axes[0].set_xlabel("Mean Cosine Similarity")
    axes[0].set_title("GT vs Non-GT Cosine")
    axes[0].legend(fontsize=8)
    axes[0].axvline(x=0, color="gray", linestyle="-", alpha=0.3)
    axes[0].invert_yaxis()

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
    plot_path = os.path.join(RESULTS_DIR, f"layer3_gt_cosine_{args.seq}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"  Plot saved: {plot_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
