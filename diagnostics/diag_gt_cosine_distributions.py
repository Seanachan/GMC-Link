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
import torch.nn.functional as F
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


def precompute_homographies(frame_dir, all_frame_ids, orb_engine):
    """
    Precompute all needed homographies for the sequence in one pass.
    Returns dict: (past_fid, curr_fid) -> (H, bg_residual)
    """
    sorted_fids = sorted(all_frame_ids)
    needed_pairs = set()
    for i, curr_fid in enumerate(sorted_fids):
        for gap in FRAME_GAPS:
            if i >= gap:
                needed_pairs.add((sorted_fids[i - gap], curr_fid))

    # Preload all needed frames (each frame loaded once)
    needed_frame_ids = set()
    for past, curr in needed_pairs:
        needed_frame_ids.add(past)
        needed_frame_ids.add(curr)

    print(f"  Preloading {len(needed_frame_ids)} frames...")
    frame_images = {}
    for fid in sorted(needed_frame_ids):
        img = cv2.imread(os.path.join(frame_dir, f"{fid:06d}.png"))
        frame_images[fid] = img

    print(f"  Computing {len(needed_pairs)} homographies...")
    cache = {}
    for idx, (past_fid, curr_fid) in enumerate(sorted(needed_pairs)):
        img_past = frame_images.get(past_fid)
        img_curr = frame_images.get(curr_fid)
        if img_past is not None and img_curr is not None:
            H, bg_res = orb_engine.estimate_homography(img_past, img_curr)
            cache[(past_fid, curr_fid)] = (H, bg_res)
        else:
            cache[(past_fid, curr_fid)] = (None, np.zeros(2))
        if (idx + 1) % 100 == 0:
            print(f"    {idx + 1}/{len(needed_pairs)} done")

    print(f"  Homography precompute complete.")
    return cache


def compute_motion_vectors_for_all_tracks(
    all_track_centroids, active_frame_ids, homography_cache, frame_shape,
    extra_features=None,
):
    """
    Compute motion vectors for ALL tracks at ALL active frames in one pass.
    Returns dict: {track_id: [(frame_id, motion_vector), ...]}
    """
    from gmc_link.dataset import compute_per_track_extras, compute_relational_extras

    h, w = frame_shape
    sorted_active = sorted(active_frame_ids)
    fid_to_idx = {f: i for i, f in enumerate(sorted_active)}
    all_results = {}

    per_track_feats = [f for f in (extra_features or [])
                       if f in ("speed_m", "heading_m", "accel", "ego_motion",
                                "accel_multiscale", "heading_sincos",
                                "ego_velocity_concat", "omf_stats")]
    relational_feats = [f for f in (extra_features or [])
                        if f in ("neighbor_mean_vel", "velocity_rank", "heading_diff",
                                 "nn_dist", "track_density")]

    # First pass: build base vectors + per-track extras, collect mid-scale data
    # {frame_id: {track_id: (mid_dx, mid_dy, cx_n, cy_n)}}
    frame_track_data = {}

    needs_accel_multiscale = "accel_multiscale" in per_track_feats

    for tid, centroids in all_track_centroids.items():
        results = []
        # Track's frames within the active set, sorted
        track_frames = sorted(f for f in centroids if f in active_frame_ids)
        if len(track_frames) < 2:
            continue

        track_scale_vels = {} if needs_accel_multiscale else None

        for i in range(len(track_frames)):
            curr_fid = track_frames[i]
            scale_velocities = []
            any_valid = False
            best_bg_residual = np.zeros(2, dtype=np.float32)
            ego_dx_m, ego_dy_m = 0.0, 0.0

            for gap_idx, gap in enumerate(FRAME_GAPS):
                if i >= gap:
                    past_fid = track_frames[i - gap]
                    cache_key = (past_fid, curr_fid)
                    H_entry = homography_cache.get(cache_key, (None, np.zeros(2)))
                    H, bg_res = H_entry

                    cx1, cy1, _, _ = centroids[past_fid]
                    cx2, cy2, _, _ = centroids[curr_fid]

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
                        ego_dx_m, ego_dy_m = ego_dx, ego_dy
                else:
                    scale_velocities.append((0.0, 0.0))

            if not any_valid:
                continue

            mid_gap = FRAME_GAPS[1]
            if i >= mid_gap:
                _, _, bw1, bh1 = centroids[track_frames[i - mid_gap]]
                _, _, bw2, bh2 = centroids[curr_fid]
                dw = (bw2 - bw1) / w * VELOCITY_SCALE
                dh = (bh2 - bh1) / h * VELOCITY_SCALE
            else:
                dw, dh = 0.0, 0.0

            cx, cy, bw, bh_val = centroids[curr_fid]
            mid_dx, mid_dy = scale_velocities[1]
            obj_speed = np.sqrt(mid_dx ** 2 + mid_dy ** 2)
            bg_mag = np.sqrt(
                (best_bg_residual[0] / w * VELOCITY_SCALE) ** 2
                + (best_bg_residual[1] / h * VELOCITY_SCALE) ** 2
            )
            snr = obj_speed / (bg_mag + 1e-6)

            base_vec = [
                scale_velocities[0][0], scale_velocities[0][1],
                scale_velocities[1][0], scale_velocities[1][1],
                scale_velocities[2][0], scale_velocities[2][1],
                dw, dh, cx / w, cy / h, bw / w, bh_val / h, snr,
            ]

            # Per-track extras: compute true-temporal accel if requested
            accel_per_scale = None
            if needs_accel_multiscale:
                accel_per_scale = []
                for gap_idx, gap in enumerate(FRAME_GAPS):
                    past_i = i - gap
                    if past_i < 0:
                        accel_per_scale.append((0.0, 0.0))
                    else:
                        past_fid = track_frames[past_i]
                        past_vels = track_scale_vels.get(past_fid)
                        if past_vels is None:
                            accel_per_scale.append((0.0, 0.0))
                        else:
                            now_dx, now_dy = scale_velocities[gap_idx]
                            past_dx, past_dy = past_vels[gap_idx]
                            accel_per_scale.append((
                                (now_dx - past_dx) / gap,
                                (now_dy - past_dy) / gap,
                            ))
                track_scale_vels[curr_fid] = list(scale_velocities)

            if per_track_feats:
                base_vec.extend(compute_per_track_extras(
                    per_track_feats, scale_velocities,
                    ego_dx_m=ego_dx_m, ego_dy_m=ego_dy_m,
                    accel_per_scale=accel_per_scale,
                ))

            # Store mid-scale data for relational post-pass
            if relational_feats:
                if curr_fid not in frame_track_data:
                    frame_track_data[curr_fid] = {}
                frame_track_data[curr_fid][tid] = (mid_dx, mid_dy, cx / w, cy / h)

            vec = np.array(base_vec, dtype=np.float32)
            results.append((curr_fid, vec))

        if results:
            all_results[tid] = results

    # Second pass: append relational extras (F5-F9) using all-track neighbor context
    if relational_feats and frame_track_data:
        frame_diag = np.sqrt(1.0 + 1.0)
        for tid, frame_vecs in all_results.items():
            for vec_idx, (fid, vec) in enumerate(frame_vecs):
                frame_neighbors = frame_track_data.get(fid, {})
                my_data = frame_neighbors.get(tid)
                if my_data is None:
                    # No mid-scale data — pad with zeros
                    n_rel = sum({"neighbor_mean_vel": 2, "velocity_rank": 1,
                                 "heading_diff": 1, "nn_dist": 1, "track_density": 1}[f]
                                for f in relational_feats)
                    padded = np.concatenate([vec, np.zeros(n_rel, dtype=np.float32)])
                    all_results[tid][vec_idx] = (fid, padded)
                    continue

                my_dx, my_dy, my_cx, my_cy = my_data
                neighbor_vels = []
                neighbor_centroids_list = []
                for other_tid, other_data in frame_neighbors.items():
                    if other_tid != tid:
                        neighbor_vels.append((other_data[0], other_data[1]))
                        neighbor_centroids_list.append((other_data[2], other_data[3]))

                rel_extras = compute_relational_extras(
                    relational_feats, my_dx, my_dy, my_cx, my_cy,
                    neighbor_vels, neighbor_centroids_list, frame_diag,
                )
                extended = np.concatenate([vec, np.array(rel_extras, dtype=np.float32)])
                all_results[tid][vec_idx] = (fid, extended)

    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=WEIGHTS_PATH)
    parser.add_argument("--seq", default=SEQUENCE)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Seq: {args.seq} | Weights: {args.weights}")

    # ── Load model ────────────────────────────────────────────────────
    checkpoint = torch.load(args.weights, map_location=device, weights_only=False)
    extra_features = None
    architecture = "mlp"
    seq_len = 10
    text_encoder_name = "all-MiniLM-L6-v2"
    lang_dim = 384
    if isinstance(checkpoint, dict):
        motion_dim = checkpoint.get("motion_dim", 13)
        extra_features = checkpoint.get("extra_features", None)
        architecture = checkpoint.get("architecture", "mlp")
        seq_len = checkpoint.get("seq_len", 10) or 10
        text_encoder_name = checkpoint.get("text_encoder", "all-MiniLM-L6-v2")
        lang_dim = checkpoint.get("lang_dim", 384)
    else:
        motion_dim = 13

    print(f"  Architecture: {architecture}" + (f" (seq_len={seq_len})" if architecture == "temporal_transformer" else ""))
    if text_encoder_name != "all-MiniLM-L6-v2":
        print(f"  Text encoder: {text_encoder_name} (lang_dim={lang_dim})")

    model = MotionLanguageAligner(
        motion_dim=motion_dim, lang_dim=lang_dim, embed_dim=256,
        architecture=architecture, seq_len=seq_len,
    ).to(device)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    if extra_features:
        print(f"  Extra features: {extra_features} ({motion_dim}D)")
        from gmc_link.dataset import compute_per_track_extras

    # ── Load data ─────────────────────────────────────────────────────
    encoder = TextEncoder(model_name=text_encoder_name, device=str(device))
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

    # ── Precompute homographies (one-time, shared across all expressions) ──
    all_frame_ids = sorted(labels_by_frame.keys())
    print(f"Sequence has {len(all_frame_ids)} labeled frames, {len(all_track_centroids)} tracks")
    homography_cache = precompute_homographies(
        frame_dir, all_frame_ids, orb_engine
    )

    # ── Compute motion vectors for ALL tracks at once ────────────────
    print("Computing motion vectors for all tracks...")
    track_motion_vecs = compute_motion_vectors_for_all_tracks(
        all_track_centroids, set(all_frame_ids), homography_cache, frame_shape,
        extra_features=extra_features,
    )
    print(f"  Tracks with motion vectors: {len(track_motion_vecs)}")

    # ── Precompute motion embeddings for all tracks ──────────────────
    if architecture == "temporal_transformer":
        print(f"Encoding motion sequences (transformer, T={seq_len})...")
        # Group per-track vectors into sliding windows, then batch-encode
        all_seqs = []
        all_masks = []
        track_emb_slices = {}

        for tid, frame_vecs in track_motion_vecs.items():
            vectors = [v for _, v in frame_vecs]
            n_frames = len(vectors)
            if n_frames == 0:
                track_emb_slices[tid] = (len(all_seqs), len(all_seqs))
                continue
            dim = vectors[0].shape[0]
            track_start = len(all_seqs)

            for win_end in range(1, n_frames + 1):
                win_start = max(0, win_end - seq_len)
                window = vectors[win_start:win_end]
                n_valid = len(window)

                if n_valid < seq_len:
                    pad = [np.zeros(dim, dtype=np.float32)] * (seq_len - n_valid)
                    window = pad + list(window)

                all_seqs.append(np.stack(window))

                mask = np.zeros(seq_len + 1, dtype=bool)
                n_padded = seq_len - n_valid
                if n_padded > 0:
                    mask[1:n_padded + 1] = True
                all_masks.append(mask)

            track_emb_slices[tid] = (track_start, len(all_seqs))

        all_motion_embs = None
        if all_seqs:
            seq_tensor = torch.tensor(np.array(all_seqs), dtype=torch.float32).to(device)
            mask_tensor = torch.tensor(np.array(all_masks), dtype=torch.bool).to(device)
            with torch.no_grad():
                all_motion_embs = F.normalize(
                    model.motion_projector(seq_tensor, mask_tensor), p=2, dim=-1
                )
            print(f"  Encoded {len(all_seqs)} motion sequences → {all_motion_embs.shape}")

    else:
        print("Encoding all motion vectors on GPU...")
        all_vecs_list = []
        vec_to_track = []
        for tid, frame_vecs in track_motion_vecs.items():
            for fid, vec in frame_vecs:
                all_vecs_list.append(vec)
                vec_to_track.append((tid, fid))

        all_motion_embs = None
        if all_vecs_list:
            all_vecs_tensor = torch.tensor(np.array(all_vecs_list), dtype=torch.float32).to(device)
            with torch.no_grad():
                all_motion_embs = F.normalize(model.motion_projector(all_vecs_tensor), p=2, dim=-1)
            print(f"  Encoded {len(all_vecs_list)} motion vectors → {all_motion_embs.shape}")

        track_emb_slices = {}
        idx = 0
        for tid, frame_vecs in track_motion_vecs.items():
            n = len(frame_vecs)
            track_emb_slices[tid] = (idx, idx + n)
            idx += n

    # ── Per-expression analysis ───────────────────────────────────────
    from scipy.stats import mannwhitneyu

    motion_exprs = [e for e in expressions if is_motion_expression(e["sentence"])]
    print(f"Motion expressions: {len(motion_exprs)}/{len(expressions)}")

    per_expr_results = []
    gt_cosines_by_expr = []
    nongt_cosines_by_expr = []

    for ei, expr in enumerate(motion_exprs):
        sentence = expr["sentence"]
        label_map = expr["label"]
        gt_track_ids = set()
        for fid_str, tids in label_map.items():
            gt_track_ids.update(tids)

        # Encode language once
        lang_emb = encoder.encode(sentence)
        with torch.no_grad():
            lang_proj = F.normalize(model.lang_projector(lang_emb.to(device)), p=2, dim=-1)  # (1, D)

        # Collect cosines from precomputed embeddings
        gt_cosines = []
        non_gt_cosines = []

        for tid, (start, end) in track_emb_slices.items():
            if all_motion_embs is None:
                continue
            embs = all_motion_embs[start:end]  # (N, D)
            cos = (embs @ lang_proj.t()).flatten().cpu().numpy()

            if tid in gt_track_ids:
                gt_cosines.extend(cos.tolist())
            else:
                non_gt_cosines.extend(cos.tolist())

        if gt_cosines and non_gt_cosines:
            gt_arr = np.array(gt_cosines)
            nongt_arr = np.array(non_gt_cosines)
            separation = gt_arr.mean() - nongt_arr.mean()

            try:
                U, p = mannwhitneyu(gt_arr, nongt_arr, alternative="greater")
                auc = U / (len(gt_arr) * len(nongt_arr))
            except ValueError:
                auc = 0.5

            per_expr_results.append({
                "sentence": sentence,
                "n_gt": len(gt_cosines),
                "n_nongt": len(non_gt_cosines),
                "gt_mean": float(gt_arr.mean()),
                "gt_std": float(gt_arr.std()),
                "nongt_mean": float(nongt_arr.mean()),
                "nongt_std": float(nongt_arr.std()),
                "separation": float(separation),
                "auc": float(auc),
            })
            gt_cosines_by_expr.append(gt_arr.astype(np.float32))
            nongt_cosines_by_expr.append(nongt_arr.astype(np.float32))

        if (ei + 1) % 10 == 0:
            print(f"  Processed {ei + 1}/{len(motion_exprs)} expressions")

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
    np.savez(
        save_path,
        results=per_expr_results,
        gt_cosines_by_expr=np.array(gt_cosines_by_expr, dtype=object),
        nongt_cosines_by_expr=np.array(nongt_cosines_by_expr, dtype=object),
    )
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
