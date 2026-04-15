"""
HOTA evaluation for TempRMOT + GMC-Link fusion.

Strategy:
  - TempRMOT has many false positives (DetPr ~7.7%).
  - GMC-Link scores each predicted track's motion against the expression.
  - Tracks with low motion-language alignment are suppressed (FP filtering).

Efficiency:
  - Ego-motion (ORB homography chain) is pre-computed ONCE per sequence.
  - Then each expression reuses the pre-computed homographies.

Usage:
    python run_hota_temprmot.py                  # run full eval
    python run_hota_temprmot.py --threshold 0.45 # adjust GMC-Link suppression threshold
"""

import os
import sys
import json
import shutil
import subprocess
from collections import defaultdict, deque

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from gmc_link.alignment import MotionLanguageAligner
from gmc_link.core import ORBHomographyEngine
from gmc_link.text_utils import TextEncoder
from gmc_link.utils import normalize_velocity, MotionBuffer, ScoreBuffer, VELOCITY_SCALE, warp_points
from gmc_link.demo_inference import classify_expression

# ── Config ──────────────────────────────────────────────────────────────
TEMPRMOT_RESULTS = "/home/seanachan/TempRMOT/exps/default_rk/results_epoch50"
IMAGE_ROOT = "/home/seanachan/data/Dataset/refer-kitti-v2/KITTI/training/image_02"
EXPR_ROOT = "/home/seanachan/GMC-Link/refer-kitti/expression"  # v1 expressions

WEIGHTS_PATH = "gmc_link_weights.pth"
TRACKEVAL_DIR = "/home/seanachan/TrackEval"
TRACKEVAL_SCRIPT = os.path.join(TRACKEVAL_DIR, "scripts", "run_mot_challenge.py")

OUTPUT_ROOT = "hota_eval/temprmot_gmclink"
FRAME_GAP = 10  # max gap for multi-scale temporal features


# ── Pre-compute ego-motion ──────────────────────────────────────────────

def precompute_ego_motion(seq: str):
    """
    Pre-compute frame-to-frame homographies and background residuals for an entire sequence.
    Returns (homographies, bg_residuals) lists (len = num_frames).
    H[t] maps frame[t-1]→frame[t], H[0] = identity. bg_residuals[0] = zeros.
    """
    frame_dir = os.path.join(IMAGE_ROOT, seq)
    frame_files = sorted(f for f in os.listdir(frame_dir) if f.endswith((".png", ".jpg")))
    num_frames = len(frame_files)

    engine = ORBHomographyEngine(max_features=1500)
    homographies = [np.eye(3, dtype=np.float32)]  # H[0] = identity
    bg_residuals = [np.zeros(2, dtype=np.float32)]  # bg[0] = zeros

    prev_frame = cv2.imread(os.path.join(frame_dir, frame_files[0]))
    for i in range(1, num_frames):
        curr_frame = cv2.imread(os.path.join(frame_dir, frame_files[i]))
        H, bg_res = engine.estimate_homography(prev_frame, curr_frame, prev_bboxes=None)
        homographies.append(H)
        bg_residuals.append(bg_res)
        prev_frame = curr_frame

        if (i + 1) % 100 == 0:
            print(f"    ego-motion: {i+1}/{num_frames} frames")

    print(f"    ego-motion: {num_frames} frames done")
    return homographies, bg_residuals


def build_cumulative_homographies(frame_to_frame_H: list, frame_gap: int) -> list:
    """
    Build cumulative homography buffers at each timestep.
    Returns list of deques, one per frame, each containing cumulative H matrices
    mapping past frames to the current frame.
    """
    n = len(frame_to_frame_H)
    cum_buffers = []

    for t in range(n):
        buf = deque(maxlen=frame_gap + 1)
        # Build cumulative H for frames [max(0, t-frame_gap) .. t]
        # H_cum[t→t] = I
        buf.append(np.eye(3, dtype=np.float32))
        for k in range(1, min(t + 1, frame_gap + 1)):
            # H maps frame[t-k] → frame[t]
            # Compose: H_{t-k→t} = H_{t-1→t} @ H_{t-2→t-1} @ ... @ H_{t-k→t-k+1}
            H_cum = np.eye(3, dtype=np.float32)
            for j in range(k):
                H_cum = frame_to_frame_H[t - j] @ H_cum
            buf.appendleft(H_cum)
        cum_buffers.append(buf)

    return cum_buffers


def build_bg_residual_buffers(bg_residuals: list, frame_gap: int) -> list:
    """
    Build per-frame background residual buffers.
    Returns list of deques, one per frame, each containing bg_residuals
    from the window [max(0, t-frame_gap) .. t].
    """
    n = len(bg_residuals)
    buffers = []
    for t in range(n):
        buf = deque(maxlen=frame_gap + 1)
        start = max(0, t - frame_gap)
        for k in range(start, t + 1):
            buf.append(bg_residuals[k])
        buffers.append(buf)
    return buffers


# ── Load TempRMOT predictions ───────────────────────────────────────────

def load_temprmot_tracks(predict_path: str):
    """
    Load predict.txt → {frame_1idx: [(track_id, x, y, w, h, raw_line), ...]}.
    Format: frame,id,x,y,w,h,conf,class,vis (1-indexed frames)
    Preserves raw_line for lossless write-back.
    """
    tracks_by_frame = defaultdict(list)
    if not os.path.exists(predict_path) or os.path.getsize(predict_path) == 0:
        return tracks_by_frame
    with open(predict_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            frame_id = int(parts[0])
            obj_id = int(parts[1])
            x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            tracks_by_frame[frame_id].append((obj_id, x, y, w, h, line))
    return tracks_by_frame


# ── Score tracks for one expression ─────────────────────────────────────

def score_expression(
    pred_tracks,       # {frame_1idx: [(tid, x, y, w, h), ...]}
    cum_buffers,       # pre-computed cumulative homographies
    lang_emb,          # (1, D) language embedding
    aligner,           # MotionLanguageAligner
    device,
    num_frames,
    img_h, img_w,
    frame_gap=FRAME_GAP,
    bg_res_buffers=None,  # pre-computed background residual buffers
    temperature=1.0,   # learned temperature for scoring
):
    """
    Score each predicted track against the expression using motion-language alignment.
    Returns {frame_1idx: {track_id: gmc_score}}.
    """
    # Per-track state
    centroid_history = {}   # tid -> deque of (frame_0idx, centroid)
    wh_history = {}         # tid -> deque of (frame_0idx, [w, h])
    motion_buffer = MotionBuffer(alpha=0.3)
    score_buffer = ScoreBuffer(alpha=0.4)

    frame_shape = (img_h, img_w)
    frame_scores = {}

    MULTI_GAPS = [2, 5, 10]  # short, mid, long — must match training
    max_gap = max(MULTI_GAPS)
    mid_gap_idx = 1  # index of primary (mid) scale

    for frame_0idx in range(num_frames):
        frame_1idx = frame_0idx + 1
        if frame_1idx not in pred_tracks:
            continue

        detections = pred_tracks[frame_1idx]
        cum_H = cum_buffers[frame_0idx]

        track_ids = []
        spatial_motions = []

        for entry in detections:
            tid, x, y, w, h = entry[0], entry[1], entry[2], entry[3], entry[4]
            cx = x + w / 2.0
            cy = y + h / 2.0
            curr_centroid = np.array([cx, cy], dtype=np.float64)

            if tid not in centroid_history:
                centroid_history[tid] = deque(maxlen=max_gap + 1)
                wh_history[tid] = deque(maxlen=max_gap + 1)

            centroid_history[tid].append((frame_0idx, curr_centroid))
            wh_history[tid].append((frame_0idx, np.array([w, h], dtype=np.float64)))

            ch = list(centroid_history[tid])
            wh = list(wh_history[tid])

            if len(ch) > 1:
                newest_f0, newest_c = ch[-1]

                # Multi-scale: residual velocity = raw - ego
                residual_velocities = []
                for gap in MULTI_GAPS:
                    if len(ch) > gap:
                        old_f0, old_c = ch[-(gap + 1)]
                        # Raw velocity
                        raw_v = normalize_velocity(newest_c - old_c, frame_shape)
                        # Per-object ego displacement
                        actual_gap = newest_f0 - old_f0
                        if actual_gap > 0 and actual_gap <= len(cum_H) - 1:
                            idx = len(cum_H) - 1 - actual_gap
                            H_old = cum_H[idx] if idx >= 0 else np.eye(3, dtype=np.float32)
                        else:
                            H_old = np.eye(3, dtype=np.float32)
                        warped_old = warp_points(np.array([old_c]), H_old)[0]
                        ego_v = normalize_velocity(warped_old - old_c, frame_shape)
                        # Residual = raw - ego
                        residual_velocities.append(raw_v - ego_v)
                    else:
                        residual_velocities.append(np.zeros(2, dtype=np.float32))

                # dw, dh from mid-scale
                mid_gap = MULTI_GAPS[mid_gap_idx]
                if len(wh) > mid_gap:
                    raw_dw_dh = wh[-1][1] - wh[-(mid_gap + 1)][1]
                else:
                    raw_dw_dh = wh[-1][1] - wh[0][1]
                dw_raw = raw_dw_dh[0] / float(img_w) * VELOCITY_SCALE
                dh_raw = raw_dw_dh[1] / float(img_h) * VELOCITY_SCALE

                full_v = np.array([
                    residual_velocities[0][0], residual_velocities[0][1],
                    residual_velocities[1][0], residual_velocities[1][1],
                    residual_velocities[2][0], residual_velocities[2][1],
                    dw_raw, dh_raw,
                ], dtype=np.float32)
                smoothed_v = motion_buffer.smooth(tid, full_v)
                dx_s, dy_s = smoothed_v[0], smoothed_v[1]
                dx_m, dy_m = smoothed_v[2], smoothed_v[3]
                dx_l, dy_l = smoothed_v[4], smoothed_v[5]
                dw_s, dh_s = smoothed_v[6], smoothed_v[7]
            else:
                dx_s, dy_s = 0.0, 0.0
                dx_m, dy_m = 0.0, 0.0
                dx_l, dy_l = 0.0, 0.0
                dw_s, dh_s = 0.0, 0.0

            # 13D: residual velocity + spatial
            w_n = w / float(img_w)
            h_n = h / float(img_h)
            cx_n = cx / float(img_w)
            cy_n = cy / float(img_h)

            # SNR from mid-scale residual speed
            obj_speed = np.sqrt(dx_m ** 2 + dy_m ** 2)
            if bg_res_buffers is not None and len(bg_res_buffers[frame_0idx]) > 0:
                bg_stack = np.array(list(bg_res_buffers[frame_0idx]))
                bg_max = np.max(np.abs(bg_stack), axis=0)
                bg_magnitude = np.sqrt(
                    (bg_max[0] / float(img_w) * VELOCITY_SCALE) ** 2
                    + (bg_max[1] / float(img_h) * VELOCITY_SCALE) ** 2
                )
            else:
                bg_magnitude = 0.0
            snr = obj_speed / (bg_magnitude + 1e-6)

            spatial_motion = np.array([
                dx_s, dy_s, dx_m, dy_m, dx_l, dy_l,
                dw_s, dh_s, cx_n, cy_n, w_n, h_n, snr,
            ], dtype=np.float32)

            track_ids.append(tid)
            spatial_motions.append(spatial_motion)

        if not spatial_motions:
            continue

        # Score with aligner
        motion_tensor = torch.tensor(np.array(spatial_motions), dtype=torch.float32).to(device)
        with torch.no_grad():
            motion_emb, l_emb = aligner.encode(motion_tensor, lang_emb.to(device))
            cosine_sim = torch.matmul(motion_emb, l_emb.t()).flatten()
            margin = 0.05  # calibrated from GT/non-GT cosine distributions
            raw_scores = torch.sigmoid((cosine_sim - margin) / temperature).cpu().numpy()

        scores = {}
        for i, tid in enumerate(track_ids):
            scores[tid] = score_buffer.smooth(tid, float(raw_scores[i]))
        frame_scores[frame_1idx] = scores

        # Clean dead tracks
        active = set(track_ids)
        motion_buffer.clear_dead_tracks(track_ids)
        score_buffer.clear_dead_tracks(track_ids)
        dead = set(centroid_history.keys()) - active
        for d in dead:
            del centroid_history[d]
            if d in wh_history:
                del wh_history[d]

    return frame_scores


# ── Filter predictions ──────────────────────────────────────────────────

def filter_predictions(pred_tracks, frame_scores, expr_type, threshold):
    """
    Suppress TempRMOT FPs using GMC-Link scores.
    For motion/stationary expressions, remove tracks below threshold.
    For appearance expressions, keep all (GMC-Link motion is less informative).
    """
    filtered = defaultdict(list)
    kept, suppressed = 0, 0

    for frame_1idx, dets in pred_tracks.items():
        scores = frame_scores.get(frame_1idx, {})
        for entry in dets:
            tid = entry[0]
            raw_line = entry[5]  # original line for lossless write-back
            gmc_score = scores.get(tid, 0.5)  # default neutral

            if expr_type in ("motion", "stationary"):
                if gmc_score >= threshold:
                    filtered[frame_1idx].append(raw_line)
                    kept += 1
                else:
                    suppressed += 1
            else:
                # Appearance: keep all, GMC-Link less reliable for pure appearance
                filtered[frame_1idx].append(raw_line)
                kept += 1

    return filtered, kept, suppressed


def write_predict_txt(filtered_tracks, out_path):
    """Write filtered predictions preserving original line format."""
    lines = []
    for frame_1idx in sorted(filtered_tracks.keys()):
        for raw_line in filtered_tracks[frame_1idx]:
            lines.append(raw_line)
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
        if lines:
            f.write("\n")


# ── Main ────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.45,
                        help="GMC-Link score threshold for motion/stationary suppression")
    parser.add_argument("--sequences", nargs="+", default=None,
                        help="Sequences to evaluate (default: all in TempRMOT results)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load aligner
    aligner = MotionLanguageAligner(motion_dim=13, lang_dim=384, embed_dim=256).to(device)
    checkpoint = torch.load(WEIGHTS_PATH, map_location=device)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        aligner.load_state_dict(checkpoint["model"])
        learned_temperature = checkpoint.get("temperature", 1.0)
    else:
        aligner.load_state_dict(checkpoint)
        learned_temperature = 1.0
    aligner.eval()
    print(f"Loaded aligner weights from {WEIGHTS_PATH} (τ={learned_temperature:.4f})")

    # Load text encoder
    encoder = TextEncoder(device=device)

    # Determine sequences
    if args.sequences:
        sequences = args.sequences
    else:
        sequences = sorted(
            d for d in os.listdir(TEMPRMOT_RESULTS)
            if os.path.isdir(os.path.join(TEMPRMOT_RESULTS, d))
        )
    print(f"Sequences: {sequences}")

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    seqmap_entries = []
    total_kept, total_suppressed = 0, 0

    for seq in sequences:
        seq_results = os.path.join(TEMPRMOT_RESULTS, seq)
        expressions = sorted(
            d for d in os.listdir(seq_results)
            if os.path.isdir(os.path.join(seq_results, d))
        )
        print(f"\n{'='*70}")
        print(f"Sequence {seq}: {len(expressions)} expressions")
        print(f"{'='*70}")

        # Get image dimensions
        frame_dir = os.path.join(IMAGE_ROOT, seq)
        frame_files = sorted(f for f in os.listdir(frame_dir) if f.endswith((".png", ".jpg")))
        num_frames = len(frame_files)
        sample_img = cv2.imread(os.path.join(frame_dir, frame_files[0]))
        img_h, img_w = sample_img.shape[:2]
        print(f"  {num_frames} frames, {img_w}x{img_h}")

        # Pre-compute ego-motion
        print(f"  Pre-computing ego-motion...")
        frame_to_frame_H, bg_residuals = precompute_ego_motion(seq)
        cum_buffers = build_cumulative_homographies(frame_to_frame_H, FRAME_GAP)
        bg_res_buffers = build_bg_residual_buffers(bg_residuals, FRAME_GAP)
        print(f"  Ego-motion ready")

        # Process each expression
        out_seq_dir = os.path.join(OUTPUT_ROOT, seq)
        for expr_idx, expr_name in enumerate(expressions):
            expr_dir_src = os.path.join(seq_results, expr_name)
            expr_dir_dst = os.path.join(out_seq_dir, expr_name)
            os.makedirs(expr_dir_dst, exist_ok=True)

            # Copy GT
            gt_src = os.path.join(expr_dir_src, "gt.txt")
            gt_dst = os.path.join(expr_dir_dst, "gt.txt")
            if os.path.exists(gt_src):
                shutil.copy2(gt_src, gt_dst)
            else:
                continue

            # Load predictions
            pred_path = os.path.join(expr_dir_src, "predict.txt")
            pred_tracks = load_temprmot_tracks(pred_path)

            if not pred_tracks:
                # Empty predictions - just copy empty predict.txt
                open(os.path.join(expr_dir_dst, "predict.txt"), "w").close()
                seqmap_entries.append(f"{seq}+{expr_name}")
                continue

            # Get expression sentence
            expr_json_path = os.path.join(EXPR_ROOT, seq, f"{expr_name}.json")
            if os.path.exists(expr_json_path):
                with open(expr_json_path) as f:
                    expr_json = json.load(f)
                sentence = expr_json["sentence"]
            else:
                sentence = expr_name.replace("-", " ")

            expr_type = classify_expression(sentence)
            lang_emb = encoder.encode(sentence)

            # Score tracks
            frame_scores = score_expression(
                pred_tracks, cum_buffers, lang_emb, aligner, device,
                num_frames, img_h, img_w, FRAME_GAP,
                bg_res_buffers=bg_res_buffers,
                temperature=learned_temperature,
            )

            # Filter
            filtered, kept, suppressed = filter_predictions(
                pred_tracks, frame_scores, expr_type, args.threshold
            )
            total_kept += kept
            total_suppressed += suppressed

            # Write filtered predictions
            write_predict_txt(filtered, os.path.join(expr_dir_dst, "predict.txt"))
            seqmap_entries.append(f"{seq}+{expr_name}")

            if (expr_idx + 1) % 50 == 0:
                print(f"  [{expr_idx+1}/{len(expressions)}] expressions processed")

        print(f"  Sequence {seq} done: {len(expressions)} expressions")

    # Write seqmap
    seqmap_path = os.path.join(OUTPUT_ROOT, "seqmap.txt")
    with open(seqmap_path, "w") as f:
        f.write("\n".join(seqmap_entries) + "\n")
    print(f"\nSeqmap: {seqmap_path} ({len(seqmap_entries)} entries)")
    print(f"Total: kept={total_kept}, suppressed={total_suppressed}")
    if total_kept + total_suppressed > 0:
        print(f"Suppression rate: {total_suppressed/(total_kept+total_suppressed)*100:.1f}%")

    # Run TrackEval
    abs_out = os.path.abspath(OUTPUT_ROOT)
    cmd = [
        sys.executable, TRACKEVAL_SCRIPT,
        "--METRICS", "HOTA", "CLEAR", "Identity",
        "--SEQMAP_FILE", os.path.abspath(seqmap_path),
        "--SKIP_SPLIT_FOL", "True",
        "--GT_FOLDER", abs_out,
        "--TRACKERS_FOLDER", abs_out,
        "--GT_LOC_FORMAT", "{gt_folder}/{video_id}/{expression_id}/gt.txt",
        "--TRACKERS_TO_EVAL", abs_out,
        "--USE_PARALLEL", "False",
        "--PLOT_CURVES", "False",
        "--PRINT_CONFIG", "False",
    ]
    print(f"\nRunning TrackEval...")
    result = subprocess.run(cmd, capture_output=True, text=True,
                            cwd=os.path.dirname(TRACKEVAL_SCRIPT))
    print("\n" + "=" * 70)
    print("TrackEval Results [TempRMOT + GMC-Link]")
    print("=" * 70)
    if result.stdout:
        # Print just the summary lines
        lines = result.stdout.strip().split("\n")
        for line in lines:
            if "COMBINED" in line or "HOTA" in line.split()[0:1] or "CLEAR" in line.split()[0:1] or "Identity" in line.split()[0:1]:
                print(line)
            # Always print COMBINED
        # Also print last few lines for timing
        print("\n--- Full COMBINED metrics ---")
        for line in lines:
            if "COMBINED" in line:
                print(line)
    if result.returncode != 0:
        print("STDERR:", result.stderr[-3000:] if result.stderr else "")

    return result.returncode


if __name__ == "__main__":
    main()
