"""
show_similarity.py

Evaluate GMC-Link motion-language similarity using the official GMCLinkManager
API in the correct way.

Main fixes:
1. Initialize GMCLinkManager only once.
2. For each frame, update internal motion/GMC state only once.
3. Reuse the same frame state to evaluate multiple expressions with update_state=False.
4. Report Avg All / Avg GT / Avg Non-GT / Margin / Top-1.
5. Add optional debug prints for GT-track overlap.

This version matches manager.py semantics:
- update_state=True  -> update homography/history/buffers
- update_state=False -> reuse current frame state for additional language prompts
"""

import os
import sys
import json
from typing import Dict, List, Tuple, Any

import cv2
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from gmc_link.manager import GMCLinkManager
from gmc_link.text_utils import TextEncoder


# =========================
# Config
# =========================
TRACK_PATH = "/home/jkaiwang/Desktop/GMC-Link/NeuralSORT/0011/car/predict.txt"
IMAGE_DIR = "/home/jkaiwang/Desktop/GMC-Link/refer-kitti/KITTI/training/image_02/0011"
GT_DIR = "/home/jkaiwang/Desktop/GMC-Link/refer-kitti/KITTI/training/image_02/0011"
WEIGHTS_PATH = "gmc_link_weights.pth"

EXPRESSIONS = [
    "moving-cars",
    "cars-in-black",
    "parking-cars",
]

FRAME_STRIDE = 5
DEBUG_PRINT_IDS = False


# =========================
# Utilities
# =========================
def safe_mean(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def safe_std(values: List[float]) -> float:
    return float(np.std(values)) if values else 0.0


def load_neuralsort_tracks(track_path: str) -> Dict[int, List[Tuple[int, float, float, float, float]]]:
    """
    Load NeuralSORT predict.txt into:
        {frame_id: [(obj_id, x, y, w, h), ...]}

    File format:
        frame,id,x,y,w,h,conf,-1,-1,-1
    Input frame index is 1-based, converted to 0-based here.
    """
    if not os.path.exists(track_path):
        raise FileNotFoundError(f"Tracking file not found: {track_path}")

    tracks_by_frame: Dict[int, List[Tuple[int, float, float, float, float]]] = {}

    data = np.loadtxt(track_path, delimiter=",")
    if data.ndim == 1:
        data = np.expand_dims(data, axis=0)

    for row in data:
        frame_id = int(row[0]) - 1
        obj_id = int(row[1])
        x, y, w, h = float(row[2]), float(row[3]), float(row[4]), float(row[5])

        tracks_by_frame.setdefault(frame_id, []).append((obj_id, x, y, w, h))

    return tracks_by_frame


def load_gt_labels(expressions: List[str], gt_dir: str) -> Dict[str, Dict[str, List[int]]]:
    """
    Load GT json for each expression.

    Expected structure:
        {
            expr: {
                "0": [1, 2],
                "5": [4],
                ...
            }
        }
    """
    gt_labels_dict: Dict[str, Dict[str, List[int]]] = {}

    for expr in expressions:
        gt_path = os.path.join(gt_dir, f"{expr}.json")
        if not os.path.exists(gt_path):
            raise FileNotFoundError(f"GT file not found: {gt_path}")

        with open(gt_path, "r") as f:
            data = json.load(f)

        if "label" not in data:
            raise KeyError(f"'label' key missing in {gt_path}")

        gt_labels_dict[expr] = data["label"]

    return gt_labels_dict


class DummyTrack:
    """
    Minimal track object compatible with GMCLinkManager.process_frame().
    Needs:
    - id
    - centroid
    - bbox
    """
    def __init__(self, obj_id: int, x: float, y: float, w: float, h: float):
        self.id = obj_id
        self.bbox = [x, y, x + w, y + h]
        self.centroid = np.array([x + w / 2.0, y + h / 2.0], dtype=np.float64)


# =========================
# Evaluation
# =========================
def evaluate_similarity():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    encoder = TextEncoder(device=device)
    manager = GMCLinkManager(weights_path=WEIGHTS_PATH, device=device)

    tracks_by_frame = load_neuralsort_tracks(TRACK_PATH)
    gt_labels_dict = load_gt_labels(EXPRESSIONS, GT_DIR)

    # Pre-encode all expressions once
    expr_embeddings: Dict[str, torch.Tensor] = {}
    for expr in EXPRESSIONS:
        expr_embeddings[expr] = encoder.encode(expr)

    all_cosines = {expr: [] for expr in EXPRESSIONS}
    gt_cosines = {expr: [] for expr in EXPRESSIONS}
    nongt_cosines = {expr: [] for expr in EXPRESSIONS}

    top1_correct = {expr: 0 for expr in EXPRESSIONS}
    top1_total = {expr: 0 for expr in EXPRESSIONS}

    frame_ids = sorted(tracks_by_frame.keys())[::FRAME_STRIDE]
    total_frames = len(frame_ids)

    for idx, frame_id in enumerate(frame_ids):
        print(f"Processing frame {frame_id} ({idx + 1}/{total_frames})...")

        image_path = os.path.join(IMAGE_DIR, f"{frame_id:06d}.png")
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"  Skipping frame {frame_id}: image not found")
            continue

        raw_tracks = tracks_by_frame.get(frame_id, [])
        if not raw_tracks:
            print(f"  Skipping frame {frame_id}: no tracks")
            continue

        active_tracks = [DummyTrack(obj_id, x, y, w, h) for obj_id, x, y, w, h in raw_tracks]
        track_ids_in_frame = {t.id for t in active_tracks}

        # IMPORTANT:
        # Use update_state=True only ONCE for the first expression of this frame.
        # Then use update_state=False for the rest, so we don't corrupt history/state.
        for expr_idx, expr in enumerate(EXPRESSIONS):
            lang_emb = expr_embeddings[expr]
            update_state = (expr_idx == 0)

            scores_dict, velocities_dict, cosine_dict = manager.process_frame(
                frame=frame,
                active_tracks=active_tracks,
                language_embedding=lang_emb,
                detections=None,
                update_state=update_state,
            )

            gt_ids = set(gt_labels_dict[expr].get(str(frame_id), []))
            valid_gt_ids = gt_ids.intersection(track_ids_in_frame)

            if DEBUG_PRINT_IDS:
                print(f"  [{expr}] track_ids={sorted(track_ids_in_frame)}")
                print(f"  [{expr}] gt_ids={sorted(gt_ids)}")
                print(f"  [{expr}] valid_gt_ids={sorted(valid_gt_ids)}")

            if not cosine_dict:
                continue

            # collect cosine stats
            for tid, cos in cosine_dict.items():
                all_cosines[expr].append(float(cos))
                if tid in valid_gt_ids:
                    gt_cosines[expr].append(float(cos))
                else:
                    nongt_cosines[expr].append(float(cos))

            # Top-1 eval only if this frame has valid GT
            if valid_gt_ids:
                best_tid = max(cosine_dict, key=cosine_dict.get)
                top1_total[expr] += 1
                if best_tid in valid_gt_ids:
                    top1_correct[expr] += 1

    # =========================
    # Summary table
    # =========================
    results = []
    for expr in EXPRESSIONS:
        avg_all = safe_mean(all_cosines[expr])
        avg_gt = safe_mean(gt_cosines[expr])
        avg_nongt = safe_mean(nongt_cosines[expr])
        margin = avg_gt - avg_nongt
        top1 = (top1_correct[expr] / top1_total[expr]) if top1_total[expr] > 0 else 0.0

        results.append({
            "expr": expr,
            "avg_all": avg_all,
            "avg_gt": avg_gt,
            "avg_nongt": avg_nongt,
            "margin": margin,
            "top1": top1,
            "n_all": len(all_cosines[expr]),
            "n_gt": len(gt_cosines[expr]),
            "n_nongt": len(nongt_cosines[expr]),
            "std_gt": safe_std(gt_cosines[expr]),
            "std_nongt": safe_std(nongt_cosines[expr]),
        })

    print("\n" + "=" * 95)
    print("Average Cosine Similarity Table")
    print("=" * 95)
    print(
        f"{'Expression':<18} {'Avg All':<10} {'Avg GT':<10} "
        f"{'Avg Non-GT':<12} {'Margin':<10} {'Top1':<10}"
    )
    print("-" * 95)

    for row in results:
        print(
            f"{row['expr']:<18} "
            f"{row['avg_all']:<10.4f} "
            f"{row['avg_gt']:<10.4f} "
            f"{row['avg_nongt']:<12.4f} "
            f"{row['margin']:<10.4f} "
            f"{row['top1']:<10.4f}"
        )

    print("=" * 95)

    print("\nDetailed Statistics")
    print("=" * 95)
    for row in results:
        print(f"[{row['expr']}]")
        print(f"  Total cosine count : {row['n_all']}")
        print(f"  GT cosine count    : {row['n_gt']}")
        print(f"  Non-GT cosine count: {row['n_nongt']}")
        print(f"  Avg GT             : {row['avg_gt']:.4f}")
        print(f"  Std GT             : {row['std_gt']:.4f}")
        print(f"  Avg Non-GT         : {row['avg_nongt']:.4f}")
        print(f"  Std Non-GT         : {row['std_nongt']:.4f}")
        if top1_total[row["expr"]] > 0:
            print(f"  Top-1 Accuracy     : {top1_correct[row['expr']]}/{top1_total[row['expr']]} = {row['top1']:.4f}")
        else:
            print("  Top-1 Accuracy     : N/A")
        print("-" * 95)


def main():
    evaluate_similarity()


if __name__ == "__main__":
    main()