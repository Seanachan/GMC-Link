"""
show_similarity.py

This script evaluates the GMC-Link model by computing cosine similarity
between object motion features and language expressions.

It loads tracking results and video frames, extracts motion features
(using GMC and velocity modeling), encodes text queries, and computes
motion-language similarity for each track.

The results are aggregated across frames to measure how well the model
aligns motion with language (e.g., moving vs. parked objects).
"""


import torch 
import numpy as np
import cv2
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from gmc_link.manager import GMCLinkManager
from gmc_link.text_utils import TextEncoder

def load_neuralsort_tracks(track_path: str) -> dict:
    """
    Load NeuralSORT predict.txt → {frame_id: [(obj_id, x, y, w, h), ...]}.

    NeuralSORT format: frame,id,x,y,w,h,conf,-1,-1,-1  (1-indexed frames)
    """
    tracks_by_frame = {}
    data = np.loadtxt(track_path, delimiter=",")
    for row in data:
        frame_id = int(row[0]) - 1  # Convert to 0-indexed
        obj_id = int(row[1])
        x, y, w, h = row[2], row[3], row[4], row[5]
        if frame_id not in tracks_by_frame:
            tracks_by_frame[frame_id] = []
        tracks_by_frame[frame_id].append((obj_id, x, y, w, h))
    return tracks_by_frame

class DummyTrack:
    """Track stub compatible with GMCLinkManager.process_frame()."""

    def __init__(self, obj_id: int, x: float, y: float, w: float, h: float):
        self.id = obj_id
        self.bbox = [x, y, x + w, y + h]
        self.centroid = np.array([x + w / 2.0, y + h / 2.0], dtype=np.float64)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # initialize the text encoder
    encoder = TextEncoder(device=device)

    # Load tracks
    track_path = "/home/jkaiwang/Desktop/GMC-Link/NeuralSORT/0011/car/predict.txt"
    tracks_by_frame = load_neuralsort_tracks(track_path)

    expressions = ["moving-cars", "cars-in-black", "parking-cars"]
    all_cosines = {expr: [] for expr in expressions}
    gt_cosines = {expr: [] for expr in expressions}
    nongt_cosines = {expr: [] for expr in expressions}

    # Load GT for each expression
    gt_labels_dict = {}
    for expr in expressions:
        gt_path = f"/home/jkaiwang/Desktop/GMC-Link/refer-kitti/KITTI/training/image_02/0011/{expr}.json"
        with open(gt_path, 'r') as f:
            gt_data = json.load(f)
        gt_labels_dict[expr] = gt_data['label']

    # Process every 5th frame
    frame_ids = sorted(tracks_by_frame.keys())[::5]
    total_frames = len(frame_ids)
    for i, frame_id in enumerate(frame_ids):
        print(f"Processing frame {frame_id} ({i+1}/{total_frames})...")
        image_path = f"/home/jkaiwang/Desktop/GMC-Link/refer-kitti/KITTI/training/image_02/0011/{frame_id:06d}.png"
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"  Skipping: image not found")
            continue

        active_tracks = [DummyTrack(obj_id, x, y, w, h) for obj_id, x, y, w, h in tracks_by_frame[frame_id]]
        print(f"  {len(active_tracks)} tracks")

        for expr in expressions:
            manager = GMCLinkManager(weights_path="gmc_link_weights.pth", device=device)
            lang_emb = encoder.encode(expr)
            scores_dict, velocities_dict, cosine_dict = manager.process_frame(frame, active_tracks, lang_emb)

            gt_ids = set(gt_labels_dict[expr].get(str(frame_id), []))
            for tid, cos in cosine_dict.items():
                all_cosines[expr].append(cos)
                if tid in gt_ids:
                    gt_cosines[expr].append(cos)
                else:
                    nongt_cosines[expr].append(cos)

    # Compute averages
    results = []
    for expr in expressions:
        cos_all = all_cosines[expr]
        cos_gt = gt_cosines[expr]
        cos_nongt = nongt_cosines[expr]
        avg_all = sum(cos_all) / len(cos_all) if cos_all else 0.0
        avg_gt = sum(cos_gt) / len(cos_gt) if cos_gt else 0.0
        avg_nongt = sum(cos_nongt) / len(cos_nongt) if cos_nongt else 0.0
        results.append((expr, avg_all, avg_gt, avg_nongt))
        print(f"Collected {len(cos_all)} total, {len(cos_gt)} GT, {len(cos_nongt)} non-GT for '{expr}'")

    # Print summary table
    print("\n" + "="*60)
    print("Average Cosine Similarity Table")
    print("="*60)
    print(f"{'Expression':<15} {'Avg All':<10} {'Avg GT':<10} {'Avg Non-GT':<12}")
    print("-"*47)
    for expr, avg_all, avg_gt, avg_nongt in results:
        print(f"{expr:<15} {avg_all:<10.4f} {avg_gt:<10.4f} {avg_nongt:<12.4f}")
    print("="*60)

if __name__ == "__main__":
    main()