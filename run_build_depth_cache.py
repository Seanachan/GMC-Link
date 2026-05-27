"""Build per-track Z time-series cache via Depth Anything V2.

Sample metric Z at bbox center (5x5 patch median) for every track in every
frame of a sequence. Writes JSON: {track_id_str: {frame_id_str: z_meters}}.

Track sources:
  - ikun: NeuralSORT/{seq}/{car,pedestrian}/predict.txt (merged like run_build_gmc_cache)
  - flexhook_v1, flexhook_v2_raw: same NeuralSORT output for now (smoke smoke);
    arch arg only switches output filename to keep evals separable.

Usage:
    python run_build_depth_cache.py --arch ikun --seq 0011
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from gmc_link.dataset import _extract_all_track_centroids
from gmc_link.demo_inference import load_neuralsort_tracks
from gmc_link.depth_cache import save_depth_cache
from gmc_link.depth_extractor import DepthExtractor

FRAME_DIR = "/home/seanachan/data/Dataset/refer-kitti/KITTI/training/image_02"
DATA_ROOT = "refer-kitti"
TRACK_DIRS = {
    "ikun": "NeuralSORT",
    "fh_v1": "NeuralSORT",
    "flexhook_v1": "NeuralSORT",
    "fh_v2": "/home/seanachan/FlexHook/FlexHook/tracker_outputs/Temp-NeuralSORT-kitti2",
    "flexhook_v2_raw": "/home/seanachan/FlexHook/FlexHook/tracker_outputs/Temp-NeuralSORT-kitti2",
}


def merged_ns(arch: str, seq: str) -> dict:
    track_dir = TRACK_DIRS[arch]
    car = load_neuralsort_tracks(os.path.join(track_dir, seq, "car", "predict.txt"))
    ped = load_neuralsort_tracks(os.path.join(track_dir, seq, "pedestrian", "predict.txt"))
    max_car = 0
    for _, dets in car.items():
        for oid, *_ in dets:
            max_car = max(max_car, oid)
    ns = defaultdict(list)
    for fid, dets in car.items():
        ns[fid].extend(dets)
    for fid, dets in ped.items():
        ns[fid].extend([(oid + max_car, x, y, w, h) for oid, x, y, w, h in dets])
    return ns


def patch_z(depth: np.ndarray, cx: int, cy: int, half: int = 2) -> float:
    H, W = depth.shape
    cx = int(np.clip(cx, half, W - 1 - half))
    cy = int(np.clip(cy, half, H - 1 - half))
    patch = depth[cy - half:cy + half + 1, cx - half:cx + half + 1]
    return float(np.median(patch))


def gt_per_frame(seq: str, frame_shape) -> dict:
    """{frame_id_1based: [(tid, cx_px, cy_px), ...]} from KITTI labels_with_ids."""
    centroids = _extract_all_track_centroids(DATA_ROOT, seq, frame_shape=frame_shape)
    per_frame: dict = defaultdict(list)
    for tid, fmap in centroids.items():
        for fid, (cx, cy, _w, _h) in fmap.items():
            per_frame[fid].append((tid, cx, cy))
    return per_frame


def build(arch: str, seq: str, out_path: str) -> None:
    if os.path.exists(out_path):
        print(f"[depth] cache exists → {out_path}, skip")
        return

    extractor = DepthExtractor(device="cuda")
    seq_frame_dir = os.path.join(FRAME_DIR, seq)
    frame_files = sorted(f for f in os.listdir(seq_frame_dir) if f.endswith((".png", ".jpg")))
    total = len(frame_files)

    if arch == "gt":
        sample_bgr = cv2.imread(os.path.join(seq_frame_dir, frame_files[0]))
        frame_shape = (sample_bgr.shape[0], sample_bgr.shape[1])
        per_frame = gt_per_frame(seq, frame_shape)
        avg_tracks = sum(len(v) for v in per_frame.values()) / max(1, total)
        print(f"[depth] arch=gt seq={seq} frames={total} tracks/frame≈{avg_tracks:.1f}")

        table: dict[str, dict[str, float]] = defaultdict(dict)
        for f0 in tqdm(range(total), desc=f"depth-gt-{seq}"):
            f1 = f0 + 1
            dets = per_frame.get(f1, [])
            if not dets:
                continue
            bgr = cv2.imread(os.path.join(seq_frame_dir, frame_files[f0]))
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            depth = extractor.extract(rgb)
            for tid, cx, cy in dets:
                z = patch_z(depth, int(round(cx)), int(round(cy)))
                table[str(tid)][str(f1)] = z
    else:
        ns = merged_ns(arch, seq)
        avg_tracks = sum(len(v) for v in ns.values()) / max(1, total)
        print(f"[depth] arch={arch} seq={seq} frames={total} tracks/frame≈{avg_tracks:.1f}")

        table = defaultdict(dict)
        for f0 in tqdm(range(total), desc=f"depth-{arch}-{seq}"):
            f1 = f0 + 1
            dets = ns.get(f1, [])
            if not dets:
                continue
            bgr = cv2.imread(os.path.join(seq_frame_dir, frame_files[f0]))
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            depth = extractor.extract(rgb)
            for oid, x, y, w, h in dets:
                cx = x + w / 2.0
                cy = y + h / 2.0
                z = patch_z(depth, int(round(cx)), int(round(cy)))
                table[str(oid)][str(f1)] = z

    save_depth_cache(table, out_path)
    n_tracks = len(table)
    n_pts = sum(len(v) for v in table.values())
    print(f"[depth] wrote {out_path}  tracks={n_tracks}  samples={n_pts}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=["ikun", "fh_v1", "fh_v2", "flexhook_v1", "flexhook_v2_raw", "gt"])
    ap.add_argument("--seq", required=True, nargs="+")
    ap.add_argument("--out-dir", default="gmc_link/depth_cache")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for s in args.seq:
        out = os.path.join(args.out_dir, f"z_track_{args.arch}_{s}.json")
        build(args.arch, s, out)
