"""Build per-track Z cache from KITTI Velodyne LiDAR (Path B depth source).

Drop-in twin of run_build_depth_cache.py (DAv2 monocular). Same track sources,
same 1-based frame keys (f1 = f0+1), same output format
{track_id_str: {frame_str: z_meters}} -> so the 17D feature path downstream is
UNCHANGED. Only the depth VALUE source differs: median camera-Z of Velodyne
points inside each bbox (vs DAv2 patch median).

Edge cases (per Path B Phase-0 spec):
  - bbox with < min_pts LiDAR points (small/distant) -> OMITTED from cache (hole),
    counted in the diag summary (coverage differs from DAv2 -> single-variable
    caveat to surface to user after depth-half sanity).
  - bbox spanning multiple Z layers -> median used; per-(track,frame) std + n_points
    written to a *_diag.json sidecar for reliability inspection.

Usage (DO NOT run until velodyne extracted):
    python run_build_depth_cache_lidar.py --arch gt --seq 0005 0011 0013
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from gmc_link.dataset import _extract_all_track_centroids
from gmc_link.depth_cache import save_depth_cache
import gmc_link.kitti_tracking_gt as K
# reuse the DAv2 builder's track plumbing so sources stay identical
from run_build_depth_cache import FRAME_DIR, DATA_ROOT, merged_ns

VELO_DIR = os.path.join(K.TRACKING_ROOT, "velodyne")


def gt_boxes_per_frame(seq: str, frame_shape) -> dict:
    """{frame_id_1based: [(tid, x1, y1, x2, y2), ...]} from KITTI labels_with_ids."""
    centroids = _extract_all_track_centroids(DATA_ROOT, seq, frame_shape=frame_shape)
    per_frame: dict = defaultdict(list)
    for tid, fmap in centroids.items():
        for fid, (cx, cy, w, h) in fmap.items():
            per_frame[fid].append((tid, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    return per_frame


def ns_boxes_per_frame(arch: str, seq: str) -> dict:
    """{frame_id_1based: [(oid, x1,y1,x2,y2)]} from NeuralSORT (xywh -> xyxy)."""
    ns = merged_ns(arch, seq)
    out: dict = defaultdict(list)
    for fid, dets in ns.items():
        for oid, x, y, w, h in dets:
            out[fid].append((oid, x, y, x + w, y + h))
    return out


def build(arch: str, seq: str, out_path: str, min_pts: int) -> None:
    if os.path.exists(out_path):
        print(f"[lidar] cache exists -> {out_path}, skip")
        return
    velo_seq = os.path.join(VELO_DIR, seq)
    if not os.path.isdir(velo_seq):
        raise FileNotFoundError(
            f"velodyne not found: {velo_seq} -- extract data_tracking_velodyne.zip first")

    import cv2
    seq_frame_dir = os.path.join(FRAME_DIR, seq)
    frame_files = sorted(f for f in os.listdir(seq_frame_dir) if f.endswith((".png", ".jpg")))
    total = len(frame_files)
    sample = cv2.imread(os.path.join(seq_frame_dir, frame_files[0]))
    frame_shape = (sample.shape[0], sample.shape[1])

    if arch == "gt":
        per_frame = gt_boxes_per_frame(seq, frame_shape)
    else:
        per_frame = ns_boxes_per_frame(arch, seq)

    calib = K.load_calib(seq)
    table: dict[str, dict[str, float]] = defaultdict(dict)
    diag: dict[str, dict[str, list]] = defaultdict(dict)
    n_box = n_hole = 0

    for f0 in tqdm(range(total), desc=f"lidar-{arch}-{seq}"):
        f1 = f0 + 1
        dets = per_frame.get(f1, [])
        if not dets:
            continue
        bin_path = os.path.join(velo_seq, f"{f0:06d}.bin")
        if not os.path.exists(bin_path):
            continue
        pts = K.load_velodyne(seq, f0)[:, :3]
        uv, z_cam = K.project_velo_to_image(pts, calib)
        for tid, x1, y1, x2, y2 in dets:
            n_box += 1
            res = K.bbox_lidar_depth(uv, z_cam, (x1, y1, x2, y2), min_pts=min_pts)
            if res is None:
                n_hole += 1
                continue
            z, std, npts = res
            table[str(tid)][str(f1)] = z
            diag[str(tid)][str(f1)] = [round(std, 3), npts]

    save_depth_cache(table, out_path)
    diag_path = out_path.replace(".json", "_diag.json")
    with open(diag_path, "w") as f:
        json.dump({"per_track": diag,
                   "summary": {"n_box": n_box, "n_hole": n_hole,
                               "hole_rate": n_hole / max(1, n_box),
                               "min_pts": min_pts}}, f)
    print(f"[lidar] wrote {out_path}  tracks={len(table)} "
          f"samples={sum(len(v) for v in table.values())} "
          f"hole_rate={n_hole/max(1,n_box):.3f} (boxes with <{min_pts} pts)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True,
                    choices=["ikun", "fh_v1", "fh_v2", "flexhook_v1", "flexhook_v2_raw", "gt"])
    ap.add_argument("--seq", required=True, nargs="+")
    ap.add_argument("--out-dir", default="gmc_link/depth_cache")
    ap.add_argument("--min-pts", type=int, default=1,
                    help="min LiDAR points in bbox; below => hole (omitted)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    for s in args.seq:
        # filename mirrors DAv2 (z_track_{arch}_{seq}) but tagged _lidar to keep both
        out = os.path.join(args.out_dir, f"z_track_lidar_{args.arch}_{s}.json")
        build(args.arch, s, out, args.min_pts)
