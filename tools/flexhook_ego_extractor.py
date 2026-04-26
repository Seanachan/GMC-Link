#!/usr/bin/env python
"""
Exp 38 — Extract ego-compensated 13D velocity cache for FlexHook.

Two sources:
  --source pred  (default): Temp-NeuralSORT predict.txt tracker output
                             Keyed by tracker-id (ped shift applied at cache
                             load time).
  --source gt              : Refer-KITTI labels.json GT tracks
                             Keyed by labels.json obj_id (no ped shift — all
                             rows tagged class=b'c' so EgoCacheLookup's shift
                             logic is a no-op).

Output layout (identical for both sources):
    {cache_root}/{seq}/ego_13d.npz with arrays:
      frames   : (N,) int32   (1-indexed MOT frame)
      track_ids: (N,) int32
      classes  : (N,) |S1     (b'c'=car, b'p'=pedestrian)
      vec13d   : (N, 13) float32

Usage — test seqs (tracker-keyed):
    python tools/flexhook_ego_extractor.py \
        --source pred \
        --tracks /home/seanachan/FlexHook-ego/tracker_outputs/Temp-NeuralSORT-kitti1 \
        --frames /home/seanachan/FlexHook-ego/datasets/refer-kitti/KITTI/training/image_02 \
        --out /home/seanachan/GMC-Link/diagnostics/results/exp38/cache/ego_speed \
        --seqs 0005 0011 0013

Usage — train seqs (GT-keyed, Exp 38-A):
    python tools/flexhook_ego_extractor.py \
        --source gt \
        --labels /home/seanachan/FlexHook-ego/datasets/refer-kitti/labels.json \
        --frames /home/seanachan/FlexHook-ego/datasets/refer-kitti/KITTI/training/image_02 \
        --out /home/seanachan/GMC-Link/diagnostics/results/exp38/cache/ego_speed \
        --seqs 0001 0002 0003 0004 0006 0007 0008 0009 0010 0012 0014 0015 0016 0018 0020
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from gmc_link.manager import GMCLinkManager


CLASS_TAG = {"car": b"c", "pedestrian": b"p"}

# Same (H, W) table FlexHook uses (data/utils.py RESOLUTION). Required to
# denormalize labels.json bboxes, which are [x_left_norm, y_top_norm, w_norm,
# h_norm]. Only V1 KITTI seqs listed — no DanceTrack.
KITTI_RESOLUTION = {
    "0000": (375, 1242), "0001": (375, 1242), "0002": (375, 1242),
    "0003": (375, 1242), "0004": (375, 1242), "0005": (375, 1242),
    "0006": (375, 1242), "0007": (375, 1242), "0008": (375, 1242),
    "0009": (375, 1242), "0010": (375, 1242), "0011": (375, 1242),
    "0012": (375, 1242), "0013": (375, 1242), "0014": (370, 1224),
    "0015": (370, 1224), "0016": (370, 1224), "0018": (374, 1238),
    "0019": (374, 1238), "0020": (376, 1241),
}

# Coord conversion: GMCLinkManager emits residual velocity in VELOCITY_SCALE=100
# units (v_pixel / img_dim * 100). FlexHook's `grid_sample` operates in normgrid
# coords where unit step = img_dim/2 pixels. Conversion:
#   normgrid = pixel_disp / (img_dim/2) = (residual / 100) * 2 = residual * 0.02
RESIDUAL_TO_NORMGRID = 0.02


def residual_to_normgrid(residual: np.ndarray) -> np.ndarray:
    """Convert GMC-Link residual velocity (VELOCITY_SCALE=100 units) to
    FlexHook `speed_normgrid` units used by `grid_sample`."""
    return residual.astype(np.float32) * RESIDUAL_TO_NORMGRID


def load_tracks(tracks_root: Path, seq: str):
    """
    Parse Temp-NeuralSORT predict.txt files for both classes.
    Returns dict: frame_idx (1-based) -> list of (class, track_id, x1, y1, w, h).
    """
    per_frame: dict[int, list] = {}
    for cls in ("car", "pedestrian"):
        path = tracks_root / seq / cls / "predict.txt"
        if not path.exists() or path.stat().st_size == 0:
            continue
        for line in path.read_text().splitlines():
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            fr = int(parts[0])
            tid = int(parts[1])
            x1, y1, w, h = (float(parts[i]) for i in range(2, 6))
            per_frame.setdefault(fr, []).append((cls, tid, x1, y1, w, h))
    return per_frame


def load_gt_tracks(labels_path: Path, seq: str):
    """
    Parse Refer-KITTI labels.json for one seq, yielding per-frame GT tracks.
    labels.json structure:
        seq -> obj_id_str -> frame_id_str -> {"bbox": [xn, yn, wn, hn], ...}
    frame_id_str is absolute 0-indexed (matches PNG filename). We emit
    1-indexed fr to match predict.txt convention downstream.

    All rows tagged cls="car" so EgoCacheLookup's ped-shift becomes a no-op
    — the training dataloader queries `f'{video}_{int(obj_id)}'` without
    class awareness.
    """
    if seq not in KITTI_RESOLUTION:
        raise KeyError(f"no resolution for seq {seq}")
    H, W = KITTI_RESOLUTION[seq]
    with labels_path.open() as f:
        all_labels = json.load(f)
    if seq not in all_labels:
        return {}
    per_frame: dict[int, list] = {}
    for obj_id_str, obj_frames in all_labels[seq].items():
        tid = int(obj_id_str)
        for frame_id_str, info in obj_frames.items():
            frame_1idx = int(frame_id_str) + 1
            xn, yn, wn, hn = info["bbox"]
            x1 = xn * W
            y1 = yn * H
            w = wn * W
            h = hn * H
            per_frame.setdefault(frame_1idx, []).append(("car", tid, x1, y1, w, h))
    return per_frame


def build_track_objs(rows):
    """Wrap predictions into objects GMCLinkManager consumes."""
    tracks = []
    detections = []
    for cls, tid, x1, y1, w, h in rows:
        cx = x1 + w / 2.0
        cy = y1 + h / 2.0
        x2 = x1 + w
        y2 = y1 + h
        # Namespace acts as cheap struct; need class-unique id to avoid
        # collision between a car and pedestrian sharing id=1.
        uid = (0 if cls == "car" else 1) * 100000 + tid
        tracks.append(
            SimpleNamespace(
                id=uid,
                centroid=(cx, cy),
                bbox=(x1, y1, x2, y2),
            )
        )
        detections.append((x1, y1, x2, y2))
    return tracks, np.array(detections, dtype=np.float32) if detections else None


def process_sequence(
    seq: str,
    frames_root: Path,
    out_root: Path,
    tracks_root: Path | None = None,
    labels_path: Path | None = None,
    source: str = "pred",
) -> None:
    if source == "pred":
        assert tracks_root is not None, "--tracks required for --source pred"
        tracks_by_frame = load_tracks(tracks_root, seq)
    elif source == "gt":
        assert labels_path is not None, "--labels required for --source gt"
        tracks_by_frame = load_gt_tracks(labels_path, seq)
    else:
        raise ValueError(f"unknown source: {source}")
    if not tracks_by_frame:
        print(f"[{seq}] no tracks, skipping")
        return
    max_frame = max(tracks_by_frame)
    frame_dir = frames_root / seq
    n_imgs = len(list(frame_dir.glob("*.png")))
    print(f"[{seq}] frames={n_imgs} tracks span 1..{max_frame} "
          f"({sum(len(v) for v in tracks_by_frame.values())} detections)")

    mgr = GMCLinkManager(weights_path=None, device="cpu", frame_gap=10)
    # Dummy language embedding (unused — we only read velocities_dict).
    dummy_lang = torch.zeros(1, 384, dtype=torch.float32)

    out_frames: list[int] = []
    out_tids: list[int] = []
    out_classes: list[bytes] = []
    out_vec: list[np.ndarray] = []

    # Iterate every frame in image dir order (mgr depends on sequential prev_frame).
    # Frame IDs: MOT 1-indexed; image file `000000.png` corresponds to frame 1.
    for fidx in range(1, n_imgs + 1):
        img_path = frame_dir / f"{fidx - 1:06d}.png"
        if not img_path.exists():
            continue
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        rows = tracks_by_frame.get(fidx, [])
        tracks, detections = build_track_objs(rows)
        if not tracks:
            # Still advance ego state with empty track list so homography chain stays intact.
            mgr.process_frame(frame, [], dummy_lang, detections=None, update_state=True)
            continue

        _, velocities, _ = mgr.process_frame(
            frame, tracks, dummy_lang,
            detections=detections,
            update_state=True,
        )
        for cls, tid, *_rest in rows:
            uid = (0 if cls == "car" else 1) * 100000 + tid
            v = velocities.get(uid)
            if v is None:
                continue
            out_frames.append(fidx)
            out_tids.append(tid)
            out_classes.append(CLASS_TAG[cls])
            out_vec.append(v.astype(np.float32))

    out_dir = out_root / seq
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ego_13d.npz"
    np.savez_compressed(
        out_path,
        frames=np.array(out_frames, dtype=np.int32),
        track_ids=np.array(out_tids, dtype=np.int32),
        classes=np.array(out_classes, dtype="|S1"),
        vec13d=np.stack(out_vec, axis=0) if out_vec else np.zeros((0, 13), np.float32),
    )
    print(f"[{seq}] wrote {out_path} (N={len(out_frames)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["pred", "gt"], default="pred")
    ap.add_argument("--tracks", type=Path, default=None,
                    help="Temp-NeuralSORT tracker root (required for --source pred)")
    ap.add_argument("--labels", type=Path, default=None,
                    help="Refer-KITTI labels.json (required for --source gt)")
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seqs", nargs="+", default=["0005", "0011", "0013"])
    args = ap.parse_args()

    if args.source == "pred" and args.tracks is None:
        ap.error("--source pred requires --tracks")
    if args.source == "gt" and args.labels is None:
        ap.error("--source gt requires --labels")

    args.out.mkdir(parents=True, exist_ok=True)
    for seq in args.seqs:
        process_sequence(
            seq,
            frames_root=args.frames,
            out_root=args.out,
            tracks_root=args.tracks,
            labels_path=args.labels,
            source=args.source,
        )


if __name__ == "__main__":
    main()
