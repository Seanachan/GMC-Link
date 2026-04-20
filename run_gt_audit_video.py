#!/usr/bin/env python
"""Render GT-annotated audit videos for Refer-KITTI V1 motion expressions."""

from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from gmc_link.dataset import is_motion_expression

BASE = Path("/home/seanachan/GMC-Link")
DATA_ROOT = Path("/home/seanachan/data/Dataset/refer-kitti")
IMG_ROOT = DATA_ROOT / "KITTI" / "training" / "image_02"
EXPR_ROOT = DATA_ROOT / "expression"
GT_TEMPLATE = BASE / "Refer-KITTI" / "gt_template"
NEURALSORT_ROOT = BASE / "NeuralSORT"

OUT_DIR = BASE / "diagnostics" / "results" / "label_audit"

SEQS = ["0005", "0011"]
FPS = 5
COLOR_GT = (0, 220, 0)
COLOR_CONTEXT = (160, 160, 160)
COLOR_TEXT = (255, 255, 255)
COLOR_TEXT_BG = (30, 30, 30)


def load_expression(path: Path) -> dict:
    """Parse expression JSON into sentence and per-frame GT track IDs."""
    with open(path) as f:
        d = json.load(f)
    gt_by_frame = {int(fid): [int(t) for t in tids] for fid, tids in d["label"].items()}
    return {"sentence": d["sentence"], "gt_by_frame": gt_by_frame}


def load_gt_template_mot(seq: str, slug: str) -> dict:
    path = GT_TEMPLATE / seq / slug / "gt.txt"
    out: dict[int, list[tuple[int, float, float, float, float]]] = defaultdict(list)
    if not path.is_file():
        return out
    with open(path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            fid = int(float(parts[0]))
            tid = int(float(parts[1]))
            x, y, w, h = (float(p) for p in parts[2:6])
            out[fid].append((tid, x, y, w, h))
    return out


def load_neuralsort_context(seq: str) -> dict:
    out: dict[int, list[tuple[int, float, float, float, float]]] = defaultdict(list)
    id_offset = 0
    for sub in ("car", "pedestrian"):
        path = NEURALSORT_ROOT / seq / sub / "predict.txt"
        if not path.is_file():
            continue
        try:
            arr = np.loadtxt(path, delimiter=",")
        except Exception:
            continue
        if arr.ndim != 2 or len(arr) == 0:
            continue
        max_id_here = int(arr[:, 1].max())
        for row in arr:
            fid = int(row[0])
            tid = int(row[1]) + id_offset
            x, y, w, h = float(row[2]), float(row[3]), float(row[4]), float(row[5])
            out[fid].append((tid, x, y, w, h))
        id_offset += max_id_here + 1
    return out


def draw_box(img: np.ndarray, x: float, y: float, w: float, h: float,
             color: tuple, label: str | None = None, thickness: int = 2) -> None:
    x1, y1 = int(x), int(y)
    x2, y2 = int(x + w), int(y + h)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if label:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


def draw_text_banner(img: np.ndarray, text: str, pos: tuple,
                     font_scale: float = 0.6) -> None:
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    x, y = pos
    cv2.rectangle(img, (x - 4, y - th - 6), (x + tw + 4, y + 4),
                  COLOR_TEXT_BG, -1)
    cv2.putText(img, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, COLOR_TEXT, 1)


def render_expression_video(seq: str, slug: str, expr_data: dict) -> Path | None:
    sentence = expr_data["sentence"]
    gt_by_frame = expr_data["gt_by_frame"]
    gt_boxes = load_gt_template_mot(seq, slug)
    context = load_neuralsort_context(seq)

    frames = sorted(gt_by_frame.keys())
    if not frames:
        return None

    img_dir = IMG_ROOT / seq
    first_img = img_dir / f"{frames[0]:06d}.png"
    if not first_img.is_file():
        print(f"  SKIP {seq}/{slug}: missing {first_img}")
        return None
    first = cv2.imread(str(first_img))
    H, W = first.shape[:2]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{seq}_{slug}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, FPS, (W, H))

    total = len(frames)
    frames_written = 0
    for i, fid in enumerate(frames):
        img_path = img_dir / f"{fid:06d}.png"
        if not img_path.is_file():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # Context tracks (gray, thin)
        for _tid, x, y, w, h in context.get(fid, []):
            draw_box(img, x, y, w, h, COLOR_CONTEXT, thickness=1)

        # GT tracks for this expression (green, thick, id-labelled)
        for tid, x, y, w, h in gt_boxes.get(fid, []):
            draw_box(img, x, y, w, h, COLOR_GT, label=f"id {tid}", thickness=2)

        draw_text_banner(img, sentence[:120], (10, 30))
        draw_text_banner(img, f"frame {fid}  ({i+1}/{total})", (max(W - 260, 10), 30))
        draw_text_banner(img, f"seq {seq}", (10, H - 15))

        writer.write(img)
        frames_written += 1

    writer.release()
    if frames_written == 0:
        out_path.unlink(missing_ok=True)
        return None
    return out_path


def write_review_csv(clips: list[tuple[str, str, str]]) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "review.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seq", "expr_slug", "sentence", "verdict", "note"])
        for seq, slug, sentence in clips:
            w.writerow([seq, slug, sentence, "", ""])
    return csv_path


def main() -> None:
    clips: list[tuple[str, str, str]] = []
    for seq in SEQS:
        expr_dir = EXPR_ROOT / seq
        if not expr_dir.is_dir():
            print(f"SKIP seq {seq}: {expr_dir} missing")
            continue
        jsons = sorted(expr_dir.glob("*.json"))
        n_motion = 0
        for expr_json in jsons:
            data = load_expression(expr_json)
            if not is_motion_expression(data["sentence"]):
                continue
            n_motion += 1
            slug = expr_json.stem
            print(f"[{seq}] {slug}: {data['sentence']}")
            out = render_expression_video(seq, slug, data)
            if out is not None:
                clips.append((seq, slug, data["sentence"]))
        print(f"  seq {seq}: {n_motion} motion expressions, "
              f"{len([c for c in clips if c[0]==seq])} rendered")

    csv_path = write_review_csv(clips)
    print(f"\nDone. {len(clips)} clips in {OUT_DIR}")
    print(f"Review template: {csv_path}")


if __name__ == "__main__":
    main()
