"""Join GT, iKUN, GMC, tracker, detector into one per-row audit table.

ID-space note: GT track_ids and tracker (NeuralSORT) track_ids live in
DIFFERENT namespaces. We bridge them via IoU matching per frame, using
the iKUN cache's frame index (which equals the tracker's frame index by
construction — both 1-indexed continuous video frames).

For each GT positive row (frame, gt_track_id, gt_bbox):
  1. Find the tracker prediction at that frame with highest IoU to gt_bbox
     (threshold 0.5). If no match → tracker_track_id = None (FN_tracker
     or FN_detector depending on det_cache).
  2. Look up (frame, tracker_track_id) in iKUN/GMC caches.
  3. Compute fusion_gate via ship recipe.

We also emit "negative" rows for non-GT tracker predictions to score FP
and TN, but those are not needed for the FN-stage attribution headline.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

from .loaders import (
    load_gt, load_ikun_logits, load_gmc_scores,
    load_tracker_assoc, load_detector_hits,
    compute_fusion_gate,
)


def _iou_xywh(a: tuple, b: tuple) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def build_cell_table(repo_root: Path, seq: str, expr: str,
                     iou_thr: float = 0.5,
                     gmc_cache_tpl: str = "gmc_scores_v1_{seq}_depth_seed1_cache.json"
                     ) -> pd.DataFrame:
    gt   = load_gt(repo_root, seq, expr)
    ikun = load_ikun_logits(repo_root, seq, expr)
    gmc  = load_gmc_scores(repo_root, seq, expr, cache_tpl=gmc_cache_tpl)
    tr   = load_tracker_assoc(repo_root, seq, expr)
    det  = load_detector_hits(repo_root, seq, expr)

    ikun_set = set((int(f), int(t)) for f, t in zip(ikun["frame"], ikun["track_id"])) \
        if not ikun.empty else set()
    ikun_lookup = {(int(f), int(t)): float(l) for f, t, l in
                   zip(ikun["frame"], ikun["track_id"], ikun["ikun_logit"])} \
        if not ikun.empty else {}
    gmc_lookup = {(int(f), int(t)): float(s) for f, t, s in
                  zip(gmc["frame"], gmc["track_id"], gmc["aligner_gmc_score"])} \
        if not gmc.empty else {}
    tr_by_frame: dict[int, list] = {}
    for _, r in tr.iterrows():
        tr_by_frame.setdefault(int(r["frame"]), []).append(r)
    det_frames = set(int(f) for f in det["frame"]) if not det.empty else set()
    ikun_frames = set(int(f) for f in ikun["frame"]) if not ikun.empty else set()

    rows = []
    for _, g in gt.iterrows():
        frame = int(g["frame"])
        gt_box = (float(g["x"]), float(g["y"]), float(g["w"]), float(g["h"]))
        candidates = tr_by_frame.get(frame, [])
        best_iou, best_tid, best_assoc = 0.0, None, "lost"
        for c in candidates:
            iou = _iou_xywh(gt_box,
                            (float(c["x"]), float(c["y"]),
                             float(c["w"]), float(c["h"])))
            if iou > best_iou:
                best_iou, best_tid, best_assoc = iou, int(c["track_id"]), c["tracker_assoc"]
        matched = best_iou >= iou_thr
        det_hit = 1 if frame in det_frames else 0  # frame-level: no track id in dets
        ikun_log = ikun_lookup.get((frame, best_tid), float("nan")) if matched else float("nan")
        gmc_s    = gmc_lookup.get((frame, best_tid), float("nan")) if matched else float("nan")
        if matched and not np.isnan(ikun_log) and not np.isnan(gmc_s):
            fuse = compute_fusion_gate(ikun_log, gmc_s, expr)
            pred = int(fuse >= 0)
        else:
            fuse = float("nan")
            pred = 0
        rows.append({
            "seq": seq, "expr": expr,
            "frame": frame, "gt_track_id": int(g["track_id"]),
            "matched_tracker_id": best_tid if matched else None,
            "best_iou": best_iou,
            "gt_match": 1,
            "detector_hit": det_hit,
            "tracker_assoc": best_assoc if matched else "lost",
            "aligner_gmc_score": gmc_s,
            "ikun_logit": ikun_log,
            "fusion_gate": fuse,
            "pred_match": pred,
            "ikun_frame_in_cache": int(frame in ikun_frames),
        })
    return pd.DataFrame(rows)
