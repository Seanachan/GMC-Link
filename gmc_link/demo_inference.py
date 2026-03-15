"""
GMC-Link Ablation Demo: iKUN Baseline vs iKUN + GMC-Link
=========================================================
Loads pre-computed iKUN CLIP similarity scores and NeuralSORT tracking boxes,
then compares:
  (A) iKUN alone (vision-language CLIP logit > 0)
  (B) iKUN + GMC-Link (fused logit: ikun_score + gmc_logit * weight)

Both pipelines use identical detections and GT matching so the only variable
is the GMC kinematic module.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import math
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import cv2
import numpy as np
import torch

from gmc_link.manager import GMCLinkManager
from gmc_link.text_utils import TextEncoder
from gmc_link.dataset import load_labels_with_ids


# ── Helpers ──────────────────────────────────────────────────────────


def load_neuralsort_tracks(track_path: str) -> Dict[int, List]:
    """
    Load NeuralSORT predict.txt → {frame_id: [(obj_id, x, y, w, h), ...]}.

    NeuralSORT format: frame,id,x,y,w,h,conf,-1,-1,-1  (1-indexed frames)
    """
    tracks_by_frame: Dict[int, list] = defaultdict(list)
    data = np.loadtxt(track_path, delimiter=",")
    for row in data:
        frame_id = int(row[0])
        obj_id = int(row[1])
        x, y, w, h = row[2], row[3], row[4], row[5]
        tracks_by_frame[frame_id].append((obj_id, x, y, w, h))
    return tracks_by_frame


def load_ikun_scores(
    results_path: str, video: str, expression: str
) -> Dict[int, Dict[int, float]]:
    """
    Load iKUN CLIP scores → {frame_id(int): {obj_id(int): mean_score}}.

    results.json structure: video → obj_id → frame_id → expression → [logit]
    Frame and obj IDs in the JSON are string keys; we return ints.
    """
    with open(results_path, "r", encoding="utf-8") as f:
        all_results = json.load(f)

    video_dict = all_results.get(video, {})
    scores: Dict[int, Dict[int, float]] = defaultdict(dict)

    for obj_id_str, obj_dict in video_dict.items():
        obj_id = int(obj_id_str)
        for frame_id_str, frame_dict in obj_dict.items():
            frame_id = int(frame_id_str)
            if expression in frame_dict:
                raw = frame_dict[expression]
                scores[frame_id][obj_id] = float(np.mean(raw))

    return scores


def compute_iou(box_a: List[float], box_b: List[float]) -> float:
    """IoU between two [x1, y1, x2, y2] boxes."""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])

    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def normalized_to_pixel(
    n_x1: float, n_y1: float, n_w: float, n_h: float, img_w: int, img_h: int
) -> List[float]:
    """Convert normalized [x1, y1, w, h] to pixel [x1, y1, x2, y2]."""
    x1 = n_x1 * img_w
    y1 = n_y1 * img_h
    w = n_w * img_w
    h = n_h * img_h
    return [x1, y1, x1 + w, y1 + h]


def match_tracks_to_gt(
    track_boxes: Dict[int, List[float]],
    gt_boxes: List[List[float]],
    iou_threshold: float = 0.3,
) -> Set[int]:
    """Greedy IoU matching: returns set of track IDs matched to a GT box."""
    matched_ids = set()
    used_gt = set()

    pairs = []
    for tid, tbox in track_boxes.items():
        for gi, gbox in enumerate(gt_boxes):
            iou = compute_iou(tbox, gbox)
            if iou >= iou_threshold:
                pairs.append((iou, tid, gi))

    pairs.sort(reverse=True)
    for _, tid, gi in pairs:
        if tid not in matched_ids and gi not in used_gt:
            matched_ids.add(tid)
            used_gt.add(gi)

    return matched_ids


class DummyTrack:
    """Track stub compatible with GMCLinkManager.process_frame()."""

    def __init__(self, obj_id: int, x: float, y: float, w: float, h: float):
        self.id = obj_id
        self.bbox = [x, y, x + w, y + h]
        self.centroid = np.array([x + w / 2.0, y + h / 2.0], dtype=np.float64)


# ── Evaluation ───────────────────────────────────────────────────────


def evaluate_predictions(
    predictions: Dict[int, bool], gt_matched_ids: Set[int], num_gt: int
) -> Tuple[int, int, int]:
    """
    Given per-track boolean predictions and GT-matched IDs, return (TP, FP, FN).
    """
    tp = sum(1 for tid, pred in predictions.items() if pred and tid in gt_matched_ids)
    fp = sum(
        1 for tid, pred in predictions.items() if pred and tid not in gt_matched_ids
    )
    fn = max(0, num_gt - tp)
    return tp, fp, fn


# ── Fusion Strategies ────────────────────────────────────────────────


def fuse_additive(ikun_logit: float, gmc_prob: float, weight: float) -> float:
    """Additive: ikun + gmc_logit * w.  Can both boost and suppress."""
    gmc_prob_c = max(1e-5, min(1.0 - 1e-5, gmc_prob))
    gmc_logit = math.log(gmc_prob_c / (1.0 - gmc_prob_c))
    return ikun_logit + gmc_logit * weight


def fuse_suppress(ikun_logit: float, gmc_prob: float, weight: float) -> float:
    """Suppress-only: GMC can penalize but never boost.  FP-safe."""
    gmc_prob_c = max(1e-5, min(1.0 - 1e-5, gmc_prob))
    gmc_logit = math.log(gmc_prob_c / (1.0 - gmc_prob_c))
    return ikun_logit + min(0.0, gmc_logit) * weight


def fuse_gate(ikun_logit: float, gmc_prob: float, weight: float) -> float:
    """Gate: if GMC confidence < threshold, zero out iKUN.  Hard AND."""
    # weight is re-used as the gmc_prob threshold
    if gmc_prob < weight:
        return ikun_logit - abs(ikun_logit)  # force negative
    return ikun_logit


FUSION_MODES = {
    "additive": fuse_additive,
    "suppress": fuse_suppress,
    "gate": fuse_gate,
}


# ── Expression Classification ────────────────────────────────────────

MOTION_KEYWORDS = {
    "moving", "turning", "driving", "going", "approaching", "passing",
    "coming", "crossing", "traveling", "heading", "accelerating",
    "slowing", "stopping", "leaving", "entering", "following",
    "walking", "faster", "direction", "horizon",
}

STATIONARY_KEYWORDS = {"parking", "parked", "stationary", "stopped"}


def classify_expression(sentence: str) -> str:
    """
    Classify a referring expression as 'motion', 'stationary', or 'appearance'.
    """
    tokens = set(sentence.lower().split())
    if tokens & MOTION_KEYWORDS:
        return "motion"
    if tokens & STATIONARY_KEYWORDS:
        return "stationary"
    return "appearance"


# ── Main Pipeline ────────────────────────────────────────────────────


def run_comparison(
    sequence: str = "0011",
    expression: str = "moving-cars",
    data_root: str = "refer-kitti",
    results_json: str = "iKUN/results.json",
    track_dir: str = "NeuralSORT",
    weights_path: str = "gmc_link_weights.pth",
    gmc_weight: float = 0.1,
    fusion_mode: str = "suppress",
    visualize: bool = False,
) -> dict:
    """
    Compare iKUN-only vs iKUN+GMC-Link on a single Refer-KITTI expression.

    Args:
        sequence:    KITTI sequence ID (e.g. "0011").
        expression:  iKUN expression key (e.g. "moving-cars").
        data_root:   Path to refer-kitti dataset root.
        results_json: Path to iKUN pre-computed results.json.
        track_dir:   Path to NeuralSORT tracking output root.
        weights_path: Path to trained GMC-Link weights.
        gmc_weight:  Scalar weight for GMC logit in fusion.
        fusion_mode: One of "additive", "suppress", "gate".
        visualize:   Whether to show OpenCV visualization.

    Returns:
        Dict with baseline and fused metrics for programmatic use.
    """
    fuse_fn = FUSION_MODES.get(fusion_mode)
    is_or_logic = fusion_mode == "or_logic"
    is_learned = fusion_mode == "learned"

    # Load learned fusion head if needed
    fusion_head_model, fusion_head_thr = None, 0.5
    if is_learned:
        from gmc_link.fusion_head import load_fusion_head
        fusion_head_model, fusion_head_thr = load_fusion_head()

    # ── 1. Initialize ──
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Device: {device}")

    # Load expression GT
    expression_path = os.path.join(
        data_root, "expression", sequence, f"{expression}.json"
    )
    with open(expression_path, "r", encoding="utf-8") as f:
        expr_json = json.load(f)
    sentence = expr_json["sentence"]
    gt_label_map = expr_json["label"]  # {frame_id_str(0-indexed): [track_ids]}
    expr_type = classify_expression(sentence)
    print(f'Expression: "{sentence}"  [{expr_type}]')
    print(f"GT spans {len(gt_label_map)} frames")

    # Load per-frame labels for bounding boxes
    labels_dir = os.path.join(
        data_root, "KITTI", "labels_with_ids", "image_02", sequence
    )
    gt_labels = load_labels_with_ids(labels_dir)

    # Image resolution (KITTI 0011: 375x1242)
    img_h, img_w = 375, 1242

    # Load NeuralSORT tracks (1-indexed frames)
    track_path = os.path.join(track_dir, sequence, "car", "predict.txt")
    ns_tracks = load_neuralsort_tracks(track_path)
    print(f"NeuralSORT: {len(ns_tracks)} frames with tracks")

    # Load iKUN pre-computed CLIP scores (1-indexed frames)
    ikun_scores = load_ikun_scores(results_json, sequence, expression)
    print(
        f"iKUN scores: {sum(len(v) for v in ikun_scores.values())} (obj,frame) entries"
    )

    # Initialize GMC-Link
    encoder = TextEncoder(device=device)
    linker = GMCLinkManager(weights_path=weights_path, device=device, lang_dim=384)
    language_embedding = encoder.encode(sentence)
    print(f"Language embedding: {language_embedding.shape}")

    # Frame directory for images
    frame_dir = os.path.join(data_root, "KITTI", "training", "image_02", sequence)
    frame_files = sorted(
        f for f in os.listdir(frame_dir) if f.endswith((".png", ".jpg"))
    )
    total_frames = len(frame_files)
    print(f"Processing {total_frames} frames...\n")

    # ── 2. Per-frame evaluation ──
    # Accumulators: (TP, FP, FN) for each method
    baseline_totals = [0, 0, 0]
    fused_totals = [0, 0, 0]
    baseline_scores_gt, baseline_scores_non_gt = [], []
    fused_scores_gt, fused_scores_non_gt = [], []
    frames_with_gt = 0

    for frame_0idx in range(total_frames):
        frame_1idx = frame_0idx + 1  # NeuralSORT / iKUN use 1-indexed
        frame_id_str = str(frame_0idx)  # GT uses 0-indexed

        # Skip frames with no tracks
        if frame_1idx not in ns_tracks:
            continue

        # Read image for GMC-Link
        frame_path = os.path.join(frame_dir, frame_files[frame_0idx])
        frame_img = cv2.imread(frame_path)
        if frame_img is None:
            continue

        # Build track objects + xyxy dict for this frame
        detections = ns_tracks[frame_1idx]
        active_tracks = []
        track_boxes_xyxy: Dict[int, List[float]] = {}

        for obj_id, x, y, w, h in detections:
            active_tracks.append(DummyTrack(obj_id, x, y, w, h))
            track_boxes_xyxy[obj_id] = [x, y, x + w, y + h]

        det_array = (
            np.array([[x, y, x + w, y + h] for _, x, y, w, h in detections])
            if detections
            else None
        )

        # Run GMC-Link on this frame
        gmc_scores, _ = linker.process_frame(
            frame_img, active_tracks, language_embedding, detections=det_array
        )

        # ── GT matching ──
        gt_track_ids = gt_label_map.get(frame_id_str, [])
        if gt_track_ids and frame_0idx in gt_labels:
            gt_bboxes = []
            for det in gt_labels[frame_0idx]:
                if det["track_id"] in gt_track_ids:
                    bbox = normalized_to_pixel(
                        det["x1_n"], det["y1_n"], det["w_n"], det["h_n"], img_w, img_h
                    )
                    gt_bboxes.append(bbox)
            matched_ids = match_tracks_to_gt(
                track_boxes_xyxy, gt_bboxes, iou_threshold=0.3
            )
        else:
            matched_ids = set()
            gt_bboxes = []

        num_gt = len(gt_track_ids)
        if num_gt == 0:
            continue
        frames_with_gt += 1

        # ── Compute predictions for both methods ──
        baseline_preds = {}
        fused_preds = {}

        for obj_id in track_boxes_xyxy:
            # iKUN baseline: raw CLIP logit > 0
            ikun_logit = ikun_scores.get(frame_1idx, {}).get(obj_id, None)
            if ikun_logit is None:
                # No iKUN score -> treat as negative for both
                baseline_preds[obj_id] = False
                fused_preds[obj_id] = False
                continue

            baseline_preds[obj_id] = ikun_logit > 0.0

            # iKUN + GMC-Link fusion
            gmc_prob = gmc_scores.get(obj_id, 0.0)

            if is_learned:
                # Learned fusion head
                is_motion_val = 1.0 if expr_type == "motion" else (0.5 if expr_type == "stationary" else 0.0)
                feat = torch.tensor([[ikun_logit, gmc_prob, is_motion_val]], dtype=torch.float32)
                with torch.no_grad():
                    prob = fusion_head_model.predict_prob(feat).item()
                fused_preds[obj_id] = prob >= fusion_head_thr
                fused_score = prob - 0.5  # center around 0 for score tracking
            elif is_or_logic:
                # OR-logic: expression-type-aware fusion
                if expr_type == "motion":
                    # For motion: positive if GMC says moving OR iKUN says match
                    fused_preds[obj_id] = (gmc_prob > gmc_weight) or (ikun_logit > 0.0)
                    fused_score = max(ikun_logit, gmc_prob - 0.5)  # synthetic score for tracking
                elif expr_type == "stationary":
                    # For stationary: iKUN primary, GMC suppress-only (same as appearance)
                    fused_score = fuse_suppress(ikun_logit, gmc_prob, 0.1)
                    fused_preds[obj_id] = fused_score > 0.0
                else:
                    # Appearance: iKUN primary, suppress-mode GMC filter
                    fused_score = fuse_suppress(ikun_logit, gmc_prob, 0.1)
                    fused_preds[obj_id] = fused_score > 0.0
            else:
                fused_score = fuse_fn(ikun_logit, gmc_prob, gmc_weight)
                fused_preds[obj_id] = fused_score > 0.0

            # Collect score distributions
            is_gt = obj_id in matched_ids
            if is_gt:
                baseline_scores_gt.append(ikun_logit)
                fused_scores_gt.append(fused_score)
            else:
                baseline_scores_non_gt.append(ikun_logit)
                fused_scores_non_gt.append(fused_score)

        # ── Evaluate ──
        b_tp, b_fp, b_fn = evaluate_predictions(baseline_preds, matched_ids, num_gt)
        f_tp, f_fp, f_fn = evaluate_predictions(fused_preds, matched_ids, num_gt)

        baseline_totals[0] += b_tp
        baseline_totals[1] += b_fp
        baseline_totals[2] += b_fn
        fused_totals[0] += f_tp
        fused_totals[1] += f_fp
        fused_totals[2] += f_fn

        # ── Visualization ──
        if visualize:
            vis = frame_img.copy()
            for obj_id, (x1, y1, x2, y2) in track_boxes_xyxy.items():
                is_gt = obj_id in matched_ids
                b_pos = baseline_preds.get(obj_id, False)
                f_pos = fused_preds.get(obj_id, False)

                # Green=fused positive, Red=fused negative, Cyan border=GT
                color = (0, 200, 0) if f_pos else (0, 0, 200)
                cv2.rectangle(
                    vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 2
                )
                if is_gt:
                    cv2.rectangle(
                        vis,
                        (int(x1) - 2, int(y1) - 2),
                        (int(x2) + 2, int(y2) + 2),
                        (255, 255, 0),
                        1,
                    )
                label = "B:{} F:{}".format(
                    "Y" if b_pos else "N", "Y" if f_pos else "N"
                )
                cv2.putText(
                    vis,
                    label,
                    (int(x1), int(y1) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    color,
                    1,
                )

            cv2.putText(
                vis,
                '"{}" | Frame {}/{}'.format(sentence, frame_0idx, total_frames),
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            cv2.imshow("iKUN vs iKUN+GMC (q to quit)", vis)
            if cv2.waitKey(30) & 0xFF == ord("q"):
                break

    if visualize:
        cv2.destroyAllWindows()

    # ── 3. Print Results ──
    def _metrics(tp, fp, fn):
        prec = tp / (tp + fp) if tp + fp > 0 else 0.0
        rec = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
        return prec, rec, f1

    b_prec, b_rec, b_f1 = _metrics(*baseline_totals)
    f_prec, f_rec, f_f1 = _metrics(*fused_totals)

    print("=" * 65)
    print("iKUN BASELINE vs iKUN + GMC-LINK  --  ABLATION RESULTS")
    print("=" * 65)
    print('Expression:  "{}"'.format(sentence))
    print("Sequence:    {}  |  Frames with GT: {}".format(sequence, frames_with_gt))
    print("Fusion:      {}  |  weight: {}  |  expr_type: {}".format(
        fusion_mode, gmc_weight, expr_type))
    print("-" * 65)
    print("{:<25} {:>18} {:>18}".format("Metric", "iKUN (baseline)", "iKUN + GMC"))
    print("-" * 65)
    print("{:.<25} {:>18} {:>18}".format("TP", baseline_totals[0], fused_totals[0]))
    print("{:.<25} {:>18} {:>18}".format("FP", baseline_totals[1], fused_totals[1]))
    print("{:.<25} {:>18} {:>18}".format("FN", baseline_totals[2], fused_totals[2]))
    print("{:.<25} {:>17.4f} {:>18.4f}".format("Precision", b_prec, f_prec))
    print("{:.<25} {:>17.4f} {:>18.4f}".format("Recall", b_rec, f_rec))
    print("{:.<25} {:>17.4f} {:>18.4f}".format("F1", b_f1, f_f1))
    print("-" * 65)

    # Score distributions
    if baseline_scores_gt:
        print(
            "{:.<25} {:>17.4f} {:>18.4f}".format(
                "GT avg logit",
                np.mean(baseline_scores_gt),
                np.mean(fused_scores_gt),
            )
        )
    if baseline_scores_non_gt:
        print(
            "{:.<25} {:>17.4f} {:>18.4f}".format(
                "Non-GT avg logit",
                np.mean(baseline_scores_non_gt),
                np.mean(fused_scores_non_gt),
            )
        )
    if baseline_scores_gt and baseline_scores_non_gt:
        b_sep = np.mean(baseline_scores_gt) - np.mean(baseline_scores_non_gt)
        f_sep = np.mean(fused_scores_gt) - np.mean(fused_scores_non_gt)
        print(
            "{:.<25} {:>+17.4f} {:>+18.4f}".format("Score separation", b_sep, f_sep)
        )

    # Delta summary
    print("=" * 65)
    delta_prec = f_prec - b_prec
    delta_rec = f_rec - b_rec
    delta_f1 = f_f1 - b_f1
    delta_fp = fused_totals[1] - baseline_totals[1]
    print(
        "GMC-Link effect:  "
        "Prec {:+.4f}  |  Rec {:+.4f}  |  "
        "F1 {:+.4f}  |  FP {:+d}".format(delta_prec, delta_rec, delta_f1, delta_fp)
    )
    print("=" * 65)

    return {
        "baseline": {"tp": baseline_totals[0], "fp": baseline_totals[1], "fn": baseline_totals[2],
                     "prec": b_prec, "rec": b_rec, "f1": b_f1},
        "fused": {"tp": fused_totals[0], "fp": fused_totals[1], "fn": fused_totals[2],
                  "prec": f_prec, "rec": f_rec, "f1": f_f1},
        "expr_type": expr_type,
    }


def run_ablation(
    sequence: str = "0011",
    expression: str = "moving-cars",
    data_root: str = "refer-kitti",
    results_json: str = "iKUN/results.json",
    track_dir: str = "NeuralSORT",
    weights_path: str = "gmc_link_weights.pth",
) -> None:
    """Sweep fusion modes and weights, print a compact comparison table."""
    configs = [
        ("suppress", 0.05),
        ("suppress", 0.10),
        ("suppress", 0.20),
        ("suppress", 0.50),
        ("additive", 0.05),
        ("additive", 0.10),
        ("gate",     0.30),
        ("gate",     0.40),
        ("gate",     0.50),
    ]

    rows = []
    baseline = None
    for mode, w in configs:
        r = run_comparison(
            sequence=sequence, expression=expression,
            data_root=data_root, results_json=results_json,
            track_dir=track_dir, weights_path=weights_path,
            gmc_weight=w, fusion_mode=mode, visualize=False,
        )
        if baseline is None:
            baseline = r["baseline"]
        rows.append((mode, w, r["fused"]))

    print("\n")
    print("=" * 75)
    print("FUSION ABLATION SWEEP")
    print("=" * 75)
    print("{:<12} {:>6}  {:>5} {:>5} {:>5}  {:>7} {:>7} {:>7}".format(
        "Mode", "Weight", "TP", "FP", "FN", "Prec", "Rec", "F1"))
    print("-" * 75)
    print("{:<12} {:>6}  {:>5} {:>5} {:>5}  {:>7.4f} {:>7.4f} {:>7.4f}  <-- baseline".format(
        "iKUN-only", "-",
        baseline["tp"], baseline["fp"], baseline["fn"],
        baseline["prec"], baseline["rec"], baseline["f1"]))
    print("-" * 75)
    for mode, w, m in rows:
        delta_fp = m["fp"] - baseline["fp"]
        marker = "  ** best" if m["prec"] > baseline["prec"] and m["rec"] >= baseline["rec"] else ""
        print("{:<12} {:>6}  {:>5} {:>5} {:>5}  {:>7.4f} {:>7.4f} {:>7.4f}  FP {:+d}{}".format(
            mode, w, m["tp"], m["fp"], m["fn"],
            m["prec"], m["rec"], m["f1"], delta_fp, marker))
    print("=" * 75)


def run_multi_expression(
    sequence: str = "0011",
    data_root: str = "refer-kitti",
    results_json: str = "iKUN/results.json",
    track_dir: str = "NeuralSORT",
    weights_path: str = "gmc_link_weights.pth",
    fusion_mode: str = "learned",
    gmc_weight: float = 0.65,
) -> None:
    """Evaluate across ALL expressions for a sequence, grouped by type."""
    import glob

    expr_dir = os.path.join(data_root, "expression", sequence)
    expr_files = sorted(glob.glob(os.path.join(expr_dir, "*.json")))

    results_by_type: Dict[str, list] = defaultdict(list)
    all_results = []

    for expr_file in expr_files:
        expr_name = os.path.splitext(os.path.basename(expr_file))[0]
        print(f"\n{'='*65}")
        print(f"Evaluating: {expr_name}")
        print(f"{'='*65}")
        try:
            r = run_comparison(
                sequence=sequence, expression=expr_name,
                data_root=data_root, results_json=results_json,
                track_dir=track_dir, weights_path=weights_path,
                gmc_weight=gmc_weight, fusion_mode=fusion_mode,
                visualize=False,
            )
            r["name"] = expr_name
            all_results.append(r)
            results_by_type[r["expr_type"]].append(r)
        except Exception as e:
            print(f"  SKIP: {e}")

    # ── Summary table ──
    print("\n\n")
    print("=" * 90)
    print(f"MULTI-EXPRESSION SUMMARY  |  Fusion: {fusion_mode}  weight: {gmc_weight}")
    print("=" * 90)
    print("{:<45} {:>5} {:>7} {:>7} {:>7} {:>7} {:>7}".format(
        "Expression", "Type", "B_Prec", "B_Rec", "B_F1", "F_Prec", "F_Rec", ))
    # Renamed: B=baseline, F=fused
    print("{:<45} {:>5} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7}".format(
        "", "", "", "", "", "", "", "F_F1"))
    print("-" * 90)

    for etype in ["motion", "stationary", "appearance"]:
        group = results_by_type.get(etype, [])
        if not group:
            continue
        print(f"\n  [{etype.upper()}]")
        for r in group:
            b, f_ = r["baseline"], r["fused"]
            delta_f1 = f_["f1"] - b["f1"]
            marker = " +++" if delta_f1 > 0.01 else (" ---" if delta_f1 < -0.01 else "")
            print("  {:<43} {:>7.4f} {:>7.4f} {:>7.4f} {:>7.4f} {:>7.4f} {:>7.4f}{}".format(
                r["name"][:43],
                b["prec"], b["rec"], b["f1"],
                f_["prec"], f_["rec"], f_["f1"], marker))

    # ── Per-type aggregates ──
    print("\n" + "=" * 90)
    print("PER-TYPE AGGREGATES")
    print("-" * 90)
    print("{:<15} {:>5} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7}".format(
        "Type", "N", "B_Prec", "B_Rec", "B_F1", "F_Prec", "F_Rec", "F_F1"))
    print("-" * 90)

    total_b = {"tp": 0, "fp": 0, "fn": 0}
    total_f = {"tp": 0, "fp": 0, "fn": 0}

    for etype in ["motion", "stationary", "appearance"]:
        group = results_by_type.get(etype, [])
        if not group:
            continue
        b_tp = sum(r["baseline"]["tp"] for r in group)
        b_fp = sum(r["baseline"]["fp"] for r in group)
        b_fn = sum(r["baseline"]["fn"] for r in group)
        f_tp = sum(r["fused"]["tp"] for r in group)
        f_fp = sum(r["fused"]["fp"] for r in group)
        f_fn = sum(r["fused"]["fn"] for r in group)

        total_b["tp"] += b_tp; total_b["fp"] += b_fp; total_b["fn"] += b_fn
        total_f["tp"] += f_tp; total_f["fp"] += f_fp; total_f["fn"] += f_fn

        def _m(tp, fp, fn):
            p = tp/(tp+fp) if tp+fp else 0.0
            r = tp/(tp+fn) if tp+fn else 0.0
            f = 2*p*r/(p+r) if p+r else 0.0
            return p, r, f

        bp, br, bf = _m(b_tp, b_fp, b_fn)
        fp_, fr, ff = _m(f_tp, f_fp, f_fn)
        print("{:<15} {:>5} {:>7.4f} {:>7.4f} {:>7.4f} {:>7.4f} {:>7.4f} {:>7.4f}".format(
            etype, len(group), bp, br, bf, fp_, fr, ff))

    # Overall
    def _m(tp, fp, fn):
        p = tp/(tp+fp) if tp+fp else 0.0
        r = tp/(tp+fn) if tp+fn else 0.0
        f = 2*p*r/(p+r) if p+r else 0.0
        return p, r, f

    bp, br, bf = _m(total_b["tp"], total_b["fp"], total_b["fn"])
    fp_, fr, ff = _m(total_f["tp"], total_f["fp"], total_f["fn"])
    print("-" * 90)
    print("{:<15} {:>5} {:>7.4f} {:>7.4f} {:>7.4f} {:>7.4f} {:>7.4f} {:>7.4f}".format(
        "OVERALL", len(all_results), bp, br, bf, fp_, fr, ff))
    print("=" * 90)


if __name__ == "__main__":
    import sys
    if "--sweep" in sys.argv:
        run_ablation()
    elif "--multi" in sys.argv:
        # Parse optional args
        mode = "learned"
        weight = 0.65
        for i, arg in enumerate(sys.argv):
            if arg == "--mode" and i + 1 < len(sys.argv):
                mode = sys.argv[i + 1]
            if arg == "--weight" and i + 1 < len(sys.argv):
                weight = float(sys.argv[i + 1])
        run_multi_expression(fusion_mode=mode, gmc_weight=weight)
    else:
        run_comparison(
            sequence="0011",
            expression="moving-cars",
            data_root="refer-kitti",
            results_json="iKUN/results.json",
            track_dir="NeuralSORT",
            weights_path="gmc_link_weights.pth",
            gmc_weight=0.65,
            fusion_mode="learned",
            visualize=False,
        )
