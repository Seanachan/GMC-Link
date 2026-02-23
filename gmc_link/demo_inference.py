"""
GMC-Link End-to-End Demo with YOLO + ByteTrack
================================================
Runs YOLOv8 + ByteTrack on refer-kitti sequence 0011, feeds real detections
into the GMC-Link pipeline, and evaluates alignment scores against ground truth.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import json
import torch
import numpy as np
from typing import Dict, List, Set, Tuple, Any, Optional

from ultralytics import YOLO
from gmc_link.manager import GMCLinkManager
from gmc_link.text_utils import TextEncoder
from gmc_link.dataset import load_labels_with_ids


# ── Helpers ──────────────────────────────────────────────────────────

class Track:
    """Lightweight track object compatible with GMCLinkManager.process_frame()."""
    def __init__(self, tid, centroid, prev_centroid=None):
        self.id = tid
        self.centroid = np.array(centroid, dtype=np.float64)
        if prev_centroid is not None:
            self.prev_centroid = np.array(prev_centroid, dtype=np.float64)


def compute_iou(box_a: List[float], box_b: List[float]) -> float:
    """Compute Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])
    
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def normalized_to_pixel(cx: float, cy: float, w: float, h: float, img_w: int, img_h: int) -> List[float]:
    """Convert normalized [cx, cy, w, h] to pixel coordinates [x1, y1, x2, y2]."""
    pw = w * img_w
    ph = h * img_h
    px = cx * img_w
    py = cy * img_h
    return [px - pw / 2, py - ph / 2, px + pw / 2, py + ph / 2]


def match_yolo_to_gt(yolo_boxes: Dict[int, List[float]], gt_boxes: List[List[float]], iou_threshold: float = 0.3) -> Set[int]:
    """
    Match YOLO detections to Ground Truth boxes using IoU bipartite matching.
    
    Args:
        yolo_boxes: Dictionary mapping YOLO track IDs to their bounding boxes [x1, y1, x2, y2].
        gt_boxes: List of Ground Truth bounding boxes [x1, y1, x2, y2] present in the current frame.
        iou_threshold: Minimum IoU required to consider a match valid.
        
    Returns:
        Set of YOLO track IDs that successfully matched to a GT box.
    """
    matched_yolo_ids = set()
    used_gt = set()
    
    # Greedy match: establish matches starting with the highest IoU pairs
    pairs = []
    for yid, ybox in yolo_boxes.items():
        for gi, gbox in enumerate(gt_boxes):
            iou = compute_iou(ybox, gbox)
            if iou >= iou_threshold:
                pairs.append((iou, yid, gi))
    
    pairs.sort(reverse=True)
    for iou, yid, gi in pairs:
        if yid not in matched_yolo_ids and gi not in used_gt:
            matched_yolo_ids.add(yid)
            used_gt.add(gi)
    
    return matched_yolo_ids


def load_ground_truth(expression_path: str, labels_dir: str) -> Tuple[str, Dict[str, list], Dict[int, list]]:
    """
    Parse the expression sequence JSON and the corresponding KITTI tracking format labels.
    
    Args:
        expression_path: Path to the refer-kitti JSON containing the sentence and frame-level tracking target IDs.
        labels_dir: Path to the refer-kitti `labels_with_ids` directory for the specific sequence.
        
    Returns:
        Tuple of:
            - sentence: The natural language motion description (e.g. "moving cars").
            - gt_label_map: Dictionary mapping string frame IDs to a list of target track IDs.
            - gt_labels: Dictionary mapping integer frame indices to a list of bounding box dicts.
    """
    with open(expression_path, 'r') as f:
        expression = json.load(f)
    sentence = expression['sentence']
    gt_label_map = expression['label']  # {frame_id_str: [track_ids]}
    
    # Load per-frame labels for all bounding boxes in the sequence
    gt_labels = load_labels_with_ids(labels_dir)
    return sentence, gt_label_map, gt_labels


def draw_frame_visualization(
    frame: np.ndarray, 
    yolo_ids: np.ndarray, 
    yolo_boxes_dict: Dict[int, List[float]], 
    scores: Dict[int, float], 
    matched_yolo_ids: Set[int], 
    velocities: Dict[int, np.ndarray],
    sentence: str,
    frame_idx: int,
    total_frames: int,
    num_gt: int
) -> None:
    """
    Apply OpenCV annotations to the image frame, drawing bounding boxes, velocity arrows, and scores.
    """
    for yid in yolo_ids:
        yid = int(yid)
        if yid not in yolo_boxes_dict:
            continue
            
        x1, y1, x2, y2 = [int(v) for v in yolo_boxes_dict[yid]]
        score = scores.get(yid, 0.0)
        is_gt = yid in matched_yolo_ids
        vel = velocities.get(yid, None)
        
        # Color by MODEL SCORE: green=high confidence match, red=low confidence match
        score_color_g = int(min(255, max(0, score * 255 * 2)))
        score_color_r = int(min(255, max(0, (1 - score) * 255 * 2)))
        color = (0, score_color_g, score_color_r)
        thickness = 2
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Highlight Ground Truth objects with a white border
        if is_gt:
            cv2.rectangle(frame, (x1-2, y1-2), (x2+2, y2+2), (255, 255, 255), 1)
        
        # Draw camera-motion-compensated world velocity arrow
        if vel is not None:
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            speed = np.linalg.norm((vel[0], vel[1]))
            arrow_scale = 8  # Velocities are pre-scaled by 100x via VELOCITY_SCALE
            end_x = int(cx + vel[0] * arrow_scale)
            end_y = int(cy + vel[1] * arrow_scale)
            cv2.arrowedLine(frame, (cx, cy), (end_x, end_y), (255, 255, 0), 2, tipLength=0.3)
            
            motion_label = f"v={speed:.1f}" if speed > 0.1 else "STATIC"
def evaluate_frame_metrics(
    scores: Dict[int, float], 
    matched_yolo_ids: Set[int], 
    num_gt_this_frame: int, 
    score_threshold: float
) -> Tuple[float, float, int, int, int, List[float], List[float]]:
    """
    Calculate Precision and Recall for a single frame based on a score threshold.
    
    Returns:
        Tuple of (precision, recall, true_positives, false_positives, false_negatives, gt_scores, non_gt_scores)
    """
    tp, fp, fn = 0, 0, 0
    gt_scores_this_frame = []
    non_gt_scores_this_frame = []
    
    for yid, score in scores.items():
        is_gt = yid in matched_yolo_ids
        if is_gt:
            gt_scores_this_frame.append(score)
        else:
            non_gt_scores_this_frame.append(score)
        
        # Threshold-based classification
        predicted_positive = score > score_threshold
        if predicted_positive and is_gt:
            tp += 1
        elif predicted_positive and not is_gt:
            fp += 1
    
    fn = max(0, num_gt_this_frame - tp)
    
    precision = tp / (tp + fp) if tp + fp > 0 else (1.0 if num_gt_this_frame == 0 else 0.0)
    recall = tp / (tp + fn) if tp + fn > 0 else 1.0
    
    return precision, recall, tp, fp, fn, gt_scores_this_frame, non_gt_scores_this_frame


# ── Main Pipeline ────────────────────────────────────────────────────

def run_e2e_evaluation(
    frame_dir: str, 
    expression_path: str, 
    labels_dir: str, 
    weights_path: str = "gmc_link_weights.pth", 
    visualize: bool = True
) -> None:
    """
    Run end-to-end YOLO + GMC-Link pipeline and evaluate against refer-kitti GT.
    
    Args:
        frame_dir: Path to image frames (e.g., refer-kitti/KITTI/training/image_02/0011)
        expression_path: Path to expression JSON (e.g., refer-kitti/expression/0011/moving-cars.json)
        labels_dir: Path to labels_with_ids (e.g., refer-kitti/KITTI/labels_with_ids/image_02/0011)
        weights_path: Path to trained GMC-Link weights
        visualize: Whether to show the OpenCV visualization
    """
    
    # ── 1. Initialize ──
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Load expression ground truth and bounding box labels
    sentence, gt_label_map, gt_labels = load_ground_truth(expression_path, labels_dir)
    print(f"Expression: \"{sentence}\"")
    print(f"GT spans {len(gt_label_map)} frames")
    
    # Initialize components
    encoder = TextEncoder(device=device)
    linker = GMCLinkManager(weights_path=weights_path, device=device, lang_dim=384)
    yolo = YOLO("yolov8n.pt")
    
    # Encode the prompt once
    language_embedding = encoder.encode(sentence)
    print(f"Language embedding shape: {language_embedding.shape}")
    
    # ── 2. Load frames ──
    frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith(('.png', '.jpg'))])
    print(f"Processing {len(frame_files)} frames...")
    
    # ── 3. Tracking state ──
    prev_centroids = {}  # {yolo_track_id: np.array([cx, cy])}
    
    # ── 4. Evaluation accumulators ──
    gt_scores = []       # Alignment scores for GT-matched detections
    non_gt_scores = []   # Alignment scores for non-GT detections
    frame_results = []   # Per-frame (precision, recall) at threshold
    score_threshold = 0.5
    
    for frame_idx, frame_name in enumerate(frame_files):
        frame_path = os.path.join(frame_dir, frame_name)
        frame = cv2.imread(frame_path)
        if frame is None:
            continue
        
        img_h, img_w = frame.shape[:2]
        frame_id_str = str(frame_idx)
        
        # ── YOLO Detection + ByteTrack Tracking ──
        # KITTI classes of interest: car(2), van(2), truck(7) → COCO: car=2, truck=7, bus=5
        results = yolo.track(frame, persist=True, verbose=False, classes=[2, 5, 7])
        
        if results[0].boxes is None or results[0].boxes.id is None:
            continue
        
        boxes = results[0].boxes
        yolo_ids = boxes.id.cpu().numpy().astype(int)
        xyxy = boxes.xyxy.cpu().numpy()  # (N, 4) in pixel coords
        
        # Build Track objects from YOLO detections
        active_tracks = []
        yolo_boxes_dict = {}  # {track_id: [x1,y1,x2,y2]} for IoU matching
        
        for i, yid in enumerate(yolo_ids):
            yid = int(yid)
            x1, y1, x2, y2 = xyxy[i]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            centroid = np.array([cx, cy])
            
            yolo_boxes_dict[yid] = [x1, y1, x2, y2]
            
            prev = prev_centroids.get(yid, None)
            track = Track(yid, centroid, prev)
            active_tracks.append(track)
            
            # Update prev centroid for next frame
            prev_centroids[yid] = centroid.copy()
        
        # ── GMC-Link Processing ──
        scores, velocities = linker.process_frame(frame, active_tracks, language_embedding)
        
        # ── GT Matching ──
        # Get GT track IDs for this frame
        gt_track_ids_this_frame = gt_label_map.get(frame_id_str, [])
        
        if gt_track_ids_this_frame and frame_idx in gt_labels:
            # Build GT bounding boxes in pixel coords
            gt_bboxes = []
            for det in gt_labels[frame_idx]:
                if det['track_id'] in gt_track_ids_this_frame:
                    bbox = normalized_to_pixel(det['cx'], det['cy'], det['w'], det['h'], img_w, img_h)
                    gt_bboxes.append(bbox)
            
            # Match YOLO detections to GT
            matched_yolo_ids = match_yolo_to_gt(yolo_boxes_dict, gt_bboxes, iou_threshold=0.3)
        else:
            matched_yolo_ids = set()
        
        # ── Collect Scores and Evaluate ──
        num_gt_this_frame = len(gt_track_ids_this_frame)
        
        precision, recall, tp, fp, fn, gt_img_scores, non_gt_img_scores = evaluate_frame_metrics(
            scores, matched_yolo_ids, num_gt_this_frame, score_threshold
        )
        
        gt_scores.extend(gt_img_scores)
        non_gt_scores.extend(non_gt_img_scores)
        
        if gt_track_ids_this_frame:  # Only count frames that have GT for summary metrics
            frame_results.append((precision, recall, tp, fp, fn))
        
        # ── Visualization ──
        if visualize:
            draw_frame_visualization(
                frame=frame,
                yolo_ids=yolo_ids,
                yolo_boxes_dict=yolo_boxes_dict,
                scores=scores,
                matched_yolo_ids=matched_yolo_ids,
                velocities=velocities,
                sentence=sentence,
                frame_idx=frame_idx,
                total_frames=len(frame_files),
                num_gt=num_gt_this_frame
            )
            
            cv2.imshow("GMC-Link E2E (press q to quit)", frame)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
    
    if visualize:
        cv2.destroyAllWindows()
    
    # ── 5. Print Evaluation Results ──
    print("\n" + "=" * 60)
    print("GMC-LINK END-TO-END EVALUATION RESULTS")
    print("=" * 60)
    print(f"Expression: \"{sentence}\"")
    print(f"Sequence: 0011 | Frames processed: {len(frame_files)}")
    print(f"Score threshold: {score_threshold}")
    print("-" * 60)
    
    if gt_scores:
        print(f"GT-matched tracks:     avg score = {np.mean(gt_scores):.4f} (n={len(gt_scores)})")
    else:
        print("GT-matched tracks:     No GT matches found")
    
    if non_gt_scores:
        print(f"Non-GT tracks:         avg score = {np.mean(non_gt_scores):.4f} (n={len(non_gt_scores)})")
    else:
        print("Non-GT tracks:         No non-GT tracks found")
    
    if gt_scores and non_gt_scores:
        separation = np.mean(gt_scores) - np.mean(non_gt_scores)
        print(f"Score separation:      {separation:+.4f} {'✅' if separation > 0 else '❌'}")
    
    if frame_results:
        avg_precision = np.mean([r[0] for r in frame_results])
        avg_recall = np.mean([r[1] for r in frame_results])
        total_tp = sum(r[2] for r in frame_results)
        total_fp = sum(r[3] for r in frame_results)
        total_fn = sum(r[4] for r in frame_results)
        
        print(f"\nPer-frame avg precision: {avg_precision:.4f}")
        print(f"Per-frame avg recall:    {avg_recall:.4f}")
        print(f"Total TP: {total_tp} | FP: {total_fp} | FN: {total_fn}")
        
        if total_tp + total_fp > 0:
            global_precision = total_tp / (total_tp + total_fp)
            print(f"Global precision:        {global_precision:.4f}")
        if total_tp + total_fn > 0:
            global_recall = total_tp / (total_tp + total_fn)
            print(f"Global recall:           {global_recall:.4f}")
    
    print("=" * 60)


if __name__ == "__main__":
    run_e2e_evaluation(
        frame_dir="refer-kitti/KITTI/training/image_02/0011",
        expression_path="refer-kitti/expression/0011/moving-cars.json",
        labels_dir="refer-kitti/KITTI/labels_with_ids/image_02/0011",
        weights_path="gmc_link_weights.pth",
        visualize=True,
    )