"""
Dataset generation for GMC-Link: spatial-temporal feature extraction from Refer-KITTI targets.
"""
import json
import os
from typing import Dict, List, Tuple, Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from gmc_link.utils import VELOCITY_SCALE, warp_points
from gmc_link.core import ORBHomographyEngine

HOMOGRAPHY_CACHE = {}


class MotionLanguageDataset(Dataset):
    """
    PyTorch Dataset for contrastive motion-language training.
    Each sample: (motion_vector, language_embedding, expression_id)
    expression_id is an integer — all samples from the same expression share the same ID.
    """

    def __init__(self, motion_data, language_data, labels):
        assert len(motion_data) == len(language_data) == len(labels)
        self.motion_data = motion_data
        self.language_data = language_data
        self.labels = labels

    def __len__(self):
        return len(self.motion_data)

    def __getitem__(self, idx):
        motion = torch.tensor(self.motion_data[idx], dtype=torch.float32)
        lang = torch.tensor(self.language_data[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return motion, lang, label


def collate_fn(batch):
    """Stack into (Batch, 8) motion, (Batch, L_dim) language, and (Batch,) integer labels."""
    motion_batch = torch.stack([item[0] for item in batch], dim=0)
    language_batch = torch.stack([item[1] for item in batch], dim=0)
    label_batch = torch.stack([item[2] for item in batch], dim=0)
    return motion_batch, language_batch, label_batch


# ── Data Loaders ─────────────────────────────────────────────────────


def load_refer_kitti_expressions(expression_dir):
    """Load all expression JSON files from a sequence's expression directory."""
    expressions = []
    for json_file in sorted(os.listdir(expression_dir)):
        if not json_file.endswith(".json"):
            continue
        with open(os.path.join(expression_dir, json_file), "r", encoding="utf-8") as f:
            expr = json.load(f)
        expressions.append(expr)
    return expressions


def load_labels_with_ids(labels_dir):
    """
    Load per-frame label files from labels_with_ids directory.
    Format: class_id track_id x1 y1 w h (normalized top-left coordinates)
    """
    labels = {}
    for txt_file in sorted(os.listdir(labels_dir)):
        if not txt_file.endswith(".txt"):
            continue
        frame_idx = int(os.path.splitext(txt_file)[0])
        frame_labels = []
        with open(os.path.join(labels_dir, txt_file), "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                frame_labels.append(
                    {
                        "class_id": int(parts[0]),
                        "track_id": int(parts[1]),
                        "x1_n": float(parts[2]),
                        "y1_n": float(parts[3]),
                        "w_n": float(parts[4]),
                        "h_n": float(parts[5]),
                    }
                )
        labels[frame_idx] = frame_labels
    return labels


# ── Training Data Builder ────────────────────────────────────────────

# Keywords that indicate motion-related expressions learnable from velocity vectors
MOTION_KEYWORDS = [
    "moving",
    "parking",
    "parked",
    "turning",
    "braking",
    "slower",
    "faster",
    "counter direction",
    "same direction",
    "in front of",
    "horizon direction",
    "following",
    "approaching",
    "overtaking",
    "stopping",
]


def is_motion_expression(sentence):
    """Check if a sentence describes motion (learnable from velocity vectors)."""
    lower = sentence.lower()
    return any(kw in lower for kw in MOTION_KEYWORDS)


def _collect_expressions(
    data_root: str, sequences: List[str], text_encoder: Any
) -> Tuple[List[Dict[str, Any]], Dict[str, np.ndarray], List[str]]:
    """
    Load JSON expressions, filter for motion keywords, and compute language embeddings.
    """
    print("  Encoding all sentences...")
    all_expressions = []
    sentence_embeddings = {}

    for seq in sequences:
        expression_dir = os.path.join(data_root, "expression", seq)
        if not os.path.exists(expression_dir):
            print(f"  Skipping {seq}: no expression dir")
            continue

        exprs = load_refer_kitti_expressions(expression_dir)
        skipped = 0
        for expr in exprs:
            sentence = expr["sentence"]
            if not is_motion_expression(sentence):
                skipped += 1
                continue

            if sentence not in sentence_embeddings:
                emb = text_encoder.encode(sentence).squeeze(0).cpu().numpy()
                sentence_embeddings[sentence] = emb

            all_expressions.append(
                {
                    "sentence": sentence,
                    "embedding": sentence_embeddings[sentence],
                    "label": expr["label"],
                    "seq": seq,
                }
            )

        if skipped:
            print(f"  {seq}: skipped {skipped} non-motion expressions")

    print(f"  Motion-related sentences: {len(sentence_embeddings)}")
    print(f"  Total expressions: {len(all_expressions)}")
    all_sentences = list(sentence_embeddings.keys())

    return all_expressions, sentence_embeddings, all_sentences


def _extract_target_centroids(
    data_root: str,
    seq: str,
    label_map: Dict[str, List[int]],
    frame_shape: Tuple[int, int] = (375, 1242),
) -> Dict[int, Dict[int, Tuple[float, float]]]:
    """
    Extract centroid coordinates and dimensions (cx, cy, w, h) in pixel space for target
    object tracks across frames.
    Uses labels_with_ids per-frame format (class_id track_id x1 y1 w h in normalized coords).

    Returns mapping: {track_id: {frame_id: (cx_px, cy_px, w_px, h_px)}}
    """
    labels_dir = os.path.join(data_root, "KITTI", "labels_with_ids", "image_02", seq)
    if not os.path.exists(labels_dir):
        return {}

    labels_by_frame = load_labels_with_ids(labels_dir)
    frame_ids = sorted([int(fid) for fid in label_map.keys()])
    h, w = frame_shape

    track_centroids = {}
    for fid in frame_ids:
        target_tracks = set(label_map[str(fid)])
        if fid not in labels_by_frame:
            continue

        for det in labels_by_frame[fid]:
            tid = det["track_id"]
            if tid in target_tracks:
                # Convert normalized coords to pixel coords and compute true centroid
                x1_px = det["x1_n"] * w
                y1_px = det["y1_n"] * h
                w_px = det["w_n"] * w
                h_px = det["h_n"] * h

                cx_px = x1_px + w_px / 2.0
                cy_px = y1_px + h_px / 2.0

                if tid not in track_centroids:
                    track_centroids[tid] = {}
                track_centroids[tid][fid] = (cx_px, cy_px, w_px, h_px)

    return track_centroids


def _generate_positive_pairs(
    track_centroids: Dict[int, Dict[int, Tuple[float, float, float, float]]],
    embedding: np.ndarray,
    expression_id: int,
    frame_gap: int,
    frame_shape: Tuple[int, int],
    seq: str = None,
    frame_dir: str = None,
    orb_engine: Any = None,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[int]]:
    """
    Generate positive (motion, language) pairs for contrastive training.
    Negatives are formed in-batch by SupervisedInfoNCE — samples with different
    expression_id values automatically contrast against each other.

    Uses ORBHomographyEngine to warp centroids to absolute world velocities.
    """
    h, w = frame_shape
    motion_data = []
    language_data = []
    labels = []

    for tid, centroids in track_centroids.items():
        sorted_frames = sorted(centroids.keys())
        for i in range(len(sorted_frames)):
            curr_fid = sorted_frames[i]
            target_fid = curr_fid + frame_gap

            best_j = None
            for j in range(i + 1, len(sorted_frames)):
                if sorted_frames[j] >= target_fid:
                    best_j = j
                    break

            if best_j is None:
                continue

            future_fid = sorted_frames[best_j]
            if future_fid - curr_fid > frame_gap * 2:
                continue

            # --- ORB Homography compensated velocity with Jitter ---
            if frame_dir is not None and orb_engine is not None and seq is not None:
                cache_key = (seq, curr_fid, future_fid)
                if cache_key in HOMOGRAPHY_CACHE:
                    homography = HOMOGRAPHY_CACHE[cache_key]
                else:
                    curr_img_path = os.path.join(frame_dir, f"{curr_fid:06d}.png")
                    future_img_path = os.path.join(frame_dir, f"{future_fid:06d}.png")

                    img_curr = cv2.imread(curr_img_path)
                    img_future = cv2.imread(future_img_path)

                    if img_curr is not None and img_future is not None:
                        homography = orb_engine.estimate_homography(
                            img_curr, img_future, prev_bboxes=None
                        )
                    else:
                        homography = None
                    HOMOGRAPHY_CACHE[cache_key] = homography

                if homography is not None:
                    cx1, cy1, bw1, bh1 = centroids[curr_fid]
                    cx2, cy2, bw2, bh2 = centroids[future_fid]

                    # Synthetic Jitter (+/- 2 pixels) to harden against YOLO noise
                    j_cx2 = cx2 + np.random.uniform(-2.0, 2.0)
                    j_cy2 = cy2 + np.random.uniform(-2.0, 2.0)
                    j_bw2 = bw2 + np.random.uniform(-2.0, 2.0)
                    j_bh2 = bh2 + np.random.uniform(-2.0, 2.0)

                    pts = np.array([[cx1, cy1]], dtype=np.float32)
                    warped_pts = warp_points(pts, homography)
                    wcx1, wcy1 = warped_pts[0]

                    # 8D Spatio-Temporal Features
                    dx = (j_cx2 - wcx1) / w * VELOCITY_SCALE
                    dy = (j_cy2 - wcy1) / h * VELOCITY_SCALE
                    dw = (j_bw2 - bw1) / w * VELOCITY_SCALE
                    dh = (j_bh2 - bh1) / h * VELOCITY_SCALE

                    cx_n, cy_n = cx1 / w, cy1 / h
                    bw_n, bh_n = bw1 / w, bh1 / h
                    motion_vec = np.array(
                        [dx, dy, dw, dh, cx_n, cy_n, bw_n, bh_n], dtype=np.float32
                    )
                else:
                    cx1, cy1, bw1, bh1 = centroids[curr_fid]
                    cx2, cy2, bw2, bh2 = centroids[future_fid]
                    dx = (cx2 - cx1) / w * VELOCITY_SCALE
                    dy = (cy2 - cy1) / h * VELOCITY_SCALE
                    dw = (bw2 - bw1) / w * VELOCITY_SCALE
                    dh = (bh2 - bh1) / h * VELOCITY_SCALE

                    cx_n, cy_n = cx1 / w, cy1 / h
                    bw_n, bh_n = bw1 / w, bh1 / h
                    motion_vec = np.array(
                        [dx, dy, dw, dh, cx_n, cy_n, bw_n, bh_n], dtype=np.float32
                    )
            else:
                # Fallback: raw centroid differences
                cx1, cy1, bw1, bh1 = centroids[curr_fid]
                cx2, cy2, bw2, bh2 = centroids[future_fid]
                dx = (cx2 - cx1) / w * VELOCITY_SCALE
                dy = (cy2 - cy1) / h * VELOCITY_SCALE
                dw = (bw2 - bw1) / w * VELOCITY_SCALE
                dh = (bh2 - bh1) / h * VELOCITY_SCALE

                cx_n, cy_n = cx1 / w, cy1 / h
                bw_n, bh_n = bw1 / w, bh1 / h
                motion_vec = np.array(
                    [dx, dy, dw, dh, cx_n, cy_n, bw_n, bh_n], dtype=np.float32
                )

            # Positive pair: this motion matches this expression
            motion_data.append(motion_vec)
            language_data.append(embedding.copy())
            labels.append(expression_id)

    return motion_data, language_data, labels


def build_training_data(
    data_root: str,
    sequences: List[str],
    text_encoder: Any,
    frame_gap: int = 5,
    frame_shape: Tuple[int, int] = (375, 1242),
) -> Tuple[List[np.ndarray], List[np.ndarray], List[int]]:
    """
    Build (motion, language, expression_id) training triples for contrastive learning.

    Each unique expression gets an integer ID. All motion vectors that correspond
    to the same expression share that ID, so SupervisedInfoNCE treats them as
    positives and everything else as in-batch negatives.

    Uses ORB Homography and centroid-difference tracking for ego-motion compensation.
    """
    all_expressions, sentence_embeddings, all_sentences = _collect_expressions(
        data_root, sequences, text_encoder
    )

    print("  Initializing ORBHomographyEngine for dataset ego-motion compensation...")
    orb_engine = ORBHomographyEngine(max_features=1500)

    print(
        "  Computing centroid-difference velocities and generating positive pairs..."
    )

    # Map each unique sentence to a stable integer class ID
    sentence_to_id = {s: i for i, s in enumerate(all_sentences)}

    motion_data = []
    language_data = []
    labels = []

    for expr in all_expressions:
        seq = expr["seq"]
        label_map = expr["label"]
        sentence = expr["sentence"]
        embedding = expr["embedding"]
        expression_id = sentence_to_id[sentence]

        # Detect actual frame dimensions from first image
        frame_dir = os.path.join(data_root, "KITTI", "training", "image_02", seq)
        actual_shape = frame_shape
        if os.path.exists(frame_dir):
            sample_frames = [f for f in os.listdir(frame_dir) if f.endswith(".png")]
            if sample_frames:
                sample = cv2.imread(os.path.join(frame_dir, sample_frames[0]))
                if sample is not None:
                    actual_shape = (sample.shape[0], sample.shape[1])

        track_centroids = _extract_target_centroids(
            data_root, seq, label_map, frame_shape=actual_shape
        )

        # Generate only positive pairs — negatives are formed in-batch by InfoNCE
        m_data, l_data, lbls = _generate_positive_pairs(
            track_centroids,
            embedding,
            expression_id,
            frame_gap,
            actual_shape,
            seq=seq,
            frame_dir=frame_dir,
            orb_engine=orb_engine,
        )

        motion_data.extend(m_data)
        language_data.extend(l_data)
        labels.extend(lbls)

    n_classes = len(sentence_to_id)
    print(f"  Unique expressions: {n_classes} | Total samples: {len(labels)}")
    return motion_data, language_data, labels
