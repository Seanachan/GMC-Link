"""
Dataset generation for GMC-Link: spatial-temporal feature extraction from Refer-KITTI targets.
"""
import hashlib
import json
import os
from typing import Dict, List, Tuple, Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from gmc_link.utils import VELOCITY_SCALE, warp_points
from gmc_link.core import ORBHomographyEngine
from gmc_link.ego.ego_router import make_ego_router
from gmc_link.features.omf_stats import per_bbox_omf_stats

HOMOGRAPHY_CACHE = {}

# Flow fields are ~3.7MB each; full V1 train set is ~45k fields total. We use a
# small LRU so expressions sharing the same seq reuse loaded flows (each seq has
# many GT tracks that overlap on frames). 256 entries ≈ 1GB RAM, a good tradeoff
# vs disk re-decompression. Callers pass .cache_clear() hints between seqs via
# ``_omf_cache_reset`` to keep memory bounded even for the largest seqs.
from functools import lru_cache


@lru_cache(maxsize=1024)
def _load_omf_field(ego_router_name: str, seq: str, frame_id: int, gap: int) -> np.ndarray:
    """Load a precomputed Farneback OMF .npz → (H, W, 2). Return None if missing."""
    path = os.path.join(
        "cache", "omf", ego_router_name, seq, f"{int(frame_id):06d}_gap{int(gap)}.npz"
    )
    if not os.path.exists(path):
        return None
    try:
        with np.load(path) as npz:
            return np.asarray(npz["flow"], dtype=np.float32)
    except Exception:
        return None


def _omf_cache_reset():
    """Clear the flow-field LRU. Call between sequences to bound peak RSS."""
    _load_omf_field.cache_clear()

# Multi-scale frame gaps for temporal velocity features
FRAME_GAPS = [2, 5, 10]  # short, mid, long

# ── Training-data disk cache ─────────────────────────────────────────
# Bump CACHE_VERSION whenever the build logic changes in a way that
# affects output (feature layout, jitter, motion keywords, etc.).
CACHE_VERSION = 1
CACHE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "cache", "training_data")
)


def _build_cache_key(
    data_root, sequences, frame_gaps, frame_shape,
    use_group_labels, extra_features, seq_len, text_encoder_name="all-MiniLM-L6-v2",
    ego_router_name="orb",
):
    """Deterministic hash of everything that affects build_training_data output.

    Note: when ``ego_router_name == "orb"`` (the historic default), the ego
    field is intentionally omitted from the hash so that existing ORB-built
    caches continue to hit. Any non-ORB backend (e.g. ``"recoverpose"``) adds
    the field explicitly, producing a distinct hash.
    """
    key_obj = {
        "version": CACHE_VERSION,
        "data_root": os.path.abspath(data_root),
        "sequences": sorted(sequences),
        "frame_gaps": list(frame_gaps),
        "frame_shape": list(frame_shape),
        "use_group_labels": bool(use_group_labels),
        "extra_features": sorted(extra_features) if extra_features else None,
        "seq_len": int(seq_len),
        "text_encoder": str(text_encoder_name),
    }
    if ego_router_name and ego_router_name != "orb":
        key_obj["ego_router"] = str(ego_router_name)
    key_str = json.dumps(key_obj, sort_keys=True)
    key_hash = hashlib.sha256(key_str.encode()).hexdigest()[:16]
    return key_hash, key_obj


def _try_load_cache(cache_key, seq_len):
    """Return tuple of lists matching build_training_data's return contract, or None."""
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.npz")
    if not os.path.exists(cache_path):
        return None
    try:
        data = np.load(cache_path, allow_pickle=False)
        if seq_len > 0:
            sm = data["seq_motion"]
            mk = data["seq_masks"]
            sl = data["seq_language"]
            return (
                [sm[i] for i in range(sm.shape[0])],
                [mk[i] for i in range(mk.shape[0])],
                [sl[i] for i in range(sl.shape[0])],
                data["seq_labels"].astype(int).tolist(),
            )
        md = data["motion_data"]
        ld = data["language_data"]
        return (
            [md[i] for i in range(md.shape[0])],
            [ld[i] for i in range(ld.shape[0])],
            data["labels"].astype(int).tolist(),
        )
    except Exception as e:
        print(f"  [cache] failed to load {cache_path}: {e}")
        return None


def _save_cache(cache_key, key_obj, payload, seq_len):
    """Persist build output as stacked arrays + JSON sidecar."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.npz")
    sidecar = os.path.join(CACHE_DIR, f"{cache_key}.json")

    if seq_len > 0:
        seq_motion, seq_masks, seq_language, seq_labels = payload
        if not seq_motion:
            return
        arrays = {
            "seq_motion": np.stack(seq_motion),
            "seq_masks": np.stack(seq_masks),
            "seq_language": np.stack(seq_language),
            "seq_labels": np.array(seq_labels, dtype=np.int64),
        }
    else:
        motion_data, language_data, labels = payload
        if not motion_data:
            return
        arrays = {
            "motion_data": np.stack(motion_data),
            "language_data": np.stack(language_data),
            "labels": np.array(labels, dtype=np.int64),
        }

    np.savez(cache_path, **arrays)
    with open(sidecar, "w") as f:
        json.dump(key_obj, f, indent=2)
    size_mb = os.path.getsize(cache_path) / (1024 * 1024)
    print(f"  [cache] saved {size_mb:.1f} MB → {cache_path}")


# ── Extra feature definitions ───────────────────────────────────────
# Each feature: (name, n_dims). Order matters for vector layout.
EXTRA_FEATURE_DIMS = {
    "speed_m": 1,       # F1: mid-scale speed magnitude
    "heading_m": 1,     # F2: mid-scale heading angle
    "accel": 2,         # F3: acceleration (long - short velocity)
    "ego_motion": 2,    # F4: ego/camera motion (dx, dy)
    "neighbor_mean_vel": 2,  # F5: mean velocity of other tracks
    "velocity_rank": 1,      # F6: speed rank among neighbors
    "heading_diff": 1,       # F7: heading vs mean flow
    "nn_dist": 1,            # F8: nearest neighbor distance
    "track_density": 1,      # F9: density of nearby tracks
    "accel_multiscale": 6,   # exp36a: (dx,dy) × {2,5,10} second-order diff
    "heading_sincos": 6,     # exp36a: (sin,cos) × {2,5,10} heading
    "ego_velocity_concat": 2,  # exp37 stage C: EMAP-style ego (vx, vy) structural conditioning
    "omf_stats": 15,           # exp37 stage B: per-bbox OMF pooling × 3 scales
}


def compute_extra_dims(extra_features):
    """Return total extra dimensions for the given feature list."""
    if not extra_features:
        return 0
    return sum(EXTRA_FEATURE_DIMS[f] for f in extra_features)


def compute_per_track_extras(extra_features, scale_velocities, ego_dx_m=0.0,
                             ego_dy_m=0.0, accel_per_scale=None,
                             omf_stats_per_scale=None):
    """
    Compute per-track extra features (F1-F4) from existing velocity data.

    Args:
        extra_features: list of feature names to compute
        scale_velocities: [(dx_s, dy_s), (dx_m, dy_m), (dx_l, dy_l)]
        ego_dx_m, ego_dy_m: mid-scale ego motion components
        accel_per_scale: optional [(ax_s, ay_s), (ax_m, ay_m), (ax_l, ay_l)]
            with true temporal acceleration (v_k[t] - v_k[t-k])/k per scale.
            If None and accel_multiscale requested, zeros are emitted.
        omf_stats_per_scale: optional sequence of 3 arrays of length 5 each
            (one per FRAME_GAPS scale) produced by
            ``gmc_link.features.omf_stats.per_bbox_omf_stats``. If None and
            ``omf_stats`` requested, zeros are emitted.

    Returns:
        list of float values to append to the base 13D vector
    """
    if not extra_features:
        return []

    extras = []
    dx_s, dy_s = scale_velocities[0]
    dx_m, dy_m = scale_velocities[1]
    dx_l, dy_l = scale_velocities[2]

    for feat in extra_features:
        if feat == "speed_m":
            extras.append(np.sqrt(dx_m ** 2 + dy_m ** 2))
        elif feat == "heading_m":
            extras.append(np.arctan2(dy_m, dx_m))
        elif feat == "accel":
            extras.extend([dx_l - dx_s, dy_l - dy_s])
        elif feat == "ego_motion":
            extras.extend([ego_dx_m, ego_dy_m])
        elif feat == "accel_multiscale":
            if accel_per_scale is None:
                extras.extend([0.0] * 6)
            else:
                for ax, ay in accel_per_scale:
                    extras.append(float(ax))
                    extras.append(float(ay))
        elif feat == "heading_sincos":
            # Smooth heading encoding per scale (avoids atan2 ±π discontinuity)
            # When |v|→0, atan2(0,0)=0 → sin=0, cos=1 (stationary encoded).
            for dx, dy in scale_velocities:
                theta = np.arctan2(dy, dx)
                extras.append(float(np.sin(theta)))
                extras.append(float(np.cos(theta)))
        elif feat == "ego_velocity_concat":
            # Exp 37 Stage C: mid-scale ego translation (vx, vy). Same upstream
            # signal as the older F4 "ego_motion" channel; distinct name marks
            # intent as structural conditioning (EMAP) for the Stage C memo.
            extras.extend([ego_dx_m, ego_dy_m])
        elif feat == "omf_stats":
            # Exp 37 Stage B: per-bbox OMF [mean_dx, mean_dy, std_dx, std_dy,
            # max_mag] × 3 FRAME_GAPS scales. Zero-fill when dense flow is
            # unavailable so the vector layout stays consistent; the Stage B
            # runner must populate ``omf_stats_per_scale`` before training to
            # exercise the feature.
            if omf_stats_per_scale is None:
                extras.extend([0.0] * 15)
            else:
                for scale_stats in omf_stats_per_scale:
                    extras.extend(float(v) for v in scale_stats)
        # F5-F9 are relational — handled separately

    return extras


def compute_relational_extras(extra_features, my_dx, my_dy, my_cx, my_cy,
                              neighbor_vels, neighbor_centroids, frame_diag):
    """
    Compute relational extra features (F5-F9) from neighbor context.

    Args:
        extra_features: list of feature names
        my_dx, my_dy: this track's mid-scale residual velocity
        my_cx, my_cy: this track's normalized centroid
        neighbor_vels: list of (dx, dy) for other tracks at this frame
        neighbor_centroids: list of (cx_n, cy_n) for other tracks
        frame_diag: frame diagonal (normalized, =sqrt(1² + 1²) if using normalized coords)
    """
    if not extra_features:
        return []

    extras = []
    has_neighbors = len(neighbor_vels) > 0

    for feat in extra_features:
        if feat == "neighbor_mean_vel":
            if has_neighbors:
                mean_dx = np.mean([v[0] for v in neighbor_vels])
                mean_dy = np.mean([v[1] for v in neighbor_vels])
            else:
                mean_dx, mean_dy = 0.0, 0.0
            extras.extend([mean_dx, mean_dy])

        elif feat == "velocity_rank":
            if has_neighbors:
                my_speed = np.sqrt(my_dx ** 2 + my_dy ** 2)
                slower_count = sum(1 for v in neighbor_vels
                                   if np.sqrt(v[0]**2 + v[1]**2) < my_speed)
                rank = slower_count / len(neighbor_vels)
            else:
                rank = 0.5
            extras.append(rank)

        elif feat == "heading_diff":
            if has_neighbors:
                my_heading = np.arctan2(my_dy, my_dx)
                mean_dx = np.mean([v[0] for v in neighbor_vels])
                mean_dy = np.mean([v[1] for v in neighbor_vels])
                mean_heading = np.arctan2(mean_dy, mean_dx)
                diff = my_heading - mean_heading
                # Normalize to [-pi, pi]
                diff = (diff + np.pi) % (2 * np.pi) - np.pi
            else:
                diff = 0.0
            extras.append(diff)

        elif feat == "nn_dist":
            if has_neighbors:
                dists = [np.sqrt((my_cx - nc[0])**2 + (my_cy - nc[1])**2)
                         for nc in neighbor_centroids]
                nn = min(dists) / (frame_diag + 1e-6)
            else:
                nn = 1.0
            extras.append(nn)

        elif feat == "track_density":
            if has_neighbors:
                radius = 0.2  # 20% of frame diagonal in normalized coords
                count = sum(1 for nc in neighbor_centroids
                            if np.sqrt((my_cx - nc[0])**2 + (my_cy - nc[1])**2) < radius)
                density = count / len(neighbor_centroids)
            else:
                density = 0.0
            extras.append(density)

    return extras


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
    """Stack into (Batch, 13) motion, (Batch, L_dim) language, and (Batch,) integer labels."""
    motion_batch = torch.stack([item[0] for item in batch], dim=0)
    language_batch = torch.stack([item[1] for item in batch], dim=0)
    label_batch = torch.stack([item[2] for item in batch], dim=0)
    return motion_batch, language_batch, label_batch


class SequenceMotionLanguageDataset(Dataset):
    """
    PyTorch Dataset for temporal contrastive motion-language training.
    Each sample: (motion_sequence, padding_mask, language_embedding, expression_id)
    """

    def __init__(self, motion_seqs, masks, language_data, labels):
        assert len(motion_seqs) == len(masks) == len(language_data) == len(labels)
        self.motion_seqs = motion_seqs
        self.masks = masks
        self.language_data = language_data
        self.labels = labels

    def __len__(self):
        return len(self.motion_seqs)

    def __getitem__(self, idx):
        seq = torch.tensor(self.motion_seqs[idx], dtype=torch.float32)
        mask = torch.tensor(self.masks[idx], dtype=torch.bool)
        lang = torch.tensor(self.language_data[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return seq, mask, lang, label


def sequence_collate_fn(batch):
    """Stack into (B, T, D) motion, (B, T+1) mask, (B, 384) language, (B,) labels."""
    seq_batch = torch.stack([item[0] for item in batch], dim=0)
    mask_batch = torch.stack([item[1] for item in batch], dim=0)
    lang_batch = torch.stack([item[2] for item in batch], dim=0)
    label_batch = torch.stack([item[3] for item in batch], dim=0)
    return seq_batch, mask_batch, lang_batch, label_batch


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
# Keywords that indicate motion-related expressions learnable from velocity vectors.
# Updated for Refer-KITTI V2 which includes richer motion vocabulary generated by LLMs.
MOTION_KEYWORDS = [
    # Basic motion states (V1 + V2)
    "moving",
    "in motion",
    "driving",
    "parking",
    "parked",
    "at rest",
    "stationary",
    "still",
    "stopping",
    "stopped",
    # Direction (V1 + V2)
    "turning",
    "counter direction",
    "same direction",
    "opposite direction",
    "contrary direction",
    "reverse direction",
    "horizon direction",
    "heading",
    "in front of",
    # Speed (V1 + V2)
    "braking",
    "brake",
    "slower",
    "faster",
    "speedier",
    "accelerat",
    "decelerat",
    "slowing",
    # Interaction (V1 + V2)
    "following",
    "approaching",
    "overtaking",
    "travelling",
    "traveling",
]


def is_motion_expression(sentence):
    """Check if a sentence describes motion (learnable from velocity vectors)."""
    lower = sentence.lower()
    return any(kw in lower for kw in MOTION_KEYWORDS)


# ── Motion-type grouping ────────────────────────────────────────────
# Maps fine-grained expressions → 6 motion-type groups that the 13D
# features can actually distinguish.  Priority order matters: "moving
# turning cars" → TURNING (more specific), not MOVING.

MOTION_TYPE_GROUPS = {
    "braking": 0,
    "slower": 0,
    "turning": 1,
    "parking": 2,
    "counter direction": 3,
    "contrary direction": 3,
    "reverse direction": 3,
    "opposite direction": 3,
    "horizon direction": 4,
    "in front of": 4,
    "same direction": 5,
    "moving": 5,
    "in motion": 5,
    "driving": 5,
    "following": 5,
    "approaching": 5,
    "travelling": 5,
    "traveling": 5,
    "overtaking": 5,
    "accelerat": 5,
    "faster": 5,
    "speedier": 5,
}

MOTION_TYPE_NAMES = {
    0: "braking",
    1: "turning",
    2: "parking",
    3: "counter_dir",
    4: "approaching",
    5: "moving",
}


def motion_type_group(sentence):
    """
    Map a sentence to its motion-type group ID.
    Returns group ID (int) or None if no motion keyword matches.
    Priority: more specific keywords checked first.
    """
    lower = sentence.lower()
    for keyword, group_id in MOTION_TYPE_GROUPS.items():
        if keyword in lower:
            return group_id
    return None


def _collect_expressions(
    data_root: str, sequences: List[str], text_encoder: Any
) -> Tuple[List[Dict[str, Any]], Dict[str, np.ndarray], List[str]]:
    """
    Load JSON expressions and compute language embeddings using batch encoding.
    """

    print("Encoding all sentences (batch mode)...")

    all_expressions = []
    raw_data = []
    sentence_set = set()

    # Step 1: Collect all sentences and metadata
    for seq in sequences:
        expression_dir = os.path.join(data_root, "expression", seq)
        if not os.path.exists(expression_dir):
            print(f"Skipping {seq}: no expression dir")
            continue

        exprs = load_refer_kitti_expressions(expression_dir)
        for expr in exprs:
            sentence = expr["sentence"]

            raw_data.append({
                "sentence": sentence,
                "label": expr["label"],
                "seq": seq,
            })

            sentence_set.add(sentence)

    # Step 2: Create a deterministic list of unique sentences
    all_sentences = sorted(sentence_set)
    print(f"Unique sentences: {len(all_sentences)}")

    # Step 3: Batch encode all sentences
    embeddings = text_encoder.encode(
        all_sentences,
        batch_size=64,
        convert_to_tensor=True,
        show_progress_bar=True
    )

    # Step 4: Build sentence-to-embedding mapping
    sentence_embeddings = {
        s: embeddings[i].cpu().numpy()
        for i, s in enumerate(all_sentences)
    }

    # Step 5: Assign embeddings back to each expression
    for item in raw_data:
        all_expressions.append({
            "sentence": item["sentence"],
            "embedding": sentence_embeddings[item["sentence"]],
            "label": item["label"],
            "seq": item["seq"],
        })

    print(f"Total expressions: {len(all_expressions)}")

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


def _extract_all_track_centroids(
    data_root: str,
    seq: str,
    frame_shape: Tuple[int, int] = (375, 1242),
) -> Dict[int, Dict[int, Tuple[float, float, float, float]]]:
    """
    Extract centroid coordinates for ALL tracks (not just GT) across all frames.
    Returns mapping: {track_id: {frame_id: (cx_px, cy_px, w_px, h_px)}}
    """
    labels_dir = os.path.join(data_root, "KITTI", "labels_with_ids", "image_02", seq)
    if not os.path.exists(labels_dir):
        return {}

    labels_by_frame = load_labels_with_ids(labels_dir)
    h, w = frame_shape

    all_centroids = {}
    for fid, detections in labels_by_frame.items():
        for det in detections:
            tid = det["track_id"]
            x1_px = det["x1_n"] * w
            y1_px = det["y1_n"] * h
            w_px = det["w_n"] * w
            h_px = det["h_n"] * h
            cx_px = x1_px + w_px / 2.0
            cy_px = y1_px + h_px / 2.0

            if tid not in all_centroids:
                all_centroids[tid] = {}
            all_centroids[tid][fid] = (cx_px, cy_px, w_px, h_px)

    return all_centroids


def _precompute_frame_track_data(
    all_track_centroids: Dict[int, Dict[int, Tuple]],
    frame_shape: Tuple[int, int],
    frame_gaps: List[int],
    seq: str,
    frame_dir: str,
    orb_engine: Any,
) -> Dict[int, Dict[int, Tuple[float, float, float, float]]]:
    """
    Pre-compute mid-scale residual velocity + normalized position for ALL tracks
    at each frame in a sequence. Used as neighbor context for relational features.

    Returns: {frame_id: {track_id: (dx_m, dy_m, cx_n, cy_n)}}
    """
    h, w = frame_shape
    primary_gap = frame_gaps[1]  # mid-scale
    frame_data = {}

    for tid, centroids in all_track_centroids.items():
        sorted_frames = sorted(centroids.keys())
        for i in range(len(sorted_frames)):
            curr_fid = sorted_frames[i]
            future_j = _find_future_frame(sorted_frames, i, primary_gap)
            if future_j is None:
                continue
            future_fid = sorted_frames[future_j]

            dx, dy, _, _, _ = _compute_velocity_at_gap(
                centroids, curr_fid, future_fid, frame_shape,
                seq, frame_dir, orb_engine,
            )

            cx_px, cy_px, _, _ = centroids[curr_fid]
            cx_n = cx_px / w
            cy_n = cy_px / h

            if curr_fid not in frame_data:
                frame_data[curr_fid] = {}
            frame_data[curr_fid][tid] = (dx, dy, cx_n, cy_n)

    return frame_data


def _find_future_frame(sorted_frames, start_idx, gap):
    """Find the best future frame index at approximately `gap` frames ahead."""
    curr_fid = sorted_frames[start_idx]
    target_fid = curr_fid + gap
    for j in range(start_idx + 1, len(sorted_frames)):
        if sorted_frames[j] >= target_fid:
            if sorted_frames[j] - curr_fid > gap * 2:
                return None
            return j
    return None


def _find_past_frame_id(vel_hist, curr_fid, gap):
    """Return frame_id in vel_hist closest to `curr_fid - gap` (within [gap, 2*gap] back)."""
    if not vel_hist:
        return None
    target = curr_fid - gap
    best_fid = None
    best_err = None
    for fid in vel_hist:
        if fid > target:
            continue
        if curr_fid - fid > 2 * gap:
            continue
        err = abs(fid - target)
        if best_err is None or err < best_err:
            best_err = err
            best_fid = fid
    return best_fid


def _compute_velocity_at_gap(
    centroids, curr_fid, future_fid, frame_shape, seq, frame_dir, orb_engine
):
    """
    Compute residual velocity (raw - ego) for a frame pair.
    Returns (res_dx, res_dy, bg_residual).
    """
    h, w = frame_shape

    homography = None
    bg_residual = np.zeros(2, dtype=np.float32)
    if frame_dir is not None and orb_engine is not None and seq is not None:
        cache_key = (seq, curr_fid, future_fid)
        if cache_key in HOMOGRAPHY_CACHE:
            homography, bg_residual = HOMOGRAPHY_CACHE[cache_key]
        else:
            curr_img_path = os.path.join(frame_dir, f"{curr_fid:06d}.png")
            future_img_path = os.path.join(frame_dir, f"{future_fid:06d}.png")
            img_curr = cv2.imread(curr_img_path)
            img_future = cv2.imread(future_img_path)
            if img_curr is not None and img_future is not None:
                homography, bg_residual = orb_engine.estimate_homography(
                    img_curr, img_future, prev_bboxes=None
                )
            HOMOGRAPHY_CACHE[cache_key] = (homography, bg_residual)

    cx1, cy1, _, _ = centroids[curr_fid]
    cx2, cy2, _, _ = centroids[future_fid]

    # Synthetic jitter (+/- 2 pixels)
    j_cx2 = cx2 + np.random.uniform(-2.0, 2.0)
    j_cy2 = cy2 + np.random.uniform(-2.0, 2.0)

    # Raw centroid difference (normalized)
    raw_dx = (j_cx2 - cx1) / w * VELOCITY_SCALE
    raw_dy = (j_cy2 - cy1) / h * VELOCITY_SCALE

    # Per-object ego displacement (normalized)
    ego_dx, ego_dy = 0.0, 0.0
    if homography is not None:
        warped = warp_points(np.array([[cx1, cy1]], dtype=np.float32), homography)[0]
        ego_dx = (warped[0] - cx1) / w * VELOCITY_SCALE
        ego_dy = (warped[1] - cy1) / h * VELOCITY_SCALE

    # Residual = raw - ego (object-only motion)
    res_dx = raw_dx - ego_dx
    res_dy = raw_dy - ego_dy

    return res_dx, res_dy, bg_residual, ego_dx, ego_dy


def _generate_positive_pairs(
    track_centroids: Dict[int, Dict[int, Tuple[float, float, float, float]]],
    embedding: np.ndarray,
    expression_id: int,
    frame_gaps: List[int],
    frame_shape: Tuple[int, int],
    seq: str = None,
    frame_dir: str = None,
    orb_engine: Any = None,
    extra_features: List[str] = None,
    all_track_centroids_for_frame: Dict[int, Dict[int, Tuple]] = None,
    track_boundaries: List = None,
    ego_router_name: str = "orb",
) -> Tuple[List[np.ndarray], List[np.ndarray], List[int]]:
    """
    Generate positive (motion, language) pairs with residual velocity (raw - ego).

    Base 13D vector: [res_dx_s, res_dy_s, res_dx_m, res_dy_m, res_dx_l, res_dy_l,
                      dw, dh, cx, cy, w, h, snr]
    + optional extra features appended.
    """
    h, w = frame_shape
    motion_data = []
    language_data = []
    labels = []

    # Determine which extras are per-track vs relational
    per_track_feats = [f for f in (extra_features or [])
                       if f in ("speed_m", "heading_m", "accel", "ego_motion",
                                "accel_multiscale", "heading_sincos",
                                "ego_velocity_concat", "omf_stats")]
    relational_feats = [f for f in (extra_features or [])
                        if f in ("neighbor_mean_vel", "velocity_rank", "heading_diff",
                                 "nn_dist", "track_density")]
    needs_neighbors = len(relational_feats) > 0

    primary_gap_idx = 1  # mid-scale (gap=5) is the primary for dw/dh/snr

    # Pre-compute per-track mid-scale velocities for relational features
    # {frame_id: {track_id: (dx_m, dy_m, cx_n, cy_n)}}
    frame_track_data = {}

    needs_accel_multiscale = "accel_multiscale" in per_track_feats
    needs_omf_stats = "omf_stats" in per_track_feats

    for tid, centroids in track_centroids.items():
        track_start_idx = len(motion_data)
        track_frame_ids = []
        sorted_frames = sorted(centroids.keys())
        # Per-track velocity history: {frame_id: [(dx,dy) per scale]}
        track_vel_hist = {} if needs_accel_multiscale else None
        for i in range(len(sorted_frames)):
            curr_fid = sorted_frames[i]

            # Find future frames at each scale — warped velocity
            scale_velocities = []
            best_bg_residual = np.zeros(2, dtype=np.float32)
            ego_dx_m, ego_dy_m = 0.0, 0.0
            any_valid = False

            for gap_idx, gap in enumerate(frame_gaps):
                future_j = _find_future_frame(sorted_frames, i, gap)
                if future_j is not None:
                    future_fid = sorted_frames[future_j]
                    dx, dy, bg_res, e_dx, e_dy = _compute_velocity_at_gap(
                        centroids, curr_fid, future_fid, frame_shape,
                        seq, frame_dir, orb_engine,
                    )
                    scale_velocities.append((dx, dy))
                    any_valid = True
                    if gap_idx == primary_gap_idx:
                        best_bg_residual = bg_res
                        ego_dx_m, ego_dy_m = e_dx, e_dy
                else:
                    scale_velocities.append((0.0, 0.0))

            if not any_valid:
                continue

            accel_per_scale = None
            if needs_accel_multiscale:
                accel_per_scale = []
                for gap_idx, gap in enumerate(frame_gaps):
                    past_fid = _find_past_frame_id(track_vel_hist, curr_fid, gap)
                    if past_fid is None:
                        accel_per_scale.append((0.0, 0.0))
                    else:
                        past_dx, past_dy = track_vel_hist[past_fid][gap_idx]
                        now_dx, now_dy = scale_velocities[gap_idx]
                        accel_per_scale.append((
                            (now_dx - past_dx) / gap,
                            (now_dy - past_dy) / gap,
                        ))
                track_vel_hist[curr_fid] = list(scale_velocities)

            # Exp 37 Stage B: per-bbox OMF pooling × 3 scales (15D).
            # Loads precomputed Farneback flow field and pools over the
            # anchor-frame bbox. Missing flow → 5D zeros for that scale.
            omf_stats_per_scale = None
            if needs_omf_stats and seq is not None:
                cx_px, cy_px, bw_px, bh_px = centroids[curr_fid]
                bbox_px = (
                    int(round(cx_px - bw_px / 2.0)),
                    int(round(cy_px - bh_px / 2.0)),
                    int(round(bw_px)),
                    int(round(bh_px)),
                )
                omf_stats_per_scale = []
                for gap in frame_gaps:
                    flow = _load_omf_field(ego_router_name, seq, curr_fid, gap)
                    if flow is None:
                        omf_stats_per_scale.append(np.zeros(5, dtype=np.float32))
                    else:
                        omf_stats_per_scale.append(per_bbox_omf_stats(flow, bbox_px))

            # dw, dh from primary scale (mid)
            primary_future_j = _find_future_frame(sorted_frames, i, frame_gaps[primary_gap_idx])
            if primary_future_j is not None:
                future_fid = sorted_frames[primary_future_j]
                _, _, bw1, bh1 = centroids[curr_fid]
                _, _, bw2, bh2 = centroids[future_fid]
                j_bw2 = bw2 + np.random.uniform(-2.0, 2.0)
                j_bh2 = bh2 + np.random.uniform(-2.0, 2.0)
                dw = (j_bw2 - bw1) / w * VELOCITY_SCALE
                dh = (j_bh2 - bh1) / h * VELOCITY_SCALE
            else:
                dw, dh = 0.0, 0.0
                _, _, bw1, bh1 = centroids[curr_fid]

            # Spatial features from anchor frame
            cx1, cy1, bw1, bh1 = centroids[curr_fid]
            cx_n, cy_n = cx1 / w, cy1 / h
            bw_n, bh_n = bw1 / w, bh1 / h

            # SNR from primary (mid) scale warped speed
            mid_dx, mid_dy = scale_velocities[primary_gap_idx]
            obj_speed = np.sqrt(mid_dx ** 2 + mid_dy ** 2)
            bg_mag = np.sqrt(
                (best_bg_residual[0] / w * VELOCITY_SCALE) ** 2
                + (best_bg_residual[1] / h * VELOCITY_SCALE) ** 2
            )
            snr = obj_speed / (bg_mag + 1e-6)

            # Base 13D vector
            base_vec = [
                scale_velocities[0][0], scale_velocities[0][1],
                scale_velocities[1][0], scale_velocities[1][1],
                scale_velocities[2][0], scale_velocities[2][1],
                dw, dh, cx_n, cy_n, bw_n, bh_n, snr,
            ]

            # Per-track extras (F1-F4)
            base_vec.extend(compute_per_track_extras(
                per_track_feats, scale_velocities,
                ego_dx_m=ego_dx_m, ego_dy_m=ego_dy_m,
                accel_per_scale=accel_per_scale,
                omf_stats_per_scale=omf_stats_per_scale,
            ))

            # Store for relational feature computation
            if needs_neighbors:
                if curr_fid not in frame_track_data:
                    frame_track_data[curr_fid] = {}
                frame_track_data[curr_fid][tid] = (mid_dx, mid_dy, cx_n, cy_n)

            motion_vec = np.array(base_vec, dtype=np.float32)
            motion_data.append(motion_vec)
            language_data.append(embedding.copy())
            labels.append(expression_id)
            track_frame_ids.append(curr_fid)

        if track_boundaries is not None and track_frame_ids:
            track_boundaries.append({
                "tid": tid,
                "start": track_start_idx,
                "end": len(motion_data),
                "frame_ids": track_frame_ids,
            })

    # Post-pass: compute relational extras (F5-F9) if needed
    # Use all_track_centroids_for_frame (ALL tracks in sequence) for neighbor context,
    # not just the GT tracks from this expression.
    neighbor_source = all_track_centroids_for_frame or frame_track_data
    if needs_neighbors and neighbor_source:
        frame_diag = np.sqrt(1.0 + 1.0)  # normalized coords
        sample_idx = 0
        for tid, centroids in track_centroids.items():
            sorted_frames = sorted(centroids.keys())
            for i in range(len(sorted_frames)):
                curr_fid = sorted_frames[i]
                if curr_fid not in frame_track_data or tid not in frame_track_data.get(curr_fid, {}):
                    continue

                my_data = frame_track_data[curr_fid][tid]
                my_dx, my_dy, my_cx, my_cy = my_data

                # Gather neighbors from ALL tracks at this frame
                frame_neighbors = neighbor_source.get(curr_fid, {})
                neighbor_vels = []
                neighbor_centroids_list = []
                for other_tid, other_data in frame_neighbors.items():
                    if other_tid != tid:
                        neighbor_vels.append((other_data[0], other_data[1]))
                        neighbor_centroids_list.append((other_data[2], other_data[3]))

                rel_extras = compute_relational_extras(
                    relational_feats, my_dx, my_dy, my_cx, my_cy,
                    neighbor_vels, neighbor_centroids_list, frame_diag,
                )

                # Append to the existing motion vector
                old_vec = motion_data[sample_idx]
                motion_data[sample_idx] = np.concatenate([old_vec, np.array(rel_extras, dtype=np.float32)])
                sample_idx += 1

    return motion_data, language_data, labels


def _vectors_to_sequences(
    motion_data: List[np.ndarray],
    language_data: List[np.ndarray],
    labels: List[int],
    track_boundaries: List[dict],
    seq_len: int = 10,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[int]]:
    """
    Convert per-frame motion vectors into sliding-window sequences.

    Args:
        motion_data: flat list of (D,) vectors, ordered by track then frame
        language_data: parallel list of (384,) embeddings
        labels: parallel list of expression IDs
        track_boundaries: list of {tid, start, end, frame_ids} from _generate_positive_pairs
        seq_len: window length T

    Returns:
        seq_motion: list of (seq_len, D) arrays
        seq_masks: list of (seq_len+1,) bool arrays (True=padded, False=valid; +1 for CLS)
        seq_language: list of (384,) arrays
        seq_labels: list of int
    """
    dim = motion_data[0].shape[0] if motion_data else 13
    seq_motion = []
    seq_masks = []
    seq_language = []
    seq_labels = []

    for tb in track_boundaries:
        start, end = tb["start"], tb["end"]
        n_frames = end - start
        if n_frames == 0:
            continue

        vectors = motion_data[start:end]
        embedding = language_data[start]
        label = labels[start]

        for win_end in range(1, n_frames + 1):
            win_start = max(0, win_end - seq_len)
            window = vectors[win_start:win_end]
            n_valid = len(window)

            if n_valid < seq_len:
                pad = [np.zeros(dim, dtype=np.float32)] * (seq_len - n_valid)
                window = pad + list(window)

            seq = np.stack(window)

            mask = np.zeros(seq_len + 1, dtype=bool)
            n_padded = seq_len - n_valid
            if n_padded > 0:
                mask[1 : n_padded + 1] = True

            seq_motion.append(seq)
            seq_masks.append(mask)
            seq_language.append(embedding.copy())
            seq_labels.append(label)

    return seq_motion, seq_masks, seq_language, seq_labels


def build_training_data(
    data_root: str,
    sequences: List[str],
    text_encoder: Any,
    frame_gaps: List[int] = None,
    frame_shape: Tuple[int, int] = (375, 1242),
    use_group_labels: bool = False,
    extra_features: List[str] = None,
    seq_len: int = 0,
    ego_router_name: str = "orb",
) -> Tuple:
    """
    Build (motion, language, expression_id) training triples for contrastive learning.

    Each unique expression gets an integer ID. All motion vectors that correspond
    to the same expression share that ID, so SupervisedInfoNCE treats them as
    positives and everything else as in-batch negatives.

    Uses multi-scale frame gaps for temporal velocity features (16D vectors with ego-motion).
    """
    if frame_gaps is None:
        frame_gaps = FRAME_GAPS

    # ── Disk cache check (skip ORB + velocity rebuild for identical config) ──
    use_cache = os.environ.get("GMCLINK_NO_CACHE", "0") != "1"
    encoder_name = getattr(text_encoder, "model_name", "all-MiniLM-L6-v2")
    cache_key, key_obj = _build_cache_key(
        data_root, sequences, frame_gaps, frame_shape,
        use_group_labels, extra_features, seq_len,
        text_encoder_name=encoder_name,
        ego_router_name=ego_router_name,
    )
    if use_cache:
        cached = _try_load_cache(cache_key, seq_len)
        if cached is not None:
            print(f"  [cache] HIT key={cache_key} — skipping ORB+velocity build")
            return cached
        print(f"  [cache] MISS key={cache_key} — building from scratch")

    all_expressions, sentence_embeddings, all_sentences = _collect_expressions(
        data_root, sequences, text_encoder
    )

    print(f"  Initializing '{ego_router_name}' ego router for dataset ego-motion compensation...")
    orb_engine = make_ego_router(ego_router_name)

    print(
        "  Computing centroid-difference velocities and generating positive pairs..."
    )

    # Map each unique sentence to a stable integer class ID
    sentence_to_id = {s: i for i, s in enumerate(all_sentences)}

    motion_data = []
    language_data = []
    labels = []

    # Filter to motion-relevant expressions only — appearance-only sentences
    # like "red cars" add noise since motion vectors can't encode color/shape
    motion_expressions = [e for e in all_expressions if is_motion_expression(e["sentence"])]
    print(f"  Motion-filtered: {len(motion_expressions)}/{len(all_expressions)} expressions")

    if use_group_labels:
        group_counts = {}
        for expr in motion_expressions:
            gid = motion_type_group(expr["sentence"])
            group_counts[gid] = group_counts.get(gid, 0) + 1
        print(f"  Motion-type groups: {len(group_counts)} — "
              + ", ".join(f"{MOTION_TYPE_NAMES.get(g, '?')}={n}" for g, n in sorted(group_counts.items())))

    # Check if relational features need all-track neighbor context
    relational_feats = [f for f in (extra_features or [])
                        if f in ("neighbor_mean_vel", "velocity_rank", "heading_diff",
                                 "nn_dist", "track_density")]
    needs_all_tracks = len(relational_feats) > 0

    # Cache per-sequence: frame shape, frame dir, and all-track data
    seq_cache = {}  # {seq: (actual_shape, frame_dir, all_frame_track_data)}

    all_track_boundaries = [] if seq_len > 0 else None

    # Process expressions grouped by seq so the OMF flow LRU stays hot.
    motion_expressions.sort(key=lambda e: e["seq"])
    _prev_seq = None

    for expr in motion_expressions:
        seq = expr["seq"]
        # Drop flow cache when crossing a seq boundary — flows from earlier seqs
        # won't be reused and we want to bound peak RSS for large seqs.
        if _prev_seq is not None and seq != _prev_seq:
            _omf_cache_reset()
        _prev_seq = seq
        label_map = expr["label"]
        sentence = expr["sentence"]
        embedding = expr["embedding"]

        if use_group_labels:
            expression_id = motion_type_group(sentence)
            if expression_id is None:
                continue
        else:
            expression_id = sentence_to_id[sentence]

        # Use cached sequence data or compute it
        if seq not in seq_cache:
            frame_dir = os.path.join(data_root, "KITTI", "training", "image_02", seq)
            actual_shape = frame_shape
            if os.path.exists(frame_dir):
                sample_frames = [f for f in os.listdir(frame_dir) if f.endswith(".png")]
                if sample_frames:
                    sample = cv2.imread(os.path.join(frame_dir, sample_frames[0]))
                    if sample is not None:
                        actual_shape = (sample.shape[0], sample.shape[1])

            all_frame_track_data = None
            if needs_all_tracks:
                print(f"  Pre-computing all-track neighbor data for seq {seq}...")
                all_centroids = _extract_all_track_centroids(
                    data_root, seq, frame_shape=actual_shape
                )
                all_frame_track_data = _precompute_frame_track_data(
                    all_centroids, actual_shape, frame_gaps, seq, frame_dir, orb_engine,
                )
                print(f"    {len(all_centroids)} tracks, "
                      f"{sum(len(v) for v in all_frame_track_data.values())} track-frame entries")

            seq_cache[seq] = (actual_shape, frame_dir, all_frame_track_data)

        actual_shape, frame_dir, all_frame_track_data = seq_cache[seq]

        track_centroids = _extract_target_centroids(
            data_root, seq, label_map, frame_shape=actual_shape
        )

        # Generate only positive pairs — negatives are formed in-batch by InfoNCE
        per_expr_boundaries = [] if seq_len > 0 else None
        m_data, l_data, lbls = _generate_positive_pairs(
            track_centroids,
            embedding,
            expression_id,
            frame_gaps,
            actual_shape,
            seq=seq,
            frame_dir=frame_dir,
            orb_engine=orb_engine,
            extra_features=extra_features,
            all_track_centroids_for_frame=all_frame_track_data,
            track_boundaries=per_expr_boundaries,
            ego_router_name=ego_router_name,
        )

        # Offset boundaries to global indices before extending
        if seq_len > 0 and per_expr_boundaries:
            offset = len(motion_data)
            for tb in per_expr_boundaries:
                tb["start"] += offset
                tb["end"] += offset
            all_track_boundaries.extend(per_expr_boundaries)

        motion_data.extend(m_data)
        language_data.extend(l_data)
        labels.extend(lbls)

    if use_group_labels:
        n_classes = len(set(labels))
        print(f"  Motion-type groups used: {n_classes} | Total samples: {len(labels)}")
    else:
        n_classes = len(sentence_to_id)
        print(f"  Unique expressions: {n_classes} | Total samples: {len(labels)}")
    if extra_features:
        extra_dims = compute_extra_dims(extra_features)
        print(f"  Extra features: {extra_features} (+{extra_dims}D → {13 + extra_dims}D)")

    # Convert to sequences if requested
    if seq_len > 0:
        print(f"  Converting to sequences (seq_len={seq_len})...")
        seq_motion, seq_masks, seq_language, seq_labels = _vectors_to_sequences(
            motion_data, language_data, labels, all_track_boundaries, seq_len=seq_len,
        )
        print(f"  Sequences: {len(seq_motion)} (from {len(motion_data)} individual vectors)")
        if use_cache:
            _save_cache(cache_key, key_obj,
                        (seq_motion, seq_masks, seq_language, seq_labels), seq_len)
        return seq_motion, seq_masks, seq_language, seq_labels

    if use_cache:
        _save_cache(cache_key, key_obj,
                    (motion_data, language_data, labels), seq_len)
    return motion_data, language_data, labels