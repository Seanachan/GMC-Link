import torch
from torch.utils.data import Dataset
import numpy as np
import json
import os
import random
from gmc_link.utils import VELOCITY_SCALE


class MotionLanguageDataset(Dataset):
    """
    PyTorch Dataset for BCE-based motion-language training.
    Each sample: (motion_vector, language_embedding, label)
    label = 1.0 for positive match, 0.0 for negative.
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
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return motion, lang, label


def collate_fn(batch):
    """Stack into (Batch, 2) motion, (Batch, L_dim) language, and (Batch,) labels."""
    motion_batch = torch.stack([item[0] for item in batch], dim=0)
    language_batch = torch.stack([item[1] for item in batch], dim=0)
    label_batch = torch.stack([item[2] for item in batch], dim=0)
    return motion_batch, language_batch, label_batch


# ── Data Loaders ─────────────────────────────────────────────────────

def load_refer_kitti_expressions(expression_dir):
    """Load all expression JSON files from a sequence's expression directory."""
    expressions = []
    for json_file in sorted(os.listdir(expression_dir)):
        if not json_file.endswith('.json'):
            continue
        with open(os.path.join(expression_dir, json_file), 'r') as f:
            expr = json.load(f)
        expressions.append(expr)
    return expressions


def load_labels_with_ids(labels_dir):
    """
    Load per-frame label files from labels_with_ids directory.
    Format: class_id track_id cx cy w h (normalized coordinates)
    """
    labels = {}
    for txt_file in sorted(os.listdir(labels_dir)):
        if not txt_file.endswith('.txt'):
            continue
        frame_idx = int(os.path.splitext(txt_file)[0])
        frame_labels = []
        with open(os.path.join(labels_dir, txt_file), 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                frame_labels.append({
                    'class_id': int(parts[0]),
                    'track_id': int(parts[1]),
                    'cx': float(parts[2]),
                    'cy': float(parts[3]),
                    'w': float(parts[4]),
                    'h': float(parts[5]),
                })
        labels[frame_idx] = frame_labels
    return labels


def load_kitti_tracking_labels(label_file):
    """
    Load KITTI tracking labels (label_02/XXXX.txt).
    Format: frame_id track_id type truncated occluded alpha x1 y1 x2 y2 h w l x y z ry
    """
    frames = {}
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 17:
                continue
            
            frame_id = int(parts[0])
            track_id = int(parts[1])
            obj_type = parts[2]
            
            if obj_type == 'DontCare' or track_id < 0:
                continue
            
            x1, y1, x2, y2 = float(parts[6]), float(parts[7]), float(parts[8]), float(parts[9])
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            
            if frame_id not in frames:
                frames[frame_id] = {}
            frames[frame_id][track_id] = {
                'type': obj_type,
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'cx': cx, 'cy': cy,
            }
    return frames


# ── Training Data Builder ────────────────────────────────────────────

# Keywords that indicate motion-related expressions learnable from velocity vectors
MOTION_KEYWORDS = [
    'moving', 'parking', 'parked', 'turning', 'braking', 'slower', 'faster',
    'counter direction', 'same direction', 'in front of', 'horizon direction',
    'following', 'approaching', 'overtaking', 'stopping',
]

def is_motion_expression(sentence):
    """Check if a sentence describes motion (learnable from velocity vectors)."""
    lower = sentence.lower()
    return any(kw in lower for kw in MOTION_KEYWORDS)


def build_training_data(data_root, sequences, text_encoder, frame_gap=5, frame_shape=(375, 1242)):
    """
    Build (motion, language) training pairs from refer-kitti.
    
    Only positive pairs are generated. The CLIP-style contrastive loss
    naturally treats all other in-batch items as negatives via the
    off-diagonal, so explicit negative sampling is NOT needed.
    
    Key design:
    1. Multi-frame velocity with `frame_gap` for larger, discriminative vectors
    2. KITTI tracking labels (pixel coords) normalized to match inference
    3. All 4 expression sequences for maximum sentence diversity
    """
    h, w = frame_shape
    
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
            sentence = expr['sentence']
            if not is_motion_expression(sentence):
                skipped += 1
                continue
            if sentence not in sentence_embeddings:
                emb = text_encoder.encode(sentence).squeeze(0).cpu().numpy()
                sentence_embeddings[sentence] = emb
            all_expressions.append({
                'sentence': sentence,
                'embedding': sentence_embeddings[sentence],
                'label': expr['label'],
                'seq': seq,
            })
        if skipped:
            print(f"  {seq}: skipped {skipped} non-motion expressions")
    
    print(f"  Motion-related sentences: {len(sentence_embeddings)}")
    print(f"  Total expressions: {len(all_expressions)}")
    all_sentences = list(sentence_embeddings.keys())
    
    print("  Computing multi-frame velocities...")
    motion_data = []
    language_data = []
    labels = []
    
    for expr in all_expressions:
        seq = expr['seq']
        label_map = expr['label']
        embedding = expr['embedding']
        
        tracking_file = os.path.join(data_root, "KITTI", "tracking", "training", "label_02", f"{seq}.txt")
        if not os.path.exists(tracking_file):
            continue
        
        tracking_data = load_kitti_tracking_labels(tracking_file)
        frame_ids = sorted([int(fid) for fid in label_map.keys()])
        
        # Collect centroids for target tracks
        track_centroids = {}
        for fid in frame_ids:
            target_tracks = set(label_map[str(fid)])
            if fid not in tracking_data:
                continue
            for tid in target_tracks:
                if tid in tracking_data[fid]:
                    det = tracking_data[fid][tid]
                    if tid not in track_centroids:
                        track_centroids[tid] = {}
                    track_centroids[tid][fid] = (det['cx'], det['cy'])
        
        # Build samples with multi-frame gap
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
                
                cx1, cy1 = centroids[curr_fid]
                cx2, cy2 = centroids[future_fid]
                
                dx = (cx2 - cx1) / w * VELOCITY_SCALE
                dy = (cy2 - cy1) / h * VELOCITY_SCALE
                
                motion_data.append(np.array([dx, dy], dtype=np.float32))
                language_data.append(embedding.copy())
                labels.append(1.0)  # Positive pair
                
                # Generate negative pairs (same motion, wrong sentence)
                for _ in range(3):
                    wrong_sentence = random.choice(all_sentences)
                    while wrong_sentence == sentence and len(all_sentences) > 1:
                        wrong_sentence = random.choice(all_sentences)
                    motion_data.append(np.array([dx, dy], dtype=np.float32))
                    language_data.append(sentence_embeddings[wrong_sentence].copy())
                    labels.append(0.0)  # Negative pair
    
    pos_count = sum(1 for l in labels if l == 1.0)
    neg_count = sum(1 for l in labels if l == 0.0)
    print(f"  Positive: {pos_count} | Negative: {neg_count} | Total: {len(labels)}")
    return motion_data, language_data, labels
