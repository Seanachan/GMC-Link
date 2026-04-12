"""
Stage 2: Learned Fusion Head
=============================
Train a small MLP on [ikun_logit, gmc_score, is_motion_flag] → P(match)
to replace hand-tuned OR-logic thresholds.

Usage:
    python gmc_link/fusion_head.py --collect   # collect training data
    python gmc_link/fusion_head.py --train     # train fusion head
    python gmc_link/fusion_head.py --eval      # evaluate on val split
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from typing import Dict, List, Tuple

import cv2

from gmc_link.manager import GMCLinkManager
from gmc_link.text_utils import TextEncoder
from gmc_link.dataset import load_labels_with_ids
from gmc_link.demo_inference import (
    load_neuralsort_tracks,
    load_ikun_scores,
    DummyTrack,
    match_tracks_to_gt,
    normalized_to_pixel,
    classify_expression,
)


# ── Model ────────────────────────────────────────────────────────────


class FusionHead(nn.Module):
    """Tiny MLP: [ikun_logit, gmc_score, is_motion] → P(match)."""

    def __init__(self, input_dim: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def predict_prob(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))


# ── Data Collection ──────────────────────────────────────────────────


def collect_training_data(
    sequences: list = None,
    data_root: str = "refer-kitti",
    results_json: str = "iKUN/results.json",
    track_dir: str = "NeuralSORT",
    weights_path: str = "gmc_link_weights.pth",
    output_path: str = "gmc_link/fusion_train_data.npz",
) -> None:
    """
    Collect (ikun_logit, gmc_score, is_motion, label, frame_idx) for all expressions.
    Supports multiple sequences — frame_idx is made globally unique across sequences.
    """
    if sequences is None:
        sequences = ["0011"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = TextEncoder(device=device)

    all_samples = []  # (ikun_logit, gmc_score, is_motion, label, global_frame_idx)
    global_frame_offset = 0

    for sequence in sequences:
        print(f"\n── Sequence {sequence} ──")

        # Load NeuralSORT tracks (merge car + pedestrian)
        car_path = os.path.join(track_dir, sequence, "car", "predict.txt")
        ped_path = os.path.join(track_dir, sequence, "pedestrian", "predict.txt")
        ns_tracks = load_neuralsort_tracks(car_path)

        # Merge pedestrian tracks with offset IDs
        max_car_id = 0
        for dets in ns_tracks.values():
            for oid, *_ in dets:
                max_car_id = max(max_car_id, oid)
        if os.path.exists(ped_path):
            ped_data = np.loadtxt(ped_path, delimiter=",")
            if ped_data.ndim == 1 and ped_data.size > 0:
                ped_data = ped_data[np.newaxis]
            if ped_data.ndim == 2 and ped_data.size > 0:
                for row in ped_data:
                    fid = int(row[0])
                    ns_tracks[fid].append((int(row[1]) + max_car_id, row[2], row[3], row[4], row[5]))

        labels_dir = os.path.join(data_root, "KITTI", "labels_with_ids", "image_02", sequence)
        gt_labels = load_labels_with_ids(labels_dir)
        img_h, img_w = 375, 1242

        frame_dir = os.path.join(data_root, "KITTI", "training", "image_02", sequence)
        frame_files = sorted(f for f in os.listdir(frame_dir) if f.endswith((".png", ".jpg")))
        total_frames = len(frame_files)

        expr_dir = os.path.join(data_root, "expression", sequence)
        expr_files = sorted(glob.glob(os.path.join(expr_dir, "*.json")))

        for expr_file in expr_files:
            expr_name = os.path.splitext(os.path.basename(expr_file))[0]

            with open(expr_file, "r") as f:
                expr_json = json.load(f)
            sentence = expr_json["sentence"]
            gt_label_map = expr_json["label"]
            expr_type = classify_expression(sentence)
            is_motion = 1.0 if expr_type == "motion" else (0.5 if expr_type == "stationary" else 0.0)

            ikun_scores = load_ikun_scores(results_json, sequence, expr_name)

            linker = GMCLinkManager(weights_path=weights_path, device=device, lang_dim=384)
            lang_emb = encoder.encode(sentence)

            print(f"  Collecting: {expr_name} [{expr_type}]")

            for frame_0idx in range(total_frames):
                frame_1idx = frame_0idx + 1
                frame_id_str = str(frame_0idx)

                if frame_1idx not in ns_tracks:
                    continue

                frame_path = os.path.join(frame_dir, frame_files[frame_0idx])
                frame_img = cv2.imread(frame_path)
                if frame_img is None:
                    continue

                detections = ns_tracks[frame_1idx]
                active_tracks = []
                track_boxes_xyxy: Dict[int, List[float]] = {}

                for obj_id, x, y, w, h in detections:
                    active_tracks.append(DummyTrack(obj_id, x, y, w, h))
                    track_boxes_xyxy[obj_id] = [x, y, x + w, y + h]

                det_array = (
                    np.array([[x, y, x + w, y + h] for _, x, y, w, h in detections])
                    if detections else None
                )

                gmc_scores, _ = linker.process_frame(
                    frame_img, active_tracks, lang_emb, detections=det_array
                )

                gt_track_ids = gt_label_map.get(frame_id_str, [])
                if gt_track_ids and frame_0idx in gt_labels:
                    gt_bboxes = []
                    for det in gt_labels[frame_0idx]:
                        if det["track_id"] in gt_track_ids:
                            bbox = normalized_to_pixel(
                                det["x1_n"], det["y1_n"], det["w_n"], det["h_n"], img_w, img_h
                            )
                            gt_bboxes.append(bbox)
                    matched_ids = match_tracks_to_gt(track_boxes_xyxy, gt_bboxes, iou_threshold=0.3)
                else:
                    matched_ids = set()

                for obj_id in track_boxes_xyxy:
                    ikun_logit = ikun_scores.get(frame_1idx, {}).get(obj_id, None)
                    if ikun_logit is None:
                        continue
                    gmc_prob = gmc_scores.get(obj_id, 0.0)
                    label = 1.0 if obj_id in matched_ids else 0.0

                    all_samples.append((ikun_logit, gmc_prob, is_motion, label,
                                        global_frame_offset + frame_0idx))

        global_frame_offset += total_frames

    samples = np.array(all_samples, dtype=np.float32)
    np.savez_compressed(output_path, samples=samples)
    n_pos = int(samples[:, 3].sum())
    n_neg = len(samples) - n_pos
    print(f"\nCollected {len(samples)} samples ({n_pos} pos, {n_neg} neg)")
    print(f"Saved to {output_path}")


# ── Dataset ──────────────────────────────────────────────────────────


class FusionDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# ── Training ─────────────────────────────────────────────────────────


def train_fusion_head(
    data_path: str = "gmc_link/fusion_train_data.npz",
    output_path: str = "gmc_link/fusion_head_weights.pth",
    train_frac: float = 0.7,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    neg_sample_ratio: float = 3.0,
) -> None:
    """Train FusionHead on collected data with frame-based train/val split."""
    data = np.load(data_path)
    samples = data["samples"]  # (N, 5): ikun, gmc, is_motion, label, frame_idx

    # Frame-based split
    total_frames = int(samples[:, 4].max()) + 1
    split_frame = int(total_frames * train_frac)
    train_mask = samples[:, 4] < split_frame
    val_mask = ~train_mask

    train_data = samples[train_mask]
    val_data = samples[val_mask]

    # Downsample negatives for training (balance)
    train_pos = train_data[train_data[:, 3] == 1.0]
    train_neg = train_data[train_data[:, 3] == 0.0]
    n_neg_keep = min(len(train_neg), int(len(train_pos) * neg_sample_ratio))
    rng = np.random.default_rng(42)
    neg_idx = rng.choice(len(train_neg), n_neg_keep, replace=False)
    train_balanced = np.concatenate([train_pos, train_neg[neg_idx]], axis=0)
    rng.shuffle(train_balanced)

    print(f"Train: {len(train_balanced)} ({int(train_balanced[:, 3].sum())} pos, "
          f"{len(train_balanced) - int(train_balanced[:, 3].sum())} neg)")
    print(f"Val:   {len(val_data)} ({int(val_data[:, 3].sum())} pos, "
          f"{len(val_data) - int(val_data[:, 3].sum())} neg)")

    train_ds = FusionDataset(train_balanced[:, :3], train_balanced[:, 3])
    val_ds = FusionDataset(val_data[:, :3], val_data[:, 3])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 4)

    # Compute class weight for BCE
    pos_weight = torch.tensor(
        [len(train_balanced) / (2 * max(1, int(train_balanced[:, 3].sum())))]
    )

    model = FusionHead(input_dim=3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_f1 = 0.0
    best_state = None

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        for feats, labels in train_loader:
            logits = model(feats)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(labels)
        train_loss /= len(train_ds)
        scheduler.step()

        # Validate
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for feats, labels in val_loader:
                probs = model.predict_prob(feats)
                val_preds.extend(probs.numpy().tolist())
                val_labels.extend(labels.numpy().tolist())

        val_preds = np.array(val_preds)
        val_labels = np.array(val_labels)

        # Find best threshold
        best_thr_f1 = 0.0
        best_thr = 0.5
        for thr in np.arange(0.3, 0.8, 0.02):
            preds = (val_preds >= thr).astype(float)
            tp = ((preds == 1) & (val_labels == 1)).sum()
            fp = ((preds == 1) & (val_labels == 0)).sum()
            fn = ((preds == 0) & (val_labels == 1)).sum()
            prec = tp / (tp + fp) if tp + fp > 0 else 0
            rec = tp / (tp + fn) if tp + fn > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0
            if f1 > best_thr_f1:
                best_thr_f1 = f1
                best_thr = thr

        if epoch % 10 == 0 or best_thr_f1 > best_val_f1:
            print(f"  Epoch {epoch:3d}  loss={train_loss:.4f}  val_F1={best_thr_f1:.4f}  thr={best_thr:.2f}")

        if best_thr_f1 > best_val_f1:
            best_val_f1 = best_thr_f1
            best_state = {
                "model": model.state_dict(),
                "threshold": float(best_thr),
                "val_f1": float(best_val_f1),
            }

    print(f"\nBest val F1: {best_val_f1:.4f}  threshold: {best_state['threshold']:.2f}")
    torch.save(best_state, output_path)
    print(f"Saved to {output_path}")

    # Final detailed eval
    model.load_state_dict(best_state["model"])
    _evaluate_model(model, val_data, best_state["threshold"])


def _evaluate_model(model: FusionHead, data: np.ndarray, threshold: float) -> None:
    """Print per-type evaluation of the fusion head."""
    model.eval()
    features = torch.tensor(data[:, :3], dtype=torch.float32)
    labels = data[:, 3]

    with torch.no_grad():
        probs = model.predict_prob(features).numpy()

    preds = (probs >= threshold).astype(float)

    # Overall
    tp = ((preds == 1) & (labels == 1)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()
    prec = tp / (tp + fp) if tp + fp > 0 else 0
    rec = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0

    print(f"\n{'='*50}")
    print(f"FUSION HEAD EVALUATION (threshold={threshold:.2f})")
    print(f"{'='*50}")
    print(f"Overall:  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}")

    # Per expression type (motion=1.0, stationary=0.5, appearance=0.0)
    for etype, val in [("motion", 1.0), ("stationary", 0.5), ("appearance", 0.0)]:
        mask = data[:, 2] == val
        if not mask.any():
            continue
        e_preds = preds[mask]
        e_labels = labels[mask]
        e_tp = ((e_preds == 1) & (e_labels == 1)).sum()
        e_fp = ((e_preds == 1) & (e_labels == 0)).sum()
        e_fn = ((e_preds == 0) & (e_labels == 1)).sum()
        e_prec = e_tp / (e_tp + e_fp) if e_tp + e_fp > 0 else 0
        e_rec = e_tp / (e_tp + e_fn) if e_tp + e_fn > 0 else 0
        e_f1 = 2 * e_prec * e_rec / (e_prec + e_rec) if e_prec + e_rec > 0 else 0
        print(f"  {etype:<12} Prec={e_prec:.4f}  Rec={e_rec:.4f}  F1={e_f1:.4f}  (n={mask.sum()})")
    print(f"{'='*50}")


# ── Integration with demo_inference ──────────────────────────────────


def load_fusion_head(
    weights_path: str = "gmc_link/fusion_head_weights.pth",
) -> Tuple[FusionHead, float]:
    """Load trained fusion head and its threshold."""
    state = torch.load(weights_path, map_location="cpu", weights_only=False)
    model = FusionHead(input_dim=3)
    model.load_state_dict(state["model"])
    model.eval()
    return model, state["threshold"]


if __name__ == "__main__":
    v1_mode = "--v1" in sys.argv

    if v1_mode:
        # V1 clean split: train on 0005+0013, evaluate on 0011
        data_path = "gmc_link/fusion_train_data_v1.npz"
        weights_out = "gmc_link/fusion_head_weights_v1.pth"
        collect_kwargs = dict(
            sequences=["0005", "0013"],
            data_root="refer-kitti",
            results_json="iKUN/ikun_results_v1.json",
            track_dir="NeuralSORT",
            weights_path="gmc_link_weights_v1train.pth",
            output_path=data_path,
        )
        print("[V1 mode] Train on 0005+0013, eval on 0011")
    else:
        data_path = "gmc_link/fusion_train_data.npz"
        weights_out = "gmc_link/fusion_head_weights.pth"
        collect_kwargs = dict(
            sequences=["0011"],
            data_root="refer-kitti",
            results_json="iKUN/results.json",
            output_path=data_path,
        )

    if "--collect" in sys.argv:
        print("Collecting training data...")
        collect_training_data(**collect_kwargs)
    elif "--train" in sys.argv:
        print("Training fusion head...")
        train_fusion_head(data_path=data_path, output_path=weights_out)
    elif "--eval" in sys.argv:
        print("Evaluating fusion head...")
        data = np.load(data_path)["samples"]
        total_frames = int(data[:, 4].max()) + 1
        split_frame = int(total_frames * 0.7)
        val_data = data[data[:, 4] >= split_frame]
        state = torch.load(weights_out, map_location="cpu")
        model = FusionHead(input_dim=3)
        model.load_state_dict(state["model"])
        _evaluate_model(model, val_data, state["threshold"])
    else:
        print("Usage: python gmc_link/fusion_head.py [--v1] --collect|--train|--eval")
