"""Math kernel + data loaders for the ego-motion information-cap ladder.

Four velocity levels per (track, frame) instance, all in per-frame units:
  L1 raw pixel       : ||v_pix||                         (px/frame)
  L2 pixel x Z/f     : ||v_pix . Z/f||  (NO ego comp)     (m/frame)
  L3 ego-comp pixel  : ||v_pix - ego_pix||                (px/frame)
  L4 metric residual : ||(vres . Z/f, dZ - ego_dZ)||      (m/frame)

Ego is estimated WITHOUT homography (no oxts available): component-wise median
over the slower-half cohort of tracks in the same frame (robust stand-in for the
ship's post-homography stationary cohort, manager.py:514).

Magnitudes are scale-confounded at L2/L4 (the x Z/f rescale was shown to be a
model no-op, world-XY NEG) -- so AUC (rank-based, scale-invariant) is the
inferential metric; magnitudes are descriptive only.
"""
from __future__ import annotations

import glob
import json
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Math kernel ──────────────────────────────────────────────────────────


def estimate_ego(
    vels: Dict[int, Tuple[float, float]],
    dzs: Dict[int, float],
) -> Tuple[float, float, float]:
    """Robust ego estimate = component-wise median over the slower-half cohort.

    Cohort = tracks whose raw pixel speed <= the frame's median speed. The median
    of their velocity vectors and dZ approximates ego-induced motion (most tracks
    are ground-static, so their apparent motion is ego). Returns (ego_vx, ego_vy,
    ego_dz). With one track, that track defines ego (residual collapses to 0).
    """
    tids = [t for t in vels if t in dzs]
    if not tids:
        return (0.0, 0.0, 0.0)
    speeds = {t: math.hypot(vels[t][0], vels[t][1]) for t in tids}
    med_speed = float(np.median([speeds[t] for t in tids]))
    cohort = [t for t in tids if speeds[t] <= med_speed]
    if not cohort:
        cohort = tids
    ego_vx = float(np.median([vels[t][0] for t in cohort]))
    ego_vy = float(np.median([vels[t][1] for t in cohort]))
    ego_dz = float(np.median([dzs[t] for t in cohort]))
    return (ego_vx, ego_vy, ego_dz)


def compute_levels(
    v_pix: Tuple[float, float],
    z_t: float,
    dz: float,
    ego: Tuple[float, float, float],
    fx: float,
    fy: float,
) -> Dict[str, float]:
    """Compute the 4 ladder magnitudes + metric components for one instance."""
    vx, vy = v_pix
    ego_vx, ego_vy, ego_dz = ego

    l1 = math.hypot(vx, vy)
    l2 = math.hypot(vx * z_t / fx, vy * z_t / fy)

    res_x, res_y = vx - ego_vx, vy - ego_vy
    l3 = math.hypot(res_x, res_y)

    dz_res = dz - ego_dz
    vm_x = res_x * z_t / fx
    vm_y = res_y * z_t / fy
    vm_z = dz_res
    l4 = math.sqrt(vm_x * vm_x + vm_y * vm_y + vm_z * vm_z)

    return {
        "L1": l1, "L2": l2, "L3": l3, "L4": l4,
        "v_metric_x": vm_x, "v_metric_y": vm_y, "v_metric_z": vm_z,
        "dz_res": dz_res,
    }


def is_sdf(
    dz_res: float,
    v_metric_x: float,
    v_metric_z: float,
    tau_z: float,
    ratio: float = 0.3,
    eps: float = 1e-3,
) -> bool:
    """Same-Direction-Faster flag (kinematic, no class filter).

    Qualifies iff: pulling away in depth (dZ_res > tau_z) AND motion is mostly
    longitudinal (|v_metric_x| / |v_metric_z| < ratio). Near-zero longitudinal
    velocity is excluded to avoid division noise.
    """
    if abs(v_metric_z) < eps:
        return False
    if dz_res <= tau_z:
        return False
    if abs(v_metric_x) / abs(v_metric_z) >= ratio:
        return False
    return True


def safe_auc(labels: List[int], scores: List[float]) -> Optional[float]:
    """ROC-AUC, or None if degenerate (empty or single-class)."""
    if len(labels) == 0 or len(set(labels)) < 2:
        return None
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(labels, scores))


# ── Motion-state label + subtype (expression semantics) ───────────────────

# Velocity-asserting: the referent is described as in motion.
VELOCITY_KEYWORDS = [
    "moving", "move", "driving", "drive", "travel", "faster", "speedier",
    "accelerat", "decelerat", "overtak", "walking", "walk", "running", "run",
    "heading", "following", "follow", "approaching", "approach",
    "same direction", "opposite direction", "counter direction",
    "contrary direction", "reverse direction", "turning", "turn", "braking",
    "brake", "slowing", "slower",
]
# Rest-asserting: the referent is described as at rest.
REST_KEYWORDS = [
    "parked", "parking", "stopped", "stopping", "stationary", "at rest",
    "still", "halted", "idle",
]


def motion_state(sentence: str) -> Optional[str]:
    """Expression-semantic motion label: 'MOVING' / 'STATIC' / None (excluded).

    Rest keywords win when both appear is impossible by design, so conflicts
    (both present) and neither-present both return None (unlabelled).
    """
    s = sentence.lower()
    has_rest = any(k in s for k in REST_KEYWORDS)
    has_vel = any(k in s for k in VELOCITY_KEYWORDS)
    if has_rest and not has_vel:
        return "STATIC"
    if has_vel and not has_rest:
        return "MOVING"
    return None  # neither, or ambiguous both


def keyword_subtype(sentence: str) -> str:
    """Descriptive subtype bucket (priority-ordered, partitions the subset)."""
    s = sentence.lower()
    if "turn" in s:
        return "turning"
    if "brak" in s:
        return "braking"
    if "walk" in s:
        return "walking"
    if any(k in s for k in ["moving", "move", "driving", "drive", "faster",
                            "speedier", "accelerat", "overtak", "direction",
                            "heading", "travel"]):
        return "moving"
    return "other-motion"


# ── Data loaders ──────────────────────────────────────────────────────────

V1_EXPR_DIR = "/home/seanachan/data/Dataset/refer-kitti/expression"
LABELS_DIR = "/home/seanachan/data/Dataset/refer-kitti/KITTI/labels_with_ids/image_02"
FRAME_DIR = "/home/seanachan/data/Dataset/refer-kitti/KITTI/training/image_02"
DEPTH_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "gmc_link", "depth_cache")
INTRINSICS = (721.5377, 721.5377, 609.5593, 172.8540)  # canonical KITTI 2011_09_26


def image_size(seq: str) -> Tuple[int, int]:
    """(W, H) from the first frame; falls back to canonical 1242x375."""
    fs = sorted(glob.glob(os.path.join(FRAME_DIR, seq, "*.png")))
    if fs:
        import cv2
        im = cv2.imread(fs[0])
        if im is not None:
            return im.shape[1], im.shape[0]
    return 1242, 375


def load_geometry(seq: str) -> Dict[int, Dict[int, Tuple[float, float, float, float]]]:
    """{frame_id: {track_id: (cx_px, cy_px, w_px, h_px)}} from labels_with_ids."""
    W, H = image_size(seq)
    seq_dir = os.path.join(LABELS_DIR, seq)
    out: Dict[int, Dict[int, Tuple[float, float, float, float]]] = {}
    for txt in sorted(glob.glob(os.path.join(seq_dir, "*.txt"))):
        fid = int(os.path.splitext(os.path.basename(txt))[0])
        frame: Dict[int, Tuple[float, float, float, float]] = {}
        with open(txt) as f:
            for line in f:
                p = line.split()
                if len(p) < 6:
                    continue
                tid = int(p[1])
                x1n, y1n, wn, hn = map(float, p[2:6])
                cx = (x1n + wn / 2.0) * W
                cy = (y1n + hn / 2.0) * H
                frame[tid] = (cx, cy, wn * W, hn * H)
        out[fid] = frame
    return out


def load_depth(seq: str) -> Dict[int, Dict[int, float]]:
    """{track_id: {frame_id: Z_meters}} from the GT depth cache."""
    path = os.path.join(DEPTH_DIR, f"z_track_gt_{seq}.json")
    raw = json.load(open(path))
    return {int(t): {int(f): float(z) for f, z in fr.items()} for t, fr in raw.items()}


def load_motion_referents(seq: str):
    """List of (sentence, motion_state, subtype, {frame_id: set(track_ids)}).

    Only expressions with a non-None motion_state (MOVING/STATIC) are returned.
    """
    out = []
    for jf in sorted(glob.glob(os.path.join(V1_EXPR_DIR, seq, "*.json"))):
        j = json.load(open(jf))
        sent = j["sentence"]
        state = motion_state(sent)
        if state is None:
            continue
        refs: Dict[int, set] = {}
        for fr, objs in j["label"].items():
            refs.setdefault(int(fr), set()).update(int(o) for o in objs)
        out.append((sent, state, keyword_subtype(sent), refs))
    return out
