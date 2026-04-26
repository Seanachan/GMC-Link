"""KITTI odometry Absolute Trajectory Error with rigid + scale Umeyama alignment.

Compares a predicted trajectory (N x 3 x 4 pose matrices) to ground-truth,
after removing any arbitrary rigid + scale offset via Umeyama closed-form
alignment. Used in Exp 37 Stage A to ablate ORB vs recoverPose ego sources
against KITTI odometry ground-truth poses (seqs 09, 10).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np


def load_kitti_poses(path: Path) -> np.ndarray:
    """Load a KITTI-format pose file: one 3x4 pose per line, 12 whitespace-separated floats."""
    lines = Path(path).read_text().strip().splitlines()
    poses = np.array([list(map(float, line.split())) for line in lines], dtype=np.float64)
    return poses.reshape(-1, 3, 4)


def _umeyama(src: np.ndarray, tgt: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    assert src.shape == tgt.shape and src.ndim == 2 and src.shape[1] == 3
    mu_s = src.mean(axis=0)
    mu_t = tgt.mean(axis=0)
    s = src - mu_s
    t = tgt - mu_t
    cov = s.T @ t / src.shape[0]
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = Vt.T @ S @ U.T
    var_s = (s ** 2).sum() / src.shape[0]
    if var_s < 1e-12:
        scale = 1.0
    else:
        scale = float(np.trace(np.diag(D) @ S) / var_s)
    trans = mu_t - scale * R @ mu_s
    return scale, R, trans


def absolute_trajectory_error(pred: np.ndarray, gt: np.ndarray) -> float:
    """Translation RMSE between aligned predicted and GT trajectories (meters).

    Applies rigid + scale Umeyama alignment before computing RMSE so the result
    is invariant to the arbitrary world-frame offset of the estimator.
    """
    assert pred.shape == gt.shape, f"Shape mismatch: {pred.shape} vs {gt.shape}"
    p = pred[:, :3, 3]
    g = gt[:, :3, 3]
    scale, R, trans = _umeyama(p, g)
    aligned = (scale * (R @ p.T)).T + trans
    err = np.linalg.norm(aligned - g, axis=1)
    return float(np.sqrt((err ** 2).mean()))


def compose_relative_poses(rel_Rs: np.ndarray, rel_ts: np.ndarray) -> np.ndarray:
    """Accumulate per-frame relative {R, t} into absolute poses.

    rel_Rs: (N, 3, 3) rotations from frame k -> k+1.
    rel_ts: (N, 3) translations.
    Returns (N+1, 3, 4) with pose[0] = identity and pose[k+1] = pose[k] @ rel[k].
    """
    n = rel_Rs.shape[0]
    out = np.zeros((n + 1, 3, 4), dtype=np.float64)
    R_acc = np.eye(3, dtype=np.float64)
    t_acc = np.zeros(3, dtype=np.float64)
    out[0, :, :3] = R_acc
    out[0, :, 3] = t_acc
    for k in range(n):
        t_acc = t_acc + R_acc @ rel_ts[k]
        R_acc = R_acc @ rel_Rs[k]
        out[k + 1, :, :3] = R_acc
        out[k + 1, :, 3] = t_acc
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="KITTI odometry ATE comparison.")
    parser.add_argument("--pred-poses", required=True, help="Path to predicted poses txt (KITTI format)")
    parser.add_argument("--gt-poses", required=True, help="Path to GT poses txt (KITTI format)")
    args = parser.parse_args()

    pred = load_kitti_poses(Path(args.pred_poses))
    gt = load_kitti_poses(Path(args.gt_poses))
    n = min(len(pred), len(gt))
    ate = absolute_trajectory_error(pred[:n], gt[:n])
    print(f"ATE = {ate:.6f} (frames={n})")


if __name__ == "__main__":
    main()
