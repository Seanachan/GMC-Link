"""KITTI-tracking GT-signal loaders for Path B (oxts ego + calib; LiDAR later).

Data is the KITTI *tracking* benchmark distribution, RE-INDEXED to tracking
sequence numbering (0000-0020) == Refer-KITTI numbering. So frame f here == image
frame f in refer-kitti (verified: oxts/0005.txt has 297 rows == 297 images).

Coordinate frames (KITTI raw README):
  IMU/GPS : x=forward, y=left, z=up
  Velodyne: x=forward, y=left, z=up
  Camera  : x=right,   y=down, z=forward
Camera +z = forward, so ego forward motion makes a static point's Z DECREASE.

Path B scope (a): this module provides oxts-derived ego vz (camera frame) to
replace the stationary-cohort dz_ego, and calib for LiDAR projection. Pixel-domain
ego (ORB homography) and the 13D base are NOT touched.
"""
from __future__ import annotations

import math
import os
from typing import Dict, List

import numpy as np

TRACKING_ROOT = "/home/seanachan/data/Dataset/kitti_tracking/training"


# ── oxts → metric pose (port of devkit convertOxtsToPose) ─────────────────

def load_oxts(seq: str) -> np.ndarray:
    """(N, 30) oxts array for a tracking sequence. Cols 0-5 = lat,lon,alt,roll,pitch,yaw;
    8=vf (forward m/s), 9=vl (left m/s); 22=wz (yaw-rate rad/s)."""
    return np.loadtxt(os.path.join(TRACKING_ROOT, "oxts", f"{seq}.txt"))


def _lat_to_scale(lat: float) -> float:
    return math.cos(lat * math.pi / 180.0)


def _latlon_to_mercator(lat: float, lon: float, scale: float):
    er = 6378137.0
    mx = scale * lon * math.pi * er / 180.0
    my = scale * er * math.log(math.tan((90.0 + lat) * math.pi / 360.0))
    return mx, my


def oxts_to_poses(oxts: np.ndarray) -> List[np.ndarray]:
    """List of 4x4 rigid poses: pose[i] maps a point in IMU frame i into the
    nav frame of frame 0 (pose[0] == identity). Port of devkit convertOxtsToPose."""
    scale = _lat_to_scale(oxts[0, 0])
    poses: List[np.ndarray] = []
    t0_inv = None
    for row in oxts:
        lat, lon, alt, roll, pitch, yaw = row[0], row[1], row[2], row[3], row[4], row[5]
        mx, my = _latlon_to_mercator(lat, lon, scale)
        t = np.array([mx, my, alt], dtype=np.float64)
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        R = Rz @ Ry @ Rx
        P = np.eye(4)
        P[:3, :3] = R
        P[:3, 3] = t
        if t0_inv is None:
            t0_inv = np.linalg.inv(P)
        poses.append(t0_inv @ P)
    return poses


# ── calibration ───────────────────────────────────────────────────────────

def load_calib(seq: str) -> Dict[str, np.ndarray]:
    """Parse tracking calib/{seq}.txt → P2 (3x4), R_rect (4x4), Tr_velo_cam (4x4),
    Tr_imu_velo (4x4). The 3x4 rigid Trs are padded to 4x4."""
    raw: Dict[str, np.ndarray] = {}
    with open(os.path.join(TRACKING_ROOT, "calib", f"{seq}.txt")) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            key, vals = line.split(":", 1) if ":" in line else line.split(None, 1)
            raw[key.strip()] = np.array([float(x) for x in vals.split()])

    def to44(v):
        M = np.eye(4)
        M[:3, :4] = v.reshape(3, 4)
        return M

    R_rect = np.eye(4)
    R_rect[:3, :3] = raw["R_rect"].reshape(3, 3)
    return {
        "P2": raw["P2"].reshape(3, 4),
        "R_rect": R_rect,
        "Tr_velo_cam": to44(raw["Tr_velo_cam"]),
        "Tr_imu_velo": to44(raw["Tr_imu_velo"]),
    }


def R_cam_from_imu(calib: Dict[str, np.ndarray]) -> np.ndarray:
    """3x3 rotation taking a displacement vector from IMU frame to camera frame."""
    T_cam_imu = calib["Tr_velo_cam"] @ calib["Tr_imu_velo"]
    return T_cam_imu[:3, :3]


def cam_to_world(calib: Dict[str, np.ndarray], pose: np.ndarray) -> np.ndarray:
    """4x4 mapping a rectified-camera point to the nav/world frame, given the
    IMU pose (imu_t -> nav0). Inverse of the projection chain
    X_cam_rect = R_rect . Tr_velo_cam . Tr_imu_velo^-1 ... -> world = pose . imu."""
    return (pose
            @ np.linalg.inv(calib["Tr_imu_velo"])
            @ np.linalg.inv(calib["Tr_velo_cam"])
            @ np.linalg.inv(calib["R_rect"]))


# ── ego depth-velocity (the dz_ego replacement) ───────────────────────────

# ── Velodyne LiDAR → per-bbox depth (Path B depth source) ─────────────────

def load_velodyne(seq: str, frame0: int) -> np.ndarray:
    """(N,4) raw Velodyne scan [x,y,z,reflectance] for 0-based image index frame0.
    Path: training/velodyne/{seq}/{frame0:06d}.bin (tracking-indexed, sync'd to images)."""
    path = os.path.join(TRACKING_ROOT, "velodyne", seq, f"{frame0:06d}.bin")
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)


def project_velo_to_image(pts_xyz: np.ndarray, calib: Dict[str, np.ndarray]):
    """Project Velodyne xyz (N,3) to image. Returns (uv (M,2), z_cam (M,)) for points
    IN FRONT of the camera (z_cam>0). Y = P2 . R_rect . Tr_velo_cam . X_velo."""
    n = pts_xyz.shape[0]
    pts_h = np.hstack([pts_xyz, np.ones((n, 1))])           # (N,4)
    cam = (calib["R_rect"] @ calib["Tr_velo_cam"] @ pts_h.T).T   # (N,4) rectified cam
    z_cam = cam[:, 2]
    front = z_cam > 0.1
    cam, z_cam = cam[front], z_cam[front]
    uvw = (calib["P2"] @ cam.T).T                            # (M,3)
    uv = uvw[:, :2] / uvw[:, 2:3]
    return uv, z_cam


def bbox_lidar_depth(uv: np.ndarray, z_cam: np.ndarray, box, min_pts: int = 1):
    """Median camera-Z of LiDAR points inside box (x1,y1,x2,y2).
    Returns (median_z, std_z, n_points) or None if fewer than min_pts points.
    std is reported for diagnostics (multi-Z-layer bboxes have large std)."""
    x1, y1, x2, y2 = box
    m = (uv[:, 0] >= x1) & (uv[:, 0] <= x2) & (uv[:, 1] >= y1) & (uv[:, 1] <= y2)
    zs = z_cam[m]
    if zs.shape[0] < min_pts:
        return None
    return float(np.median(zs)), float(np.std(zs)), int(zs.shape[0])


def ego_dz_camera(poses: List[np.ndarray], calib: Dict[str, np.ndarray],
                  t: int, gap: int) -> float:
    """Ego-induced ΔZ (camera frame, meters) of a static point between frame
    t-gap and t. = camera-z component of the static point's apparent displacement,
    which is -(ego translation in IMU frame), rotated into camera frame.

    Replaces stationary-cohort median ΔZ: dz_residual = dz_track - ego_dz_camera.
    """
    if t - gap < 0 or t >= len(poses):
        return 0.0
    rel = np.linalg.inv(poses[t - gap]) @ poses[t]      # frame t origin in (t-gap) IMU frame
    ego_trans_imu = rel[:3, 3]                          # how far ego moved (IMU frame)
    static_disp_cam = R_cam_from_imu(calib) @ (-ego_trans_imu)
    return float(static_disp_cam[2])                     # camera z-component
