"""Sanity tests for KITTI-tracking oxts→pose + ego-dz frame math (real seq 0005)."""
import os
import numpy as np
import pytest

import kitti_tracking_gt as G

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(G.TRACKING_ROOT, "oxts", "0005.txt")),
    reason="KITTI tracking oxts not downloaded",
)

DT = 0.1  # KITTI ~10 Hz


def test_pose0_is_identity():
    poses = G.oxts_to_poses(G.load_oxts("0005"))
    assert np.allclose(poses[0], np.eye(4), atol=1e-9)


def test_forward_translation_matches_integrated_vf():
    # Pose forward displacement over a gap must match integral of forward velocity vf.
    oxts = G.load_oxts("0005")
    poses = G.oxts_to_poses(oxts)
    gap = 10
    t = 100
    rel = np.linalg.inv(poses[t - gap]) @ poses[t]
    fwd_from_pose = rel[0, 3]                       # IMU x = forward
    fwd_from_vf = oxts[t - gap:t, 8].sum() * DT     # integrate vf
    # within 10% — validates Mercator scale + rotation order
    assert abs(fwd_from_pose - fwd_from_vf) / fwd_from_vf < 0.10


def test_ego_dz_sign_and_magnitude():
    # Forward driving -> static point gets closer -> ego_dz (camera Z) negative,
    # magnitude ~ forward distance traveled over the gap.
    oxts = G.load_oxts("0005")
    poses = G.oxts_to_poses(oxts)
    calib = G.load_calib("0005")
    gap = 5
    t = 100
    dz = G.ego_dz_camera(poses, calib, t, gap)
    fwd_dist = oxts[t - gap:t, 8].sum() * DT        # ~ vf * 0.5s
    assert dz < 0                                    # closing
    assert abs(abs(dz) - fwd_dist) / fwd_dist < 0.15


# ── LiDAR projection (calib-only, no velodyne files needed) ───────────────

def test_velo_projection_forward_point():
    # A Velodyne point 12 m forward, on axis -> camera Z ~ 12, u near principal pt.
    calib = G.load_calib("0005")
    pts = np.array([[12.0, 0.0, 0.0]])              # velo: x=fwd, y=left, z=up
    uv, z_cam = G.project_velo_to_image(pts, calib)
    assert z_cam.shape[0] == 1
    assert abs(z_cam[0] - 12.0) / 12.0 < 0.10        # forward distance preserved
    assert 0 <= uv[0, 0] <= 1242                     # lands in image width


def test_velo_projection_culls_behind():
    # Point behind the vehicle (velo x negative) must be culled (z_cam <= 0).
    calib = G.load_calib("0005")
    uv, z_cam = G.project_velo_to_image(np.array([[-8.0, 0.0, 0.0]]), calib)
    assert z_cam.shape[0] == 0


def test_bbox_lidar_depth_median_and_hole():
    calib = G.load_calib("0005")
    pts = np.array([[10.0, 0.0, 0.0], [10.5, 0.1, 0.0], [11.0, -0.1, 0.0]])
    uv, z_cam = G.project_velo_to_image(pts, calib)
    box = (uv[:, 0].min() - 5, uv[:, 1].min() - 5, uv[:, 0].max() + 5, uv[:, 1].max() + 5)
    res = G.bbox_lidar_depth(uv, z_cam, box, min_pts=1)
    assert res is not None
    med, std, n = res
    assert n == 3 and 10.0 <= med <= 11.0
    # empty box -> hole
    assert G.bbox_lidar_depth(uv, z_cam, (0, 0, 1, 1), min_pts=1) is None
