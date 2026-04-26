"""Tests for KITTI odometry ATE (Umeyama-aligned translation RMSE)."""
from pathlib import Path

import numpy as np
import pytest

from eval.ego_ate import (
    absolute_trajectory_error,
    compose_relative_poses,
    load_kitti_poses,
)


def _identity_trajectory(n: int) -> np.ndarray:
    poses = np.zeros((n, 3, 4), dtype=np.float64)
    poses[:, :3, :3] = np.eye(3)
    return poses


def test_ate_zero_on_identical_trajectories():
    a = _identity_trajectory(10)
    a[:, :, 3] = np.random.RandomState(0).randn(10, 3)
    assert absolute_trajectory_error(a, a) < 1e-9


def test_ate_invariant_to_rigid_translation_offset():
    a = _identity_trajectory(10)
    a[:, :, 3] = np.arange(10)[:, None] * np.array([1.0, 0.5, 0.0])
    b = a.copy()
    b[:, :, 3] += np.array([100.0, -50.0, 7.0])
    err = absolute_trajectory_error(a, b)
    assert err < 1e-6, f"Umeyama should cancel rigid translation, got {err}"


def test_ate_invariant_to_global_rotation():
    a = _identity_trajectory(10)
    a[:, :, 3] = np.arange(10)[:, None] * np.array([1.0, 0.2, 0.0])
    theta = 0.7
    R = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    b = a.copy()
    b[:, :, 3] = (R @ a[:, :, 3].T).T
    err = absolute_trajectory_error(a, b)
    assert err < 1e-6, f"Umeyama should cancel global rotation, got {err}"


def test_ate_detects_per_frame_noise():
    a = _identity_trajectory(20)
    a[:, :, 3] = np.arange(20)[:, None] * np.array([1.0, 0.0, 0.0])
    b = a.copy()
    rng = np.random.default_rng(0)
    b[:, :, 3] += rng.normal(scale=0.1, size=(20, 3))
    err = absolute_trajectory_error(a, b)
    assert err > 0.01, f"Non-zero per-frame noise should leak through alignment, got {err}"


def test_compose_relative_poses_identity_stays_at_origin():
    rel_Rs = np.stack([np.eye(3) for _ in range(5)])
    rel_ts = np.zeros((5, 3))
    poses = compose_relative_poses(rel_Rs, rel_ts)
    assert poses.shape == (6, 3, 4)
    np.testing.assert_allclose(poses[:, :, 3], np.zeros((6, 3)), atol=1e-12)


def test_compose_relative_poses_pure_translation_accumulates():
    rel_Rs = np.stack([np.eye(3) for _ in range(4)])
    rel_ts = np.tile(np.array([1.0, 0.0, 0.0]), (4, 1))
    poses = compose_relative_poses(rel_Rs, rel_ts)
    expected_x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(poses[:, 0, 3], expected_x)


def test_compose_relative_poses_rotation_redirects_translation():
    rel_Rs = np.stack(
        [
            np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
            np.eye(3),
        ]
    )
    rel_ts = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    poses = compose_relative_poses(rel_Rs, rel_ts)
    np.testing.assert_allclose(poses[2, :, 3], np.array([1.0, 1.0, 0.0]), atol=1e-12)


@pytest.mark.skipif(
    not Path("/home/seanachan/data/Dataset/kitti_odometry/poses/09.txt").is_file(),
    reason="KITTI odometry poses not extracted yet",
)
def test_load_kitti_poses_real_seq_09_has_expected_shape():
    poses = load_kitti_poses(Path("/home/seanachan/data/Dataset/kitti_odometry/poses/09.txt"))
    assert poses.ndim == 3
    assert poses.shape[1:] == (3, 4)
    assert poses.shape[0] > 1500
