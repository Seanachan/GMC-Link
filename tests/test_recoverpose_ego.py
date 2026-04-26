"""Tests for the cv2.recoverPose ego adapter (SparseMFE fallback)."""
import numpy as np

from gmc_link.ego.ego_router import available_ego_routers, make_ego_router
from gmc_link.ego.recoverpose_ego import (
    KITTI_DEFAULT_K,
    RecoverPoseEgoRouter,
)


def _rand_frame(seed: int, shape=(256, 832)):
    rng = np.random.default_rng(seed)
    return (rng.random((*shape, 3)) * 255).astype(np.uint8)


def test_recoverpose_registered():
    assert "recoverpose" in available_ego_routers()
    assert isinstance(make_ego_router("recoverpose"), RecoverPoseEgoRouter)


def test_shape_and_residual_types():
    router = RecoverPoseEgoRouter(max_features=1500)
    H, residual = router.estimate_homography(_rand_frame(0), _rand_frame(1))
    assert H.shape == (3, 3)
    assert H.dtype == np.float32
    assert residual.shape == (2,)


def test_identity_on_same_frame():
    router = RecoverPoseEgoRouter(max_features=1500)
    frame = _rand_frame(42)
    H, residual = router.estimate_homography(frame, frame)
    np.testing.assert_allclose(H, np.eye(3), atol=1.0)


def test_too_few_features_returns_identity():
    router = RecoverPoseEgoRouter(max_features=1500)
    blank = np.zeros((64, 64, 3), dtype=np.uint8)
    H, residual = router.estimate_homography(blank, blank)
    np.testing.assert_allclose(H, np.eye(3), atol=1e-6)
    np.testing.assert_allclose(residual, np.zeros(2), atol=1e-6)


def test_default_K_is_kitti_intrinsics():
    router = RecoverPoseEgoRouter()
    np.testing.assert_allclose(router.K, KITTI_DEFAULT_K)


def test_custom_K_accepted():
    K = np.eye(3) * 500.0
    K[2, 2] = 1.0
    router = RecoverPoseEgoRouter(K=K)
    np.testing.assert_allclose(router.K, K)


def test_foreground_mask_does_not_crash():
    router = RecoverPoseEgoRouter(max_features=500)
    f0 = _rand_frame(0)
    f1 = _rand_frame(1)
    bboxes = [(10.0, 10.0, 50.0, 50.0), (100.0, 100.0, 200.0, 150.0)]
    H, residual = router.estimate_homography(f0, f1, prev_bboxes=bboxes)
    assert H.shape == (3, 3)
