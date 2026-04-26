"""Tests for the EgoRouter registry + ORB adapter."""
import numpy as np
import pytest

from gmc_link.ego.ego_router import (
    OrbEgoRouter,
    available_ego_routers,
    make_ego_router,
)


def _rand_frame(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random((64, 64, 3)) * 255).astype(np.uint8)


def test_orb_registered_by_default():
    assert "orb" in available_ego_routers()


def test_make_ego_router_returns_orb_by_name():
    router = make_ego_router("orb")
    assert isinstance(router, OrbEgoRouter)


def test_make_ego_router_unknown_raises():
    with pytest.raises(KeyError):
        make_ego_router("nonexistent")


def test_orb_router_returns_3x3_homography_and_2vec_residual():
    router = OrbEgoRouter(max_features=500)
    H, residual = router.estimate_homography(_rand_frame(0), _rand_frame(1))
    assert H.shape == (3, 3)
    assert H.dtype in (np.float32, np.float64)
    assert residual.shape == (2,)


def test_orb_router_identity_on_same_frame():
    router = OrbEgoRouter(max_features=500)
    frame = _rand_frame(42)
    H, residual = router.estimate_homography(frame, frame)
    np.testing.assert_allclose(H, np.eye(3), atol=0.5)
    assert float(residual.sum()) < 2.0
