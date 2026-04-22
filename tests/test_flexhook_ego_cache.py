"""Exp 38 Day 1 — sanity tests for the ego-compensated 13D velocity cache and
the GMC-Link → FlexHook coord conversion formula.

These tests assume the extractor has been run against the V1 held-out seqs:
    python tools/flexhook_ego_extractor.py \
        --tracks ~/FlexHook/tracker_outputs/Temp-NeuralSORT-kitti1 \
        --frames ~/FlexHook/datasets/refer-kitti/KITTI/training/image_02 \
        --out diagnostics/results/exp38/cache/ego_speed \
        --seqs 0005 0011 0013
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import sys
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.flexhook_ego_extractor import (
    RESIDUAL_TO_NORMGRID,
    residual_to_normgrid,
)
from gmc_link.utils import VELOCITY_SCALE


CACHE_ROOT = _ROOT / "diagnostics/results/exp38/cache/ego_speed"
EXPECTED_SEQS = {
    "0005": 1023,
    "0011": 2551,
    "0013": 988,
}


def _require_cache(seq: str) -> Path:
    path = CACHE_ROOT / seq / "ego_13d.npz"
    if not path.exists():
        pytest.skip(f"cache not generated for {seq}: {path}")
    return path


@pytest.mark.parametrize("seq,expected_n", EXPECTED_SEQS.items())
def test_cache_schema_and_count(seq: str, expected_n: int):
    path = _require_cache(seq)
    d = np.load(path)
    assert set(d.files) == {"frames", "track_ids", "classes", "vec13d"}
    assert d["vec13d"].shape == (expected_n, 13)
    assert d["vec13d"].dtype == np.float32
    assert d["frames"].shape == (expected_n,)
    assert d["track_ids"].shape == (expected_n,)
    assert d["classes"].shape == (expected_n,)
    assert d["frames"].min() >= 1
    # Classes must be a subset of {car, pedestrian}
    assert set(np.unique(d["classes"]).tolist()).issubset({b"c", b"p"})


@pytest.mark.parametrize("seq", EXPECTED_SEQS.keys())
def test_residual_velocity_distribution_nontrivial(seq: str):
    """Cold-start rows get zero velocity; past-history rows must have variance."""
    path = _require_cache(seq)
    v = np.load(path)["vec13d"]
    # Cols 0..5 are residual dx/dy at gaps {2,5,10}.
    # At least the short-gap columns should be mostly non-zero after warmup.
    gap2_nz = np.abs(v[:, :2]).sum(axis=1) > 0
    assert gap2_nz.mean() > 0.6, f"{seq}: gap=2 residual too sparse ({gap2_nz.mean():.2f})"
    # Std must be bounded but non-zero (a degenerate cache has std=0).
    for c in range(6):
        std = v[v[:, :6].any(axis=1), c].std()
        assert 0.0 < std < 50.0, f"{seq}: col {c} std={std:.3f} out of sanity range"


@pytest.mark.parametrize("seq", EXPECTED_SEQS.keys())
def test_spatial_cols_always_populated(seq: str):
    """cx, cy, w, h (cols 8..11) come from the bbox — must be set on every row."""
    path = _require_cache(seq)
    v = np.load(path)["vec13d"]
    spatial = v[:, 8:12]
    assert (spatial > 0).all(), f"{seq}: spatial cols have zero entries"
    # Normalized coords must stay in (0, 1].
    assert spatial.max() <= 1.5, f"{seq}: normalized coord > 1.5 suggests unnormalized input"


def test_residual_to_normgrid_constant():
    # pixel_disp=10, img_dim=1000: residual(VELOCITY_SCALE=100) = 1.0,
    # normgrid = 10 / (1000/2) = 0.02 → formula residual*0.02 = 0.02.
    assert RESIDUAL_TO_NORMGRID == pytest.approx(2.0 / VELOCITY_SCALE)


def test_residual_to_normgrid_round_trip():
    """Round-trip: pixel → residual → normgrid → back to pixel (per img_dim)."""
    img_w, img_h = 1242, 375  # KITTI resolution
    pixel_dx = np.array([10.0, -25.0, 0.0], dtype=np.float32)
    pixel_dy = np.array([5.0, 8.0, -3.0], dtype=np.float32)
    residual_dx = pixel_dx / img_w * VELOCITY_SCALE
    residual_dy = pixel_dy / img_h * VELOCITY_SCALE
    ng_dx = residual_to_normgrid(residual_dx)
    ng_dy = residual_to_normgrid(residual_dy)
    back_px_dx = ng_dx * (img_w / 2.0)
    back_px_dy = ng_dy * (img_h / 2.0)
    np.testing.assert_allclose(back_px_dx, pixel_dx, atol=1e-4)
    np.testing.assert_allclose(back_px_dy, pixel_dy, atol=1e-4)


def test_residual_to_normgrid_vectorized_on_cache():
    """Apply conversion to a real cache row — output must sit in plausible
    normgrid range (|ng| < 0.5 means < 25% of frame per-frame, reasonable)."""
    path = _require_cache("0005")
    v = np.load(path)["vec13d"]
    ng = residual_to_normgrid(v[:, :6])
    assert np.isfinite(ng).all()
    # KITTI fg motion rarely exceeds ~half a frame per 10-frame gap.
    assert np.abs(ng).max() < 1.0, f"extreme normgrid velocity: {np.abs(ng).max():.3f}"
