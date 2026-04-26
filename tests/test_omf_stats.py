"""Tests for per-bbox Object Motion Field (OMF) pooling stats.

Spec §3 row "Stage B OMF 15 extra dims": Per-bbox [mean_dx, mean_dy, std_dx,
std_dy, max_mag] concatenated across FRAME_GAPS = (2, 5, 10) → 15 extra dims.
This test file covers the single-scale primitive that produces 5 stats; the
per-scale concatenation lives in dataset.py (Task 10).
"""
import numpy as np

from gmc_link.features.omf_stats import per_bbox_omf_stats


def test_shape_is_5():
    omf = np.zeros((64, 64, 2), dtype=np.float32)
    out = per_bbox_omf_stats(omf, (10, 10, 20, 20))
    assert out.shape == (5,), f"Expected (5,), got {out.shape}"


def test_uniform_flow_matches_expected_stats():
    """Constant dx=1, dy=2 everywhere → mean=(1,2), std=(0,0), mag=sqrt(5)."""
    omf = np.zeros((64, 64, 2), dtype=np.float32)
    omf[..., 0] = 1.0
    omf[..., 1] = 2.0
    out = per_bbox_omf_stats(omf, (10, 10, 20, 20))
    np.testing.assert_allclose(out[0], 1.0)
    np.testing.assert_allclose(out[1], 2.0)
    np.testing.assert_allclose(out[2], 0.0, atol=1e-6)
    np.testing.assert_allclose(out[3], 0.0, atol=1e-6)
    np.testing.assert_allclose(out[4], np.sqrt(5.0), atol=1e-6)


def test_empty_bbox_returns_zeros():
    omf = np.zeros((64, 64, 2), dtype=np.float32)
    out = per_bbox_omf_stats(omf, (0, 0, 0, 0))
    np.testing.assert_array_equal(out, np.zeros(5, dtype=np.float32))


def test_out_of_bounds_bbox_clips_to_frame():
    """Partially out-of-frame bbox clipped, still returns finite values."""
    omf = np.ones((64, 64, 2), dtype=np.float32)
    out = per_bbox_omf_stats(omf, (60, 60, 20, 20))
    assert not np.any(np.isnan(out))
    assert not np.any(np.isinf(out))


def test_fully_out_of_frame_returns_zeros():
    """bbox entirely outside frame → zeros (defensive; prevents NaN downstream)."""
    omf = np.ones((64, 64, 2), dtype=np.float32)
    out = per_bbox_omf_stats(omf, (200, 200, 10, 10))
    np.testing.assert_array_equal(out, np.zeros(5, dtype=np.float32))


def test_dtype_is_float32():
    omf = np.ones((64, 64, 2), dtype=np.float32)
    out = per_bbox_omf_stats(omf, (0, 0, 10, 10))
    assert out.dtype == np.float32
