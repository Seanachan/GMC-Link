"""Unit tests for the velocity-ladder math kernel.

Locks sign/scale conventions for the 4-level information-cap ladder:
  L1 raw pixel  | L2 pixel x Z/f (no ego) | L3 ego-comp pixel | L4 full metric residual

Run: python -m pytest diagnostics/ego_motion_ladder/test_ladder_lib.py -q
"""
import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import ladder_lib as L


FX = FY = 721.5377  # canonical KITTI 2011_09_26


def approx(a, b, tol=1e-6):
    return abs(a - b) <= tol


def test_no_motion_all_levels_zero():
    # Single static track: zero pixel motion, zero depth change.
    vels = {1: (0.0, 0.0)}
    dzs = {1: 0.0}
    ego = L.estimate_ego(vels, dzs)
    assert ego == (0.0, 0.0, 0.0)
    lv = L.compute_levels(v_pix=(0.0, 0.0), z_t=20.0, dz=0.0, ego=ego, fx=FX, fy=FY)
    assert approx(lv["L1"], 0.0) and approx(lv["L2"], 0.0)
    assert approx(lv["L3"], 0.0) and approx(lv["L4"], 0.0)


def test_pure_ego_static_object_cancels_at_L3_L4():
    # Three tracks all swept identically by ego (rigid pixel translation + uniform dZ).
    # Ego estimate must equal that common motion, so residual (L3) and metric (L4) vanish,
    # while raw L1/L2 stay large -> demonstrates ego compensation.
    common = (6.0, -2.0)
    common_dz = 1.5
    vels = {1: common, 2: common, 3: common}
    dzs = {1: common_dz, 2: common_dz, 3: common_dz}
    ego = L.estimate_ego(vels, dzs)
    assert approx(ego[0], 6.0) and approx(ego[1], -2.0) and approx(ego[2], 1.5)
    lv = L.compute_levels(v_pix=common, z_t=25.0, dz=common_dz, ego=ego, fx=FX, fy=FY)
    assert lv["L1"] > 1.0           # raw pixel motion is large
    assert approx(lv["L3"], 0.0)    # ego removed
    assert approx(lv["L4"], 0.0)    # ego removed in metric too


def test_object_moving_beyond_ego_survives_compensation():
    # Two ~static tracks (carried by ego) + one object with extra (3,0) px/frame.
    ego_motion = (5.0, 0.0)
    obj = (8.0, 0.0)  # ego + (3,0)
    vels = {1: ego_motion, 2: ego_motion, 3: obj}
    dzs = {1: 2.0, 2: 2.0, 3: 2.0}
    ego = L.estimate_ego(vels, dzs)
    # slower-half cohort = the two ego tracks -> ego estimate = (5,0,2)
    assert approx(ego[0], 5.0) and approx(ego[1], 0.0) and approx(ego[2], 2.0)
    lv = L.compute_levels(v_pix=obj, z_t=20.0, dz=2.0, ego=ego, fx=FX, fy=FY)
    assert approx(lv["L3"], 3.0)    # residual lateral pixel speed = 3 px/frame
    assert lv["L1"] == 8.0          # raw includes ego


def test_L2_scales_pixel_by_Z_over_f():
    # No ego: L2 = ||v_pix * Z/f||.
    ego = (0.0, 0.0, 0.0)
    lv = L.compute_levels(v_pix=(10.0, 0.0), z_t=72.15377, dz=0.0, ego=ego, fx=FX, fy=FY)
    # 10 px/frame * (72.15377/721.5377) = 10 * 0.1 = 1.0 m/frame
    assert approx(lv["L2"], 1.0, tol=1e-4)


def test_metric_residual_includes_depth_axis():
    # Object pulling away in depth (positive residual dZ), small lateral residual.
    ego = (0.0, 0.0, 0.0)
    lv = L.compute_levels(v_pix=(0.0, 0.0), z_t=30.0, dz=3.0, ego=ego, fx=FX, fy=FY)
    # lateral metric ~0, depth residual = 3.0 -> L4 = 3.0, but L1/L3 (pixel) = 0
    assert approx(lv["L1"], 0.0)
    assert approx(lv["L3"], 0.0)
    assert approx(lv["L4"], 3.0, tol=1e-6)
    assert approx(lv["v_metric_z"], 3.0)
    assert approx(lv["v_metric_x"], 0.0)


def test_sdf_flag_logic():
    tau = 1.0
    # qualifies: dZ_res > tau, lateral/long ratio < 0.3
    assert L.is_sdf(dz_res=2.0, v_metric_x=0.1, v_metric_z=2.0, tau_z=tau)
    # fails: dZ_res below threshold
    assert not L.is_sdf(dz_res=0.5, v_metric_x=0.1, v_metric_z=2.0, tau_z=tau)
    # fails: lateral too large (ratio 1.0/2.0 = 0.5 > 0.3)
    assert not L.is_sdf(dz_res=2.0, v_metric_x=1.0, v_metric_z=2.0, tau_z=tau)
    # fails: near-zero longitudinal velocity (division noise guard)
    assert not L.is_sdf(dz_res=2.0, v_metric_x=0.0, v_metric_z=1e-4, tau_z=tau)


def test_auc_perfect_and_degenerate():
    # perfect separation
    a = L.safe_auc([0, 0, 1, 1], [0.1, 0.2, 0.9, 0.8])
    assert approx(a, 1.0)
    # single-class -> None (degenerate, cannot compute)
    assert L.safe_auc([1, 1, 1], [0.1, 0.2, 0.3]) is None
    assert L.safe_auc([], []) is None
