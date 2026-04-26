"""Tests for EMAP-style ego-velocity concat feature.

Spec §§3, 7 (Stage C): concat the 2D ego-translation velocity (vx, vy averaged
over FRAME_GAPS window) into the aligner input. Distinct from the older F4
``ego_motion`` channel by *intent* (structural conditioning signal rather than
an ablated motion cue), even though the emitted values overlap in practice.
"""
from gmc_link.dataset import (
    EXTRA_FEATURE_DIMS,
    compute_extra_dims,
    compute_per_track_extras,
)


def test_ego_velocity_concat_registered():
    assert "ego_velocity_concat" in EXTRA_FEATURE_DIMS
    assert EXTRA_FEATURE_DIMS["ego_velocity_concat"] == 2


def test_compute_extra_dims_sums_new_feature():
    assert compute_extra_dims(["ego_velocity_concat"]) == 2
    assert compute_extra_dims(["ego_velocity_concat", "speed_m"]) == 3


def test_emits_ego_vx_vy_from_mid_scale():
    """ego_velocity_concat should emit [ego_dx_m, ego_dy_m] (same mid-scale
    ego-translation cue that structural conditioning consumes)."""
    scale_vels = [(1.0, 0.0), (0.5, 0.5), (0.2, 0.8)]
    out = compute_per_track_extras(
        ["ego_velocity_concat"],
        scale_vels,
        ego_dx_m=3.14,
        ego_dy_m=-2.71,
    )
    assert out == [3.14, -2.71]


def test_does_not_leak_when_not_requested():
    out = compute_per_track_extras(
        ["speed_m"],
        [(1.0, 0.0), (0.5, 0.5), (0.2, 0.8)],
        ego_dx_m=99.0,
        ego_dy_m=99.0,
    )
    # speed_m emits sqrt(0.5^2 + 0.5^2) — ego vals must not appear
    assert len(out) == 1
    assert 99.0 not in out
