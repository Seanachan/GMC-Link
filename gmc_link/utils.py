"""
Utility functions for GMC-Link covering geometric warping and kinematics buffer smoothing.
"""
from typing import Dict, List, Tuple
import numpy as np
# Scale factor for normalized velocities so the MLP operates on ~1.0 magnitude values
# Calibrated for frame_gap=5 (5-frame window produces clean, robust centroid diffs)
VELOCITY_SCALE = 100


def warp_points(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    """
    Transforms coordinates from frame t-1 to frame t using a homography matrix.

    Args:
        points: (N, 2) array representing the (x, y) coordinates of the points.

        homography: (3, 3) matrix representing the background translation.

    Returns:
        warped_points: (N, 2) array representing the warped (x', y') coordinates.
    """
    if points is None or len(points) == 0:
        return np.empty((0, 2))  # Return an empty array if there are no points to warp

    num_points = points.shape[0]
    homogeneous_points = np.hstack(
        (points, np.ones((num_points, 1)))
    )  # Convert to homogeneous
    warped_homogeneous = homogeneous_points @ homography.T  # Apply homography

    # Convert back to Cartesian coordinates
    warped_points = warped_homogeneous[:, :2] / warped_homogeneous[:, 2:3]
    return warped_points


def normalize_velocity(v_comp: np.ndarray, frame_shape: Tuple[int, int]) -> np.ndarray:
    """
    Scales pixel velocity to a normalized range based on frame dimensions.
    This ensures GMC-Link is scale-invariant (works the same on 720p vs 4K).

    Args:
        v_comp: (N, 2) array of [dx, dy] pixel velocities.
        frame_shape: (height, width) of the video frame.

    Returns:
        v_norm: (N, 2) array of normalized velocities.
    """
    if len(v_comp) == 0:
        return v_comp

    h, w = frame_shape
    # Ensure v_comp is a numpy array
    v_comp = np.array(v_comp, dtype=np.float32)
    # Normalize by dimensions so a movement of 'half the screen' is 0.5, regardless of pixel count.
    v_norm = v_comp / np.array([w, h], dtype=np.float32)
    # Scale up so the MLP operates on values around ~1.0 instead of ~0.01
    v_norm *= VELOCITY_SCALE
    return v_norm


class MotionBuffer:
    """
    Simple temporal smoothing to prevent 'jitter' in the reasoning module.
    Calculates an Exponential Moving Average (EMA) of velocities.
    """

    def __init__(self, alpha: float = 0.8) -> None:
        self.alpha: float = alpha
        self.registry: Dict[int, np.ndarray] = {}  # {track_id: last_v}

    def smooth(self, track_id: int, v_new: np.ndarray) -> np.ndarray:
        """
        Update the track's velocity with EMA smoothing.
        """
        if track_id not in self.registry:
            self.registry[track_id] = v_new
            return v_new

        # EMA: alpha * new + (1-alpha) * old
        v_smoothed = (self.alpha * v_new) + ((1 - self.alpha) * self.registry[track_id])
        self.registry[track_id] = v_smoothed
        return v_smoothed

    def clear_dead_tracks(self, active_track_ids: List[int]) -> None:
        """
        Remove tracks from the registry that are no longer active to prevent memory bloat.
        """
        dead = set(self.registry.keys()) - set(active_track_ids)
        for d in dead:
            del self.registry[d]


class ScoreBuffer:
    """
    Temporal smoothing for alignment scores to prevent flickering in visualization.
    Uses EMA to stabilize per-track scores across frames.
    """

    def __init__(self, alpha: float = 0.4) -> None:
        self.alpha: float = alpha
        self.registry: Dict[int, float] = {}  # {track_id: smoothed_score}

    def smooth(self, track_id: int, raw_score: float) -> float:
        """Apply EMA smoothing to a track's alignment score."""
        if track_id not in self.registry:
            self.registry[track_id] = raw_score
            return raw_score

        smoothed = self.alpha * raw_score + (1 - self.alpha) * self.registry[track_id]
        self.registry[track_id] = smoothed
        return smoothed

    def clear_dead_tracks(self, active_track_ids: List[int]) -> None:
        """
        Remove inactive track score registries to prevent memory bloat.
        """
        dead = set(self.registry.keys()) - set(active_track_ids)
        for d in dead:
            del self.registry[d]


def velocity_confidence(
    velocity: np.ndarray, threshold: float = 0.3, steepness: float = 10.0
) -> float:
    """
    Compute a [0, 1] confidence that the object is actually moving,
    based on velocity magnitude. Uses a sigmoid gate centered at `threshold`.

    Objects with near-zero compensated velocity (i.e. stationary after camera
    compensation) get confidence → 0, suppressing their alignment score.

    Args:
        velocity: (2,) normalized velocity vector [dx, dy].
        threshold: Velocity magnitude below which confidence drops sharply.
        steepness: Controls how sharp the sigmoid transition is.
    """
    speed = float(np.linalg.norm(velocity))
    # Sigmoid: 1 / (1 + exp(-steepness * (speed - threshold)))
    return 1.0 / (1.0 + np.exp(-steepness * (speed - threshold)))
