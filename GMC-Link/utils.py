import numpy as np
import torch


def warp_points(points, homography):
    """
    Transforms coordinates from framt t-1 to the coordinate sysmtem of frame t using homography matrix.
    Args:
        points: A numpy array of shape (N, 2) representing the (x, y) coordinates of the points to be warped.
        homography: A 3x3 homography matrix.
    Returns:
        warped_points: A numpy array of shape (N, 2) representing the warped (x', y') coordinates of the points.
    """
    if points is None or len(points) == 0:
        return np.empty((0, 2))  # Return an empty array if there are no points to warp

    num_points = points.shape[0]
    homogeneous_points = np.hstack((points, np.ones((num_points, 1))))  # Convert to homogeneous coordinates

    warped_homogeneous = homogeneous_points @ homography.T  # Apply homography

    warped_points = warped_homogeneous[:, :2] / warped_homogeneous[:, 2:3]  # Convert back to Cartesian coordinates
    return warped_points


def normalize_velocity(v_comp, frame_shape):
    """
    Scales pixel velocity to a normalized range [-1, 1] based on frame dimensions.
    This ensures GMC-Link is scale-invariant (works the same on 720p vs 4K).
    
    Args:
        v_comp: (N, 2) array of [dx, dy]
        frame_shape: (height, width)
    """
    if len(v_comp) == 0:
        return v_comp
        
    h, w = frame_shape
    # We normalize by the dimensions so a movement of 'half the screen'
    # always results in 0.5, regardless of pixel count.
    v_norm = v_comp / np.array([w, h], dtype=np.float32)
    return v_norm


class MotionBuffer:
    """
    Simple temporal smoothing to prevent 'jitter' in the reasoning module.
    Calculates an Exponential Moving Average (EMA) of velocities.
    """
    def __init__(self, alpha=0.8):
        self.alpha = alpha
        self.registry = {} # Stores {track_id: last_v}

    def smooth(self, track_id, v_new):
        if track_id not in self.registry:
            self.registry[track_id] = v_new
            return v_new
        
        # EMA: alpha * new + (1-alpha) * old
        v_smoothed = (self.alpha * v_new) + ((1 - self.alpha) * self.registry[track_id])
        self.registry[track_id] = v_smoothed
        return v_smoothed

    def clear_dead_tracks(self, active_ids):
        """Removes IDs that are no longer being tracked to save memory."""
        self.registry = {tid: v for tid, v in self.registry.items() if tid in active_ids}