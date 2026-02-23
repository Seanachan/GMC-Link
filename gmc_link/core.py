# GMC-Link/core.py
import cv2
import numpy as np
from typing import Optional, List, Tuple


class DenseFlowEngine:
    """
    Computes dense optical flow between consecutive frames using Farneback.
    Returns a per-pixel (H, W, 2) flow field instead of a single 3x3 homography,
    enabling per-object velocity extraction that handles parallax at different depths.
    """
    def __init__(self) -> None:
        self.prev_gray: Optional[np.ndarray] = None

    def estimate(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Compute dense optical flow between the current and previous frame.
        
        Args:
            frame: (H, W, 3) Current video frame in BGR format.
            
        Returns:
            flow: (H, W, 2) per-pixel flow field [dx, dy], or None for the first frame.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            return None

        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray,
            flow=None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )

        self.prev_gray = gray
        return flow


def extract_object_velocity(
    flow: np.ndarray,
    bbox: Tuple[float, float, float, float],
    frame_shape: Tuple[int, int],
) -> np.ndarray:
    """
    Average the dense flow vectors inside a bounding box to get that object's apparent motion.
    
    Args:
        flow: (H, W, 2) dense optical flow field.
        bbox: (x1, y1, x2, y2) bounding box in pixel coordinates.
        frame_shape: (height, width) of the frame.
        
    Returns:
        velocity: (2,) mean flow [dx, dy] within the bounding box.
    """
    h, w = frame_shape
    x1, y1, x2, y2 = int(max(0, bbox[0])), int(max(0, bbox[1])), int(min(w, bbox[2])), int(min(h, bbox[3]))

    if x2 <= x1 or y2 <= y1:
        return np.zeros(2, dtype=np.float32)

    roi_flow = flow[y1:y2, x1:x2]
    return np.mean(roi_flow, axis=(0, 1)).astype(np.float32)


def extract_background_flow(
    flow: np.ndarray,
    bboxes: List[Tuple[float, float, float, float]],
    frame_shape: Tuple[int, int],
) -> np.ndarray:
    """
    Compute median flow across background pixels (excluding object bounding boxes).
    This represents the camera's ego-motion.
    
    Args:
        flow: (H, W, 2) dense optical flow field.
        bboxes: List of (x1, y1, x2, y2) bounding boxes to exclude.
        frame_shape: (height, width) of the frame.
        
    Returns:
        bg_flow: (2,) median background flow [dx, dy].
    """
    h, w = frame_shape
    mask = np.ones((h, w), dtype=bool)

    for bbox in bboxes:
        x1, y1, x2, y2 = int(max(0, bbox[0])), int(max(0, bbox[1])), int(min(w, bbox[2])), int(min(h, bbox[3]))
        mask[y1:y2, x1:x2] = False

    bg_pixels = flow[mask]  # (N, 2)
    if len(bg_pixels) == 0:
        return np.zeros(2, dtype=np.float32)

    return np.median(bg_pixels, axis=0).astype(np.float32)