# GMC-Link/core.py
import cv2
import numpy as np
from typing import Optional, List, Tuple


class ORBHomographyEngine:
    """
    Compute rigid background motion (ego-motion) between frames using ORB features
    and RANSAC homography estimation. Masking foreground objects ensures we
    only track the true camera motion.
    """
    def __init__(self, max_features: int = 1500) -> None:
        self.orb = cv2.ORB_create(max_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def estimate_homography(
        self, 
        prev_frame: np.ndarray, 
        curr_frame: np.ndarray,
        prev_bboxes: Optional[List[Tuple[float, float, float, float]]] = None
    ) -> np.ndarray:
        """
        Estimate the 3x3 homography matrix H_prev_to_curr that transforms points
        from prev_frame to curr_frame coordinates.
        """
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY) if len(prev_frame.shape) == 3 else prev_frame
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY) if len(curr_frame.shape) == 3 else curr_frame

        mask = None
        if prev_bboxes:
            h, w = prev_gray.shape
            mask = np.ones((h, w), dtype=np.uint8) * 255
            for bbox in prev_bboxes:
                x1, y1, x2, y2 = map(int, bbox)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 > x1 and y2 > y1:
                    mask[y1:y2, x1:x2] = 0

        kp1, des1 = self.orb.detectAndCompute(prev_gray, mask=mask)
        kp2, des2 = self.orb.detectAndCompute(curr_gray, mask=None)

        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            return np.eye(3, dtype=np.float32)

        matches = self.matcher.knnMatch(des1, des2, k=2)
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:  # Lowe's ratio test
                    good_matches.append(m)
            elif len(match_pair) == 1:
                good_matches.append(match_pair[0])

        if len(good_matches) < 4:
            return np.eye(3, dtype=np.float32)

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if H is None:
            return np.eye(3, dtype=np.float32)
            
        return H.astype(np.float32)
