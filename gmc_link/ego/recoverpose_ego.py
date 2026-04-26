"""6DoF ego estimator via cv2.recoverPose on ORB correspondences.

Spec §10 kill switch #1 fallback: when SparseMFE is not publicly released,
substitute classical 5-point essential-matrix decomposition as the "non-ORB"
ego source for Stage A. Returns a 3x3 homography projected from the recovered
rotation via the plane-at-infinity approximation (H_inf = K R K^-1) so
GMCLinkManager centroid warping remains backwards-compatible.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from gmc_link.ego.ego_router import register_ego_router

KITTI_DEFAULT_K = np.array(
    [[721.5377, 0.0, 609.5593], [0.0, 721.5377, 172.854], [0.0, 0.0, 1.0]],
    dtype=np.float64,
)


class RecoverPoseEgoRouter:
    """Classical 6DoF ego recovered from ORB correspondences via essential matrix.

    Projects the resulting {R, t} into a 3x3 plane-at-infinity homography so
    downstream GMCLinkManager centroid warping uses the same 3x3 interface as
    ORBHomographyEngine.
    """

    def __init__(
        self,
        max_features: int = 1500,
        K: Optional[np.ndarray] = None,
    ) -> None:
        self.orb = cv2.ORB_create(max_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.K = K if K is not None else KITTI_DEFAULT_K
        self.K_inv = np.linalg.inv(self.K)

    def estimate_homography(
        self,
        prev_frame: np.ndarray,
        curr_frame: np.ndarray,
        prev_bboxes: Optional[List[Tuple[float, float, float, float]]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        prev_gray = (
            cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            if len(prev_frame.shape) == 3
            else prev_frame
        )
        curr_gray = (
            cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            if len(curr_frame.shape) == 3
            else curr_frame
        )

        mask = self._build_foreground_mask(prev_gray.shape, prev_bboxes)
        kp1, des1 = self.orb.detectAndCompute(prev_gray, mask=mask)
        kp2, des2 = self.orb.detectAndCompute(curr_gray, mask=None)

        if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
            return np.eye(3, dtype=np.float32), np.zeros(2, dtype=np.float32)

        good = self._lowe_ratio_matches(des1, des2)
        if len(good) < 8:
            return np.eye(3, dtype=np.float32), np.zeros(2, dtype=np.float32)

        src = np.float32([kp1[m.queryIdx].pt for m in good])
        dst = np.float32([kp2[m.trainIdx].pt for m in good])

        E, inlier_mask = cv2.findEssentialMat(
            src, dst, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0
        )
        if E is None:
            return np.eye(3, dtype=np.float32), np.zeros(2, dtype=np.float32)

        _, R, t, pose_mask = cv2.recoverPose(E, src, dst, self.K, mask=inlier_mask)

        H = self._project_rotation_to_homography(R)
        bg_residual = self._compute_residual(src, dst, H, pose_mask)
        return H, bg_residual

    def _build_foreground_mask(self, shape, prev_bboxes):
        if not prev_bboxes:
            return None
        h, w = shape
        mask = np.ones((h, w), dtype=np.uint8) * 255
        for bbox in prev_bboxes:
            x1, y1, x2, y2 = map(int, bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = 0
        return mask

    def _lowe_ratio_matches(self, des1, des2, ratio: float = 0.7):
        matches = self.matcher.knnMatch(des1, des2, k=2)
        good = []
        for pair in matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < ratio * n.distance:
                    good.append(m)
            elif len(pair) == 1:
                good.append(pair[0])
        return good

    def _project_rotation_to_homography(self, R: np.ndarray) -> np.ndarray:
        H_inf = self.K @ R @ self.K_inv
        H_inf = H_inf / H_inf[2, 2]
        return H_inf.astype(np.float32)

    def _compute_residual(
        self,
        src: np.ndarray,
        dst: np.ndarray,
        H: np.ndarray,
        pose_mask: Optional[np.ndarray],
    ) -> np.ndarray:
        if pose_mask is None or pose_mask.sum() == 0:
            return np.zeros(2, dtype=np.float32)
        idx = pose_mask.ravel().astype(bool)
        if idx.sum() == 0:
            return np.zeros(2, dtype=np.float32)
        src_in = src[idx]
        dst_in = dst[idx]
        ones = np.ones((src_in.shape[0], 1), dtype=np.float32)
        src_h = np.hstack([src_in, ones])
        warped = (H @ src_h.T).T
        warped = warped[:, :2] / warped[:, 2:3]
        residuals = np.abs(dst_in - warped)
        return np.median(residuals, axis=0).astype(np.float32)


register_ego_router("recoverpose", RecoverPoseEgoRouter)
