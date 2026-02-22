# GMC-Link/core.py
import cv2
import numpy as np


class GlobalMotion:
    def __init__(self, method="orb", max_features=500):
        """
        Purpose: Detect camera movement while ignoring tracked objects.
        Args:
            method: The feature detection method to use ("orb" or "sift"). Lowe's ratio test is applied to reduce the noise in urban repetitive scenes.
            max_features: The maximum number of features to detect.
        """
        self.method = method.lower()
        if method == "orb":
            self.detector = cv2.ORB_create(nfeatures=max_features)
            self.matcher = cv2.BFMatcher(
                cv2.NORM_HAMMING, crossCheck=False
            )  # since orb uses binary descriptors, we use NORM_HAMMING for matching
        elif method == "sift":
            self.detector = cv2.SIFT_create(nfeatures=max_features)
            self.matcher = cv2.BFMatcher(
                cv2.NORM_L2, crossCheck=False
            )  # since sift uses float descriptors, we use NORM_L2 for matching
        else:
            raise ValueError("Unsupported method: {}".format(method))

        self.prev_gray = None
        self.prev_keypoints = None
        self.prev_descriptors = None

    def _update_state(self, gray, kp, des):
        """Internal state update for the next frame."""
        self.prev_gray = gray
        self.prev_keypoints = kp
        self.prev_descriptors = des

    def estimate(self, frame, detections=None):
        """
        Estimate the global motion between the current frame and the previous frame.
        Args:
            frame: The current video frame (BGR format).
            detections: A list of bounding boxes for detected objects in the current frame, in the format [(x1, y1, x2, y2)] to be masked out.
        Returns:
            A 3x3 homography matrix representing the global motion, or np.eye(3) if estimation fails.
        """

        # Grayscale for feature extraction
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Mask out detected objects, focusing on background features for motion estimation
        mask = np.ones_like(gray, dtype=np.uint8) * 255
        if detections is not None and len(detections) > 0:
            for det in detections:
                x1, y1, x2, y2 = map(int, det)

                # Expand by 5 pixels to ensure we exclude features near the edges of the bounding box
                cv2.rectangle(mask, (x1 - 5, y1 - 5), (x2 + 5, y2 + 5), 0, -1)

        # Feature Extraction
        keypoints, descriptors = self.detector.detectAndCompute(gray, mask=mask)

        if self.prev_gray is None or descriptors is None or self.prev_descriptors is None:
            self._update_state(gray, keypoints, descriptors)
            return np.eye(3, dtype=np.float32)  # No motion for the first frame, return identity matrix
        
        # Lowe's Ratio Test (knnMatch)
        try:
            matches = self.matcher.knnMatch(descriptors, self.prev_descriptors, k=2)
        except cv2.error:
            # In case of an error during matching (e.g., no descriptors), return identity matrix
            self._update_state(gray, keypoints, descriptors)
            return np.eye(3, dtype=np.float32)

        good_matches = []
        for match_pair in matches:
            # knnMatch may return fewer than k matches for some descriptors
            if len(match_pair) == 2:
                m, n = match_pair
                # 0.75 is a common threshold for Lowe's ratio test
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

        # Homography Estimation with RANSAC
        if len(good_matches) < 4:
            self._update_state(gray, keypoints, descriptors)
            return np.eye(3, dtype=np.float32)  # Not enough matches, return identity matrix

        src_pts = np.float32([self.prev_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        self._update_state(gray, keypoints, descriptors)

        if H is None:
            return np.eye(3, dtype=np.float32)  # If homography estimation fails, return identity matrix
        
        return H.astype(np.float32)