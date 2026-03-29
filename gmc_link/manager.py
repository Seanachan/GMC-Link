"""
Inference Manager linking tracks to text prompts using physical vectors and spatial models.

UPDATED: Now uses cumulative homography method for better numerical stability
and debugging capabilities.
"""
from typing import Dict, List, Tuple, Any, Optional
from collections import deque
import torch
import numpy as np

from .utils import (
    normalize_velocity,
    MotionBuffer,
    ScoreBuffer,
    warp_points,
    VELOCITY_SCALE,
)
from .alignment import MotionLanguageAligner
from .core import ORBHomographyEngine


class GMCLinkManager:
    """
    GMC-Link Manager: Orchestrates Motion Estimation and Reasoning modules.

    Uses cumulative homography method:
    - Stores ORIGINAL centroid coordinates (never modified)
    - Stores CUMULATIVE homographies: H[t] transforms frame[t] -> current_frame
    - Warps coordinates ONCE when computing velocities (not every frame)
    
    This provides better numerical stability and debugging capabilities.
    """

    # Multi-scale frame gaps matching training
    FRAME_GAPS = [2, 5, 10]  # short, mid, long

    def __init__(
        self,
        weights_path: str = None,
        device: str = "cpu",
        lang_dim: int = 384,
        frame_gap: int = 10,  # max gap for buffer sizing
        fx: float = 718.856,
        cx: float = 607.1928,
        x_range: float = 50.0,
        z_min: float = 2.0,
        z_max: float = 100.0,
    ) -> None:
        self.device = device
        self.frame_gap = frame_gap
        self.fx = fx
        self.cx = cx
        self.x_range = x_range
        self.z_min = z_min
        self.z_max = z_max
        self.motion_buffer = MotionBuffer(alpha=0.3)
        self.score_buffer = ScoreBuffer(alpha=0.4)
        self.aligner = MotionLanguageAligner(
            motion_dim=13, lang_dim=lang_dim, embed_dim=256
        ).to(device)

        self.temperature = 1.0  # default (no scaling)
        if weights_path:
            checkpoint = torch.load(weights_path, map_location=device)
            if isinstance(checkpoint, dict) and "model" in checkpoint:
                self.aligner.load_state_dict(checkpoint["model"])
                self.temperature = checkpoint.get("temperature", 1.0)
            else:
                self.aligner.load_state_dict(checkpoint)
        self.aligner.eval()

        self.ego_engine = ORBHomographyEngine(max_features=1500)
        self.prev_frame = None
        self.prev_detections = None

        # CUMULATIVE HOMOGRAPHY: Store original coordinates (never warped)
        self.centroid_history: Dict[int, deque] = {}
        self.wh_history: Dict[int, deque] = {}

        # Store cumulative homographies: H[i] warps frame[t-i] -> current frame
        self.homography_buffer: deque = deque(maxlen=frame_gap + 1)

        # Background residual buffer for noise floor estimation
        self.bg_residual_buffer: deque = deque(maxlen=frame_gap + 1)

    @staticmethod
    def _estimate_depth_from_bbox(bbox_h: float) -> float:
        # Fast monocular proxy: taller box => closer object.
        if bbox_h <= 1e-3:
            return 30.0
        f = 720.0
        h_real = 1.5
        z = (f * h_real) / bbox_h
        return float(np.clip(z, 2.0, 100.0))

    def _pixel_to_xz(self, u: float, z: float) -> Tuple[float, float]:
        # Back-project image x into camera-centric ground-plane X using depth Z.
        x = (u - self.cx) * z / self.fx
        return float(x), float(z)

    def _normalize_xz(self, X: float, Z: float) -> Tuple[float, float]:
        # Clamp to stable ranges so feature scale stays training-friendly.
        Xn = (np.clip(X, -self.x_range, self.x_range) + self.x_range) / (2.0 * self.x_range)
        Zn = (np.clip(Z, self.z_min, self.z_max) - self.z_min) / (self.z_max - self.z_min + 1e-6)
        return float(Xn), float(Zn)

    def process_frame(
        self,
        frame: np.ndarray,
        active_tracks: List[Any],
        language_embedding: torch.Tensor,
        detections: Optional[np.ndarray] = None,
        update_state: bool = True,
    ) -> Tuple[Dict[int, float], Dict[int, np.ndarray]]:
        """
        Process a frame: compute centroid-difference velocities per tracked object,
        and return alignment scores against a language prompt.

        Args:
            frame: (H, W, 3) The current video frame.
            active_tracks: List of track objects (must have `id` and `centroid`).
            language_embedding: (1, L_dim) Tensor representing the language prompt.
            detections: (N, 4) array of bounding boxes for ego-motion masking.
            update_state: Whether to update internal state (for multiple evaluations per frame)
        """
        if not active_tracks:
            return {}, {}

        img_h, img_w = frame.shape[:2]
        frame_shape = (img_h, img_w)

        # CUMULATIVE HOMOGRAPHY UPDATE
        if update_state:
            if self.prev_frame is not None:
                # Estimate H_{t-1 -> t} and background warp residual
                H_prev_to_curr, bg_residual = self.ego_engine.estimate_homography(
                    self.prev_frame, frame, self.prev_detections
                )
                self.bg_residual_buffer.append(bg_residual)

                # Update ALL cumulative homographies by composing with new homography
                updated_homographies = deque(maxlen=self.frame_gap + 1)
                for H_old in self.homography_buffer:
                    # H_old maps frame[t_old] -> frame[t-1]
                    # H_prev_to_curr maps frame[t-1] -> frame[t]
                    # Composition: frame[t_old] -> frame[t]
                    H_cumulative = H_prev_to_curr @ H_old
                    updated_homographies.append(H_cumulative)
                
                # Current frame has identity homography (maps to itself)
                updated_homographies.append(np.eye(3, dtype=np.float32))
                self.homography_buffer = updated_homographies
            else:
                # First frame: identity homography
                self.homography_buffer.append(np.eye(3, dtype=np.float32))

            self.prev_frame = frame.copy()
            if detections is not None:
                self.prev_detections = [tuple(d) for d in detections]
            else:
                self.prev_detections = None

        track_ids = []
        compensated_velocities = []

        for track in active_tracks:
            if not hasattr(track, "centroid") or track.centroid is None:
                continue

            tid = track.id
            curr_centroid = np.array(track.centroid, dtype=np.float64)

            if tid not in self.centroid_history:
                self.centroid_history[tid] = deque(maxlen=self.frame_gap + 1)
                self.wh_history[tid] = deque(maxlen=self.frame_gap + 1)

            # Store ORIGINAL coordinates (never warp!)
            if update_state:
                self.centroid_history[tid].append(curr_centroid)

            if hasattr(track, "bbox") and track.bbox is not None:
                bx1, by1, bx2, by2 = track.bbox
                curr_w = bx2 - bx1
                curr_h = by2 - by1
            else:
                curr_w, curr_h = 0.0, 0.0

            if update_state:
                self.wh_history[tid].append(np.array([curr_w, curr_h], dtype=np.float64))

            # Get original coordinate history
            centroid_hist = list(self.centroid_history[tid])
            wh_hist = list(self.wh_history[tid])

            if len(centroid_hist) > 1:
                # WARP ORIGINAL COORDINATES TO CURRENT FRAME (warp once!)
                T = len(centroid_hist)
                homographies = list(self.homography_buffer)[-T:]

                # Ensure we have enough homographies
                while len(homographies) < T:
                    homographies.insert(0, np.eye(3, dtype=np.float32))

                # Warp ORIGINAL coordinates to current frame
                world_frame_centroids = []
                for centroid, H in zip(centroid_hist, homographies):
                    warped = warp_points(np.array([centroid]), H)
                    world_frame_centroids.append(warped[0])
                
                world_frame_xz = []
                for c2d, wh2d in zip(world_frame_centroids, wh_hist):
                    # Convert homography-compensated 2D center to pseudo-3D (X,Z).
                    z_est = self._estimate_depth_from_bbox(float(wh2d[1]))
                    Xw, Zw = self._pixel_to_xz(float(c2d[0]), z_est)
                    world_frame_xz.append(np.array([Xw, Zw], dtype=np.float32))

                # Multi-scale velocity: compute dx, dy at each gap
                scale_velocities = []
                for gap in self.FRAME_GAPS:
                    if T > gap:
                        # Multi-scale world motion: delta in X/Z between two compensated times.
                        v_raw_xz = world_frame_xz[-1] - world_frame_xz[-(gap + 1)]
                        scale_velocities.append(v_raw_xz.astype(np.float32))
                    else:
                        scale_velocities.append(np.zeros(2, dtype=np.float32))

                # dw, dh from mid-scale (index 1, gap=5) or full history
                mid_gap = self.FRAME_GAPS[1]
                if T > mid_gap:
                    raw_dw_dh = wh_hist[-1] - wh_hist[-(mid_gap + 1)]
                else:
                    raw_dw_dh = wh_hist[-1] - wh_hist[0]
                dw_raw = raw_dw_dh[0] / float(img_w) * VELOCITY_SCALE
                dh_raw = raw_dw_dh[1] / float(img_h) * VELOCITY_SCALE

                # Smooth the full 8D multi-scale kinematic vector
                full_raw_v = np.array([
                    scale_velocities[0][0], scale_velocities[0][1],  # short
                    scale_velocities[1][0], scale_velocities[1][1],  # mid
                    scale_velocities[2][0], scale_velocities[2][1],  # long
                    dw_raw, dh_raw,
                ], dtype=np.float32)
                if update_state:
                    smoothed_v = self.motion_buffer.smooth(tid, full_raw_v)
                else:
                    if tid in self.motion_buffer.registry:
                        alpha = self.motion_buffer.alpha
                        smoothed_v = (alpha * full_raw_v) + ((1 - alpha) * self.motion_buffer.registry[tid])
                    else:
                        smoothed_v = full_raw_v
                dx_s, dy_s = smoothed_v[0], smoothed_v[1]
                dx_m, dy_m = smoothed_v[2], smoothed_v[3]
                dx_l, dy_l = smoothed_v[4], smoothed_v[5]
                dw, dh = smoothed_v[6], smoothed_v[7]
            else:
                # First appearance: zero velocity
                smoothed_v = np.zeros(8, dtype=np.float32)
                dx_s, dy_s = 0.0, 0.0
                dx_m, dy_m = 0.0, 0.0
                dx_l, dy_l = 0.0, 0.0
                dw, dh = 0.0, 0.0

            # Build 13D Multi-Scale Spatial-Motion Vector
            w_n = curr_w / float(img_w)
            h_n = curr_h / float(img_h)
            z_curr = self._estimate_depth_from_bbox(float(curr_h))
            X_curr, Z_curr = self._pixel_to_xz(float(curr_centroid[0]), z_curr)
            X_n, Z_n = self._normalize_xz(X_curr, Z_curr)

            # SNR from mid-scale velocity
            obj_speed = np.sqrt(dx_m ** 2 + dy_m ** 2)
            if len(self.bg_residual_buffer) > 0:
                bg_stack = np.array(list(self.bg_residual_buffer))
                bg_max = np.max(np.abs(bg_stack), axis=0)
                bg_magnitude = np.sqrt(
                    (bg_max[0] / float(img_w) * VELOCITY_SCALE) ** 2
                    + (bg_max[1] / float(img_h) * VELOCITY_SCALE) ** 2
                )
            else:
                bg_magnitude = 0.0
            snr = obj_speed / (bg_magnitude + 1e-6)

            spatial_motion = np.array(
                # 13D feature with world-space position cues (X_n, Z_n) instead of (cx, cy).
                [dx_s, dy_s, dx_m, dy_m, dx_l, dy_l, dw, dh,
                 X_n, Z_n, w_n, h_n, snr], dtype=np.float32
            )

            track_ids.append(tid)
            compensated_velocities.append(spatial_motion)

        if not compensated_velocities:
            return {}, {}

        # Align motion with language via cosine similarity
        motion_tensor = torch.tensor(
            np.array(compensated_velocities), dtype=torch.float32
        ).to(self.device)

        with torch.no_grad():
            motion_emb, lang_emb = self.aligner.encode(
                motion_tensor, language_embedding.to(self.device)
            )
            # Cosine similarity scaled by learned temperature → sigmoid to [0, 1]
            cosine_sim = torch.matmul(motion_emb, lang_emb.t()).flatten()
            raw_scores = torch.sigmoid(cosine_sim / self.temperature).cpu().numpy()

        # Apply score smoothing for temporal consistency
        scores_dict = {}
        velocities_dict = {}
        for i, tid in enumerate(track_ids):
            if update_state:
                smoothed_score = self.score_buffer.smooth(tid, float(raw_scores[i]))
            else:
                if tid in self.score_buffer.registry:
                    alpha = self.score_buffer.alpha
                    smoothed_score = alpha * float(raw_scores[i]) + (1 - alpha) * self.score_buffer.registry[tid]
                else:
                    smoothed_score = float(raw_scores[i])
            scores_dict[tid] = smoothed_score
            velocities_dict[tid] = compensated_velocities[i]

        if update_state:
            # Clean up dead tracks
            active_ids = set(track_ids)
            self.motion_buffer.clear_dead_tracks(track_ids)
            self.score_buffer.clear_dead_tracks(track_ids)
            dead_centroids = set(self.centroid_history.keys()) - active_ids
            for d in dead_centroids:
                del self.centroid_history[d]
                if d in self.wh_history:
                    del self.wh_history[d]

        return scores_dict, velocities_dict
