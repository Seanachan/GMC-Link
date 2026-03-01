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

    def __init__(
        self,
        weights_path: str = None,
        device: str = "cpu",
        lang_dim: int = 384,
        frame_gap: int = 5,
    ) -> None:
        self.device = device
        self.frame_gap = frame_gap

        self.motion_buffer = MotionBuffer(alpha=0.3)
        self.score_buffer = ScoreBuffer(alpha=0.4)
        self.aligner = MotionLanguageAligner(
            motion_dim=8, lang_dim=lang_dim, embed_dim=256
        ).to(device)

        if weights_path:
            self.aligner.load_state_dict(torch.load(weights_path, map_location=device))
        self.aligner.eval()

        self.ego_engine = ORBHomographyEngine(max_features=1500)
        self.prev_frame = None
        self.prev_detections = None

        # CUMULATIVE HOMOGRAPHY: Store original coordinates (never warped)
        self.centroid_history: Dict[int, deque] = {}
        self.wh_history: Dict[int, deque] = {}
        
        # Store cumulative homographies: H[i] warps frame[t-i] -> current frame
        self.homography_buffer: deque = deque(maxlen=frame_gap + 1)

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
                # Estimate H_{t-1 -> t}
                H_prev_to_curr = self.ego_engine.estimate_homography(
                    self.prev_frame, frame, self.prev_detections
                )
                
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
                
                # Compute velocity from world-frame coordinates
                world_frame_first = world_frame_centroids[0]
                world_frame_last = world_frame_centroids[-1]
                
                raw_velocity = world_frame_last - world_frame_first
                norm_velocity = normalize_velocity(raw_velocity, frame_shape)

                # Z-axis depth scaling velocity (dw, dh)
                raw_dw_dh = wh_hist[-1] - wh_hist[0]
                dw_raw = raw_dw_dh[0] / float(img_w) * VELOCITY_SCALE
                dh_raw = raw_dw_dh[1] / float(img_h) * VELOCITY_SCALE

                # Smooth the FULL 4D kinematic vector to absorb YOLO jitter
                full_raw_v = np.array(
                    [norm_velocity[0], norm_velocity[1], dw_raw, dh_raw],
                    dtype=np.float32,
                )
                if update_state:
                    smoothed_v = self.motion_buffer.smooth(tid, full_raw_v)
                else:
                    if tid in self.motion_buffer.registry:
                        alpha = self.motion_buffer.alpha
                        smoothed_v = (alpha * full_raw_v) + ((1 - alpha) * self.motion_buffer.registry[tid])
                    else:
                        smoothed_v = full_raw_v
                dx, dy, dw, dh = (
                    smoothed_v[0],
                    smoothed_v[1],
                    smoothed_v[2],
                    smoothed_v[3],
                )
            else:
                # First appearance: zero velocity
                smoothed_v = np.zeros(4, dtype=np.float32)
                dx, dy, dw, dh = 0.0, 0.0, 0.0, 0.0

            # Build 8D Spatial-Motion Vector
            w_n = curr_w / float(img_w)
            h_n = curr_h / float(img_h)
            cx_n = curr_centroid[0] / float(img_w)
            cy_n = curr_centroid[1] / float(img_h)

            spatial_motion = np.array(
                [dx, dy, dw, dh, cx_n, cy_n, w_n, h_n], dtype=np.float32
            )

            track_ids.append(tid)
            compensated_velocities.append(spatial_motion)

        if not compensated_velocities:
            return {}, {}

        # Align motion with language
        motion_tensor = torch.tensor(
            np.array(compensated_velocities), dtype=torch.float32
        ).to(self.device)

        with torch.no_grad():
            logits = self.aligner(motion_tensor, language_embedding.to(self.device))

        raw_scores = torch.sigmoid(logits).cpu().numpy().flatten()

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
