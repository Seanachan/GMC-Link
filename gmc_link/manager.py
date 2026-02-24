import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

from .utils import normalize_velocity, MotionBuffer, ScoreBuffer
from .alignment import MotionLanguageAligner


class GMCLinkManager:
    """
    GMC-Link Manager: Orchestrates Motion Estimation and Reasoning modules.
    
    Uses centroid-difference velocities from the tracker for motion estimation,
    matching the training pipeline for consistent performance.
    """
    def __init__(
        self, 
        weights_path: str = None, 
        device: str = 'cpu', 
        lang_dim: int = 384
    ) -> None:
        self.device = device
        
        self.motion_buffer = MotionBuffer(alpha=0.3)
        self.score_buffer = ScoreBuffer(alpha=0.4)
        self.aligner = MotionLanguageAligner(lang_dim=lang_dim, embed_dim=256).to(device)
        
        if weights_path:
            self.aligner.load_state_dict(torch.load(weights_path, map_location=device))
        self.aligner.eval()
        
        # Store previous centroids for velocity computation
        self.prev_centroids: Dict[int, np.ndarray] = {}

    def process_frame(
        self, 
        frame: np.ndarray, 
        active_tracks: List[Any], 
        language_embedding: torch.Tensor,
        detections: Optional[np.ndarray] = None
    ) -> Tuple[Dict[int, float], Dict[int, np.ndarray]]:
        """
        Process a frame: compute centroid-difference velocities per tracked object,
        and return alignment scores against a language prompt.
        
        Args:
            frame: (H, W, 3) The current video frame.
            active_tracks: List of track objects (must have `id` and `centroid`).
            language_embedding: (1, L_dim) Tensor representing the language prompt.
            detections: (N, 4) array of [x1, y1, x2, y2] bounding boxes (unused, kept for API compat).
        """
        if not active_tracks: 
            return {}, {}

        img_h, img_w = frame.shape[:2]
        frame_shape = (img_h, img_w)

        track_ids = []
        compensated_velocities = []

        for track in active_tracks:
            if not hasattr(track, 'centroid') or track.centroid is None:
                continue

            tid = track.id
            curr_centroid = np.array(track.centroid, dtype=np.float64)

            if tid in self.prev_centroids:
                # Centroid-difference velocity (pixels/frame)
                raw_velocity = curr_centroid - self.prev_centroids[tid]
                norm_velocity = normalize_velocity(raw_velocity, frame_shape)
                smoothed_v = self.motion_buffer.smooth(tid, norm_velocity)
            else:
                # First appearance: zero velocity
                smoothed_v = np.zeros(2, dtype=np.float32)

            self.prev_centroids[tid] = curr_centroid
            track_ids.append(tid)
            compensated_velocities.append(smoothed_v)

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
            smoothed_score = self.score_buffer.smooth(tid, float(raw_scores[i]))
            scores_dict[tid] = smoothed_score
            velocities_dict[tid] = compensated_velocities[i]

        # Clean up dead tracks
        active_ids = set(track_ids)
        self.motion_buffer.clear_dead_tracks(track_ids)
        self.score_buffer.clear_dead_tracks(track_ids)
        dead_centroids = set(self.prev_centroids.keys()) - active_ids
        for d in dead_centroids:
            del self.prev_centroids[d]

        return scores_dict, velocities_dict