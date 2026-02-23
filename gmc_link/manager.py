import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

from .core import DenseFlowEngine, extract_object_velocity, extract_background_flow
from .utils import normalize_velocity, MotionBuffer
from .alignment import MotionLanguageAligner


class GMCLinkManager:
    """
    GMC-Link Manager: Orchestrates Dense Optical Flow, Compensation, and Reasoning modules.
    Processes each frame and active tracks, returning alignment scores against a language prompt.
    """
    def __init__(
        self, 
        weights_path: str = None, 
        device: str = 'cpu', 
        lang_dim: int = 384
    ) -> None:
        self.device = device
        
        self.flow_engine = DenseFlowEngine()
        self.motion_buffer = MotionBuffer(alpha=0.3)
        self.aligner = MotionLanguageAligner(lang_dim=lang_dim, embed_dim=256).to(device)
        
        if weights_path:
            self.aligner.load_state_dict(torch.load(weights_path, map_location=device))
        self.aligner.eval()

    def process_frame(
        self, 
        frame: np.ndarray, 
        active_tracks: List[Any], 
        language_embedding: torch.Tensor,
        detections: Optional[np.ndarray] = None
    ) -> Tuple[Dict[int, float], Dict[int, np.ndarray]]:
        """
        Process a frame: compute dense optical flow, extract per-object world velocities,
        and return alignment scores.
        
        Args:
            frame: (H, W, 3) The current video frame.
            active_tracks: List of track objects (must have `id` and `centroid`).
            language_embedding: (1, L_dim) Tensor representing the language prompt.
            detections: (N, 4) array of [x1, y1, x2, y2] bounding boxes for all tracked objects.
        """
        if not active_tracks: 
            return {}, {}

        # 1. Compute dense optical flow
        flow = self.flow_engine.estimate(frame)
        if flow is None:
            return {}, {}  # First frame, no flow yet

        img_h, img_w = frame.shape[:2]
        frame_shape = (img_h, img_w)

        # 2. Estimate camera motion from background flow (excluding objects)
        bboxes = []
        if detections is not None:
            bboxes = [tuple(d) for d in detections]
        bg_flow = extract_background_flow(flow, bboxes, frame_shape)

        # 3. Per-object: extract flow inside bbox, subtract background = world velocity
        track_ids = []
        compensated_velocities = []

        for track in active_tracks:
            if not hasattr(track, 'centroid') or track.centroid is None:
                continue

            # Find this track's bbox from detections
            bbox = self._find_bbox_for_track(track, detections)
            if bbox is None:
                continue

            # Average flow inside the object's bounding box
            obj_flow = extract_object_velocity(flow, bbox, frame_shape)

            # World velocity = object's apparent motion - camera motion
            world_velocity = obj_flow - bg_flow

            # Normalize and smooth
            norm_velocity = normalize_velocity(world_velocity, frame_shape)
            smoothed_v = self.motion_buffer.smooth(track.id, norm_velocity)

            track_ids.append(track.id)
            compensated_velocities.append(smoothed_v)

        if not compensated_velocities:
            return {}, {}

        # 4. Align motion with language
        motion_tensor = torch.tensor(
            np.array(compensated_velocities), dtype=torch.float32
        ).to(self.device)

        with torch.no_grad():
            logits = self.aligner(motion_tensor, language_embedding.to(self.device))

        self.motion_buffer.clear_dead_tracks(track_ids)

        scores = torch.sigmoid(logits).cpu().numpy().flatten()
        scores_dict = dict(zip(track_ids, scores))
        velocities_dict = dict(zip(track_ids, compensated_velocities))
        return scores_dict, velocities_dict

    def _find_bbox_for_track(
        self, track: Any, detections: Optional[np.ndarray]
    ) -> Optional[Tuple[float, float, float, float]]:
        """Find the bounding box closest to a track's centroid."""
        if detections is None or len(detections) == 0:
            return None

        cx, cy = track.centroid
        best_bbox = None
        best_dist = float('inf')

        for det in detections:
            x1, y1, x2, y2 = det
            det_cx = (x1 + x2) / 2
            det_cy = (y1 + y2) / 2
            dist = (cx - det_cx) ** 2 + (cy - det_cy) ** 2
            if dist < best_dist:
                best_dist = dist
                best_bbox = (x1, y1, x2, y2)

        return best_bbox