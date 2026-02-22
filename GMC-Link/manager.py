import torch
import numpy as np
from .core import GlobalMotion
from .utils import warp_points, normalize_velocity, MotionBuffer
from .alignment import MotionLanguageAligner

class GMCLinkManager:
    """
    GMC-Link Manager: The 'Brain' that orchestrates the Geometry, Compensation, and Reasoning modules. 
    This class is designed to be called from your main tracking loop, where it processes each frame and active tracks, returning alignment scores for each track against the language prompt.
    """
    def __init__(self, weights_path=None, device='cpu'):
        """
        Args:
            weights_path: Optional path to pre-trained weights for the MotionLanguageAligner.
            device: 'cpu' or 'cuda' for running the model.  Default is 'cpu' for compatibility.
        """
        self.device = device
        
        # 1. Initialize Components
        self.gmc_engine = GlobalMotion()
        self.motion_buffer = MotionBuffer(alpha=0.8) # Smoothes out jitter
        self.aligner = MotionLanguageAligner().to(device)
        
        # 2. Load trained knowledge if available
        if weights_path:
            self.aligner.load_state_dict(torch.load(weights_path, map_location=device))
        self.aligner.eval()

    def process_frame(self, frame, active_tracks, language_embedding):
        """
        Main method to process each frame and return alignment scores for active tracks.
        Args:
            frame: The current video frame (as a NumPy array).
            active_tracks: A list of track objects, each with attributes like 'id', 'centroid', and 'prev_centroid'.
            language_embedding: A tensor representing the language prompt embedding (e.g., from CLIP or BERT).
        Returns:
            A dictionary mapping track IDs to their alignment scores with the language prompt.  Scores are in the range [0, 1], where higher means more aligned with the prompt.
        """
        if not active_tracks: return {}

        # Geometric Motion Compensation (GMC) to find camera movement
        # Find how the camera moved between the last frame and this one
        homography = self.gmc_engine.estimate(frame)

        # Compensation: Warp previous positions to cancel out camera motion, then calculate smoothed velocities
        track_ids = []
        compensated_velocities = []

        for track in active_tracks:
            # We need the previous position to find the direction
            # If the track is new, we can't calculate motion yet
            if not hasattr(track, 'prev_centroid'):
                track.prev_centroid = track.centroid
                continue

            # Warp the previous point into the current frame's coordinate system
            # This cancels out the camera's panning/tilting
            warped_prev = warp_points(np.array([track.prev_centroid]), homography)[0]
            
            # Calculate world velocity: [current_pos - corrected_prev_pos]
            velocity = track.centroid - warped_prev
            
            # Normalize based on frame size (Direction-focused)
            norm_velocity = normalize_velocity(velocity, frame.shape[:2])
            
            # Smooth the direction using the buffer
            smoothed_v = self.motion_buffer.smooth(track.id, norm_velocity)
            
            track_ids.append(track.id)
            compensated_velocities.append(smoothed_v)

        if not compensated_velocities:
            return {}

        # Reasoning: Align the compensated motion with the language embedding
        # Convert list of motions to a (N, 2) tensor
        motion_tensor = torch.tensor(np.array(compensated_velocities), dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            # Get alignment scores for all objects against the prompt
            # Output shape: (N, 1)
            logits = self.aligner(motion_tensor, language_embedding.to(self.device))

        # Clean up old tracks from the motion buffer to prevent memory bloat
        self.motion_buffer.clear_dead_tracks(track_ids)

        # Return a dictionary of {track_id: score} for easy look-up in your tracker
        scores = logits.cpu().numpy().flatten()
        return dict(zip(track_ids, scores))