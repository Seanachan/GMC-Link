# GMC-Link/core.py
import cv2
import torch
import numpy as np
from typing import Optional, List, Tuple
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
from torchvision.transforms import functional as F


class RAFTFlowEngine:
    """
    Compute dense optical flow using RAFT (Recurrent All-Pairs Field Transforms).
    GPU-accelerated via MPS/CUDA. Produces significantly more accurate flow
    than classical methods like Farneback.
    """
    def __init__(self, device: str = "cpu") -> None:
        self.device = torch.device(device)
        weights = Raft_Small_Weights.DEFAULT
        self.model = raft_small(weights=weights).to(self.device)
        self.model.eval()
        self.transforms = weights.transforms()
        self.prev_tensor: Optional[torch.Tensor] = None

    @torch.no_grad()
    def estimate(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Compute dense optical flow between the current and previous frame using RAFT.
        Automatically pads frames to be divisible by 8 (RAFT requirement).
        
        Args:
            frame: (H, W, 3) Current video frame in BGR format.
            
        Returns:
            flow: (H, W, 2) per-pixel flow field [dx, dy], or None for the first frame.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().to(self.device)
        orig_h, orig_w = tensor.shape[1], tensor.shape[2]

        if self.prev_tensor is None:
            self.prev_tensor = tensor
            return None

        # Pad to multiple of 8 (RAFT requirement)
        pad_h = (8 - orig_h % 8) % 8
        pad_w = (8 - orig_w % 8) % 8
        if pad_h > 0 or pad_w > 0:
            prev_padded = torch.nn.functional.pad(self.prev_tensor, (0, pad_w, 0, pad_h))
            curr_padded = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h))
        else:
            prev_padded = self.prev_tensor
            curr_padded = tensor

        img1_batch, img2_batch = self.transforms(
            prev_padded.unsqueeze(0), curr_padded.unsqueeze(0)
        )

        flow_predictions = self.model(img1_batch, img2_batch)
        flow = flow_predictions[-1]  # (1, 2, H_pad, W_pad)

        self.prev_tensor = tensor

        # Crop back to original dimensions and convert to numpy
        return flow[0, :, :orig_h, :orig_w].permute(1, 2, 0).cpu().numpy()


def extract_object_velocity(
    flow: np.ndarray,
    bbox: Tuple[float, float, float, float],
    frame_shape: Tuple[int, int],
) -> np.ndarray:
    """
    Average the dense flow vectors inside a bounding box to get that object's apparent motion.
    """
    h, w = frame_shape
    x1 = int(max(0, bbox[0]))
    y1 = int(max(0, bbox[1]))
    x2 = int(min(w, bbox[2]))
    y2 = int(min(h, bbox[3]))

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
    """
    h, w = frame_shape
    mask = np.ones((h, w), dtype=bool)

    for bbox in bboxes:
        x1 = int(max(0, bbox[0]))
        y1 = int(max(0, bbox[1]))
        x2 = int(min(w, bbox[2]))
        y2 = int(min(h, bbox[3]))
        mask[y1:y2, x1:x2] = False

    bg_pixels = flow[mask]
    if len(bg_pixels) == 0:
        return np.zeros(2, dtype=np.float32)

    return np.median(bg_pixels, axis=0).astype(np.float32)