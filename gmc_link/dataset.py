import torch
from torch.utils.data import Dataset
import numpy as np


class MotionLanguageDataset(Dataset):
    """
    A PyTorch Dataset for loading motion and language data for GMC-Link.
    Each sample consists of:
    - motion_features: (N, 2) Tensor of normalized world velocities [dx, dy] for N tracks.
    - language_features: (1, L_dim) or (N, L_dim) Tensor of text features from the language model.
    """

    def __init__(self, motion_data, language_data):
        """
        Args:
            motion_data: List of Numpy arrays or Tensors containing motion features for each track.
            language_data: List of Numpy arrays or Tensors containing language features for each track or a single description.
        """
        self.motion_data = motion_data
        self.language_data = language_data

    def __len__(self):
        return len(self.motion_data)

    def __getitem__(self, idx):
        motion_features = torch.tensor(self.motion_data[idx], dtype=torch.float32)
        language_features = torch.tensor(self.language_data[idx], dtype=torch.float32)
        return motion_features, language_features

    def collate_fn(self, batch):
        """
        Standarize how data is grouped into batches. 
        Required because alignment.py expects (N,2) for motion and (N, L_dim) or (1, L_dim) for language.
        """
        # item[0] is the motion vector [dx, dy]
        # item[1] is the language embedding

        # Use torch.stack with dim=0 to create a single 2D tensor for the whole batch
        motion_batch = torch.stack([item[0] for item in batch], dim=0)     # Shape: (Batch_Size, 2)
        language_batch = torch.stack([item[1] for item in batch], dim=0)   # Shape: (Batch_Size, L_dim)

        return motion_batch, language_batch
