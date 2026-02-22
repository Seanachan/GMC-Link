'''
Take stabilized velocity vector from utils.py, and align it with language features from the language model using a small MLP.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MotionLanguageAligner(nn.Module):
    """
    Reasoning Head of GMC-Link that aligns motion features with language features.
    Args:
      motion_dim: The dimensionality of the motion features.
      language_dim: The dimensionality of the language features.
      hidden_dim: The dimensionality of the hidden layer for alignment.
    """
    def __init__(self, vis_dim=512, lang_dim=768, embed_dim=256):
        super().__init__()
        # Motion Encoder: Project (dx, dy) into a semantic vector.
        # Use small MLP because 'meaning' of motion is non-linear.
        self.motion_projector = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        self.lang_projector = nn.Sequential(
            nn.Linear(lang_dim, embed_dim),
            nn.ReLU(),
            nn.LayerNorm(embed_dim)
        )

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))  # Learnable temperature parameter for scaling

    def forward(self, motion_feats, lang_feats):
        """
        The 'Thinking' phase: Link geometric motion to linguistic intent.

        Args:
            motion_feats: (N, 2) Tensor - The normalized world velocities [dx, dy].
            lang_feats: (1, L_dim) or (N, L_dim) Tensor - The text features. (1,L_dim) if using a single language description for all tracks, or (N, L_dim) if each track has its own description.

        Returns:
            aligment_logits: (N, N) Matrix of simlilarity scores between motion and language features, where M is the number of language features (e.g., tokens).
        """

        # 1. Project to Shared Latent Space (latent -> 'thought space'), now having same dim (256)
        motion_latents = self.motion_projector(motion_feats)
        language_latents = self.lang_projector(lang_feats)

        # 2. L2 Normalization
        # Standardizes vectors to a length of 1.0 for Cosine Similarity, both (N, 256)  now.
        motion_latents = F.normalize(motion_latents, p=2, dim=-1)
        language_latents = F.normalize(language_latents, p=2, dim=-1)

        # 3. Compute Similarity (The Alignment)
        # (N, shared_dim) @ (shared_dim, M) -> (N, M)
        raw_similarity = torch.matmul(motion_latents, language_latents.t())

        # 4. Temperature Scaling
        # Sharpen the scores so the best match is mathematically distinct
        alignment_logits = raw_similarity * self.logit_scale.exp()

        return alignment_logits

