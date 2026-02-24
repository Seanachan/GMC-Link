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
    """
    def __init__(self, motion_dim: int = 6, lang_dim: int = 768, embed_dim: int = 256) -> None:
        super().__init__()
        # Motion Encoder: Project (dx, dy, cx, cy, w, h) into a semantic vector.
        # Deeper MLP to learn nuanced motion semantics (e.g., turning vs moving forward)
        self.motion_projector = nn.Sequential(
            nn.Linear(motion_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        self.lang_projector = nn.Sequential(
            nn.Linear(lang_dim, embed_dim),
            nn.ReLU(),
            nn.LayerNorm(embed_dim)
        )

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))  # Learnable temperature parameter for scaling

    def forward(self, motion_feats: torch.Tensor, lang_feats: torch.Tensor) -> torch.Tensor:
        """
        The 'Thinking' phase: Link geometric motion to linguistic intent.

        Args:
            motion_feats: (N, 6) Tensor of normalized world velocities [dx, dy] and positions [cx, cy, w, h].
            lang_feats: (1, L_dim) Tensor of text features representing the prompt.

        Returns:
            alignment_logits: (N, 1) Matrix of similarity scores between each motion and the language concept.
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

    def score_pairs(self, motion_feats: torch.Tensor, lang_feats: torch.Tensor) -> torch.Tensor:
        """
        Compute per-pair similarity scores for BCE training.
        
        Args:
            motion_feats: (N, 6) Tensor of velocity and position vectors.
            lang_feats: (N, L_dim) Tensor of language embeddings (one per motion).
            
        Returns:
            scores: (N,) Tensor of scalar similarity scores per pair.
        """
        motion_latents = F.normalize(self.motion_projector(motion_feats), p=2, dim=-1)
        language_latents = F.normalize(self.lang_projector(lang_feats), p=2, dim=-1)
        
        # Element-wise dot product → per-pair cosine similarity
        scores = (motion_latents * language_latents).sum(dim=-1)
        scores = scores * self.logit_scale.exp()
        return scores
