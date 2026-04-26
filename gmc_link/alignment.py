"""
Take stabilized velocity vector from utils.py, and align it
with language features from the language model using a small MLP or transformer.
"""

from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F


class TemporalMotionEncoder(nn.Module):
    """
    Transformer-based motion encoder that processes a sequence of T per-frame
    motion vectors and produces a single embedding via a [CLS] token.
    """

    def __init__(
        self,
        motion_dim: int = 13,
        d_model: int = 64,
        n_head: int = 4,
        dim_feedforward: int = 128,
        num_layers: int = 1,
        seq_len: int = 10,
        embed_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len

        # Normalize inputs before projection — SNR and velocity can have extreme
        # outliers (e.g. snr → ∞ when ego-motion ≈ 0) that would blow up attention
        # softmax and produce NaN gradients.
        self.input_norm = nn.LayerNorm(motion_dim)

        # Per-frame projection: 13D → d_model
        self.input_proj = nn.Linear(motion_dim, d_model)

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Learnable positional encoding for T+1 positions (CLS + T frames)
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len + 1, d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Output projection: d_model → embed_dim
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(
        self, x: torch.Tensor, padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, T, motion_dim) — sequence of per-frame motion vectors
            padding_mask: (batch, T+1) bool — True=padded/ignore, False=valid
                          Length T+1 because [CLS] is prepended. CLS position
                          should always be False (valid).
        Returns:
            (batch, embed_dim) — embedding from [CLS] token output
        """
        B, T, _ = x.shape

        # Normalize per-frame features, then project to d_model
        x = self.input_norm(x)
        x = self.input_proj(x)  # (B, T, d_model)

        # Prepend [CLS] token
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1)  # (B, T+1, d_model)

        # Add positional encoding (handle variable T <= seq_len)
        x = x + self.pos_encoding[:, : T + 1, :]

        # Transformer encoder with optional padding mask
        x = self.transformer(x, src_key_padding_mask=padding_mask)

        # Take [CLS] output (position 0)
        cls_out = x[:, 0, :]  # (B, d_model)

        # Project to shared embedding space
        return self.output_proj(cls_out)  # (B, embed_dim)


class MotionLanguageAligner(nn.Module):
    """
    Reasoning Head of GMC-Link that aligns motion features with language features.

    Supports two architectures:
    - "mlp": single-frame MLP projector (original, default)
    - "temporal_transformer": sequence-based transformer encoder with [CLS] token

    Trained with Supervised InfoNCE. At inference, use encode() to get L2-normalized
    embeddings and compute cosine similarity directly.
    """

    def __init__(
        self,
        motion_dim: int = 13,
        lang_dim: int = 384,
        embed_dim: int = 256,
        architecture: str = "mlp",
        seq_len: int = 10,
    ) -> None:
        super().__init__()
        self.architecture = architecture

        if architecture == "temporal_transformer":
            self.motion_projector = TemporalMotionEncoder(
                motion_dim=motion_dim,
                seq_len=seq_len,
                embed_dim=embed_dim,
            )
        else:
            # Original MLP: 13D → 128 → 256 → 512 → embed_dim
            self.motion_projector = nn.Sequential(
                nn.Linear(motion_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, embed_dim),
                nn.LayerNorm(embed_dim),
            )

        # Language Projector: two-layer projection with bottleneck (unchanged)
        self.lang_projector = nn.Sequential(
            nn.Linear(lang_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def encode(
        self,
        motion_feats: torch.Tensor,
        lang_feats: torch.Tensor,
        padding_mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project motion and language inputs into the shared latent space.

        Args:
            motion_feats: (N, motion_dim) for MLP or (N, T, motion_dim) for transformer.
            lang_feats:   (N, lang_dim) or (M, lang_dim) language embeddings.
            padding_mask: (N, T+1) bool tensor for transformer. None for MLP.

        Returns:
            motion_emb:   (N, embed_dim) L2-normalized motion embeddings.
            lang_emb:     (N, embed_dim) or (M, embed_dim) L2-normalized language embeddings.
        """
        if self.architecture == "temporal_transformer":
            motion_emb = F.normalize(
                self.motion_projector(motion_feats, padding_mask), p=2, dim=-1
            )
        else:
            motion_emb = F.normalize(self.motion_projector(motion_feats), p=2, dim=-1)
        lang_emb = F.normalize(self.lang_projector(lang_feats), p=2, dim=-1)
        return motion_emb, lang_emb

    def forward(
        self,
        motion_feats: torch.Tensor,
        lang_feats: torch.Tensor,
        padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute cosine similarity scores between motion and language embeddings.

        Args:
            motion_feats: (N, motion_dim) for MLP or (N, T, motion_dim) for transformer.
            lang_feats:   (M, L_dim) language embeddings.
            padding_mask: (N, T+1) bool tensor for transformer. None for MLP.

        Returns:
            scores: (N, M) cosine similarity in [-1, 1].
        """
        motion_emb, lang_emb = self.encode(motion_feats, lang_feats, padding_mask)
        return torch.matmul(motion_emb, lang_emb.t())
