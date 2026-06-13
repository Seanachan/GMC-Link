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

    Supports three architectures:
    - "mlp": single-frame MLP projector (original, default)
    - "temporal_transformer": sequence-based transformer encoder with [CLS] token
    - "shared_weight": per-modality Linear adapter → shared 2-hidden MLP → LN → L2
      (symmetric two-tower with shared nonlinear core; ~628k params, same as mlp)

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
        use_clip_feat: bool = False,
        clip_feat_dim: int = 512,
        clip_proj_dim: int = 64,
        fusion_site: str = "input_concat",
        lang_passthrough: bool = False,
        app_proj_dim: int = 256,
        identity_init_depth: bool = False,
    ) -> None:
        super().__init__()
        self.architecture = architecture
        self.use_clip_feat = use_clip_feat
        self.fusion_site = fusion_site
        self.lang_passthrough = lang_passthrough
        self.app_proj_dim = app_proj_dim
        self.motion_dim = motion_dim
        self.identity_init_depth = identity_init_depth

        if fusion_site not in ("input_concat", "late_concat"):
            raise ValueError(
                f"fusion_site must be 'input_concat' or 'late_concat', got {fusion_site!r}"
            )
        if fusion_site == "late_concat" and not use_clip_feat:
            raise ValueError("fusion_site='late_concat' requires use_clip_feat=True")

        if architecture == "shared_weight":
            if lang_passthrough:
                raise ValueError("shared_weight does not support lang_passthrough")
            if identity_init_depth:
                raise ValueError("shared_weight does not support identity_init_depth")
            if use_clip_feat and fusion_site != "input_concat":
                raise ValueError("shared_weight CLIP supports input_concat only")
            if use_clip_feat:
                self.clip_proj = nn.Linear(clip_feat_dim, clip_proj_dim)
                # Zero-init gate: clip_proj(x)=0 → concat=[motion, zeros] → bit-exact
                # to the no-CLIP shared_weight baseline at step 0.
                nn.init.zeros_(self.clip_proj.weight)
                nn.init.zeros_(self.clip_proj.bias)
                adapter_in = motion_dim + clip_proj_dim
            else:
                adapter_in = motion_dim
            self.motion_adapter = nn.Linear(adapter_in, embed_dim)
            self.lang_adapter = nn.Linear(lang_dim, embed_dim)
            self.shared_weight = nn.Sequential(
                nn.Linear(embed_dim, 768),
                nn.ReLU(),
                nn.Dropout(0.05),
                nn.Linear(768, 768),
                nn.ReLU(),
                nn.Dropout(0.05),
                nn.Linear(768, embed_dim),
                nn.LayerNorm(embed_dim),
            )
            self.out_dim = embed_dim
            return

        # ── Motion side ──────────────────────────────────────────────
        if use_clip_feat and fusion_site == "input_concat":
            self.clip_proj = nn.Linear(clip_feat_dim, clip_proj_dim)
            # Zero-init: clip_proj(x) = 0 → concat = [motion, zeros] → bit-exact
            # 13D-only baseline at step 0.
            nn.init.zeros_(self.clip_proj.weight)
            nn.init.zeros_(self.clip_proj.bias)
            effective_motion_dim = motion_dim + clip_proj_dim
        else:
            effective_motion_dim = motion_dim

        if architecture == "temporal_transformer":
            self.motion_projector = TemporalMotionEncoder(
                motion_dim=effective_motion_dim,
                seq_len=seq_len,
                embed_dim=embed_dim,
            )
        else:
            self.motion_projector = nn.Sequential(
                nn.Linear(effective_motion_dim, 128),
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

        # Identity-init depth columns: zero W[:, 13:motion_dim] of the first
        # Linear so the depth tail multiplies to zero at step 0. Bit-exact
        # gate vs a 13D aligner whose first-13 weight columns match.
        if (
            identity_init_depth
            and architecture == "mlp"
            and motion_dim > 13
            and not (use_clip_feat and fusion_site == "input_concat")
        ):
            with torch.no_grad():
                self.motion_projector[0].weight[:, 13:motion_dim].zero_()

        # ── Late-concat: parallel appearance arm ────────────────────
        if use_clip_feat and fusion_site == "late_concat":
            self.app_projector = nn.Sequential(
                nn.Linear(clip_feat_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, app_proj_dim),
                nn.LayerNorm(app_proj_dim),
            )
            # Zero-init the Linear before LN. LN(zeros) = zeros (β=0, γ=1
            # applied to (0-0)/sqrt(eps) = 0). At step 0: a256 = 0, combined
            # = [m256, 0_256] → identity gate against motion-only-with-zero-pad
            # reference.
            nn.init.zeros_(self.app_projector[3].weight)
            nn.init.zeros_(self.app_projector[3].bias)
            self.out_dim = embed_dim + app_proj_dim
        else:
            self.out_dim = embed_dim

        # ── Language side ───────────────────────────────────────────
        if lang_passthrough:
            if lang_dim != self.out_dim:
                raise ValueError(
                    f"lang_passthrough requires lang_dim ({lang_dim}) == "
                    f"out_dim ({self.out_dim})"
                )
            self.lang_projector = nn.Identity()
        else:
            self.lang_projector = nn.Sequential(
                nn.Linear(lang_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, self.out_dim),
                nn.LayerNorm(self.out_dim),
            )

    def encode_lang(self, lang_feats: torch.Tensor) -> torch.Tensor:
        """Project language features → L2-normalized embedding.
        Arch-dispatched so callers don't reach into `lang_projector` directly."""
        if self.architecture == "shared_weight":
            l = self.lang_adapter(lang_feats)
            return F.normalize(self.shared_weight(l), p=2, dim=-1)
        return F.normalize(self.lang_projector(lang_feats), p=2, dim=-1)

    def encode(
        self,
        motion_feats: torch.Tensor,
        lang_feats: torch.Tensor,
        padding_mask: torch.Tensor = None,
        clip_feats: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project motion and language inputs into the shared latent space.

        Args:
            motion_feats: (N, motion_dim) for MLP or (N, T, motion_dim) for transformer.
            lang_feats:   (N, lang_dim) or (M, lang_dim) language embeddings.
            padding_mask: (N, T+1) bool tensor for transformer. None for MLP.
            clip_feats:   (N, clip_feat_dim) MLP or (N, T, clip_feat_dim) transformer,
                          required when use_clip_feat=True. Concatenated to motion_feats
                          along the last dim after projection through clip_proj.

        Returns:
            motion_emb:   (N, embed_dim) L2-normalized motion embeddings.
            lang_emb:     (N, embed_dim) or (M, embed_dim) L2-normalized language embeddings.
        """
        if self.architecture == "shared_weight":
            if self.use_clip_feat:
                if clip_feats is None:
                    raise ValueError("clip_feats required when use_clip_feat=True")
                motion_feats = torch.cat(
                    [motion_feats, self.clip_proj(clip_feats)], dim=-1
                )
            m = self.motion_adapter(motion_feats)
            l = self.lang_adapter(lang_feats)
            motion_emb = F.normalize(self.shared_weight(m), p=2, dim=-1)
            lang_emb = F.normalize(self.shared_weight(l), p=2, dim=-1)
            return motion_emb, lang_emb

        if self.use_clip_feat and self.fusion_site == "input_concat":
            if clip_feats is None:
                raise ValueError("clip_feats required when use_clip_feat=True")
            clip_p = self.clip_proj(clip_feats)
            motion_feats = torch.cat([motion_feats, clip_p], dim=-1)

        if self.architecture == "temporal_transformer":
            m_out = self.motion_projector(motion_feats, padding_mask)
        else:
            m_out = self.motion_projector(motion_feats)

        if self.use_clip_feat and self.fusion_site == "late_concat":
            if clip_feats is None:
                raise ValueError("clip_feats required when fusion_site='late_concat'")
            a_out = self.app_projector(clip_feats)
            m_out = torch.cat([m_out, a_out], dim=-1)

        motion_emb = F.normalize(m_out, p=2, dim=-1)
        lang_emb = F.normalize(self.lang_projector(lang_feats), p=2, dim=-1)
        return motion_emb, lang_emb

    def forward(
        self,
        motion_feats: torch.Tensor,
        lang_feats: torch.Tensor,
        padding_mask: torch.Tensor = None,
        clip_feats: torch.Tensor = None,
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
        motion_emb, lang_emb = self.encode(
            motion_feats, lang_feats, padding_mask, clip_feats=clip_feats
        )
        return torch.matmul(motion_emb, lang_emb.t())
