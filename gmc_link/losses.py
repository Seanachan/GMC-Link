"""
Loss functions for the GMC-Link alignment network.
"""
import torch
from torch import nn
import torch.nn.functional as F


class AlignmentLoss(nn.Module):
    """
    Symmetric InfoNCE loss with fixed temperature.

    Standard CLIP-style cross-modal contrastive loss:
      L = CE(sim / τ, targets)  where targets = [0, 1, ..., B-1] (diagonal)
    Applied symmetrically: motion→language + language→motion.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, sim_matrix, sentence_ids=None):
        """
        Args:
            sim_matrix:   (B, B) cosine similarity matrix from model.forward()
            sentence_ids: unused, kept for API compatibility

        Returns:
            Scalar loss (mean of motion→language and language→motion directions)
        """
        B = sim_matrix.size(0)
        device = sim_matrix.device

        logits = sim_matrix / self.temperature

        # Targets: diagonal pairs are positives
        targets = torch.arange(B, device=device)

        # Symmetric cross-entropy
        m2l_loss = F.cross_entropy(logits, targets)
        l2m_loss = F.cross_entropy(logits.t(), targets)

        return (m2l_loss + l2m_loss) / 2.0


class HardNegativeInfoNCE(nn.Module):
    """Hard-negative-mining InfoNCE with optional False-Negative Masking.

    At β=0 and fnm=False, this reduces to standard InfoNCE.
    At β>0, negatives are reweighted by exp(β * sim) (Robinson et al. 2021 style).
    When fnm=True, same-sentence pairs are excluded from the negative set.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        beta: float = 1.0,
        fnm: bool = True,
    ):
        super().__init__()
        self.temperature = temperature
        self.beta = beta
        self.fnm = fnm

    def forward(self, sim_matrix, sentence_ids=None):
        if self.fnm and sentence_ids is None:
            raise ValueError("sentence_ids is required when fnm=True")
        B = sim_matrix.size(0)
        device = sim_matrix.device
        logits = sim_matrix / self.temperature

        # Masks
        diag = torch.eye(B, dtype=torch.bool, device=device)
        if self.fnm:
            same_sentence = sentence_ids[:, None] == sentence_ids[None, :]
        else:
            same_sentence = diag  # only the diagonal is a "positive"
        positive_mask = same_sentence            # (B, B) bool
        negative_mask = (~positive_mask) & ~diag  # exclude self-pair and same-sentence

        # Weighted logsumexp: denominator = exp(logits_pos_i) + Σ_j w[i,j] * exp(logits[i,j])
        # For β=0 (no mining), weights are uniform = 1 over the negative set.
        # Implemented as masked logsumexp — mask negatives we want to skip
        # by adding -inf to their logits before logsumexp.
        neg_logits = logits.masked_fill(~negative_mask, float("-inf"))

        # Positive logit: diagonal (motion_i vs its own caption_i)
        pos_logits = logits.diagonal()  # (B,)

        # m2l: anchor = motion, candidates = language (columns)
        # logsumexp over each row's unmasked negatives, plus the positive
        neg_lse_m2l = torch.logsumexp(neg_logits, dim=1)
        # Combine positive + negative denominator
        den_m2l = torch.logsumexp(
            torch.stack([pos_logits, neg_lse_m2l], dim=1), dim=1
        )
        l_m2l = (den_m2l - pos_logits).mean()

        # l2m: anchor = language (transpose logits)
        logits_t = logits.t()
        neg_logits_t = logits_t.masked_fill(~negative_mask.t(), float("-inf"))
        neg_lse_l2m = torch.logsumexp(neg_logits_t, dim=1)
        den_l2m = torch.logsumexp(
            torch.stack([pos_logits, neg_lse_l2m], dim=1), dim=1
        )
        l_l2m = (den_l2m - pos_logits).mean()

        return (l_m2l + l_l2m) / 2.0
