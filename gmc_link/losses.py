"""
Loss functions for the GMC-Link alignment network.
"""
import torch
from torch import nn
import torch.nn.functional as F


class AlignmentLoss(nn.Module):
    """
    Supervised InfoNCE (SupInfoNCE) with False-Negative Masking.

    Standard InfoNCE treats only the diagonal (i, i) as positive and all
    off-diagonal pairs as negative.  When a batch contains multiple samples
    sharing the same expression class, those same-class off-diagonal pairs
    become *false negatives* — the loss pushes apart embeddings that should
    be close, creating contradictory gradients and capping accuracy.

    FNM fixes this by masking same-class off-diagonal logits to −inf before
    the softmax, so they contribute as neither positive nor negative.

    Reference: Supervised Contrastive Learning (Khosla et al., 2020)
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, sim_matrix, sentence_ids=None):
        """
        Args:
            sim_matrix:   (B, B) cosine similarity matrix from model.forward()
            sentence_ids: (B,) integer class labels.  If provided, same-class
                          off-diagonal pairs are masked (FNM).  If None, falls
                          back to standard diagonal-only InfoNCE.

        Returns:
            Scalar loss (mean of motion→language and language→motion directions)
        """
        B = sim_matrix.size(0)
        device = sim_matrix.device

        logits = sim_matrix / self.temperature

        # Targets: diagonal pairs are positives
        targets = torch.arange(B, device=device)

        if sentence_ids is not None:
            # False-Negative Mask: same-class off-diagonal → −inf
            # (B, B) boolean: True where same class AND not on diagonal
            same_class = sentence_ids.unsqueeze(0) == sentence_ids.unsqueeze(1)
            diag_mask = torch.eye(B, dtype=torch.bool, device=device)
            false_neg_mask = same_class & ~diag_mask

            logits = logits.masked_fill(false_neg_mask, float("-inf"))

        # Symmetric cross-entropy
        m2l_loss = F.cross_entropy(logits, targets)
        l2m_loss = F.cross_entropy(logits.t(), targets)

        return (m2l_loss + l2m_loss) / 2.0
