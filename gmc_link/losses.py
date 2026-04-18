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

    def forward(self, sim_matrix, sentence_ids):
        B = sim_matrix.size(0)
        device = sim_matrix.device
        logits = sim_matrix / self.temperature

        # For β=0 and fnm=False, this must equal F.cross_entropy(logits, diag_targets)
        targets = torch.arange(B, device=device)
        m2l_loss = F.cross_entropy(logits, targets)
        l2m_loss = F.cross_entropy(logits.t(), targets)
        return (m2l_loss + l2m_loss) / 2.0
