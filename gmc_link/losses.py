"""
Loss functions for the GMC-Link alignment network.
"""
# pylint: disable=too-few-public-methods
from torch import nn

class AlignmentLoss(nn.Module):
    """
    Binary cross-entropy loss for motion-language alignment.

    For each (motion, language) pair, the model predicts a scalar similarity
    score. Positive pairs (correct match) should score high, negative pairs
    (wrong sentence) should score low.

    This replaces the CLIP-style contrastive loss which breaks down when
    many samples share the same sentence.
    """

    def __init__(self):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, scores, labels):
        """
        Args:
            scores: (N,) similarity scores from the aligner
            labels: (N,) binary labels (1.0 = positive match, 0.0 = negative)
        """
        return self.loss_fn(scores, labels)
