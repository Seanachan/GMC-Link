"""
Loss functions for the GMC-Link alignment network.
"""
import torch
from torch import nn
import torch.nn.functional as F


class AlignmentLoss(nn.Module):
    """
    SupInfoNCE (multi-positive + false-negative masking)
    symmetric: motion ↔ language
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, sim_matrix, sentence_ids):
        B = sim_matrix.size(0)

        # ---------------------------
        # normalize
        # ---------------------------
        logits = sim_matrix / self.temperature

        # ---------------------------
        # build masks
        # ---------------------------
        sentence_ids = sentence_ids.view(-1, 1)

        # positives: same sentence
        pos_mask = (sentence_ids == sentence_ids.T).float()

        # ❗ remove self-pair
        pos_mask.fill_diagonal_(0)

        # ❗ false-negative masking
        # only true negatives in denominator
        neg_mask = (sentence_ids != sentence_ids.T).float()

        # ---------------------------
        # motion → language
        # ---------------------------
        logits_max, _ = logits.max(dim=1, keepdim=True)
        logits_stable = logits - logits_max.detach()

        exp_logits = torch.exp(logits_stable) * neg_mask
        log_prob = logits_stable - torch.log(
            exp_logits.sum(dim=1, keepdim=True) + 1e-8
        )

        pos_count = pos_mask.sum(dim=1).clamp(min=1)

        loss_m2l = -(pos_mask * log_prob).sum(dim=1) / pos_count

        # ---------------------------
        # language → motion
        # ---------------------------
        logits_t = logits.t()

        logits_max_t, _ = logits_t.max(dim=1, keepdim=True)
        logits_t = logits_t - logits_max_t.detach()

        exp_logits_t = torch.exp(logits_t) * neg_mask.t()
        log_prob_t = logits_t - torch.log(
            exp_logits_t.sum(dim=1, keepdim=True) + 1e-8
        )

        pos_count_t = pos_mask.t().sum(dim=1).clamp(min=1)

        loss_l2m = -(pos_mask.t() * log_prob_t).sum(dim=1) / pos_count_t

        return (loss_m2l.mean() + loss_l2m.mean()) / 2