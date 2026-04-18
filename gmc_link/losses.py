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

        diag = torch.eye(B, dtype=torch.bool, device=device)
        if self.fnm:
            same_sentence = sentence_ids[:, None] == sentence_ids[None, :]
        else:
            same_sentence = diag
        positive_mask = same_sentence
        negative_mask = (~positive_mask) & ~diag

        pos_logits = logits.diagonal()  # (B,)

        # ── β-weighted denominator for motion→language direction ──
        # w_raw[i,j] = exp(β * sim[i,j]) on negatives, 0 elsewhere.
        # Then normalize: w[i,:] = w_raw[i,:] / w_raw[i,:].sum() * N_neg[i]
        # so Σⱼ w[i,j] = N_neg[i] (preserves β=0 → uniform weights = 1).
        def weighted_neg_lse(lg, nm, sim_for_weights):
            log_w_raw = (self.beta * sim_for_weights).masked_fill(~nm, float("-inf"))
            log_w_norm = log_w_raw - torch.logsumexp(log_w_raw, dim=1, keepdim=True)
            n_neg = nm.sum(dim=1, keepdim=True).clamp_min(1).to(lg.dtype)
            log_w = log_w_norm + torch.log(n_neg)  # rescale so Σw = N_neg
            # Fully-masked rows produce NaN from (-inf) - (-inf); clear them so
            # they contribute -inf to the logsumexp below (i.e., no negatives).
            log_w = log_w.masked_fill(~nm, float("-inf"))

            # Weighted logsumexp: logsumexp_j( log(w[i,j]) + logits[i,j] ) over negatives
            masked_logits = lg.masked_fill(~nm, float("-inf"))
            return torch.logsumexp(log_w + masked_logits, dim=1)

        neg_lse_m2l = weighted_neg_lse(logits,     negative_mask,     sim_matrix)
        neg_lse_l2m = weighted_neg_lse(logits.t(), negative_mask.t(), sim_matrix.t())

        den_m2l = torch.logsumexp(torch.stack([pos_logits, neg_lse_m2l], dim=1), dim=1)
        den_l2m = torch.logsumexp(torch.stack([pos_logits, neg_lse_l2m], dim=1), dim=1)

        l_m2l = (den_m2l - pos_logits).mean()
        l_l2m = (den_l2m - pos_logits).mean()
        return (l_m2l + l_l2m) / 2.0
