"""Unit + integration tests for HardNegativeInfoNCE."""
import torch
import pytest

from gmc_link.losses import AlignmentLoss, HardNegativeInfoNCE


def test_hninfo_beta_zero_equals_alignment_loss():
    """β=0, FNM off should exactly match the standard InfoNCE loss."""
    torch.manual_seed(0)
    sim = torch.randn(4, 4)
    # All distinct sentence IDs — FNM is a no-op regardless
    expr_ids = torch.tensor([10, 11, 12, 13])

    base = AlignmentLoss(temperature=0.07)(sim)
    hn = HardNegativeInfoNCE(temperature=0.07, beta=0.0, fnm=False)(sim, expr_ids)

    assert torch.allclose(base, hn, atol=1e-5), f"base={base.item()} hn={hn.item()}"
