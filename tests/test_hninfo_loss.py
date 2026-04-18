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


def test_fnm_excludes_same_sentence_pairs():
    """With FNM on, a same-sentence negative at large cosine should NOT
    penalize the loss. Without FNM, the same input should yield a larger loss
    because that off-diagonal entry acts as a hard (but false) negative.
    """
    # Batch of 3, items 0 and 2 share sentence_id=5 (duplicate), item 1 differs.
    sentence_ids = torch.tensor([5, 9, 5])

    # Construct a similarity matrix where the same-sentence off-diagonal
    # (positions [0,2] and [2,0]) has a very high cosine. The diagonal has a
    # moderate positive cosine, so a same-sentence "false negative" at [0,2]
    # should be masked out (ignored) when FNM is on.
    sim = torch.tensor([
        [0.3, 0.0, 0.95],
        [0.0, 0.3, 0.0 ],
        [0.95, 0.0, 0.3],
    ])

    loss_fnm_on  = HardNegativeInfoNCE(temperature=0.07, beta=0.0, fnm=True )(sim, sentence_ids)
    loss_fnm_off = HardNegativeInfoNCE(temperature=0.07, beta=0.0, fnm=False)(sim, sentence_ids)

    # FNM off: the 0.95 off-diagonal acts as a hard negative, driving loss UP.
    # FNM on: it's masked, loss goes DOWN.
    assert loss_fnm_on.item() < loss_fnm_off.item() - 0.1, (
        f"FNM should reduce loss by masking same-sentence pairs; "
        f"got fnm_on={loss_fnm_on.item():.4f}, fnm_off={loss_fnm_off.item():.4f}"
    )
