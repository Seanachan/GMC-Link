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


def test_beta_amplifies_hard_negative_gradient():
    """Higher β should produce a larger gradient norm at the hardest negative's
    input position, relative to easy negatives."""
    torch.manual_seed(42)

    # 4 samples with all-distinct sentences → negatives are 3 off-diagonal
    # entries per row. Build a similarity matrix with one clear hard negative
    # (row 0, column 2 — cosine 0.8, near the positive 0.9) and one easy
    # negative (row 0, column 3 — cosine -0.5).
    sim_base = torch.tensor([
        [0.9, 0.1, 0.8, -0.5],
        [0.1, 0.9, 0.0,  0.0],
        [0.8, 0.0, 0.9,  0.0],
        [-0.5, 0.0, 0.0, 0.9],
    ])
    sentence_ids = torch.tensor([0, 1, 2, 3])

    def grad_at(beta, col):
        sim = sim_base.clone().requires_grad_(True)
        loss = HardNegativeInfoNCE(temperature=0.07, beta=beta, fnm=False)(sim, sentence_ids)
        loss.backward()
        return sim.grad[0, col].abs().item()

    g_hard_small = grad_at(beta=0.5, col=2)
    g_easy_small = grad_at(beta=0.5, col=3)
    g_hard_large = grad_at(beta=2.0, col=2)
    g_easy_large = grad_at(beta=2.0, col=3)

    ratio_small = g_hard_small / (g_easy_small + 1e-9)
    ratio_large = g_hard_large / (g_easy_large + 1e-9)

    assert ratio_large > ratio_small, (
        f"β=2.0 should amplify hard-vs-easy gradient ratio more than β=0.5; "
        f"got ratio_small={ratio_small:.3f}, ratio_large={ratio_large:.3f}"
    )


def test_beta_amplifies_l2m_direction_with_asymmetric_sim():
    """On an asymmetric similarity matrix, β>0 should correctly route
    hard-negative weights through sim.t() in the l2m direction.

    Construct a matrix where the motion→language direction has its hard
    negative at a DIFFERENT column than the language→motion direction.
    Only correct per-direction weighting recovers both gradient asymmetries.
    """
    torch.manual_seed(0)

    # Asymmetric sim: diagonal = positives. Off-diagonal:
    # - Row 0 (motion 0 anchor): hard negative at col 2 (0.8)
    # - Col 0 (language 0 anchor, i.e. row 0 of sim.t()): hard negative at col 1
    #   (in sim.t(), row 0 == col 0 of sim) — engineered so sim[1,0] = 0.8.
    # Row 0 col 1 = 0.0 (easy), Row 2 col 0 = 0.1 (middling), Row 1 col 0 = 0.8 (hard for l2m anchor 0).
    sim_base = torch.tensor([
        [0.9, 0.0, 0.8, -0.5],   # motion 0 row: hard neg at col 2
        [0.8, 0.9, 0.0,  0.0],   # sim[1,0] = 0.8 → language 0 (in l2m) sees motion 1 as hard
        [0.1, 0.0, 0.9,  0.0],
        [-0.5, 0.0, 0.0, 0.9],
    ])
    sentence_ids = torch.tensor([0, 1, 2, 3])

    def l2m_grad_ratio(beta):
        sim = sim_base.clone().requires_grad_(True)
        loss = HardNegativeInfoNCE(temperature=0.07, beta=beta, fnm=False)(sim, sentence_ids)
        loss.backward()
        # For the l2m direction anchored at language 0 (column 0 of sim),
        # the hard negative is sim[1,0] = 0.8 and the easy one is sim[3,0] = -0.5.
        g_hard = sim.grad[1, 0].abs().item()
        g_easy = sim.grad[3, 0].abs().item()
        return g_hard / (g_easy + 1e-9)

    r_small = l2m_grad_ratio(beta=0.5)
    r_large = l2m_grad_ratio(beta=2.0)
    assert r_large > r_small, (
        f"β=2.0 should amplify l2m hard/easy ratio more than β=0.5; "
        f"got r_small={r_small:.3f}, r_large={r_large:.3f}"
    )


def test_fully_masked_row_is_finite():
    """When FNM masks an entire row's negatives (all same sentence), loss and
    gradients must stay finite rather than NaN."""
    sim = torch.randn(3, 3, requires_grad=True)
    sids = torch.tensor([7, 7, 7])  # all same sentence → every off-diagonal is a "false negative"
    loss = HardNegativeInfoNCE(temperature=0.07, beta=2.0, fnm=True)(sim, sids)
    assert torch.isfinite(loss).item(), f"loss not finite: {loss.item()}"
    loss.backward()
    assert torch.isfinite(sim.grad).all().item(), f"non-finite grad: {sim.grad}"
    # With all rows fully masked, there's no contrastive signal, so loss should be ~0.
    assert loss.item() < 1e-4, f"loss should be ~0 when no negatives; got {loss.item()}"
