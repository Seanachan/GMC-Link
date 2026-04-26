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


def test_weight_normalization_invariant():
    """For any β, the per-anchor negative weights must sum to N_neg[i]."""
    torch.manual_seed(1)
    B = 8
    sim = torch.randn(B, B)
    # 3 groups of duplicate sentences → N_neg varies per anchor
    sentence_ids = torch.tensor([0, 0, 1, 1, 1, 2, 3, 4])

    for beta in [0.0, 0.5, 1.0, 2.0]:
        loss_fn = HardNegativeInfoNCE(temperature=0.07, beta=beta, fnm=True)
        w, n_neg = loss_fn.compute_negative_weights(sim, sentence_ids)
        # w shape: (B, B); n_neg shape: (B,)
        assert torch.allclose(w.sum(dim=1), n_neg.to(w.dtype), atol=1e-4), (
            f"β={beta}: weights don't sum to N_neg; "
            f"got sums={w.sum(dim=1)}, expected={n_neg}"
        )


def test_sentence_ids_passed_through_training_contract():
    """Verify AlignmentLoss and HardNegativeInfoNCE both accept (sim, ids)
    so train_one_epoch's call site `loss_func(sim_matrix, expr_ids)` works
    for either loss with no branching at the call site.
    """
    import inspect
    from gmc_link.losses import AlignmentLoss, HardNegativeInfoNCE

    align_params = inspect.signature(AlignmentLoss.forward).parameters
    hn_params    = inspect.signature(HardNegativeInfoNCE.forward).parameters

    # Both must accept at least (self, sim_matrix, <ids-like arg>)
    assert len(align_params) >= 3, "AlignmentLoss.forward must accept sentence_ids"
    assert len(hn_params) >= 3, "HardNegativeInfoNCE.forward must accept sentence_ids"

    # Smoke: can call each without TypeError
    sim = torch.randn(4, 4)
    ids = torch.tensor([0, 1, 2, 3])
    _ = AlignmentLoss(temperature=0.07)(sim, ids)
    _ = HardNegativeInfoNCE(temperature=0.07, beta=0.0, fnm=False)(sim, ids)


def test_end_to_end_forward_backward_one_batch():
    """Simulate one training iteration end-to-end: build the real model,
    run HN-InfoNCE forward, backward, optimizer step, and assert no NaN."""
    from gmc_link.alignment import MotionLanguageAligner

    torch.manual_seed(0)
    device = torch.device("cpu")
    model = MotionLanguageAligner(motion_dim=13, lang_dim=384, embed_dim=256).to(device)
    loss_fn = HardNegativeInfoNCE(temperature=0.07, beta=1.0, fnm=True)
    optim_obj = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Realistic batch: 8 samples, some duplicate sentences (FNM exercise)
    motion = torch.randn(8, 13)
    lang   = torch.randn(8, 384)
    sentence_ids = torch.tensor([10, 11, 10, 12, 13, 11, 14, 15])

    model.train()
    sim = model(motion, lang)
    loss = loss_fn(sim, sentence_ids)
    assert torch.isfinite(loss).item(), f"loss is not finite: {loss}"

    optim_obj.zero_grad()
    loss.backward()
    # Every trainable param should receive a finite gradient
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"param {name} has no grad"
            assert torch.isfinite(p.grad).all(), f"param {name} has non-finite grad"
    optim_obj.step()

    # Second forward should still be finite (catches NaN from bad backward)
    sim2 = model(motion, lang)
    loss2 = loss_fn(sim2, sentence_ids)
    assert torch.isfinite(loss2).item(), f"loss after step is not finite: {loss2}"
