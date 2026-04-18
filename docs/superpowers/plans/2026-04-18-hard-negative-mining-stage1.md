# Hard-Negative Mining Finetune on Stage 1 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement HN-InfoNCE loss with False-Negative Masking, wire it into the existing training CLI, and build a shell driver that finetunes the V1 stage1 weights at three β values and drops the results into the existing multi-seq eval pipeline.

**Architecture:** Additive change to `gmc_link/losses.py` (new `HardNegativeInfoNCE` class), a two-flag extension to `gmc_link/train.py` (`--loss`, `--beta`), and a single shell driver that loops the 3 β values and invokes the existing `diagnostics/diag_gt_cosine_distributions.py` + `diagnostics/aggregate_multiseq.py` pipeline with a 14-weight argument list (11 Exp 33 baselines + 3 new HN finetunes).

**Tech Stack:** PyTorch (loss + training), bash (driver), pytest (tests), existing `aggregate_multiseq.py` and `diag_gt_cosine_distributions.py` (evaluation).

**Spec:** `docs/superpowers/specs/2026-04-18-hard-negative-mining-stage1-design.md`

---

## File Structure

**Modified:**
- `gmc_link/losses.py` — adds `HardNegativeInfoNCE` class; `AlignmentLoss` stays untouched.
- `gmc_link/train.py` — adds `--loss` and `--beta` CLI args; threads them into loss construction.

**Created:**
- `tests/test_hninfo_loss.py` — 4 unit tests + 1 integration smoke test.
- `run_hn_finetune.sh` — driver: 3-β sweep + multi-seq eval.

**Generated (gitignored):**
- `gmc_link_weights_v1train_hninfo_beta{0.5,1.0,2.0}.pth`
- `gmc_link_weights_v1train_hninfo_beta{β}_curves.png`
- `diagnostics/results/multiseq/layer3_{0005,0011,0013}_v1train_hninfo_beta{β}.npz`
- `diagnostics/results/multiseq/layer3_multiseq_v1train_hninfo_beta{β}.{json,md}` (regenerated)
- `diagnostics/results/multiseq/layer3_multiseq_comparison.md` (regenerated with 14 rows)

---

## Task 1: Scaffold `HardNegativeInfoNCE` (β=0, no FNM) ≡ `AlignmentLoss`

**Rationale:** Start with the weakest version that still has the right interface. At β=0 and FNM off, the class must produce identical loss to `AlignmentLoss` so we know the skeleton math is correct before adding complexity.

**Files:**
- Modify: `gmc_link/losses.py` (append new class at end)
- Create: `tests/test_hninfo_loss.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_hninfo_loss.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_hninfo_loss.py::test_hninfo_beta_zero_equals_alignment_loss -v`
Expected: FAIL with `ImportError` on `HardNegativeInfoNCE`.

- [ ] **Step 3: Implement the minimal class**

Append to `gmc_link/losses.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_hninfo_loss.py::test_hninfo_beta_zero_equals_alignment_loss -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add gmc_link/losses.py tests/test_hninfo_loss.py
git commit -m "feat(losses): scaffold HardNegativeInfoNCE with β=0 equivalence"
```

---

## Task 2: Add False-Negative Masking

**Rationale:** Before adding hard-negative weighting, make FNM work. Mining will amplify same-sentence "false negatives" so the mask must be in place first.

**Files:**
- Modify: `gmc_link/losses.py:HardNegativeInfoNCE`
- Modify: `tests/test_hninfo_loss.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_hninfo_loss.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_hninfo_loss.py::test_fnm_excludes_same_sentence_pairs -v`
Expected: FAIL — current implementation ignores `sentence_ids`, so `loss_fnm_on == loss_fnm_off`.

- [ ] **Step 3: Implement FNM by replacing the forward method**

Replace the body of `HardNegativeInfoNCE.forward` in `gmc_link/losses.py` with:

```python
    def forward(self, sim_matrix, sentence_ids):
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
```

- [ ] **Step 4: Run both tests to verify they pass**

Run: `pytest tests/test_hninfo_loss.py -v`
Expected: Both tests PASS. (Task 1 test still passes because with all-distinct `sentence_ids` and `fnm=False`, the new code path is equivalent to `F.cross_entropy(logits, diag_targets)`.)

- [ ] **Step 5: Commit**

```bash
git add gmc_link/losses.py tests/test_hninfo_loss.py
git commit -m "feat(losses): add False-Negative Masking to HardNegativeInfoNCE"
```

---

## Task 3: Add β-Weighted Hard-Negative Mining

**Rationale:** Now add the Robinson-style hardness reweighting. With β>0, harder negatives (higher similarity) get larger weights in the denominator, concentrating loss gradients on them.

**Files:**
- Modify: `gmc_link/losses.py:HardNegativeInfoNCE.forward`
- Modify: `tests/test_hninfo_loss.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_hninfo_loss.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_hninfo_loss.py::test_beta_amplifies_hard_negative_gradient -v`
Expected: FAIL — β is currently ignored, so ratios match exactly.

- [ ] **Step 3: Add β-weighting to the forward method**

Replace the body of `HardNegativeInfoNCE.forward` in `gmc_link/losses.py` with the β-weighted version:

```python
    def forward(self, sim_matrix, sentence_ids):
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
        def weighted_neg_lse(lg, nm):
            neg_sim = sim_matrix.masked_fill(~nm, 0.0)
            # exp(β * sim) with masked entries set to 0 via additive -inf on log side
            log_w_raw = (self.beta * neg_sim).masked_fill(~nm, float("-inf"))
            log_w_norm = log_w_raw - torch.logsumexp(log_w_raw, dim=1, keepdim=True)
            n_neg = nm.sum(dim=1, keepdim=True).clamp_min(1).to(lg.dtype)
            log_w = log_w_norm + torch.log(n_neg)  # rescale so Σw = N_neg

            # Weighted logsumexp: logsumexp_j( log(w[i,j]) + logits[i,j] ) over negatives
            masked_logits = lg.masked_fill(~nm, float("-inf"))
            return torch.logsumexp(log_w + masked_logits, dim=1)

        neg_lse_m2l = weighted_neg_lse(logits,   negative_mask)
        neg_lse_l2m = weighted_neg_lse(logits.t(), negative_mask.t())

        den_m2l = torch.logsumexp(torch.stack([pos_logits, neg_lse_m2l], dim=1), dim=1)
        den_l2m = torch.logsumexp(torch.stack([pos_logits, neg_lse_l2m], dim=1), dim=1)

        l_m2l = (den_m2l - pos_logits).mean()
        l_l2m = (den_l2m - pos_logits).mean()
        return (l_m2l + l_l2m) / 2.0
```

- [ ] **Step 4: Run all three tests**

Run: `pytest tests/test_hninfo_loss.py -v`
Expected: All three tests PASS.
- `test_hninfo_beta_zero_equals_alignment_loss` — β=0 still reduces to standard InfoNCE because at β=0 all exp(β·sim) = 1, so `log_w_norm = -log(N_neg)`, plus `log(N_neg)` cancels → `log_w = 0`, which means `w = 1` per negative. This matches the plain masked logsumexp from Task 2.
- `test_fnm_excludes_same_sentence_pairs` — still passes (β=0 in that test).
- `test_beta_amplifies_hard_negative_gradient` — now passes because β>0 upweights the col=2 logit.

- [ ] **Step 5: Commit**

```bash
git add gmc_link/losses.py tests/test_hninfo_loss.py
git commit -m "feat(losses): add β-weighted hard-negative mining to HardNegativeInfoNCE"
```

---

## Task 4: Verify Weight-Normalization Invariant

**Rationale:** Guardrail. The spec requires `Σⱼ w[i,j] = N_neg[i]` so that β=0 genuinely recovers uniform weights = 1. Add a test that inspects this directly (rather than relying on the β=0 equivalence test) to catch regressions if the normalization is ever rewritten.

**Files:**
- Modify: `gmc_link/losses.py:HardNegativeInfoNCE` (expose helper)
- Modify: `tests/test_hninfo_loss.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_hninfo_loss.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_hninfo_loss.py::test_weight_normalization_invariant -v`
Expected: FAIL with `AttributeError: 'HardNegativeInfoNCE' object has no attribute 'compute_negative_weights'`.

- [ ] **Step 3: Expose the helper by extracting the mask+weight computation**

Add a method to `HardNegativeInfoNCE` in `gmc_link/losses.py` (directly above `forward`):

```python
    def compute_negative_weights(self, sim_matrix, sentence_ids):
        """Return (w, n_neg) for inspection.

        w: (B, B) tensor of normalized negative weights (positives are 0).
        n_neg: (B,) tensor of negative counts per anchor.

        Invariant: w.sum(dim=1) == n_neg for all rows with n_neg > 0.
        """
        B = sim_matrix.size(0)
        device = sim_matrix.device
        diag = torch.eye(B, dtype=torch.bool, device=device)
        if self.fnm:
            same_sentence = sentence_ids[:, None] == sentence_ids[None, :]
        else:
            same_sentence = diag
        negative_mask = (~same_sentence) & ~diag

        log_w_raw = (self.beta * sim_matrix).masked_fill(~negative_mask, float("-inf"))
        log_w_norm = log_w_raw - torch.logsumexp(log_w_raw, dim=1, keepdim=True)
        n_neg = negative_mask.sum(dim=1).to(sim_matrix.dtype)
        # log_w = log_w_norm + log(N_neg). Convert to linear scale, zero out masked.
        log_w = log_w_norm + torch.log(n_neg.clamp_min(1)).unsqueeze(1)
        w = log_w.exp().masked_fill(~negative_mask, 0.0)
        return w, n_neg.long()
```

- [ ] **Step 4: Run the test**

Run: `pytest tests/test_hninfo_loss.py::test_weight_normalization_invariant -v`
Expected: PASS.

- [ ] **Step 5: Run the whole suite to confirm nothing regressed**

Run: `pytest tests/test_hninfo_loss.py -v`
Expected: 4 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add gmc_link/losses.py tests/test_hninfo_loss.py
git commit -m "test(losses): expose compute_negative_weights for invariant check"
```

---

## Task 5: Wire `--loss` and `--beta` CLI Flags in `train.py`

**Rationale:** Let the training entrypoint select between `AlignmentLoss` (default, existing behavior) and `HardNegativeInfoNCE` (new). Default behavior must not change.

**Files:**
- Modify: `gmc_link/train.py` (argparse + `setup_model_and_optimizer` + `train_one_epoch` sentence_ids plumbing)

- [ ] **Step 1: Add argparse flags**

In `gmc_link/train.py`, inside `main()` after the existing `--grad-clip` line (around line 371), add:

```python
    parser.add_argument("--loss", default="infonce", choices=["infonce", "hninfo"],
                        help="Contrastive loss (default: infonce; hninfo = hard-negative InfoNCE with FNM)")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="Hard-negative concentration (only used when --loss hninfo; 0=uniform, 1.0 typical)")
```

- [ ] **Step 2: Update `setup_model_and_optimizer` to accept the loss choice**

Replace the signature and body of `setup_model_and_optimizer` in `gmc_link/train.py` (around line 150):

```python
def setup_model_and_optimizer(
    device: torch.device, lang_dim: int, learning_rate: float, epochs: int,
    learnable_temp: bool = False, motion_dim: int = 13,
    architecture: str = "mlp", seq_len: int = 10,
    loss_name: str = "infonce", beta: float = 1.0,
) -> Tuple[
    MotionLanguageAligner, nn.Module, optim.Optimizer, optim.lr_scheduler.LRScheduler
]:
    """Initialize model, loss, and AdamW optimizer."""
    model = MotionLanguageAligner(
        motion_dim=motion_dim, lang_dim=lang_dim, embed_dim=256,
        architecture=architecture, seq_len=seq_len,
    ).to(device)

    if loss_name == "hninfo":
        from gmc_link.losses import HardNegativeInfoNCE
        if learnable_temp:
            raise ValueError("--learnable-temp is not supported with --loss hninfo")
        criterion = HardNegativeInfoNCE(temperature=0.07, beta=beta, fnm=True)
    else:
        criterion = AlignmentLoss(temperature=0.07, learnable=learnable_temp)

    params = list(model.parameters())
    if learnable_temp:
        params += list(criterion.parameters())

    optimizer = optim.AdamW(params, lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )
    return model, criterion, optimizer, scheduler
```

- [ ] **Step 3: Thread loss choice through `_run_single_stage`**

Replace `_run_single_stage`'s signature (around line 294) to accept `loss_name` and `beta`, and pass them to `setup_model_and_optimizer`:

```python
def _run_single_stage(
    device: torch.device,
    data_root: str,
    sequences: list,
    batch_size: int,
    lang_dim: int,
    lr: float,
    epochs: int,
    save_path: str,
    use_group_labels: bool = False,
    resume_path: str = None,
    warmup_epochs: int = 0,
    learnable_temp: bool = False,
    grad_clip: float = 0.0,
    extra_features: list = None,
    architecture: str = "mlp",
    seq_len: int = 10,
    loss_name: str = "infonce",
    beta: float = 1.0,
) -> None:
    """Run a single training stage."""
    if loss_name == "hninfo" and use_group_labels:
        raise ValueError(
            "--loss hninfo requires sentence-level labels; "
            "not compatible with --stage 1 (group labels)"
        )

    dataloader = setup_data(device, data_root, sequences, batch_size,
                            use_group_labels=use_group_labels,
                            extra_features=extra_features,
                            seq_len=seq_len if architecture == "temporal_transformer" else 0)
    if dataloader is None:
        print("ERROR: No training data found.")
        return

    motion_dim = 13 + compute_extra_dims(extra_features)
    model, criterion, optimizer, scheduler = setup_model_and_optimizer(
        device, lang_dim, lr, epochs, learnable_temp=learnable_temp,
        motion_dim=motion_dim,
        architecture=architecture, seq_len=seq_len,
        loss_name=loss_name, beta=beta,
    )

    if resume_path is not None:
        print(f"  Loading weights from {resume_path}...")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
        else:
            model.load_state_dict(checkpoint)

    train_loop(model, dataloader, optimizer, scheduler, criterion, device, epochs,
               save_path=save_path, warmup_epochs=warmup_epochs, grad_clip=grad_clip)

    checkpoint = torch.load(save_path, map_location=device, weights_only=False)
    checkpoint["motion_dim"] = motion_dim
    checkpoint["extra_features"] = extra_features
    checkpoint["architecture"] = architecture
    checkpoint["seq_len"] = seq_len if architecture == "temporal_transformer" else None
    checkpoint["loss_name"] = loss_name
    checkpoint["beta"] = beta if loss_name == "hninfo" else None
    torch.save(checkpoint, save_path)
```

- [ ] **Step 4: Thread loss choice from `main()` into the single-stage call**

Replace the final `_run_single_stage(...)` call in `main()` (around line 460) with:

```python
    _run_single_stage(
        device, data_root, sequences, batch_size, lang_dim,
        lr=args.lr, epochs=args.epochs, save_path=save_path,
        use_group_labels=use_group, resume_path=args.resume,
        warmup_epochs=args.warmup_epochs, learnable_temp=args.learnable_temp,
        grad_clip=args.grad_clip, extra_features=extra_features,
        architecture=args.architecture, seq_len=args.seq_len,
        loss_name=args.loss, beta=args.beta,
    )
```

(The curriculum branch keeps `loss_name="infonce"` implicitly via defaults — HN-InfoNCE does not apply to curriculum mode in this experiment.)

- [ ] **Step 5: Smoke-test the `--help` output**

Run: `python -m gmc_link.train --help 2>&1 | grep -E "(--loss|--beta)"`
Expected: two lines showing `--loss {infonce,hninfo}` and `--beta BETA`.

- [ ] **Step 6: Verify default path unchanged**

Run: `python -c "from gmc_link.train import setup_model_and_optimizer; import torch; m, c, o, s = setup_model_and_optimizer(torch.device('cpu'), 384, 1e-3, 10); print(type(c).__name__)"`
Expected: `AlignmentLoss`

- [ ] **Step 7: Commit**

```bash
git add gmc_link/train.py
git commit -m "feat(train): add --loss and --beta CLI flags for HN-InfoNCE"
```

---

## Task 6: Update `train_one_epoch` to Pass `sentence_ids` to HN-InfoNCE

**Rationale:** `HardNegativeInfoNCE.forward(sim_matrix, sentence_ids)` requires a real sentence-ID tensor. The existing `train_one_epoch` already collates `expr_ids` from the dataset; when `use_group_labels=False` (required for hninfo per Task 5's guard), these `expr_ids` ARE sentence-level IDs from `sentence_to_id`. So no dataset change needed — just confirm the tensor is passed through.

Inspect `train_one_epoch` (around line 58): it already does `loss = loss_func(sim_matrix, expr_ids)`. ✓ Contract is satisfied. This task is a verification-only task.

**Files:**
- Modify: `tests/test_hninfo_loss.py` (add plumbing check)

- [ ] **Step 1: Write a lightweight plumbing test**

Append to `tests/test_hninfo_loss.py`:

```python
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
```

- [ ] **Step 2: Run the test**

Run: `pytest tests/test_hninfo_loss.py::test_sentence_ids_passed_through_training_contract -v`
Expected: PASS. Both losses already accept `(sim_matrix, sentence_ids)`.

- [ ] **Step 3: Commit**

```bash
git add tests/test_hninfo_loss.py
git commit -m "test(losses): lock in sentence_ids training-contract compatibility"
```

---

## Task 7: Integration Smoke Test — Single Mini-Epoch

**Rationale:** Make sure the full pipeline (dataset → model → HN-InfoNCE → optimizer step) runs without NaN/shape errors on a tiny synthetic batch. This catches any issue the unit tests can't — e.g., device placement, optimizer-parameter interaction, or checkpoint serialization.

**Files:**
- Modify: `tests/test_hninfo_loss.py`

- [ ] **Step 1: Write the failing integration test**

Append to `tests/test_hninfo_loss.py`:

```python
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
```

- [ ] **Step 2: Run the test**

Run: `pytest tests/test_hninfo_loss.py::test_end_to_end_forward_backward_one_batch -v`
Expected: PASS.

- [ ] **Step 3: Run the full test suite as a sanity check**

Run: `pytest tests/test_hninfo_loss.py -v`
Expected: 6 tests PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/test_hninfo_loss.py
git commit -m "test(losses): end-to-end HN-InfoNCE integration smoke test"
```

---

## Task 8: Build the `run_hn_finetune.sh` Driver

**Rationale:** Automate the 3-β finetune sweep plus the multi-seq eval in a single idempotent script. Mirrors the structure of `run_multiseq_eval.sh` (pre-flight, seq-outer/weight-inner eval loop, aggregator invocation).

**Files:**
- Create: `run_hn_finetune.sh`

- [ ] **Step 1: Write the driver**

Create `run_hn_finetune.sh`:

```bash
#!/usr/bin/env bash
# Hard-Negative Mining Finetune driver for Exp 34.
#
# For each β in {0.5, 1.0, 2.0}:
#   1. Finetune v1train_stage1.pth with HN-InfoNCE for 30 epochs at LR=1e-4.
#   2. Run the Layer 3 diagnostic on each held-out seq (0005, 0011, 0013).
#   3. Relocate the .npz into diagnostics/results/multiseq/ with the HN tag.
#
# Finally, invoke aggregate_multiseq.py with a 14-weight list (11 Exp 33
# baselines + 3 HN finetunes) to regenerate layer3_multiseq_comparison.md.
#
# Spec: docs/superpowers/specs/2026-04-18-hard-negative-mining-stage1-design.md
# Plan: docs/superpowers/plans/2026-04-18-hard-negative-mining-stage1.md
#
# Usage: bash run_hn_finetune.sh

set -euo pipefail

PY=~/miniconda/envs/RMOT/bin/python
DIAG=diagnostics/diag_gt_cosine_distributions.py
RESULTS_DIR=diagnostics/results
MULTISEQ_DIR="${RESULTS_DIR}/multiseq"
mkdir -p "${MULTISEQ_DIR}"

BETAS=(0.5 1.0 2.0)
SEQS=(0005 0011 0013)
STAGE1_WEIGHTS=gmc_link_weights_v1train_stage1.pth

# Pre-flight
if [[ ! -f "${STAGE1_WEIGHTS}" ]]; then
  echo "ERROR: stage1 weights not found: ${STAGE1_WEIGHTS}" >&2
  exit 1
fi

# ── Training loop: one finetune per β ──────────────────────────────
for beta in "${BETAS[@]}"; do
  tag="v1train_hninfo_beta${beta}"
  save_path="gmc_link_weights_${tag}.pth"
  echo "============================================================"
  echo "Finetune β=${beta} → ${save_path}"
  echo "============================================================"
  "${PY}" -m gmc_link.train \
    --split v1 \
    --loss hninfo --beta "${beta}" \
    --resume "${STAGE1_WEIGHTS}" \
    --epochs 30 --lr 1e-4 --batch-size 256 \
    --save-path "${save_path}"
done

# ── Evaluation loop: diagnostic per (seq, β) ───────────────────────
for seq in "${SEQS[@]}"; do
  echo "============================================================"
  echo "Diagnostic | seq ${seq}"
  echo "============================================================"
  for beta in "${BETAS[@]}"; do
    tag="v1train_hninfo_beta${beta}"
    weights="gmc_link_weights_${tag}.pth"
    echo "--- ${seq} / ${tag} ---"
    "${PY}" "${DIAG}" --weights "${weights}" --seq "${seq}"
    src="${RESULTS_DIR}/layer3_gt_cosine_${seq}.npz"
    dst="${MULTISEQ_DIR}/layer3_${seq}_${tag}.npz"
    if [[ ! -f "${src}" ]]; then
      echo "ERROR: expected ${src} but diagnostic did not produce it" >&2
      exit 2
    fi
    mv "${src}" "${dst}"
    src_png="${RESULTS_DIR}/layer3_gt_cosine_${seq}.png"
    if [[ -f "${src_png}" ]]; then
      mv "${src_png}" "${MULTISEQ_DIR}/layer3_${seq}_${tag}.png"
    fi
  done
done

# ── Aggregate: 11 baseline weights + 3 HN weights = 14 total ───────
ALL_WEIGHTS=(
  "v1train_stage1=gmc_link_weights_v1train_stage1.pth"
  "v1train_F1_speed=gmc_link_weights_v1train_F1_speed.pth"
  "v1train_F2_heading=gmc_link_weights_v1train_F2_heading.pth"
  "v1train_F3_accel=gmc_link_weights_v1train_F3_accel.pth"
  "v1train_F4_ego=gmc_link_weights_v1train_F4_ego.pth"
  "v1train_F5_nbrmean=gmc_link_weights_v1train_F5_nbrmean.pth"
  "v1train_F6_velrank=gmc_link_weights_v1train_F6_velrank.pth"
  "v1train_F7_headdiff=gmc_link_weights_v1train_F7_headdiff.pth"
  "v1train_F8_nndist=gmc_link_weights_v1train_F8_nndist.pth"
  "v1train_F9_density=gmc_link_weights_v1train_F9_density.pth"
  "v1train_temporal=gmc_link_weights_v1train_temporal.pth"
)
for beta in "${BETAS[@]}"; do
  tag="v1train_hninfo_beta${beta}"
  ALL_WEIGHTS+=("${tag}=gmc_link_weights_${tag}.pth")
done

LEGACY=(
  "v1train_stage1=0.779"
  "v1train_temporal=0.770"
)

echo "============================================================"
echo "Aggregating 14 weights × 3 seqs → multiseq reports"
echo "============================================================"
"${PY}" diagnostics/aggregate_multiseq.py \
  --results-dir "${MULTISEQ_DIR}" \
  --output-dir "${MULTISEQ_DIR}" \
  --weights "${ALL_WEIGHTS[@]}" \
  --seqs "${SEQS[@]}" \
  --legacy-seq-0011 "${LEGACY[@]}"

echo "Done. See ${MULTISEQ_DIR}/layer3_multiseq_comparison.md"
```

- [ ] **Step 2: Make executable**

Run: `chmod +x run_hn_finetune.sh`

- [ ] **Step 3: Dry-run inspection (no execution)**

Run: `bash -n run_hn_finetune.sh && echo "syntax OK"`
Expected: `syntax OK` (no syntax errors; does not execute).

- [ ] **Step 4: Commit**

```bash
git add run_hn_finetune.sh
git commit -m "feat: HN-InfoNCE finetune + multi-seq eval driver script"
```

---

## Task 9: Add HN-InfoNCE Weight Outputs to `.gitignore`

**Rationale:** Weight files are gitignored by pattern (`*.pth`), but curve PNGs and generated `.npz` files also need to stay out of git. The existing `.gitignore` covers most of this, but the new filenames should be cross-checked.

**Files:**
- Modify: `.gitignore` (verify, add if missing)

- [ ] **Step 1: Check what's already covered**

Run: `git check-ignore -v gmc_link_weights_v1train_hninfo_beta1.0.pth gmc_link_weights_v1train_hninfo_beta1.0_curves.png diagnostics/results/multiseq/layer3_0011_v1train_hninfo_beta1.0.npz`

Expected: each line printed = that path IS ignored. If any of the three prints nothing, add the corresponding pattern in Step 2; otherwise skip Steps 2–4.

- [ ] **Step 2: If any pattern is missing, append to `.gitignore`**

Only if Step 1 showed a gap. For `.pth` files the existing `*.pth` should cover it. For curve PNGs, the existing `*_curves.png` should cover it. For multi-seq `.npz` under `diagnostics/results/multiseq/`, the existing `diagnostics/results/` wildcard should cover it.

If a gap exists, append the most specific pattern that closes it (e.g. `diagnostics/results/multiseq/*.npz`).

- [ ] **Step 3: Verify**

Run: `git check-ignore -v gmc_link_weights_v1train_hninfo_beta1.0.pth gmc_link_weights_v1train_hninfo_beta1.0_curves.png diagnostics/results/multiseq/layer3_0011_v1train_hninfo_beta1.0.npz`
Expected: all three now ignored.

- [ ] **Step 4: Commit (only if a change was made)**

```bash
git add .gitignore
git commit -m "chore(gitignore): ensure HN-InfoNCE generated artifacts are ignored"
```

---

## Task 10: Final End-to-End Dry-Run (No Training)

**Rationale:** Before the user kicks off the ~50-minute finetune, verify that a 2-epoch run of HN-InfoNCE on a tiny batch succeeds end-to-end. This catches the class of bug that unit tests can't — interaction with the actual dataset builder, actual optimizer, and actual checkpoint format.

**Files:** none (runs existing code)

- [ ] **Step 1: Run a 2-epoch sanity finetune on a single sequence**

Run:
```bash
~/miniconda/envs/RMOT/bin/python -m gmc_link.train \
  --split v1 \
  --loss hninfo --beta 1.0 \
  --resume gmc_link_weights_v1train_stage1.pth \
  --epochs 2 --lr 1e-4 --batch-size 64 \
  --save-path /tmp/hninfo_sanity.pth
```

Expected:
- Training completes 2 epochs without error.
- Loss is finite and > 0 at both epochs.
- `/tmp/hninfo_sanity.pth` exists on disk.

- [ ] **Step 2: Inspect the saved checkpoint**

Run:
```bash
~/miniconda/envs/RMOT/bin/python -c "
import torch
ckpt = torch.load('/tmp/hninfo_sanity.pth', map_location='cpu', weights_only=False)
print('keys:', sorted(ckpt.keys()))
print('loss_name:', ckpt.get('loss_name'))
print('beta:', ckpt.get('beta'))
print('architecture:', ckpt.get('architecture'))
"
```

Expected output must include:
- `loss_name: hninfo`
- `beta: 1.0`
- `architecture: mlp`
- keys include `model` and `temperature`

- [ ] **Step 3: Clean up sanity artifact**

Run: `rm -f /tmp/hninfo_sanity.pth /tmp/hninfo_sanity_curves.png`

- [ ] **Step 4: No commit needed**

This task is verification only — nothing to commit.

---

## Task 11: Kick Off the Real Experiment

**Rationale:** With the pipeline verified, run the full 3-β sweep + evaluation.

**Files:** none

- [ ] **Step 1: Run the finetune driver**

Run: `bash run_hn_finetune.sh 2>&1 | tee diagnostics/results/multiseq/_hninfo_run_log.txt`

Expected wall clock: ~50 minutes on the V1 training machine. Log file captures per-epoch losses and eval AUCs.

- [ ] **Step 2: Inspect the updated comparison table**

Run: `cat diagnostics/results/multiseq/layer3_multiseq_comparison.md | head -30`

Expected: 14-row Markdown table sorted by `mean_auc_micro` descending. Three new rows present: `v1train_hninfo_beta0.5`, `v1train_hninfo_beta1.0`, `v1train_hninfo_beta2.0`.

- [ ] **Step 3: Apply the decision rule**

Read the best HN-β micro AUC from the comparison table:
- **≥ 0.795** → H₁ confirmed (loss-bound ceiling). Next step: write Exp 34 retrospective in `RESEARCH_NOTES.md`, recommend full-budget (γ) run.
- **0.779 ± 0.015** → Inconclusive. Record findings; do not pursue further hard-negative work on the 13D representation.
- **< 0.764** → Mining actively harms. Record findings; pivot to representation-side experiments.

- [ ] **Step 4: Write the retrospective**

Append a new section to `RESEARCH_NOTES.md`:

```
## Exp 34: Hard-Negative Mining Finetune on V1 Stage 1 (2026-04-18)

**Goal:** Test H₀ (ceiling is representation-bound) vs H₁ (ceiling is loss-bound) by
finetuning gmc_link_weights_v1train_stage1.pth with HN-InfoNCE + FNM at β ∈ {0.5, 1.0, 2.0}.

**Method:** 30 epochs, LR=1e-4, cosine decay, batch=256, sentence-level FNM.

**Results** (micro AUC on held-out seqs 0005/0011/0013):
| β   | mean_auc_micro | mean_auc_macro ± std | worst_seq       |
|-----|----------------|----------------------|-----------------|
| 0.5 | <fill>         | <fill>               | <fill>          |
| 1.0 | <fill>         | <fill>               | <fill>          |
| 2.0 | <fill>         | <fill>               | <fill>          |

**Verdict:** <loss-bound / inconclusive / representation-bound per §1 decision rule>.

**Next:** <full-budget run / alternate loss families / pivot to representation>.
```

Fill in the three `<fill>` rows from `layer3_multiseq_comparison.md` and the verdict from Step 3's decision rule.

- [ ] **Step 5: Commit results**

```bash
git add RESEARCH_NOTES.md diagnostics/results/multiseq/layer3_multiseq_comparison.md \
        diagnostics/results/multiseq/layer3_multiseq_v1train_hninfo_beta*.{json,md} \
        diagnostics/results/multiseq/_hninfo_run_log.txt
git commit -m "docs: Exp 34 hard-negative mining results and verdict"
```

---

## Spec Coverage Self-Review

Checking the plan against the spec, section by section:

- **§1 Hypothesis & decision rule** → Task 11 Step 3 applies the exact ±0.015 band.
- **§2 Scope (what changes/doesn't)** → Tasks 1–4 touch only `losses.py`; Task 5 touches `train.py`. No dataset, no model, no eval script changes. ✓
- **§3 Loss formulation** → Tasks 1→2→3 build up to the exact equations. β=0 recovery verified by Task 1 test; FNM by Task 2; β-weighting gradient amplification by Task 3; weight-sum invariant by Task 4. ✓
- **§4 Training & eval flow** → Task 8's `run_hn_finetune.sh` executes the 5-step flow exactly as specified. Task 11 runs it. ✓
- **§5 File structure** → All files listed in §5 appear as Create/Modify targets in Tasks 1–9.
- **§6 Dataset path (sentence-level IDs)** → Covered by Task 5 Step 3 (guard rejects `hninfo + use_group_labels=True`) and Task 6 (plumbing verification). Existing `build_training_data(use_group_labels=False)` already yields sentence IDs; no dataset code changes.
- **§7 Testing strategy** → 5 tests required; plan produces 6 (the extra is the plumbing contract test in Task 6, added because it caught a real risk during plan-writing).
- **§8 Risks & mitigations** →
  - Collapse: curves PNG is automatic via existing `save_training_curves`.
  - Breaking existing paths: Task 5 Step 6 verifies default = `AlignmentLoss`.
  - Misinterpretation: Task 11 Step 3 binds interpretation to the spec's decision rule.
  - Early instability: LR=1e-4 + cosine decay set in Task 8.
  - Empty N_neg guard: covered by `clamp_min(1)` in Tasks 3 and 4.
  - Micro up / macro down: Task 11 Step 2 and retrospective table capture both.
- **§9 Out-of-scope** → Plan does not touch batch size, representation, V2, fusion HOTA, or alternate loss families.
- **§10 Success criteria** → Task 11 artifact list matches §10 exactly.

**Placeholder scan:** None found. All code snippets are complete, all commands exact, all expected outputs specific.

**Type consistency:** `HardNegativeInfoNCE(temperature, beta, fnm)` used identically in Tasks 1, 2, 3, 4, 5, 6, 7. Helper method `compute_negative_weights(sim_matrix, sentence_ids) → (w, n_neg)` defined once (Task 4) and referenced only there. CLI flags `--loss {infonce,hninfo}` and `--beta FLOAT` used identically in Tasks 5, 8, 10.

**Gaps:** None relative to the spec.
