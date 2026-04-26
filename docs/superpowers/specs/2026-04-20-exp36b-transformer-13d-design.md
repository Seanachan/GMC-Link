# Exp 36B — Temporal Transformer on 13D Design

**Date:** 2026-04-20
**Branch:** `exp/hn-mining` (continues after Exp 36A + 36A-v2 negative)
**Status:** Spec

## Goal

Test whether architecture capacity (temporal context) is the lever that Exp 36A (features) was not. Train `MotionLanguageAligner(architecture="temporal_transformer", seq_len=30)` on the original 13D motion vector and compare held-out AUC against stage1 MLP baseline.

## Why Now

Exp 36A + v2 ruled out features: both scale-difference and spec-correct temporal-derivative accel_multiscale + sin/cos heading produced 0.747 and 0.741 (vs stage1 0.779). Remaining open lever per retrospective: temporal context / architecture capacity.

## Scope

### Architecture

Existing `MotionLanguageAligner(architecture="temporal_transformer", seq_len=30)` in `gmc_link/alignment.py` — no code changes needed. Already:
- 2-layer transformer encoder, d_model=128, nhead=4, GELU
- CLS token pooling → 256D shared embedding
- Positional encoding learned, sized seq_len+1

### Training config

| Param | Value |
|---|---|
| Split | V1 train |
| Motion dim | **13** (no extras) |
| Architecture | `temporal_transformer` |
| seq_len T | **30** (3 sec at 10fps) |
| Epochs | 5 (match stage1) |
| Batch size | 128 |
| LR | 3e-4 |
| Save path | `gmc_link_weights_v1train_exp36b.pth` |

### Files

No source edits. Only new orchestrator `run_exp36b.sh`.

### Eval

Same held-out multi-seq protocol as 36A:

```bash
for seq in 0005 0011 0013; do
  python diagnostics/diag_gt_cosine_distributions.py \
      --weights gmc_link_weights_v1train_exp36b.pth --seq "$seq"
done
```

Diag already branches on `architecture == "temporal_transformer"` (line 345) and builds sliding windows of seq_len motion vectors.

### Aggregation

Four-way table: stage1, exp36a (scale-diff), exp36a_v2 (temporal-deriv), exp36b (transformer on 13D).

## Decision rule (pre-registered)

Based on `mean_auc_micro` over {0005, 0011, 0013}:

| Result | Verdict | Action |
|---|---|---|
| ≥ 0.80 | **Win** | Temporal context is the lever. Publish; consider Exp 36C = transformer on 25D. |
| 0.79 ≤ x < 0.80 | **Marginal** | Seed sweep {42, 123, 7} for n=3 average. |
| < 0.79 | **Flat** | Neither features nor arch help at 5 epochs on V1. Ceiling is data-bound (label density, not quality). Next lever = supervision expansion (V1+V2 joint train, pseudo-labels, or curriculum). |

## Cost

- Training: ~30 min (transformer slower than MLP; seq_len=30 × 147k samples)
- Eval 3 seqs: ~15 min (sliding-window encoding per track)
- Aggregate: < 1 min
- **Total ~45 min**

## Risks

| Risk | Mitigation |
|---|---|
| seq_len=30 exceeds some track lengths | Diag script pads + masks already (code path verified lines 366–373) |
| Transformer under-trained at 5 epochs | Stage1 is also 5 epochs; comparable compute budget. If marginal, extend. |
| Memory on batch 128 × T=30 × 13D | Small: ~5 MB/batch. Fine on any GPU. |

## Non-goals

- No feature expansion (that was 36A, ruled out).
- No encoder swap.
- No loss variant change.
