# Exp 36A — Rich Motion Features (13D → 25D, MLP) Design

**Date:** 2026-04-20
**Branch:** `exp/hn-mining` (continues from label-audit confirmation of 0% noise)
**Author:** Seanachan (human) + Claude (agent)
**Status:** Spec, pending user review

## Goal

Beat the V1 held-out ceiling (micro-AUC 0.779) by expanding the motion
representation from 13D to 25D. Keep the MLP aligner unchanged to isolate
feature-gain from architecture-gain. If 36A wins, skip 36B (transformer on
13D); if 36A flat, 36B tests whether temporal context is the lever.

## Why Now

- Label audit (2026-04-20) returned 0% noise rate after legend clarification.
  Labels trustworthy → ceiling is representation-bound per
  `project_exp34_representation_bound.md`.
- Exp 34 retrospective: "positive-EV = richer motion features, stronger
  language encoder, OR wider/deeper motion head". Richer features is the
  cheapest of the three (no new encoder, no arch change).
- Flagged motion verbs: brake, turn, park, same/counter direction. These
  map literally to acceleration + heading — exactly the features added.

## Scope

### Feature additions (13 → 25)

Current 13D (unchanged):
```
[0:3]  res_dx × {2,5,10} scales    (ego-compensated residual velocity)
[3:6]  res_dy × {2,5,10} scales
[6:8]  dw, dh                       (size change rate)
[8:10] cx, cy                       (center position, normalized)
[10:12] w, h                        (size, normalized)
[12]   snr                          (signal-to-noise ratio)
```

New 12D appended:
```
[13:16] accel_x × {2,5,10}          NEW: (res_dx_t - res_dx_{t-k})/k × 100
[16:19] accel_y × {2,5,10}          NEW: same for dy
[19:22] sin(heading) × {2,5,10}     NEW: sin(atan2(res_dy, res_dx))
[22:25] cos(heading) × {2,5,10}     NEW: cos(atan2(res_dy, res_dx))
```

### Architecture

`MotionLanguageAligner` in `gmc_link/alignment.py`:
- `motion_dim=25` (was 13) — constructor default bumped
- `motion_mlp: 25 → 256 → 256` (LayerNorm + GELU, unchanged otherwise)
- `lang_mlp: 384 → 256 → 256` (unchanged)
- Shared 256D embedding, cosine sim remapped to [0, 1]

### Loss

- Symmetric InfoNCE + False-Negative Masking (`gmc_link/losses.py`)
- Temperature τ=0.07
- No changes

### Training config (match stage1)

| Param | Value |
|---|---|
| Split | V1 train |
| Epochs | 5 |
| Batch size | 128 |
| LR | 3e-4 |
| Optimizer | AdamW (default in `train.py`) |
| Seed | 42 |
| Save path | `gmc_link_weights_v1train_exp36a.pth` |

### Files to modify

| File | Change |
|---|---|
| `gmc_link/manager.py` | `compute_motion_vec()` appends 12 features from stored velocity/position history |
| `gmc_link/dataset.py` | `_build_motion_vector()` (or equivalent builder) appends 12 features in same order |
| `gmc_link/alignment.py` | `MotionLanguageAligner(motion_dim=25)` default |
| `gmc_link/train.py` | Verify `--motion-dim` flag flows through; default to 25 |
| `diagnostics/diag_gt_cosine_distributions.py` | No change — reads motion_dim from state dict |

### Feature computation details

**Acceleration (6D):**
- `accel_x_k[t] = (res_dx_k[t] - res_dx_k[t-k]) / k × VELOCITY_SCALE`
- Same scale as velocity (×100). Needs one extra stored past velocity per scale.
- When t < k frames into a track, return 0 (no accel history).

**Heading (6D = 3 sin + 3 cos):**
- `θ_k = atan2(res_dy_k, res_dx_k)`
- Output `(sin θ_k, cos θ_k)` pair — smooth across ±π wraparound
- When motion magnitude `|v| < ε` (e.g., 1e-3), `sin θ → 0, cos θ → 1` naturally
  from atan2(0, 0) = 0. No explicit mask needed.

### Data/config symmetry

Critical: feature order and scaling MUST match between `GMCLinkManager`
(inference) and `dataset.py` (training). Off-by-one or different
normalization silently breaks the aligner.

Unit test (inline smoke): same synthetic 30-frame trajectory through both
paths → identical 25-D vector modulo expected EMA difference.

## Eval

### Held-out multi-seq

```bash
for seq in 0005 0011 0013; do
  python diagnostics/diag_gt_cosine_distributions.py \
      --weights gmc_link_weights_v1train_exp36a.pth \
      --seq "$seq" \
      --arch aligner  # default, matches stage1 topology
done
```

### Aggregation

`diagnostics/aggregate_multiseq.py` with both stage1 and exp36a weights in
`--weights` so the comparison markdown emits side-by-side micro/macro AUC.

## Decision rule (pre-registered)

Based on `mean_auc_micro` across {0005, 0011, 0013}:

| Result | Verdict | Action |
|---|---|---|
| ≥ 0.80 (≥ +0.02 vs 0.779) | **Win** | Publish finding. Consider Exp 36C = features + transformer for stretch goal. |
| 0.79 ≤ x < 0.80 | **Marginal** | Re-run with seeds {42, 123, 7} for n=3 average. Check if signal is stable. |
| < 0.79 | **Flat** | Features are not the lever. Move to Exp 36B = 13D + transformer T=30. |

Also report per-seq to detect hidden gains: if 0011 (worst) moves ≥ +0.03
but mean micro stays flat, that's a targeted signal worth preserving even
without threshold win.

## Cost estimate

- Training: ~25 min (same magnitude as stage1)
- Eval per seq: ~2 min × 3 seqs = 6 min
- Aggregation: < 1 min
- **Total ~35 min** wall-clock

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Accel amplifies jitter on short scale (k=2) | Multi-scale: larger k=10 is naturally smoother. If k=2 dominates and is noisy, drop it next iteration. |
| Heading undefined at near-zero motion | `atan2(0, 0)=0` → sin=0, cos=1. Degenerate but smooth. Natural "stationary" encoding. |
| 25D → 256D projection could under-regularize | Current MLP has LayerNorm + GELU, already regularized. Only 12 new params per hidden neuron, 3k new params total — trivial. |
| Feature code drift between manager.py and dataset.py | Inline smoke test comparing synthetic trajectory outputs. |
| Stage1 at 0.779 used exact same train/eval split — no benchmark contamination | Same data, same eval, only feature expansion. Direct comparison valid. |

## Non-goals

- No transformer (that is 36B if needed).
- No language encoder swap (MiniLM stays).
- No new loss variants (InfoNCE+FNM stays).
- No factorial over architecture (single condition: 25D + MLP).

## Self-review (inline)

- [x] Placeholder scan: no TBD/TODO/vague spots
- [x] Internal consistency: feature count 12 (6 accel + 6 sin/cos heading)
      = Option B heading with sin/cos split. 13+12=25. Checks out.
- [x] Scope check: one experiment, one config, one decision rule. Single plan.
- [x] Ambiguity check: feature order fixed by indices [0:25]; scaling fixed
      by VELOCITY_SCALE constant; heading wraparound fix is explicit
      (sin/cos pair, not raw angle).
