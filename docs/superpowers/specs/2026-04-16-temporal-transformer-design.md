# Temporal Transformer Motion Encoder

**Date:** 2026-04-16
**Branch:** `exp/temporal-transformer` (off `exp/training-improvements`)
**Motivation:** The MLP-based motion projector takes a single 13D snapshot per track, producing embeddings that cluster in a narrow cosine band (mean separation +0.195, AUC 0.779). Even enriching the vector with 9 additional features (Exp 31) couldn't break the ~0.79 ceiling. The bottleneck is not missing features — it's missing temporal context. A transformer encoder over T consecutive frames can learn temporal patterns (braking, acceleration, lane changes) that a single-frame MLP cannot represent, producing more distinctive embeddings with wider cosine separation.

## Baseline

Stage 1 group-level model, 13D features, AUC **0.779**, mean separation **+0.195** on Layer 3 GT cosine diagnostic (seq 0011).

## Architecture

### TemporalMotionEncoder

Replaces the `motion_projector` MLP inside `MotionLanguageAligner`.

```
Input: (batch, T, 13)  — T=10 frames of 13D motion vectors per track

1. Linear(13 → d_model=64)         — per-frame projection
2. Prepend learnable [CLS] token   — sequence becomes (T+1, 64)
3. + Learnable positional encoding  — (T+1, d_model)
4. TransformerEncoderLayer          — 1 layer, n_head=4, d_model=64, dim_feedforward=128
5. Take [CLS] output               — (batch, 64)
6. Linear(64 → 256) + LayerNorm    — project to shared embedding space
```

- **d_model=64**: input is only 13D; larger dims risk overfitting
- **n_head=4**: each head sees 16 dims, can specialize on different motion aspects
- **1 layer**: minimal starting point; scale up only if warranted
- **~20K new parameters**: lightweight compared to the current MLP's ~200K
- **Bidirectional self-attention**: all T frames attend to all other frames. No causal masking — the entire window is past data at inference time
- **Padding mask**: tracks with < T history frames are zero-padded; attention ignores padded positions via key_padding_mask

### Language projector

**Unchanged.** Still `Linear(384→512) → ReLU → Dropout → Linear(512→256) → LayerNorm`.

### Per-frame 13D vector

**Unchanged.** Each frame in the sequence uses the same 13D feature layout: `[res_dx_s, res_dy_s, res_dx_m, res_dy_m, res_dx_l, res_dy_l, dw, dh, cx, cy, w, h, snr]`. Multi-scale velocity is kept per frame for backward compatibility and to give the transformer richer per-timestep context.

## Training Data Pipeline

### Current behavior

`_generate_positive_pairs` produces one 13D vector per track per valid frame. Each vector is an independent training sample.

### New behavior

Group consecutive frames per track into sliding windows of length T.

```
Track 5 has frames [80, 81, 82, ..., 120]:
  Sample at frame 89: [13D_f80, 13D_f81, ..., 13D_f89]  (T=10)
  Sample at frame 90: [13D_f81, 13D_f82, ..., 13D_f90]  (T=10)
  ...stride-1 sliding window
```

- **Sliding window, stride 1**: maximizes training data. A track visible for N frames yields N-T+1 sequence samples
- **Short tracks (< T frames)**: zero-pad from the left, provide a boolean padding mask following PyTorch convention: `[True, ..., True, False, ..., False]` where **True = padded (ignore), False = valid**. Example: a track with 4 frames and T=10 gets mask `[T,T,T,T,T,T,F,F,F,F]`
- **Expected sample count**: ~400-600K (up from ~147K), since each track contributes multiple overlapping windows
- **Language embedding and label**: identical for all windows of the same track+expression pair

### Dataset output format

`MotionLanguageDataset.__getitem__` returns:
- `motion_seq`: `(T, 13)` float tensor — the frame sequence
- `padding_mask`: `(T+1,)` bool tensor — PyTorch convention: **True = padded (ignore), False = valid**. Length T+1 because [CLS] is prepended in the model. The [CLS] position is always False (always valid)
- `lang_emb`: `(384,)` float tensor — unchanged
- `label`: int tensor — unchanged

### Collate function

Stacks into `(batch, T, 13)` motion, `(batch, T+1)` mask, `(batch, 384)` language, `(batch,)` labels.

## Inference Pipeline

### GMCLinkManager changes

Currently `process_frame()` builds one 13D vector per track from centroid history. Changes:

1. For each track, gather the last T frames of 13D vectors from history (instead of collapsing into one)
2. If fewer than T frames available, zero-pad and create a mask
3. Pass `(1, T, 13)` sequence through the transformer encoder instead of the MLP
4. Rest of the pipeline unchanged: L2-normalize → cosine sim → margin → sigmoid → EMA smoothing

The centroid/homography history buffers already store up to 11 frames (`frame_gap=10`), so T=10 fits naturally.

### Key change: per-frame velocity computation

Currently, `process_frame()` computes multi-scale residual velocities using gaps [2, 5, 10] relative to the current frame. For the temporal sequence, each historical frame needs its own 13D vector computed relative to *its own* past frames.

Approach: store the per-frame 13D vectors in a deque as they're computed each frame, rather than recomputing history. Each call to `process_frame()` computes one new 13D vector for the current frame and appends it to the track's history buffer.

## Diagnostics

### diag_gt_cosine_distributions.py changes

Currently computes one 13D vector per track per frame and batches through the MLP. Changes:

1. After computing all per-track per-frame vectors (existing `compute_motion_vectors_for_all_tracks`), group them into T-length sliding windows per track
2. Pass sequences through the transformer encoder
3. Auto-detect architecture from checkpoint (new `architecture` key: `"mlp"` or `"temporal_transformer"`)

## Training Protocol

- Stage 1 group-level training (same as Exp 30 baseline)
- 100 epochs, lr=1e-3, batch_size=256, cosine annealing
- V1 split (train on all except seq 0011)
- Supervised InfoNCE + FNM loss (unchanged)
- No extra features (base 13D)

## Success Criteria

- **Primary**: Mean AUC > 0.800 (Δ > +0.021 over baseline 0.779)
- **Secondary**: Mean separation > +0.250 (currently +0.195) — the core "scores should spread out" metric
- **Tertiary**: Per-expression worst-case AUC improvement (currently some expressions near chance at 0.5)

## Future Scaling (if successful)

If 1-layer transformer improves over baseline:
- Test 2-3 layers
- Test d_model=128
- Test larger T (15, 20)
- Test cross-attention with language tokens (Approach B from brainstorming)

If it doesn't improve: the temporal signal is not the bottleneck, and we should investigate other directions (different loss functions, multi-sequence evaluation, or architecture search on the language side).
