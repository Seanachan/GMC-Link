# FlexHook-Adjacent Alignment with Ego-Compensated Motion Tokens — Design Spec

**Date:** 2026-04-19
**Author:** Seanachan (with Claude Opus 4.7)
**Related:** Exp 34 retrospective (commit `f55713f`), `RESEARCH_NOTES.md` §Exp 34; FlexHook (arXiv:2503.07516, code at `~/FlexHook/`)
**Experiment ID:** Exp 35 (tentative)
**Branch target:** `exp/flexhook-adjacent` (off `main` or off `exp/hn-mining`)

---

## 0. One-Sentence Pitch

Replace the 13D→MLP Layer-3 aligner with a compact joint cross-attention decoder that consumes CLIP text tokens, CLIP ViT patch tokens (pooled under each track's bbox neighborhood), and 3 ego-compensated motion tokens produced by a tiny temporal transformer — keeping all current training signals (InfoNCE + FNM) and evaluation gates (multi-seq Layer-3 micro AUC).

## 1. Hypothesis & Falsification Criterion

**Context.** Exp 34's HN-InfoNCE β-grid proved the 0.779 multi-seq micro AUC ceiling is **representation-bound**: the 13D motion → 384D language shared-MLP architecture cannot discriminate finer than Exp 33's ceiling, regardless of the contrastive loss. Further loss-side work is negative-EV. The next-step levers are all representation-side: richer motion encoder, stronger language encoder, or a richer visual/multimodal context.

**Null hypothesis (H₀).** A FlexHook-adjacent joint decoder with CLIP text + CLIP ViT patch tokens + tiny temporal-transformer motion tokens, trained with the existing InfoNCE+FNM signal, fails to raise Layer-3 multi-seq micro AUC meaningfully above the representation-bound 0.779 ceiling.

**Alternative (H₁).** The new decoder moves the ceiling: a richer multimodal query — text tokens + localized visual patches + temporally-aware ego-compensated motion — lets the aligner disentangle expressions the 13D→MLP could not.

**Pre-registered decision rule** on multi-seq micro AUC (macro- and per-seq reported for robustness):

- **≥ 0.830** — Strong positive. Representation ceiling moved; invest in Exp 36 scaling (more motion tokens, fine-tune CLIP top blocks, ablation breadth).
- **0.795 – 0.829** — Positive. Meaningful improvement beyond Exp 33 ±1σ; run the ablation battery (§7) before declaring a headline number.
- **0.779 ± 0.015 (0.764 – 0.794)** — Inconclusive. Could be noise; inspect qualitative ranking shifts on the 5 expressions Exp 33 bottomed on (parking pedestrians, faster-than-ours, horizon-direction). If no shift, declare this architecture family insufficient.
- **< 0.764** — Strongly negative. The added multimodal context is actively hurting — probably a training-regime issue (CLIP gradients, token imbalance). Debug once; if not fixed in one iteration, pivot to Approach B (per-frame motion tokens) or Approach C (reference-point conditioning).

The ±0.015 band matches the Exp 33 weight-to-weight spread on micro AUC.

**Critical ablation (§7) — the ego-compensation proof.** Whether the architecture beats 0.779 or not, we run the **raw-displacement vs ego-compensated** head-to-head inside the new decoder. The delta between these two variants is the paper-worthy claim about global-motion compensation, independent of whether the overall architecture sets new SOTA.

## 2. Scope: What Changes, What Doesn't

### Does NOT change
- `gmc_link/core.py` — `ORBHomographyEngine` (ego-motion estimation stays identical).
- `gmc_link/manager.py` — `GMCLinkManager` multi-scale residual velocity computation stays identical. `FRAME_GAPS = [2, 5, 10]`, `VELOCITY_SCALE = 100`.
- `gmc_link/utils.py` — `MotionBuffer`, `ScoreBuffer` EMAs stay identical.
- `gmc_link/losses.py` — `AlignmentLoss` (symmetric InfoNCE + FNM) reused without modification.
- `gmc_link/fusion_head.py` — downstream fusion MLP stays for end-to-end eval; Layer 5 pipeline unchanged.
- `run_hota_eval*.py` — HOTA evaluation drivers unchanged.
- Cache format for existing 13D snapshots stays readable (but Exp 35 writes a *new* cache key).
- Refer-KITTI V1 train split, multi-seq held-out (0005, 0011, 0013) split.

### Does change (additive; a new module alongside existing code paths)
- **New:** `gmc_link/aligner_v2.py` — `FlexHookAdjacentAligner` module (architecture in §4). Alongside `gmc_link/alignment.py`, not replacing it.
- **New:** `gmc_link/motion_encoder.py` — `TemporalMotionEncoder` (tiny temporal transformer over multi-scale residual-velocity windows; reuses the encoder shape sketched in `2026-04-16-temporal-transformer-design.md` but outputs 3 scale-tokens rather than one `[CLS]`).
- **New:** `gmc_link/visual_encoder.py` — `ClipPatchExtractor` (CLIP ViT-B/16 frozen; pools ViT patch tokens under an expanded bbox per track).
- **New:** `gmc_link/text_encoder_clip.py` — thin wrapper around `open_clip` or `transformers` CLIP text encoder (frozen by default).
- `gmc_link/dataset.py` — extends Exp 35 pipeline:
  - Stores per-track motion *windows* (length T=10 multi-scale vectors) instead of a single snapshot.
  - Stores the frame's bbox in image coordinates (for patch-token pooling at train time).
  - Stores the reference frame ID (so we can load the corresponding image on demand).
  - **New cache key** — does not overwrite the existing 13D cache.
- `gmc_link/train.py` — add `--arch {mlp,flexhook}` flag; default stays `mlp`. Loads the v2 aligner + dataset when `--arch flexhook`.
- **New driver:** `run_flexhook_adjacent.sh` — trains one checkpoint, runs multi-seq eval across (0005, 0011, 0013), aggregates into `diagnostics/results/multiseq/layer3_multiseq_comparison.md` as a 15th row.
- **New ablation driver:** `run_flexhook_ablations.sh` — trains the 3 variants in §7.

### Explicit non-goals for Exp 35
- We do **NOT** replace the fusion head (Layer 4) or re-run Layer-5 HOTA as the primary gate. Exp 35's eval surface is Layer-3 multi-seq AUC.
- We do **NOT** fine-tune CLIP. Both the text encoder and ViT stay frozen. Fine-tuning is deferred to Exp 36 conditional on a positive Exp 35.
- We do **NOT** implement FlexHook's PCD masked cross-attention (Approach C). That is a follow-up.
- We do **NOT** implement per-frame motion tokens (Approach B). That is a conditional escalation if Approach A stalls near 0.779.
- We do **NOT** change the train/held-out seq split, the motion keyword list, or the `is_motion_flag` logic.

## 3. Architecture Overview (Approach A)

```
                                Per (track, expression, frame-window) query
                                           │
                                           ▼
                     ┌─────────────────────────────────────────┐
                     │   Joint Cross-Attention Decoder (ours)   │
                     │   2 layers, d_model=256, 4 heads         │
                     │   learned query q_0 ∈ ℝ^{1×256}          │
                     └─────────────────────────────────────────┘
                         ▲                ▲                ▲
                         │                │                │
           ┌─────────────┘        ┌───────┘        └───────────────┐
           │                      │                                │
    Text tokens             Motion tokens                    ViT patch tokens
    (CLIP txt, frozen)      (TemporalMotionEncoder)          (CLIP ViT-B/16, frozen,
    (L_txt, 512)  → Linear  (T=10 window of 13D, multi-      pooled under track bbox)
    → (L_txt, 256)          scale residual velocity)          (~10–30 patches, 768)
                            → (3 tokens × 256) one per        → Linear → (~20, 256)
                              gap ∈ {2, 5, 10}
```

### 3.1 Motion encoder (`TemporalMotionEncoder`)

Input: `(batch, T=10, 13)` multi-scale residual velocity + bbox + SNR (schema unchanged).

```
x = Linear(13 → 128)(x)                              # per-frame projection
x = x + learned_positional_embed[:T, :]              # T=10 pos embeddings
x = TransformerEncoder(                              # 2 layers
        d_model=128, nhead=4, dim_ff=256,
        num_layers=2
    )(x)                                             # (B, T, 128)
# 3 scale-specific pool queries: short, mid, long
pool_q = learned_scale_queries                       # (3, 128)
motion_tokens = MHA(pool_q, x, x)                    # (3, 128)
motion_tokens = Linear(128 → 256)(motion_tokens)     # (3, 256)
```

**Why 3 scale-pool queries instead of one `[CLS]`:** the three temporal gaps `{2, 5, 10}` already partition motion into short/mid/long. A scale-specific query preserves that inductive bias while letting attention choose *which frames* within the window contribute to each scale. One token per scale == 3 motion tokens to the joint decoder.

Parameter budget: ~200K. Unit-testable standalone.

### 3.2 Text encoder (CLIP text, frozen)

- Model: `ViT-B/16` CLIP (OpenAI release or OpenCLIP; whichever is on disk first).
- Keep all token embeddings (not just `[EOS]`): a query of form `"moving cars"` produces 3 content tokens; `"left vehicles which are parking"` produces 7. Padding tokens masked out via `key_padding_mask`.
- Project `(L_txt, 512) → Linear → (L_txt, 256)`, LayerNorm. This projector is the only trainable part of the text path.

### 3.3 Visual encoder (CLIP ViT, frozen)

- Same `ViT-B/16` backbone (shared with text via the same CLIP checkpoint; saves disk & memory).
- For each `(frame, track)`, compute the reference frame's full ViT patch tokens once: `(196, 768)` on a 224×224 resize.
- **Pool under expanded bbox:** expand the track's bbox by 1.5× in each dimension, clamp to frame bounds, select the subset of patch tokens whose `16×16` grid cell overlaps the expanded bbox. Typical count: 10–30 tokens.
- Project `(N_patch, 768) → Linear → (N_patch, 256)`, LayerNorm. This projector is the only trainable part of the visual path.

**Why expanded bbox rather than the full 14×14 grid:** reduces attention-pool size by 7–20× and matches FlexHook's philosophy of conditioning visual features on object location. Expanded (not tight) so context (neighboring vehicles, lane geometry) leaks in. Not language-conditioned reference points — that's Approach C.

### 3.4 Joint decoder

```
q = learned_query.expand(batch, 1, 256)             # one query per sample
kv_tokens = concat([
    text_tokens,        # (L_txt, 256) + learned [TEXT_TYPE] emb
    motion_tokens,      # (3, 256)     + learned [MOT_TYPE]  emb
    patch_tokens        # (N_patch, 256) + learned [VIS_TYPE] emb
], dim=1)

key_padding_mask = concat([
    text_padding,
    zeros(3),           # motion always present
    patch_padding       # rare: bbox-off-frame case
])

for layer in [DecoderLayer, DecoderLayer]:           # 2 layers
    q = layer(q, kv_tokens, key_padding_mask)        # cross-attn → self-attn → FFN

score_emb = q.squeeze(1)                             # (batch, 256)
```

- Each `DecoderLayer`: MHA cross-attn (q → kv_tokens) → MHA self-attn (single-query, effectively identity here since batch-of-one) → FFN (256 → 512 → 256) → LayerNorm.
- Learned type embeddings distinguish text / motion / vision within the mixed KV pool (standard trick).
- Output: a single 256D embedding per sample.

### 3.5 Alignment score

Kept identical to current Layer-3 behavior: the decoder output is the **motion-side** 256D vector. The language-side 256D vector is the mean-pooled (over content tokens) projected CLIP text embedding. Cosine similarity → remapped to `[0, 1]` as alignment score.

This preserves compatibility with the existing diagnostic pipeline (`diagnostics/diag_gt_cosine_distributions.py`) — same `.npz` schema.

## 4. Data Flow

### Per training sample (one positive pair)

```
1. Dataset yields: (track_id, expression_id, ref_frame_id, bbox_xyxy,
                    motion_window[T=10, 13], language_text)
2. Image loader: load ref_frame → resize to 224×224 → CLIP mean/std normalize
3. ViT forward → (196, 768) patch tokens → select under expanded bbox → project → (N_patch, 256)
4. Motion encoder: (T=10, 13) → (3, 256)
5. Text encoder: tokenize(language_text) → CLIP txt forward → (L_txt, 512) → project → (L_txt, 256)
6. Joint decoder: concat KV, cross-attn → (256) motion-side embedding
7. Language-side: mean-pool projected text tokens over non-pad positions → (256)
8. Cosine → score → InfoNCE loss
```

### Per evaluation sample (one track vs one expression, frame-level AUC)

Same as training, except we run the decoder over every GT-frame of the track (frame-level scores aggregated to track-level as before — current Layer-3 protocol).

### Image caching

- Refer-KITTI V1 train split has ~10K valid (track, frame) pairs. ViT-B/16 on a 3060 Ti processes ~50 images/sec at 224×224. Pre-compute patch tokens once per frame, cache to `diagnostics/cache/vit_patches_{split}.npz` keyed by `frame_id`. Amortized cost: ~200s one-off per split.
- Cache shape: `(N_frames, 196, 768)` as float16 — 144 MB for 10K frames. Fits in RAM.

## 5. Training

### Loss

**Identical to current.** Symmetric InfoNCE + FNM:

```
logits[i, j] = cos(motion_emb_i, lang_emb_j) / τ
positive_mask[i, j] = (sentence_id[i] == sentence_id[j])
L = 0.5 * (InfoNCE_m2l + InfoNCE_l2m)
```

with τ = 0.07, FNM applied via `positive_mask`. **β=0 (no hard-negative mining)** — Exp 34 showed HN hurts this family; we stay with vanilla InfoNCE for Exp 35's headline run.

### Regime

| Knob | Value | Notes |
|---|---|---|
| Optimizer | AdamW | β=(0.9, 0.999), wd=1e-4 |
| LR | 3e-4 (decoder, projectors, motion encoder); **0.0** (CLIP text, CLIP ViT) | Two param groups |
| LR schedule | Cosine w/ 1-epoch warmup | |
| Batch size | 128 | Smaller than 256 because ViT tokens add memory |
| Epochs | 40 | Compare against Exp 33 Stage 1's 60 |
| Precision | fp16 autocast for CLIP forwards, fp32 for decoder | CLIP stability |
| Cache regeneration | On first run only | |
| Seed | 42 | For the headline run |

### Data

- Train: Refer-KITTI V1 train split, existing expression-annotation pipeline.
- Valid: 10% held-out from train (same as Exp 33), sentence-level split (not track-level).
- No changes to motion-keyword list, `is_motion_flag`, or augmentation (±2px jitter still applied to bbox before CLIP patch pooling; velocity normalization unchanged).

### What training looks like per epoch (order of magnitude)

~10K pairs / 128 batch = 78 steps/epoch. Each step: 128 CLIP txt forwards (cheap, frozen), 128 motion-encoder forwards (cheap), 128 ViT patch lookups (cached), 128 decoder forwards (small). Expected <30s/epoch on 3060 Ti. 40 epochs ≈ 20 min. Cache generation dominates wall-time on first run.

## 6. Evaluation

### Primary gate
`diagnostics/aggregate_multiseq.py` over `{0005, 0011, 0013}`, micro-AUC pooled across the 33 expressions.

Report:
- Micro AUC (primary gate; apply §1 decision rule).
- Macro AUC (per-seq averaged, n=12 expressions in ≥2 seqs).
- Per-seq AUC on 0011 (legacy reference — the systematically-worst seq per memory).
- Full per-expression breakdown table (as in `layer3_multiseq_v1train_stage1.md`).

### Secondary (diagnostic, not gate)
- Attention weights on a handful of queries: which KV tokens does the learned query attend to most? Expected: motion-dominant for "moving cars", visual-dominant for "red car", text-dominant for out-of-distribution.
- GT-cosine histogram on seq 0011 — compare to Exp 33's +0.195 mean separation.

### Artifacts

Produced by `run_flexhook_adjacent.sh`, written to:
- `gmc_link_weights_v1train_flexhook_adj.pth` (checkpoint, gitignored).
- `diagnostics/results/multiseq/layer3_{seq}_v1train_flexhook_adj.{npz,png}`.
- `diagnostics/results/multiseq/layer3_multiseq_v1train_flexhook_adj.md`.
- `diagnostics/results/multiseq/layer3_multiseq_comparison.md` (updated to 15 rows).

## 7. Ablations (Required, Not Optional)

These run **after** the headline Approach A completes, using the same code path with different config flags. Each ablation trains from scratch at seed 42, same regime.

### A1 — Raw displacement (FlexHook-style motion)

Motion encoder input schema is kept at 13D (so the `TemporalMotionEncoder` architecture is unchanged across A and A1 — only the feature values differ). In A1, the 6 residual-velocity dims `[res_dx_s, res_dy_s, res_dx_m, res_dy_m, res_dx_l, res_dy_l]` are replaced with raw single-frame displacement `[raw_dx, raw_dy]` replicated across all three scale slots (i.e., `[raw_dx, raw_dy, raw_dx, raw_dy, raw_dx, raw_dy]`). The bbox geometry dims `[dw, dh, cx, cy, w, h]` and `snr` stay as in A. **This isolates the ego-compensation + multi-scale contribution in a single ablation**; the `dataset.py` v2 path gets an `--ablation raw-disp` flag that swaps the feature values at generation time. Everything else (decoder, ViT, CLIP, InfoNCE) identical. **Purpose:** isolates the ego-compensation contribution. This is the headline ablation — deltas A vs A1 are the proof of the global-motion thesis.

### A2 — Motion-free control
Drop motion tokens entirely. Decoder sees only text + ViT patches. **Purpose:** lower-bound — what does a pure text+vision model achieve? If A2 beats our headline A, motion doesn't help the new architecture; write that up honestly.

### A3 — ViT-free control
Drop ViT patch tokens. Decoder sees only text + motion tokens. **Purpose:** isolates the visual-context contribution. If A3 matches A, ViT cross-attention is cosmetic and we should simplify.

### Ablation report
Single table in `RESEARCH_NOTES.md` §Exp 35 with rows `{A, A1, A2, A3}`, columns `{micro AUC, macro AUC, Δ vs A, Δ vs Exp 34 β=0 baseline (0.779)}`. Commit as the same commit that lands the Exp 35 retrospective.

## 8. Risks & Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| CLIP's ViT-B/16 patch features dominate and motion tokens get ignored by the decoder (motion attention → ~0). | Medium | Log attention weights per epoch; if motion attention < 10% consistently, add a motion-attention auxiliary loss or warm-start by pre-training A3 (motion + text only) for 10 epochs before adding ViT. |
| ViT patch pooling under bbox is noisy (occlusion, poor tracking). | Medium | Expanded bbox (1.5×) already mitigates; additionally mask out patches whose center falls outside the track's predicted next-frame position. |
| CLIP tokenizer truncates long expressions (some Refer-KITTI V2 expressions are 10+ tokens). | Low | CLIP supports 77 tokens; all Refer-KITTI V1 expressions are < 15 words. |
| Image-loading I/O becomes the bottleneck during training. | Medium | Patch-token cache (§4) eliminates per-step I/O once warm. |
| The `expanded_bbox → patch` selection has edge cases (bbox leaves frame). | Low | Clamp to frame bounds; if 0 patches selected, fall back to the 9 center patches of the frame and log a warning. |
| InfoNCE over the new embedding geometry has a different optimal τ. | Medium | Defer to Exp 36 — for Exp 35, τ=0.07 stays (one variable at a time). |
| CLIP-frozen gradients still flow through projectors → numerical noise. | Low | fp32 decoder + LayerNorm on projector output handles it. |
| N_patch varies per sample → batch-inefficient attention. | Medium | Pad patches to max-per-batch (typical max ~30); use `key_padding_mask` to ignore padding. |

## 9. Open Questions (to resolve before implementation plan)

1. **CLIP variant:** OpenAI CLIP ViT-B/16 or OpenCLIP ViT-B/16 (laion2b)? → **Default to OpenCLIP ViT-B/16 laion2b** (stronger on open-vocab benchmarks, pip-installable via `open_clip_torch`).
2. **Patch-token cache location:** `diagnostics/cache/` under gitignore, or `~/data/Dataset/refer-kitti-v2/cache/`? → **Default to dataset path** so multiple branches share the cache.
3. **Does the existing `AlignmentLoss` (symmetric InfoNCE + FNM) need modification?** → No. Keep as-is; it operates on `(motion_emb, lang_emb)` pairs and doesn't care how they were produced.
4. **Do we keep `VELOCITY_SCALE=100` and the 13D layout unchanged inside the temporal transformer?** → **Yes.** The motion encoder is architecturally new but the input tensor schema matches `GMCLinkManager`'s output exactly, preserving inference-time compatibility.
5. **Should `run_flexhook_adjacent.sh` also run the three ablations, or separate driver?** → **Separate driver** (`run_flexhook_ablations.sh`) so a failed ablation doesn't block updating `layer3_multiseq_comparison.md` with the headline number.

## 10. Exit Criteria

Exp 35 is complete when **all** of the following are true:

- [ ] Approach A headline checkpoint trained, multi-seq diagnostic run, micro AUC logged against §1 decision rule.
- [ ] Ablations A1 (raw-displacement), A2 (no-motion), A3 (no-ViT) all trained and evaluated.
- [ ] `layer3_multiseq_comparison.md` updated with 4 new rows (A, A1, A2, A3).
- [ ] `RESEARCH_NOTES.md` §Exp 35 retrospective written: headline number, ablation table, attention-weight observations, verdict against §1, next-step recommendation.
- [ ] At least the ego-compensation ablation (A vs A1) delta has been called out explicitly — positive or negative — as the contribution claim for this line of work.

---

## Appendix A: File-by-file diff sketch

```
gmc_link/
├── alignment.py               ← unchanged
├── aligner_v2.py              ← NEW: FlexHookAdjacentAligner
├── motion_encoder.py          ← NEW: TemporalMotionEncoder (3 scale-tokens out)
├── visual_encoder.py          ← NEW: ClipPatchExtractor (frozen ViT + bbox pool)
├── text_encoder_clip.py       ← NEW: thin CLIP txt wrapper
├── dataset.py                 ← MODIFY: add v2 path w/ motion window + bbox + ref_frame
├── losses.py                  ← unchanged
├── train.py                   ← MODIFY: --arch {mlp,flexhook}
├── core.py, manager.py        ← unchanged
└── fusion_head.py             ← unchanged

diagnostics/
├── diag_gt_cosine_distributions.py     ← MODIFY: --arch flexhook path
└── aggregate_multiseq.py               ← unchanged

run_flexhook_adjacent.sh       ← NEW: headline run + multi-seq eval
run_flexhook_ablations.sh      ← NEW: A1, A2, A3 ablation grid
```

## Appendix B: Parameter count (approximate)

- TemporalMotionEncoder: ~200K
- Text projector (512 → 256 + LN): ~130K
- Vision projector (768 → 256 + LN): ~200K
- Type embeddings (3 × 256): ~800
- Joint decoder (2 layers, d=256, 4 heads, ffn=512): ~2.1M
- Learned query + scale pool queries: ~1K
- **Trainable total: ~2.6M** (CLIP's ~150M params frozen)

Compared to current MLP aligner (~500K trainable): ~5× larger but still a fraction of CLIP itself. Well within 3060 Ti memory for batch=128.
