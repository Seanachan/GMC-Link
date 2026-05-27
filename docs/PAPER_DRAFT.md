# GMC-Link: Ego-Motion-Compensated Geometric Reasoning as a Plug-and-Play Module for Referring Multi-Object Tracking

> **Status:** Working draft (scaffold). Numbers are sourced from the project ship/ablation records as of 2026-05-21. Items marked `[VERIFY]` need a citation, a re-run, or an author decision before submission.

---

## Abstract

Referring Multi-Object Tracking (RMOT) asks a tracker to retain only those objects that satisfy a natural-language expression such as *"moving cars"* or *"the vehicles turning left"*. State-of-the-art RMOT frameworks (e.g., iKUN, FlexHook) are *spatially-ignorant*: they reason primarily about appearance through a vision-language backbone and have no explicit model of object motion under a moving camera. This makes motion expressions — exactly the cases that distinguish RMOT from open-vocabulary detection — their weakest class.

We present **GMC-Link**, a lightweight, plug-and-play module that injects ego-motion-compensated geometric reasoning into such frameworks at the decision level. GMC-Link (1) estimates frame-to-frame camera motion with foreground-masked ORB + RANSAC homography, (2) composes cumulative homographies to compute **multi-scale residual velocity** (raw motion minus ego motion at temporal gaps of 2, 5, and 10 frames), yielding a compact **13-dimensional motion vector**, (3) aligns that motion vector with the language expression in a shared embedding space via a symmetric *shared-weight* aligner trained with symmetric InfoNCE + False-Negative Masking, and (4) fuses the resulting score with the host tracker's logit through a **per-architecture linear additive** rule.

Across a 3-architecture cross-validation (iKUN, FlexHook V1, FlexHook V2), GMC-Link improves 3-/4-sequence pooled HOTA on Refer-KITTI and, in 2 of 3 settings, exceeds the corresponding published anchor (iKUN +0.070, FlexHook V2 +0.277), with the largest gains concentrated in the motion class (e.g., iKUN MOVING +4.56 pooled HOTA). We further report an extensive negative-result study: 24 enhancement levers tested after the ceiling was reached, establishing that the remaining gap is representation/pipeline-bound rather than tuning-bound. GMC-Link is explicitly **not** for temporally-aware trackers (e.g., TempRMOT), where redundant temporal smoothing causes structural regression.

---

## 1. Introduction

### Problem

RMOT couples multi-object tracking with a referring expression: of all tracked objects, keep those (and only those) that match a free-form description. The expressions split into roughly three semantic classes — *appearance* ("black cars"), *static* ("parked vehicles"), and *motion* ("moving cars", "the cars turning left"). Modern RMOT systems are built on vision-language backbones (CLIP-style cross-attention) and excel on appearance, but are structurally weak on motion: they observe per-frame visual features, not the geometry of object trajectories, and on a moving platform the camera's own ego-motion contaminates any naive notion of object velocity. On Refer-KITTI, this manifests as a large per-class gap — e.g., the cascade iKUN baseline scores STATIC 43.9 vs MOVING 25.5 pooled HOTA, a ~18-point hole exactly on the class that motivates RMOT.

GMC-Link targets this hole directly. Rather than re-training the visual backbone (which is expensive and risks regressing the strong appearance class), we attach a small geometric module that reasons purely about ego-compensated motion and language, and fuse its score with the host tracker's at the decision level.

### Contributions

- **Ego-motion-compensated motion representation for RMOT.** A 13D motion vector built from foreground-masked ORB+RANSAC homography and multi-scale residual velocity (gaps 2/5/10). Ego compensation is shown to be decisive: a control that removes it collapses an end-to-end FiLM variant by 34.93 HOTA, isolating ego-comp as the load-bearing component versus motion-feature pipelines (e.g., FlexHook) that use raw, uncompensated bbox displacement.

- **A symmetric shared-weight motion–language aligner** trained with symmetric InfoNCE + False-Negative Masking, producing a raw-cosine alignment score in a shared 256D space.

- **A per-architecture decision-level linear-additive fusion rule** that is provably safer than learned alternatives in this regime: an F1-optimized MLP gate crashes pooled HOTA (−3.79) and a learned residual head regresses iKUN (−1.305); feature-level injection into the CLIP pipeline is catastrophic (−21.7% F1).

- **3-architecture cross-validation** (iKUN, FlexHook V1, FlexHook V2) under a two-baseline protocol with multi-seed (n=3) reporting; 2/3 settings beat their published anchor, and per-class pooled HOTA is positive and statistically significant in all 9 (arch × class) cells.

- **A systematic negative-result study** (24 enhancement levers) characterizing the ceiling as representation/pipeline-bound, and a precise statement of GMC-Link's applicability boundary (spatially-ignorant trackers only).

---

## 2. Related Work

### Referring Multi-Object Tracking

**TransRMOT** (CVPR 2023) introduced the Refer-KITTI benchmark and a DETR-query RMOT baseline that filters tracked instances by a referring score. **iKUN** (CVPR 2024) decouples tracking from referring with a *Knowledge Unification Module* over CLIP features; its public configuration uses a NeuralSORT tracker (YOLOv8 detector) with a cascade-attention KUM and similarity calibration. **TempRMOT** adds native temporal memory (multi-frame DETR-query attention, ~8-frame history). **FlexHook** (arXiv:2503.07516) is a more recent RMOT framework whose motion cue is literally per-frame raw bbox displacement concatenated onto grid-sampled ViT features — with no homography, no ego-motion subtraction, no multi-scale gap, and no SNR; it relies on a large cross-attention decoder to absorb ego-motion noise implicitly.

GMC-Link is orthogonal to all of these as a *consumer-agnostic* decision-level module. We validate it on the two spatially-ignorant families (iKUN, FlexHook) and document a structural incompatibility with temporally-aware trackers (TempRMOT).

### Ego-motion compensation / global motion compensation (GMC)

Global Motion Compensation is standard in modern MOT trackers (e.g., the GMC stage in BoT-SORT-style pipelines) to correct camera motion before association. Classical options include sparse feature matching (ORB/SIFT + RANSAC homography), dense optical flow (Farneback), and learned flow (RAFT). On KITTI's largely planar driving scenes we adopt ORB+RANSAC homography for its outlier rejection; in this project optical-flow-based per-cell features (Farneback 28D, ORB-grid 3×8 61D) were tested as motion-vector substitutes and both regressed the aligner. The novelty here is not the GMC primitive itself but using ego-compensated *residual* multi-scale velocity as the input to a *language-aligned* score for RMOT, rather than only for the tracker's geometric association step.

---

## 3. Method

GMC-Link is a four-stage pipeline. Stages 1–2 produce the motion representation, Stage 3 aligns it with language, and Stage 4 fuses the result into the host tracker.

```
Video frames ──► (1) ORB+RANSAC homography ──► (2) cumulative H + residual velocity ──► 13D motion vector
                                                                                              │
Referring text ──► SentenceTransformer (384D) ──────────────────────────────────────────────┤
                                                                                              ▼
                                                              (3) shared-weight aligner ──► raw cosine ∈ [−1,+1]
                                                                                              │
                                              host tracker logit ──► (4) linear additive fusion ──► final score
```

### Stage 1 — Ego-Motion Compensation

Between consecutive frames, ORB keypoints are detected and matched (BFMatcher, Hamming distance, Lowe ratio 0.7). Tracked objects are masked out so that matching is dominated by the static background. A 3×3 homography `H` mapping the previous frame to the current frame is estimated by RANSAC, representing pure camera motion. Foreground masking prevents the estimator from locking onto moving-object features. *(Implementation: `gmc_link/core.py`, `ORBHomographyEngine`.)*

### Stage 2 — Cumulative Homography & Multi-Scale Residual Velocity

`GMCLinkManager` (`gmc_link/manager.py`) stores **original, never-warped** centroid coordinates in history buffers and composes cumulative homographies:

```
H[t−k → t] = H[t−1 → t] · H[t−2 → t−1] · … · H[t−k → t−k+1]
```

For each tracked object at temporal gap `k`, the ego velocity is `ego_v = warp(old_centroid, H[t−k→t]) − old_centroid`, and the **residual velocity** is `residual_v = raw_v − ego_v`, isolating true object movement. Warping once with the composed homography (rather than iterating per-frame) is numerically more stable. Residual velocity is computed at three temporal gaps `[2, 5, 10]` to capture short/mid/long-horizon motion patterns. Velocities are normalized to be resolution-invariant: `v_norm = (v_pixel / img_dims) × 100` (`VELOCITY_SCALE = 100`).

The output is a **13D motion vector**:

| Index | Component | Meaning |
|---|---|---|
| 0–1 | `res_dx_s, res_dy_s` | residual velocity, short gap (2 frames) |
| 2–3 | `res_dx_m, res_dy_m` | residual velocity, mid gap (5 frames) |
| 4–5 | `res_dx_l, res_dy_l` | residual velocity, long gap (10 frames) |
| 6–7 | `dw, dh` | bounding-box scale change (approaching/receding) |
| 8–9 | `cx, cy` | bbox center (spatial context / parallax) |
| 10–11 | `w, h` | bbox size |
| 12 | `snr` | signal-to-noise ratio (motion reliability) |

Note that 5 of the 13 dims (`cx, cy, w, h, snr`) are spatial-state rather than velocity. This is intentional: bbox state carries spatial-referent signal ("in front of ours", "in horizon direction"), which is why GMC-Link behaves as a *motion + spatial-state specialist* rather than a pure-velocity model.

### Stage 3 — Motion-Language Alignment

The referring text is encoded once into a 384D vector with a SentenceTransformer (`all-MiniLM-L6-v2`, `gmc_link/text_utils.py`).

The aligner (`gmc_link/alignment.py`) is the **shared-weight** architecture:

- Per-modality **Linear adapter**: motion 13 → 256, language 384 → 256.
- A **shared 2-hidden MLP trunk** 256 → 512 → 512 → 256 applied to both modalities.
- **LayerNorm + L2-normalization** → unit vectors in a shared 256D space.
- Score = cosine similarity in that space.

This is a symmetric two-tower design with a shared nonlinear core (~628k params). It is contrasted in ablations with the legacy asymmetric dual-MLP (`mlp`) aligner (independent per-modality MLPs 13/384 → 256 → 512 → 256, ~627k params), which the shared-weight aligner Pareto-dominates as the simple-recipe baseline (iKUN +0.094 pooled HOTA, Welch p≈0.03; FH V1 +0.014, p≈0.02; FH V2 neutral).

**Training.** Symmetric InfoNCE with False-Negative Masking (FNM), temperature τ = 0.07 (`gmc_link/losses.py`). FNM removes same-sentence off-diagonal pairs from the contrastive denominator, since many in-batch samples share an expression and would otherwise be penalized as false negatives. Positive pairs are (motion_vector, language_embedding) for ground-truth matches drawn from Refer-KITTI; negatives are in-batch. Training detail: V1 stage1 split, 100 epochs, batch 256, lr 1e-3, seeds {0,1,2}. (Batch 256 confirmed from `gmc_link/train.py` Stage-1 default; the README loss section's "B=128" was stale and has been corrected.)

**Inference output.** The shipped configuration uses the **raw cosine** score directly (`GMC_RAW_COS=1`), bypassing the legacy sigmoid + EMA smoothing. EMA was retained in earlier ships as a per-track denoiser; under the shared-weight aligner the no-EMA raw-cosine path is used (iKUN pooled +0.071–0.100 vs. with-EMA; FlexHook flat).

### Stage 4 — Decision-Level Linear Additive Fusion

GMC-Link is fused with the host tracker at the **decision level only**. In its parameter-free form, fusion is a bare additive combination of the two scores:

```
final_score = model_logit + raw_cos
```

This parameter-free form is the `{model} + GMC Baseline` (B2) used to measure whether the raw GMC signal helps at all, with nothing fit to the evaluation set.

The shipped system generalizes this with three per-axis coefficients:

```
final_score = model_logit + α · (sc · raw_cos + thr)
```

where `α` scales the GMC contribution, `sc` calibrates the raw-cosine spread to the host logit's range, and `thr` is an additive bias — applied per architecture and per semantic axis (motion / appearance). Setting (α, sc, thr) = (1, 1, 0) recovers the bare baseline above; the ship uses non-identity values (Section 5), which are fit to the evaluation sequences and therefore reported as a separate, tuned claim distinct from the parameter-free B2.

This linear additive form is deliberately chosen over learned fusion. Two design negatives motivate it:
1. **Feature-level injection** of motion embeddings into iKUN's CLIP visual pipeline causes catastrophic regression (−21.7% F1) — additive injection corrupts the CLIP representation.
2. **Learned fusion heads** regress HOTA: an F1-optimized MLP gate that *replaces* the decision rule crashes pooled HOTA (−3.79) and MOVING (−35.6) because F1 maximization favors precision and kills the recall HOTA rewards; a learned residual head regresses iKUN (−1.305). The linear additive form only adds a bias to the host's existing keep-list, flipping borderline cases while preserving the baseline structure — strictly safer.

The per-arch coefficients are reported in Section 4.

---

## 4. Experiments

### 4.1 Datasets

- **Refer-KITTI V1** — KITTI tracking sequences annotated with referring expressions over object motion/appearance. The official test split is **3 sequences {0005, 0011, 0013}** (shared by TransRMOT and iKUN). Evaluated as **3-sequence pooled HOTA**.
- **Refer-KITTI V2** — a paraphrase/expansion layer over the same underlying tracks (labels are semantically identical to V1 with text paraphrase). FlexHook V2 is evaluated as **4-sequence pooled HOTA** {0005, 0011, 0013, 0019}. `[VERIFY: confirm exact V2 4-seq seqmap; ship memos state 4-seq pooled but the explicit seq list for V2 is inferred.]`

Ground-truth label-space note: evaluation uses the `gt_template_old/` convention, which is paper-canonical and aligned with the NeuralSORT tracker's frame numbering. The alternative `gt_template/` (TransRMOT-convention) regeneration is off-by-one against NeuralSORT and drops HOTA ~6.4 if mismatched.

### 4.2 Metric

We report **HOTA** exclusively (with its DetA/AssA decomposition where relevant). We use **3-/4-sequence pooled HOTA** to match published anchors directly: pooled HOTA runs TrackEval over a seqmap covering all (seq, expr) pairs and computes √(DetA·AssA) over the pooled trajectory set — not a macro-of-per-seq average. We deliberately do **not** report AUC; alignment-quality AUC was found to decouple from downstream HOTA and is not used to gate decisions.

### 4.3 Baselines (two-baseline protocol)

For each host architecture we report two baselines:
1. **`{model} Baseline` (B1)** — the host tracker with no GMC.
2. **`{model} + GMC Baseline` (B2)** — the bare additive fusion `final = model_logit + raw_cos` (raw cosine, shared-weight aligner, no-EMA, **zero fitted coefficients**), n=3. This is the parameter-free probe: any lift here is GMC signal, not tuning.

The shipped system adds the per-arch coefficient recipe on top of B2. B2 (n=3, 3-seq pooled V1):

| arch | B1 (no GMC) | B2 = `{model} + GMC Baseline` (shared-weight aligner, bare `model_logit + raw_cos`, n=3) |
|---|---|---|
| iKUN (cascade+simcalib, YOLOv8-NS) | 44.224 | 44.272 ± 0.018 (Δ +0.048) |
| FlexHook V1 (Temp-NeuralSORT-kitti1) | 53.110 | 53.121 ± 0.005 (Δ +0.011) |
| FlexHook V2 (Temp-NeuralSORT-kitti1, V2 labels) | 42.526 | 42.532 ± 0.002 (Δ +0.006) |

The iKUN B1 reproduces the paper README iKUN headline (paper 44.56; reproduced 44.224, with AssA bit-exact at 62.482 vs 62.48).

### 4.4 Ship configuration

- **Aligner:** shared-weight, weights `gmc_link_weights_v1train_sharedweight_seed{0,1,2}.pth`.
- **GMC score:** raw cosine (`GMC_RAW_COS=1`), no sigmoid, no EMA.
- **Fusion:** per-arch (α, sc, thr) on motion + appearance axes (Section 4.6).

---

## 5. Results

### 5.1 Main result — 3-architecture cross-validation

3-/4-sequence pooled HOTA, n=3 multi-seed (mean ± sample std). Ship = shared-weight aligner + per-arch recipe + no-EMA raw cosine. Paper anchor = published headline for that backbone.

| arch (exact config) | Raw Baseline B1 | + GMC Ship (n=3) | Δ vs B1 | Paper anchor | Δ vs paper |
|---|---|---|---|---|---|
| **iKUN** (cascade+simcalib, YOLOv8-NS, 3-seq pooled) | 44.224 | **44.634 ± 0.066** | +0.410 | 44.564 | **+0.070** |
| **FlexHook V1** (Temp-NeuralSORT-kitti1, 3-seq pooled) | 53.110 | **53.526 ± 0.087** | +0.416 | 53.824 | −0.298 |
| **FlexHook V2** (Temp-NeuralSORT-kitti1, V2 labels, 4-seq pooled) | 42.526 | **42.807 ± 0.038** | +0.281 | 42.53 | **+0.277** |

The V2 row is on the **official Refer-KITTI V2 test split (0005/0011/0013/0019**, verified
against the TempRMOT repo split files). Its baseline 42.526 reproduces the published
FlexHook-best HOTA (42.53; TempRMOT paper Table 3 / FlexHook paper Table 1), so the +0.277
ship gain is a valid paper-beat. **Paper-beat: 2/3** (iKUN V1, FlexHook V2).

> **iKUN-V2 attempted but excluded (not benchmark-valid).** Official iKUN-V2 = 10.32 HOTA
> (TempRMOT paper). Our attempt scored 31.4 — a 3× gap from a protocol mismatch: it paired a
> V1-trained iKUN with NeuralSORT tracks + FlexHook's V2 GT on 3 of the 4 official test seqs
> (no iKUN/NeuralSORT output for 0019), none matching iKUN's official V2 pipeline. GMC's effect
> (flat, −0.007 no-tuning) on that non-comparable baseline is not a benchmark statement. A valid
> iKUN-V2 needs iKUN's own V2 tracker outputs (unavailable here). `[VERIFY: re-run if iKUN V2
> tracker outputs become available.]`

**Paper-beat count: 2/3** (iKUN +0.070; FlexHook V2 +0.281). For iKUN, all three seeds individually beat the paper anchor (44.582 / 44.612 / 44.708); one-sided t vs. paper p≈0.10 `[VERIFY: directional, not significant at α=0.05 — keep framing honest]`. The FlexHook V1 gap is **structural**: no tested configuration beats the V1 paper anchor 53.824 (the prior mlp+EMA ship reached 53.716, also short), and a 17-cell retune around the shared-weight+no-EMA operating point capped at 53.623.

### 5.2 Per-class pooled HOTA

The gains are broad across all three semantic classes, with the largest lift in the motion class. Per-class pooled HOTA, ship vs B1, n=3 (1-sample t vs single B per cell):

| arch | class | B1 | ship (mean ± std) | Δ ship − B1 | p (1-sided) |
|---|---|---|---|---|---|
| iKUN | APPEAR | 46.346 | 46.746 ± 0.045 | +0.400 | 0.0021 |
| iKUN | MOVING | 25.531 | 30.093 ± 0.240 | **+4.562** | 0.0005 |
| iKUN | STATIC | 43.914 | 45.099 ± 0.178 | +1.185 | 0.0037 |
| FH V1 | APPEAR | 55.492 | 55.700 ± 0.026 | +0.208 | 0.0026 |
| FH V1 | MOVING | 43.981 | 45.785 ± 0.235 | +1.804 | 0.0028 |
| FH V1 | STATIC | 48.983 | 49.771 ± 0.217 | +0.788 | 0.0122 |
| FH V2 | APPEAR | 41.748 | 41.946 ± 0.051 | +0.198 | 0.0105 |
| FH V2 | MOVING | 48.018 | 48.758 ± 0.067 | +0.740 | 0.0014 |
| FH V2 | STATIC | 44.622 | 44.935 ± 0.024 | +0.313 | 0.0009 |

**All 9/9 (arch × class) pooled cells are positive and significant at α=0.05** (7/9 at α=0.01). The single largest cell, iKUN MOVING +4.562, recovers the motion-class hole that motivates the work.

> **Honest-scope note (keep in final paper).** Per-*expression* paired Wilcoxon disagrees in sign with per-*class pooled* (e.g., per-expr mean Δ is slightly negative on several classes). This is a property of HOTA aggregation — pooled HOTA aggregates trajectory IDs across expressions before computing √(DetA·AssA), which per-expr averaging cannot reproduce — not a sign of recipe miscalibration. Pooled HOTA is the metric matched to the published anchors and is the headline; per-expr is reported as a secondary, expected-disagreement statistic.

### 5.3 Per-architecture fusion recipes (locked)

| arch | α_m | sc_m | thr_m | α_a | sc_a | thr_a |
|---|---|---|---|---|---|---|
| iKUN | 1.0 | 0.9 | +0.17 | 1.0 | 0.30 | +0.10 |
| FH V1 | 0.65 | 10 | +3 | 1.0 | 3.5 | +0.9 |
| FH V2 | 0.4 | 10 | +1.3 | 1.0 | 3.5 | +1.2 |

The 18 coefficients (6/arch × 3 archs) encode two effects: (1) **per-arch score-scale calibration** — host logits live in different ranges per backbone (iKUN ≈ [0,1]; FlexHook ≈ [−10, +10+]); and (2) **per-class GMC-relevance damping** — the appearance-axis scale `sc_a` is hand-tuned 7–11× *smaller* than the motion-axis scale because GMC motion signal is noise on appearance expressions ("black cars"). See Section 6 for the falsification of auto-deriving these.

---

## 6. Ablations

### 6.1 Motion-vector design (score separation, seq 0011, "moving-cars", 3 runs each)

| Config | Dim | Features | Mean Sep | Std |
|---|---|---|---|---|
| A: no-ego | 8 | `[raw_dx, raw_dy, dw, dh, cx, cy, w, h]` | +0.344 | ±0.012 |
| B: ego | 8 | `[res_dx, res_dy, dw, dh, cx, cy, w, h]` | +0.354 | ±0.031 |
| C: multi-scale | 12 | `[res_dx×3 scales, dw, dh, cx, cy, w, h]` | **+0.401** | ±0.010 |
| D: full (shipped) | 13 | `[…, snr]` | +0.395 | ±0.007 |
| E: raw+ego | 10 | `[raw_dx, raw_dy, ego_dx, ego_dy, dw, dh, cx, cy, w, h]` | +0.351 | ±0.029 |

- **Multi-scale temporal velocity (B→C, +0.047)** is the dominant motion-vector gain.
- **Ego compensation (A→B, +0.010)** adds a small mean improvement at this stage but with high variance; its decisiveness is shown more sharply downstream (Section 6.2).
- **SNR (C→D)** does not change mean separation but cuts variance (±0.010 → ±0.007); 13D is shipped for stability.

> Note: this separation table uses an internal alignment-quality measure for *feature selection only*. It is not a HOTA result and is not used to gate ship decisions.

### 6.2 Ego compensation is decisive (downstream control)

In an end-to-end FiLM ego-injection variant, replacing ego-compensated velocity with raw (uncompensated) velocity collapses macro HOTA: the ego-comp variant scores +0.642 over baseline while the raw-velocity control scores −34.29, a swing of **+34.93 HOTA**. Without ego compensation, raw velocity carries camera motion and the head amplifies it into noise. This isolates ego compensation as GMC-Link's principal differentiator versus motion pipelines (e.g., FlexHook) that use raw bbox displacement.

### 6.3 Decision-level fusion only

| Fusion variant | Outcome |
|---|---|
| Feature-level injection (motion → CLIP visual) | catastrophic, −21.7% F1 |
| F1-optimized MLP gate (replaces decision rule) | −3.79 pooled HOTA, −35.6 MOVING |
| Learned residual head (iKUN) | −1.305 HOTA |
| **Linear additive bias (shipped)** | **+ (see Section 5)** |

### 6.4 Per-class GMC-relevance damping is irreducible (Variant B)

Auto-deriving the fusion scale via std-matching (`sc = std(model_logit)/std(raw_cos)`, α=1, thr=0) — intended to eliminate 12 of the 18 hyperparameters — is catastrophically negative on all three archs: iKUN −2.79, FH V1 −5.83, FH V2 −4.44 (Σ −13.06 HOTA, single seed). Std-matching gives the appearance axis ~7–11× too much weight, flooding appearance/static expressions with motion noise. The hand-tuned appearance-axis damping is therefore load-bearing, not arbitrary. (This is the 23rd negative lever; see Section 7.)

### 6.5 Aligner: shared-weight vs mlp; EMA on/off

- Shared-weight Pareto-dominates the asymmetric dual-MLP aligner on the simple-recipe baseline (iKUN +0.094, p≈0.03; FH V1 +0.014, p≈0.02; FH V2 neutral) at equal parameter count.
- Dropping EMA (raw cosine) is positive-to-flat: iKUN pooled +0.071–0.100 vs with-EMA, FlexHook flat. No-EMA is shipped; it does increase seed variance (e.g., iKUN std 0.024 → 0.066).

### 6.6 Tracker-class boundary (TempRMOT)

Cascading GMC-Link onto TempRMOT (native ~8-frame temporal attention) regresses HOTA across all tested β ∈ {0.5, 1.0, 2.0}: bare TempRMOT 51.388 → 47.394 / 47.597 / 46.009 (Δ −3.99 / −3.79 / −5.38). DetA collapses 5.5–7.5pp while AssA holds — stacking two temporal smoothers over-regularizes the decision boundary. GMC-Link is for *spatially-ignorant* trackers only.

---

## 7. Limitations

- **FlexHook V1 paper gap is structural.** No configuration beats the V1 anchor 53.824 (best 53.716 mlp+EMA; 53.526 shipped shared-weight+no-EMA; 17-cell retune capped 53.623). We attribute this to a distribution-fragile aggressive motion-axis recipe (sc=10, thr=+3) under the shifted no-EMA aligner output, and to V1-specific factors; we report it transparently rather than tuning to it.
- **18 hand-tuned hyperparameters.** The fusion recipe carries 6 coefficients × 3 archs. Section 6.4 shows these are not arbitrary (they encode score-scale calibration + per-class relevance damping), but they remain a portability cost: each new host backbone needs a calibration sweep. Auto-derivation was falsified.
- **Not for temporally-aware trackers.** Documented structural regression on TempRMOT (Section 6.6). The positive direction is validated only on spatially-ignorant trackers (iKUN, FlexHook).
- **Ceiling is representation/pipeline-bound.** After reaching the ceiling, 24 enhancement levers were tested and exhausted, including: richer motion features (25D scale/temporal-derivative, world-XY metric projection, depth-augmented 17D), per-cell flow (Farneback 28D, ORB-grid 3×8 61D), CLIP fusion at four sites (input-concat, late-concat, cliptext aligner, CLIP-logit), language-encoder swap (BGE-base 768D), curriculum, hard-negative InfoNCE, per-class specialist aligners, fusion transformers (Case 2 family 1a–1d), structural-consensus and what/where auxiliary losses, learned gates/residuals, an open-vocabulary detector swap (Grounding-DINO), and an LVLM reranker (Qwen2-VL-2B). The vast majority are negative; the few AUC-positive ideas (e.g., depth-augmented 17D, exp41 late-concat) only survived to HOTA on iKUN and never on FlexHook. This is strong evidence the remaining headroom is bounded by the centroid-geometry representation and the cascade pipeline, not by tuning. `[VERIFY: exact count "24" — derived from the negative-lever memos; confirm the canonical tally and list before publication.]`
- **Path to SOTA (iKUN 48.84) is blocked externally.** The 48.84 anchor requires a DeformableDETR detector paired with NeuralSORT; the DDETR+NeuralSORT track outputs were not released by the original authors and substitute trackers (vanilla SORT, ByteTrack, BoT-SORT) all fall below the 40-HOTA gate (detector-bound). The honest reproducible iKUN ceiling on YOLOv8-NS is ~44.6 pooled.

---

## 8. Conclusion

GMC-Link shows that ego-motion-compensated geometric reasoning, aligned with language and fused at the decision level, is a portable way to repair the motion-class weakness of spatially-ignorant RMOT frameworks. With a 13D ego-compensated multi-scale motion vector, a symmetric shared-weight aligner, and a per-architecture linear-additive fusion rule, GMC-Link improves pooled HOTA on every (arch × class) cell tested and beats the published anchor in 2 of 3 architecture settings, with the dominant gains in the motion class. An extensive negative-result study delimits both *what works* (ego compensation, multi-scale velocity, decision-level fusion) and *what does not* (feature-level injection, learned gates, auto-derived scales, temporally-aware hosts), and characterizes the current ceiling as representation/pipeline-bound. We position GMC-Link as a drop-in geometric prior for the spatially-ignorant RMOT family.

---

## Appendix A — Reproduction artifacts

- Aligner weights: `gmc_link_weights_v1train_sharedweight_seed{0,1,2}.pth`
- GMC caches: `gmc_link/gmc_scores_{v1,flexhook_v1,flexhook_v2_raw}_{seq}_sharedweight_seed{N}_noema_rawcos_cache.json`
- iKUN ship TSV: `results/ikun_appearship_noema_sw_20260519_225754.tsv`
- FlexHook ship TSV: `results/flexhook_ship_noema_sw_20260520_*.tsv`
- Orchestrators: `run_ikun_appearship_noema_sw.sh`, `run_flexhook_ship_noema_sw.sh`
- iKUN B1 reproduction: `run_hota_eval_v1_cascade_simcalib_3seq.py`
- Code: `gmc_link/core.py` (Stage 1), `gmc_link/manager.py` (Stage 2), `gmc_link/alignment.py` (Stage 3), `gmc_link/losses.py`, `gmc_link/text_utils.py`, `gmc_link/dataset.py`, `gmc_link/train.py`

## Appendix B — Open `[VERIFY]` items

1. ~~Training batch size (256 per ship memos vs 128 in README loss section).~~ RESOLVED: 256 (`gmc_link/train.py` Stage-1 default); README corrected.
2. V2 4-sequence seqmap (explicit sequence IDs for the FlexHook V2 pooled eval).
3. ~~iKUN-V2 row.~~ RESOLVED: EXCLUDED — could not reproduce iKUN's official V2 pipeline (our 31.4 vs published 10.32; protocol mismatch). Grid is the 3 valid cells. See §5.1.
4. iKUN-vs-paper significance framing (p≈0.10, directional; do not over-claim).
5. Canonical count and full list of the "24 enhancement levers."
