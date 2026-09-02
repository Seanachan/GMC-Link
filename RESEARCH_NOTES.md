# GMC-Link Research Notes

Experiment log for the GMC-Link Referring Multi-Object Tracking (RMOT) module.
Authoritative chronological-by-theme record with exact metric values. Spans the
early aligner bring-up (Exp 1-26), the V1 multi-sequence ceiling investigation
(Exp 27-43), the paper-reproduction + fusion-ship discovery campaign (2026-04 to
2026-05), and the 2026-05 ship-saga (shared_weight aligner, EMA-drop, CLIP-fusion
arch-split, ship simplification reversal).

> **Metric policy.** HOTA is the only ship metric reported in the modern sections.
> AUC was dropped as a metric mid-project (it decoupled from HOTA on every
> aligner-internal lever); historical AUC numbers from Exp 26-43 are retained only
> where they are the literal record of what was measured at the time, and every
> such lever has a HOTA verdict where one exists.
>
> **Result-citation policy.** Every modern result cites the exact recipe + n + std.
> "Ship", "baseline", "ours" without a recipe are not used. The canonical anchors are:
> - **B1 = `{model} Baseline`** (no GMC, deterministic): iKUN 44.224, FH V1 53.110, FH V2 42.526.
> - **B2 = `{model} + GMC Baseline`** (sw aligner, simple fusion α=1 sc=1 thr=0, raw cos, no EMA, n=3): iKUN 44.272±0.018, FH V1 53.121±0.005, FH V2 42.532±0.002.
> - **Paper claims**: iKUN 44.564, FH V1 53.824, FH V2 42.526 (3-seq pooled, YOLOv8-NS / NeuralSORT, `gt_template_old`).

---

## 0. Project Recap

GMC-Link is a plug-and-play module that fuses ego-motion-compensated object geometry
with natural-language descriptions to score which tracked objects match a referring
expression ("moving cars", "turning vehicles", "parked cars"). It is a decision-level
add-on for spatially-ignorant vision-language RMOT frameworks (iKUN, FlexHook,
TransRMOT).

Pipeline (see `CLAUDE.md` for the full architecture):

1. **Ego-motion compensation** (`core.py`) — ORB + RANSAC homography per frame.
2. **Cumulative homography + multi-scale residual velocity** (`manager.py`) — warp
   original centroids by composed H, compute residual velocity at frame gaps (2, 5, 10),
   emit the **13D motion vector** `[res_dx×3, res_dy×3, dw, dh, cx, cy, w, h, snr]`.
3. **Motion-language alignment** (`alignment.py`) — dual-tower aligner trained with
   InfoNCE + False-Negative Masking (τ=0.07); cosine similarity is the GMC score.
4. **Decision-level fusion** — additive logit bias into the host tracker's score.

The current ship (Section 7) trains the aligner on Refer-KITTI V1 and reports
3-seq pooled HOTA on the V1 held-out test split (seqs 0005, 0011, 0013).

---

## 1. Aligner Bring-up (Exp 1-26)

Early single-sequence work establishing the geometry pipeline, loss function, and
13D feature vector. Reported in F1 / score-separation terms (HOTA not yet in use).
Preserved verbatim from the original notes for provenance; **superseded by the
multi-sequence HOTA regime from Exp 27 onward.**

### Exp 1-6: Loss + scaling shakeout

CLIP-style symmetric cross-entropy stalled at the `ln(N)` random floor (Exp 1-5)
because single-frame velocities were ~0.001 and every batch had many same-sentence
false negatives. `VELOCITY_SCALE=100` (Exp 6) amplified inputs to ~1.0 magnitude.

### Exp 7: Switch to BCE loss

Per-pair `BCEWithLogitsLoss` + 3:1 negative sampling + proper train/test split
(train 15/16/18, test 11). Train loss 0.29, acc 82.2%, but **separation on held-out
seq 0011 was −0.04** (no generalization).

### Exp 8-9: GMC object masking + deeper MLP + hard negatives

Passing YOLO bboxes to the GMC engine stopped object features contaminating the
background homography (FP −50%). Deeper motion projector + hard negatives (zero/inverted
velocity) → **separation +0.5229** on seq 0011 (GT 0.7344 vs non-GT 0.2115). Best
early result.

### Exp 10-15: Optical-flow bake-off + ego-motion restoration

| Exp | Method | Separation | Note |
|-----|--------|-----------|------|
| 10 | Farneback dense flow | +0.2750 | noisier than ORB |
| 11 | RAFT learned flow | +0.1056 | best train acc 89.91%, worst test sep |
| 14 | Centroid-diff + ORB ego-comp | +0.1086 | physically locks world velocity; FP 389, TP 235 |
| 15 | + YOLOv8x detector + label-format fix | +0.1080 | more detections, parallax FP returns |

**Decision: ORB+RANSAC homography beats Farneback and RAFT** on KITTI planar scenes
(better outlier rejection). Ego-motion compensation is mathematically necessary —
disabling it (Exp 12-13) made co-moving cars look stationary and parked cars look
moving, destroying motion semantics.

### Exp 16-17: 6D → 8D geometry-aware vector

`[dx, dy]` → `[dx, dy, cx, cy, w, h]` (Exp 16, sep **+0.3059**) → add depth-scaling
velocities `[dw, dh]` + ±2px synthetic jitter + 4D EMA smoothing (Exp 17, TP peak 369).
Spatial context grants implicit 3D parallax correction.

### Exp 18-19: Tracker integration — the temporal-tracker dichotomy

| Integration | HOTA | Verdict |
|-------------|------|---------|
| TransRMOT + GMC (`min(vit, gmc)`) | 42.61 (vs 38.06 baseline) | **+4.55**, AssA +18.4 |
| TempRMOT + GMC (thr=0.4) | 43.18 (vs 49.93 baseline) | **−6.75**, structural regression |

**Key design decision: GMC-Link helps spatially-ignorant frameworks (TransRMOT,
iKUN) but is destructive when force-coupled with natively temporal trackers
(TempRMOT) — redundant temporal constraints over-prune.** Confirmed again in 2026-04
(Exp 37 Stage D, below).

### Exp 20-22: iKUN fusion — decision-level wins, feature injection catastrophic

- **Stage 1 OR-logic** (no training): Overall F1 0.5730→0.5863 (+1.3%).
- **Stage 2 learned MLP fusion head**: Overall F1 →0.5895 (+1.7%).
- **Stage 3 feature-level injection** into iKUN's CLIP visual pipeline: gate opened →
  **−21.7% F1 catastrophic collapse** (Exp 21). The 256D motion embedding occupies a
  different manifold than CLIP visual features.
- **Exp 22 InfoNCE+FNM aligner → fusion head**: Overall F1 **0.6569 (+8.4%)**, Motion
  F1 0.7328, the headline early result. InfoNCE's structured metric space transfers
  far better to fusion than BCE's pointwise scores.

**Design decision: decision-level fusion only. Never inject motion features into the
visual backbone.**

### Exp 23-26: 9D → 13D motion vector

- **Exp 23 (9D + SNR + fixed τ=0.07)**: added signal-to-noise ratio feature; learnable
  τ collapsed to 0.0099 (saturation), so τ fixed at 0.07. Seq-0011 "moving cars"
  separation **+0.263**.
- **Exp 24 (13D multi-scale + motion-only filter)**: residual velocity at 3 frame gaps
  (2,5,10) + skip appearance-only training expressions. Separation **+0.362** (+42%
  vs Exp 23). This locked the **13D motion vector** that is still shipped.
- **Exp 25 (encoder swap mpnet-768D)**: +0.009 vs MiniLM-384D — negligible; **kept
  MiniLM-L6-v2 (384D)**.
- **Exp 26 (inference margin calibration)**: margin=0.05 shifted sigmoid reference,
  separation 0.336→0.415. Later superseded by the raw-cosine fusion regime.

---

## 2. Multi-Sequence Ceiling Investigation (Exp 27-34)

Transition from seq-0011-only diagnostics to **3-seq held-out (0005, 0011, 0013)**
evaluation. The recurring finding: the aligner is **representation-bound**, not
loss-bound or capacity-bound. (Historical AUC is the literal record here; the
representation-bound conclusion is what carries forward.)

### Exp 27: Additive logit fusion replaces the MLP head

The learned MLP fusion head (3→32→16→1) over-recalled on unseen seqs (HOTA −1.40).
Replaced with **1-parameter additive logit fusion**: `final = ikun_logit + α·logit(gmc)`
for motion/stationary, identity for appearance. Best **α=0.07 → HOTA 43.02 (+1.87)** on
seq 0011. The optimal training-data α (0.4) was ~6× too large to generalize — the
additive form wins by preserving iKUN's calibrated decision boundary. This is the
ancestor of the modern linear-additive ship recipe.

### Exp 28-29: Contrastive fixes + training-dynamics ablation (reverted)

- FNM, motion-type grouping, and z-score normalization (Exp 28): grouping helped train
  acc (15%→67%) but z-norm hurt generalization; **all three reverted** (`38e43d2`).
- Longer epochs / LR warmup / learnable τ / grad clip (Exp 29): all within ±0.017 AUC
  of baseline 0.759. **Bottleneck is feature separability, not training dynamics.**

### Exp 30: Group-level (Stage 1) training — the production baseline

Training on 6 motion-type groups (coarse) instead of 82 expressions reached **AUC
0.779 (+0.020)**, the best single improvement found and the basis for "stage1". Stage 2
fine-tuning on 82 classes degraded it (0.779→0.777). **Stage 1 group-level training
became the production aligner recipe.**

### Exp 31-32: Feature enrichment + temporal transformer (negative)

- **Exp 31 (9 candidate features)**: best F3 acceleration 0.788 (+0.009); none reached
  the >0.800 success bar. 13D space has a hard ~0.79 ceiling.
- **Exp 32 (temporal transformer T=10, [CLS] aggregation)**: AUC 0.770 (−0.009).
  Temporal context is **not** the bottleneck.

### Exp 33: Multi-sequence re-evaluation of Exp 30-32

Aggregating across 0005/0011/0013 confirmed: seq 0011 is representative but
systematically the **worst** seq; the ~0.78 ceiling is real, not seq-0011 variance;
Exp 32's transformer regression is genuine (gap widens to 0.032 pooled). Seq 0013 has
only n=2 expressions — its per-seq numbers are noise.

### Exp 34: HN-InfoNCE β-grid — the ceiling is representation-bound

Hard-Negative InfoNCE (Robinson) finetune at β∈{0.5,1.0,2.0} **monotonically degraded**:

| Model | micro AUC | Δ vs stage1 0.779 |
|-------|-----------|-------------------|
| stage1 baseline | **0.779** | — |
| HN β=0.5 | 0.753 | −0.026 |
| HN β=1.0 | 0.746 | −0.033 |
| HN β=2.0 | 0.733 | −0.046 |

**Verdict: REPRESENTATION-BOUND.** Upweighting hard negatives amplifies noise once the
13D/MLP encoder is saturated. Stop iterating on contrastive-loss variants. This framing
governs all subsequent aligner-side levers (Exp 36-43).

---

## 3. Representation-Side Levers at the 0.779 Ceiling (Exp 35-41 AUC phase)

A systematic exhaustion of feature, architecture, encoder, supervision, and fusion-site
levers at the stage1 aligner. **All AUC-negative.** Several were later HOTA-revisited
(Section 6) and a subset flipped POS for iKUN — the origin of the "never kill at AUC"
rule. AUC values below are the literal record of the kill decision.

| Exp | Lever | micro AUC | Δ vs 0.779 | Verdict |
|-----|-------|-----------|-----------|---------|
| 35 | FlexHook-adjacent cross-attn decoder | 0.741 | −0.038 | NEG (HOTA revisit deferred) |
| 36A | 25D MLP (scale-diff accel) | 0.747 | −0.032 | NEG |
| 36A-v2 | 25D MLP (temporal-deriv accel) | 0.741 | −0.038 | NEG |
| 36B | transformer T=30, 5ep | 0.754 | −0.025 | NEG |
| 36B-long | transformer T=30, 25ep | 0.761 | −0.018 | NEG |
| 36C | V1+V2 joint train | 0.767 micro | flat | macro +0.005 / −27% std; micro flat |
| 36D | BGE-base 768D encoder | 0.735 | −0.044 | NEG (worst of 7) |
| 36E | curriculum (100ep+50ep) | 0.762 | −0.017 | NEG |
| 37-A | ego source swap | — | −0.036 | NEG |
| 37-B | OMF 28D Farneback per-cell flow | 0.624 | −0.155 | NEG (worst lever) |
| 37-C | EMAP concat | — | −0.024 | NEG |
| 37 | ORB-grid 3×8 (61D) | — | −0.107 | NEG, 4 motion sign-flips |
| 38-A/B/C | ego-injection (replace/concat/all13d) | — | best −0.496 (HOTA) | NEG (3 recipes) |
| 39 | CLIP B/32 visual 64D early-concat → 13D | 0.7223 | −0.057 | NEG (feature drowning) |
| 41 | CLIP-text late-concat (256⊕256) | 0.731 | −0.048 | NEG at AUC → **HOTA revisit flips iKUN** (§6) |

**Conclusions from this phase:**
- Features, capacity, encoder, supervision, and aligner-internal fusion site are all
  exhausted at AUC. The 0.779 ceiling is pipeline/representation-bound.
- **Farneback per-cell flow actively corrupts the 13D signal** (37-B worst at 0.624);
  sparse per-cell ORB is also worse than dense (3×8 grid −0.107). The single dense
  ORB+homography ego estimate is the right granularity.
- Exp 37 Stage D (β-grid on **TempRMOT**) was NEG by −3.8 to −5.4 HOTA — re-confirming
  the Exp 19 dichotomy: do not cascade GMC onto temporally-aware trackers.

---

## 4. Paper Reproduction + Fusion-Ship Discovery (2026-04-28 to 2026-05-03)

Pivot from the AUC ceiling to **HOTA on the real tracker pipeline**. The fusion recipe
is the lever, not the aligner. This phase produced the multi-seed paper-beating ship.

### 4.1 Paper reproduction at 3-seq pooled HOTA

Three different statistics were being conflated. The paper headline is **3-seq POOLED
HOTA** (TrackEval pools all seq+expr trajectory IDs before √(DetA·AssA)):

| statistic | recipe | ours | paper |
|-----------|--------|------|-------|
| 3-seq pooled (paper headline) | iKUN cascade+simcalib, YOLOv8-NS | **44.224** | 44.56 |
| └ AssA | | 62.482 | 62.48 (bit-exact) |
| 3-seq pooled + DDETR | cascade★ DDETR-NeuralSORT | unreachable | 48.84 |

**iKUN cascade+simcalib(a=8,b=−0.1,τ=100) reproduced at 44.224 vs paper 44.56**
(Δ−0.336, within seed/CLIP-version noise). The 48.84 SOTA row requires DeformableDETR
detections paired with NeuralSORT, which the paper authors refused to release (iKUN
issues #25/#32/#33/#35) and NeuralSORT code is unreleased — **path to 48.84 is closed;
honest pooled ceiling on public components is ~44.564.**

Two GT-label conventions exist: **`gt_template_old/` is paper-canonical**
(NeuralSORT-aligned); the local `gt_template/` regen is off-by-one and drops HOTA ~6.4.
All ship numbers use `gt_template_old/`.

### 4.2 The fusion recipe — scale-matched linear-additive, per axis

The fusion form is `fused = model_logit + b_simcalib + α·(gmc−0.5)·sc`, kept iff
`fused > thr`, applied **separately per axis** (motion expressions vs appearance/static).
The invariant is **bias_magnitude / score_magnitude ≈ 30-50%** — fixed-magnitude recipes
across architectures are wrong (this is why early iKUN attempts at FlexHook's sc=10 were
NEG: iKUN cosine ∈ [0,1] needs ~30× smaller bias).

**iKUN motion-axis (scale-matched):** α=1, sc=0.9, thr=+0.17 → pool **44.400 (+0.176
vs B1)**. 32-arm ridge fully mapped; plateau 44.388-44.400, not knife-edge.

**iKUN APPEAR-axis extension:** add α_a=1, sc_a=0.30, thr_a=+0.10 → pool **44.602
(+0.378 vs B1, +0.038 vs paper 44.564 — BEATS PAPER)**. 24-arm ridge, plateau width
0.015. 13D bbox-state slots (cx,cy,w,h) carry appearance signal even on "appearance"
expressions, so the second axis is an independent gain that stacks.

**FlexHook V1 (motion + APPEAR):** motion α=0.65 sc=10 thr=+3 → 53.607; + appear α_a=1
sc_a=3.5 thr_a=+0.9 → pool **53.696 (+0.586 vs B1; paper gap −0.128)**. 17-arm appear
sweep, 9-arm plateau width 0.012.

**FlexHook V2 (raw_sentence encoding fix):** encoding the V1-canonical `raw_sentence`
field instead of V2 paraphrase rescued a prior NEG (the V1-trained projector is OOD on
paraphrases). Ship motion α=0.4 sc=10 thr=+1.3 → 42.584; + appear α_a=1 sc_a=3.5
thr_a=+1.2 → pool **42.799 (+0.273 vs paper 42.526 — biggest paper-beat margin)**.

### 4.3 Statistical defense

- **Per-class POOL Δ (3-arch × 3-class = 9 cells):** all 9 POSITIVE and stat-sig at
  α=0.05 (7/9 at α=0.01). Smallest t=6.28 (V1 STATIC), biggest Δ=+4.562 (iKUN MOVING).
  Strongest possible pool defense.
- **Per-expr Wilcoxon disagrees** for V1/V2 (pool gain doesn't survive per-expr because
  77% of frames are appearance and pool aggregates trajectory IDs cross-expr before the
  √-product). iKUN survives per-expr too (MOVING Cliff δ=+0.222). The pool-vs-per-expr
  disagreement is **mathematical (HOTA aggregation), not artifact** — confirmed by
  GT_Dets-weighted Δ staying NEG and by STATIC recipe-split being NEG/neutral on both
  V1 and V2 (the per-expr STATIC "loss" is not a recipe miscalibration).

### 4.4 Multi-seed confirmation (n=3, seeds 0/1/2) — the prior mlp ship

Aligner retrained per seed; per-seed caches; per-seed ship eval.

| arch | recipe | pool HOTA n=3 | vs paper | sig |
|------|--------|---------------|----------|-----|
| iKUN | mlp + motion(1,0.9,+0.17) + appear(1,0.30,+0.10) + sigmoid+EMA | **44.608 ± 0.024** | +0.044 | t=3.14, p_one=0.044 |
| FH V1 | mlp + motion(0.65,10,+3) + appear(1,3.5,+0.9) | 53.716 ± 0.068 | −0.108 | p_less=0.056 (structural undershoot) |
| FH V2 | mlp + motion(0.4,10,+1.3) + appear(1,3.5,+1.2) | 42.799 ± 0.047 | +0.273 | t=10.13, p=0.005 |

**2/3 archs beat paper.** V1 undershoots paper 53.824 but beats local B1 by +0.606
(p=0.002) — the V1 paper gap is cli-fork baseline drift (local B1 53.110 vs paper 53.824),
not a GMC failure. Reference single-seed runs sit at seed-1 typical, not cherry-picked.
This was the shipped configuration **until the 2026-05 saga (Section 7) replaced the
aligner arch and dropped EMA.**

### 4.5 FiLM ego-injection (architectural, research result)

FiLM γ⊙x+β at pre-text-attention site A on cascade KUM: ep19 macro **+0.642** vs B;
the **rawvel ablation collapses (ΔΔ=+34.93 HOTA)** — ego-compensation in the 13D vector
is decisive. Site-A > site-B by +0.97 macro. But 3-seq **pooled** gain is only +0.053
(macro gain evaporates under frame imbalance), and the ceiling needs DDETR retrain.
Research result, not shipped.

---

## 5. SOTA-Reach Levers — DDETR Substitutes (all NEG)

The +4.28 HOTA from paper SOTA (48.84) needs DDETR+NeuralSORT. Every public substitute
failed the recall gate:

| Substitute | result | verdict |
|-----------|--------|---------|
| DDETR ep30 + vanilla SORT + cascade | 0011 = 36.44 (vs 48.84) | SORT bottleneck |
| DDETR + ByteTrack | 39.0-39.8 pooled | <40 gate |
| DDETR + BoT-SORT (NS-parity Re-ID+ECC) | 35.12 | <40 gate |
| FlexHook Temp-NeuralSORT-kitti1 tracker | 39.547 pooled (Δ−5.02) | detector recall bottleneck |
| DETR-NS detector for cascade KUM | B=32.4 (vs YOLOv8 39.4 macro) | DETR≠DETR |
| Grounding-DINO-Tiny + OC-SORT (Path A) | recall 0.50-0.75 (need 0.90) | open-vocab geometric drift |

**Detector-bound, not tracker-bound.** Path to 48.84 closed without the unreleased
DDETR+NeuralSORT pair.

---

## 6. "Never Kill at AUC" — HOTA Revisits of AUC-Killed Levers (2026-05-07 to 2026-05-18)

AUC and HOTA decoupled on aligner-internal levers (AUC = per-frame discrimination;
HOTA = trajectory continuity / AssA). Several AUC-killed levers were HOTA-revisited;
**iKUN repeatedly flipped POS** (its cascade+simcalib pruning absorbs higher-magnitude
score distributions as gain) while FlexHook stayed NEG.

| Lever | AUC verdict | HOTA revisit verdict |
|-------|-------------|----------------------|
| Depth-augmented 17D (DAv2 metric-Z + ego-comp ΔZ) | KILL 0.7567 | iKUN +0.215 (p=0.016, sig POS); FH V1 +0.048 / V2 +0.034 within seed noise. **iKUN-only ship candidate.** |
| World-XY projection (metric dX,dY via inverse pinhole) | — | FLAT vs 17D all 3 archs (iKUN Δ0.000, p>0.29). Aligner absorbs unit scale. NEUTRAL. |
| CLIP-visual 128D concat (exp39 HOTA) | KILL 0.7223 | NEG vs depth-aug iKUN −0.139; FH −0.096/−0.229; iKUN MOVING −2.72. |
| CLIP-text encoder swap (exp40) | KILL | iKUN +0.032 single-seed POS; FH V1/V2 NEG (score-scale mismatch). |
| CLIP-text late-concat (exp41) | KILL 0.731 | iKUN +0.203 single-seed (~8σ POS); FH V1 −0.566 (~8σ NEG). Arch-split. |
| Arm B raw-cosine (skip sigmoid+EMA) | — | iKUN −0.179, FH V1 −0.051, FH V2 multi-seed +0.060 (t=1.22 n.s.). EMA = per-track denoiser. KILL (later revisited in §7). |

**Rule established: AUC is a fast prefilter, HOTA is the ship gate. Every AUC-killed
aligner lever deserves a HOTA cross-check.** 3 of 4 such revisits flipped to HOTA-POS
on iKUN. The exp39/41 CLIP-fusion direction is iKUN-only — FlexHook uses a native RoI
visual backbone, so CLIP features are redundant + dim-mismatched.

---

## 7. The 2026-05 Ship Saga (shared_weight + EMA-drop + CLIP-fusion + simplification)

A two-week sequence of arch/pipeline changes, two withdrawn ship-swaps, and a final
reversal that landed on the current ship.

### 7.1 shared_weight aligner — two-baseline protocol (2026-05-19)

New `shared_weight` (sw) arch: per-modality Linear adapter (motion 13→256, lang
384→256) → shared 2-hidden MLP (256→512→512→256) → LN → L2. 628k params (≈ mlp 627k).
Symmetric two-tower, shared nonlinear core. Trained V1 stage1, InfoNCE+FNM, 100ep,
batch 256, lr 1e-3, seeds {0,1,2}.

Two-baseline B2 protocol (simple fusion α=1 sc=1 thr=0, raw cos), n=3 pool HOTA:

| arch | B1 | mlp B2 | sw B2 | Δ sw vs mlp |
|------|----|--------|-------|-------------|
| iKUN | 44.224 | 44.178±0.054 | **44.272±0.018** | +0.094 (Welch t≈2.9, p≈0.03) |
| FH V1 | 53.110 | 53.107±0.005 | **53.121±0.005** | +0.014 (t≈3.4, p≈0.02) |
| FH V2 | 42.526 | 42.533±0.001 | 42.532±0.002 | −0.001 (NEU) |

**sw beats mlp 2/3 archs** at stat-sig but trivial absolute gain; wins the simplicity
tiebreaker via symmetric inductive bias. Also confirmed: **simple-recipe GMC is
pool-neutral vs B1 on all 3 archs** — past ship gains came entirely from per-arch
coefficient tuning, not raw GMC signal. iKUN MOVING +4.08 survives pool-neutral due to
frame imbalance.

### 7.2 EMA-drop validation (2026-05-19)

Dropped all EMA (MotionBuffer + ScoreBuffer + cosine_buffer); pure raw-cosine path.

| arch | aligner | w/EMA pool | no-EMA pool | Δ |
|------|---------|-----------|-------------|---|
| iKUN | sw | 44.272±0.018 | **44.343±0.060** | +0.071 |
| iKUN | mlp | 44.178±0.054 | 44.278±0.029 | +0.100 |
| FH V1/V2 | sw/mlp | — | — | flat (±0.005) |

iKUN gains (its sc=0.9/0.30 lets the aligner drive per-frame ranking; EMA was smoothing
away discriminative spikes); FlexHook flat (sc=3.5 saturates the sigmoid; EMA was
deadweight). No-EMA exposes seed variance (std grows). Note this **inverts** the earlier
Exp-42 Arm-B finding for FlexHook — the difference is the sw aligner + scale-matched
recipe; the EMA-as-denoiser conclusion held for the mlp+FlexHook-scale regime.

> Both 7.1 and 7.2 were briefly adopted as framework defaults then **reverted in code**
> (commit `8225022`, mlp+EMA restored as defaults) so the ship recipe stays opt-in via
> `--architecture shared_weight` + `GMC_RAW_COS=1`. The measurements remain valid.

### 7.3 CLIP-fusion ship-swap — attempted then WITHDRAWN

Multi-seed n=3 CLIP-fusion at the **legacy mlp+EMA pipeline**:

| variant | arch | pool n=3 | Δ vs mlp ship | verdict |
|---------|------|----------|---------------|---------|
| exp39 early-concat | iKUN | 44.812±0.134 | +0.204 | POS (pool) |
| exp41 late-concat | iKUN | 44.801±0.094 | +0.193 | POS (pool, p≈0.04) |
| exp39/41 | FH V1 | 53.611 / 53.233 | −0.105 / −0.483 | NEG sig |
| exp39/41 | FH V2 | 42.628 / 42.683 | −0.171 / −0.116 | NEG sig |

iKUN exp41 was briefly swapped in as ship (44.801, +0.237 vs paper). **WITHDRAWN
2026-05-21** after per-class audit: the +0.193 pool gain is a **trajectory-pooling
artifact** (cross-class trajectory-ID consistency), while within-class HOTA REGRESSES —
STATIC −1.743, MOVING −1.034. Frame-weighted within-class Δ = −0.34. This reverses the
§4.3 per-class-pool defense, so the swap was rejected on all 3 archs.

### 7.4 Ship simplification ("sw + simple") — adopted then REVERSED same day

User asked for sw arch + "just simple": ship = B2 (uniform α=1 sc=1 thr=0, 0 free
hyperparams). iKUN 44.272 / FH V1 53.121 / FH V2 42.532. **Reversed within hours** when
user clarified "paper-beating matters" — sw+simple **loses the iKUN paper-beat**
(44.272 < paper 44.564, Δ−0.292). The per-arch recipe is needed to restore it. The
sw+simple configuration was demoted to the **B2 baseline anchor** (not the ship).

### 7.5 Final ship adoption (2026-05-21): sw + per-arch recipe + no-EMA

The configuration that satisfies both user priorities (sw arch + paper-beat) =
**sw aligner + per-arch recipe + no-EMA + raw cosine.** See Section 8.

### 7.6 Post-ship genericization levers (23rd, 24th — both NEG)

- **Variant B (auto-derive sc via std-matching), 23rd lever:** set sc = std(model_logit)/std(raw_cos)
  per arch per axis (eliminate 12 of 18 hyperparams). Catastrophic NEG: iKUN −2.794,
  FH V1 −5.831, FH V2 −4.436 (Σ −13.06). The hand-tuned APPEAR-axis sc is **7-11× SMALLER**
  than std-matching predicts — hand-tuning intentionally **damps GMC on appearance
  expressions** (motion signal is noise for "black cars"). Std-matching floods APPEAR
  with motion noise. **The 18-param recipe is irreducible.**
- **Phase-1 exp41 per-class routing gate, 24th lever:** re-eval exp41 at the **current
  sw+no-EMA raw-cos pipeline**: all 3 pool NEG (iKUN −0.408, V1 −0.353, V2 −0.140) AND
  all 12 per-class cells NEG. iKUN MOVING −5.45. exp41's marginal pool gain **required
  the EMA pipeline** — raw cos amplifies its CLIP-text encoder noise. No per-class
  asymmetry to route → Phase 2 skipped. **CLIP-fusion direction closed at the current
  pipeline.**

---

## 8. Lever Exhaustion Summary

~24+ levers tested at the post-reproduction ceiling, all NEG / neutral relative to the
ship except the depth-aug iKUN candidate (AUC-NEG → HOTA-POS, iKUN-only).

| # | Lever | Δ / verdict | Date |
|---|-------|-------------|------|
| — | HN-InfoNCE β-grid (Exp 34) | −0.026 to −0.046 AUC; representation-bound | 2026-04-19 |
| — | Feature enrichment F1-F9 (Exp 31) | best +0.009 AUC, <0.800 bar | 2026-04-16 |
| — | Temporal transformer (Exp 32) | −0.009 AUC | 2026-04-17 |
| — | 25D MLP accel (Exp 36A) | −0.032 / −0.038 AUC | 2026-04-20 |
| — | V1+V2 joint train (Exp 36C) | micro flat, macro +0.005 | 2026-04-20 |
| — | BGE-base 768D encoder (Exp 36D) | −0.044 AUC (worst encoder) | 2026-04-20 |
| — | Curriculum (Exp 36E) | −0.017 AUC | 2026-04-20 |
| — | Ego-source swap (Exp 37-A) | −0.036 AUC | 2026-04-22 |
| — | OMF 28D Farneback flow (Exp 37-B) | −0.155 AUC (worst lever) | 2026-04-24 |
| — | EMAP concat (Exp 37-C) | −0.024 AUC | 2026-04-22 |
| — | ORB-grid 3×8 61D (Exp 37) | −0.107 AUC | 2026-04-27 |
| — | TempRMOT β-grid (Exp 37-D) | −3.8 to −5.4 HOTA; tracker-class dichotomy | 2026-04-22 |
| — | Ego-injection 3 recipes (Exp 38) | best −0.496 HOTA | 2026-04-24 |
| — | CLIP early-concat 64D (Exp 39 AUC) | −0.057 AUC | 2026-05-05 |
| — | CLIP-visual 128D concat (Exp 39 HOTA) | iKUN −0.139, FH −0.096/−0.229 | 2026-05-10 |
| — | CLIP late-concat (Exp 41 AUC) | −0.048 AUC (HOTA iKUN flips POS) | 2026-05-06 |
| — | Strict motion-filter (138 pure) | micro −0.142, macro −0.132 | 2026-04-30 |
| — | World-XY projection | FLAT vs 17D all archs (p>0.29) | 2026-05-10 |
| — | CLIP-logit decision-level (Exp 43) | best −0.243 HOTA (4th CLIP site closed) | 2026-05-12 |
| — | Arm B raw-cosine (Exp 42, mlp+EMA regime) | iKUN −0.179, FH V2 +0.060 n.s. | 2026-05-07 |
| — | Tier-B per-class specialist aligners | iKUN −1.819, V1 −1.319, V2 −0.404 | 2026-05-07 |
| — | Learned-fusion MLP gate (FlexHook) | −3.79 pool, −35.6 MOVING | 2026-05-01 |
| — | Learned-residual MLP (iKUN) | −1.305 | 2026-05-01 |
| — | GMC seed-ensemble cache | sign-POS within ±std (reproducibility, not gain) | 2026-05-12 |
| — | Case 2 1a fusion-transformer (iKUN+motion KV) | −1.0 vs ship (AUC 0.90 PASS) | 2026-05-14 |
| — | Case 2 1b POS-decoupled branches | −1.17 vs ship | 2026-05-15 |
| — | Case 2 1c +ego-state 3rd KV | −1.19 vs ship (turning-verb wall lever dead) | 2026-05-15 |
| — | Case 2 1d FiLM-on-visual | −0.73 vs ship (strongest of 4); 1d-on-ship stack also NEG | 2026-05-15 |
| — | Lever A CDRMOT structural-consensus aux loss | λ=0.5 −0.210, λ=0.1 −0.152; manifold-collapse | 2026-05-16 |
| — | Lever B CDRMOT what/where dual-cosine (spaCy POS) | best −3.67 (stub inputs break cascade) | 2026-05-16 |
| — | Path A Grounding-DINO + OC-SORT (20th) | recall 0.50-0.75, gate FAIL | 2026-05-16 |
| — | Path C Qwen2-VL-2B LVLM rerank (21st) | degenerate output (capacity-bound) | 2026-05-17 |
| 22 | Arm B raw-cosine re-validation (sw regime) | superseded by §7.2 (no-EMA adopted) | 2026-05-19 |
| 23 | Variant B std-matching sc | iKUN −2.794, V1 −5.831, V2 −4.436 | 2026-05-21 |
| 24 | exp41 per-class routing gate (raw-cos pipeline) | all 12 per-class + 3 pool NEG | 2026-05-21 |

**Cross-cutting verdicts:**
- The 13D bbox-state slots carry appearance signal; do NOT short-circuit the appearance
  axis, and do NOT inject CLIP (redundant). Decision-level additive fusion is the only
  POS path; learned fusion (MLP gate or residual) is strictly worse.
- The 18-param ship recipe is irreducible: the per-class APPEAR-axis damping is the
  load-bearing gain, not arbitrary tweaks.
- iKUN (cascade+simcalib) absorbs higher-magnitude aligner distributions as gain;
  FlexHook (native RoI backbone) does not — every CLIP/encoder lever is arch-split.
- The turning-cars/turning-vehicles motion class on seq 0011 is universally
  unrecoverable across all levers (threshold FP 3.9:1, IoU-bridge NS recall miss,
  curvature spike all dead).

---

## 9. Current Ship (2026-05-21) — SUPERSEDED

> **SUPERSEDED 2026-08-19 by Option B** (§10 A22–A36): road-plane ego chain on all three host
> settings; iKUN two-α (0.7 / 0.1) 44.847 ± 0.107; FlexHook single α. This section documents the
> 13D / 18-hyperparameter recipe ship and is kept for provenance.

**Aligner:** `shared_weight` arch — per-modality Linear adapter (motion 13→256, lang
384→256) → shared 2-hidden MLP (256→512→512→256) → LN → L2-norm. 628k params. Trained
V1 stage1, InfoNCE+FNM (τ=0.07), 100ep, batch 256, lr 1e-3, seeds {0,1,2}.
Weights: `gmc_link_weights_v1train_sharedweight_seed{0,1,2}.pth`.

**GMC score:** raw cosine ∈ [−1,+1] (`GMC_RAW_COS=1`); **no sigmoid, no EMA.**

**Fusion (per arch, per axis):** `final = model_logit + α·(sc·raw_cos + thr)`, motion
axis for motion expressions, appear axis otherwise.

| arch | α_m | sc_m | thr_m | α_a | sc_a | thr_a |
|------|-----|------|-------|-----|------|-------|
| iKUN | 1.0 | 0.9 | +0.17 | 1.0 | 0.30 | +0.10 |
| FH V1 | 0.65 | 10 | +3 | 1.0 | 3.5 | +0.9 |
| FH V2 | 0.4 | 10 | +1.3 | 1.0 | 3.5 | +1.2 |

**Ship HOTA (3-seq pooled V1, n=3 mean ± sample std):**

| arch | B1 (no GMC) | B2 (sw simple) | **Ship (sw+recipe+no-EMA)** | vs paper |
|------|-------------|----------------|-----------------------------|----------|
| iKUN | 44.224 | 44.272±0.018 | **44.634 ± 0.066** | **+0.070** (paper 44.564), all 3 seeds beat; one-sided t vs paper p≈0.10 |
| FH V1 | 53.110 | 53.121±0.005 | **53.526 ± 0.087** | −0.298 (paper 53.824, structural cli-fork gap; +0.416 vs B1) |
| FH V2 | 42.526 | 42.532±0.002 | **42.807 ± 0.038** | **+0.281** (paper 42.526) |

**2/3 archs beat their paper claim, with bigger margins than the prior mlp+EMA ship**
(iKUN +0.070 vs +0.044; V2 +0.281 vs +0.273). The V1 paper gap is structural — paper
53.824 is not beaten in **any** tested configuration (the prior mlp ship also undershot
at 53.716); the local V1 −0.190 vs the mlp ship is an accepted trade for sw uniformity
+ the better iKUN/V2 paper margins.

**Honest claims to cite going forward (exact recipe + n + std):**
- "iKUN cascade+simcalib + GMC sw-aligner linear-additive (motion α=1.0 sc=0.9 thr=+0.17,
  appear α=1.0 sc=0.30 thr=+0.10, no-EMA, n=3) = 44.634 ± 0.066, +0.070 vs paper 44.564."
- "FH V2 + GMC sw-aligner (motion α=0.4 sc=10 thr=+1.3, appear α=1.0 sc=3.5 thr=+1.2,
  n=3) = 42.807 ± 0.038, +0.281 vs paper 42.526."
- New levers report Δ against these ship numbers (44.634 / 53.526 / 42.807), and
  "+X helps" comparisons anchor against B2 (44.272 / 53.121 / 42.532).

---

## 10. Full-Pipeline Audit (2026-08-13) — defects found, none previously documented

13-agent audit of every stage (12D single-α ship), all candidates adversarially
verified against `docs/CLOSED_LEVERS.md`. Ranked plan: `docs/IMPROVEMENT_PLAN_2026_08_13.md`.
Memory: `project_full_audit_2026_08_13`. Findings (status at audit time):

| # | Defect | Evidence | Status |
|---|--------|----------|--------|
| A1 | Ship stage-1 training has **NO False-Negative Masking** (docs claimed "InfoNCE + FNM"); default `AlignmentLoss.forward` ignores `sentence_ids`; hninfo+FNM hard-blocked for stage 1. ~30% of in-batch negatives are same-group false negatives at B=256 / 6 group labels (~117/255 for "moving" anchors) | `losses.py:34-60`, `train.py:458-462`, measured on `cache/training_data/120ae7403763dfc8.npz` | **MEASURED NEGATIVE 2026-08-14 — FNM-at-HOTA CLOSED** (`GMC_FNM=1` group-mask in `AlignmentLoss`, 3 seeds retrained, full protocol, `results/fnm{,_warm11}/`): pooled NEG on all archs at every α>0 — iKUN peak 44.284±0.015 @0.2 (−0.23 vs no-mask ship), collapses to 42.2@1.0; FH V1 53.092@1 declining; FH V2 42.529@1 declining. MOVING sharpens at low α (+1.0 @0.2) but STATIC/APPEAR crash harder — same-group in-batch negatives are load-bearing for appearance/static discrimination. Doc mismatch stays fixed; `GMC_FNM` guard retained for reproducibility of the falsification |
| A2 | **Inference-only MotionBuffer EMA (α=0.3)** on the 8 velocity dims; training features are raw. 2026-08-10 "EMA removal" was score-side only. Prior removal measurement POSITIVE for iKUN (pool +0.07..+0.10, MOVING +0.75..1.02; commit b00d232), reverted, never tested on 12D ship | `manager.py:372-375` vs `dataset.py` (no EMA); memory `project_noema_validation_2026_05_19` | **MEASURED NEUTRAL 2026-08-13** (`GMC_MOTION_EMA=0` caches, full-test n=3, `results/nomema/`): iKUN @0.5 44.514±0.082 (+0.002 vs no-mask ship — 13D-era positive does NOT reproduce under 12D single-α), FH V1 @5 53.198±0.026 (+0.041), FH V2 @5 42.663±0.041 (−0.021). All flat within ~1σ; warm11 dominates on all archs. Combined arm (nomema+warm11, `results/nomema_warm11/`, n=3): statistically identical to warm11-only — iKUN @0.5 44.615±0.073 (−0.019), FH V1 @5 53.246±0.008 (+0.011), FH V2 @5 42.658±0.030 (−0.012), V2 MOVING @2–3 slightly positive (48.025/48.030 vs 48.018). ⇒ MotionBuffer EMA deletes for FREE under the warm11 mask (EMA only mattered on the masked early frames). Simplification candidate: raw features + T≥11 mask, no MotionBuffer |
| A3 | **Warmup garbage fused at full α**: 25.1% of NeuralSORT test track-frames lack long-gap dims, 5.5% all-zero velocity (= "stationary"-coded), 2.7% fully-zero first-frame vector; on FH the GMC term also sits inside the detection gate | `manager.py:353-354,380-386`; measured over 4,950 track-frames | **MEASURED POSITIVE 2026-08-13** (`filter_warmup_cache.py` T≥11 mask, zero hyperparams, full-test sweeps n=3, `results/warm11/`): iKUN @α=0.5 pooled 44.634±0.095 (vs no-mask ship 44.512±0.104, **+0.122**; = old 18-param recipe number; mean beats paper-pure 44.564), MOVING 30.043±0.351 (−0.18 vs ship, within noise), APPEAR 46.517 (+0.21). FH V1 @α=5 53.235±0.004 (+0.078 vs ship@2; +0.125 vs native). FH V2 @α=5 42.670±0.040 (flat vs ship 42.684); slug-MOVING anomaly gone at α=1–3 (Δ≈0.00). α=0 ≡ natives exactly all archs. LOSO complete (`results/warm11/loso_*`): α* = 0.5 / 5 / 5 (fold argmaxes iKUN {0.2, 1.0-boundary, 0.5}; V1 {5,7,5}; V2 {5,3,5,7}) → full-test at α* = the numbers above; gate PASSED on iKUN/V1, V2 non-inferior (−0.014, 1σ). Ship adoption pending user/professor decision |
| A4 | **V2 per-class grouping label-space broken**: classifier runs on paraphrase slugs with an ad-hoc keyword list, cache scores canonical raw_sentence; 108/862 exprs disagree; V2 "MOVING" row = 38% canonical-APPEARANCE; 25 true-MOVING hidden in APPEARANCE. The V2 MOVING-negative anomaly is confounded by this AND localizes to seq 0019 (hold-0019 LOSO fold: MOVING rises with α) | `run_flexhook_v2_raw_sweep.py:33-49,143`; `results/loso_fh_v2_hold0019/` | **ADJUDICATED 2026-08-13** (`run_v2_canonical_regroup.py`, n=3, regenerated full-test predicts): canonical-MOVING (136 exprs) α5 38.096±0.035 vs α0 38.154, Δ=−0.058 — still ≤0 ⇒ anomaly is STRUCTURAL (host-deficit inverse law), not a grouping artifact. BUT slug "MOVING" 48.018 was inflated by the 66 direction paraphrases (their own row: 55.35, Δ+0.011 flat); true canonical MOVING baseline = 38.15. STATIC Δ+0.196, APPEAR Δ+0.169. Paper must report canonical grouping. `results/v2_canonical_regroup.json` |
| A5 | **LOSO clobber live on disk**: `hota_eval_flexhook_v2_raw_gmc_sw12d_seed*/alpha*/result.json` held 3-seq fold outputs (pooled 51.006) instead of full-test 42.684; fold runs and full-test share output paths | verified on disk 2026-08-13; memory `project_loso_outsuffix_clobber_landmine_2026_08_11` | FIXING — DO-NOW 1 |
| A6 | FH eval scripts WARN-and-continue on missing GMC cache → α>0 silently evaluates as native (typo'd `GMC_SUFFIX` = flat sweep labeled as fused); iKUN hard-crashes — inconsistent | `run_flexhook_phase5_gmc_sweep.py:177-183`, `run_flexhook_v2_raw_sweep.py:180-186` | FIXING — DO-NOW 1 |
| A7 | iKUN LOSO fragile: fold argmaxes {0.2, 1.0-censored-at-grid-boundary, 0.5}; fold-chosen α=1.0 is full-test −0.444; grid step 0.2 leaves peak unresolved | `results/loso_ikun_hold*/` | LATER 2 |
| A8 | Keyword router has 15 stems (iKUN) / 25 (V2), docs claimed "~38"; 14/126 V1 direction exprs (counter/same/horizon-direction) misroute to APPEARANCE | `run_ikun_linear_additive.py:44-45` | LATER 5 |
| A9 | 7/916 frame transitions emit wild homographies (max 5592px corner disp), poisoning ~4.3% of (frame,gap) ego slots via cumulative composition; identity fallback never fires (0/916) | instrumented over 0005/0011/0013 | **MEASURED 2026-08-14** — `GMC_HGATE=1` (reject \|h31\|,\|h32\|>1e-3 or corner-disp>150px → reuse last-good-H; thresholds from measured p99=134px, not HOTA-tuned). Instrumented: fires on 9/916 (0.98%), 0 fallbacks, no over-firing (`diag_hgate_instrument.py`); 2 of the 9 are seq-0013 frame-skip wide-baselines (A10 interaction). HOTA (stacked on nomema+warm11, iKUN n=3, `results/hgate/`): pooled 44.625±0.083 (+0.010, flat), MOVING 30.215±0.220 (+0.109, ~1σ). Free robustness rider — adoption optional; FH skipped per ego-iKUN-only rule |
| A10 | Cache builders recompute ego per-expression (num_exprs× redundant ORB+RANSAC) with cv2 process-global RNG → same frame pair can get different H across expressions | `run_build_gmc_cache.py:88-90` | LATER 7 |
| A11 | 193/862 V2 test expr JSONs lack `raw_sentence` → ship caches silently encode paraphrase text for ~24% of exprs (mostly appearance/spatial) | `run_build_gmc_cache_flexhook_v2_raw.py:158` | LATER 7 |

| A12 | **Keypoint detector A/B** (user-raised 2026-08-14; same pipeline, only detector swapped via `GMC_FEAT`, 916 transitions, real NS masks, `diag_feature_ab.py`): ORB wild=9 / bg_res p50 **0.898px** / **18.5ms**; AKAZE 3 / 0.986 / 57.8; BRISK 6 / 1.074 / 92.2; SIFT 4 / 0.969 / 87.7. ORB = best median accuracy AND speed (1500 kps → tight RANSAC beats "more accurate" descriptors with fewer points); AKAZE only wins the tail, which GMC_HGATE handles for free. Textbook ranking (SIFT most accurate) does not survive unequal keypoint budgets. Extended screen (same replay): orb1500+MAGSAC wild=6/res 0.935/14.7ms (admits a 15280px monster RANSAC rejects); orb3000+RANSAC 6/0.923/22.0; orb3000+MAGSAC 4/0.989/22.1; orb5000+RANSAC 4/0.968/34.0 — every robustness variant trades median accuracy for tail; 8 configs total, none Pareto-beats ship | measured 2026-08-14 | **CLOSED — keep ORB1500+RANSAC (+HGATE)**; design decision now evidence-backed across detector, keypoint budget, and robust-estimator axes |

| A13 | **Motion-model DOF screen** (user-raised 2026-08-14; `GMC_MODEL`, same 916-transition replay): homography 8DOF wild=8 / disp p99 130.7 / max 3361.7 / bg_res 0.896px; affine 6DOF wild=1 / 57.4 / 265.6 / 0.994; **similarity 4DOF wild=0 / 37.6 / 132.7 / 1.034** — all ~14.5ms. Lower-DOF models kill the wild-H tail BY CONSTRUCTION (no thresholds), cost +0.1-0.14px median model bias. Consecutive-frame KITTI ego ≈ similarity; homography's extra DOF mostly fits noise/degenerate configs (BoT-SORT GMC precedent). HOTA (iKUN, stacked on nomema+warm11, n=3, `results/model_{aff,sim}/`): affine @0.5 pooled 44.654±0.088 / MOV 30.103±0.139; **similarity @0.5 pooled 44.656±0.078 / MOV 30.045±0.091** — both sign-above the homography candidate ship (44.615±0.073) and above the H-gate arm (44.625), all within ~1σ (statistically tied at n=3). α=0 ≡ 44.224 exact both. Per the simplicity-adjudicates-ties rule: **similarity preferred** — replaces GMC_HGATE entirely (no thresholds), 4-DOF model, tail-free by construction. Before ship adoption: FH V1/V2 rebuild+confirm (model change touches their caches too) + all-arch LOSO | measured 2026-08-14 | **iKUN PASS; FH confirm + LOSO pending ship decision** |

| A14 | **Composition-vs-direct referee + parallax wall** (user-raised 2026-08-15). Loop-consistency: composed H(t-10→t) vs direct disagree p50=37px (direct never fails, 0/176). Referee = GT-static-track residual (parked cars, true disp = pure ego): **composed WINS every gap/seq** — 0005 gap2/5/10: 13.7/30.6/57.3px vs direct 14.3/34.1/80.5; 0011: 2.8/5.5/8.4 vs 3.1/7.3/12.3. (a) Accumulation hypothesis FALSIFIED — composition is the better side; direct wide-baseline dies to parallax; temporal levers (Kalman-on-H, direct-per-gap, keyframe) dead. (b) Training's direct-estimate ego is WORSE than inference's composed — asymmetry exists but inference is on the good side. (c) **Real wall = depth parallax**: object-level ego error 3-57px vs estimator inlier residual 0.9px — two orders of magnitude; no global 2D transform of any DOF can fix it; consistent with oxts-perfect-ego only +0.285, monocular-depth ladder NEG, info-cap ladder NEG. Ego stage CLOSED: final form = ORB1500 + similarity 4DOF + composition + warmup abstention. Parallax table = rebuttal ammo for "why not a stronger ego estimator" | measured 2026-08-15 | **EGO STAGE CLOSED** |

| A15 | **D2 — train-time composed ego: NEG, closed** (`GMC_TRAIN_COMPOSED_EGO=1` + `GMC_MODEL=similarity`, 3 seeds retrained, iKUN full protocol, `results/cego/`): peak pooled 44.453±0.045 @0.3 vs sim-arm baseline 44.656±0.078 @0.5 (−0.20); MOVING @0.5 29.556±0.464 vs 30.045 (−0.49); every α below baseline; α=0 ≡ 44.224 ✓. Mechanism: training's dirtier direct ego = free noise augmentation teaching parallax invariance (same lesson as strict-filter NEG); cleaning it overfits the aligner to a clean-ego world that inference (parallax 3-57px) never provides. TD-2 train/eval-ego parity family closed in BOTH directions (A14: inference→direct dead; A15: train→composed dead) — the asymmetry is load-bearing, leave it | measured 2026-08-15 | **CLOSED — ego stage now closed from every direction**; cego weights/caches deletable |

| A16 | **T-probe representation diagnosis** (pre-registered `docs/TPROBE_PLAN.md`, 13,901 GT-track samples, 15 train seqs, seq-held-out, `results/tprobe/`): (1) temporal info EXISTS for moving/counter/direction (flat-T16 − T1 = +0.105/+0.107/+0.10 macro-F1) BUT ship-12D's built-in multiscale already banks 75-85% of it (ship12-T1: 0.845/0.770/0.932 vs T16 0.873/0.794/0.940) — remaining headroom +0.02-0.03; (2) **shuffle ≡ ordered on EVERY task (Δ=0.000)** — temporal ORDERING carries zero extractable information even on clean GT trajectories ⇒ **GRU/Transformer/Mamba closed at the information level** (upgrades the 2026-06-11 convergent kill); (3) **turning ≈ chance (0.55) at every T** — turning semantics absent from 2D image-plane kinematics entirely; needs a new information source (world-frame heading via ground-plane is the natural candidate, probe pending ground-arm verdict); (4) future-ADE saturates at T=4-8 (43.8→25.7px), consistent with gap {2,5,10} design; (5) kinematic labels parallax-noisy as predicted — expression labels primary | measured 2026-08-15 | temporal-encoder architecture bet CLOSED; 12D validated near-information-complete for 2D kinematics; open lead = new information (ground-plane/world-frame), not new encoders |

| A17 | **T-probe round 2 — coordinate-frame arms** (`diagnostics/tprobe2.py`, 17,943 samples, flatten T=16, `results/tprobe/tprobe2_results.json`): raw(no-ego)/gmc/ground/relational = moving 0.858/0.856/0.814/0.839; turning 0.576/0.541/0.559/**0.624**; counter 0.786/**0.808**/0.776/0.769. Verdicts: (1) raw ≈ gmc at probe level — with 16 frames of GT trajectory a linear probe implicitly separates camera motion (no conflict with the frame-level HOTA ablation, different regime); (2) **ground-plane arm does NOT unlock turning** (0.559) and slightly hurts moving/counter — road-H image residual ≠ world heading; (3) surprise: relational arm best on TURNING (+0.08) not counter — neighbor-median subtraction removes common-mode ego error; (4) ego-rotation stratification INVALID — image-plane similarity rotation ≈ 0.002-0.1° on KITTI (ego yaw manifests as image translation, not in-plane rotation); redo with oxts yaw if pursued. Round 3 (oracle 3D: KITTI label_02 rotation_y + oxts ego yaw, tid mapping via IoU vote) = decisive test running | measured 2026-08-15 | round-3 oracle pending |

| A18 | **T-probe round 3 — ORACLE 3D/world frame** (`diagnostics/tprobe3_oracle.py`, KITTI label_02 3D + oxts ego yaw, center-distance tid mapping ≥5-frame vote, 19,068 samples / 493 tracks, `results/tprobe/tprobe3_oracle_results.json`): turning **0.718** (vs image-plane best 0.624/0.541 → world-frame heading carries +0.09-0.18 real turning information — Hypothesis B partially CONFIRMED); moving 0.782 < image 0.858 (image box-size/position cues absent from 3D feature set); counter 0.622 ≪ image 0.808 (counter is SCENE-RELATIVE — absolute world heading cannot read it; image lane-position already encodes it). Even GT 3D caps turning at 0.72: expression labels are TRACK-level (whole trajectory labeled "turning" incl. straight segments) — annotation granularity is the residual ceiling. Deployment would need per-object monocular 3D heading (heavy, off plug-in story) for ~+0.1-0.3 HOTA EV on a small expression subset → NOT for this paper; the probe suite itself is next-paper analysis material | measured 2026-08-15 | coordinate-frame diagnosis COMPLETE: turning=world-frame info + label granularity; counter=scene-relative (image OK); moving=image sufficient |

| A19 | **T-probe round 4 — stratification + component ablation OVERTURN the world-frame story** (`diagnostics/tprobe4_strat_components.py`, `results/tprobe/tprobe4_results.json`): (A) turning × ego-YAW terciles (oxts, correct variable): stable 0.48-0.54 / rotating 0.65-0.68 on ALL image arms — **OPPOSITE of the ego-pollution prediction**; ego turns correlate with object turns (intersections; rotating stratum has 2× positives) — probe reads scene context, not motion; with a stable camera turning is still chance ⇒ pollution mechanism FALSIFIED. (B) oracle component ablation: position_XZ alone 0.667 (location/intersection prior!), **Δheading — the physically real turning signal — only 0.555 ≈ image-plane arms**; world heading 0.462; full 0.718 rides on the location shortcut. counter: position/velocity (lane side) 0.74 ≈ everything, image already encodes it. VERDICT: the oracle world-frame turning gain is mostly a SCENE-LOCATION prior, not motion information; **turning is bounded by annotation granularity + label-location confound, unreachable by ANY representation/coordinate change** — label-side (segment-level annotation) or host-side problem → next-paper territory. Representation layer fully exonerated | measured 2026-08-15 | coordinate-frame campaign CLOSED end-to-end (all 8 phases of the external plan adjudicated) |

| A20 | **Turning label dataset-bias decomposition** (`results/tprobe/tprobe5_locbias.json`; caveat: within-seq split is sample-hash fallback, track-frame leakage inflates XZ/full — seqid row unaffected): seq-ID ALONE = 0.673 within-seq (per-sequence base-rate bias; 0.483≈chance seq-held-out as expected); XZ+seqid 0.751 ≥ oracle full 0.741 — the oracle world-frame gain decomposes ENTIRELY into which-sequence + where-in-scene; **Δheading (true turning dynamics) = 0.555/0.572 under both splits — the honest motion-only turning ceiling, ≈ image plane**. Broader implication: ANY method claiming to learn turning/motion semantics on Refer-KITTI may be reading location priors — dataset-paper-grade finding, next-paper ammo | measured 2026-08-15 | Turning branch CLOSED with cause-of-death certificate: track-level labels × geography bias; motion dynamics signal ≈ absent in any frame |

| A21 | **A14 CORRECTION (user-challenged): "static" expression tracks verified against KITTI 3D+oxts world speed** — 0005's 6 static-keyword tracks move at median **1.22 m/frame (~44 km/h)**, 52.3% frames >0.3 m/f ("parking" = action in progress; track-level label noise, echoes A20); 0011's are genuinely static (0.037 m/f, 8.9% contaminated). Verified-static referee (world speed <0.1 m/f per window): **composed still beats direct** (0011 n=676/526: 2.73/8.65 vs 3.11/12.21 px at gap 2/10) — ranking conclusion SAFE; **parallax magnitude corrected to ~2.7px@gap2 → 8.7px@gap10** (3-10× estimator noise; wall stands, moderated) — 0005's dramatic 13.7-57.3px were object-motion artifacts (verified-static n collapses to 4-12). **Ground-plane screen's "0005 residual halving" INVALIDATED** (measured on moving cars); 0011 road-vs-sim was marginal ⇒ ground arm HOTA expected flat, close on arrival. Methodology rule added: expression keywords are NEVER motion ground truth — verify with 3D GT | measured 2026-08-15 | A14 magnitudes corrected; composed-wins intact; ground-arm prior downgraded |

| A22 | **GT-feature oracle — motion module at practical ceiling** (`build_oracle_motion_cache.py`, GT trajectories → ship sw12d aligner → standard fusion, NS↔GT center-distance mapping, warm11-style T≥11 abstention, n=3, `results/gtoracle/`): α=0 ≡ 44.224 ✓; **pooled @0.5 = 44.549±0.052 vs candidate ship 44.656±0.078 — oracle ≈ ship (−0.107, ~1σ)**; MOVING @1.0 32.264 vs 31.14 (+1.1, GT features do carry more motion signal but APPEAR pays, same Pareto). Verdict per pre-registered rule: **feature-level tracker noise is NOT the residual bottleneck — the motion line is CLOSED at its practical ceiling**; remaining +6.13 label-oracle gap lives in aligner/fusion/host-veto, unreachable per prior closures. Caveat: oracle covers only GT-matchable NS tracks (34-53/seq, rest fall back native) — masks effects <~0.1 | measured 2026-08-15 | **MOTION LINE CLOSED** — answers "current vs GT motion at HOTA" |

| A22b | **A22 scope caveat (user-challenged, 2026-08-15)**: A22 is an INFERENCE-SUBSTITUTION oracle (GT trajectories → dirty-trained aligner). It bounds feature QUALITY/noise, not feature DEFINITION (A23 ground arm MOVING@1.0 32.684 > oracle 32.264 proves the definition axis is outside its scope). Domain-shift objection ("aligner overfit to noise") is weakened by training already using GT centroid tracks (`dataset.py` labels_with_ids), but the fully matched cell is untested → matched-retrain oracle (cego weights × GT-oracle caches) pre-registered (`docs/PREREG_CEGO_ORACLE_2026_08_15.md`), queued. Status: closed-pending-seal | registered 2026-08-15 | **SEALED by A26** |

| A23 | **Ground-plane arm HOTA — pre-registered "expected flat" verdict OVERTURNED on MOVING** (train `GMC_GROUND=1` 3 seeds + warm11, suffix `_sw12d_ground_seed{s}_warm11`, iKUN full protocol n=3, `results/ground/`; pre-reg baseline = sim arm 44.656±0.078 / MOV 30.045±0.091 @0.5): α=0 ≡ 44.224 ✓. Pooled @0.5 **44.591±0.043** (−0.065, non-inferior, within 1σ — pooled gate PASSES). **MOVING @0.5 = 30.710±0.305 = +0.665, Welch t≈3.6 — SIGNIFICANT**, and MOVING @1.0 = 32.684±0.770 = campaign-best (vs sim 31.144, oracle 32.264). STATIC @0.5 44.040±0.168 pays −0.41 vs sim 44.448±0.087; APPEAR flat (46.476 vs 46.572). NOTE: the A21-invalidated screening story does NOT explain this — the MECHANISM IS UNPROVEN; gain could be the ground-contact warp POINT (bottom-center vs centroid), the road-H transform, or train-domain interaction. Status: **LIVE CANDIDATE RIDER — NOT adopted.** User signed off attribution as top priority: 2×2 split (G1 point / G2 road) pre-registered (`docs/PREREG_GROUND_ATTRIBUTION_2026_08_15.md`), `GMC_GROUND_MODE` env split landed, queue running. Owed after: ground-arm LOSO α*, user+professor ship decision | measured 2026-08-15 | **pre-reg "≥2σ positive" branch taken — flagged to user; attribution in flight** |

| A24 | **iKUN LOSO grid repair — α\*=0.5 CONFIRMED on dense un-censored grid** (pre-reg `docs/PREREG_LOSO_GRID_REPAIR_2026_08_15.md`; sim-arm caches, UNION 16-α grid with 0.05 mid-steps + {1.25,1.5,2.0} extension, 3 folds + full-test, n=3, `results/model_sim/loso_ikun_hold*` + 16-point `results/model_sim/alpha_sweep_ikun.*`): fold argmaxes {0.2, 1.5, 0.5} — extension UN-CENSORED hold0011 (was 1.0-boundary, A7); zero censored folds; median α\* = 0.5 = current pick, Δ=0.000. Full-test @0.5 = 44.656±0.078 / MOV 30.045±0.091 unchanged; dense peak 0.55 @44.663 (+0.007, noise). α=0 ≡ 44.224 ✓. Doubles as the sim arm's OWN LOSO (was inherited from nomema_warm11, A13 gap). A7 grid-fragility attack CLOSED; fold heterogeneity real ({0.2,1.5,0.5}) but median-robust — report in paper as robustness note | measured 2026-08-15 | **α\*=0.5 sealed; protocol hardening complete, no number changes** |

| A25 | **Ground-arm attribution 2×2 — mechanism = ROAD-PLANE CHAIN, not the ground-contact point** (pre-reg `docs/PREREG_GROUND_ATTRIBUTION_2026_08_15.md`, `GMC_GROUND_MODE` split, end-to-end per arm, n=3, `results/ground_{point,road}/`; baselines sim 44.656±0.078 / MOV 30.045±0.091 @0.5, full-arm MOV Δ=+0.665): α=0 ≡ 44.224 ✓ both. **G1 point: MOV Δ=+0.180, t=0.55, 27% recovery — ≈0, not significant.** **G2 road: MOV Δ=+1.458, t=3.88, 219% recovery — SIGNIFICANT and EXCEEDS the full arm**; pooled 44.721±0.127 (+0.065 vs sim, t=0.76, nominal BEST of campaign, non-inferior-plus); STATIC 44.522 (no cost, unlike full arm's −0.41); MOV@1.0 = 33.130±0.977 = new campaign best (vs full 32.684, t=0.62 ns). Pre-reg rule fires cleanly: attribute to ROAD; bottom-point factor ≈ noise alone and mildly sub-additive combined (full < road on MOV and STATIC). Sharpening contrast: cego (composed-GLOBAL train, A15) was −0.20 ⇒ it is the road-plane transform per se, not chain composition. Ground-full rider SUPERSEDED by ground-road as live candidate. Owed before ship talk: ground-road LOSO α*, n=5 confirm optional, user+professor sign-off | measured 2026-08-15 | **ROAD arm = live candidate; point factor closed** |

| A26 | **Matched-retrain oracle — A22 SEALED** (pre-reg `docs/PREREG_CEGO_ORACLE_2026_08_15.md`; cego weights (GT tracks + composed-similarity ego train = oracle-domain-matched) × GT-trajectory caches, n=3, `results/cego_gtoracle/`): α=0 ≡ 44.224 ✓; peak pooled 44.591±0.102 @0.5 ≤ seal threshold 44.81 (ship+2σ) → **domain-shift objection FALSIFIED**; matched oracle ≈ substitution oracle (44.549) ≈ ship (44.656); MOV@1.0 32.590±1.371 ≈ A22's 32.264. Feature-QUALITY ceiling stands on both training domains; A22b closed-pending-seal → **CLOSED-SEALED**. The definition axis (A25 road) remains the only live motion lever | measured 2026-08-15 | **A22 sealed; challenge #1 fully adjudicated** |

| A27 | **Ground-road LOSO + FH confirmation — road gain is real but NOT LOSO-reachable** (pre-reg `docs/PREREG_GROAD_LOSO_FH_2026_08_16.md`, n=3, α=0 ≡ natives exact all archs): **iKUN LOSO** (16-α dense): fold argmaxes {0.2, **2.0-CENSORED**, 0.5} — hold0011 pushes past even the extended grid (road keeps feeding the worst seq); 1 censored <2 ⇒ proceed, α\* = even-median{0.2,0.5} = **0.35** (on-grid). Full-test @0.35 = 44.616±0.098 / MOV 29.429±0.289 vs sim ship at ITS LOSO α\*=0.5 (44.656±0.078 / 30.045±0.091): pooled −0.040 (t=0.55, tie), **MOVING −0.62 (t=3.5) — LOSO-honest road LOSES MOVING**. The A25 headline numbers (44.721 / MOV 31.503 @0.5, MOV 33.130 @1.0) are real measurements but sit at α the road arm's own folds cannot select — per-seq α heterogeneity (0.2 vs >2.0) too extreme for single-α selection; exactly the tuned-on-test trap LOSO exists to catch (A7 precedent). **FH confirmation**: V1 best @7 53.280±0.045 (+0.034 vs 53.246, t=1.29) NON-INFERIOR nominally-positive; V2 best @5 42.625±0.032 (−0.033 vs 42.658, t=−1.30) NON-INFERIOR — road chain is FH-SAFE. Ship-decision menu (user+professor): (a) keep sim candidate ship — the LOSO-honest winner; (b) road ship requires abandoning LOSO-honest α (not defensible); (c) road as paper mechanism/analysis section: parallax-bypass demonstrated, MOVING headroom +1.5-3.1 shown at fixed α, selection gap named as the honest limitation | measured 2026-08-16 | **road = mechanism-real / selection-blocked; sim ship stays LOSO-honest champion; decision pack complete** |

| A28 | **Road-arm per-seq decomposition — gain is DISTRIBUTED, scene-overfitting objection falsified** (pre-reg `docs/PREREG_ROAD_DIAG_2026_08_16.md`, per-seq TrackEval on existing predicts, n=3, `results/road_diag/perseq.json`): MOVING Δ(road−sim) @0.5 = 0005 **+0.95** / 0011 **+2.23** / 0013 −0.38±1.68(noise, expr-thin seq); @1.0 = **+1.72 / +2.54 / +1.76 — all three positive**. Pre-reg rule: ≥2 seqs positive ⇒ DISTRIBUTED ⇒ road narrative holds (not a flat-scene trick on this set). Bonus: largest gain on 0011 = the systematically-worst seq — host-deficit inverse law reappears at sequence level. Pooled Δ: 0005/0011 positive, 0013 pays (−0.64/−0.86) — 0013's tiny MOVING base (HOTA 8-16, 2-expr thinness memory) makes it the noise floor. α-heterogeneity note (probe 3): fold-argmax spread pre-exists on sim arm ({0.2,1.5,0.5}), road amplifies ({0.2,2.0c,0.5}); 0011 prefers large α on BOTH arms. **Probe 2 (cache-level separation) INVALID by construction** — NS-tid vs GT-tid id-space mismatch (Case 2 landmine repeated); road-vs-sim signal-level WHY remains open, needs IoU-joined redo if pursued. Scope note stands: claim limited to Refer-KITTI's 3 test seqs; terrain-stratified validation impossible here (all urban KITTI) | measured 2026-08-16 | **challenge-#1 core falsified (DISTRIBUTED); mechanism-WHY still open (probe 2 invalid)** |

| A29 | **Probe-2 REDONE with IoU join — road's mechanism located** (Track B2; NS tid ↔ GT box greedy IoU≥0.5 per (MOVING-expr, frame); per-frame conditional separation = mean(cos pos) − mean(cos neg); `results/road_diag/separation_iou.json`): separations all POSITIVE post-fix (prior id-space run was garbage, confirmed). sim/road per seq: 0005 0.521/0.521 (unchanged, already strong); **0011 0.022 → 0.135 (6×)** — road repairs score discrimination precisely where the sim-chain signal is near-zero; 0013 0.208 → 0.178 (−0.03). Converges with A28 per-seq HOTA (0011 MOVING +2.23 = largest). **A25's WHY: road chain restores moving-expression score separation on the worst-host sequence**; parallax-bypass narrative now has signal-level support on 0011, stated as seq-scoped. Caveat: GT match rate 0.70-0.80 (tracker recall bound, not join failure — probe evaluates the NS-tracked population GMC actually scores) | measured 2026-08-16 | mechanism-WHY filled (seq-scoped); A25 wording can cite this |

| A30 | **V2 canonical regroup — candidate ship (Track B1)** (`run_v2_canonical_regroup.py --suffix _nomema_warm11`, α∈{0,5}, n=3, `results/v2_canonical_regroup_candidate_ship.json`): canonical rows @α=5 vs α=0 — MOVING (136) **+0.048**±0.060, STATIC (93) **+0.294**±0.037, APPEARANCE (633) **+0.125**±0.030, DIRECTION (66) +0.021±0.120. **All four canonical rows non-negative**; the A4-era canonical-MOVING negative (−0.058, old ship) flips to +0.048 under nomema+warm11 — the V2 MOVING anomaly is RESOLVED at the candidate ship (within ~1σ, but sign no longer adverse). Paper V2 table uses these rows | measured 2026-08-16 | V2 per-class table ready; anomaly resolved |

| A31 | **FH V1 official-seqmap re-evaluation — "structural reproduction gap" RESOLVED as eval-list mismatch** (Track A; pre-reg `docs/PREREG_FHV1_OFFICIAL_SEQMAP_2026_08_16.md`; TrackEval-only rescore of existing predicts on FlexHook's official 150-expr seqmap; `results/fh_v1_official/`): **α=0 = 53.824 EXACT** (= FlexHook's own recorded reproduction; paper's 53.83 flagged as typo in their infer.sh). Root cause: our seqmap enumerated 158 exprs (150 official + 8 pathological: braking×2/horizon×2/back-to-camera×4, degenerate GT) — predicts were byte-identical all along. Candidate ship (nomema_warm11) on official protocol: **54.011±0.025 @α\*=7** (LOSO folds {7,7,7} unanimous, uncensored) = **+0.187 vs native AND +0.19 above the host's published score**; MOV 44.802±0.207 (+0.49), STA 49.435 (+0.45), APP 56.562 (+0.09) — all classes positive. groad arm full-test ≈ ship arm (53.98@5-7, A27 non-inferiority re-confirmed on official list). Old "structural cli-fork gap" conclusions (COMPARISON.md, §7) SUPERSEDED. Clobbered FlexHook native summary restored (53.824 regenerated); guard: our evals never write into ~/FlexHook. Paper: FH V1 row becomes official-protocol-comparable, reproduction-gap disclaimer deleted, α\* = 7 (was 5 on the 158-list) | measured 2026-08-16 | **FH V1 line: native ≡ published, +GMC beats published; gap narrative dead** |

| A32 | **Two-α × road — FIRST LOSO-HONEST POOLED WIN of the campaign** (Track C1, pre-reg `docs/PREREG_TWO_ALPHA_ROAD_2026_08_16.md`, user-authorized; keyword-routed α_mot [MOVING+STATIC] / α_app [APPEARANCE], road caches, 6×5 grid LOSO, n=3, `results/two_alpha_road/`): fold argmaxes {(0.7,0.1), (2.0,0.7)-censored-both-axes, (0.7,0.2)} → (α_mot\*, α_app\*) = **(0.7, 0.1)**. Full-test: **pooled 44.847±0.107 (+0.191 vs candidate ship 44.656±0.078, t=2.50 — EXCEEDS the pre-registered upgrade threshold 44.812)**; **MOVING 32.606±0.654 (+2.56, t=6.7)**; STATIC 44.584 (+0.14); APPEAR 46.468 (−0.10, t=1.3 ns). **Sim control (3×3 grid): star (0.5,0.2) → 44.672±0.028 (+0.016, t=0.34, FLAT)** — two-α alone does nothing; the unlock is the road×two-α combination (road supplies the headroom [A25/A28/A29], class routing supplies the LOSO-reachable harvest [absorbs A27's per-seq α tension]). Integrity: diagonals am=aa=0.5 reproduce single-α bit-exact both arms; α=0 ≡ 44.224. Old "per-axis refit 44.316" precedent SUPERSEDED for this cell (different form: 2-param keyword routing on road caches, not 4-param per-axis on sim). Cost acknowledged: 2 hyperparams vs 1 (reverses part of the 2026-08-10 simplification) — **adoption = user+professor decision; ship menu now has option (b′): road + two-α = LOSO-honest champion 44.847 / MOV 32.6** | measured 2026-08-16 | **pooled ceiling honestly moved for the first time; awaiting sign-off** |

| A33 | **TTC (C²RMOT-style confidence-gated temporal calibration) — NEGATIVE, folds select "off"** (Track C2, pre-reg `docs/PREREG_TTC_CALIBRATION_2026_08_16.md`; memory+positive-residual core, θ×w LOSO on sim caches, `results/ttc_sim/`): integrity w=0 ≡ ship exact ✓. Fold argmaxes {(0.6,0.5), (0.8,0.5), (0.8,0.5)} — θ=0.8 = gate-never-fires = **pooled ≡ ship bit-exact in all folds**; 2/3 folds censor at this "off" boundary ⇒ θ axis unresolved per pre-reg, no selection. Active settings: only hold0005 ever benefits (+0.09 @θ=0.6) while hold0011 pays −0.62 at the same cell; strong calibration (θ=0.5, w=1.0) catastrophic (−2.2..−2.9). Extension pointless (θ→1 converges to ship monotonically). Mechanism: our two-stage fused scores are already temporally stable; positive-residual memory locks in early highs → FP inflation on diverging tracks. C²RMOT's regime (query-based hosts, frame-wise logit fluctuation, embedding winner-gate) does not transfer. Scope caveat: embedding competition gate untested (no per-instance embeddings in caches) — closure is for the memory+residual core on two-stage per-frame hosts | measured 2026-08-16 | **lever CLOSED (scoped); test-time calibration family dead on this architecture** |

| A34 | **n=5 ablations for BOTH ship options + FPS** (Track D hedge, pre-reg `docs/PREREG_ABLATION_N5_2026_08_16.md`, fixed LOSO operating points, 0 FAILs, `results/ablation_n5_hedge.json` + `results/fps_profile.json`): full arms at n=5 confirm n=3 numbers (A 44.649±0.087 / MOV 30.102±0.132; B 44.803±0.103 / MOV 32.295±0.632; B−A pooled t≈2.6 holds). Ablations ALL significant, no sign flips vs 08-11 priors: **A: −ego pooled −0.398 (t=5.8) / MOV −1.89 (t=15.9); −multiscale −0.207 (t=3.6) / −0.77 (t=2.9, prior "marginal" now significant). B: −ego −0.637 (t=11.7) / MOV −3.81 (t=11.8); −multiscale −0.318 (t=6.7) / −1.85 (t=6.2)** — option B's larger drops show its gain rides on the ego/multiscale machinery (design claims strengthen under B). FPS (CPU, seq 0011, 200f, machine under load): sim 48.0 / road 35.3 process-only — re-measure clean before paper. Table 2 material complete for whichever option is chosen | measured 2026-08-17 | **Track D experiments complete; awaiting ship decision only** |
| A35 | **FH two-α × road LOSO — BOTH HOSTS UNRESOLVED, stay single-α** (pre-reg `docs/PREREG_TWO_ALPHA_FH_2026_08_18.md`; canonical-text routing [V1: sentence, V2: raw_sentence, V1 lists iKUN order = A30; locked counts V1 25/12/113 of official-150, V2 136/93/633 of 862]; road caches `_sw12d_groad_seed{0,1,2}_warm11`; V1 grid 5×5 am{3,5,7,10,15}×aa{0.5,1,2,3,5} under official seqmap [generation-time filter, `_off150` trees], V2 grid am{2,3,5,7,10}×aa{0.5,1,2,3,5}, 4 folds; `results/two_alpha_road_fh_{v1,v2}/`): integrity gates ALL passed — α=0 ≡ published natives EXACT (V1 official 53.824, V2 42.526; validates generation-time filter ≡ A31 rescore), diagonals bit-exact (V1 am=aa=7 ≡ alpha7 53.994 groad-off150; V2 am=aa=5 ≡ alpha5 42.633 groad). Fold argmaxes V1 {(10,5),(3,3),(15,5)}, V2 {(5,5),(2,3),(5,5),(10,5)} — **α_app censored at grid max in 2/3 (V1) and 3/4 (V2) folds ⇒ aa axis UNRESOLVED per pre-reg, no selection, no full-test**. Folds push α_app toward the diagonal = routing buys nothing on FlexHook; road×two-α combination is iKUN-specific, consistent with the inverse law (A32's unlock rode iKUN's host motion deficit; FH hosts have native temporal modeling). Sim control arm not run (pre-reg: only on threshold pass). **Frozen fallback wording applies: paper FlexHook rows stay single-α (V1 54.011±0.025 @7 official, V2 42.658±0.030 @5); method section states per-host LOSO selects α_mot=α_app on FlexHook, degenerating to a single α**. Closure-verification probe (2026-08-18, diagnostic not selection): aa∈{7,10} extension cells at the 3 steepest-edge folds ALL turn over past aa=5 (V1 hold0005 am=10: 52.024→52.008→51.922; hold0013 am=15: 56.114→56.083→55.971; V2 hold0019 am=10: 50.964→50.955→50.880) — grid edge sat ON the aa optimum; censoring = boundary peak, not truncation; in-fold routing margins (+0.00–0.10) are the measured ceiling | measured 2026-08-17/18 | **FH two-α closed (measured, not extrapolated); Option B = iKUN two-α + FH single-α, honest & method-consistent** |
| A36 | **FPS clean re-measure** (pre-reg in `docs/PREREG_TWO_ALPHA_FH_2026_08_18.md` §A36; seq 0011, n=500, GMC_MODEL=similarity, GMC_MOTION_EMA=0; 8 reps before sweeps launched, reps 1–3 cold-cache excluded, median of warm 5; `results/fps_profile.json`): **sim 42.8 / road 31.8 FPS process-only** (incl-io 31.1 / 24.5). Residual load: gnome-shell ~0.6 core (desktop session, irreducible without killing user session); warm dispersion ~6% vs 5% gate — documented, medians stable. Replaces under-load A34 values (48.0/35.3, 200f) and the stale paper 68 FPS (13D pipeline). More conservative than prior claims → safe direction | measured 2026-08-17 | **paper FPS numbers final: sim 42.8 / road 31.8 (CPU, process-only)** |

| A37 | **FlexHook road-chain single-α LOSO — ship unified on the road chain, and the motion class IMPROVES** (pre-reg `docs/PREREG_FH_ROAD_SINGLE_ALPHA_2026_08_19.md`, committed before any fold data; grid α∈{0,1,2,3,5,7,10}, n=3, V1 official-150 3 folds / V2 862 4 folds; `results/road_loso_fh_{v1,v2}/single_alpha_campaign.json`): native gates EXACT (V1 53.824, V2 42.526). Fold argmaxes V1 {7,1,7} → **α\*=7**, V2 {5,3,5,7} → **α\*=5**; no censoring on either axis. Full-test at α\* reproduces the A27/A31 groad numbers **bit-identically** (pooled delta 0.0 — one set of numbers, not two). **V1 official: 53.980±0.059 (+0.156 vs native, t=4.6), MOVING 44.979±0.126 (+0.67, t=9.2), STATIC 49.293 (+0.31, t=4.3), APPEAR 56.529 (+0.05). V2: 42.625±0.032 (+0.099, t=5.4); canonical per-class (A30 regrouping on the road tree, `results/v2_canonical_regroup_road_ship.json`) MOVING 38.338±0.057 (+0.184, t=5.6), STATIC +0.256, APPEAR +0.083, DIRECTION +0.022 — all four positive.** Trade vs the sim chain: pooled −0.031 (V1) / −0.033 (V2), but **MOVING gain nearly doubles on V1 (+0.49→+0.67) and quadruples on V2 canonical (+0.048→+0.184, from within-1σ to ~3σ)** — the ego-estimation upgrade pays exactly where the mechanism predicts, and V2's motion class stops being a null result. Supporting measurement (`results/road_fallback_rate.json` — ad-hoc run, no producing script in the repo; reproduced as a subset and extended to all 19 seqs by `diag_road_chain.py`, A39): the road fit succeeds on **2065/2065 adjacent frame pairs across all four eval sequences — the global-ORB fallback never fires**, so "camera motion estimated from road-region correspondences" needs no hedge. Ship is now uniform: road chain on all three host settings, iKUN two-α, FlexHook single α | measured 2026-08-19 | **Option B complete and internally uniform; all paper numbers final** |
| A38 | **Welch statistics for the n=5 ablation (A34 arms) — REBUILT 2026-08-25** (`diagnostics/welch_ablation_n5.py` → `results/ablation_n5_welch.json`; the #29 comment of 2026-08-25 cited both as A38 but neither was on this tree; scipy `ttest_ind(equal_var=False)`, two-sided, t>0 = full > ablated, Welch–Satterthwaite df): **Option B (ship: road chain, (0.7,0.1)) −ego: MOVING −3.805 t=11.8 df=6.2 p=1.75e-5; pooled −0.637 t=11.7 df=6.8 p=9.81e-6; STATIC −0.843 t=7.7 p=2.84e-4; APPEAR −0.042 t=1.6 p=0.15 (ns). −multiscale: MOVING −1.846 t=6.2 df=4.8 p=1.84e-3; pooled −0.318 t=6.7 df=4.4 p=1.76e-3; STATIC −0.117 t=2.1 p=0.076 (ns); APPEAR −0.050 t=2.1 p=0.080 (ns).** Option A (sim chain @0.5) −ego: MOVING −1.890 t=15.9 p=2.4e-6, pooled −0.398 t=5.7 p=6.9e-4, STATIC −0.600 t=7.5 p=8.6e-5; −multiscale: MOVING −0.772 t=2.9 p=0.040, pooled −0.207 t=3.6 p=7.4e-3, STATIC/APPEAR ns. Means reproduce A34 exactly; the four B-arm t/p values match the #29 comment set | measured 2026-08-25 (rebuilt) | **paper §4.3: quote t and p — ego and multiscale both p<0.01 on MOVING and pooled; STATIC significant only for −ego; APPEAR never** |
| A39 | **Road-chain estimator diagnostic — four open reviewer questions closed at the estimator level** (`diag_road_chain.py`, pre-reg in its docstring, amended after an 8-pair smoke with the amendments recorded there; every adjacent frame pair on all 19 V1 seqs = 7,690 pairs [4 eval 2,065 + 15 train 5,625]; eval 0005/0011/0013 with NeuralSORT masks = inference condition, 0019 + train seqs unmasked = training condition — `dataset.py:1057` passes no boxes, so train/inference road fits differ in foreground handling [newly noted, not a #29 item]; referee = photometric residual mean|warp(I_{t-1},H) − I_t| over band-0.5-minus-boxes, vs identity and vs the similarity-chain H; `results/road_diag/road_chain_diag{,_evalnear}.json` — kept LOCAL/untracked by decision 2026-08-26; regenerate with the script, deterministic): **(Q4) fallback 0/7,690** — extends A37's 2,065/2,065 to every training sequence. **(Q1) wild fits: GMC_HGATE bounds are INVALID for the road chain** — a road-plane H under forward motion carries h32 ≈ v/(f·d) physically (oxts-verified: 0005 17.6 m/s → −1.1e-3, 0011 5.3 m/s → −4.2e-4); the bound flags 815/7,690 = ordinary fast driving (0004: 185/313 at 13.4 m/s). Photometric referee: road beats sim on 6,934/7,690 (90%); p50 residual identity 18.95 → sim 15.51 → road 10.53. Pre-registered "road worse than identity" fires 792× but 727 are stationary-camera pairs (vf<0.5 m/s, H≈I, sign = noise); among 6,520 moving pairs 27 exceed +1 gray level, ALL on unmasked training seqs at 11–17 m/s (0010/0008/0020/0015), **0 on the four eval seqs** (eval moving pairs road<sim 295/296, 346/346, 332/339, 833/861; road worse than I by >1: 0/0/0/0). Grid displacement is not a wild criterion here (max 1,178 px on 0020 f0715 still aligns the road better than sim/identity). **(Q2) road_band = 0.5 is data-justified**: photometric over the full lower half (where objects live) has 0.5 as the minimum on every eval seq (ALL p50 for 0.3/0.4/0.5/0.6/0.7 = 11.92/11.01/10.41/11.02/12.48); over the near road only (inside every band's sampling — fair test) 0.5–0.7 tie (8.23/8.10/8.39, per-pair winners 453/540/591) while 0.3/0.4 lose everywhere (10.78/8.91): bands starting above the KITTI horizon (cy/h ≈ 0.46) admit non-road; 0.5 = widest band below the horizon = the only one of the tied three that samples the mid-range object region instead of extrapolating to it. HOTA band sweep NOT warranted (near-road tie; {0.5,0.6} would be the only meaningful cell). **(Q3) ORB on the same band does NOT starve** — kp 1,498 / good 188 / inliers 159 (p50), good<12 on 0.6% of pairs, 9/7,690 fails — but its H is worse: inlier res 1.24 vs LK 0.88 px, photometric 14.37 ≈ sim 15.51 vs road 10.53, ORB-H vs LK-H grid disagreement p50 37 px. The design claim "asphalt is too low-texture for ORB" is FALSE as stated; the measured claim is that ORB's band keypoints do not fit the road plane (worst on 0011: good p50 42, 10.8% <12 — the seq where the road chain helped most, A29). Self-check: inline fit ≡ engine H exactly (OpenCV RANSAC fixed seed). Road fit 5.0 ms/pair | measured 2026-08-25 | **paper §3.1/§4.1 material: fallback 0/7,690 (train+eval); band 0.5 = horizon-bounded, photometrically justified; ORB rationale must be rewritten (wrong points, not too few); never apply HGATE-style gates to the road chain** |
| A40 | **Coupling story from existing sweeps + E4 estimator follow-ups (no HOTA runs).** **(E2) single-α road sweep** (`results/ground_road/alpha_sweep_ikun.csv`, n=3 seeds 0–2, warm11): APPEAR peaks at α≈0.2 (46.509) and falls monotonically (0.35: 46.475, 0.5: 46.390, 0.7: 46.096, 1.0: 45.674, 2.0: 43.910) while MOVING rises to α=1.0 (25.53→27.66→29.43→31.50→32.61→33.13) and STATIC to 0.7–1.0 (44.584); pooled peaks at 0.5 (44.721) — a single α is a forced compromise between classes wanting α≤0.2 and α≥1.0. Two-α (0.7,0.1) = 44.847/32.606/44.584/46.468: MOVING/STATIC equal the α=0.7 row BY CONSTRUCTION (router and per-class grouping share `classify()`); the entire +0.261 pooled gain over single-α@0.7 is APPEAR +0.372 from α_app 0.7→0.1. The LOSO single-α pick 0.35 (A27) sat where MOVING is 29.4. **(E4a) ORB-vs-LK mechanism on the band** (`diag_road_chain.py --seqs eval --tag e4`, 2,065 pairs, `results/road_diag/road_chain_diag_e4.json` — local/untracked, regenerate with the script): RANSAC threshold is NOT the cause — ORB refit at 3 px gives photometric 15.01 vs 14.79 at 5 px vs LK-road 10.22 (moving pairs 15.39/15.04/10.41). Inlier LOCATION is: horizontally ORB is as central as LK (central-corridor fraction 0.81 vs 0.75 — the "curbs/margins" guess is wrong); vertically ORB inliers sit at y_p50 = 0.56h vs LK 0.64h (0011: 0.55 vs 0.71). ORB's distinctive keypoints cluster in the far field just below the horizon (0.46h), where motion is small and off-plane structure mixes in; Shi-Tomasi/LK points extend down the near road where the plane is well observed. Mechanism = depth of the sampled points — not brightness, not texture count, not threshold. ORB-band H ≈ the global sim fit overall (15.04 vs 15.03). **(E4b) train/inference mask mismatch quantified** (masked eval pairs n=919 on 0005/0011/0013; no-mask = training condition, `dataset.py:1057`): grid disagreement vs the masked fit p50 9.8 px / p95 47.5 px; photometric masked 8.41 vs no-mask 8.63; no-mask worse by >1 gray level on 113 pairs, masked worse on 83 (0011 47/17, 0013 29/37). Second-order at the estimator level (median ≈ ¼ of the ORB-vs-LK gap, not one-sided); consistent with A15 (dirtier train-side ego harmless or augmenting). NOT a retrain trigger; open item. Re-run reproduces A39's eval columns exactly; self-check 0.00 | measured 2026-08-25 | **coupling = measured (APPEAR is the class that caps a single α); ORB rationale = far-field sampling; mask mismatch = documented, not actioned** |
| A41 | **Lazy global fallback in road mode + FPS re-measured — the shipped chain is 2.3× faster than the previous ship, not 26% slower** (user decision 2026-08-25: measure the system the paper describes instead of writing the never-firing fallback into it). Code: `manager.py` road branch now calls `estimate_homography` ONLY when `estimate_road_homography` returns None (0/7,690 pairs, A39); the global cumulative buffer is no longer maintained in road mode (its only reader was the non-road branch; repo-wide grep: no external consumer). **Equivalence proof**: 0011 seed0 road cache rebuilt with the new code (`_lazychk`, then deleted) vs the existing `_sw12d_groad_seed0` cache — 183,872 entries, 0 missing, 0 extra, max |Δ| = 0.00 → every HOTA number is unchanged. Tests: 12/14 pass; the 2 failures (`test_manager_17d.py` expects 13D/17D) predate this change (12D since 2026-08-10). **FPS** (`profile_inference.py --seq 0011 --n 500` = 373 frames, GMC_MODEL=similarity, CPU, base py3.13/torch 2.11, idle machine, 16 reps in two batches, reps 1–3 excluded, median+IQR of 13 warm reps; `results/fps_lazy/summary.json` (all 16 reps inside); A36 record = the unchanged tracked `results/fps_profile.json`): **road 149.3 FPS process-only (IQR 147.5–150.8; 6.7 ms/frame), 67.2 incl. I/O; sim 63.9 (IQR 63.8–64.0; 15.6 ms), 42.3 incl. I/O.** Caveats: (a) sim's code path is unchanged yet reads 63.9 vs A36's 42.8 → A36 absolute numbers were machine-state-specific (A36 noted residual load); only same-session comparisons are valid, and this session has both; (b) road dispersion min/max 122.5–169.6 across all 16 (two outliers) — IQR is tight; (c) 0019 cannot be profiled with this script (needs NeuralSORT tracks; FH V2 seq). Per-frame budget now: road fit ≈ 5 ms + feature/aligner ≈ 1.7 ms | measured 2026-08-25 | **paper §4.1 FPS: 31.8 → 149 (process-only, CPU); no fallback sentence needed; CLAUDE.md FPS line updated** |
| A42 | **iKUN evaluated on the official 150-expression seqmap — the 158-list was our own over-enumeration; every iKUN number re-derived, α unchanged** (found 2026-08-29 while checking the Table 2 caption "27/158 vs 25/150". Procedure, TrackEval-only: for every run dir under `hota_eval_ikun_linear_additive_sw12d_{groad,noego_*_nm,nomulti_*_road}_seed*_warm11/` [485 dirs = 71 full-test + 414 LOSO fold dirs] filter its `seqmap.txt` to the official list, split per class with `classify()`, rerun `run_mot_challenge.py --METRICS HOTA` with the same GT/tracker arguments as `run_te()`, write `result_off150.json` beside the untouched `result.json`; then aggregate (n=3/n=5 means, Welch, LOSO argmax/censor/median rule of `run_two_alpha_sweep.py`) → `results/official150/ikun_official150.json` [tracked]. The one-off driver `rescore_official150.py` and `diagnostics/aggregate_official150.py` are kept LOCAL/untracked by user decision; official list tracked as `seqmaps/refer_kitti_v1_test_official_150.txt`). Provenance: TransRMOT `datasets/data_path/seqmap.txt` (fetched upstream) = FlexHook `kitti-1.txt` = complement of iKUN `utils.py EXPRESSIONS['dropped']` (8 exprs: braking×2 / horizon×2 / back-to-the-camera×4; iKUN `test.py:236,274` skips them; their GT is NOT degenerate — 99/179/215/11 rows — A31's "degenerate GT" wording was wrong). Our 158 came from `build_seqmap.py` (11aee02, 2026-02-26) listing TempRMOT's output dir, repeated by `run_ikun_linear_additive.py:107` `os.listdir(expression/{seq})`. In our trees the 8 are pure damage (braking predict=0 vs gt=99; men-back-to-the-camera predict=205 vs gt=11). TrackEval-only rescore, predictions byte-identical: **native 44.543** (published 44.56; 158-list 44.224 → gap −0.34 becomes −0.02, same shape as A31), MOV 25.778 / STA 43.914 / APP 46.743. **Ship two-α (0.7,0.1) n=3: 45.158±0.104 (+0.615 vs native, +0.598 vs published), MOVING 32.902±0.660 (+7.12), STATIC 44.584±0.122 (+0.67), APPEAR 46.852±0.043 (+0.11)**; n=5: 45.117±0.099 / 32.584±0.642. LOSO re-selection on 150: two-α fold argmaxes identical cells {(0.7,0.1),(2.0,0.7),(0.7,0.2)} → (0.7, 0.1) unchanged; single-α {0.2, 2.0, 0.5} → 0.35 unchanged. Ablation B-arm n=5 (supersedes A38 for the paper): −ego 44.473±0.065 / MOV 28.716±0.351 / STA 43.702±0.219 (Δ −0.644 t=12.1 p=6.7e-6; −3.868 t=11.8 p=1.8e-5; −0.843 t=7.7 p=2.8e-4); −multiscale 44.799±0.022 / 30.700±0.204 / 44.428±0.061 (Δ −0.319 t=7.0 p=1.5e-3; −1.885 t=6.3 p=1.8e-3; STATIC −0.117 t=2.1 p=0.076 n.s.). STATIC values unchanged (none of the 8 is static). Side measurement (user question): a merged MOVING∪STATIC eval class (37 exprs, 17,398 GT dets, 63% static) gains only +2.34 (38.133 → 40.472±0.289) — det-mass dilution; keep the classes separate in reporting. FlexHook rows unaffected (already on the official lists). `run_ikun_linear_additive.py` still enumerates 158 — read `result_off150.json`, never `result.json`, for iKUN paper numbers | measured 2026-08-29 | **iKUN native ≡ published (−0.02); ship +0.60 above published; all iKUN gains slightly larger; α unchanged; A31 lesson applied to iKUN** |
| A43 | **MOVING class redefined (user-supplied keyword list) — shared classifier drives BOTH the α router and the per-class rows; iKUN re-run, FlexHook regrouped** (2026-08-29, user decision after finding mis-aligned classifications. New `gmc_link/moving_kw.py`: MOVING = {moving, in motion, driving, walking, running, jogging, crossing, riding, travelling, traveling, braking, brake, accelerat, decelerat, slowing down, speeding up, approaching, overtaking, receding}; STATIC = the old 7 stems; else APPEARANCE; `-`→space so slug ≡ sentence. All three eval scripts + `run_v2_canonical_regroup.py` import it (pre-A43 copies in git history; V2 slug `classify()` kept for legacy rows). Counts: V1 official-150 MOVING 25→**21** (→APPEARANCE: `0011+turning-cars/-vehicles`, `0011+cars/vehicles-which-are-faster-than-ours`), STATIC 12, APPEAR 117; V2 canonical 136→**111** (−29 faster/turning, +4 "braked"), STATIC 93, APPEAR 658; no MOVING∩STATIC. Procedure: iKUN two-α (0.7,0.1) re-run for 5 seeds × {groad, noego, nomulti} into `*_mkw` trees (old trees untouched); gate A: vs old trees only the two `faster-than-ours` predict.txt differ — `turning-*` have NO entries in `iKUN/ikun_results_v1_cascade_full.json`, so their predict.txt is empty under any α (silent fail, gt=30 rows); official-150 rescore (`rescore_official150.py --out-name result_off150_mkw.json`, local) + aggregate (`diagnostics/aggregate_official150.py --tree-suffix _mkw`, local) → `results/moving_kw/ikun_official150_mkw.json`; gate B: STATIC bit-identical in all 15 dirs. FlexHook: routing degenerate (α_mot=α_app) → TrackEval regroup only (`regroup_fh_mkw.py` local → `result_mkw.json`; V2 via `run_v2_canonical_regroup.py --tree …groad_seed{seed} --suffix _warm11`) → `results/moving_kw/{fh_mkw,v2_canonical_regroup_mkw}.json`; gates C/D: STATIC identical, counts exact. **iKUN (official-150)**: native pooled 44.543 / STATIC 43.914 unchanged; MOVING 25.778→**27.697**, APPEAR 46.743→46.300. Ship n=3 **45.261±0.086** (+0.718 vs native, **+0.701 vs published**; was 45.158±0.104/+0.598), **MOVING 36.279±0.627 (+8.58**; was 32.902±0.660/+7.12), STATIC 44.584±0.122 (same), APPEAR 46.372±0.045 (+0.07). n=5: 45.227±0.082 / MOV 35.929±0.653 / STA 44.545±0.107. Ablation n=5: −ego 44.692±0.060 / MOV 32.353±0.363 / STA 43.702±0.219 (Welch Δ −0.535 t=11.8 p=5.0e-6; −3.576 t=10.7 p=3.0e-5; −0.843 t=7.7 p=2.8e-4); −multiscale 44.997±0.014 / MOV 34.401±0.129 / STA 44.428±0.061 (Δ −0.230 t=6.2 p=2.9e-3; −1.529 t=5.1 p=5.5e-3; STATIC −0.117 t=2.1 p=0.076 n.s.). Pooled rises because the two `faster-than-ours` exprs score better at α_app=0.1 than at 0.7. **FH V1 official-150**: MOVING native 44.31→**47.896**, ship 44.979±0.126→**48.330±0.189**, Δ +0.67 (t=9.2)→**+0.434 (t=4.0)**; pooled 53.980±0.059 / STATIC 49.293 unchanged; APPEAR 55.825±0.049 (+0.104). **FH V2 canonical**: MOVING native 38.154→**38.561**, ship 38.338±0.057→**39.255±0.044**, Δ +0.184 (t=5.6)→**+0.694 (t=27)**; STATIC +0.256 / pooled 42.625±0.032 unchanged; APPEAR 42.604±0.036 (+0.062); legacy DIRECTION row (slug-MOVING ∩ canonical-APPEAR) now 83 exprs, −0.224 — not reported. **LOSO re-selection under the A43 router (same evening, user-requested)**: the two folds that contain the reclassified 0011 exprs (hold-0005 → select on 0011+0013; hold-0013 → select on 0005+0011) re-run for the full 6×5 grid × 3 seeds (180 fold dirs in the `_mkw` trees, pooled-only official-150 TrackEval via local `loso_mkw_driver.py`); hold-0011 selects on 0005+0013 and contains none of the changed exprs → its pre-A43 fold dirs are reused (predictions provably identical). am=aa cells are bit-identical old vs new (routing-invariant), every am≠aa cell shifted. Fold argmaxes: hold-0005 **(1.0, 0.1)** @44.854 (was (0.7,0.1) @44.692), hold-0011 (2.0, 0.7) censored both axes (unchanged), hold-0013 **(1.0, 0.1)** @47.758 (was (0.7,0.2) @47.606) → `run_two_alpha_sweep.py` rule gives **(α_mot\*, α_app\*) = (1.0, 0.1)**, not (0.7, 0.1) (`results/moving_kw/loso_two_alpha_mkw.json`, local `diagnostics/loso_two_alpha_mkw.py`). Mechanism: the two `faster-than-ours` exprs were the members of the α_mot group that lose with larger α; once routed to α_app the group's optimum moves up. Full-test at (1.0, 0.1), 5 seeds × 3 arms (`*_mkw/am1.0_aa0.1`, `results/moving_kw/ikun_official150_mkw_am1.0.json`): ship n=3 **45.304±0.115 (+0.744 vs published), MOVING 37.139±0.923 (+9.44)**, STATIC 44.584±0.196, APPEAR 46.372±0.045 (APPEAR identical to (0.7,0.1) — α_app unchanged); n=5 45.282±0.090 / MOV 36.988±0.684 / STA 44.532±0.159; −ego 44.609±0.069 / 32.372±0.386 / 43.397±0.229 (Welch Δ −0.673 t=13.2 p=1.8e-6; −4.616 t=13.1 p=8.1e-6; −1.135 t=9.1 p=3.5e-5); −multiscale 44.999±0.022 / 35.143±0.384 / 44.348±0.122 (Δ −0.283 t=6.8 p=1.6e-3; −1.846 t=5.3 p=1.6e-3; STATIC −0.184 t=2.1 p=0.076 n.s.). Both α sets are on disk; **user decision (same evening, on the reviewer argument that the stated LOSO procedure must produce the reported value): v3 reports (1.0, 0.1)**; (0.7, 0.1)-under-new-router numbers stay in `ikun_official150_mkw.json` as the comparison. FH α=7 / 5 unaffected (routing degenerate). Paper (`gmc_v3.tex` = committed `gmc_v2.tex` 148b6be + A42 + A43; the A42 edits that sat uncommitted in v2 were reverted 2026-08-30, so v2 stays as committed) §3.4 α sentence, §4.1 "one-seventh" (edited from one-sixth), and §5 "14 of 126 direction expressions treated as appearance" (now by design, not a misread) need user prose | measured 2026-08-29 | **all three MOVING gains re-derived on the user's class definition: iKUN +9.44 (α re-selected by LOSO to (1.0, 0.1)), FH V1 +0.43, FH V2 +0.69; iKUN pooled +0.744 vs published** |
| A44 | **Paper-content additions from the RMOT survey (`docs/RESEARCH_RMOT_CONTENT_SURVEY_2026_08_30.md` §6): HOTA sub-metrics, three-host single-α sweep, single-α n=5 row, STATIC column** (2026-08-30; user decisions: DO A/B-table/E/F + FH V1 sweep, C/D/G/H sentence drafts record-only). **Sub-metrics** (one pooled TrackEval per dir on the official seqmaps, gate HOTA == recorded pooled PASS; `results/moving_kw/submetrics.json`, local `submetrics_mkw.py`), native → ship, n=3: iKUN HOTA 44.54→45.30, DetA 32.04→32.90 (**+0.87**), AssA 62.48→62.94 (+0.46), DetRe 48.53→50.06 (+1.52), DetPr +0.40, AssRe +0.59, AssPr +0.03, LocA +0.04; FH V1 DetA 43.35→43.51 (+0.15), AssA 66.92→67.07 (+0.15), DetRe +0.42, DetPr −0.13; FH V2 DetA 30.63→30.75 (+0.12), AssA 59.19→59.23 (+0.04), DetRe −0.20, DetPr +0.28. **Single-α sweep under the A43 class** (single α is routing-invariant, so the pre-A43 sweep predictions stand; iKUN 16 α × 3 seeds rescored `--out-name result_off150_mkw.json`; FH V1 α∈{1,2,3,5,10} × 3 seeds = **15 new fusions** on the official list into the `_off150` trees; FH V2 canonical regroup α∈{0,1,2,3,5,7,10}; `results/moving_kw/alpha_sweep_mkw.json`, `v2_canonical_regroup_mkw_sweep.json`, local `alpha_sweep_mkw.py`), pooled/MOVING: iKUN 0: 44.543/27.697 · 0.2: 44.820/30.339 · 0.35: 44.910/32.453 · 0.5: **45.008**/34.878 (single-α pooled max) · 0.7: 44.866/36.279 · 1.0: 44.558/37.139 · 1.25: 44.276/**37.539** · 2.0: 42.960/36.897. FH V1 0: 53.824/47.896 · 1: 53.869/47.942 · 2: 53.904/47.998 · 3: 53.929/48.060 · 5: 53.981/48.161 · 7: 53.980/48.330 · 10: 53.887/48.317 (pooled within 0.1 of max over α∈[2,10]). FH V2 0: 42.526/38.561 · 1: 42.553/38.683 · 2: 42.585/38.851 · 3: 42.603/38.979 · 5: 42.625/39.255 · 7: 42.576/39.537 · 10: 42.413/39.839 (pooled within 0.05 of max over α∈[2,7]; MOVING rises to 10). **Single α = 0.35, n=5** (A42 single-α LOSO; still the LOSO value under A43; seeds 3–4 run 2026-08-30; gate seeds 0–2 == A42 sweep PASS; `single_alpha_0.35_n5.json`): pooled 44.948±0.112, MOV 32.423±0.247, STA 44.349±0.064 → two-α (1.0, 0.1) n=5 is **+0.33 pooled / +4.57 MOVING / +0.18 STATIC** over the single-weight LOSO choice, and +0.30 pooled / +2.26 MOVING over the best single weight (0.5, n=3). Paper (`gmc_v3.tex`): Table 1 + ΔDetA/ΔAssA columns; Table 3 + STATIC column + single-α row; tables \footnotesize. An α-sensitivity Table 4 (5 rows × 3 hosts) + one reference sentence was added first and pushed the text ~30 lines past the 4-page limit; **user decision Z (same day): retract only the height-costing additions — Table 4 and its sentence removed, captions shortened, original prose untouched** → content ends on p4, p5 references-only, 0 overfull. The α-sweep numbers stay here and in the JSON for a later version. C/D/G/H drafts (TempRMOT negative, parameters, statistics, frame-convention footnote) recorded in the survey §6, not inserted | measured 2026-08-30 | **DetA carries iKUN's gain (+0.87 vs AssA +0.46); FH gains split evenly; FH pooled flat over α; iKUN's single-α optimum is narrow → the two-weight routing is worth +0.33 pooled / +4.6 MOVING** |
| A45 | **教授改稿合併(wording=教授 / facts=v3)+ Table 3 Others 欄**(2026-08-31;來源 `gmc_v1_by_WTC.pdf` 五頁重建;OTHERS = official-150 非 MOVING 的 129 條,TrackEval 21 目錄,`results/moving_kw/others_mkw.json`,local 程式碼在 session 内):native 45.870、full 46.051±0.053、single-α(0.35) 45.990±0.122、−ego 45.772±0.038、−multiscale 45.960±0.036 — 非運動類在各變體間變動 ≤0.28,−ego 仍最低(呼應 STATIC 結論但幅度較稀釋)。Table 1 改列 DetA/AssA 數值對、刪 Gain 欄;Table 3 欄 Moving/Others/Pooled;§5 未再引用 STATIC 數字。教授刪的 v1 句照刪(future work、Third-two-weights、ORB、§2.2 技術引用句);v2+ 事實句全保留(150 名單、44.543 揭露、7690、FPS、Welch、warm-up、1/g、horizon)| measured 2026-08-31 | **v3 散文 = 教授版;數字層不變;Others 欄取代 STATIC 欄** |
| A46 | **TransRMOT third-host campaign — prereg committed (126b2f5, docs/PREREG_TRANSRMOT_HOST_2026_09.md) + advisor side-track S1/S2 done; inference blocked on the NVIDIA driver** (2026-09-02. P0 diagnosis: kernel 7.0.0-30 has no nvidia module — `nvidia-driver-580-open` installed but no dkms and no prebuilt module for this kernel; repair = `apt install dkms linux-headers-$(uname -r) nvidia-dkms-580-open` + modprobe/reboot, user-side. Host facts verified: official checkpoint0099 on disk (`~/RMOT/exps/default/`), native gate threshold = `filter_dt_by_ref_scores(...,0.5)` inference.py:575, predict.txt has NO score column → P2 = score-dumping inference re-run, GPU-only in practice (CPU fallback exists but ~150k frame-forwards). Exp 18's old TransRMOT+GMC (+4.55, min-fusion) is superseded and not comparable. **S2 router audit** (`results/moving_kw/s2_router_audit.json`): independent semantic labels of the 150 official V1 test expressions under the A43 three-class definition vs `gmc_link.moving_kw.classify` → **150/150 agreement** (21/12/117); the only 'misrouting' left is the DEFINITIONAL exclusion of direction/turning/speed exprs (214/818), already disclosed in §5 — the 'router misfires' question is closed. **S1 ego-magnitude stratification** (`results/moving_kw/s1_ego_stratification.json`, local `s1_ego_stratification.py`; ego = mean px displacement of a 16×8 road-band grid under the road homography, averaged over each expression's GT frames; gain = per-expr HOTA ship(am1.0_aa0.1, n=3) − native from TrackEval detailed CSV — NB detailed CSV is 0–1 scale, ×100): mean ego 0005 52.3 px / 0011 22.6 / 0013 24.3. MOVING class (n=21): Spearman ρ=0.41, p=0.065 — positive but MARGINAL, and ego variance is mostly between-sequence while gains at the same ego level span 0…+30 (e.g. 26.2 px: +0.4 vs +30.2). All-150: ρ=0.12 n.s. Verdict: the advisor's strong claim ('gain tracks ego magnitude') is NOT supported at expression level; a mild positive trend only — usable as one cautious sentence, not a figure. The measured explanation of the iKUN-vs-FH gap remains host motion deficit (2026-05-26), which ego magnitude logically cannot explain (same videos). Largest per-expr gains: 0013+left-people/persons-who-are-walking +29/+30, 0013+left-pedestrian +17.5, moving-cars 0011/0005 +13.3/+12.3; zeros are host silent-fails | prereg + S-track measured 2026-09-02 | **campaign P2–P6 pending GPU; S2 closes the router question (150/150); S1: ego-magnitude stratification is weak evidence — do not oversell** |

Killed at verification (do not pursue): `max(gmc,0)` clamp (twice-closed fusion family);
per-expression mean-centering as ship change (threshold family, Gate C; diagnostic-only);
V2 3-α arm (recipe-split family); V2 canonical-text A/B (already the ship — `_raw` = raw_sentence).

## Key Bugs Fixed Along the Way

| File | Bug | Fix |
|------|-----|-----|
| `core.py` | `len(cv2.DMatch)` crash in Lowe's ratio test | check `len(match_pair)==2` first |
| `core.py` | mask initialized to all zeros (no features) | `np.ones * 255` |
| `manager.py` | object bboxes not passed to GMC engine | added `detections` parameter |
| `manager.py` | `clip_feat_dim=None` int cast crash on new ckpts | `int(... or 512)` (Lever A wiring) |
| `dataset.py` | `_try_load_cache` decompressed NpzFile N× → OOM at 27GB | hoist `data[key]` out of comprehension |
| `alignment.py` | `vis_dim` parameter misleading | renamed to `motion_dim` |
| `train.py` | relative imports fail as script | absolute imports + `sys.path.insert` |
| Case 2 / Path A | id-space mismatch in window-score join | id-fix before CV-OOF |
| failure_audit | "FN_ikun_coverage" was a loader schema-misread | corrected: turning-verbs FN_fusion 60%, ped-walking FN_tracker 44% |

---

## Open Questions / Status

- **Path to paper SOTA 48.84 is closed** on public components (DDETR+NeuralSORT pair
  unreleased; open-vocab detector + LVLM substitutes all NEG). Honest pooled ceiling is
  ~44.6 on YOLOv8-NS.
- **Ceiling-break campaign is exhausted** at 24+ levers. The aligner is
  representation-bound; the fusion recipe is irreducible at 18 params.
- The **depth-augmented 17D iKUN** candidate (AUC-NEG, HOTA +0.215 sig) remains a live
  iKUN-only direction not folded into the current sw ship — re-validation under
  sw+no-EMA pending if revisited.
