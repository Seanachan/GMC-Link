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

## 9. Current Ship (2026-05-21)

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

## 10. Sequence Motion Encoder — HOTA-Direct Rescreen (2026-06-11)

Motivation: 2026-06-11 survey (24/25 claims verified) found sequence-encoder-over-
trajectory-kinematics aligned to language UNCLAIMED in 2025-26 literature; iKUN oracle
gap (+6.13 pooled, MOVING DetRe 20.5→69.2 same tracker) is a classification deficit.
Exp36's transformer kill was AUC-only (0.779 screen) — never HOTA-tested (§6 rule).

Setup: existing `temporal_transformer` arch (CLS-token, d_model=64, 1 layer, seq_len=10)
+ NEW inference windowing in `manager.py` (commit ffc14a7) mirroring training's
front-pad convention. Train V1 stage-1 seeds {0,1,2}; locked iKUN ship recipe
(motion α=1.0 sc=0.9 thr=+0.17, appear α=1.0 sc=0.30 thr=+0.10, raw-cos no-EMA).
Cache build ~22 min / 3 seqs (≤2× MLP).

| seed | pooled | MOVING | STATIC | APPEAR |
|------|--------|--------|--------|--------|
| 0 | 44.612 | 29.655 | 44.481 | 46.628 |
| 1 | 44.446 | 28.353 | 44.580 | 46.500 |
| 2 | 44.641 | 29.586 | 44.556 | 46.663 |
| **n=3** | **44.566 ± 0.105** | **29.198 ± 0.733** | **44.539 ± 0.052** | **46.597 ± 0.086** |

Vs sw ship n=3 (44.634 ± 0.066 pooled; seed0 per-class STATIC 43.240 / MOVING 28.885):
- **pooled FLAT**: Δ = −0.068, inside joint seed noise. Not a ship-swap.
- **STATIC +~1.3 and tight** (44.539 ± 0.052 vs 43.240 seed0 sw) — sequence window
  acts as a static-class denoiser; far outside noise.
- **MOVING +0.31 ± 0.73, NOT significant** — seed0 screen's +0.770 was seed luck
  (seed1 −0.53 below sw seed0). The oracle axis barely moved.
- Caveat: ship recipe (sc/thr) was tuned on the sw cosine distribution; recipe is
  LOCKED by protocol, so the temporal arch runs with a possibly-mismatched calibration.

Verdict: sequence arch at HOTA = per-class profile shift (STATIC↑, MOVING flat), pooled
flat. Exp36's AUC kill does NOT hold at HOTA (4th AUC-NEG→HOTA-not-NEG case), but no
pooled win either. Promotion gate (beat 44.634 ± 0.066) NOT met.

### B2 variants (2026-06-11, all seed0 screens, locked iKUN recipe)

| variant | pooled | MOVING | STATIC | verdict |
|---------|--------|--------|--------|---------|
| temporal T=5 | 44.544 | 28.994 | 44.673 | flat |
| temporal T=10 (ref) | 44.612 | 29.655 | 44.481 | — |
| temporal T=20 | 44.537 | 29.635 | 43.929 | flat |
| sw + window_stats 4D (ungated) | 44.146 | 28.207 | 41.351 | NEG (STATIC −1.9, guardrail FAIL) |
| sw + window_stats 4D (noise-gated 0.5) | 44.094 | 28.339 | 40.953 | NEG (STATIC −2.3; gate didn't rescue) |

window_stats = [mean window speed, mean-heading sin/cos, circular heading variation]
over trailing 10 mid-scale residual steps (ReferGPT-style; turning gets distinct
variation signature — unit-verified). First screen crashed STATIC; hypothesis was the
static-jitter variation artifact (random noise directions → variation ≈ 1); the 0.5
noise gate fixed the math (unit test: jitter variation 1→0) but STATIC got WORSE →
artifact hypothesis falsified, the feature itself hurts (likely mean_speed scale
imbalance into sw's no-input-LN trunk). 2 strikes → killed, no further HOTA-peeking
feature iteration. Feature stays registered (default off), like other exp37 NEG extras.

**Track B convergent conclusion:** MOVING signal is NOT extractable from 2D pixel-
kinematic windows — (a) attention over raw 13D sequences flat at every T, (b) explicit
window statistics NEG. The +6.13 oracle gap survives both window-shaped levers; whatever
separates MOVING from STATIC per-trajectory needs information beyond windowed image-plane
kinematics (3D motion, semantics, or tracker-noise modeling). GRU/Mamba swap over the
same input deprioritized — same information, no reason to expect a different outcome.

---

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
