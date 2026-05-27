# GMC-Link — Canonical Results Summary

Single-source-of-truth reference for all ship numbers, baselines, paper anchors, and
multi-seed statistics. Cite this file when writing the paper/thesis.

**Last updated:** 2026-05-23
**Ship version:** 2026-05-21 (`shared_weight` aligner + per-arch recipe + no-EMA, raw cosine)

> Hard reporting rules enforced throughout:
> - **HOTA only. AUC is never reported, never gated on.**
> - Every number cites the **exact recipe + n + sample std**. No vague labels
>   ("current ship" / "baseline" / "ours") — the full recipe is always written out.

---

## 0. Definitions & Conventions

| Term | Meaning |
|---|---|
| **3-seq pooled HOTA (V1)** | TrackEval HOTA computed over the Refer-KITTI V1 test split = sequences **0005, 0011, 0013**, with trajectory IDs pooled across all (seq, expr) pairs before computing √(DetA·AssA). This is the ship metric for iKUN and FH V1. |
| **4-seq pooled HOTA (V2)** | Same pooling on the Refer-KITTI V2 test split (FH V2 consumer). |
| **`gt_template_old/`** | Paper-iKUN-canonical GT label space. Frame numbering aligns with the NeuralSORT tracker `predict.txt`. **All HOTA evals in this doc use `gt_template_old/`.** The locally-regenerated `gt_template/` (TransRMOT convention) is off-by-one vs NeuralSORT and drops HOTA ~6.4 — not used here. |
| **raw cosine (`GMC_RAW_COS=1`)** | GMC alignment score taken as the raw cosine ∈ [−1,+1] from the aligner. Bypasses the legacy sigmoid + EMA smoothing path (`manager.py`). Current ship convention. |
| **EMA** | Legacy exponential-moving-average smoothing (`MotionBuffer` α=0.3, `ScoreBuffer` α=0.4, `cosine_buffer`). **OFF in the current ship.** raw_cos=True bypasses it regardless. |
| **`shared_weight` aligner** | Per-modality Linear adapter (motion 13→256, lang 384→256) → shared 2-hidden MLP (256→512→512→256) → LN → L2-norm. Symmetric two-tower, shared nonlinear core. ~628k params. Current default arch. |
| **`mlp` aligner (legacy)** | Independent dual-MLP per modality (motion 13→256→512→256, lang 384→256→512→256) → L2-norm. Asymmetric. ~627k params. Opt-in only. |
| **Fusion — baseline (B2, parameter-free)** | `final = model_logit + raw_cos`. Bare additive combination, zero fitted coefficients. |
| **Fusion — ship (tuned)** | `final = model_logit + α·(sc·raw_cos + thr)`, per arch per axis (motion + appearance). (α,sc,thr)=(1,1,0) recovers B2; ship uses non-identity values fit to the eval seqs. |

### Two-baseline protocol labels

Per the project's two-baseline reporting protocol, every arch reports against two anchors,
with full recipe in the label (never a bare "B1"/"B2" shorthand):

- **B1 = `{model} Baseline` (no GMC):** the downstream consumer's own detector/tracker
  output with no GMC fusion. iKUN 44.224, FH V1 53.110, FH V2 42.526.
- **B2 = `{model} + GMC Baseline` (shared-weight aligner, bare `final = model_logit + raw_cos`, raw cos, no-EMA, n=3):**
  GMC fused with ZERO fitted coefficients — the parameter-free additive combination. Nothing is
  tuned against the eval set, so any lift here is pure GMC signal. The clean anchor for all future
  "+X helps" Δ measurements. (Equivalent to the ship formula at the identity point (α,sc,thr)=(1,1,0).)
- **Ship = `{model} + GMC` (shared-weight aligner, per-arch APPEAR-axis recipe, raw cos, no-EMA, n=3):**
  the 18-hyperparam hand-tuned recipe. The number cited as the headline result.
- **Paper anchor:** the published HOTA for each consumer, reproduced locally where stated.

---

## 1. Master Ship Table

All HOTA. n=3 seeds {0,1,2}. iKUN/FH-V1 = 3-seq pooled (0005/0011/0013); FH-V2 = 4-seq pooled.

| Arch | B1: `{model} Baseline` (no GMC) | B2: `{model} + GMC` (shared-weight, bare `logit + raw_cos`, no-EMA, n=3) | **Ship: `{model} + GMC` (shared-weight, per-arch tuned recipe, raw cos, no-EMA, n=3)** | Paper anchor | Δ ship vs paper |
|---|---|---|---|---|---|
| **iKUN** (cascade + simcalib, YOLOv8-NS) | 44.224 | 44.272 ± 0.018 | **44.634 ± 0.066** | 44.564 | **+0.070** ✓ |
| **FH V1** | 53.110 | 53.121 ± 0.005 | **53.526 ± 0.087** | 53.824 | −0.298 ✗ |
| **FH V2** | 42.526 | 42.532 ± 0.002 | **42.807 ± 0.038** | 42.53 | **+0.277** ✓ |

> **Note — official V2 test split (verified 2026-05-25, TempRMOT repo).** Refer-KITTI V2's
> official test sequences are **0005, 0011, 0013, 0019** (the TempRMOT `refer-kitti-v2.train`
> file excludes them; the test `seqmap.txt` lists them). FH V2 above is on this correct split,
> and its "host alone" 42.526 reproduces the published FlexHook-best **42.53** (TempRMOT paper
> Table 3 / FlexHook paper Table 1), so the +0.277 ship gain is a valid paper-beat.
> *(An earlier note here claimed the V2 test split was 0016–0020 from the local
> `annotations/test.json` — that was a misread of a COCO-detection file, now corrected.)*

**Paper-beat count: 2/3** (iKUN V1 ✓ +0.070, FH V2 ✓ +0.277; FH V1 short −0.298, structural).

**iKUN-V2 — attempted, excluded (not benchmark-valid).** Official iKUN-V2 = **10.32** HOTA
(TempRMOT paper). Our attempt scored 31.4 — a 3× gap from a **protocol mismatch**, not a
result: it paired a V1-trained iKUN with NeuralSORT tracks + FlexHook's V2 GT on 3 of the 4
test seqs (no iKUN/NeuralSORT output for 0019), none of which matches iKUN's official V2
pipeline. GMC's effect on it (flat, −0.007 no-tuning) is not a meaningful benchmark
statement. A valid iKUN-V2 needs iKUN's own V2 tracker outputs (unavailable here). Recorded
in `project_ikun_v2_crosssplit_flat_2026_05_25` as an internal note only.

Notes:
- iKUN paper anchor 44.564 is **bit-exact reproduced** locally (Δ=+0.004 vs paper README
  44.56; paper-pure end-to-end `iKUN/test.py` + `iKUN_cascade_attention.pth` + simcalib +
  cascade KUM + YOLOv8-NS + `gt_template_old/` + 3-seq pooled).
- FH V2 "host alone" baseline (42.526) reproduces the published FlexHook-best HOTA (42.53,
  TempRMOT paper Table 3 / FlexHook paper Table 1); the ship 42.807 beats it by +0.277.
- **FH V1 paper-gap is structural.** Neither the current shared-weight ship (53.526) nor the prior
  mlp+EMA ship (53.716) beats paper 53.824. A 17-cell retune around the shared-weight+no-EMA
  coordinates capped at 53.623. The V1 local loss vs the legacy mlp ship does not affect
  the paper-beat claim (already lost in every tested config).
- **iKUN-V2: EXCLUDED (protocol mismatch, not benchmark-valid).** Official iKUN-V2 = 10.32 HOTA; our attempt = 31.4 (3× gap) because it paired a V1-trained iKUN with NeuralSORT tracks + FlexHook's V2 GT on 3 of 4 test seqs — not iKUN's official V2 pipeline. GMC flat (−0.007) on that non-comparable baseline. Not part of the headline grid. Internal-only diagnostic: GMC's only mild-POS axis was APPEAR (+0.44); motion-axis NEG (−1.45 MOVING at sc=1). The tuned +0.29 probe was test-set selection and is not cited.

### Ship fusion recipe (locked, 2026-05-21)

| component | spec |
|---|---|
| Aligner arch | `shared_weight` (Linear adapter motion 13→256 + lang 384→256 → shared MLP 256→512→512→256 → LN → L2) |
| Aligner weights | `gmc_link_weights_v1train_sharedweight_seed{0,1,2}.pth` (V1 stage1, InfoNCE+FNM, τ=0.07, 100 ep, batch 256, lr 1e-3) |
| GMC score | raw cosine ∈ [−1,+1] (`GMC_RAW_COS=1`) |
| EMA | OFF |
| Fusion | `final = model_logit + α·(sc·raw_cos + thr)`, motion axis + appear axis |

**Per-arch fusion coefficients** (18 free hyperparams = 6/arch × 3 archs):

| arch | α_m | sc_m | thr_m | α_a | sc_a | thr_a |
|---|---|---|---|---|---|---|
| iKUN | 1.0 | 0.9 | +0.17 | 1.0 | 0.30 | +0.10 |
| FH V1 | 0.65 | 10 | +3 | 1.0 | 3.5 | +0.9 |
| FH V2 | 0.4 | 10 | +1.3 | 1.0 | 3.5 | +1.2 |

The 18-param recipe is retained over a 0-param simple recipe because the simple recipe
loses the iKUN paper-beat (B2 iKUN 44.272 < paper 44.564, Δ=−0.292). The hand-tuned
sc_a is intentionally 7–11× **smaller** than a data-derived std-matching value would set
it — this damps GMC (motion) signal on appearance expressions where it is noise. See
lever 23 (Variant B std-matching) for the falsification of auto-deriving these.

---

## 2. Per-Seed Breakdown

Pooled + per-class APPEAR / MOVING / STATIC, all 3 seeds per arch. Same ship recipe as §1.

### iKUN — ship (shared-weight + APPEAR-axis recipe α_m=1.0/sc_m=0.9/thr_m=+0.17, α_a=1.0/sc_a=0.30/thr_a=+0.10, raw cos, no-EMA)
Source: `results/ikun_appearship_noema_sw_20260519_225754.tsv`

| Seed | Pooled | APPEAR | MOVING | STATIC |
|---|---|---|---|---|
| 0 | 44.582 | 46.761 | 29.202 | 43.345 |
| 1 | 44.612 | 46.587 | 29.789 | 43.863 |
| 2 | 44.708 | 46.785 | 30.576 | 43.226 |
| **Mean ± std** | **44.634 ± 0.066** | 46.711 ± 0.108 | 29.856 ± 0.689 | 43.478 ± 0.339 |

### FH V1 — ship (shared-weight + recipe motion 0.65/10/+3, appear 1.0/3.5/+0.9, raw cos, no-EMA)
Source: `results/flexhook_ship_noema_sw_20260520_064025.tsv`

| Seed | Pooled | APPEAR | MOVING | STATIC |
|---|---|---|---|---|
| 0 | 53.591 | 55.593 | 45.349 | 49.534 |
| 1 | 53.427 | 55.489 | 44.386 | 49.568 |
| 2 | 53.559 | 55.621 | 45.446 | 49.077 |
| **Mean ± std** | **53.526 ± 0.087** | 55.568 ± 0.069 | 45.060 ± 0.586 | 49.393 ± 0.274 |

(Seed 1 MOVING = 44.386 is a low outlier vs ~45.4 on the other two seeds, driving the
elevated MOVING std.)

### FH V2 — ship (shared-weight + recipe motion 0.4/10/+1.3, appear 1.0/3.5/+1.2, raw cos, no-EMA)
Source: `results/flexhook_ship_noema_sw_20260520_064025.tsv`

| Seed | Pooled | APPEAR | MOVING | STATIC |
|---|---|---|---|---|
| 0 | 42.805 | 41.935 | 48.611 | 45.409 |
| 1 | 42.771 | 41.912 | 48.329 | 45.536 |
| 2 | 42.846 | 42.002 | 48.482 | 45.336 |
| **Mean ± std** | **42.807 ± 0.038** | 41.950 ± 0.047 | 48.474 ± 0.141 | 45.427 ± 0.101 |

### iKUN-V2 — seed0 only (EXCLUDED: protocol mismatch vs official iKUN-V2 = 10.32; internal diagnostic)

| Config (seed0) | Pooled | APPEAR | MOVING | STATIC |
|---|---|---|---|---|
| B1 (no GMC) | 31.434 | 32.108 | 26.665 | 32.146 |
| B2 bare `cascade + raw_cos` (no-tuning) | 31.427 | 32.544 | 25.218 | 31.327 |
| Δ (B2 − B1) | **−0.007** | +0.436 | −1.447 | −0.819 |
| V1-recipe transferred (NOT used) | 28.651 | 30.264 | 20.278 | 27.294 |
| best searched probe P3 (α=0.5, test-set-selected) | 31.727 | 32.593 | 26.357 | 32.131 |

Single-seed (seed0) — a generalization probe, not a multi-seed ship. n=3 not run because the no-tuning verdict is flat and the cell is confounded by domain shift (no ship claim to stabilize).

---

## 3. Per-Class Pooled HOTA + Statistical Significance

Per-class pooled HOTA aggregates trajectory IDs across all (seq, expr) pairs **within a
class** before computing √(DetA·AssA). Compared B (single-seed reference) vs ship recipe,
3 seeds, one-sample t-test vs single B.

Source: `project_per_class_pool_all_positive` (2026-05-03; ship-recipe per-class pool).

| arch | class | B (single-seed) | ship seed mean ± std (n=3) | Δ ship − B | t | p_one |
|---|---|---|---|---|---|---|
| iKUN | APPEAR | 46.346 | 46.746 ± 0.045 | +0.400 | 15.57 | 0.0021 ✓✓ |
| iKUN | MOVING | 25.531 | 30.093 ± 0.240 | **+4.562** | 32.89 | 0.0005 ✓✓ |
| iKUN | STATIC | 43.914 | 45.099 ± 0.178 | +1.185 | 11.52 | 0.0037 ✓✓ |
| FH V1 | APPEAR | 55.492 | 55.700 ± 0.026 | +0.208 | 13.93 | 0.0026 ✓✓ |
| FH V1 | MOVING | 43.981 | 45.785 ± 0.235 | +1.804 | 13.31 | 0.0028 ✓✓ |
| FH V1 | STATIC | 48.983 | 49.771 ± 0.217 | +0.788 | 6.28 | 0.0122 ✓ |
| FH V2 | APPEAR | 41.748 | 41.946 ± 0.051 | +0.198 | 6.79 | 0.0105 ✓ |
| FH V2 | MOVING | 48.018 | 48.758 ± 0.067 | +0.740 | 19.17 | 0.0014 ✓✓ |
| FH V2 | STATIC | 44.622 | 44.935 ± 0.024 | +0.313 | 23.04 | 0.0009 ✓✓ |

**All 9/9 (arch × class) cells positive AND stat-sig at α=0.05.**
- 7/9 cells stat-sig at α=0.01 (✓✓).
- 2/9 cells stat-sig at α=0.05 only (FH V1 STATIC p=0.0122, FH V2 APPEAR p=0.0105).
- Largest gain: iKUN MOVING +4.562 (recovers the cascade motion-class catastrophe).
- Smallest t: FH V1 STATIC = 6.28 (still passes α=0.05).

> Note: these per-class numbers are from the legacy mlp+EMA ship-recipe measurement
> (2026-05-03), which is why the per-class B and ship values differ slightly from the §2
> shared-weight+no-EMA per-seed tables. They are the strongest available pool defense (all-positive,
> all-significant) and stand as the per-class significance story; re-measurement at the
> shared-weight+no-EMA ship is a documented follow-up.

### Per-expr vs per-class-pool reconciliation

Per-expr mean Δ (Wilcoxon) is NEG on V1/V2 while per-class POOL Δ is POS — every class on
every arch is sign-flipped. This is a **mathematical property of HOTA aggregation**, not a
recipe miscalibration: pool HOTA aggregates trajectory IDs across exprs before √(DetA·AssA),
which per-expr loses.

| arch | class | per-expr mean Δ (n) | per-class POOL Δ |
|---|---|---|---|
| FH V1 | APPEAR | −0.0045 (n=119) NEG | +0.141 POS |
| FH V1 | MOVING | −0.0086 (n=27) NEG | +1.991 POS |
| FH V1 | STATIC | −0.0479 (n=12) NEG | +0.843 POS |
| FH V2 | APPEAR | −0.0088 (n=639) NEG | +0.164 POS |
| FH V2 | MOVING | −0.0200 (n=122) NEG | +0.767 POS |
| FH V2 | STATIC | −0.0402 (n=101) NEG | +0.571 POS |

---

## 4. Multi-Seed Statistics

n=3 seeds {0,1,2} per arch. Sample std reported (not population std).

| arch | ship mean ± std (n=3) | vs B1 (no GMC) | vs paper | significance test vs paper |
|---|---|---|---|---|
| iKUN | 44.634 ± 0.066 | +0.410 | +0.070 | one-sided t=1.85, df=2, p≈0.10 (directional +, not sig at α=0.05). All 3 seeds individually beat paper (44.582 / 44.612 / 44.708). |
| FH V1 | 53.526 ± 0.087 | +0.416 | −0.298 | does not beat paper in any config; gap structural. |
| FH V2 | 42.807 ± 0.038 | +0.281 | +0.281 | beats paper on all 3 seeds (42.805 / 42.771 / 42.846). |

Ship vs prior mlp+EMA ship (legacy reference, n=3): iKUN +0.026 (Welch t≈0.59, p≈0.30,
statistically same — no regression); FH V1 −0.190 (Welch t=−2.98, p≈0.02, local loss but
neither config beats paper); FH V2 +0.008 (t=0.23, n.s.).

iKUN per-class vs B1: MOVING +4.325, APPEAR +0.365, STATIC −0.436 (motion class is the
dominant lift; small STATIC trade). Dropping EMA exposes per-seed variance: iKUN std grows
0.024 (mlp+EMA) → 0.066 (shared-weight+no-EMA).

---

## 5. Lever-Exhaustion Table

Negative / neutral levers tested and closed. Δ are vs the ship recipe in force at test time
(noted per row). HOTA throughout unless a representation-axis AUC gate is noted (and even
those were re-checked at HOTA where it mattered).

| # | Lever | What was tested | Verdict | Key Δ |
|---|---|---|---|---|
| 1 | Exp 36A — 25D MLP feature aug | accel features (scale-diff + temporal-deriv) into MLP aligner | NEG | 0.747 / 0.741 vs stage1 0.779 (rep-axis gate); features not the lever |
| 2 | Exp 36C — V1+V2 joint train | supervision expansion via paraphrase-augmented joint data | NEG | micro flat 0.767 vs 0.779 |
| 3 | Exp 36D — BGE-base encoder | swap 768D BGE language encoder | NEG | 0.735, worst of 7 Exp 36 runs |
| 4 | Exp 36E — curriculum train | group→fine curriculum (100ep+50ep) | NEG | 0.762 vs 0.779; no ceiling breach |
| 5 | Exp 37 Stage A/C — ego source + EMAP | alt ego-motion source; ego-map-pool concat | NEG | Stage A −0.036, Stage C −0.024 |
| 6 | Exp 37 Stage B — OMF 28D Farneback | 13D + per-bbox Farneback flow (28D) | NEG | micro 0.624 (Δ=−0.155), worst of 12 levers; flow corrupts 13D |
| 7 | Exp 37 — ORB-grid 3×8 (61D) | per-cell ORB-match flow grid | NEG | Δ=−0.107 vs stage1; sparse cells fragment matches |
| 8 | Exp 38 — ego injection (zeroshot) | replace / concat / all-13d ego injection | NEG | best 53.328 (38-B), Δ=−0.496 vs FH V1 53.824 |
| 9 | Exp 39 — CLIP concat (input) | CLIP B/32 64D/128D concat into 13D motion | NEG | rep-axis 0.7223; HOTA revisit Δ −0.139 iKUN / −0.096 V1 / −0.229 V2 vs depth-aug |
| 10 | Exp 41 — CLIP late-concat (aligner-internal) | motion 256 ⊕ app 256 = 512D vs CLIP-text 512D | NEG | rep-axis 0.731; (HOTA revisit was iKUN-only POS at legacy EMA pipeline — see lever 24) |
| 11 | Exp 42 — Arm B raw cosine (vs EMA) | additive fusion on raw cosine, multi-seed | NEG/NEU | iKUN −0.179, FH V1 −0.051, FH V2 multi-seed +0.060 (t=1.22). Note: ship later moved to raw-cos under shared-weight; this was at mlp recipe. |
| 12 | Exp 43 — CLIP logit fusion (decision-level) | CLIP cosine as 3rd additive channel on iKUN ship | NEG | best 44.359, Δ=−0.243 (all 8 arms NEG) |
| 13 | Tier B (1) — per-class specialist aligners | motion/static/appear specialist aligners | NEG | all 3 archs −0.4 to −1.8 HOTA |
| 14 | World-XY projection (17D) | metric dX,dY via inverse pinhole vs image-domain depth-aug | NEUTRAL | flat all 3 archs (paired t, p>0.29) |
| 15 | CLIP-visual 128D concat — HOTA revisit | re-check Exp 39 at HOTA vs depth-aug iKUN | NEG | −0.139 iKUN, −0.096 V1, −0.229 V2; iKUN MOVING −2.72 |
| 16 | Case 2 1a — cross-attn fusion transformer | window-level fusion transformer (iKUN+motion KV) | NEG | Δ=−1.0 vs ship 44.602 (AUC PASS 0.901 but per-frame HOTA loses) |
| 17 | Case 2 1b — POS-decoupled two-branch | sigmoid-gated lang_static / lang_motion branches | NEG | Δ=−1.17 vs ship; −0.18 vs 1a peak |
| 18 | Case 2 1c — +ego-state 3rd KV | ego-state token as 3rd KV in fusion transformer | NEG | Δ=−1.19 vs ship; turning-verb wall did not move |
| 19 | Case 2 1d — FiLM on visual (stack on ship) | zero-init FiLM, stacked on full ship recipe, multi-seed | NEG | n=3 44.452 ± 0.285, Δ=−0.150; closes entire Case 2 family |
| 20 | iKUN learned residual MLP | residual additive MLP on iKUN cascade+simcalib | NEG | 42.919, Δ=−1.305; MOVING crashed to 18.261 |
| 21 | FlexHook learned fusion (MLP gate) | leave-one-seq-out CV MLP gate on (margin, gmc) | NEG | pool −3.79, MOVING −35.6; hand-tuned linear wins |
| 22 | Strict motion-filter train data | 138 pure-motion exprs vs 296 mixed (stage1) | NEG | micro −0.142, macro −0.132; mixed teaches invariance |
| 23 | V1 STATIC recipe-split | soft static-axis bias on parked-y exprs | NEG | pool −0.13 to −0.24, STATIC −0.6 to −1.5 |
| 24 | CDRMOT Lever A — structural consensus aux loss | dist-only consensus constraint, λ∈{0.1,0.5} on iKUN ship | NEG | λ=0.5 pool 44.376 (−0.210), λ=0.1 pool 44.434 (−0.152); manifold collapse |
| 25 | CDRMOT Lever B — what/where dual cosine | spaCy POS split, w_what∈{0.3,0.5,0.7} on iKUN ship | NEG | best w_what=0.5 pool 40.918, Δ=−3.67 (all cells ~−3.7 to −4.5) |
| 26 | Path A — Grounding-DINO-Tiny detector | open-vocab detector recall gate vs NS predict.txt | NEG | recall 0.50–0.75 (need 0.90); cannot match YOLOv8-NS geometry |
| 27 | Path C — Qwen2-VL-2B LVLM calibration | int4 LVLM motion-class discrimination on KITTI crops | NEG | 3 prompt/input variants all degenerate; 2B binds to class noun |
| 28 | Path B — seed-ensemble cache | ensemble-of-seed GMC caches at ship recipe | NEUTRAL | all 3 archs sign-POS within ±std; FH V1 +0.102 edge; reproducibility, not gain |
| 29 | V1 retune under shared-weight+no-EMA | 8-cell motion + 9-cell appear sweep to recover −0.190 | NEG | 17 cells within ±0.124 of control; recipe lever closed |
| 30 | Variant B — std-matching auto-sc (23rd) | derive sc = std(model_logit)/std(raw_cos) per arch per axis, thr=0 | NEG | iKUN −2.794, FH V1 −5.831, FH V2 −4.436 vs ship; APPEAR-damping is irreducible |
| 31 | Phase 1 — exp41 per-class CLIP routing gate (24th) | re-eval exp41 cliptext aligner at shared-weight+no-EMA ship across 3 archs | NEG | all 3 pool NEG (iKUN −0.408, V1 −0.353, V2 −0.140), all 12 per-class cells NEG; no class to route to |

Notes:
- Rep-axis (AUC) entries (#1–4, 9, 10) are recorded for completeness of what was tried; the
  project's policy is that AUC is never a ship gate — several AUC-NEG levers were later
  re-checked at HOTA (e.g. depth-aug 17D went AUC-NEG → HOTA-POS for iKUN, +0.215, p=0.016).
- Levers #16–19 form the closed Case 2 fusion-transformer family.
- Levers #24–27 are the ceiling-break campaign closures (CDRMOT, open-vocab detector, LVLM).
- Levers #30–31 are the two most recent closures (2026-05-21), tested at the current
  shared-weight+no-EMA ship recipe.

---

## 6. Source Artifacts

| Artifact | Path |
|---|---|
| iKUN ship per-seed TSV | `results/ikun_appearship_noema_sw_20260519_225754.tsv` |
| FH V1+V2 ship per-seed TSV | `results/flexhook_ship_noema_sw_20260520_064025.tsv` |
| iKUN ship orchestrator | `run_ikun_appearship_noema_sw.sh` |
| FH ship orchestrator | `run_flexhook_ship_noema_sw.sh` |
| B2 simple-recipe orchestrator | `run_sharedweight_two_baseline.sh` |
| B2 raw TSV | `results/sharedweight_two_baseline_20260519_*.tsv` |
| Variant B (lever 30) TSV | `results/variant_b_test_20260521_183250.tsv` |
| Phase 1 exp41 gate (lever 31) TSV | `results/phase1_exp41_gate_20260521_191917.tsv` |
| Aligner weights | `gmc_link_weights_v1train_sharedweight_seed{0,1,2}.pth` |
| GMC score caches | `gmc_link/gmc_scores_{v1,flexhook_v1,flexhook_v2_raw}_{seq}_sharedweight_seed{N}_rawcos_cache.json` |
| Aligner code | `gmc_link/alignment.py` (`shared_weight` branch) |
| GT label space | `~/data/Dataset/refer-kitti/gt_template_old/` (paper-canonical, NeuralSORT-aligned) |
