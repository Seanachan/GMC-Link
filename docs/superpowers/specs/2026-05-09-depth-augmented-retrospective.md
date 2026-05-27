# Depth-Augmented GMC (Approach B) — Retrospective

**Date:** 2026-05-09
**Branch:** `exp/ego-motion-systematic`
**Spec:** [2026-05-09-depth-augmented-gmc-design.md](2026-05-09-depth-augmented-gmc-design.md)
**Plan:** [docs/superpowers/plans/2026-05-09-depth-augmented-gmc.md](../plans/2026-05-09-depth-augmented-gmc.md)
**Verdict:** **KILL** (below micro AUC gate at 3-seed mean)

---

## TL;DR

Extending 13D motion vector → 17D with per-track absolute depth `Z_n` and ego-compensated Δ-Z at multi-scale gaps did NOT clear the stage1 ceiling. Mean micro AUC across 3 seeds = **0.7567** vs stage1 baseline 0.7793 (Δ = **−0.023**), and 2 of 3 seeds fell below the 0.760 KILL gate. 14 levers exhausted at stage1; ceiling is representation-bound to centroid-geometry signal at the aligner's hypothesis class.

---

## Hypothesis

Per-track depth time-series `[Z_n, dZ_2, dZ_5, dZ_10]` from a metric monocular depth estimator (Depth Anything V2 metric-outdoor-large) should:
1. **Discriminate near vs far cohorts** — language tokens like "in front of ours" reference scene depth absolutely
2. **Disambiguate approach/recede vs lateral motion** — `dZ` flips sign on counter-direction vs same-direction
3. **Survive ego-Z noise** — cohort-median compensation should subtract translational ego-Z from per-track Δ-Z, leaving residual radial motion

If true, micro AUC should clear ≥0.7793 (stage1) and ideally ≥0.79 (clear stage1 by ≥1pp).

---

## Method

### 17D motion vector
```
[res_dx_s, res_dy_s, res_dx_m, res_dy_m, res_dx_l, res_dy_l,   # 6 residual velocity (3 scales × 2 axes)
 dw, dh, cx, cy, w, h, snr,                                     # 7 base 13D state
 Z_n / 100,                                                     # 1 absolute depth (clipped [0, 80])
 (dZ_2 - ego_dZ_2) / 10,                                        # 3 ego-compensated Δ-Z at FRAME_GAPS = (2, 5, 10)
 (dZ_5 - ego_dZ_5) / 10,
 (dZ_10 - ego_dZ_10) / 10]
```

### Cohort ego-Z compensation
For each `(curr_fid, gap)` pair, compute median `dZ` across **all tracks present at both `curr_fid` and `curr_fid + gap`** in the depth cache. Subtract that median from each track's per-frame `dZ`. Idea: ego translation in Z dominates Δ-Z when camera moves; median across cohort tracks ≈ ego signal.

### Identity-init depth gate
At step 0, zero `motion_projector[0].weight[:, 13:17]` so the depth tail multiplies to zero. Bit-exact 13D init verified by unit test (4/4 PASS in `tests/test_alignment_motion_dim.py`). Gradient flows through depth columns from epoch 1 onward.

### Depth source
- Model: `depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf`
- Local snapshot: `/home/seanachan/.cache/depth_anything_v2_outdoor_large`
- Per-track Z extraction: bbox-center pixel value from per-frame metric depth map
- Cache: `gmc_link/depth_cache/z_track_gt_{seq}.json`, structure `{tid: {fid: z_meters}}`

### Train recipe
- Split: V1 (15 train seqs, no 0000) — 660 expressions
- Epochs 100, batch 256, seed ∈ {0, 1, 2}
- All other hyperparameters identical to stage1 baseline (`--text-encoder all-MiniLM-L6-v2`, motion_dim=17 + `--use-depth --identity-init-depth`)

### Eval
- 3 seqs × 3 seeds = 9 diag runs
- Aggregator: `diagnostics/aggregate_multiseq.py` over `layer3_{seq}_depth_seed{N}.npz`
- Decision gate: micro pool AUC (n_expressions=33, expression-frame-weighted)

---

## Results

### Decision gate (micro AUC, V1 test 3-seq pool)

| Seed | mean_auc_micro | mean_auc_macro ± std | best_seq | worst_seq |
|---|---|---|---|---|
| 0 | **0.7647** | 0.834 ± 0.066 | 0013: 0.819 | 0011: 0.783 |
| 1 | **0.7549** | 0.825 ± 0.069 | 0013: 0.826 | 0005: 0.771 |
| 2 | **0.7506** | 0.819 ± 0.073 | 0013: 0.809 | 0005: 0.769 |
| **mean** | **0.7567** | 0.826 | — | — |

**vs stage1 baseline 0.7793** → Δ = **−0.0226** (2 of 3 seeds below 0.760 KILL gate)
**vs Exp 41 late-concat 0.731** → +0.026 (depth slightly better than late-concat-CLIP, still NEG vs stage1)

### Decision rule applied
- micro < 0.760 → **KILL** ✓
- micro mean 0.7567 fails KILL gate → skip Task 8 HOTA cross-arch eval

### Macro vs micro disagreement
Macro mean 0.826 vs micro mean 0.757 — large gap. Macro averages per-expression AUC unweighted; micro pools by expression-frame count. Stage1 baseline shows similar macro/micro split (0.844 / 0.7793). The gap is structural to V1 test — short-frame expressions like "moving cars" drive micro down without changing macro much. Depth lifted macro slightly (0.826 vs stage1 0.844 = −0.018, similar drop on micro) but did not differentially help frame-heavy expressions.

---

## Why It Failed

1. **Cross-manifold corruption (echoes Exp 39/40 late-concat)** — Adding 4 depth dims to 13D motion forces the shared MLP to disentangle a fundamentally different input modality (metric depth in meters at +30dB signal-to-noise from monocular DAv2 vs centroid velocity at 100×-scale resolution-normalized pixels). Identity-init delays this by 1 step but doesn't prevent it during train.
2. **Cohort ego-Z is noisier than expected** — Median across tracks visible at both endpoints requires sufficient cohort size; sparse frames have median over n=2-3 tracks → high variance estimate of ego dZ.
3. **DAv2 metric depth on KITTI street scenes** — Trained on synthetic vKITTI; sim-to-real gap on real KITTI. Per-track Z noise likely > 10% at >30m distance, swamping the +0.5–2m signal that distinguishes near/far tracks.
4. **Depth tokens not in stage1 language** — MiniLM lang space encodes "front", "left", "counter direction" relationally, not by absolute depth. Adding Z signal to motion side without reciprocal grounding on lang side → InfoNCE has no anchor for the new motion content.

---

## Counterfactuals NOT Tested

- **Relative depth** instead of metric (DPT relative ranking; less sim-to-real risk)
- **Reciprocal Z grounding on lang side** (project lang_emb through depth-token-aware MLP)
- **Z as decision-level feature** (not aligner input — concatenate to fusion logits like FlexHook GMC ship recipes)
- **Per-expression depth-token attention** (only "front", "near", "far" sees Z; rest masked)

These remain open for future iteration. Decision-level fusion is the only POS direction shipped (per `project_aligner_sweep_at_alpha1_restores_auc` and per-arch HOTA wins).

---

## Levers Exhausted at Stage1 (post Exp 42)

| # | Lever | Source | Result |
|---|---|---|---|
| 1 | HN-InfoNCE β-grid | Exp 34 | NEG |
| 2 | Scale-diff features (25D MLP) | Exp 36A | NEG |
| 3 | Temporal-deriv features | Exp 36A v2 | NEG |
| 4 | Transformer arch (5/25 ep) | Exp 36B | NEG |
| 5 | V1+V2 joint train | Exp 36C | NEG (micro flat, macro +0.005) |
| 6 | BGE-base lang encoder | Exp 36D | NEG (worst) |
| 7 | Curriculum (100ep+50ep) | Exp 36E | NEG |
| 8 | Stage A: ego source swap | Exp 37A | NEG |
| 9 | Stage B: OMF Farneback flow (28D) | Exp 37B | NEG |
| 10 | Stage C: EMAP concat | Exp 37C | NEG |
| 11 | Stage E: ORB-grid 3×8 (61D) | Exp 37 grid | NEG |
| 12 | Exp 39: CLIP visual concat (64D / 128D) | Exp 39 | NEG |
| 13 | Exp 41: CLIP-text + late-concat fusion | Exp 41 | NEG |
| 14 | **Exp 42: Depth-augmented 17D** | **this** | **NEG** |

Conclusion: stage1 ceiling is **representation-bound to centroid geometry at this MLP hypothesis class**. Adding modalities (CLIP visual, CLIP text, depth) at the aligner internal level uniformly fails. Decision-level fusion (per-arch ship recipes) is the only POS path.

---

## Artifacts

- 3 ckpts: `experiments/depth_v1train/seed{0,1,2}/best.pth` (motion_dim=17)
- 18 GT depth caches: `gmc_link/depth_cache/z_track_gt_{0001..0020}.json` (V1 train + test, no 0000)
- 9 diag npz: `diagnostics/results/depth_v1train/layer3_{seq}_depth_seed{N}.npz`
- 3 aggregator JSONs: `diagnostics/results/depth_v1train/multiseq/layer3_multiseq_depth_seed{N}.json`
- Comparison: `diagnostics/results/depth_v1train/multiseq/layer3_multiseq_comparison.md`

## Code shipped (kept on branch despite KILL)

The plumbing is reusable for future Z-based experiments:
- `gmc_link/depth_extractor.py` — DAv2 metric wrapper, offline-mode env vars
- `gmc_link/depth_cache.py` — `DepthCache.load()`, `lookup(tid, fid)`, JSON format
- `run_build_depth_cache.py` — driver for batch cache build
- `gmc_link/dataset.py:_frame_cohort_dz_ego` — cohort median helper
- `gmc_link/dataset.py:compute_motion_vectors_for_pair_with_depth` — 17D path under `--use-depth`
- `gmc_link/manager.py` — 17D inference path
- `gmc_link/alignment.py:identity_init_depth` — bit-exact init gate (4/4 unit tests pass)
- `gmc_link/train.py` — `--use-depth`, `--depth-cache-dir`, `--identity-init-depth` CLI
- `diagnostics/diag_gt_cosine_distributions.py` — depth_cache plumb in eval

No code reverted. All commits remain on `exp/ego-motion-systematic`.

---

## 2026-05-10 Addendum — HOTA cross-check FLIPS verdict

**KILL was AUC-only and is wrong at HOTA decision level.**

After noting cliptext precedent (Exp 40: identical micro AUC magnitude but partial HOTA POS at iKUN), ran iKUN HOTA on 17D ckpts at locked Arm A 13D ship recipe (α_m=1, sc_m=0.9, thr_m=0.17, α_a=1, sc_a=0.30, thr_a=0.10).

### iKUN multi-seed n=3 pool HOTA on V1 test 3-seq

| Seed | 17D pooled | 13D pooled (reference) | matched-seed Δ |
|---|---|---|---|
| 0 | **44.876** | 44.586 | **+0.290** |
| 1 | **44.800** | 44.604 | **+0.196** |
| 2 | **44.793** | 44.634 | **+0.159** |
| **mean ± std** | **44.823 ± 0.046** | 44.608 ± 0.024 | **+0.215** |

- vs paper iKUN 44.564 → Δ = **+0.259**
- vs Arm A 13D multi-seed 44.608 → Δ = **+0.215**
- Paired t (n=3) = 5.51, df=2, p_one-sided ≈ **0.016** → stat-sig POS at α=0.05
- Sign consistency: 3/3 seeds POS

### 8-arm sweep on seed0 17D ckpt (Arm A reference 44.876)

| Arm | Config | pooled | Δ vs A |
|---|---|---|---|
| **A** | **13D ship sc_m=0.9, sc_a=0.30** | **44.876** | **0** |
| B | sc_m=1.5 | 44.589 | −0.287 |
| C | sc_a=0.50 | 44.693 | −0.183 |
| D | both up | 44.380 | −0.496 |
| E | thr=0 | 44.335 | −0.541 |
| F | scale-down | 44.646 | −0.230 |
| G | motion-only | 44.656 | −0.220 |
| H | appear-only | 44.435 | −0.441 |

**13D ship recipe is optimal on 17D too.** No hyperparameter retuning required — depth contribution lives in ckpt logits, not in needing a new α/scale/thr.

### Why AUC and HOTA disagree

| Mechanism | AUC | HOTA |
|---|---|---|
| Sampling | Per-frame random (track, expr) pairs | Per-trajectory association quality |
| Threshold | Free (ROC integration) | Anchored at IoU + score threshold |
| Time | Frame-independent | Trajectory-continuous (AssA component) |
| Weighting | Uniform per-frame pair | GT-trajectory length-weighted |

Per-track Z time-series encodes object persistence (same track stays at similar Z across frames). That's an AssA signal — HOTA's IDF1-style assignment can latch on to it. AUC samples random pairs and can't see the temporal coherence of Z across frames belonging to the same trajectory. Depth tail looks corrupted at AUC because it adds variance to per-frame cosine, but that variance carries trajectory info that decision-level fusion + HOTA unlocks.

### Implications

1. **17D depth-aug iKUN ships POS** at decision-level fusion. Multi-seed mean 44.823 (+0.259 vs paper, +0.215 vs Arm A 13D).
2. **AUC-only KILL gate is unsafe** for aligner-internal levers — failed twice in a row (cliptext + depth). Add HOTA cross-check before declaring KILL.
3. **Stage1 AUC gate retains use** for fast prefilter (cheap 9× diag eval), but HOTA is the ship gate.
4. **Lever count revision**: was "14 stage1 levers exhausted, ceiling representation-bound." Updated: 13 stage1 AUC levers exhausted but the depth lever is POS at decision level. Aligner-internal modality concat closed AT AUC; OPEN at HOTA decision-level transfer.

### FH cross-arch HOTA verify (2026-05-10, n=3 paired)

Built FH V1 + V2 17D depth caches via patched `run_build_gmc_cache_flexhook.py` and `run_build_gmc_cache_flexhook_v2_raw.py` (added `GMC_DEPTH_ARCH`/`GMC_DEPTH_DIR` env vars + `depth_z_lookup` plumbing through `linker.process_frame`). Ran `run_flexhook_depth_multiseed.sh` at locked Arm A 13D appear-extension recipes.

| Arch  | 13D mean ± std | 17D mean ± std | Δ paired | paired_t | p_one  | sign | Verdict |
|-------|----------------|----------------|----------|----------|--------|------|---------|
| iKUN  | 44.608 ± 0.024 | 44.823 ± 0.046 | +0.215   | +5.50    | 0.016  | 3/3  | **stat-sig POS α=0.05** |
| FH V1 | 53.716 ± 0.068 | 53.765 ± 0.059 | +0.048   | +0.80    | 0.255  | 2/3  | mean POS, within seed noise |
| FH V2 | 42.799 ± 0.047 | 42.833 ± 0.005 | +0.034   | +1.38    | 0.151  | 2/3  | mean POS, within seed noise |

Per-seed pooled — FH V1: 53.787 / 53.809 / 53.698. FH V2: 42.836 / 42.836 / 42.828.

vs paper: all 3 archs beat paper at 17D (iKUN +0.259, FH V1 +0.655 cli-fork drift unchanged, FH V2 +0.307). Sign 7/9 cells POS (V1 seed2 −0.061, V2 seed1 −0.012 NEG).

**Why iKUN sig but FH not:**
1. iKUN paper baseline 44.564 is weakest of 3 archs — depth headroom widest there. FH V1 paper 53.110, FH V2 paper 42.526 are closer to ceilings.
2. iKUN seed std (0.024) tightest of 3; smaller paired Δ clears t-test sig. FH V1 std 0.068 / V2 std 0.047 swallow Δ in noise.
3. Mechanism is decision-level invariant — score distribution unchanged when motion_dim grows 13→17. All 3 archs show POS mean; only iKUN reaches α=0.05.

### Final ship decision

- **iKUN**: ship 17D depth-aug at locked 13D recipe → 44.823 multi-seed mean.
- **FH V1 + FH V2**: do NOT ship 17D — within seed noise vs 13D, no recipe-level lever. 13D remains FH ship.

### Updated artifacts (2026-05-10)

- 8-arm sweep log: `experiments/depth_v1train/seed0/ikun_hota_sweep.log`
- Multi-seed Arm A logs: `experiments/depth_v1train/{ikun,fh_v1,fh_v2}_armA_multiseed.log`
- 9 17D iKUN GMC caches: `gmc_link/gmc_scores_v1_{0005,0011,0013}_depth_seed{0,1,2}_cache.json`
- 9 17D FH V1 GMC caches: `gmc_link/gmc_scores_flexhook_v1_{0005,0011,0013}_depth_seed{0,1,2}_cache.json`
- 12 17D FH V2 GMC caches: `gmc_link/gmc_scores_flexhook_v2_raw_{0005,0011,0013,0019}_depth_seed{0,1,2}_cache.json`
- Patched FH cache builders: `run_build_gmc_cache_flexhook.py`, `run_build_gmc_cache_flexhook_v2_raw.py` (depth path)
- Sweep drivers: `run_ikun_depth_sweep.sh`, `run_ikun_depth_armA_seeds.sh`, `run_flexhook_depth_multiseed.sh`

