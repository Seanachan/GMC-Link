# Path B — Complete Result Report (2026-05-28)

**Goal:** push RMOT SOTA on Refer-KITTI by replacing the noisy monocular ego-motion
signal (DAv2 depth + stationary-cohort ego-Z) with GT-quality signal (Velodyne LiDAR
depth + oxts GPS/IMU ego-Z) in the 17D motion vector. Single-variable: feature structure
(17D), model (shared_weight), recipe, eval all unchanged — only the depth-axis signal
source flips. Host = iKUN, V1 3-seq pooled HOTA, locked 13D-ship fusion recipe.

---

## 1. Validation gates passed (pre-training)
| Gate | Result |
|---|---|
| Phase-0 mechanism (2×2, label_02 depth) | GT+oxts 0.048m vs GT+cohort 0.312m → **6.47×** static-residual cohort artifact confirmed |
| Phase-1 depth-half sanity (velodyne) | holes **0.4%**, LiDAR/DAv2 ratio 1.03, static \|dZ−ego\| 0.099m → **policy(a) single run** |
| Frame-alignment unit test (n=1361 static) | offset sweep **off0 minimal** (0.196 vs 0.223/0.204); ego-comp 1.902→0.196m (~10×) → convention proven, off-by-one guarded |
| Monocular reproduce (NEW code, end-to-end) | **44.584 / 42.605 / 30.869 = R exact** (bit-identical 3 dp) → no cache/fuse/eval drift; attribution anchor solid |

## 2. Design — 3 settings × n=3 seeds
- **A = lidar_oxts** (full Path B: LiDAR depth + oxts ego)
- **B = lidar_cohort** (ego ablation: LiDAR depth + cohort ego)
- **M = monocular** (control: DAv2 depth + cohort ego)

## 3. Raw per-seed (pooled / MOVING / STATIC / APPEAR HOTA)
```
A lidar_oxts    s0 45.020 32.263 43.921 46.768
                s1 44.952 31.506 44.126 46.721
                s2 44.785 30.850 44.019 46.609
B lidar_cohort  s0 44.742 31.196 42.870 46.782
                s1 44.710 30.679 43.371 46.670
                s2 44.598 30.263 43.460 46.600
M monocular     s0 44.584 30.869 42.605 46.713
                s1 44.609 30.147 43.014 46.688
                s2 44.489 29.574 43.039 46.665
```

## 4. Means ± std (n=3)
| Setting | pooled | MOVING | STATIC | APPEAR |
|---|---|---|---|---|
| **A lidar_oxts** | **44.919** ±0.121 | **31.540** ±0.707 | **44.022** ±0.103 | 46.699 ±0.082 |
| B lidar_cohort | 44.683 ±0.076 | 30.713 ±0.467 | 43.234 ±0.318 | 46.684 ±0.092 |
| M monocular | 44.561 ±0.063 | 30.197 ±0.649 | 42.886 ±0.244 | 46.689 ±0.024 |
| 13D ship (ref) | 44.634 (n3) | 28.885 (s0) | 43.240 (s0) | — |

## 5. Attribution decomposition (paired per-seed; df=2, t_crit=4.303 @ α=0.05)
| effect | pooled | MOVING | STATIC | APPEAR |
|---|---|---|---|---|
| **EGO** (A−B) | +0.236 (t8.9)✓ | **+0.827** (t6.0)✓ | **+0.788** (t5.5)✓ | +0.015 (t0.8) ns |
| **DEPTH** (B−M) | +0.123 (t6.9)✓ | +0.516 (t4.9)✓ | +0.348 (t7.7)✓ | −0.005 ns |
| **TOTAL** (A−M) | +0.358 (t8.7)✓ | +1.343 (t38)✓ | +1.136 (t12)✓ | +0.011 ns |

Both depth and ego effects real (all sig); **ego > depth on every metric**. APPEAR flat (correct — motion module).

## 6. Path B (A) vs deployed 13D ship
- pooled **+0.285** · MOVING **+2.655** · STATIC **+0.782** · APPEAR flat.
- vs paper iKUN 44.564: **+0.355**. Best iKUN result to date.
- **STATIC regression fixed**: monocular 17D depth-aug rejected for STATIC 42.886 < 43.2 guardrail (the −0.47).
  A restores STATIC to 44.022 (≥43.2 ✓, +0.78 vs ship). Phase-0 STATIC mechanism confirmed at HOTA.

## 7. Verdict → **Row 1: POS, ego compensation IS the lever; paper headline holds**
- A STATIC 44.022 ≥ 43.2 ✓ (improved, not just met).
- A−B ≥ +0.5: MOVING +0.827 ✓, STATIC +0.788 ✓ (pooled +0.236 under +0.5 but sig; pooled 73%-APPEAR-diluted).
- A MOVING ↑ ≥ +1.5: vs ship **+2.66** ✓ / vs monocular +1.34 (just under — baseline-dependent).
- A ≉ B on MOVING/STATIC → **not** row 2 (depth-only).

## 8. Caveats
- **Single host (iKUN).** Cross-host FH V1/V2 not yet run — required before SOTA/generalization claim.
  iKUN historically gains most from GMC (motion-blind host); FH may be neutral-to-marginal.
- **Static-residual diagnostic under-predicted ego value** (oxts≈cohort 0.196≈0.219m on parked-car magnitude);
  HOTA overturned it — ego is the bigger lever. Proxy necessary-not-sufficient; HOTA adjudicates.
- **MOVING-↑ baseline**: +2.66 (vs ship) vs +1.34 (vs monocular) — choice anchors the headline.
- **pooled gain modest (+0.285)**: expected — pooled APPEARANCE-dominated; motion story is the MOVING slice (+2.66).
- n=3, single split (V1). Implementation uncommitted on `exp/pathB-lidar-17d`.

## 9. Reproduce
- Driver: `run_pathB_AB_n3.sh`. Caches: `gmc_link/depth_cache_lidar/z_track_lidar_{gt,ikun}_*.json`.
- Weights: `gmc_link_weights_v1train_sw_{lidaroxts,lidarcohort}_seed{0,1,2}.pth`.
- Flag: `--use-depth --depth-source {lidar_oxts,lidar_cohort} --depth-cache-dir gmc_link/depth_cache_lidar`.
- Frame-align test: `tests/test_pathB_oxts_frame_alignment.py`. Code: `gmc_link/kitti_tracking_gt.py:ego_dz_camera/ego_dz_between/seq_poses_calib`.

## 10. Awaiting decision (nothing fires)
1. MOVING-↑ headline baseline (ship vs monocular).
2. Cross-host FH V1/V2 — go / no-go.
3. Commit the Path B implementation.
