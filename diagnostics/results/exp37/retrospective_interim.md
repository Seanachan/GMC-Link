# Exp 37 Interim Retrospective — Stages A & C settled; B & D pending

Date: 2026-04-22
Spec: `docs/superpowers/specs/2026-04-22-ego-motion-systematic-design.md`
Plan: `docs/superpowers/plans/2026-04-22-exp37-ego-motion-systematic.md`

## Status snapshot

| Stage | Hypothesis | Status | ΔAUC vs C1 baseline | Decision |
|---|---|---|---|---|
| A | ORB vs recoverPose ego source | **Executed** | −0.036 micro | **NEGATIVE** — ORB wins |
| B | 13D vs 13D + OMF (28D) | **Deferred** | — | Pending OMF cache build + dataset plumb |
| C | Structural EMAP ego-velocity concat | **Executed** | −0.024 micro | **NEGATIVE** |
| D | TempRMOT cross-tracker portability | **Blocked** | — | Gated on user consent for submodule + pip install |

Baseline C1 = `gmc_link_weights_v1train_stage1.pth` (micro AUC 0.779, the
Exp 36e representation-bound ceiling memory confirms as the current
pipeline-bound ceiling).

## Stages A and C confirm representation bound

Both aligner-side levers regress rather than lift the 0.779 ceiling:

- **Stage A** (`stage_a_memo.md`): recoverPose 5-point essential-matrix
  ego estimation is structurally shakier on KITTI — foreground motion
  contaminates RANSAC inliers more than ORB's planar-homography RANSAC
  because the essential matrix has 5 DoF vs homography's 3 DoF. Per-seq
  std roughly doubles (0.064 → 0.121).

- **Stage C** (`stage_c_memo.md`): EMAP-style ego-velocity concatenation
  is redundant with the residual velocities already encoded as
  `res_v_k = raw_v_k − ego_v_k` in the base 13D vector. Appending
  absolute `(ego_dx_m, ego_dy_m)` widens the aligner's input surface
  without adding new geometric information.

Combined with the eight Exp 36 levers exhausted in `project_exp36e_curriculum_negative`
(features F1-F9, transformer arch, V1+V2 joint, BGE encoder, curriculum),
and now Stage A (ego source) + Stage C (structural conditioning),
**ten independent levers on 13D centroid geometry have all failed to lift
the 0.779 ceiling.** The ceiling is clearly representation-bound at this point;
no aligner-quality lever that stays inside centroid geometry is likely to move it.

## Stage B deferment rationale

Stage B tests the 28D OMF lever. Two reasons the execution was deferred
in favor of writing the Stage C memo and flagging Stage D:

1. **Expected outcome under the representation-bound hypothesis:** OMF
   adds per-bbox dense-flow statistics. Given the ten prior levers
   failed and the pattern suggests centroid-based supervision is the
   bottleneck (not feature richness), Stage B's prior on lifting AUC is low.
2. **Infrastructure cost:** OMF requires Farneback precompute (~30-60 min
   over 15 V1 seqs × 3 FRAME_GAPS) plus a dataset-side integration to
   actually load the cache and feed `omf_stats_per_scale` through
   `compute_per_track_extras`. Currently that path is zero-filled — the
   feature is registered but silently no-ops.

If Stage B is revived, the integration steps are:
- Run `run_exp37_stage_b.sh` OMF precompute heredoc (builds `cache/omf/orb/<seq>/<fid>_gap<N>.npz`)
- Patch `gmc_link/dataset.py:build_training_data` to load the per-(seq,fid,gap)
  cache and pass `omf_stats_per_scale=[per_bbox_omf_stats(flow, bbox) for flow in …]`
- Re-run training, eval, write stage_b_memo.md

## Stage D consent gate

Per spec §10 kill switch #2, Stage D (portability to TempRMOT) is now the
sole surviving Exp 37 publishable headline. Execution blocked on shared-state
changes that require explicit user approval per project rules:

1. `git submodule add https://github.com/zyn213/TempRMOT third_party/TempRMOT`
   (modifies `.gitmodules`, adds tracked directory from unvetted third-party code)
2. `pip install -e third_party/TempRMOT` (modifies RMOT conda env, harder to roll back)
3. Any conda-env dependency conflicts would activate spec §10 kill switch #3
   ("portability scope deferred") and narrow Exp 37 to A/B/C + ATE only.

**User action needed:** confirm Stage D acquisition to proceed, or decide to
narrow scope to A/B/C.

## Fix committed en route

Two parallel whitelist bugs caught during Stage C execution:

- `gmc_link/dataset.py:build_training_data` per-track feature whitelist was
  missing `ego_velocity_concat` and `omf_stats` — feature-dispatch silently
  dropped them before compute_per_track_extras ran, leaving the motion
  tensor 13D while the aligner expected 15D/28D.
- `diagnostics/diag_gt_cosine_distributions.py` had the same whitelist bug
  on the inference side.

Both fixed in commits `c938092` and `05e5ce0` on `exp/ego-motion-systematic`
branch. Pre-fix `57101dcd362ad422.npz` cache was discarded as part of the fix.
