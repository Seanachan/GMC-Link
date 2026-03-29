# Change Notes (feature/3d-projection)

## What Was Changed

### 1) `gmc_link/manager.py`
- Added pseudo-3D projection parameters in `GMCLinkManager.__init__`:
  - `fx`, `cx`, `x_range`, `z_min`, `z_max`
- Added helper methods:
  - `_estimate_depth_from_bbox(bbox_h)`
  - `_pixel_to_xz(u, z)`
  - `_normalize_xz(X, Z)`
- Updated motion feature construction pipeline:
  - Keep homography-based ego-motion compensation (`warp_points` on centroid history).
  - Convert compensated 2D centers to projected `X/Z`.
  - Compute multi-scale motion from `XZ` deltas (short/mid/long gaps).
  - Replace spatial slots in 13D vector from `(cx_n, cy_n)` to `(X_n, Z_n)`.
- Added concise English comments around key 3D projection logic.

### 2) `gmc_link/dataset.py`
- Added projection constants/helpers to mirror manager behavior:
  - `FX`, `CX`, `X_RANGE`, `Z_MIN`, `Z_MAX`
  - `_estimate_depth_from_bbox`, `_pixel_to_xz`, `_normalize_xz`
- Updated `_compute_velocity_at_gap(...)`:
  - Returns projected `(dX, dZ, bg_residual_xz)` semantics.
- Updated positive-pair generation:
  - 13D vector now uses projected position terms `(X_n, Z_n)` instead of `(cx_n, cy_n)`.
  - Kept vector length as 13 to stay compatible with current aligner shape.

### 3) `README.md`
- Rewritten to match current branch behavior and 13D XZ-projection semantics.
- Removed stale mixed descriptions and added focused usage + limitations.

## Current 13D Feature (Now)

`[dX_s, dZ_s, dX_m, dZ_m, dX_l, dZ_l, dw, dh, X_n, Z_n, w_n, h_n, snr]`

## Validation Done
- Syntax check passed:
  - `python -m py_compile gmc_link/manager.py gmc_link/dataset.py gmc_link/alignment.py gmc_link/train.py`

## What Still Needs To Be Done

### Priority A (before claiming performance gains)
1. Retrain model weights using updated dataset semantics (train/infer feature definition changed).
2. Re-run evaluation (`demo_inference.py` / your benchmark scripts) and compare against baseline.
3. Verify score distributions (GT vs non-GT) did not collapse after projection update.

### Priority B (quality improvements)
1. Replace bbox-height depth heuristic with stronger depth source:
   - monocular depth model, stereo depth, or dataset-provided depth.
2. Add simple temporal filtering for depth (EMA/Kalman) to reduce far-range jitter.
3. Calibrate projection ranges (`x_range`, `z_min`, `z_max`) per dataset/sequence.

### Priority C (tracker integration)
1. If applying to external tracker association, plug the same frame homography into matching cost.
2. Ensure decision-level association and representation-level motion use the same `H`.

## Suggested Next Commits
1. `feat(3d): project 13D motion features to XZ space in manager and dataset`
2. `docs: rewrite README for 3d-projection branch`
3. `docs: add change notes and follow-up plan`

