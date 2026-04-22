# SparseMFE Gatekeeper

**Decision date:** 2026-04-22
**Upstream status:** NOT_PUBLIC

## Probe results

| Check | Result |
|-------|--------|
| `github.com/kashyap7x/SparseMFE` | HTTP 404 |
| `github.com/kashyap7x` (user) | HTTP 200 — user exists, 8 public repos, none named SparseMFE |
| `github.com/anuragranj/sparsemfe` | HTTP 000 (intermittent) — per api search: no such repo |
| GitHub search `SparseMFE` | 1 unrelated hit (`s-will/SparseMFEFold` = RNA folding) |
| GitHub search `sparse motion field ego` | 0 hits |

Author Kashyap publishes to github but has not released the TNNLS 2021 sparse-motion paper code. Weights not available either.

## Decision

**FALLBACK_RECOVERPOSE** per spec §10 kill switch #1.

Implement `gmc_link/ego/recoverpose_ego.py` using `cv2.findEssentialMat` + `cv2.recoverPose` on ORB correspondences. This gives us a 6DoF {R, t} ego estimate directly — the same output SparseMFE would produce — but via classical geometry instead of a learned autoencoder.

## Consequences

- Stage A becomes "ORB 2D-planar homography" vs "ORB correspondences → 6DoF via recoverPose".
- Smaller expected ΔAUC (paper-reported SparseMFE advantage came from learned sparse motion-field representations, not just 6DoF recovery).
- KITTI ATE still meaningful: compares 2D-planar homography vs 6DoF essential-matrix ego.
- Stage B OMF features: need alternate flow source. Use RAFT (already a candidate in spec §11 Q2). Defer choice to Stage B kickoff.

## Rationale

Spec §10: "If Stage A SparseMFE install blocked for > 1 day, pivot to Stage A-alt: use `cv2.recoverPose` on ORB correspondences as a 6DoF-without-autoencoder baseline. Smaller expected gain, zero install risk." Two-day probe already exceeded the 1-day threshold (factoring in session gaps), and the repo is verifiably absent rather than merely unreachable.
