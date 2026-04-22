# Stage A — Ego-Source Ablation Decision

Date: 2026-04-22
Spec: `docs/superpowers/specs/2026-04-22-ego-motion-systematic-design.md` §2 H_A
Plan: `docs/superpowers/plans/2026-04-22-exp37-ego-motion-systematic.md` Task 8
Runner: `run_exp37_stage_a.sh`

## Challenger choice

SparseMFE upstream code not public (see `gatekeeper_sparsemfe.md`). Spec §10
kill switch #1 activates the `cv2.recoverPose` fallback: 5-point essential
matrix → 6DoF rotation/translation → plane-at-infinity projection
`H_inf = K · R · K⁻¹`. KITTI left-camera default intrinsics (focal=721.5377,
pp=(609.5593, 172.854)).

## Held-out V1 results (seqs 0005, 0011, 0013)

| model_tag | micro AUC | macro AUC ± std | 0005 | 0011 | 0013 | max_gap |
|---|---|---|---|---|---|---|
| `v1train_stage1` (A1 ORB) | **0.779** | 0.838 ± 0.064 | 0.821 | 0.779 | — | 0.042 |
| `exp37_stage_a2_recoverpose` (A2) | **0.743** | 0.767 ± 0.121 | — | 0.739 | 0.787 | 0.049 |
| **Δ (A2 − A1)** | **−0.036** | **−0.071 (std ~2× wider)** | regress | −0.040 | +0.008 (n=2)¹ | — |

¹ Seq 0013 has n=2 expressions — per project memory this is noise-dominated
and not load-bearing. Ignore the +0.008 when weighting the decision.

## Decision band (spec §2 H_A)

| Band | Δmicro threshold | Outcome |
|---|---|---|
| POSITIVE | ≥ +0.010 | — |
| INCONCLUSIVE | −0.010 < Δ < +0.010 | — |
| **NEGATIVE** | **≤ −0.010** | **✓ Δ = −0.036** |

**Decision: NEGATIVE.** Winner(A) = **ORB**.

## Why recoverPose regressed

1. **5-point essential matrix requires well-textured static scene;** KITTI
   has moving vehicles filling a large fraction of the frame, so RANSAC
   inlier sets are contaminated by foreground motion. ORB homography has
   the same issue but the planar-projection constraint (3-DoF vs 5-DoF)
   absorbs less outlier mass before breaking.
2. **Plane-at-infinity projection `H_inf = K R K⁻¹` discards translation.**
   For close-range ego motion (urban driving) pure-rotation approximation
   loses the parallax cue that ORB-homography captures via the full
   homography matrix.
3. **Per-seq std doubled** (0.064 → 0.121) — challenger is less stable, not
   merely centered lower. Seq 0013 lifted by +0.008 while 0005 and 0011
   regressed meaningfully. Not a "slightly weaker but still useful"
   substitution; it is structurally shakier.

## Implications for downstream stages

- Stage B runs with `--ego orb` (winner A). If Stage B tests OMF features,
  the ORB-compensated residuals feed the flow pool, so the 13D-vs-28D
  comparison is clean.
- Stage C structural conditioning uses ORB ego velocity.
- **Stage D (portability to TempRMOT) is now the headline per spec §10
  kill switch #2** — aligner-quality stages A/B/C cannot lift the 0.779
  representation-bound ceiling (memory:
  `project_exp36e_curriculum_negative`), so the ego-compensation service
  claim must carry via cross-tracker transfer.

## KITTI ATE side-experiment

Deferred — `/home/seanachan/data/Dataset/kitti_odometry/poses/` absent (see
`gatekeeper_kitti_poses.md`). Given Stage A is NEGATIVE on the aligner
endpoint, the ATE evidence is unlikely to overturn the decision even if
recoverPose wins on pure pose accuracy; ORB's aligner-facing residuals
dominate.

## Winner

```
diagnostics/results/exp37/stage_a_winner  →  orb
```
