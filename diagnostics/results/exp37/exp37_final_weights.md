# Exp 37 Final Weights Sentinel

Date: 2026-04-22

## Winner

**`gmc_link_weights_v1train_stage1.pth`** (symlinked as `gmc_link_weights.pth`).

- V1 micro AUC: **0.779** (Exp 37 baseline, Exp 36e representation-bound ceiling)
- V1 macro AUC: 0.838 ± 0.064
- Encoder: MiniLM-L6-v2 (384D)
- Motion dim: 13D (no ego_velocity_concat, no OMF, no 6DoF)
- Ego source: ORB + RANSAC homography

## Fusion config

- Consumer tracker: iKUN (spatially-ignorant)
- α = 1.0 (from `project_aligner_sweep_at_alpha1_restores_auc`)
- HOTA on iKUN+GMC-Link: 35.848 (+2.57 vs iKUN bare 32.878)

## Exp 37 arm-by-arm winners

| Stage | Winner | Loser | Notes |
|---|---|---|---|
| A | ORB | recoverPose | 5-point essential matrix too noisy on KITTI foreground |
| B | — | deferred | OMF cache not built; expected NEG per representation-bound pattern |
| C | 13D baseline | 15D EMAP concat | Structural conditioning redundant with residual velocities |
| D | bare TempRMOT | all α arms | Tracker class dichotomy: GMC-Link regresses temporally-aware trackers |

## Do NOT

- Do not cascade GMC-Link onto TempRMOT or any DETR-query tracker with temporal
  attention across frames. Structural regression (−3.8 to −5.4 HOTA).
- Do not use Stage C's 15D `ego_velocity_concat` weights
  (`gmc_link_weights_exp37_stage_c2_orb.pth`) — they regress micro AUC by 0.024.
- Do not trust Stage A's recoverPose weights if any exist — Δ = −0.036 vs ORB.

## Rollback instructions

TempRMOT: `cd /home/seanachan/TempRMOT && git checkout main` restores pre-Exp 37
state. The `exp37-stage-d-baseline` branch retains both the user's pre-existing
integration state (`babef6e`) and the Exp 37 env-var gated fusion
(`9015c68` + `3cc8972` fix).

GMC-Link: `gmc_link_weights.pth` remains a symlink to `v1train_stage1.pth`. No
rollback needed unless a future experiment wants a different baseline.
